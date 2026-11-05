import Foundation

/// App composition only: existing capture, durable store and authenticated
/// transport owners stay separate. Nothing here starts or stops Jarvis voice.
@MainActor
final class JarvisRecordingAppController {
    struct Status {
        let recordingID: UUID?
        let capturing: Bool
        let message: String
        var phase: JarvisRecordingSession.State? = nil
    }
    struct Saved: Sendable {
        let ownerID: String
        let backendNamespace: String
        let sessionID: UUID
        let fileID: UUID
        let versionID: String
        let downloadRequestID: UUID?
    }
    private struct StreamRun {
        let binding: JarvisRecordingManifest.Binding
        let generationID: UUID
        let target: JarvisRecordingStreamingUpload.Target
        let stream: JarvisRecordingByteStream
        let transfer: Task<JarvisRecordingByteStream.Summary, Error>
    }
    typealias Context = JarvisRecordingSession.Context
    typealias Write = @MainActor (JarvisRecordingControl.Outbound, Context) async throws -> Void
    private let write: Write
    private let changed: @MainActor (Status) -> Void
    private let saved: @MainActor (Saved) -> Void
    private var identity = UUID()
    private var context: Context?
    private var current: (@MainActor () -> Bool)?
    private var video: JarvisRecordingVideoSource?
    private var visionReady = false
    private var controller: JarvisRecordingSession?
    private var retiring: [UUID: JarvisRecordingSession] = [:]
    private var transfers: JarvisRecordingTransfers?
    private var preparation: Task<Void, Never>?
    private var capabilityTail: Task<Void, Never>?
    private var lastCapability: Bool?
    private var activeRecordingID: UUID?
    private var lastPhase: JarvisRecordingSession.State?
    private var streamRuns: [UUID: StreamRun] = [:]

    init(write: @escaping Write, changed: @escaping @MainActor (Status) -> Void,
         saved: @escaping @MainActor (Saved) -> Void = { _ in }) {
        self.write = write; self.changed = changed; self.saved = saved
    }

    func configure(context: Context, audio: JarvisDuplexAudioDevice,
                   video: JarvisRecordingVideoSource, api: JarvisAPIClient,
                   current: @escaping @MainActor () -> Bool) {
        if self.context == context { refresh(visionReady: visionReady); return }
        retire()
        let token = UUID(); identity = token
        self.context = context; self.current = current; self.video = video
        preparation = Task { [weak self] in
            do {
                // This runs independently of voice readiness and input handling.
                let authenticated = try await api.fetchAuthenticatedContext(sessionID: context.sessionID)
                try Task.checkCancellation()
                guard let self, self.identity == token, current() else { return }
                let scope = try JarvisRecordingPendingStore.Scope(context: authenticated)
                let store = try JarvisRecordingPendingStore()
                let live: @MainActor () -> Bool = { [weak self] in self?.identity == token && current() }
                let transfers = JarvisRecordingTransfers(store: store, scope: scope, current: live,
                    request: { entry, reconcile in
                        let reply = try await api.recordingResponse(recordingID: entry.recordingID,
                            sessionID: entry.sessionID, file: reconcile ? nil : entry.fileURL,
                            descriptor: reconcile ? nil : entry.descriptor)
                        return (reply.0, reply.1.statusCode)
                    }, changed: { [weak self] id, result in
                        guard let self, live(), self.activeRecordingID == nil || self.activeRecordingID == id else { return }
                        let message: String
                        switch result {
                        case .saved: message = "Recording saved. The original is also kept on this phone."
                        case .pending: message = "Recording kept on this phone; server save is still unconfirmed."
                        case .unavailable: message = "Recording kept on this phone. Upload can be retried later."
                        }
                        self.changed(Status(recordingID: id, capturing: false, message: message))
                    })
                self.transfers = transfers
                self.controller = try JarvisRecordingSession.native(context: context, current: live,
                    audio: audio, video: video,
                    prepare: { [weak self] command in
                        let binding = try JarvisRecordingManifest.Binding(command: command)
                        let target = try await api.recordingUploadTarget(command: command)
                        guard let self, live(), !Task.isCancelled, self.streamRuns[command.recordingID] == nil else {
                            throw CancellationError()
                        }
                        let stream = JarvisRecordingByteStream(), generation = UUID()
                        let transport = JarvisRecordingStorageTransport()
                        let upload = try JarvisRecordingStreamingUpload(target: target,
                            send: { try await transport.send($0, data: $1) })
                        let transfer = Task {
                            defer { transport.close() }
                            return try await upload.run(stream: stream)
                        }
                        self.streamRuns[command.recordingID] = StreamRun(binding: binding, generationID: generation,
                            target: target, stream: stream, transfer: transfer)
                        return .stream(.init(generationID: generation, bytes: stream))
                    }, retain: { [weak self] command, completed in
                        guard let self, live(), let run = self.streamRuns[command.recordingID] else {
                            throw CancellationError()
                        }
                        defer {
                            run.stream.fail(); run.transfer.cancel()
                            self.streamRuns[command.recordingID] = nil
                        }
                        guard run.binding == (try JarvisRecordingManifest.Binding(command: command)),
                              !completed.writer.file.isStoredLocally else {
                            run.stream.fail(); run.transfer.cancel()
                            throw JarvisRecordingStreamingUpload.Failure.invalidBinding
                        }
                        let coverage = completed.mediaCoverage
                        let metrics: [String: UInt64]? = completed.selectedMediaAudio ? [
                            "consumed_output_frames": coverage.consumedOutputFrames,
                            "resolved_source_frames": coverage.resolvedSourceFrames,
                            "missing_source_frames": coverage.missingSourceFrames,
                            "unknown_output_frames": coverage.unknownOutputFrames,
                            "receipt_dropped_output_frames": coverage.receiptDroppedOutputFrames,
                            "zero_gain_source_frames": coverage.zeroGainSourceFrames,
                            "admitted_source_frames": coverage.admittedSourceFrames,
                            "rejected_source_frames": coverage.rejectedSourceFrames,
                            "recorded_source_frames": coverage.recordedSourceFrames,
                            "output_capture_dropped_frames": coverage.outputCaptureDroppedFrames] : nil
                        let descriptor = try JarvisRecordingManifest.make(binding: run.binding, completed: completed.writer,
                            expectedWriterGenerationID: run.generationID, mediaCoverage: metrics)
                        let uploaded = try await run.transfer.value
                        guard live(), uploaded.generationID == run.generationID,
                              uploaded.byteCount == completed.writer.file.byteCount,
                              uploaded.contentSHA256 == completed.writer.file.contentSHA256 else {
                            throw JarvisRecordingStreamingUpload.Failure.invalidBinding
                        }
                        let reply = try await api.finalizeRecordingUpload(recordingID: command.recordingID,
                            sessionID: command.sessionID, uploadID: run.target.uploadID, descriptor: descriptor)
                        guard reply.1.statusCode == 200 else { throw JarvisRecordingStreamingUpload.Failure.invalidResponse }
                        let receipt = try JarvisRecordingManifest.savedReceipt(reply.0, binding: run.binding, ownerID: scope.ownerID)
                        guard live(), !Task.isCancelled else { return }
                        let fields = try JSONSerialization.jsonObject(with: reply.0) as? [String: Any]
                        let request = (fields?["download_request_id"] as? String).flatMap(UUID.init(uuidString:))
                        self.saved(Saved(ownerID: scope.ownerID, backendNamespace: scope.backendNamespace,
                            sessionID: command.sessionID, fileID: receipt.fileID,
                            versionID: receipt.versionID, downloadRequestID: request))
                    }, write: { [weak self] event, bound in
                        guard let self, live() else { throw CancellationError() }
                        try await self.write(event, bound)
                    }, changed: { [weak self] id, state in
                        guard let self, live() else { return }
                        self.observe(id: id, state: state)
                    }, issue: { [weak self] id, _ in
                        guard let self, live() else { return }
                        self.changed(Status(recordingID: id, capturing: self.controller?.isCapturing ?? false,
                            message: "Recording status could not reach Jarvis.", phase: self.lastPhase))
                    })
                self.preparation = nil
                self.refresh(visionReady: self.visionReady)
                // Already accepted work retains its original session identity.
                // No synthetic new admission is created on reconnect.
                let pending = try await store.pending(scope: scope)
                guard live(), !Task.isCancelled else { return }
                for entry in pending where entry.confirmedLinkage == nil {
                    transfers.submit(entry, recovering: true)
                }
            } catch {
                guard let self, self.identity == token, current(), !Task.isCancelled else { return }
                self.preparation = nil
                self.changed(Status(recordingID: nil, capturing: false,
                    message: "Recording storage is unavailable. Jarvis can still talk and see."))
                self.refresh(visionReady: self.visionReady)
            }
        }
    }

    func refresh(visionReady: Bool) {
        self.visionReady = visionReady
        guard let context, let current, current() else { return }
        let available = controller != nil && visionReady && video?.latest != nil
            && !retiring.values.contains(where: { $0.isCapturing })
        guard lastCapability != available else { return }
        lastCapability = available
        let previous = capabilityTail, token = identity
        capabilityTail = Task { [weak self] in
            await previous?.value
            guard let self, self.identity == token, current() else { return }
            do {
                try await self.write(.capabilities(generationID: context.generationID, available: available,
                    includeJarvisAudio: true, maxBytes: JarvisRecordingControl.maximumBytes,
                    maxDurationMilliseconds: JarvisRecordingControl.maximumDurationMilliseconds,
                    protocolVersion: context.protocolVersion), context)
            } catch {
                // A later source/readiness change can retry this hint; no timer.
                if self.identity == token { self.lastCapability = nil }
            }
        }
    }

    func receive(_ values: [String: Any]) {
        guard let controller, current?() == true else { return }
        do { try controller.receive(JarvisRecordingControl.Command(values: values)) }
        catch {
            changed(Status(recordingID: activeRecordingID, capturing: controller.isCapturing,
                           message: "The recording command was rejected; no replacement capture was started.", phase: lastPhase))
        }
    }

    func stop() {
        guard let id = activeRecordingID else { return }
        controller?.stopLocally(recordingID: id)
    }

    func retire() {
        identity = UUID(); preparation?.cancel(); preparation = nil
        for run in streamRuns.values { run.stream.fail(); run.transfer.cancel() }
        streamRuns.removeAll()
        transfers?.retire(); transfers = nil
        capabilityTail?.cancel(); capabilityTail = nil; lastCapability = nil
        if let old = controller {
            let id = old.context.connectionID
            old.retire(); retiring[id] = old
            Task { [weak self] in
                await old.drained()
                self?.retiring[id] = nil
                if let self { self.refresh(visionReady: self.visionReady) }
            }
        }
        controller = nil; context = nil; current = nil; video = nil
        visionReady = false; activeRecordingID = nil; lastPhase = nil
        changed(Status(recordingID: nil, capturing: false, message: "Recording is not active."))
    }

    private func observe(id: UUID, state: JarvisRecordingSession.State) {
        lastPhase = state
        if [.captureFailed, .retentionFailed, .stoppedBeforeStart].contains(state),
           let run = streamRuns.removeValue(forKey: id) {
            run.stream.fail(); run.transfer.cancel()
        }
        let capturing = [.starting, .recording, .stopping].contains(state)
        if capturing { activeRecordingID = id }
        else if activeRecordingID == id { activeRecordingID = nil }
        let message: String
        switch state {
        case .starting: message = "Preparing recording..."
        case .recording: message = "Recording. Jarvis is still available."
        case .stopping: message = "Stopping recording..."
        case .retaining: message = "Finishing upload…"
        case .retained: message = "Saved to Audiovisual."
        case .captureFailed: message = "Recording failed."
        case .retentionFailed: message = "The recording save could not be confirmed."
        case .stoppedBeforeStart: message = "Recording stopped before capture began."
        }
        changed(Status(recordingID: id, capturing: capturing, message: message, phase: state))
    }
}
