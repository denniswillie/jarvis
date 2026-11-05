import Foundation
import CoreFoundation
import os

@MainActor
final class JarvisRealtimeClient {
#if DEBUG
    var evaluationRunID: UUID?
    var onReplayManualWrite: ((JarvisReplayManualWireObservation) -> Void)?
#endif
    var onEvent: ((JarvisRealtimeServerEvent) -> Void)?
    var onDisconnect: ((Error?) -> Void)?
    var onSocketOpened: ((UUID, TimeInterval) -> Void)?

    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "realtime-audio")
    private let urlSession: URLSession
    private let sendQueue = JarvisWebSocketSendQueue()
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var isDisconnectingIntentionally = false
    private var diagnostics = RealtimeDiagnostics()
    private var diagnosticContext: [String: Any] = [:]
    private(set) var connectionID = UUID()
    private var playbackReceiptSessionID: String?
    private var socketOpenObserver: JarvisControlSocketOpenObserver?

    init(urlSession: URLSession = .shared) {
        self.urlSession = urlSession
    }

    var isConnected: Bool {
        webSocketTask != nil
    }

    func connect(audioWebSocketURL: URL, bearerToken: String) {
        disconnect()
        connectionID = UUID()
        isDisconnectingIntentionally = false
        diagnostics = RealtimeDiagnostics()
        diagnosticContext = ["connection_id": connectionID.uuidString]

        var request = URLRequest(url: audioWebSocketURL)
        request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
#if DEBUG
        JarvisReplayScope.applyRunHeader(evaluationRunID, to: &request)
#endif
        logger.info("Connecting audio websocket host=\(audioWebSocketURL.host ?? "-", privacy: .public) path=\(audioWebSocketURL.path, privacy: .public)")

        let task = urlSession.webSocketTask(with: request)
        let openingConnection = connectionID
        let observer = JarvisControlSocketOpenObserver { [weak self] openedTask, observedAt in
            Task { @MainActor [weak self, weak openedTask] in
                guard let self, let openedTask, self.webSocketTask === openedTask,
                      self.connectionID == openingConnection, !self.isDisconnectingIntentionally else { return }
                self.onSocketOpened?(openingConnection, observedAt)
            }
        }
        socketOpenObserver = observer
        task.delegate = observer
        webSocketTask = task
        task.resume()
        startReceiveLoop(task: task)
    }

    func disconnect() {
        playbackReceiptSessionID = nil
        if webSocketTask != nil {
            logger.info("Disconnecting audio websocket")
        }
        isDisconnectingIntentionally = true
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        socketOpenObserver = nil
    }

    func sendAudioAppend(_ data: Data, segmentId: String) async throws {
        try await sendEvent(type: "audio.input.append", values: [
            "segment_id": segmentId,
            "audio_base64": data.base64EncodedString(),
        ])
    }

    func sendWebRTCOffer(_ sdp: String, captureMode: JarvisAudioCaptureMode,
                         voiceRoute: [String: Any]? = nil, conversationMode: String = "legacy") async throws {
        var values: [String: Any] = [
            "sdp": sdp, "capture_mode": captureMode == .autoDetect ? "auto" : "manual",
        ]
        if let voiceRoute { values["voice_route"] = voiceRoute }
        if conversationMode != "legacy" { values["conversation_mode"] = conversationMode }
        try await sendEvent(type: "audio.webrtc.offer", values: values)
    }

    func sendVoiceRouteEvent(_ event: [String: Any], connectionID: UUID) async throws {
        try Task.checkCancellation()
        guard self.connectionID == connectionID, webSocketTask != nil else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        guard let type = event["type"] as? String, type.hasPrefix("voice_route.") else {
            throw JarvisVoiceRouteError.invalidWire
        }
        var values = event
        values.removeValue(forKey: "type")
        try await sendEvent(type: type, values: values)
    }

    /// Borrow the current authenticated socket without the legacy extra wire
    /// fields. Guard immediately before URLSession send, not before a slow queue.
    func sendLiveAppEvent(_ payload: [String: Any], connectionID: UUID,
                          current: @escaping @MainActor () -> Bool) async throws {
        try Task.checkCancellation()
        guard self.connectionID == connectionID, let socket = webSocketTask, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        let text = try JarvisLiveAppWire.encode(payload)
        guard self.connectionID == connectionID, webSocketTask === socket, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        try await socket.send(.string(text))
        try Task.checkCancellation()
        guard self.connectionID == connectionID, webSocketTask === socket, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
    }

    /// Shared resource context uses the selected authenticated control socket.
    /// It is metadata only and carries no command or legacy event identifier.
    func sendWorkspaceResourceContext(_ payload: [String: Any], connectionID: UUID,
                                      current: @escaping @MainActor () -> Bool) async throws {
        try Task.checkCancellation()
        guard self.connectionID == connectionID, let socket = webSocketTask, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        guard payload["type"] as? String == "workspace.resource_context",
              payload["client_event_id"] == nil else { throw JarvisVoiceRouteError.invalidWire }
        let data = try JSONSerialization.data(withJSONObject: payload, options: .sortedKeys)
        guard data.count <= 65_536, let text = String(data: data, encoding: .utf8) else {
            throw JarvisVoiceRouteError.invalidWire
        }
        guard self.connectionID == connectionID, webSocketTask === socket, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        try await socket.send(.string(text))
        try Task.checkCancellation()
        guard self.connectionID == connectionID, webSocketTask === socket, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
    }

    /// Recording's owner serializes these small control writes. Preserve the
    /// exact schema instead of adding legacy client_event_id fields. A write
    /// is not backend acceptance, capture success or a saved-file receipt.
    func sendRecordingEvent(_ event: JarvisRecordingControl.Outbound, connectionID: UUID,
                            current: @escaping @MainActor () -> Bool) async throws {
        try Task.checkCancellation()
        guard self.connectionID == connectionID, let socket = webSocketTask, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        let text = try JarvisRecordingControl.encode(event)
        guard self.connectionID == connectionID, webSocketTask === socket, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        try await socket.send(.string(text))
        try Task.checkCancellation()
        guard self.connectionID == connectionID, webSocketTask === socket, current() else {
            throw JarvisVoiceRouteError.identityMismatch
        }
    }

    func sendCaptureMode(_ mode: JarvisAudioCaptureMode) async throws {
        try await sendEvent(type: "audio.capture_mode", values: [
            "capture_mode": mode == .autoDetect ? "auto" : "manual",
        ])
    }

    func sendAudioInputBegin(segmentId: String) async throws {
        try await sendEvent(type: "audio.input.begin", values: ["segment_id": segmentId])
    }

    func sendAudioInputAbort(segmentId: String) async throws {
        try await sendEvent(type: "audio.input.abort", values: ["segment_id": segmentId])
    }

    func sendAudioCommit(segmentId: String, startedAt: Date?, endedAt: Date?) async throws {
        var values: [String: Any] = ["segment_id": segmentId]
        if let startedAt {
            values["audio_started_at"] = startedAt.jarvisISO8601String()
        }
        if let endedAt {
            values["audio_ended_at"] = endedAt.jarvisISO8601String()
        }
        try await sendEvent(type: "audio.input.commit", values: values)
    }

    func sendResponseCreate() async throws {
        try await sendEvent(type: "response.create", values: [:])
    }

    func sendResponseCancel() async throws {
        try await sendEvent(type: "response.cancel", values: [:])
    }

    func sendPlaybackTruncated(itemId: String, contentIndex: Int = 0, audioEndMs: Int) async throws {
        try await sendEvent(type: "playback.truncated", values: [
            "item_id": itemId,
            "content_index": contentIndex,
            "audio_end_ms": max(0, audioEndMs),
        ])
    }

    func sendSessionClose() async throws {
        try await sendEvent(type: "session.close", values: [:])
    }

    func sendPlaybackReceipt(_ receipt: AudioPlaybackReceipt, audioSessionID: String,
                             connectionID: UUID) async throws -> Bool {
        try Task.checkCancellation()
        // Unsupported older backends stay quiet; a queued callback from a
        // retired socket cannot select the replacement connection.
        guard self.connectionID == connectionID, playbackReceiptSessionID == audioSessionID,
              webSocketTask != nil else { return false }
        guard receipt.playedSamples >= 0, !receipt.itemId.isEmpty, !receipt.responseId.isEmpty,
              receipt.itemId.count <= 256, receipt.responseId.count <= 256 else {
            throw JarvisAudioSessionError.serverError("Invalid playback receipt identity.")
        }
        try await sendEvent(type: "playback.\(receipt.state.rawValue)", values: [
            "protocol_version": 1, "audio_session_id": audioSessionID,
            "response_id": receipt.responseId, "item_id": receipt.itemId, "content_index": 0,
            "played_samples": receipt.playedSamples, "sample_rate": 24_000,
            "boundary": "av_audio_data_played_back_lower_bound",
        ])
        return true
    }

    func sendContextResetCleared(_ notice: JarvisContextResetNotice, connectionID: UUID) async throws -> Bool {
        try Task.checkCancellation()
        guard self.connectionID == connectionID, webSocketTask != nil else { return false }
        try await sendEvent(type: "session.context_cleared", values: [
            "protocol_version": 1, "reset_id": notice.id.uuidString.lowercased(),
            "audio_session_id": notice.audioSessionID,
        ])
        return true
    }

    private func learnPlaybackCapabilities(_ event: JarvisRealtimeServerEvent) {
        guard event.type == "session.created" else { return }
        playbackReceiptSessionID = nil
        guard let capabilities = event.values["playback_receipts"] as? [String: Any],
              let version = capabilities["protocol_version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              let rate = capabilities["sample_rate"] as? NSNumber,
              CFGetTypeID(rate) != CFBooleanGetTypeID(), rate == 24_000,
              let identity = event.audioSessionId, !identity.isEmpty else { return }
        playbackReceiptSessionID = identity
    }

    private func sendEvent(type: String, values: [String: Any]) async throws {
        try Task.checkCancellation()
        guard let webSocketTask else {
            throw JarvisAudioSessionError.realtimeNotConnected
        }

        var payload = values
        payload["type"] = type
        payload["client_event_id"] = UUID().uuidString

        let data = try JSONSerialization.data(withJSONObject: payload, options: [])
        guard let text = String(data: data, encoding: .utf8) else {
            throw JarvisAudioSessionError.serverError("Failed to encode realtime event.")
        }
        if type != "audio.input.append" {
            logger.debug("Sending realtime event \(type, privacy: .public)")
        }
        let context = diagnosticContext
        let writeStarted = ProcessInfo.processInfo.systemUptime
        if type != "audio.input.append",
           let trace = RealtimeDiagnostics.record("client_control_write_started", fields:
                context.merging(["event_type": type, "boundary": "before_urlsession_send"]) { _, new in new }) {
            logger.info("realtime_trace \(trace, privacy: .public)")
        }
#if DEBUG
        let replayObserver = onReplayManualWrite
        let replayWrite = evaluationRunID.flatMap {
            JarvisReplayManualWireObservation(runID: $0, connectionID: connectionID,
                payload: payload, context: context, at: writeStarted)
        }
        if let replayWrite { replayObserver?(replayWrite) }
#endif
        do {
            try await sendQueue.send(.string(text), task: webSocketTask)
        } catch {
#if DEBUG
            if var observation = replayWrite {
                observation.stage = .failed; observation.at = ProcessInfo.processInfo.systemUptime
                replayObserver?(observation)
            }
#endif
            throw error
        }
#if DEBUG
        if var observation = replayWrite {
            observation.stage = self.webSocketTask === webSocketTask
                && self.connectionID == observation.connectionID && evaluationRunID == observation.runID ? .written : .retired
            observation.at = ProcessInfo.processInfo.systemUptime
            replayObserver?(observation)
        }
#endif
        if type != "audio.input.append",
           let diagnostic = RealtimeDiagnostics.record("client_\(type)_written", fields:
                context.merging(payload) { _, new in new }.merging([
                    "write_elapsed_ms": (ProcessInfo.processInfo.systemUptime - writeStarted) * 1_000,
                    "boundary": "urlsession_send_completed_not_provider_ack",
                ]) { _, new in new }) {
            logger.info("realtime_trace \(diagnostic, privacy: .public)")
        }
    }

    private func startReceiveLoop(task: URLSessionWebSocketTask) {
        let connectionID = self.connectionID
        let timingLogger = logger
        // Receive/decode off MainActor. This measures application receipt, not
        // packet arrival; ordered handling still takes place on MainActor.
        receiveTask = Task.detached { [weak self, weak task] in
            guard let task else { return }
            while !Task.isCancelled {
                do {
                    let message = try await task.receive()
                    let received = ProcessInfo.processInfo.systemUptime
                    guard let event = try Self.decode(message: message) else { continue }
                    let decoded = ProcessInfo.processInfo.systemUptime
                    let fields: [String: Any] = ["connection_id": connectionID.uuidString,
                        "event_type": event.type, "boundary": "urlsession_receive_completed",
                        "receive_monotonic_s": received, "decode_elapsed_ms": (decoded - received) * 1_000]
                    if event.type != "audio.output.delta",
                       let trace = RealtimeDiagnostics.record("client_control_decoded", fields: fields) {
                        timingLogger.info("realtime_trace \(trace, privacy: .public)")
                    }
                    await MainActor.run { [weak self] in
                        let handling = ProcessInfo.processInfo.systemUptime
                        defer {
                            if event.type != "audio.output.delta",
                               let trace = RealtimeDiagnostics.record("client_control_handled", fields: fields.merging([
                                "boundary": "main_actor_handler_completed",
                                "actor_wait_ms": (handling - decoded) * 1_000,
                                "handler_elapsed_ms": (ProcessInfo.processInfo.systemUptime - handling) * 1_000,
                               ]) { _, new in new }) {
                                timingLogger.info("realtime_trace \(trace, privacy: .public)")
                            }
                        }
                        if self?.webSocketTask === task {
                            self?.recordReceivedDiagnostic(event)
                        }
                        guard let self, self.webSocketTask === task else { return }
                        self.learnPlaybackCapabilities(event)
                        self.onEvent?(event)
                    }
                } catch {
                    await MainActor.run { [weak self] in
                        guard self?.webSocketTask === task,
                              self?.isDisconnectingIntentionally != true else { return }
                        self?.webSocketTask = nil
                        self?.logger.error("Audio websocket disconnected: \(error.localizedDescription, privacy: .public)")
                        self?.onDisconnect?(error)
                    }
                    return
                }
            }
        }
    }

    private func recordReceivedDiagnostic(_ event: JarvisRealtimeServerEvent) {
        if ["session.created", "audio.provider.selected", "audio.provider.reconnected"].contains(event.type) {
            for key in ["provider", "audio_session_id", "session_id", "connection_generation"] {
                if let value = event.values[key] { diagnosticContext[key] = value }
            }
        }
        let fields = diagnosticContext.merging(event.values) { _, new in new }
        let diagnostic: String?
        if event.type == "audio.output.delta", event.audioBase64?.isEmpty == false {
            diagnostic = diagnostics.firstAudio(fields: fields)
        } else if ["session.created", "audio.provider.selected", "audio.provider.reconnecting",
                   "audio.provider.reconnected", "audio.output.clear", "response.done",
                   "audio.webrtc.answer", "audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started",
                   "audio.webrtc.stopped", "audio.webrtc.cleared", "audio.input.started",
                   "audio.input.stopped"].contains(event.type) {
            diagnostic = RealtimeDiagnostics.record("client_\(event.type)_received", fields: fields)
        } else {
            diagnostic = nil
        }
        if let diagnostic { logger.info("realtime_trace \(diagnostic, privacy: .public)") }
    }

    nonisolated private static func decode(message: URLSessionWebSocketTask.Message) throws -> JarvisRealtimeServerEvent? {
        let data: Data
        switch message {
        case .data(let payload):
            data = payload
        case .string(let text):
            guard let payload = text.data(using: .utf8) else { return nil }
            data = payload
        @unknown default:
            return nil
        }

        let object = try JSONSerialization.jsonObject(with: data)
        guard let values = object as? [String: Any], let type = values["type"] as? String else {
            return nil
        }
        return JarvisRealtimeServerEvent(type: type, values: values)
    }
}

/// The callback is an observed Foundation open, not task creation/resume or a
/// provider receipt. The negotiated protocol string is deliberately discarded.
final class JarvisControlSocketOpenObserver: NSObject, URLSessionWebSocketDelegate, @unchecked Sendable {
    private let opened: @Sendable (URLSessionWebSocketTask, TimeInterval) -> Void
    init(opened: @escaping @Sendable (URLSessionWebSocketTask, TimeInterval) -> Void) {
        self.opened = opened
    }
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        opened(webSocketTask, ProcessInfo.processInfo.systemUptime)
    }
}

private actor JarvisWebSocketSendQueue {
    func send(_ message: URLSessionWebSocketTask.Message, task: URLSessionWebSocketTask) async throws {
        try await task.send(message)
    }
}

@MainActor
final class JarvisAudioSendQueue {
    private struct Operation {
        let send: @MainActor () async throws -> Void
        let onFailure: @MainActor (Error) -> Void
    }

    private let maxPendingOperations: Int
    private var pending: [Operation] = []
    private var worker: Task<Void, Never>?
    private var generation = UUID()

    init(maxPendingOperations: Int = 32) {
        precondition(maxPendingOperations > 0)
        self.maxPendingOperations = maxPendingOperations
    }

    // Enqueue synchronously at the audio producer, before any network await.
    // A commit/create or cancel/truncate pair is one indivisible operation.
    func enqueue(
        _ send: @escaping @MainActor () async throws -> Void,
        onFailure: @escaping @MainActor (Error) -> Void
    ) throws {
        guard pending.count < maxPendingOperations else {
            throw JarvisAudioSessionError.serverError("Audio send queue is full.")
        }
        pending.append(Operation(send: send, onFailure: onFailure))
        guard worker == nil else { return }
        let token = generation
        worker = Task { await drain(generation: token) }
    }

    func reset() {
        generation = UUID()
        pending.removeAll()
        worker?.cancel()
        worker = nil
    }

    private func drain(generation token: UUID) async {
        defer {
            if generation == token { worker = nil }
        }
        while generation == token, !Task.isCancelled, !pending.isEmpty {
            let operation = pending.removeFirst()
            do {
                try await operation.send()
            } catch {
                guard generation == token else { return }
                reset()
                operation.onFailure(error)
                return
            }
        }
    }
}
