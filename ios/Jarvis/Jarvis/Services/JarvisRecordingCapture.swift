@preconcurrency import CoreMedia
import CoreVideo
import CoreImage
import Foundation

/// Coordinates recording consumers only. The serial worker owns all mapping,
/// mixing and writer admission. Source callbacks do bounded enqueue work only.
final class JarvisRecordingCapture: @unchecked Sendable {
    enum StopReason: String, Sendable { case durationLimit, sourceTerminated, failure }
    enum Code: String, Sendable {
        case invalidConfiguration, videoUnavailable, videoAttachFailed, audioAttachFailed
        case startupTimeout, cancelled, audioFenceFailed, fenceTimeout
        case generationMismatch, mappingFailed, mixerFailed, writerFailed
    }
    struct Coverage: Sendable {
        var videoBaseline: JarvisRecordingVideoSource.Counters?
        var videoFinal: JarvisRecordingVideoSource.Counters?
        var audioBaseline: [JarvisRecordingAudioSource.Coverage] = []
        var audioFinal: [JarvisRecordingAudioSource.Coverage] = []
        var omittedAudioRoutes = 0
        var queueDroppedVideo = 0
        var queueDroppedAudioFrames = 0
        var queueDroppedMetadata = 0
        var rejectedVideo = 0
        var rejectedAudio = 0
        var outsideWindow = 0
        var audioFenced = false
        var videoFenced = false
        var attachmentPending = false
    }
    struct Times: Sendable {
        let startedAt: Date
        let startHostTime: UInt64
        let stopRequestedAt: Date
        let requestedStopHostTime: UInt64
        let requestedEnd: CMTime
        let mediaEnd: CMTime
        let mediaEndHostTime: UInt64
        let fencedHostTime: UInt64
    }
    // Mixer.Evidence/Gap are value-only snapshots, copied after its worker is
    // finished. No live mixer or mutable reference escapes this wrapper.
    struct Completed: @unchecked Sendable {
        let captureID: UUID
        let writer: JarvisRecordingWriter.Completed
        let mixerEvidence: JarvisRecordingMixer.Evidence
        let coverage: Coverage
        let times: Times
        let videoClock: JarvisRecordingTimeline.VideoClock
        let includeJarvisAudio: Bool
        var selectedMediaAudio: Bool = false
        var mediaCoverage = JarvisMediaCaptureCoverage()
    }
    struct Failure: Error, Sendable {
        let code: Code
        let coverage: Coverage
        let writerEvidence: JarvisRecordingWriter.Evidence?
    }
    struct Options: Sendable {
        var videoClock: JarvisRecordingTimeline.VideoClock = .datGlassesReceiptAnchored
        var startupTimeoutMilliseconds = 3_000
        var fenceTimeoutMilliseconds = 3_000
        var mixWaitMilliseconds = 250
        var nowHost: @Sendable () -> UInt64 = { mach_absolute_time() }
        var makeWriter: @Sendable (JarvisRecordingWriter.Configuration) async throws -> JarvisRecordingWriter = {
            try await JarvisRecordingWriter.make(configuration: $0)
        }
    }
    static let maximumPendingEvents = 128
    static let maximumPendingBytes = 16 * 1024 * 1024
    let captureID: UUID
    private let audio: JarvisRecordingAudioSource
    private let video: JarvisRecordingVideoSource
    private let snapshot: JarvisRecordingVideoSource.Snapshot
    private let writer: JarvisRecordingWriter
    private let timeline: JarvisRecordingTimeline
    private let includeOutput: Bool
    private let includeMedia: Bool
    private let options: Options
    private let startedAt: Date
    private let startHost: UInt64
    private let onStopped: @Sendable (StopReason) -> Void
    private let worker = DispatchQueue(label: "jarvis.recording.capture", qos: .utility)
    private let intake = NSLock()
    private var accepting = true
    private var pendingEvents = 0
    private var pendingBytes = 0
    private var droppedVideo = 0
    private var droppedAudio = 0
    private var droppedMetadata = 0
    private var droppedMediaSourceFrames: UInt64 = 0
    private var videoEndQueued = false
    private var audioEndQueued = false
    private let startup = RecordingCaptureResult<Void>()
    private let recordingReady = RecordingCaptureResult<Void>()
    private var confirmedRecording = false
    private let terminal = RecordingCaptureResult<Completed>()

    // Worker-confined state.
    private enum Phase { case idle, starting, running, stopping, finishing, terminal }
    private var phase = Phase.idle
    private var mixer: JarvisRecordingMixer!
    private var timer: DispatchSourceTimer?
    private var audioAttempted = false
    private var audioAttached: Bool?
    private var audioDetaching = false
    private var audioFenced = false
    private var audioFenceFailed = false
    private var videoAttached = false
    private var videoFenced = false
    private var failure: Code?
    private var automaticReason: StopReason?
    private var notified = false
    private var requestedStopHost: UInt64?
    private var requestedStopDate: Date?
    private var maximumAdmittedEnd = CMTime.zero
    private var report = Coverage()
    private var audioRoute: UInt64 = 0
    private var mediaThroughFrame: Int64 = 0
    private var mediaCoverage = JarvisMediaCaptureCoverage()
    private lazy var videoResizeContext = CIContext(options: [.cacheIntermediates: false])

    static func start(audio: JarvisRecordingAudioSource, video: JarvisRecordingVideoSource,
                      includeJarvisAudio: Bool, includeMediaAudio: Bool = false, maximumDurationMilliseconds: Int,
                      maxBytes: Int = 1024 * 1024 * 1024,
                      destination: JarvisRecordingWriter.Destination? = nil,
                      streaming: JarvisRecordingWriter.Streaming? = nil,
                      onStopped: @escaping @Sendable (StopReason) -> Void) async throws -> JarvisRecordingCapture {
        try await start(audio: audio, video: video, includeJarvisAudio: includeJarvisAudio, includeMediaAudio: includeMediaAudio,
            maximumDurationMilliseconds: maximumDurationMilliseconds, maxBytes: maxBytes, destination: destination, streaming: streaming,
            onStopped: onStopped, options: Options())
    }

    // Explicit internal test seam; production uses the overload above.
    static func start(audio: JarvisRecordingAudioSource, video: JarvisRecordingVideoSource,
                      includeJarvisAudio: Bool, includeMediaAudio: Bool = false, maximumDurationMilliseconds: Int,
                      maxBytes: Int = 1024 * 1024 * 1024,
                      destination: JarvisRecordingWriter.Destination? = nil,
                      streaming: JarvisRecordingWriter.Streaming? = nil,
                      onStopped: @escaping @Sendable (StopReason) -> Void,
                      options: Options) async throws -> JarvisRecordingCapture {
        guard (1...1_800_000).contains(maximumDurationMilliseconds),
              (1...(1024 * 1024 * 1024)).contains(maxBytes),
              (1...10_000).contains(options.startupTimeoutMilliseconds),
              (1...10_000).contains(options.fenceTimeoutMilliseconds),
              (1...1_000).contains(options.mixWaitMilliseconds) else {
            throw Failure(code: .invalidConfiguration, coverage: Coverage(), writerEvidence: nil)
        }
        guard let latest = video.latest else {
            throw Failure(code: .videoUnavailable, coverage: Coverage(), writerEvidence: nil)
        }
        var configuration = JarvisRecordingWriter.Configuration(width: latest.width, height: latest.height)
        configuration.destination = destination
        configuration.streaming = streaming
        configuration.maximumDuration = CMTime(value: Int64(maximumDurationMilliseconds), timescale: 1_000)
        configuration.maximumFileBytes = maxBytes
        let writer = try await options.makeWriter(configuration)
        let capture: JarvisRecordingCapture
        do {
            try Task.checkCancellation()
            capture = try JarvisRecordingCapture(audio: audio, video: video, snapshot: latest, writer: writer,
                includeOutput: includeJarvisAudio, includeMedia: includeMediaAudio, maximumMilliseconds: maximumDurationMilliseconds,
                options: options, onStopped: onStopped)
        } catch { writer.cancel(generationID: writer.generationID); throw error }
        return try await withTaskCancellationHandler(operation: {
            capture.worker.async { capture.begin() }
            try await capture.startup.value()
            if Task.isCancelled {
                try await capture.cancel()
                throw CancellationError()
            }
            return capture
        }, onCancel: { capture.requestStop(code: .cancelled) })
    }

    private init(audio: JarvisRecordingAudioSource, video: JarvisRecordingVideoSource,
                 snapshot: JarvisRecordingVideoSource.Snapshot, writer: JarvisRecordingWriter,
                 includeOutput: Bool, includeMedia: Bool, maximumMilliseconds: Int, options: Options,
                 onStopped: @escaping @Sendable (StopReason) -> Void) throws {
        captureID = UUID(); self.audio = audio; self.video = video; self.snapshot = snapshot
        self.writer = writer; self.includeOutput = includeOutput; self.options = options
        self.includeMedia = includeMedia
        self.onStopped = onStopped; startedAt = Date(); startHost = options.nowHost()
        timeline = try JarvisRecordingTimeline(startHostTime: startHost, maximumDurationMilliseconds: maximumMilliseconds,
            videoClock: options.videoClock, maximumVideoDeliveryDelay: CMTime(value: 2, timescale: 1),
            videoReference: (snapshot.presentationTime, snapshot.receivedHostTime))
        mixer = JarvisRecordingMixer(timeline: timeline, includeJarvisAudio: includeOutput,
                                     includeMediaAudio: includeMedia) { [weak self] chunk in
            self?.writeAudio(chunk)
        }
    }

    /// Caller cancellation does not cancel a shared finish. Explicit cancel()
    /// is separate. Every finish caller observes the same terminal result/file.
    func finish() async throws -> Completed {
        requestStop()
        return try await terminal.value()
    }

    /// Attaching consumers is not proof that frames can be encoded. Native UI
    /// and Voice wait for real writes on both tracks before announcing a start.
    func waitUntilRecording() async throws {
        try await withTaskCancellationHandler(operation: {
            worker.async { [self] in
                confirmRecordingIfReady()
                worker.asyncAfter(deadline: .now() + .milliseconds(options.startupTimeoutMilliseconds)) { [self] in
                    guard !confirmedRecording, phase == .starting || phase == .running else { return }
                    stop(code: .startupTimeout, automatic: .failure)
                }
            }
            try await recordingReady.value()
        }, onCancel: { self.requestStop(code: .cancelled) })
    }

    private func confirmRecordingIfReady() {
        guard !confirmedRecording, phase == .running, failure == nil else { return }
        let evidence = writer.evidence
        guard evidence.video.appended > 0, evidence.audio.appended > 0 else { return }
        confirmedRecording = true
        recordingReady.resolve(.success(()))
    }
    func cancel() async throws {
        requestStop(code: .cancelled)
        do { _ = try await terminal.value() }
        catch let result as Failure where result.code == .cancelled { return }
    }

    private func begin() {
        guard phase == .idle else { return }
        mixer.enableEchoReduction()
        phase = .starting
        let source = DispatchSource.makeTimerSource(queue: worker)
        source.schedule(deadline: .now() + .milliseconds(50), repeating: .milliseconds(50))
        // Retain the capture until actual stop/failure, even if a caller loses
        // its handle. Cancelling/clearing the handler below breaks the cycle.
        source.setEventHandler { [self] in tick() }; timer = source; source.resume()
        guard let baseline = video.subscribe(captureID: captureID, sourceGeneration: snapshot.sourceGeneration,
            sink: { [weak self] in self?.receiveVideo($0) }) else {
            videoFenced = true; audioFenced = true
            stop(code: .videoAttachFailed, automatic: .failure); return
        }
        videoAttached = true; report.videoBaseline = baseline.counters
        audioAttempted = true
        audio.attach(captureID, { [weak self] in self?.receiveAudio($0) }, { [self] ok in
            worker.async { self.attachedAudio(ok) }
        })
        worker.asyncAfter(deadline: .now() + .milliseconds(options.startupTimeoutMilliseconds)) { [weak self] in
            guard let self, self.phase == .starting else { return }
            self.stop(code: .startupTimeout, automatic: .failure)
        }
    }
    private func attachedAudio(_ ok: Bool) {
        guard audioAttached == nil else { return }
        audioAttached = ok
        if !ok { audioFenced = true }
        if phase == .terminal || phase == .finishing {
            // The startup/fence deadline can expire before the real ADM attach
            // callback. Its retained completion ALWAYS removes a late sink.
            if ok { audio.detach(captureID) { _ in } }
            return
        }
        if phase == .stopping { detachAudioIfReady(); finalizeIfFenced(); return }
        guard ok else { stop(code: .audioAttachFailed, automatic: .failure); return }
        phase = .running; startup.resolve(.success(()))
    }

    private func receiveVideo(_ event: JarvisRecordingVideoSource.Event) {
        switch event {
        case .frame(let frame):
            guard frame.captureID == captureID else { return }
            let bytes = CMSampleBufferGetImageBuffer(frame.sampleBuffer).map(CVPixelBufferGetDataSize) ?? 0
            enqueue(bytes: bytes, video: true, frames: 0) { self.processVideo(frame) }
        case .ended(let id, let generation, _, let counters):
            guard id == captureID, generation == snapshot.sourceGeneration else { return }
            let schedule = intake.withLock { if videoEndQueued { return false }; videoEndQueued = true; return true }
            if schedule { worker.async { [self] in
                videoFenced = true; report.videoFinal = counters
                if phase == .starting || phase == .running { stop(automatic: .sourceTerminated) }
                finalizeIfFenced()
            } }
        }
    }
    private func receiveAudio(_ event: JarvisRecordingAudioSource.Event) {
        if case .boundary(let id, _, _, let isTerminal, let fenced) = event, isTerminal {
            guard id == captureID else { return }
            let schedule = intake.withLock { if audioEndQueued { return false }; audioEndQueued = true; return true }
            if schedule { worker.async { [self] in
                if fenced { audioFenced = true }
                if phase == .starting || phase == .running { stop(automatic: .sourceTerminated) }
                finalizeIfFenced()
            } }
            return
        }
        let bytes: Int, frames: Int, mediaFrames: Int
        if case .pcm(let p) = event {
            bytes = p.pcm16Interleaved.count + p.mediaSpans.reduce(0) { $0 + $1.pcm16Interleaved.count }
            frames = p.frames
            mediaFrames = includeMedia ? p.mediaSpans.reduce(0) { $0 + max(0, $1.frames) } : 0
        } else { bytes = 0; frames = 0; mediaFrames = 0 }
        enqueue(bytes: bytes, video: false, frames: max(0, frames), mediaFrames: mediaFrames) { self.processAudio(event) }
    }
    private func enqueue(bytes: Int, video: Bool, frames: Int, mediaFrames: Int = 0, work: @escaping @Sendable () -> Void) {
        let admitted = intake.withLock {
            guard accepting else { return false }
            guard bytes >= 0, pendingEvents < Self.maximumPendingEvents,
                  bytes <= Self.maximumPendingBytes - pendingBytes else {
                if video { droppedVideo += 1 }
                else if frames > 0 { droppedAudio += frames }
                else { droppedMetadata += 1 }
                droppedMediaSourceFrames += UInt64(mediaFrames)
                return false
            }
            pendingEvents += 1; pendingBytes += bytes; return true
        }
        guard admitted else { return }
        worker.async { [self] in
            if phase == .starting || phase == .running || phase == .stopping { work() }
            intake.withLock { pendingEvents -= 1; pendingBytes -= bytes }
        }
    }
    private func processVideo(_ frame: JarvisRecordingVideoSource.Frame) {
        guard frame.snapshot.sourceGeneration == snapshot.sourceGeneration else {
            stop(code: .generationMismatch, automatic: .failure); return
        }
        // Frame dimensions may change without replacing the camera stream.
        // Keep the encoder's original canvas and the original sample timing.
        guard let recordingSample = fixedSizeVideoSample(frame.sampleBuffer) else {
            report.rejectedVideo += 1; return
        }
        do {
            guard let sample = try timeline.videoCopy(recordingSample, receivedHostTime: frame.snapshot.receivedHostTime) else {
                report.outsideWindow += 1; return
            }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample), duration = CMSampleBufferGetDuration(sample)
            let explicit = duration.isNumeric && CMTimeCompare(duration, .zero) > 0
            let sampleEnd = explicit ? CMTimeAdd(pts, duration)
                : CMTimeMinimum(timeline.end, CMTimeAdd(pts, CMTime(value: 1, timescale: 24)))
            // Preserve real durations. A sample extending beyond the immutable
            // envelope is rejected, not silently shortened to fit it.
            guard CMTimeCompare(sampleEnd, timeline.end) <= 0 else { report.rejectedVideo += 1; return }
            switch writer.offerVideo(sample, generationID: writer.generationID) {
            case .queued: maximumAdmittedEnd = CMTimeMaximum(maximumAdmittedEnd, sampleEnd)
            case .rejected(let reason):
                report.rejectedVideo += 1
                if reason != .queueFull && reason != .backpressure && reason != .outOfOrder {
                    stop(code: .writerFailed, automatic: .failure)
                }
            }
        } catch { stop(code: .mappingFailed, automatic: .failure) }
    }

    private func fixedSizeVideoSample(_ sample: CMSampleBuffer) -> CMSampleBuffer? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sample) else { return nil }
        let width = CVPixelBufferGetWidth(imageBuffer), height = CVPixelBufferGetHeight(imageBuffer)
        if width == snapshot.width && height == snapshot.height { return sample }
        guard width > 0, height > 0 else { return nil }
        var resized: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, snapshot.width, snapshot.height,
            kCVPixelFormatType_32BGRA, attributes, &resized) == kCVReturnSuccess,
              let resized else { return nil }
        let bounds = CGRect(x: 0, y: 0, width: snapshot.width, height: snapshot.height)
        let scale = min(CGFloat(snapshot.width) / CGFloat(width), CGFloat(snapshot.height) / CGFloat(height))
        let image = CIImage(cvPixelBuffer: imageBuffer)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: (bounds.width - CGFloat(width) * scale) / 2,
                                               y: (bounds.height - CGFloat(height) * scale) / 2))
            .composited(over: CIImage(color: .black).cropped(to: bounds))
        videoResizeContext.render(image, to: resized, bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())
        var format: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
            imageBuffer: resized, formatDescriptionOut: &format) == noErr, let format else { return nil }
        var timing = CMSampleTimingInfo(duration: CMSampleBufferGetDuration(sample),
            presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(sample),
            decodeTimeStamp: CMSampleBufferGetDecodeTimeStamp(sample))
        var output: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault,
            imageBuffer: resized, formatDescription: format, sampleTiming: &timing,
            sampleBufferOut: &output) == noErr else { return nil }
        return output
    }
    private func processAudio(_ event: JarvisRecordingAudioSource.Event) {
        switch event {
        case .format(let f):
            guard f.voiceGeneration == audio.voiceGeneration, f.routeGeneration > 0 else {
                stop(code: .generationMismatch, automatic: .failure); return
            }
        case .coverage(let c):
            if let index = report.audioFinal.firstIndex(where: { $0.routeGeneration == c.routeGeneration }) {
                report.audioFinal[index] = c
            } else if report.audioFinal.count < 64 {
                report.audioBaseline.append(c); report.audioFinal.append(c)
            } else { report.omittedAudioRoutes += 1 }
        case .boundary: break
        case .pcm(let p):
            guard p.captureID == captureID else { return }
            guard p.format.voiceGeneration == audio.voiceGeneration else {
                stop(code: .generationMismatch, automatic: .failure); return
            }
            if p.track == .renderedOutput && !includeOutput && !includeMedia && !mixer.isEchoReductionEnabled { return }
            guard p.format.routeGeneration >= audioRoute else { report.rejectedAudio += 1; return }
            do {
                guard let window = try timeline.audioWindow(hostTime: p.hostTime, timestampFlags: p.timestampFlags,
                    sampleRate: p.format.sampleRate, frames: p.frames) else { report.outsideWindow += 1; return }
                // Bounded missing-coverage expiry, not compression of the gap.
                let requiredCursor = max(mixer.cursor, window.outputEndFrame - Int64(JarvisRecordingMixer.capacity))
                if requiredCursor > mixer.cursor { try mixer.drain(through: requiredCursor, expiringMissing: true) }
                if p.track == .microphone || includeOutput || mixer.isEchoReductionEnabled {
                    let result = try mixer.offer(.init(track: p.track == .microphone ? .microphone : .renderedOutput,
                        routeGeneration: p.format.routeGeneration, hostTime: p.hostTime, timestampFlags: p.timestampFlags,
                        sampleTime: p.sampleTime, sampleRate: p.format.sampleRate,
                        channels: p.track == .microphone ? p.format.inputChannels : p.format.outputChannels,
                        frames: p.frames, pcm16Interleaved: p.pcm16Interleaved))
                    if case .accepted = result {
                        audioRoute = p.format.routeGeneration
                        maximumAdmittedEnd = CMTimeMaximum(maximumAdmittedEnd, window.end)
                    }
                }
                if (includeMedia || mixer.isEchoReductionEnabled) && p.track == .renderedOutput {
                    mediaCoverage.add(p.mediaCoverage)
                    for span in p.mediaSpans {
                        do {
                            let flags = AudioTimeStampFlags.hostTimeValid.rawValue
                            guard let mediaWindow = try timeline.audioWindow(hostTime: span.hostTime,
                                timestampFlags: flags, sampleRate: 48_000, frames: span.frames) else { continue }
                            let admission = try mixer.offer(.init(track: .media, routeGeneration: p.format.routeGeneration,
                                hostTime: span.hostTime, timestampFlags: flags, sampleTime: 0, sampleRate: 48_000,
                                channels: 2, frames: span.frames, pcm16Interleaved: span.pcm16Interleaved))
                            if case .accepted = admission {
                                mediaCoverage.admittedSourceFrames += UInt64(mediaWindow.sourceFrames.count)
                                maximumAdmittedEnd = CMTimeMaximum(maximumAdmittedEnd, mediaWindow.end)
                            }
                        } catch { mediaCoverage.rejectedSourceFrames += UInt64(max(0, span.frames)) }
                    }
                    // This output callback is complete even when it consumed no media.
                    mediaThroughFrame = max(mediaThroughFrame, window.outputEndFrame)
                }
                // With voice excluded, microphone arrival must not flush ahead
                // of the output packet carrying its corresponding media sidecar.
                let through = (includeMedia || mixer.isEchoReductionEnabled)
                    ? min(window.outputEndFrame, mediaThroughFrame) : window.outputEndFrame
                try mixer.drain(through: max(mixer.cursor, through))
            } catch let error as JarvisRecordingMixer.MixerError where error == .lateInput || error == .staleRoute {
                report.rejectedAudio += 1
            } catch { stop(code: .mixerFailed, automatic: .failure) }
        }
    }
    private func writeAudio(_ chunk: JarvisRecordingMixer.Chunk) {
        switch writer.offerAudio(chunk.samples, at: chunk.presentationTime, generationID: writer.generationID) {
        case .queued:
            maximumAdmittedEnd = CMTimeMaximum(maximumAdmittedEnd,
                CMTime(value: chunk.startFrame + Int64(chunk.samples.count), timescale: 48_000))
        case .rejected(let reason):
            report.rejectedAudio += 1
            if reason != .queueFull && reason != .backpressure { failure = .writerFailed }
        }
    }
    private func tick() {
        guard phase == .starting || phase == .running else { return }
        confirmRecordingIfReady()
        do {
            let now = try timeline.relativeHostTime(options.nowHost())
            if CMTimeCompare(now, timeline.end) >= 0 { stop(automatic: .durationLimit); return }
            if failure != nil { stop(code: failure, automatic: .failure); return }
            let frame = CMTimeConvertScale(now, timescale: 48_000, method: .roundTowardNegativeInfinity).value
            let expired = max(mixer.cursor, frame - Int64(options.mixWaitMilliseconds) * 48)
            if expired > mixer.cursor { try mixer.drain(through: expired, expiringMissing: true) }
        } catch { stop(code: .mixerFailed, automatic: .failure) }
    }

    private func requestStop(code: Code? = nil) {
        let host = options.nowHost(), date = Date()
        worker.async { [self] in stop(code: code, host: host, date: date) }
    }
    private func stop(code: Code? = nil, automatic: StopReason? = nil, host: UInt64? = nil, date: Date? = nil) {
        guard phase != .terminal, phase != .finishing else { return }
        if failure == nil { failure = code }
        if phase == .stopping { finalizeIfFenced(); return }
        if phase == .idle || phase == .starting, failure == nil { failure = .startupTimeout }
        if !confirmedRecording {
            recordingReady.resolve(.failure(Failure(code: failure ?? .startupTimeout,
                coverage: coverageSnapshot(), writerEvidence: writer.evidence)))
        }
        phase = .stopping; automaticReason = automatic
        requestedStopHost = host ?? options.nowHost(); requestedStopDate = date ?? Date()
        timer?.setEventHandler {}; timer?.cancel(); timer = nil
        if !videoAttached { videoFenced = true }
        else if !videoFenced {
            _ = video.detach(captureID: captureID) { [self] in worker.async {
                self.videoFenced = true; self.finalizeIfFenced()
            } }
            // false can mean sourceEnded's ordered end marker is still queued.
        }
        if !audioAttempted { audioFenced = true }
        detachAudioIfReady()
        worker.asyncAfter(deadline: .now() + .milliseconds(options.fenceTimeoutMilliseconds)) { [weak self] in
            guard let self, self.phase == .stopping else { return }
            // Best effort now, plus the retained late-attach completion above.
            if !self.audioFenced { self.audio.detach(self.captureID) { _ in } }
            self.failure = .fenceTimeout; self.completeFailure()
        }
        finalizeIfFenced()
    }
    private func detachAudioIfReady() {
        guard !audioFenced, !audioDetaching, audioAttached != nil else { return }
        guard audioAttached == true else { audioFenced = true; return }
        audioDetaching = true
        audio.detach(captureID) { [self] ok in worker.async {
            self.audioDetaching = false
            if ok { self.audioFenced = true }
            else if !self.audioFenced { self.audioFenceFailed = true; self.failure = .audioFenceFailed }
            self.finalizeIfFenced()
        } }
    }
    private func finalizeIfFenced() {
        guard phase == .stopping, videoFenced, audioFenced || audioFenceFailed else { return }
        intake.withLock { accepting = false }
        if failure != nil { completeFailure(); return }
        do {
            let requested = try timeline.relativeHostTime(requestedStopHost ?? options.nowHost())
            let requestedEnd = CMTimeMinimum(timeline.end, CMTimeMaximum(.zero, requested))
            let target = CMTimeMaximum(requestedEnd, maximumAdmittedEnd)
            let frame = max(Int64(1), CMTimeConvertScale(target, timescale: 48_000,
                method: .roundTowardPositiveInfinity).value)
            let end = CMTime(value: frame, timescale: 48_000)
            guard CMTimeCompare(end, timeline.end) <= 0 else { throw JarvisRecordingMixer.MixerError.invalidFlush }
            try mixer.finish(through: frame)
            if failure != nil { completeFailure(); return }
            let times = Times(startedAt: startedAt, startHostTime: startHost,
                stopRequestedAt: requestedStopDate ?? Date(), requestedStopHostTime: requestedStopHost ?? options.nowHost(),
                requestedEnd: requestedEnd, mediaEnd: end,
                mediaEndHostTime: CMClockConvertHostTimeToSystemUnits(CMTimeAdd(timeline.origin, end)),
                fencedHostTime: options.nowHost())
            let coverage = coverageSnapshot(), evidence = MixerSnapshot(value: mixer.evidence)
            var selectedCoverage = mediaCoverage
            if includeMedia {
                selectedCoverage.recordedSourceFrames = UInt64(max(0, mixer.evidence.mediaFrames))
                selectedCoverage.rejectedSourceFrames += intake.withLock { droppedMediaSourceFrames }
                for final in report.audioFinal {
                    if let baseline = report.audioBaseline.first(where: { $0.routeGeneration == final.routeGeneration }) {
                        selectedCoverage.outputCaptureDroppedFrames += final.outputDroppedFrames >= baseline.outputDroppedFrames
                            ? final.outputDroppedFrames - baseline.outputDroppedFrames : 0
                    }
                }
            }
            let finalMediaCoverage = selectedCoverage
            phase = .finishing; notifyStopped()
            Task { [self] in
                do {
                    let completed = try await writer.finish(at: end, generationID: writer.generationID)
                    let result = Completed(captureID: captureID, writer: completed, mixerEvidence: evidence.value,
                        coverage: coverage, times: times, videoClock: timeline.videoClock, includeJarvisAudio: includeOutput,
                        selectedMediaAudio: includeMedia, mediaCoverage: finalMediaCoverage)
                    worker.async { self.phase = .terminal; self.terminal.resolve(.success(result)) }
                } catch {
                    worker.async { self.failure = .writerFailed; self.completeFailure() }
                }
            }
        } catch { failure = .mixerFailed; completeFailure() }
    }
    private struct MixerSnapshot: @unchecked Sendable { let value: JarvisRecordingMixer.Evidence }
    private func coverageSnapshot() -> Coverage {
        var value = report
        intake.withLock {
            value.queueDroppedVideo = droppedVideo; value.queueDroppedAudioFrames = droppedAudio
            value.queueDroppedMetadata = droppedMetadata
        }
        value.audioFenced = audioFenced; value.videoFenced = videoFenced
        value.attachmentPending = audioAttempted && audioAttached == nil
        return value
    }
    private func completeFailure() {
        guard phase != .terminal else { return }
        intake.withLock { accepting = false }
        writer.cancel(generationID: writer.generationID)
        let error = Failure(code: failure ?? .writerFailed, coverage: coverageSnapshot(), writerEvidence: writer.evidence)
        phase = .terminal; timer?.setEventHandler {}; timer?.cancel(); timer = nil
        startup.resolve(.failure(error)); recordingReady.resolve(.failure(error))
        terminal.resolve(.failure(error)); notifyStopped()
    }
    private func notifyStopped() {
        guard !notified, let automaticReason else { return }
        notified = true
        DispatchQueue.global(qos: .utility).async { [onStopped] in onStopped(automaticReason) }
    }
}

private final class RecordingCaptureResult<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Value, Error>?
    private var waiters: [CheckedContinuation<Value, Error>] = []
    func value() async throws -> Value {
        try await withCheckedThrowingContinuation { continuation in
            let ready = lock.withLock { () -> Result<Value, Error>? in
                if let result { return result }
                waiters.append(continuation); return nil
            }
            if let ready { continuation.resume(with: ready) }
        }
    }
    func resolve(_ value: Result<Value, Error>) {
        let pending = lock.withLock { () -> [CheckedContinuation<Value, Error>] in
            guard result == nil else { return [] }
            result = value; let current = waiters; waiters.removeAll(); return current
        }
        for continuation in pending { continuation.resume(with: value) }
    }
}
