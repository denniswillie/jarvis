import AVFoundation
import CoreMedia
import Foundation
import WebRTC

/// Hardware-backed opt-in device. iOS 17; bridge JarvisDuplexAudioIO.h and build
/// its .m with ARC. One instance/factory per voice-session generation. Never
/// replace this device merely to start/stop a recording.
///
/// RTCAudioDevice lifecycle/properties run on WebRTC's ADM thread. Core entry
/// points below dispatch there; the mailbox lock is NEVER used by I/O callbacks.
/// Core must not hold RTCAudioSession's configuration lock while calling these
/// entry points or creating the peer. A failed unit fence forbids replacement.
final class JarvisDuplexAudioDevice: NSObject, RTCAudioDevice, @unchecked Sendable {
    struct Route: Sendable {
        let identity: String // Opaque route identity, not a user-visible device name.
        let sampleRate: Double
        let inputChannels: Int
        let outputChannels: Int
        let ioBufferDuration: TimeInterval
        let inputLatency: TimeInterval
        let outputLatency: TimeInterval
    }

    struct SessionLease: Sendable {
        /// Core configures/activates the existing shared session and returns its
        /// ACTUAL route. Suggested values are preferences, not negotiated truth.
        /// Required policy: playAndRecord/voiceChat, appropriate HFP routing.
        /// No default WebRTC ADM or second engine may own hardware concurrently.
        let acquire: @Sendable (_ preferredRate: Double, _ preferredBuffer: TimeInterval) throws -> Route
        let release: @Sendable () -> Void
    }

    struct Format: Sendable {
        let voiceGeneration: UUID
        let routeGeneration: UInt64
        let route: Route
        let inputChannels: Int
        let outputChannels: Int
    }

    struct PCMBlock: Sendable {
        enum Track: Sendable { case microphone, renderedOutput }
        let captureID: UUID
        let format: Format
        let track: Track
        let pcm16Interleaved: Data
        let frames: UInt32
        // Native AudioTimeStamp fields, not Date()/queue-arrival timestamps.
        // hostTime is mach absolute ticks ONLY when hostTimeValid is set.
        // sampleTime may restart on a new route generation. No DAT mapping here.
        let hostTime: UInt64
        let sampleTime: Double
        let timestampFlags: UInt32
        let softwareGateOpen: Bool
        var mediaSpans: [JarvisMediaCaptureSpan] = []
        var mediaCoverage = JarvisMediaCaptureCoverage()
    }

    struct Coverage: Sendable, Equatable {
        let routeGeneration: UInt64
        let microphoneDroppedPackets: UInt64
        let microphoneDroppedFrames: UInt64
        let outputDroppedPackets: UInt64
        let outputDroppedFrames: UInt64
        let microphoneErrors: UInt64
        let outputErrors: UInt64
        let lastMicrophoneStatus: Int32
        let lastOutputStatus: Int32
    }

    enum CaptureEvent: Sendable {
        case format(Format)
        case pcm(PCMBlock)
        // Cumulative per route generation; first event is the capture baseline.
        // A count gives lost coverage, not the exact locations of all gaps.
        case coverage(Coverage)
        case boundary(captureID: UUID, routeGeneration: UInt64?, reason: String)
    }

    struct Status: Sendable {
        let voiceGeneration: UUID
        let routeGeneration: UInt64
        let reason: String
        let osStatus: Int32?
    }
    typealias Sink = @Sendable (CaptureEvent) -> Void
    typealias Completion = @Sendable (Bool) -> Void

    private final class Handle: @unchecked Sendable {
        let io: JRDDuplexIO
        let format: Format
        init(_ io: JRDDuplexIO, _ format: Format) { self.io = io; self.format = format }
    }

    private static let externalMicLock = NSLock()
    nonisolated(unsafe) private static var activeExternalIO: JRDDuplexIO?

    static func pushActiveExternalMicrophoneSamples(_ samples: UnsafePointer<Int16>,
                                                    count: UInt32,
                                                    sampleRate: Double) {
        externalMicLock.lock()
        let io = activeExternalIO
        externalMicLock.unlock()
        io?.pushExternalMicrophoneSamples(samples, count: count, sampleRate: sampleRate)
    }

    static func clearActiveExternalMicrophoneSamples() {
        externalMicLock.lock()
        let io = activeExternalIO
        externalMicLock.unlock()
        io?.clearExternalMicrophoneSamples()
    }

    private static func setActiveExternalIO(_ io: JRDDuplexIO) {
        externalMicLock.lock()
        activeExternalIO = io
        externalMicLock.unlock()
    }

    private static func clearActiveExternalIO(_ io: JRDDuplexIO) {
        externalMicLock.lock()
        if activeExternalIO === io { activeExternalIO = nil }
        externalMicLock.unlock()
    }

    let voiceGeneration: UUID
    private let lease: SessionLease
    private let statusSink: @Sendable (Status) -> Void
    private let captureQueue = DispatchQueue(label: "jarvis.duplex.recording", qos: .utility)
    private let mailbox = NSLock() // Core/ADM only, never callbacks.
    private var delegateMailbox: RTCAudioDeviceDelegate?
    private var handleMailbox: Handle?
    private var microphoneGate = false
    private var outputGate = false
    private var silentRecordingIDs: Set<UUID> = []
    private var mediaSerial: UInt64 = 0
    private var activeMediaSerial: UInt64?
    private var mediaReady = false
    private var mediaGain: Float = 1
    private var mediaMuted = false
    private var mediaUserSpeaking = false
    private var mediaSuppressionIDs: Set<UUID> = []
    private struct MediaOriginal {
        let generation: UInt64
        let start: Int64
        let pcm: Data
        var end: Int64 { start + Int64(pcm.count / 4) }
    }
    // Mailbox-confined, <=64 * 960 stereo frames (245760 bytes). Keep older
    // generations until eviction so already-rendered receipts can still resolve.
    private var mediaOriginals: [MediaOriginal] = []
    // Capture-queue confined. Never participates in the live audio path.
    private var ioDiagnostic: JarvisRecordingIODiagnostic?

    /// Playback-only contract. Convert incoming media off the audio thread to
    /// this exact hardware format, preserving the original for later recording.
    /// A new hardware route requires a fresh attachment, not relabelled PCM.
    struct MediaInput: Sendable {
        let voiceGeneration: UUID
        let routeGeneration: UInt64
        let sourceGeneration: UInt64
        let sampleRate: Double
        let channels: Int
    }

    func attachMediaSource() -> MediaInput? {
        mailbox.lock(); defer { mailbox.unlock() }
        guard mediaReady, let current = handleMailbox, mediaSerial < UInt64.max else { return nil }
        mediaSerial += 1; activeMediaSerial = mediaSerial
        current.io.selectMediaSource(mediaSerial)
        applyMediaPolicyLocked()
        return MediaInput(voiceGeneration: voiceGeneration,
            routeGeneration: current.format.routeGeneration, sourceGeneration: mediaSerial,
            sampleRate: current.format.route.sampleRate, channels: current.format.outputChannels)
    }

    @discardableResult
    func enqueueMediaPCM16(_ data: Data, frames: Int, input: MediaInput, sourceFrame: Double = .nan) -> Bool {
        mailbox.lock(); defer { mailbox.unlock() }
        guard mediaReady, let current = handleMailbox,
              input.voiceGeneration == voiceGeneration,
              input.routeGeneration == current.format.routeGeneration,
              input.sourceGeneration == activeMediaSerial,
              input.sampleRate == current.format.route.sampleRate,
              input.channels == current.format.outputChannels,
              (1...1920).contains(frames), data.count == frames * input.channels * 2 else { return false }
        return data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return false }
            return current.io.pushMediaSamples(base.assumingMemoryBound(to: Int16.self),
                frames: UInt32(frames), generation: input.sourceGeneration, sourceFrame: sourceFrame)
        }
    }

    @discardableResult
    func cacheMediaPCM16(_ data: Data, sourceFrame: Int64, input: MediaInput) -> Bool {
        mailbox.lock(); defer { mailbox.unlock() }
        guard mediaReady, input.voiceGeneration == voiceGeneration,
              input.routeGeneration == handleMailbox?.format.routeGeneration,
              input.sourceGeneration == activeMediaSerial,
              sourceFrame >= 0, sourceFrame < (1 << 50),
              data.count % 4 == 0, (1...960).contains(data.count / 4) else { return false }
        if mediaOriginals.count == 64 { mediaOriginals.removeFirst() }
        mediaOriginals.append(MediaOriginal(generation: input.sourceGeneration, start: sourceFrame, pcm: data))
        return true
    }

    // Capture consumer only; no work here runs on the AudioUnit thread. A cache
    // miss stays a gap. Arrival time and packet boundaries never supply a PTS.
    private func resolveMedia(_ packet: JRDDuplexPacket, format: Format)
        -> ([JarvisMediaCaptureSpan], JarvisMediaCaptureCoverage) {
        var coverage = JarvisMediaCaptureCoverage()
        coverage.consumedOutputFrames = UInt64(packet.mediaOutputFrames)
        coverage.unknownOutputFrames = UInt64(packet.mediaUnknownFrames)
        coverage.receiptDroppedOutputFrames = UInt64(packet.mediaDroppedFrames)
        guard packet.timestampFlags & AudioTimeStampFlags.hostTimeValid.rawValue != 0 else {
            coverage.unknownOutputFrames = coverage.consumedOutputFrames
            return ([], coverage)
        }
        let rate = format.route.sampleRate, step = 48_000 / rate
        let host = CMClockMakeHostTimeFromSystemUnits(packet.hostTime)
        var result: [JarvisMediaCaptureSpan] = []
        mailbox.lock(); defer { mailbox.unlock() }
        for span in packet.mediaSpans {
            let gains = span.gains.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            guard !gains.isEmpty, gains.count <= 8192,
                  UInt64(span.outputOffset) + UInt64(gains.count) <= UInt64(packet.frameCount),
                  span.sourceFrame.isFinite, abs(span.sourceFrame) < Double(1 << 50),
                  gains.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
                coverage.unknownOutputFrames += UInt64(gains.count); continue
            }
            if span.sourceFrame < 0 {
                coverage.unknownOutputFrames += UInt64(min(gains.count, Int(ceil(-span.sourceFrame / step))))
            }
            var cursor = Int64(ceil(max(0, span.sourceFrame) - 0.0000001))
            let end = Int64(ceil(max(0, span.sourceFrame + Double(gains.count) * step) - 0.0000001))
            while cursor < end {
                guard let original = mediaOriginals.first(where: {
                    $0.generation == span.sourceGeneration && $0.start <= cursor && cursor < $0.end
                }) else {
                    let next = min(end, mediaOriginals.filter {
                        $0.generation == span.sourceGeneration && $0.start > cursor
                    }.map(\.start).min() ?? end)
                    coverage.missingSourceFrames += UInt64(next - cursor)
                    cursor = next; continue
                }
                let count = Int(min(end, original.end) - cursor)
                var pcm = [Int16](repeating: 0, count: count * 2)
                original.pcm.withUnsafeBytes { bytes in
                    for i in 0..<count {
                        let source = cursor + Int64(i)
                        let index = min(gains.count - 1, max(0, Int(floor((Double(source) - span.sourceFrame) / step))))
                        let gain = gains[index]
                        if gain == 0 { coverage.zeroGainSourceFrames += 1 }
                        for channel in 0..<2 {
                            let offset = Int(source - original.start) * 4 + channel * 2
                            let value = Int16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: Int16.self))
                            pcm[i * 2 + channel] = Int16(clamping: Int((Float(value) * gain).rounded(.toNearestOrEven))).littleEndian
                        }
                    }
                }
                let offset = Double(span.outputOffset) / rate + (Double(cursor) - span.sourceFrame) / 48_000
                let start = CMTimeAdd(host, CMTime(seconds: offset, preferredTimescale: 48_000_000))
                result.append(JarvisMediaCaptureSpan(sourceGeneration: span.sourceGeneration,
                    hostTime: CMClockConvertHostTimeToSystemUnits(start), frames: count,
                    pcm16Interleaved: pcm.withUnsafeBytes { Data($0) }))
                coverage.resolvedSourceFrames += UInt64(count)
                cursor += Int64(count)
            }
        }
        return (result, coverage)
    }

    func hasRenderedMedia(_ input: MediaInput) -> Bool {
        mailbox.lock(); defer { mailbox.unlock() }
        guard mediaReady, let current = handleMailbox,
              input.voiceGeneration == voiceGeneration,
              input.routeGeneration == current.format.routeGeneration,
              input.sourceGeneration == activeMediaSerial else { return false }
        return current.io.hasRenderedMediaSource(input.sourceGeneration)
    }

    @discardableResult
    func detachMediaSource(_ input: MediaInput) -> Bool {
        mailbox.lock(); defer { mailbox.unlock() }
        guard input.voiceGeneration == voiceGeneration,
              input.routeGeneration == handleMailbox?.format.routeGeneration,
              input.sourceGeneration == activeMediaSerial else { return false }
        activeMediaSerial = nil; handleMailbox?.io.selectMediaSource(0)
        return true
    }

    @discardableResult
    func setMediaVolume(_ volume: Float, muted: Bool) -> Bool {
        guard volume.isFinite, (0...1).contains(volume) else { return false }
        mailbox.lock(); defer { mailbox.unlock() }
        mediaGain = volume; mediaMuted = muted; applyMediaPolicyLocked()
        return true
    }

    func setMediaUserSpeaking(_ speaking: Bool) {
        mailbox.lock(); defer { mailbox.unlock() }
        mediaUserSpeaking = speaking
        handleMailbox?.io.setMediaUserSpeaking(speaking)
    }

    /// Future voiceover control uses its capture ID; unrelated ordinary
    /// recordings that exclude Jarvis do not automatically silence media.
    func setMediaOutputSuppressed(_ suppressed: Bool, recordingID: UUID) {
        mailbox.lock(); defer { mailbox.unlock() }
        if suppressed { mediaSuppressionIDs.insert(recordingID) }
        else { mediaSuppressionIDs.remove(recordingID) }
        applyMediaPolicyLocked()
    }

    private func applyMediaPolicyLocked() {
        // A Voice interruption closes its output gate while the microphone
        // remains open. Do not stop music for that; both gates closed is safety.
        handleMailbox?.io.setMedia(allowed: microphoneGate || outputGate,
            muted: mediaMuted || !mediaSuppressionIDs.isEmpty, gain: mediaGain)
        handleMailbox?.io.setMediaUserSpeaking(mediaUserSpeaking)
    }

    // ADM-thread state.
    private var deviceDelegate: RTCAudioDeviceDelegate?
    private var handle: Handle?
    private var preparedIO: JRDDuplexIO?
    private var leaseHeld = false
    private var authorized = false
    private var suspended = false
    private var retired = false
    private var teardownPending = false
    private var routeGeneration: UInt64 = 0
    private(set) var isInitialized = false
    private(set) var isPlayoutInitialized = false
    private(set) var isRecordingInitialized = false
    private(set) var isPlaying = false
    private(set) var isRecording = false
    // ADM preferences before hardware activation; replaced with actual values
    // and notifyAudio*ParametersChange BEFORE starting the hardware callbacks.
    private(set) var deviceInputSampleRate = 48_000.0
    private(set) var deviceOutputSampleRate = 48_000.0
    private(set) var inputIOBufferDuration = 0.01
    private(set) var outputIOBufferDuration = 0.01
    private(set) var inputNumberOfChannels = 1
    private(set) var outputNumberOfChannels = 1
    private(set) var inputLatency = 0.0
    private(set) var outputLatency = 0.0

    // Capture-queue state. Timer only DRAINS hardware-produced samples; it never
    // drives microphone capture, playout or WebRTC clocks.
    private var captureHandle: Handle?
    private var captureID: UUID?
    private var captureToken: UInt64 = 0
    private var nextToken: UInt64 = 0
    private var sink: Sink?
    private var timer: DispatchSourceTimer?
    private var lastCoverage: Coverage?

    init(voiceGeneration: UUID, sessionLease: SessionLease,
         onStatus: @escaping @Sendable (Status) -> Void) {
        self.voiceGeneration = voiceGeneration; self.lease = sessionLease; self.statusSink = onStatus
        super.init()
    }

    func initialize(with delegate: RTCAudioDeviceDelegate) -> Bool {
        guard !retired, !isInitialized else { return false }
        deviceDelegate = delegate; isInitialized = true
        mailbox.lock(); delegateMailbox = delegate; mailbox.unlock()
        return true
    }
    func initializePlayout() -> Bool {
        guard isInitialized, !retired else { return false }
        isPlayoutInitialized = true; return true
    }
    func initializeRecording() -> Bool {
        guard isInitialized, !retired else { return false }
        isRecordingInitialized = true; return true
    }
    func startPlayout() -> Bool {
        guard isPlayoutInitialized, !retired else { return false }
        isPlaying = true
        if reconcileHardware() { return true }
        isPlaying = false; handle?.io.setRequests(recording: isRecording, playout: false); return false
    }
    func startRecording() -> Bool {
        guard isRecordingInitialized, !retired else { return false }
        isRecording = true
        if reconcileHardware() { return true }
        isRecording = false; handle?.io.setRequests(recording: false, playout: isPlaying); return false
    }
    func stopPlayout() -> Bool {
        isPlaying = false; return reconcileHardware()
    }
    func stopRecording() -> Bool {
        isRecording = false; return reconcileHardware()
    }
    func terminateDevice() -> Bool {
        authorized = false; setGates(microphone: false, output: false)
        isPlaying = false; isRecording = false
        guard retireHardware(reason: "device_terminated") else { return false }
        retired = true; isInitialized = false; isPlayoutInitialized = false; isRecordingInitialized = false
        mailbox.lock(); delegateMailbox = nil; mailbox.unlock()
        deviceDelegate = nil
        captureQueue.async { [self] in
            _ = endSink(reason: "device_terminated")
            timer?.cancel(); timer = nil
        }
        return true
    }

    /// Core's replacement for relying on default-ADM isAudioEnabled. SDK start
    /// requests can arrive during peer construction: they are held until this
    /// explicit session authorization, without activating any hardware early.
    func authorizeSession(completion: @escaping Completion) {
        onADM(completion: completion) { self.authorized = true; return self.reconcileHardware() }
    }

    enum SessionControl: Equatable { case authorize, interruptionBegan, interruptionEnded, routeChanged, retire }

    /// Existing transport activation/retirement is synchronous. Never hold the
    /// RTCAudioSession configuration lock across this reentrant SDK dispatch.
    func controlSynchronously(_ command: SessionControl) -> Bool {
        mailbox.lock(); let target = delegateMailbox; mailbox.unlock()
        guard let target else { return command == .retire }
        if command != .authorize && command != .interruptionEnded {
            setGates(microphone: false, output: false)
        }
        var result = false
        target.dispatchSync { [self] in
            if command == .retire { result = terminateDevice(); return }
            guard isInitialized, !retired else { return }
            switch command {
            case .authorize: authorized = true; result = reconcileHardware()
            case .interruptionBegan: suspended = true; result = retireHardware(reason: "interruption_began")
            case .interruptionEnded: suspended = false; result = reconcileHardware()
            case .routeChanged: result = retireHardware(reason: "route_changed") && reconcileHardware()
            case .retire: break
            }
        }
        return result
    }

    /// Apply existing route/response fences here as well as track.isEnabled.
    /// This does NOT stop ambient recording; privacy-stop also detaches its sink.
    /// This atomic switch is not an acknowledged in-flight callback fence.
    func setGates(microphone: Bool, output: Bool) {
        mailbox.lock()
        microphoneGate = microphone; outputGate = output
        handleMailbox?.io.setGates(microphone: microphone,
                                  output: output && silentRecordingIDs.isEmpty)
        applyMediaPolicyLocked()
        mailbox.unlock()
    }

    /// Keep draining remote audio, but render silence during a recording that
    /// excludes Jarvis. Microphone delivery and the normal route gates remain
    /// independent; ending one capture cannot unmute another capture.
    func setRecordingOutputSuppressed(_ suppressed: Bool, recordingID: UUID) {
        mailbox.lock()
        if suppressed { silentRecordingIDs.insert(recordingID) }
        else { silentRecordingIDs.remove(recordingID) }
        handleMailbox?.io.setGates(microphone: microphoneGate,
                                  output: outputGate && silentRecordingIDs.isEmpty)
        mailbox.unlock()
    }

#if DEBUG
    /// Counter-only snapshot; never reads or stores microphone contents.
    func inputDiagnosticSnapshot() -> [String: Any] {
        mailbox.lock(); let target = delegateMailbox; mailbox.unlock()
        guard let target else { return ["hardware_present": false] }
        var snapshot: [String: Any] = ["hardware_present": false]
        // Serialize AudioUnit property reads with configure/close, not with
        // the real-time callbacks. No hardware settings change here.
        target.dispatchSync { [self] in
            guard let current = handle else { return }
            let counters = current.io.counters()
            snapshot = ["hardware_present": true,
                "route_generation": current.format.routeGeneration,
                "unit_state": current.io.controlDiagnosticSnapshot(),
                "input_callbacks": counters.inputCallbackCount,
                "rendered_input_frames": counters.renderedInputFrames,
                "input_delivery_calls": counters.inputDeliveryCount,
                "last_input_delivery_status": counters.lastInputDeliveryStatus,
                "input_peak_before_gate": counters.inputPeakBeforeGate,
                "external_input_frames": counters.externalInputFrames,
                "output_callbacks": counters.outputCallbackCount,
                "output_peak_after_gate": counters.outputPeakAfterGate,
                "playout_requested": counters.playoutRequested.boolValue,
                "output_gate_open": counters.outputGateOpen.boolValue,
                "output_errors": counters.outputErrors,
                "last_output_status": counters.lastOutputStatus,
                "recording_requested": counters.recordingRequested.boolValue,
                "microphone_gate_open": counters.microphoneGateOpen.boolValue,
                "microphone_errors": counters.microphoneErrors,
                "last_microphone_status": counters.lastMicrophoneStatus]
#if os(iOS)
            let session = AVAudioSession.sharedInstance()
            snapshot["session_sample_rate"] = session.sampleRate
            snapshot["session_input_channels"] = session.inputNumberOfChannels
            snapshot["session_output_channels"] = session.outputNumberOfChannels
            snapshot["session_io_buffer_duration"] = session.ioBufferDuration
            snapshot["session_category"] = session.category.rawValue
            snapshot["session_mode"] = session.mode.rawValue
            snapshot["session_category_options"] = session.categoryOptions.rawValue
            snapshot["session_output_volume"] = session.outputVolume
            snapshot["session_input_ports"] = session.currentRoute.inputs.map { $0.portType.rawValue }
            snapshot["session_output_ports"] = session.currentRoute.outputs.map { $0.portType.rawValue }
#endif
        }
        return snapshot
    }
#endif

    func interruptionBegan(completion: @escaping Completion) {
        setGates(microphone: false, output: false)
        onADM(completion: completion) {
            self.suspended = true
            return self.retireHardware(reason: "interruption_began")
        }
    }
    /// Core decides whether resumption is authorized by iOS/user/session state.
    /// Successful resume does not reopen gates; Core explicitly re-admits them.
    func interruptionEnded(allowResume: Bool, completion: @escaping Completion) {
        onADM(completion: completion) {
            guard allowResume else { return true }
            self.suspended = false
            return self.reconcileHardware()
        }
    }
    /// Call after Core has reconciled the route or media-services reset. Only
    /// a hardware change rebuilds the unit; recording changes never do so.
    func routeDidChange(completion: @escaping Completion) {
        setGates(microphone: false, output: false)
        onADM(completion: completion) {
            guard self.retireHardware(reason: "route_changed") else { return false }
            return self.reconcileHardware()
        }
    }

    /// One sink, on a dedicated utility queue. Sink must enqueue promptly, not
    /// synchronously call back into ADM. Slow sinks lose recording coverage but
    /// cannot block audio callbacks. Completion executes on the same queue.
    func attachSink(captureID: UUID, sink: @escaping Sink, completion: @escaping Completion) {
        onADM(completion: { ok in if !ok { completion(false) } }) {
            self.captureQueue.async { [self] in
                guard self.sink == nil, let current = captureHandle,
                      nextToken < UInt64.max, current.io.endCapture() else { completion(false); return }
                // Discard any previous detached-capture packets before baseline.
                for track in 0...1 { while current.io.takePacket(track) != nil {} }
                nextToken += 1; captureToken = nextToken
                self.captureID = captureID; self.sink = sink; lastCoverage = nil
                sink(.format(current.format)); emitCoverage(current)
                guard current.io.beginCapture(captureToken) else {
                    self.sink = nil; self.captureID = nil; completion(false); return
                }
                ioDiagnostic = JarvisRecordingIODiagnostic.configured(
                    sampleRate: current.format.route.sampleRate,
                    inputChannels: current.format.inputChannels, outputChannels: current.format.outputChannels,
                    context: ["capture_id": captureID.uuidString,
                              "voice_generation": current.format.voiceGeneration.uuidString,
                              "route_generation": current.format.routeGeneration,
                              "route_identity": current.format.route.identity,
                              "input_latency": current.format.route.inputLatency,
                              "output_latency": current.format.route.outputLatency,
                              "io_buffer_duration": current.format.route.ioBufferDuration])
                let source = DispatchSource.makeTimerSource(queue: captureQueue)
                source.schedule(deadline: .now(), repeating: .milliseconds(20))
                source.setEventHandler { [weak self] in self?.drainCapture() }
                timer = source; source.resume()
                completion(true)
            }
            return true
        }
    }
    func detachSink(captureID: UUID, completion: @escaping Completion) {
        captureQueue.async { [self] in
            guard self.captureID == captureID else { completion(false); return }
            completion(endSink(reason: "capture_detached"))
        }
    }

    private func onADM(completion: @escaping Completion, operation: @escaping @Sendable () -> Bool) {
        mailbox.lock(); let target = delegateMailbox; mailbox.unlock()
        guard let target else { captureQueue.async { completion(false) }; return }
        target.dispatchAsync { [self] in
            guard isInitialized, !retired else { captureQueue.async { completion(false) }; return }
            let ok = operation()
            captureQueue.async { completion(ok) }
        }
    }

    private func reconcileHardware() -> Bool {
        guard isInitialized, !retired else { return false }
        // A retained unit may have failed to stop, uninitialize, or dispose.
        // Never restart it or allocate another owner until close has succeeded.
        if teardownPending, !retireHardware(reason: "hardware_teardown_retry") { return false }
        handle?.io.setRequests(recording: isRecording, playout: isPlaying)
        guard authorized, !suspended, isRecording || isPlaying else {
            return retireHardware(reason: suspended ? "suspended" : "hardware_idle")
        }
        if let handle { return handle.io.start() == noErr }
        guard let delegate = deviceDelegate, routeGeneration < UInt64.max else { return false }
        do {
            routeGeneration += 1
            // Match native WebRTC: allocate the unit before activating the
            // session. These provisional values are replaced with the actual
            // route before configuring formats or starting any callback.
            guard let io = JRDDuplexIO(delegate: delegate, sampleRate: 48_000,
                                     inputChannels: 1, outputChannels: 1,
                                     routeGeneration: routeGeneration) else { throw DeviceError.allocation }
            preparedIO = io
            let prepared = io.prepare()
            guard prepared == noErr else { throw DeviceError.audioUnit(prepared) }
            let route = try lease.acquire(delegate.preferredInputSampleRate, delegate.preferredInputIOBufferDuration)
            leaseHeld = true
            guard route.sampleRate.isFinite, (8_000...192_000).contains(route.sampleRate),
                  route.inputChannels >= 1, route.outputChannels >= 1,
                  route.ioBufferDuration.isFinite, route.ioBufferDuration > 0,
                  route.inputLatency.isFinite, route.inputLatency >= 0,
                  route.outputLatency.isFinite, route.outputLatency >= 0 else { throw DeviceError.invalidRoute }
            // Interleaved PCM16, at the active route's rate; use up to two
            // channels and let the AudioUnit negotiate the hardware boundary.
            let inputs = min(2, route.inputChannels), outputs = min(2, route.outputChannels)
            let format = Format(voiceGeneration: voiceGeneration, routeGeneration: routeGeneration,
                                route: route, inputChannels: inputs, outputChannels: outputs)
            let created = Handle(io, format)
            // Retain even a partially configured unit until close succeeds.
            handle = created
            preparedIO = nil
            mailbox.lock(); handleMailbox = created
            io.setGates(microphone: microphoneGate,
                        output: outputGate && silentRecordingIDs.isEmpty); mailbox.unlock()
            let status = io.configure(sampleRate: route.sampleRate, inputChannels: inputs, outputChannels: outputs)
            guard status == noErr else { throw DeviceError.audioUnit(status) }
            deviceInputSampleRate = io.sampleRate; deviceOutputSampleRate = io.sampleRate
            inputNumberOfChannels = io.inputChannels; outputNumberOfChannels = io.outputChannels
            inputIOBufferDuration = route.ioBufferDuration; outputIOBufferDuration = route.ioBufferDuration
            inputLatency = route.inputLatency; outputLatency = route.outputLatency
            // The SDK requires the wrapper even when already on its ADM thread.
            delegate.dispatchSync {
                delegate.notifyAudioInputParametersChange()
                delegate.notifyAudioOutputParametersChange()
            }
            io.setRequests(recording: isRecording, playout: isPlaying)
            captureQueue.async { [self] in
                captureHandle = created; lastCoverage = nil
                if let sink, let captureID {
                    sink(.boundary(captureID: captureID, routeGeneration: format.routeGeneration,
                                   reason: "hardware_generation_started"))
                    sink(.format(format)); emitCoverage(created)
                    if !created.io.beginCapture(captureToken) { _ = endSink(reason: "capture_attach_failed") }
                }
            }
            let started = io.start()
            guard started == noErr else { throw DeviceError.audioUnit(started) }
            mailbox.lock(); mediaReady = true; applyMediaPolicyLocked(); mailbox.unlock()
            Self.setActiveExternalIO(io)
            report("hardware_started")
            return true
        } catch {
            let code: Int32?
            if case DeviceError.audioUnit(let value) = error { code = value } else { code = nil }
            report("hardware_start_failed", code: code)
            _ = retireHardware(reason: "hardware_start_failed")
            return false
        }
    }

    private func retireHardware(reason: String) -> Bool {
        // A failed lease acquisition still owns the unit prepared before it.
        // Preserve failed-close ownership just as for an initialized handle.
        if let pending = preparedIO {
            guard pending.close() else {
                teardownPending = true
                report("hardware_fence_failed")
                return false
            }
            preparedIO = nil
            teardownPending = false
        }
        if let current = handle {
            mailbox.lock()
            mediaReady = false; activeMediaSerial = nil
            current.io.selectMediaSource(0)
            mailbox.unlock()
            Self.clearActiveExternalIO(current.io)
            current.io.setRequests(recording: false, playout: false)
            guard current.io.close() else {
                teardownPending = true
                report("hardware_fence_failed")
                return false
            }
            teardownPending = false
            // Thread continuity has ended; tell native ADM before any callbacks
            // from a replacement route's possibly different hardware thread.
            if let delegate = deviceDelegate {
                delegate.dispatchSync {
                    delegate.notifyAudioInputInterrupted()
                    delegate.notifyAudioOutputInterrupted()
                }
            }
            mailbox.lock(); handleMailbox = nil; mailbox.unlock()
            handle = nil
            captureQueue.async { [self] in
                drainCapture()
                ioDiagnostic?.finish(reason: "hardware_boundary_" + reason)
                ioDiagnostic = nil
                if let sink, let captureID {
                    sink(.boundary(captureID: captureID, routeGeneration: current.format.routeGeneration, reason: reason))
                }
                captureHandle = nil; lastCoverage = nil
            }
        }
        if leaseHeld { lease.release(); leaseHeld = false }
        report(reason)
        return true
    }

    private func drainCapture() {
        guard let current = captureHandle, let sink, let captureID else { return }
        // Bounded consumer work per tick even if a producer keeps filling.
        for track in 0...1 {
            for _ in 0..<32 {
                guard let packet = current.io.takePacket(track) else { break }
                guard packet.captureToken == captureToken else { continue }
                if let ioDiagnostic {
                    if let timing = packet.diagnosticTiming,
                       let pcm = track == 0 ? packet.pcm16 : packet.diagnosticPlaybackPCM16 {
                        var metadata: [String: Any] = timing
                        metadata["host_time"] = packet.hostTime
                        metadata["sample_time"] = packet.sampleTime
                        metadata["timestamp_flags"] = packet.timestampFlags
                        metadata["voice_gate_open"] = packet.softwareGateOpen
                        metadata["capture_token"] = packet.captureToken
                        ioDiagnostic.append(track: track, pcm16: pcm, frameCount: Int(packet.frameCount), metadata: metadata)
                    } else {
                        ioDiagnostic.finish(reason: "native_sidecar_unavailable")
                    }
                }
                let media = track == 1 ? resolveMedia(packet, format: current.format) : ([], JarvisMediaCaptureCoverage())
                sink(.pcm(PCMBlock(captureID: captureID, format: current.format,
                    track: track == 0 ? .microphone : .renderedOutput,
                    pcm16Interleaved: packet.pcm16, frames: packet.frameCount,
                    hostTime: packet.hostTime, sampleTime: packet.sampleTime,
                    timestampFlags: packet.timestampFlags, softwareGateOpen: packet.softwareGateOpen,
                    mediaSpans: media.0, mediaCoverage: media.1)))
            }
        }
        emitCoverage(current)
    }
    private func emitCoverage(_ current: Handle) {
        let c = current.io.counters()
        let value = Coverage(routeGeneration: current.format.routeGeneration,
            microphoneDroppedPackets: c.microphoneDroppedPackets, microphoneDroppedFrames: c.microphoneDroppedFrames,
            outputDroppedPackets: c.outputDroppedPackets, outputDroppedFrames: c.outputDroppedFrames,
            microphoneErrors: c.microphoneErrors, outputErrors: c.outputErrors,
            lastMicrophoneStatus: c.lastMicrophoneStatus, lastOutputStatus: c.lastOutputStatus)
        if lastCoverage != value {
            ioDiagnostic?.event(["type": "coverage", "microphone_dropped_frames": c.microphoneDroppedFrames,
                                 "output_dropped_frames": c.outputDroppedFrames,
                                 "microphone_errors": c.microphoneErrors, "output_errors": c.outputErrors])
            sink?(.coverage(value)); lastCoverage = value
        }
    }
    private func endSink(reason: String) -> Bool {
        let fenced = captureHandle?.io.endCapture() ?? true
        drainCapture()
        ioDiagnostic?.finish(reason: fenced ? reason : "capture_fence_failed")
        ioDiagnostic = nil
        if let captureID {
            sink?(.boundary(captureID: captureID, routeGeneration: captureHandle?.format.routeGeneration,
                            reason: fenced ? reason : "capture_fence_failed"))
        }
        sink = nil; captureID = nil; captureToken = 0; lastCoverage = nil
        timer?.cancel(); timer = nil
        return fenced
    }
    private func report(_ reason: String, code: Int32? = nil) {
        let status = Status(voiceGeneration: voiceGeneration, routeGeneration: routeGeneration,
                            reason: reason, osStatus: code)
        captureQueue.async { [statusSink] in statusSink(status) }
    }
    private enum DeviceError: Error { case invalidRoute, allocation, audioUnit(Int32) }
}
