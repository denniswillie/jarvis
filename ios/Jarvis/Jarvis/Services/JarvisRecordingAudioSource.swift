import Foundation

/// Source-rate PCM resolved only from a native output-consumption receipt.
struct JarvisMediaCaptureSpan: Sendable {
    let sourceGeneration: UInt64
    let hostTime: UInt64
    let frames: Int
    let pcm16Interleaved: Data // 48 kHz stereo little-endian, actual gain applied.
}

struct JarvisMediaCaptureCoverage: Codable, Sendable {
    var consumedOutputFrames: UInt64 = 0
    var resolvedSourceFrames: UInt64 = 0
    var missingSourceFrames: UInt64 = 0
    var unknownOutputFrames: UInt64 = 0
    var receiptDroppedOutputFrames: UInt64 = 0
    var zeroGainSourceFrames: UInt64 = 0
    var admittedSourceFrames: UInt64 = 0
    var rejectedSourceFrames: UInt64 = 0
    var recordedSourceFrames: UInt64 = 0
    var outputCaptureDroppedFrames: UInt64 = 0

    mutating func add(_ other: Self) {
        consumedOutputFrames += other.consumedOutputFrames
        resolvedSourceFrames += other.resolvedSourceFrames
        missingSourceFrames += other.missingSourceFrames
        unknownOutputFrames += other.unknownOutputFrames
        receiptDroppedOutputFrames += other.receiptDroppedOutputFrames
        zeroGainSourceFrames += other.zeroGainSourceFrames
    }
}

/// Recording-only subscription to an existing audio owner. No engine/session
/// creation, microphone permissions, playback gates, or speech interruption API.
/// Callbacks for an attachment must be ordered. Attach completion is exactly
/// once; detach(true) means all accepted callbacks have been delivered. A late
/// attach completion remains required after a caller's timeout/cancellation.
struct JarvisRecordingAudioSource: Sendable {
    struct Format: Sendable {
        let voiceGeneration: UUID
        let routeGeneration: UInt64
        let sampleRate: Double
        let inputChannels: Int
        let outputChannels: Int
    }
    struct PCM: Sendable {
        enum Track: Sendable { case microphone, renderedOutput }
        let captureID: UUID
        let format: Format
        let track: Track
        let pcm16Interleaved: Data
        let frames: Int
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
    enum Event: Sendable {
        case format(Format)
        case pcm(PCM)
        case coverage(Coverage)
        case boundary(captureID: UUID, routeGeneration: UInt64?, reason: String,
                      terminal: Bool, consumerFenced: Bool)
    }
    typealias Sink = @Sendable (Event) -> Void
    typealias Completion = @Sendable (Bool) -> Void
    let voiceGeneration: UUID
    let attach: @Sendable (UUID, @escaping Sink, @escaping Completion) -> Void
    let detach: @Sendable (UUID, @escaping Completion) -> Void
}

// Real production wrapper. Host checks inject closures rather than pretending
// that a timer/mock is a hardware-backed RTCAudioDevice.
#if canImport(WebRTC) && !JARVIS_RECORDING_CAPTURE_HOST_TESTS
extension JarvisRecordingAudioSource {
    init(device: JarvisDuplexAudioDevice) {
        self.init(voiceGeneration: device.voiceGeneration, attach: { id, sink, done in
            device.attachSink(captureID: id, sink: { event in
                func format(_ f: JarvisDuplexAudioDevice.Format) -> Format {
                    Format(voiceGeneration: f.voiceGeneration, routeGeneration: f.routeGeneration,
                        sampleRate: f.route.sampleRate, inputChannels: f.inputChannels, outputChannels: f.outputChannels)
                }
                switch event {
                case .format(let f): sink(.format(format(f)))
                case .pcm(let p):
                    sink(.pcm(PCM(captureID: p.captureID, format: format(p.format),
                        track: p.track == .microphone ? .microphone : .renderedOutput,
                        pcm16Interleaved: p.pcm16Interleaved, frames: Int(p.frames), hostTime: p.hostTime,
                        sampleTime: p.sampleTime, timestampFlags: p.timestampFlags, softwareGateOpen: p.softwareGateOpen,
                        mediaSpans: p.mediaSpans, mediaCoverage: p.mediaCoverage)))
                case .coverage(let c):
                    sink(.coverage(Coverage(routeGeneration: c.routeGeneration,
                        microphoneDroppedPackets: c.microphoneDroppedPackets, microphoneDroppedFrames: c.microphoneDroppedFrames,
                        outputDroppedPackets: c.outputDroppedPackets, outputDroppedFrames: c.outputDroppedFrames,
                        microphoneErrors: c.microphoneErrors, outputErrors: c.outputErrors,
                        lastMicrophoneStatus: c.lastMicrophoneStatus, lastOutputStatus: c.lastOutputStatus)))
                case .boundary(let captureID, let route, let reason):
                    // A route replacement can continue on the same H0. A speech
                    // interruption only changes output gates, not these boundaries.
                    let transient = reason == "route_changed" || reason == "hardware_generation_started"
                    let fenced = reason == "capture_detached" || reason == "device_terminated"
                    sink(.boundary(captureID: captureID, routeGeneration: route, reason: reason,
                                   terminal: !transient, consumerFenced: fenced))
                }
            }, completion: done)
        }, detach: { id, done in device.detachSink(captureID: id, completion: done) })
    }
}
#endif
