import AVFoundation
import Foundation

/// Test sink only: production network parsing/conversion is compiled unchanged.
/// This does not claim AudioUnit, phone, DAT or physical playback acceptance.
final class JarvisDuplexAudioDevice: @unchecked Sendable {
    struct MediaInput { let sampleRate = 16_000.0; let channels = 1 }
    private let lock = NSLock()
    private var frames = 0, packets = 0, nonzero = 0
    private var detached = false
    func attachMediaSource() -> MediaInput? { MediaInput() }
    func detachMediaSource(_ input: MediaInput) -> Bool { lock.withLock { detached = true }; return true }
    func setMediaVolume(_ volume: Float, muted: Bool) -> Bool { true }
    // This receiver fixture has no hardware renderer. Queueing is not playing.
    func hasRenderedMedia(_ input: MediaInput) -> Bool { false }
    func cacheMediaPCM16(_ data: Data, sourceFrame: Int64, input: MediaInput) -> Bool {
        sourceFrame >= 0 && data.count > 0 && data.count <= 3840 && data.count % 4 == 0
    }
    func enqueueMediaPCM16(_ data: Data, frames count: Int, input: MediaInput, sourceFrame: Double = .nan) -> Bool {
        guard data.count == count * 2, count > 0, count <= 1920 else { return false }
        let countNonzero = data.withUnsafeBytes { $0.bindMemory(to: Int16.self).filter { $0 != 0 }.count }
        lock.withLock { frames += count; packets += 1; nonzero += countNonzero }
        return true
    }
    func evidence() -> [String: Any] {
        lock.withLock { ["frames": frames, "packets": packets, "nonzero_samples": nonzero,
                        "sample_rate": 16000, "channels": 1, "detached": detached,
                        "passed": frames > 16000 && nonzero > 1000 && detached] }
    }
}

@main struct MediaTransportChecks {
    static func main() async throws {
        if CommandLine.arguments.count == 1 { try offline(); return }
        guard CommandLine.arguments.count == 3,
              let url = URL(string: CommandLine.arguments[1]) else { fatalError("URL and receipt path required") }
        let device = JarvisDuplexAudioDevice()
        let task = Task {
            try await JarvisMediaStream().run(request: URLRequest(url: url), device: device, volume: 0.65, receiving: {})
        }
        try await Task.sleep(for: .seconds(8))
        task.cancel()
        let outcome = await task.result
        var result = device.evidence()
        if case .failure(let error) = outcome, !(error is CancellationError),
           (error as NSError).code != NSURLErrorCancelled {
            result["error_type"] = String(describing: type(of: error))
            result["error"] = String(describing: error)
            result["passed"] = false
        }
        let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
        print(String(decoding: data, as: UTF8.self))
        if result["passed"] as? Bool != true { exit(1) }
    }

    static func offline() throws {
        let payload = Data([0x80, 0xe0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0x20, 0, 0xe0, 0])
        let packet = try JarvisMediaStream.Packet(payload)
        precondition(packet.frames == 1 && packet.samples == [8192, -8192])
        for rate in [8000.0, 16000.0, 24000.0, 44100.0, 48000.0] {
            let sourceFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48000, channels: 2, interleaved: true)!
            let destination = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: rate, channels: 1, interleaved: true)!
            let converter = AVAudioConverter(from: sourceFormat, to: destination)!
            converter.primeMethod = .none
            var inputFrames = 0, outputFrames: Int64 = 0
            var peak = 0, peakSource = 0.0
            for count in [137, 823, 333, 627, 960, 960, 960] {
                let source = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(count))!
                source.frameLength = AVAudioFrameCount(count)
                let samples = source.int16ChannelData![0]
                for i in 0..<count {
                    let value: Int16 = inputFrames + i == 1500 ? 25000 : 0
                    samples[i * 2] = value; samples[i * 2 + 1] = value
                }
                let output = AVAudioPCMBuffer(pcmFormat: destination,
                    frameCapacity: AVAudioFrameCount(ceil(Double(count) * rate / 48000)) + 32)!
                var supplied = false, error: NSError?
                let status = converter.convert(to: output, error: &error) { _, state in
                    if supplied { state.pointee = .noDataNow; return nil }
                    supplied = true; state.pointee = .haveData; return source
                }
                precondition(error == nil && status != .error)
                let phase = JarvisMediaStream.recordingSourceFrame(outputFrames: outputFrames,
                    outputRate: rate, trailingFrames: converter.primeInfo.trailingFrames)
                for i in 0..<Int(output.frameLength) {
                    let magnitude = abs(Int(output.int16ChannelData![0][i]))
                    if magnitude > peak { peak = magnitude; peakSource = phase + Double(i) * 48000 / rate }
                }
                outputFrames += Int64(output.frameLength); inputFrames += count
            }
            precondition(peak > 0 && abs(peakSource - 1500) <= max(3, 48000 / rate),
                         "Cumulative converter phase/latency mismatch at \(rate): \(peakSource)")
            print("PASS: cumulative 48k source phase with irregular input chunks at \(Int(rate))Hz, impulse source error \(peakSource - 1500)")
        }
        print("PASS: offline RTP parser and production conversion-phase helper; no socket or credentials")
    }
}
