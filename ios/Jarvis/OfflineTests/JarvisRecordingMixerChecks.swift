@preconcurrency import AVFoundation
import CoreAudio
import CoreVideo
import Foundation

@main
struct RecordingMixerChecks {
    typealias Mixer = JarvisRecordingMixer
    static let origin = CMTime(value: 120, timescale: 1)
    static let flags = AudioTimeStampFlags.hostTimeValid.rawValue | AudioTimeStampFlags.sampleTimeValid.rawValue
    static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError(message) }
    }
    static func ticks(_ ms: Int64) -> UInt64 {
        CMClockConvertHostTimeToSystemUnits(CMTimeAdd(origin, CMTime(value: ms, timescale: 1_000)))
    }
    static func timeline(_ ms: Int = 60_000) throws -> JarvisRecordingTimeline {
        try JarvisRecordingTimeline(startHostTime: ticks(0), maximumDurationMilliseconds: ms,
            videoClock: .dat07PhoneHostUnqualified, maximumVideoDeliveryDelay: CMTime(value: 2, timescale: 1))
    }
    static func input(_ track: Mixer.Track, at ms: Int64, rate: Int = 48_000,
                      route: UInt64 = 1, channels: Int = 1, amplitude: Float = 0.2,
                      frequency: Double = 0) -> Mixer.Input {
        let frames = rate / 100
        var data = [Int16]()
        for frame in 0..<frames {
            let time = Double(ms) / 1_000 + Double(frame) / Double(rate)
            let value = frequency == 0 ? Double(amplitude) : Double(amplitude) * sin(2 * Double.pi * frequency * time)
            for _ in 0..<channels { data.append(Int16((value * 32_767).rounded()).littleEndian) }
        }
        return Mixer.Input(track: track, routeGeneration: route, hostTime: ticks(ms), timestampFlags: flags,
            sampleTime: Double(ms) * Double(rate) / 1_000, sampleRate: Double(rate), channels: channels,
            frames: frames, pcm16Interleaved: data.withUnsafeBytes { Data($0) })
    }
    static func expect(_ expected: Mixer.MixerError, _ action: () throws -> Void) {
        do { try action(); fatalError("Expected \(expected)") }
        catch let error as Mixer.MixerError { require(error == expected, "Wrong mixer error") }
        catch { fatalError("Unexpected error: \(error)") }
    }

    static func main() async throws {
        var mixed: [Mixer.Chunk] = []
        let both = Mixer(timeline: try timeline(), includeJarvisAudio: true) { mixed.append($0) }
        try both.offer(input(.microphone, at: 0, channels: 2, amplitude: 0.2))
        try both.drain(through: 480)
        require(mixed.isEmpty, "Committed before other track was available")
        try both.offer(input(.renderedOutput, at: 0, amplitude: 0.3))
        try both.drain(through: 480)
        require(mixed.flatMap(\.samples).count == 480, "Mixed frame count")
        require(mixed.flatMap(\.samples).allSatisfy { abs($0 - 0.5) < 0.0001 }, "Include/downmix value")
        try both.finish(through: 480)
        var excluded: [Mixer.Chunk] = []
        let micOnly = Mixer(timeline: try timeline(), includeJarvisAudio: false) { excluded.append($0) }
        try micOnly.offer(input(.renderedOutput, at: 0, amplitude: 0.9))
        try micOnly.offer(input(.microphone, at: 0, amplitude: 0.2))
        try micOnly.finish(through: 480)
        require(excluded.flatMap(\.samples).allSatisfy { abs($0 - 0.2) < 0.0001 }, "Jarvis output leaked into excluded track")
        print("PASS 1: actual PCM downmix, include/exclude, and bounded-wait coverage")

        for rate in [16_000, 44_100, 48_000] {
            var converted: [Mixer.Chunk] = []
            let mixer = Mixer(timeline: try timeline(), includeJarvisAudio: false) { converted.append($0) }
            for ms in stride(from: Int64(0), to: 1_000, by: 10) {
                try mixer.offer(input(.microphone, at: ms, rate: rate, frequency: 440))
                try mixer.drain(through: (ms + 10) * 48)
            }
            try mixer.finish(through: 48_000)
            let samples = converted.flatMap(\.samples)
            require(samples.count == 48_000, "Persistent SRC duration at \(rate): \(samples.count)")
            require(mixer.evidence.lateConvertedFrames == 0 && mixer.evidence.duplicateConvertedFrames == 0,
                    "Per-packet resampling drift")
            let correlation = zip(samples[480..<47_520], 480..<47_520).reduce(0.0) { sum, pair in
                sum + Double(pair.0) * sin(2 * Double.pi * 440 * Double(pair.1) / 48_000)
            } / Double(47_040)
            require(correlation > 0.09, "SRC phase or signal lost at \(rate): \(correlation)")
        }
        print("PASS 2: real streaming Apple SRC at16k/44.1k/48k preserves cumulative duration and phase")

        var gaps: [Mixer.Chunk] = []
        let restarted = Mixer(timeline: try timeline(), includeJarvisAudio: false) { gaps.append($0) }
        try restarted.offer(input(.microphone, at: 100, rate: 16_000))
        try restarted.offer(input(.microphone, at: 500, rate: 48_000, route: 2))
        expect(.staleRoute) { _ = try restarted.offer(input(.microphone, at: 600, rate: 16_000)) }
        expect(.outOfOrder) { _ = try restarted.offer(input(.microphone, at: 500, route: 2)) }
        try restarted.finish(through: 28_800)
        require(gaps.map(\.startFrame) == [4_800, 24_000], "Route gap or origin lost")
        require(restarted.evidence.missingMicrophoneFrames == 27_840, "Explicit route-gap coverage")
        require(restarted.evidence.routeChanges == 1, "Route boundary not recorded")
        print("PASS 3: route changes preserve original offsets and gaps; stale/duplicate input rejected")

        let bounded = Mixer(timeline: try timeline(), includeJarvisAudio: true) { _ in }
        expect(.bufferFull) { _ = try bounded.offer(input(.microphone, at: 2_000)) }
        try bounded.offer(input(.microphone, at: 0))
        try bounded.drain(through: 480, expiringMissing: true)
        require(bounded.evidence.missingRenderedFrames == 480, "Expired wait invented playback data")
        expect(.lateInput) { _ = try bounded.offer(input(.renderedOutput, at: 0)) }
        try bounded.finish(through: 480)
        expect(.closed) { _ = try bounded.offer(input(.microphone, at: 10)) }
        print("PASS 4: bounded storage, explicit missing-coverage expiry and closed fences")

        var highRate: [Mixer.Chunk] = []
        let selected = Mixer(timeline: try timeline(), includeJarvisAudio: false, includeMediaAudio: true) { highRate.append($0) }
        try selected.offer(input(.microphone, at: 0, amplitude: 0))
        try selected.offer(input(.renderedOutput, at: 0, amplitude: 0.9))
        try selected.offer(input(.media, at: 0, channels: 2, frequency: 12000))
        try selected.drain(through: 480)
        require(highRate.flatMap(\.samples).count == 480, "Media contribution did not drain with microphone")
        let energy = highRate.flatMap(\.samples).reduce(0.0) { $0 + Double($1 * $1) } / 480
        require(energy > 0.019 && energy < 0.021, "Original 12kHz media bandwidth was lost or excluded voice leaked")
        try selected.offer(input(.microphone, at: 10, amplitude: 0))
        try selected.drain(through: 960)
        require(highRate.flatMap(\.samples).count == 960, "Absent media stalled ordinary drain")
        require(selected.evidence.mediaFrames == 480 && selected.evidence.renderedFrames == 0, "Media coverage")
        try selected.finish(through: 960)
        print("PASS: optional original-rate additive media retains 12kHz content, excludes voice and never requires absent media")

        try await rejectedFileLimit()
        try await realEncode()
    }

    static func videoFrame(at time: CMTime) throws -> CMSampleBuffer {
        var pixel: CVPixelBuffer?
        require(CVPixelBufferCreate(kCFAllocatorDefault, 32, 32, kCVPixelFormatType_32BGRA,
                                   nil, &pixel) == kCVReturnSuccess, "Pixel allocation")
        CVPixelBufferLockBaseAddress(pixel!, [])
        if let bytes = CVPixelBufferGetBaseAddress(pixel!) { memset(bytes, 100, CVPixelBufferGetDataSize(pixel!)) }
        CVPixelBufferUnlockBaseAddress(pixel!, [])
        var format: CMVideoFormatDescription?
        require(CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
            imageBuffer: pixel!, formatDescriptionOut: &format) == noErr, "Video format")
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 24),
                                       presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        require(CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault,
            imageBuffer: pixel!, formatDescription: format!, sampleTiming: &timing,
            sampleBufferOut: &sample) == noErr, "Video sample")
        return sample!
    }

    static func rejectedFileLimit() async throws {
        for limit in [0, JarvisRecordingWriter.maximumFileBytes + 1] {
            var invalid = JarvisRecordingWriter.Configuration(width: 32, height: 32)
            invalid.maximumFileBytes = limit
            do { try invalid.validate(); fatalError("Invalid file policy accepted") }
            catch JarvisRecordingWriter.Reason.invalidConfiguration { }
        }
        var configuration = JarvisRecordingWriter.Configuration(width: 32, height: 32)
        configuration.maximumFileBytes = 1
        let writer = try await JarvisRecordingWriter.make(configuration: configuration)
        let sample = try videoFrame(at: .zero)
        require(writer.offerVideo(sample, generationID: writer.generationID) == .queued, "Bounded video admission")
        require(writer.offerAudio([Float](repeating: 0.1, count: 480), at: .zero,
                                  generationID: writer.generationID) == .queued, "Bounded audio admission")
        do {
            _ = try await writer.finish(at: CMTime(value: 1, timescale: 20), generationID: writer.generationID)
            fatalError("Oversized encoded output was accepted")
        } catch let failure as JarvisRecordingWriter.Failure {
            require(failure.reason == .fileFailed, "Wrong byte-limit failure")
        }
        print("PASS 5: per-recording byte limits validate and reject actual oversized encoded output")
    }

    static func realEncode() async throws {
        guard let directory = ProcessInfo.processInfo.environment["JARVIS_RECORDING_TEST_OUTPUT"] else {
            fatalError("Explicit private artifact directory required")
        }
        let timeline = try timeline(2_000)
        let writer = try await JarvisRecordingWriter.make(configuration: .init(width: 32, height: 32))
        var audioRejected = false
        let mixer = Mixer(timeline: timeline, includeJarvisAudio: true) { chunk in
            if writer.offerAudio(chunk.samples, at: chunk.presentationTime, generationID: writer.generationID) != .queued {
                audioRejected = true
            }
        }
        if ProcessInfo.processInfo.environment["JARVIS_RECORDING_ECHO_TEST"] == "1" {
            mixer.enableEchoReduction()
            require(mixer.isEchoReductionEnabled, "Echo encode check must use the real processor")
        }
        var nextVideo = 0
        for ms in stride(from: Int64(0), to: 2_000, by: 10) {
            while nextVideo < 48, nextVideo * 1_000 / 24 <= Int(ms) {
                let pts = CMTimeAdd(origin, CMTime(value: Int64(nextVideo), timescale: 24))
                let original = try videoFrame(at: pts)
                let mapped = try timeline.videoCopy(original, receivedHostTime: ticks(ms + 100))!
                require(writer.offerVideo(mapped, generationID: writer.generationID) == .queued, "Video not queued")
                nextVideo += 1
            }
            // Declared leading/trailing50ms and one middle50ms gap on both
            // actual synthetic inputs. Do not send zero packets to hide gaps.
            if ms >= 50, ms < 1_950, !(800..<850).contains(ms) {
                try mixer.offer(input(.microphone, at: ms, rate: 16_000, frequency: 440))
                try mixer.offer(input(.renderedOutput, at: ms, frequency: 880))
            }
            try mixer.drain(through: max(0, (ms - 100) * 48))
            try await Task.sleep(for: .milliseconds(3))
        }
        try mixer.finish(through: 96_000)
        require(!audioRejected, "Writer rejected mixed audio")
        let completed = try await writer.finish(at: CMTime(value: 2, timescale: 1), generationID: writer.generationID)
        let artifact = URL(fileURLWithPath: directory).appendingPathComponent("mixed-recording.mp4")
        try FileManager.default.copyItem(at: completed.file.urlForReading(), to: artifact)
        completed.file.discard()
        require(completed.evidence.video.appended == 48 && completed.evidence.video.dropped == 0, "Video encoding coverage")
        require(completed.evidence.sourceAudioFrames == 88_800 && completed.evidence.paddedAudioFrames == 7_200,
                "Actual mixed source/padding counts: \(completed.evidence.sourceAudioFrames)/\(completed.evidence.paddedAudioFrames)")

        let asset = AVURLAsset(url: artifact)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        require(tracks.count == 1, "Expected one mixed AAC track")
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: tracks[0], outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMIsFloatKey: true,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsNonInterleaved: false])
        reader.add(output); require(reader.startReading(), "Decode start")
        var decoded: [Float] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sample) else { fatalError("Decoded audio bytes missing") }
            let length = CMBlockBufferGetDataLength(block)
            var bytes = [Float](repeating: 0, count: length / 4)
            let status = bytes.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!)
            }
            require(status == noErr, "Decode copy"); decoded.append(contentsOf: bytes)
        }
        require(reader.status == .completed && decoded.count >= 96_000, "Decoded timeline too short")
        func rms(_ lo: Int, _ hi: Int) -> Double {
            sqrt(decoded[lo..<hi].reduce(0.0) { $0 + Double($1 * $1) } / Double(hi - lo))
        }
        require(rms(39_360, 40_320) < 0.015, "Declared gap was compressed or not silent")
        require(rms(43_200, 48_000) > 0.10, "Audio after gap missing")
        for frequency in [440.0, 880.0] {
            let magnitude = (9_600..<33_600).reduce(0.0) {
                $0 + Double(decoded[$1]) * sin(2 * Double.pi * frequency * Double($1) / 48_000)
            } / 24_000
            require(magnitude > 0.07, "Mixed tone absent or shifted: \(frequency) \(magnitude)")
        }
        print("PASS 6: actual H264/AAC encode/decode,48video frames,88800source+7200padding, both tones and gap retained")
        print("Artifact: \(artifact.path)")
        print("PASS: 6 mixer/integration groups; synthetic PCM and video, no physical capture/hearing claim")
    }
}
