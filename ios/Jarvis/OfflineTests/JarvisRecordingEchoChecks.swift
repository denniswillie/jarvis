import AVFoundation
import CoreAudio
import Foundation

@main
struct RecordingEchoChecks {
    typealias Mixer = JarvisRecordingMixer
    static let rate = 48_000
    static let origin = CMTime(value: 120, timescale: 1)
    static let flags = AudioTimeStampFlags.hostTimeValid.rawValue | AudioTimeStampFlags.sampleTimeValid.rawValue

    static func require(_ value: @autoclosure () -> Bool, _ message: String) {
        if !value() { fatalError(message) }
    }
    static func timeline() throws -> JarvisRecordingTimeline {
        try .init(startHostTime: CMClockConvertHostTimeToSystemUnits(origin), maximumDurationMilliseconds: 30_000,
                  videoClock: .dat07PhoneHostUnqualified, maximumVideoDeliveryDelay: CMTime(value: 2, timescale: 1))
    }
    static func input(_ track: Mixer.Track, start: Int, values: [Float], route: UInt64 = 1) -> Mixer.Input {
        let pcm = values.map { Int16(max(-32768, min(32767, ($0 * 32768).rounded()))).littleEndian }
        return .init(track: track, routeGeneration: route,
                     hostTime: CMClockConvertHostTimeToSystemUnits(CMTimeAdd(origin, CMTime(value: Int64(start), timescale: 48_000))),
                     timestampFlags: flags, sampleTime: Double(start), sampleRate: 48_000,
                     channels: 1, frames: values.count, pcm16Interleaved: pcm.withUnsafeBytes { Data($0) })
    }
    static func energy(_ samples: [Float]) -> Double { samples.reduce(0) { $0 + Double($1) * Double($1) } }

    static func main() throws {
        // No playback: irregular 3/7/11 ms chunks must remain bit-exact after
        // input quantization, including a final incomplete 10 ms DSP block.
        var chunks: [Mixer.Chunk] = []
        let plain = Mixer(timeline: try timeline(), includeJarvisAudio: false) { chunks.append($0) }
        plain.enableEchoReduction()
        require(plain.isEchoReductionEnabled, "The production C filter must be linked for these tests")
        var position = 0
        for length in [144, 336, 528, 144, 336] {
            try plain.offer(input(.microphone, start: position, values: [Float](repeating: 0.125, count: length)))
            position += length
            try plain.drain(through: Int64(position), expiringMissing: true)
        }
        try plain.finish(through: Int64(position))
        require(chunks.flatMap(\.samples).count == position, "Partial block lost or padded real audio")
        require(chunks.flatMap(\.samples).allSatisfy { abs($0 - 0.125) < 0.00001 }, "No-reference speech was changed")
        var end: Int64 = 0
        for chunk in chunks {
            require(chunk.startFrame == end, "DSP changed the media timeline")
            end += Int64(chunk.samples.count)
        }
        print("PASS: absent playback preserves microphone, irregular chunks and final partial timestamps")

        let count = 20 * rate
        var reference = [Float](repeating: 0, count: count)
        var desired = reference
        var noise: UInt64 = 47
        var smooth = 0.0
        for i in 0..<count {
            let t = Double(i) / Double(rate)
            noise = noise &* 6364136223846793005 &+ 1
            let random = Double(Int32(truncatingIfNeeded: noise >> 32)) / Double(Int32.max)
            smooth = 0.9 * smooth + 0.1 * random
            let envelope = 0.55 + 0.45 * sin(2 * .pi * 1.7 * t)
            reference[i] = Float(envelope * (0.15 * sin(2 * .pi * (173 * t + 9 * sin(t))) + 0.13 * smooth))
            if (8..<15).contains(t) {
                desired[i] = Float(0.08 * sin(2 * .pi * 631 * t) + 0.035 * sin(2 * .pi * 1117 * t))
            }
        }
        var microphone = desired
        for i in 0..<count {
            if i >= 10_560 { microphone[i] += 0.45 * reference[i - 10_560] }
            if i >= 12_048 { microphone[i] += 0.14 * reference[i - 12_048] }
        }

        // Both sources remain available to AEC even when either digital source
        // is excluded from the saved recording.
        for includeVoice in [false, true] {
            for includeMusic in [false, true] {
                var output: [Mixer.Chunk] = []
                let mixer = Mixer(timeline: try timeline(), includeJarvisAudio: includeVoice,
                                  includeMediaAudio: includeMusic) { output.append($0) }
                mixer.enableEchoReduction()
                for start in stride(from: 0, to: count, by: 480) {
                    let range = start..<(start + 480)
                    try mixer.offer(input(.microphone, start: start, values: Array(microphone[range])))
                    try mixer.offer(input(.renderedOutput, start: start, values: reference[range].map { $0 * 0.6 }))
                    try mixer.offer(input(.media, start: start, values: reference[range].map { $0 * 0.4 }))
                    try mixer.drain(through: Int64(start + 480))
                }
                try mixer.finish(through: Int64(count))
                let mixed = output.flatMap(\.samples)
                require(mixed.count == count, "Mixed sample count changed")
                let gain: Float = (includeVoice ? 0.6 : 0) + (includeMusic ? 0.4 : 0)
                let residual = zip(mixed, reference).map { $0 - gain * $1 }
                let quiet = (4 * rate)..<(7 * rate)
                let reduction = 10 * log10((energy(Array(microphone[quiet])) + 1e-9) / (energy(Array(residual[quiet])) + 1e-9))
                let near = (10 * rate)..<(14 * rate)
                let projection = zip(residual[near], desired[near]).reduce(0.0) { $0 + Double($1.0 * $1.1) } / energy(Array(desired[near]))
                require(reduction > 8, "Echo did not improve: \(reduction) dB")
                require(projection > 0.7, "Overlapping user speech was suppressed: \(projection)")
                print("PASS: voice=\(includeVoice), music=\(includeMusic), echo reduction=\(String(format: "%.1f", reduction)) dB, user gain=\(String(format: "%.2f", projection))")
            }
        }

        // A real gap/new route is not compressed into continuous audio.
        var gapped: [Mixer.Chunk] = []
        let gap = Mixer(timeline: try timeline(), includeJarvisAudio: false) { gapped.append($0) }
        gap.enableEchoReduction()
        try gap.offer(input(.microphone, start: 4800, values: [Float](repeating: 0.125, count: 144)))
        try gap.drain(through: 4944, expiringMissing: true)
        try gap.offer(input(.microphone, start: 24000, values: [Float](repeating: 0.125, count: 144), route: 2))
        try gap.finish(through: 24144)
        require(gapped.map(\.startFrame) == [4800, 24000], "Gap or route reset changed timestamps")
        require(gapped.allSatisfy { $0.samples.count == 144 }, "Gap inserted fake source samples")
        print("PASS: gap/route reset preserves offsets without inventing source audio")
        print("PASS: production recording-only echo reducer; synthetic signals, not hardware acceptance")
    }
}
