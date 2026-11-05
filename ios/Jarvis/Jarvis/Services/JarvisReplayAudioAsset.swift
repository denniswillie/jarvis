#if DEBUG
import CryptoKit
import Foundation

/// Local runner manifest only. Never serialize this into provider or frame payloads.
struct JarvisReplayAudioAsset: Decodable {
    enum Format: String, Decodable { case wavPCM16Mono48000 = "wav_pcm16_mono_48000", rawPCM16LEMono48000 = "raw_pcm16le_mono_48000" }
    let fileURL: URL
    let format: Format
    let sha256: String
    let sampleCount: Int
    let thresholdPCM16: Int
    // Zero-based sample index, not an exclusive end position. Nil for all-silent input.
    let lastAboveThresholdSampleIndex: Int?
    enum CodingKeys: String, CodingKey {
        case fileURL = "file_url", format, sha256, sampleCount = "sample_count"
        case thresholdPCM16 = "threshold_pcm16", lastAboveThresholdSampleIndex = "last_above_threshold_sample_index"
    }

    func loadPCM16() throws -> Data {
        guard fileURL.isFileURL, sampleCount > 0, sampleCount <= 60 * 48_000,
              (0...32767).contains(thresholdPCM16) else { throw JarvisReplayError.invalidPlan }
        let size = try fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max
        guard size <= 60 * 96_000 + 65_536 else { throw JarvisReplayError.invalidPlan }
        let file = try Data(contentsOf: fileURL)
        return try validatedPCM16(file)
    }

    func validatedPCM16(_ file: Data) throws -> Data {
        guard sampleCount > 0, sampleCount <= 60 * 48_000,
              (0...32767).contains(thresholdPCM16), file.count <= 60 * 96_000 + 65_536,
              sha256.count == 64,
              SHA256.hash(data: file).map({ String(format: "%02x", $0) }).joined() == sha256.lowercased() else {
            throw JarvisReplayError.invalidPlan
        }
        let pcm = try format == .wavPCM16Mono48000 ? Self.decodeWAV(file) : file
        guard pcm.count == sampleCount * 2 else { throw JarvisReplayError.invalidPlan }
        var last: Int?
        for index in 0..<sampleCount {
            let bits = UInt16(pcm[index * 2]) | UInt16(pcm[index * 2 + 1]) << 8
            if abs(Int(Int16(bitPattern: bits))) > thresholdPCM16 { last = index }
        }
        guard last == lastAboveThresholdSampleIndex else { throw JarvisReplayError.invalidPlan }
        return pcm
    }

    private static func decodeWAV(_ data: Data) throws -> Data {
        func word(_ at: Int, _ count: Int) -> Int {
            (0..<count).reduce(0) { $0 | Int(data[at + $1]) << ($1 * 8) }
        }
        guard data.count >= 44, String(data: data.prefix(4), encoding: .ascii) == "RIFF",
              String(data: data[8..<12], encoding: .ascii) == "WAVE",
              word(4, 4) + 8 == data.count else { throw JarvisReplayError.invalidPlan }
        var position = 12
        var validFormat = false
        var pcm: Data?
        while position + 8 <= data.count {
            let kind = String(data: data[position..<(position + 4)], encoding: .ascii)
            let length = word(position + 4, 4)
            position += 8
            guard length <= data.count - position else { throw JarvisReplayError.invalidPlan }
            if kind == "fmt " {
                guard !validFormat, length >= 16, word(position, 2) == 1,
                      word(position + 2, 2) == 1, word(position + 4, 4) == 48_000,
                      word(position + 8, 4) == 96_000, word(position + 12, 2) == 2,
                      word(position + 14, 2) == 16 else { throw JarvisReplayError.invalidPlan }
                validFormat = true
            } else if kind == "data" {
                guard pcm == nil, length.isMultiple(of: 2) else { throw JarvisReplayError.invalidPlan }
                pcm = Data(data[position..<(position + length)])
            }
            position += length + length % 2
        }
        guard position == data.count, validFormat, let pcm, !pcm.isEmpty else { throw JarvisReplayError.invalidPlan }
        return pcm
    }
}
#endif
