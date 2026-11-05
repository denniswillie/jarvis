import Foundation

@main struct EchoDiagnosticChecks {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { fatalError("Private output directory required") }
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let directory = root.appendingPathComponent("bounded")
        let probe = JarvisRecordingEchoDiagnostic(directory: directory, maximumFrames: 6)!
        let mic: [Float] = [0.1, 0.2, 0.3], ref: [Float] = [0.4, 0.5, 0.6]
        let clean: [Float] = [0.01, 0.02, 0.03], direct: [Float] = [0.7, 0.8, 0.9]
        probe.append(startFrame: 100, microphone: mic, reference: ref, cleaned: clean, direct: direct, count: 3)
        probe.append(startFrame: 104, microphone: mic, reference: ref, cleaned: clean, direct: direct, count: 3)
        probe.append(startFrame: 110, microphone: mic, reference: ref, cleaned: clean, direct: direct, count: 3)
        probe.finish(); probe.finish()
        let bytes = try Data(contentsOf: directory.appendingPathComponent("capture.wav"))
        precondition(bytes.count == 44 + 6 * 16)
        precondition(String(decoding: bytes.prefix(4), as: UTF8.self) == "RIFF")
        func word(_ offset: Int) -> UInt32 {
            (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << ($1 * 8) }
        }
        precondition(word(40) == 6 * 16)
        for channel in 0..<4 {
            precondition(Float(bitPattern: word(44 + channel * 4)) == [mic[0], ref[0], clean[0], direct[0]][channel])
            precondition(word(44 + 3 * 16 + channel * 4) == 0) // Preserve one missing frame.
        }
        let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("capture.json"))) as! [String: Any]
        precondition(metadata["frames"] as? Int == 6 && metadata["first_recording_frame"] as? Int == 100)
        precondition((metadata["gaps"] as? [[String: Int]]) == [["start_frame": 103, "frames": 1]])
        precondition(JarvisRecordingEchoDiagnostic(directory: directory) == nil)
        precondition(JarvisRecordingEchoDiagnostic(directory: root, maximumFrames: 960001) == nil)
        precondition(JarvisRecordingEchoDiagnostic.configured() == nil, "Must be opt-in")
        print("PASS: channel identity, offsets, gaps, finite cap, idempotent finish, no overwrite and default-off capture")
    }
}
