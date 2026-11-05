import Foundation

@main struct RecordingIODiagnosticChecks {
    static func main() throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let directory = root.appendingPathComponent("native-io")
        let probe = JarvisRecordingIODiagnostic(directory: directory, sampleRate: 16000,
            inputChannels: 1, outputChannels: 2, frameLimit: 4)!
        let mic = Data([1, 2, 3, 4, 5, 6])
        let output = Data((0..<12).map(UInt8.init))
        let timing: [String: Any] = ["host_time": UInt64(9007199254740993), "external_microphone": true,
            "external_first_submit_host_time": 123, "external_last_submit_host_time": 456]
        probe.append(track: 0, pcm16: mic, frameCount: 3, metadata: timing)
        probe.append(track: 1, pcm16: output, frameCount: 3, metadata: ["host_time": 700])
        probe.append(track: 0, pcm16: mic, frameCount: 3, metadata: timing)
        probe.append(track: 1, pcm16: output, frameCount: 3, metadata: ["host_time": 701])
        probe.append(track: 0, pcm16: mic, frameCount: 3, metadata: timing)
        probe.finish(reason: "extra_finish")
        let micResult = try Data(contentsOf: directory.appendingPathComponent("microphone.pcm"))
        let playback = try Data(contentsOf: directory.appendingPathComponent("playback.pcm"))
        precondition(micResult == mic + mic.prefix(2))
        precondition(playback == output + output.prefix(4))
        let rows = try String(contentsOf: directory.appendingPathComponent("packets.jsonl"), encoding: .utf8)
            .split(separator: "\n").map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        precondition(rows.count == 6 && rows.last?["reason"] as? String == "frame_cap")
        precondition((rows[1]["host_time"] as? NSNumber)?.uint64Value == 9007199254740993)
        precondition(rows[1]["external_first_submit_host_time"] as? Int == 123)
        precondition(rows[3]["byte_offset"] as? Int == 6 && rows[3]["frames_written"] as? Int == 1)
        precondition(JarvisRecordingIODiagnostic(directory: directory, sampleRate: 16000,
            inputChannels: 1, outputChannels: 2) == nil, "Must never overwrite an earlier probe")
        let invalidDirectory = root.appendingPathComponent("invalid-packet")
        let invalid = JarvisRecordingIODiagnostic(directory: invalidDirectory, sampleRate: 48000,
            inputChannels: 1, outputChannels: 1)!
        invalid.append(track: 3, pcm16: mic, frameCount: 3, metadata: [:])
        let invalidRows = try String(contentsOf: invalidDirectory.appendingPathComponent("packets.jsonl"), encoding: .utf8)
        precondition(invalidRows.contains("invalid_packet"))
        let attrs = try FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent("microphone.pcm").path)
        precondition((attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        precondition(JarvisRecordingIODiagnostic.configured(sampleRate: 48000, inputChannels: 1,
            outputChannels: 1, context: [:]) == nil)
        print("PASS: exact native PCM, channel layout, timing precision, frame cap, one finish, no overwrite, invalid packets, private files, default off")
    }
}
