import Foundation

@main
struct RecordingControlChecks {
    typealias Wire = JarvisRecordingControl
    static let session = UUID(), generation = UUID(), recording = UUID(), operation = UUID()
    static func values() -> [String: Any] {
        ["type": "recording.command", "version": 1, "action": "start",
         "operation_id": operation.uuidString.lowercased(), "recording_id": recording.uuidString.lowercased(),
         "session_id": session.uuidString.lowercased(), "generation_id": generation.uuidString.lowercased(),
         "audio_session_id": "current-audio-session", "include_jarvis_audio": true,
         "max_bytes": 32 * 1024 * 1024, "max_duration_ms": 60_000]
    }
    static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError(message) }
    }
    static func rejected(_ action: () throws -> Void) {
        do { try action(); fatalError("Invalid wire was accepted") }
        catch is Wire.WireError { }
        catch { fatalError("Unexpected error: \(error)") }
    }
    static func decoded(_ text: String) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(text.utf8)) as! [String: Any]
    }
    static func main() throws {
        let start = try Wire.Command(values: values())
        require(start.action == .start && start.recordingID == recording && start.includeJarvisAudio, "Start decode")
        var stop = values(); stop["action"] = "stop"; stop["operation_id"] = UUID().uuidString
        let stopping = try Wire.Command(values: stop)
        require(stopping.action == .stop && stopping.recordingID == start.recordingID,
                "Stop must keep original recording identity")
        print("PASS 1: typed start/stop retain recording and independent operation identities")

        try start.requireCurrent(sessionID: session, audioSessionID: "current-audio-session", generationID: generation)
        rejected { try start.requireCurrent(sessionID: UUID(), audioSessionID: "current-audio-session", generationID: generation) }
        rejected { try start.requireCurrent(sessionID: session, audioSessionID: "retired", generationID: generation) }
        rejected { try start.requireCurrent(sessionID: session, audioSessionID: "current-audio-session", generationID: UUID()) }
        print("PASS 2: account-session/audio-session/device generation fences")

        for (field, value) in [("version", true), ("version", "1"), ("version", 2),
                               ("include_jarvis_audio", 1), ("include_jarvis_audio", "true"),
                               ("action", "upload"), ("recording_id", "../recording"),
                               ("audio_session_id", ""), ("max_bytes", true)] as [(String, Any)] {
            var invalid = values(); invalid[field] = value
            rejected { _ = try Wire.Command(values: invalid) }
        }
        var extra = values(); extra["path"] = "/tmp/untrusted.mp4"
        rejected { _ = try Wire.Command(values: extra) }
        for key in values().keys {
            var missing = values(); missing.removeValue(forKey: key)
            rejected { _ = try Wire.Command(values: missing) }
        }
        print("PASS 3: strict schema rejects missing/extra fields, type coercion and arbitrary actions")

        for (field, value) in [("max_bytes", 0), ("max_bytes", Wire.maximumBytes + 1),
                               ("max_duration_ms", 0), ("max_duration_ms", Wire.maximumDurationMilliseconds + 1)] {
            var invalid = values(); invalid[field] = value
            rejected { _ = try Wire.Command(values: invalid) }
        }
        print("PASS 4: device cannot expand admitted size/duration bounds")
        var thirtyMinuteValues = values()
        thirtyMinuteValues["max_duration_ms"] = 1_800_000
        thirtyMinuteValues["max_bytes"] = 1024 * 1024 * 1024
        let thirtyMinute = try Wire.Command(values: thirtyMinuteValues)
        require(thirtyMinute.maxDurationMilliseconds == 1_800_000 && thirtyMinute.maxBytes == 1024 * 1024 * 1024,
                "Current finite recording limits are accepted without a new protocol mode")

        let capabilities = try decoded(Wire.encode(.capabilities(generationID: generation, available: false,
            includeJarvisAudio: false, maxBytes: Wire.maximumBytes, maxDurationMilliseconds: 60_000)))
        require(Set(capabilities.keys) == ["type", "version", "generation_id", "available",
            "include_jarvis_audio", "max_bytes", "max_duration_ms"], "Unexpected capability fields")
        require(capabilities["available"] as? Bool == false, "Unavailable became available")
        for state in [Wire.ReportState.started, .stopped, .failed] {
            let report = try decoded(Wire.encode(.report(generationID: generation, recordingID: recording,
                sequence: 1, state: state)))
            require(Set(report.keys) == ["type", "version", "generation_id", "recording_id", "sequence", "state"], "Unexpected report fields")
            require(report["state"] as? String == state.rawValue && report["client_event_id"] == nil,
                    "Device report claims additional authority")
        }
        for sequence in [0, -1, 1_000_000_001] {
            rejected { _ = try Wire.encode(.report(generationID: generation, recordingID: recording, sequence: sequence, state: .started)) }
        }
        rejected { _ = try Wire.encode(.capabilities(generationID: generation, available: true,
            includeJarvisAudio: true, maxBytes: 0, maxDurationMilliseconds: 60_000)) }
        print("PASS 5: exact bounded capabilities/reports; no fabricated save or legacy extras")
        print("PASS: 5 recording-control groups; codec only, no live capture or delivery claim")
    }
}
