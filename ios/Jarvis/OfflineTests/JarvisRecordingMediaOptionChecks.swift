import Foundation

@main struct RecordingMediaOptionChecks {
    static func main() throws {
        let generation = UUID()
        let base: [String: Any] = ["type": "recording.command", "version": 1, "action": "start",
            "operation_id": UUID().uuidString, "recording_id": UUID().uuidString,
            "session_id": UUID().uuidString, "generation_id": generation.uuidString,
            "audio_session_id": "audio-test", "include_jarvis_audio": false,
            "max_bytes": 100000, "max_duration_ms": 1800000]
        let legacy = try JarvisRecordingControl.Command(values: base)
        precondition(legacy.version == 1 && !legacy.includeMediaAudio && legacy.recordingMode == "ordinary")
        var ordinary = base
        ordinary["version"] = 2; ordinary["recording_mode"] = "ordinary"; ordinary["include_media_audio"] = true
        let command = try JarvisRecordingControl.Command(values: ordinary)
        precondition(command.includeMediaAudio && !command.includeJarvisAudio)
        var voiceover = ordinary
        voiceover["recording_mode"] = "voiceover"; voiceover["include_media_audio"] = false
        let voiceoverCommand = try JarvisRecordingControl.Command(values: voiceover)
        precondition(voiceoverCommand.recordingMode == "voiceover")
        var bad = voiceover; bad["include_media_audio"] = true
        do { _ = try JarvisRecordingControl.Command(values: bad); fatalError("Invalid voiceover accepted") }
        catch JarvisRecordingControl.WireError.invalidCommand { }
        bad = ordinary; bad["version"] = 1
        do { _ = try JarvisRecordingControl.Command(values: bad); fatalError("Unknown v1 fields accepted") }
        catch JarvisRecordingControl.WireError.invalidCommand { }
        for version in [1, 2] {
            let text = try JarvisRecordingControl.encode(.capabilities(generationID: generation, available: true,
                includeJarvisAudio: true, maxBytes: 100000, maxDurationMilliseconds: 1800000, protocolVersion: version))
            let value = try JSONSerialization.jsonObject(with: Data(text.utf8)) as! [String: Any]
            precondition(value["version"] as? Int == version)
            precondition((value["include_media_audio"] as? Bool == true) == (version == 2))
        }
        print("PASS: legacy recording wire, ordinary music inclusion, explicit voiceover and negotiated v2 capabilities")
    }
}
