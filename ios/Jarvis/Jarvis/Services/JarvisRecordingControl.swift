import CoreFoundation
import Foundation

/// Versioned recording-only wire messages. No arbitrary operation, URL, path,
/// credential or model-authored file policy is accepted at the device boundary.
enum JarvisRecordingControl {
    static let maximumBytes = 1024 * 1024 * 1024
    static let maximumDurationMilliseconds = 1_800_000
    enum WireError: Error, Equatable { case invalidCommand, invalidBounds, identityMismatch, invalidReport, encodingFailed }
    enum Action: String, Sendable { case start, stop }
    enum ReportState: String, Sendable { case started, stopped, failed }

    struct Command: Sendable, Equatable {
        let action: Action
        let operationID: UUID
        let recordingID: UUID
        let sessionID: UUID
        let generationID: UUID
        let audioSessionID: String
        let includeJarvisAudio: Bool
        let maxBytes: Int
        let maxDurationMilliseconds: Int
        let version: Int
        let recordingMode: String
        let includeMediaAudio: Bool

        init(values: [String: Any]) throws {
            var keys: Set<String> = ["type", "version", "action", "operation_id", "recording_id",
                "session_id", "generation_id", "audio_session_id", "include_jarvis_audio", "max_bytes", "max_duration_ms"]
            let version = integer(values["version"]) ?? 0
            if version == 2 { keys.formUnion(["recording_mode", "include_media_audio"]) }
            guard Set(values.keys) == keys, values["type"] as? String == "recording.command",
                  version == 1 || version == 2,
                  let actionString = values["action"] as? String, let action = Action(rawValue: actionString),
                  let operationID = uuid(values["operation_id"]), let recordingID = uuid(values["recording_id"]),
                  let sessionID = uuid(values["session_id"]), let generationID = uuid(values["generation_id"]),
                  let audioSessionID = values["audio_session_id"] as? String,
                  !audioSessionID.isEmpty, audioSessionID.utf8.count <= 128, !audioSessionID.contains("\0"),
                  let include = boolean(values["include_jarvis_audio"]),
                  let maxBytes = integer(values["max_bytes"]), let maxDuration = integer(values["max_duration_ms"]) else {
                throw WireError.invalidCommand
            }
            try bounds(bytes: maxBytes, milliseconds: maxDuration)
            let mode = version == 1 ? "ordinary" : values["recording_mode"] as? String
            let media = version == 1 ? false : boolean(values["include_media_audio"])
            guard let mode, ["ordinary", "voiceover"].contains(mode), let media,
                  mode != "voiceover" || (!include && !media) else { throw WireError.invalidCommand }
            self.version = version; self.recordingMode = mode; self.includeMediaAudio = media
            self.action = action; self.operationID = operationID; self.recordingID = recordingID
            self.sessionID = sessionID; self.generationID = generationID; self.audioSessionID = audioSessionID
            self.includeJarvisAudio = include; self.maxBytes = maxBytes; self.maxDurationMilliseconds = maxDuration
        }

        /// Recheck at the actual start/stop boundary, not only when queued.
        func requireCurrent(sessionID: UUID, audioSessionID: String, generationID: UUID) throws {
            guard self.sessionID == sessionID, self.audioSessionID == audioSessionID,
                  self.generationID == generationID else { throw WireError.identityMismatch }
        }
    }

    enum Outbound: Sendable {
        case capabilities(generationID: UUID, available: Bool, includeJarvisAudio: Bool,
                          maxBytes: Int, maxDurationMilliseconds: Int, protocolVersion: Int = 1)
        case report(generationID: UUID, recordingID: UUID, sequence: Int, state: ReportState)
    }

    static func encode(_ event: Outbound) throws -> String {
        let values: [String: Any]
        switch event {
        case let .capabilities(generationID, available, include, bytes, duration, version):
            try bounds(bytes: bytes, milliseconds: duration)
            guard version == 1 || version == 2 else { throw WireError.invalidCommand }
            var capabilities: [String: Any] = ["type": "recording.capabilities", "version": version,
                "generation_id": generationID.uuidString.lowercased(), "available": available,
                "include_jarvis_audio": include, "max_bytes": bytes, "max_duration_ms": duration]
            if version == 2 { capabilities["include_media_audio"] = true; capabilities["voiceover"] = true }
            values = capabilities
        case let .report(generationID, recordingID, sequence, state):
            guard (1...1_000_000_000).contains(sequence) else { throw WireError.invalidReport }
            values = ["type": "recording.report", "version": 1,
                "generation_id": generationID.uuidString.lowercased(),
                "recording_id": recordingID.uuidString.lowercased(), "sequence": sequence, "state": state.rawValue]
        }
        // Deliberately no legacy client_event_id or free-form status text. A
        // device report can never claim a saved/uploaded/heard outcome.
        let data = try JSONSerialization.data(withJSONObject: values, options: [.sortedKeys])
        guard let text = String(data: data, encoding: .utf8) else { throw WireError.encodingFailed }
        return text
    }

    private static func bounds(bytes: Int, milliseconds: Int) throws {
        guard (1...maximumBytes).contains(bytes), (1...maximumDurationMilliseconds).contains(milliseconds) else {
            throw WireError.invalidBounds
        }
    }
    private static func uuid(_ value: Any?) -> UUID? {
        guard let text = value as? String, text.utf8.count == 36, let id = UUID(uuidString: text),
              id.uuidString.lowercased() == text.lowercased() else { return nil }
        return id
    }
    private static func boolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }
    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              !["f", "d"].contains(String(cString: number.objCType)),
              let exact = Int(number.stringValue) else { return nil }
        return exact
    }
}
