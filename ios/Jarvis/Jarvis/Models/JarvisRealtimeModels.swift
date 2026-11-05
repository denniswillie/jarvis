import Foundation

enum JarvisAudioConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case failed(String)

    var label: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting"
        case .connected: return "Connected"
        case .failed(let message): return "Failed: \(message)"
        }
    }

    var isConnected: Bool {
        if case .connected = self { return true }
        return false
    }
}

enum JarvisAudioTurnState: Equatable {
    case idle
    case listening
    case processingInput
    case thinking
    case speaking
    case failed(String)

    var label: String {
        switch self {
        case .idle: return "Idle"
        case .listening: return "Listening"
        case .processingInput: return "Processing"
        case .thinking: return "Thinking"
        case .speaking: return "Speaking"
        case .failed(let message): return "Failed: \(message)"
        }
    }
}

enum JarvisAudioCaptureMode: String, CaseIterable, Identifiable {
    case autoDetect
    case holdToTalk

    var id: String { rawValue }

    var label: String {
        switch self {
        case .autoDetect: return "Auto detect speech"
        case .holdToTalk: return "Hold to talk"
        }
    }
}

struct JarvisSpeechSegmentState {
    private(set) var id: String?
    private(set) var startedAt: Date?

    mutating func begin(at date: Date) {
        id = UUID().uuidString
        startedAt = date
    }

    mutating func finishCommit(segmentID: String) -> Bool {
        // A network await may have allowed another segment or session to start.
        guard id == segmentID else { return false }
        reset()
        return true
    }

    mutating func reset() {
        id = nil
        startedAt = nil
    }
}

struct JarvisRealtimeServerEvent {
    let type: String
    let values: [String: Any]

    var text: String? { values["text"] as? String }
    var role: String? { values["role"] as? String }
    var audioBase64: String? { values["audio_base64"] as? String }
    var responseId: String? { values["response_id"] as? String }
    var itemId: String? { values["item_id"] as? String }
    var code: String? { values["code"] as? String }
    var message: String? { values["message"] as? String }
    var recoverable: Bool? { Self.boolValue(values["recoverable"]) }
    var sessionId: String? { values["session_id"] as? String }
    var audioSessionId: String? { values["audio_session_id"] as? String }
    var visualSessionId: String? { values["visual_session_id"] as? String }
    var frameId: String? { values["frame_id"] as? String }
    var frameSequence: Int64? { Self.int64Value(values["frame_sequence"]) }
    var clientKind: String? { values["client_kind"] as? String }
    var reason: String? { values["reason"] as? String }
    var reminderId: String? { values["reminder_id"] as? String }
    var reminderEventId: String? { values["reminder_event_id"] as? String }
    var memoryChangeId: String? { values["change_id"] as? String }
    var memoryChangedAt: String? { values["changed_at"] as? String }
    var affectedMemoryIds: [String] { Self.stringArrayValue(values["affected_memory_ids"]) }
    var affectedMemoryTypes: [String] { Self.stringArrayValue(values["affected_memory_types"]) }
    var memoryChangeType: String? { values["change_type"] as? String }
    var workingMemoryRevision: Int64? { Self.int64Value(values["revision"]) }
    var workingMemoryOwnerRevision: Int64? { Self.int64Value(values["owner_revision"]) }
    var sessionWorkingMemoryAvailable: Bool? { Self.boolValue(values["session_available"]) }
    var ownerWorkingMemoryAvailable: Bool? { Self.boolValue(values["owner_available"]) }
    var workingMemoryObservedAt: String? { values["observed_at"] as? String }
    var workingMemoryExpiresAt: String? { values["expires_at"] as? String }
    var workingMemorySceneSummary: String? { values["scene_summary"] as? String }
    var workingMemoryActivitySummary: String? { values["activity_summary"] as? String }

    var memoryRefreshHint: JarvisMemoryRefreshHint? {
        guard type == "memory.updated" else { return nil }
        return JarvisMemoryRefreshHint(
            changeId: memoryChangeId,
            changedAt: memoryChangedAt,
            affectedMemoryIds: affectedMemoryIds,
            affectedMemoryTypes: affectedMemoryTypes,
            reason: reason,
            changeType: memoryChangeType
        )
    }

    var workingMemoryRefreshHint: JarvisWorkingMemoryRefreshHint? {
        guard type == "working_memory.updated" else { return nil }
        return JarvisWorkingMemoryRefreshHint(
            sessionId: sessionId,
            revision: workingMemoryRevision,
            ownerRevision: workingMemoryOwnerRevision,
            sessionAvailable: sessionWorkingMemoryAvailable,
            ownerAvailable: ownerWorkingMemoryAvailable,
            observedAt: workingMemoryObservedAt,
            expiresAt: workingMemoryExpiresAt,
            sceneSummary: workingMemorySceneSummary,
            activitySummary: workingMemoryActivitySummary
        )
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        if let value = value as? Bool { return value }
        if let value = value as? NSNumber { return value.boolValue }
        return nil
    }

    private static func int64Value(_ value: Any?) -> Int64? {
        if let value = value as? Int64 { return value }
        if let value = value as? Int { return Int64(value) }
        if let value = value as? NSNumber { return value.int64Value }
        if let value = value as? String { return Int64(value) }
        return nil
    }

    private static func stringArrayValue(_ value: Any?) -> [String] {
        if let values = value as? [String] { return values }
        if let values = value as? [Any] { return values.compactMap { $0 as? String } }
        return []
    }
}

struct JarvisMemoryRefreshHint: Equatable {
    let changeId: String?
    let changedAt: String?
    let affectedMemoryIds: [String]
    let affectedMemoryTypes: [String]
    let reason: String?
    let changeType: String?
}

struct JarvisWorkingMemoryRefreshHint: Equatable {
    let sessionId: String?
    let revision: Int64?
    let ownerRevision: Int64?
    let sessionAvailable: Bool?
    let ownerAvailable: Bool?
    let observedAt: String?
    let expiresAt: String?
    let sceneSummary: String?
    let activitySummary: String?
}

struct CapturedAudioChunk {
    let pcm16Data: Data
    let sampleRate: Double
    let frameCount: Int
    let capturedAt: Date

    var durationSeconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(frameCount) / sampleRate
    }
}

struct PlaybackTruncationSnapshot {
    let itemId: String
    let responseId: String?
    let audioEndMs: Int
}

struct JarvisTranscriptEntry: Identifiable, Equatable {
    enum Role: String {
        case user
        case assistant
        case system
        case error
    }

    let id = UUID()
    let role: Role
    let text: String
    let date: Date
}

struct AudioRouteSnapshot: Equatable {
    let category: String
    let mode: String
    let options: String
    let inputName: String
    let inputType: String
    let inputUID: String
    let outputName: String
    let outputType: String
    let outputUID: String
    let appearsToUseGlasses: Bool

    var summary: String {
        if appearsToUseGlasses {
            return "Ray-Ban route active"
        }
        if inputName == "-" && outputName == "-" {
            return "Audio route unavailable"
        }
        return "Input: \(inputName), Output: \(outputName)"
    }
}

enum JarvisAudioSessionError: LocalizedError {
    case missingConfiguration
    case invalidRealtimeURL(String)
    case realtimeNotConnected
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .missingConfiguration:
            return "Jarvis backend configuration is missing."
        case .invalidRealtimeURL(let value):
            return "Jarvis returned an invalid realtime URL: \(value)."
        case .realtimeNotConnected:
            return "Realtime audio is not connected."
        case .serverError(let message):
            return message
        }
    }
}

extension Date {
    func jarvisISO8601String() -> String {
        ISO8601DateFormatter.jarvisFormatter.string(from: self)
    }
}

extension ISO8601DateFormatter {
    static let jarvisFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
