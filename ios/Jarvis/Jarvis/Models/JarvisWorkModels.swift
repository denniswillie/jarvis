import Foundation

protocol JarvisWorkAccess {
    func getWorkTasks(offset: Int) async throws -> [JarvisWorkTask]
    func getWorkTask(id: UUID) async throws -> JarvisWorkTask
    func getWorkInbox(offset: Int, includeDismissed: Bool, taskID: UUID?) async throws -> [JarvisInboxItem]
    func getWorkInboxItem(id: UUID) async throws -> JarvisInboxDetail
    func changeWorkTask(id: UUID, operation: JarvisTaskOperation, body: JarvisTaskChange) async throws -> JarvisWorkTask
    func changeWorkInbox(id: UUID, operation: JarvisInboxOperation, body: JarvisInboxChange) async throws -> JarvisInboxChangeReceipt
}

struct JarvisWorkTask: Decodable, Identifiable, Equatable {
    let id: UUID
    let title: String?
    let requestedOutcome: String?
    let status: String
    let revision: Int
    let reason: String?
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id, title, status, revision, reason
        case requestedOutcome = "requested_outcome", updatedAt = "updated_at"
    }

    var displayTitle: String { title ?? "Background task" }
    var statusLabel: String {
        switch status {
        case "queued": return "Queued"
        case "running": return "Working"
        case "waiting_for_input": return "Needs your input"
        case "paused": return "Paused"
        case "succeeded": return "Work finished"
        case "failed": return "Failed"
        case "cancel_requested": return "Cancellation requested"
        case "cancelled": return "Cancelled"
        case "outcome_unknown": return "Outcome uncertain"
        default: return "Unrecognized status"
        }
    }
    var canUpdate: Bool {
        ["queued", "running", "waiting_for_input", "paused", "succeeded", "failed", "cancelled"].contains(status)
    }
    var canCancel: Bool {
        ["queued", "running", "waiting_for_input", "paused", "failed", "outcome_unknown"].contains(status)
    }
    var canResume: Bool { ["paused", "waiting_for_input", "failed", "cancelled"].contains(status) }
}

struct JarvisInboxItem: Decodable, Identifiable, Equatable {
    let id: UUID
    let sourceNamespace: String
    let taskID: UUID?
    let taskRevision: Int?
    let state: String
    let generation: Int
    let attemptCount: Int
    let sourceAvailable: Bool
    let sourceUnavailableReason: String?
    let phoneNoticeState: String?
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id, state, generation
        case sourceNamespace = "source_namespace", taskID = "task_id", taskRevision = "task_revision"
        case attemptCount = "attempt_count", sourceAvailable = "source_available"
        case sourceUnavailableReason = "source_unavailable_reason", updatedAt = "updated_at"
        case phoneNoticeState = "phone_notice_state"
    }
    var sourceLabel: String {
        switch sourceNamespace {
        case "task_result": return "Expert answer"
        case "reminder_event": return "Saved reminder"
        case "visual_alert", "prospective_event": return "Visual notification"
        default: return "Notification"
        }
    }
    var stateLabel: String {
        if sourceAvailable, ["pending", "unknown", "failed"].contains(state), let phoneNoticeState {
            switch phoneNoticeState {
            case "claimed": return "Preparing phone notification"
            case "submitted": return "Submitting phone notification"
            case "accepted": return "Notification submitted"
            case "unknown": return "Notification delivery uncertain"
            default: break
            }
        }
        switch state {
        case "pending": return "Waiting to speak"
        case "claimed": return "Preparing speech"
        case "awaiting_receipt": return "Awaiting playback confirmation"
        case "played": return "Playback reported complete"
        case "unknown": return "Playback uncertain"
        case "failed": return "Delivery failed"
        case "dismissed": return "Dismissed"
        case "revoked": return "Source unavailable"
        case "expired": return "Expired"
        default: return "Unrecognized delivery state"
        }
    }
    var canDismiss: Bool {
        ["pending", "claimed", "awaiting_receipt", "played", "unknown", "failed"].contains(state)
    }
}

struct JarvisExpertAnswer: Decodable, Equatable {
    let answer: String
    let status: String?
    let uncertainties: [String]
}

struct JarvisWorkNotification: Decodable, Equatable {
    let title: String?
    let message: String
    let authoredBy: String?
    let sourceCapturedAt: String?
    enum CodingKeys: String, CodingKey {
        case title, message
        case authoredBy = "authored_by", sourceCapturedAt = "source_captured_at"
    }
}

struct JarvisTaskArtifact: Decodable, Identifiable, Equatable {
    let fileID: UUID
    let filename: String
    var id: UUID { fileID }
    enum CodingKeys: String, CodingKey { case fileID = "file_id", filename }
}

struct JarvisInboxDetail: Decodable, Equatable {
    let item: JarvisInboxItem
    let expertResult: JarvisExpertAnswer?
    let notification: JarvisWorkNotification?
    let speechPrepared: Bool
    let artifacts: [JarvisTaskArtifact]

    enum CodingKeys: String, CodingKey {
        case expertResult = "expert_result", notification, speechPrepared = "speech_prepared", artifacts
    }
    init(from decoder: Decoder) throws {
        item = try JarvisInboxItem(from: decoder)
        let values = try decoder.container(keyedBy: CodingKeys.self)
        // Even an inconsistent response must not display a revoked body.
        if item.sourceAvailable {
            expertResult = try values.decodeIfPresent(JarvisExpertAnswer.self, forKey: .expertResult)
            notification = try values.decodeIfPresent(JarvisWorkNotification.self, forKey: .notification)
            artifacts = expertResult == nil ? [] : try values.decodeIfPresent([JarvisTaskArtifact].self, forKey: .artifacts) ?? []
        } else {
            expertResult = nil
            notification = nil
            artifacts = []
        }
        speechPrepared = try values.decode(Bool.self, forKey: .speechPrepared)
    }
    var canReplay: Bool {
        item.sourceAvailable && speechPrepared && !["claimed", "submitted"].contains(item.phoneNoticeState ?? "") &&
            (["played", "unknown", "failed", "dismissed"].contains(item.state) ||
             (item.state == "pending" && ["accepted", "unknown"].contains(item.phoneNoticeState ?? "")))
    }
}

enum JarvisTaskOperation: String { case update = "updates", cancel, resume }
enum JarvisInboxOperation: String { case replay, dismiss }

struct JarvisTaskChange: Encodable, Equatable {
    let requestID: UUID
    let expectedRevision: Int
    let message: String
    enum CodingKeys: String, CodingKey {
        case requestID = "request_id", expectedRevision = "expected_revision", message
    }
}

struct JarvisInboxChange: Encodable, Equatable {
    let requestID: UUID
    let expectedGeneration: Int
    enum CodingKeys: String, CodingKey {
        case requestID = "request_id", expectedGeneration = "expected_generation"
    }
}

struct JarvisInboxChangeReceipt: Decodable {
    let id: UUID
    let generation: Int
}

enum JarvisPendingWorkAction: Equatable {
    case task(UUID, JarvisTaskOperation, JarvisTaskChange)
    case inbox(UUID, JarvisInboxOperation, JarvisInboxChange)
}
