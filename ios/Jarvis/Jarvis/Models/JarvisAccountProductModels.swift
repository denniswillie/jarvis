import Foundation

protocol JarvisPreferencesAccess {
    func fetchBehaviorProfile() async throws -> JarvisBehaviorProfile
    func updateBehaviorProfile(_ update: JarvisBehaviorProfileUpdate) async throws -> JarvisBehaviorProfile
}

struct JarvisBehaviorProfile: Codable, Equatable, Sendable {
    let ownerID: String
    let behaviorInstructions: String
    let revision: Int64
    let createdAt: String?
    let updatedAt: String?

    enum CodingKeys: String, CodingKey {
        case ownerID = "owner_id", behaviorInstructions = "behavior_instructions", revision
        case createdAt = "created_at", updatedAt = "updated_at"
    }

    var isValid: Bool {
        !ownerID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        revision >= 0 && behaviorInstructions.unicodeScalars.count <= 12_000
    }
}

struct JarvisBehaviorProfileUpdate: Encodable, Equatable, Sendable {
    let behaviorInstructions: String
    let expectedRevision: Int64
    enum CodingKeys: String, CodingKey {
        case behaviorInstructions = "behavior_instructions", expectedRevision = "expected_revision"
    }
}

/// A backend commit is not proof that a voice provider applied its instructions.
struct JarvisBehaviorProfileCommitted: Decodable, Equatable, Sendable {
    let type: String
    let eventID: String
    let ownerID: String
    let revision: Int64
    let behaviorInstructions: String
    let committedAt: String
    enum CodingKeys: String, CodingKey {
        case type, eventID = "event_id", ownerID = "owner_id", revision
        case behaviorInstructions = "behavior_instructions", committedAt = "committed_at"
    }
}

extension JarvisRealtimeServerEvent {
    var behaviorProfileCommitted: JarvisBehaviorProfileCommitted? {
        guard type == "behavior_profile.committed" else { return nil }
        return decodeAccountEvent(JarvisBehaviorProfileCommitted.self)
    }
    private func decodeAccountEvent<T: Decodable>(_ result: T.Type) -> T? {
        var payload = values
        payload["type"] = type
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload) else { return nil }
        return try? JSONDecoder().decode(result, from: data)
    }
}
