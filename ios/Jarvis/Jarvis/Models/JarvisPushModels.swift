import Foundation

enum JarvisPushPermission: String, Codable {
    case notDetermined = "not_determined", denied, authorized, provisional, ephemeral
    var allowsNotifications: Bool { [.authorized, .provisional, .ephemeral].contains(self) }
}

struct JarvisPushStatus: Decodable {
    let state: String
    let reason: String?
    let environment: String?
    let topic: String?
    let providerAccess: String
    enum CodingKeys: String, CodingKey {
        case state, reason, environment, topic, providerAccess = "provider_access"
    }
}

struct JarvisPushDevice: Decodable, Equatable {
    let id: UUID
    let revision: Int
    let environment: String
    let topic: String
    let permission: JarvisPushPermission
    let enabled: Bool
    let quietUntil: String?
    let invalidReason: String?
    enum CodingKeys: String, CodingKey {
        case id, revision, environment, topic, permission, enabled
        case quietUntil = "quiet_until", invalidReason = "invalid_reason"
    }
}

struct JarvisDeviceMutation: Encodable, Equatable {
    let requestID: UUID
    let expectedRevision: Int
    enum CodingKeys: String, CodingKey {
        case requestID = "request_id", expectedRevision = "expected_revision"
    }
}

struct JarvisDeviceRegistration: Encodable, Equatable {
    let mutation: JarvisDeviceMutation
    let environment: String
    let topic: String
    let token: String
    let permission: JarvisPushPermission
    let enabled: Bool
    let changesQuietTime: Bool
    let quietUntil: String?

    enum CodingKeys: String, CodingKey {
        case environment, topic, token, permission, enabled, quietUntil = "quiet_until"
    }
    func encode(to encoder: Encoder) throws {
        try mutation.encode(to: encoder)
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(environment, forKey: .environment)
        try values.encode(topic, forKey: .topic)
        try values.encode(token, forKey: .token)
        try values.encode(permission, forKey: .permission)
        try values.encode(enabled, forKey: .enabled)
        // Omission preserves a saved quiet preference; explicit null clears it.
        if changesQuietTime { try values.encode(quietUntil, forKey: .quietUntil) }
    }
}

enum JarvisPendingPushChange: Equatable {
    case register(JarvisDeviceRegistration)
    case revoke(JarvisDeviceMutation)
    var mutation: JarvisDeviceMutation {
        switch self {
        case let .register(body): return body.mutation
        case let .revoke(body): return body
        }
    }
}

struct JarvisPushReceipt: Decodable {
    let device: JarvisPushDevice
    let acceptedRevision: Int
    enum CodingKeys: String, CodingKey { case device, acceptedRevision = "accepted_revision" }
}

// No device token or request body is written to local storage.
struct JarvisPushPreferences: Codable {
    var deviceID = UUID()
    var wantsEnabled = false
    var serverMayBeEnabled = false
    var reconciliationNeeded = false
}

@MainActor protocol JarvisPushPersistence {
    func load() throws -> JarvisPushPreferences
    func save(_ preferences: JarvisPushPreferences) throws
}

@MainActor protocol JarvisPushPlatform: AnyObject {
    var environment: String? { get }
    var topic: String { get }
    var presentationEnabled: Bool { get set }
    func permission() async -> JarvisPushPermission
    func requestPermission() async throws -> JarvisPushPermission
    func register()
    func unregister()
}

protocol JarvisPushAccess {
    func getPushStatus() async throws -> JarvisPushStatus
    func getPushDevices() async throws -> [JarvisPushDevice]
    func changePushDevice(id: UUID, change: JarvisPendingPushChange) async throws -> JarvisPushReceipt
}

struct JarvisNotificationLink: Identifiable, Equatable {
    let id = UUID()
    let intentID: UUID

    static func parse(_ userInfo: [AnyHashable: Any]) -> JarvisNotificationLink? {
        struct Pointer: Decodable { let version: Int; let intent_id: UUID }
        guard let value = userInfo["jarvis"],
              JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value),
              let pointer = try? JSONDecoder().decode(Pointer.self, from: data),
              pointer.version == 1 else { return nil }
        return JarvisNotificationLink(intentID: pointer.intent_id)
    }
}
