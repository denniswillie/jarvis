import Foundation

enum JarvisAppState: Equatable {
    case launching
    case needsSetup
    case ready
    case connecting
    case active
    case error(String)
}

struct JarvisConfiguration: Equatable {
    static let defaultBackendBaseURLString = "https://your-jarvis.example.invalid"

    var backendBaseURL: URL
    var staticBearerToken: String
}

struct JarvisSession: Codable, Identifiable, Equatable {
    let id: String
    let title: String
    let status: String
    let createdAt: String
    let updatedAt: String
    let closedAt: String?
    let lastActivityAt: String
    let behaviorInstructions: String

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case status
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case closedAt = "closed_at"
        case lastActivityAt = "last_activity_at"
        case behaviorInstructions = "behavior_instructions"
    }
}

struct JarvisRealtimeConfig: Codable, Equatable {
    let audioWebsocketURL: String
    let visualWebsocketURL: String

    enum CodingKeys: String, CodingKey {
        case audioWebsocketURL = "audio_websocket_url"
        case visualWebsocketURL = "visual_websocket_url"
    }
}

struct StartSessionResponse: Codable, Equatable {
    let session: JarvisSession
    let realtime: JarvisRealtimeConfig
}

struct ActiveSessionResponse: Codable, Equatable {
    let session: JarvisSession?
    let realtime: JarvisRealtimeConfig?
}

struct SessionDetailResponse: Codable, Equatable {
    let session: JarvisSession
    let realtime: JarvisRealtimeConfig
}

struct JarvisAPIError: Error, LocalizedError, Equatable {
    let code: String
    let message: String
    let statusCode: Int?
    // Server-advised seconds measured when the response arrived, not retry permission.
    let retryAfter: TimeInterval?

    init(code: String, message: String, statusCode: Int? = nil) {
        self.init(code: code, message: message, statusCode: statusCode, retryAfter: nil)
    }

    init(code: String, message: String, statusCode: Int?, retryAfter: TimeInterval?) {
        self.code = code
        self.message = message
        self.statusCode = statusCode
        self.retryAfter = retryAfter
    }

    var errorDescription: String? {
        message
    }
}

// Server-verified local journal scope. Never derive this from a bearer or session.
struct JarvisAuthenticatedContext: Decodable, Equatable, Sendable {
    let ownerID: String
    let backendNamespace: String
    let sessionID: UUID

    enum CodingKeys: String, CodingKey {
        case ownerID = "owner_id"
        case backendNamespace = "backend_namespace"
        case sessionID = "session_id"
    }

    var hasValidIdentity: Bool {
        guard !ownerID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              ownerID.utf8.count <= 512,
              backendNamespace.hasPrefix("jarvis-storage-v1:") else { return false }
        let digest = backendNamespace.dropFirst("jarvis-storage-v1:".count)
        return digest.count == 64 && digest.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}
