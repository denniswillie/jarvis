import Foundation

struct JarvisSandboxTarget: Codable, Equatable, Identifiable {
    var id: UUID { allocationID }
    let taskID: UUID?
    let allocationID: UUID
    let generation: String
    let title: String
    let expiresAt: String?
    let hermesSessionID: String?
    let state: String?
    let capability: String?
    let canView: Bool?
    let canResumeAndView: Bool?
    let observedAt: String?
    let stale: Bool?
    var isPaused: Bool { state == "paused" && stale != true }
    var isDesktop: Bool { capability == "desktop" || (capability == nil && taskID != nil) }
    var canOpen: Bool { isDesktop && stale != true && (canView ?? (state == nil || state == "running")) }
    var stateLabel: String {
        if capability == "terminal" { return "Terminal only" }
        if stale == true { return "Refresh status" }
        switch state {
        case "running": return "Running"
        case "paused": return "Paused"
        case "pausing": return "Pausing"
        case "resuming": return "Resuming"
        default: return taskID == nil ? "Status unavailable" : "Desktop"
        }
    }
    init(taskID: UUID? = nil, allocationID: UUID, generation: String, title: String, expiresAt: String? = nil,
         hermesSessionID: String? = nil, state: String? = nil, capability: String? = nil,
         canView: Bool? = nil, canResumeAndView: Bool? = nil, observedAt: String? = nil, stale: Bool? = nil) {
        self.taskID = taskID; self.allocationID = allocationID; self.generation = generation
        self.title = title; self.expiresAt = expiresAt; self.hermesSessionID = hermesSessionID
        self.state = state; self.capability = capability; self.canView = canView
        self.canResumeAndView = canResumeAndView; self.observedAt = observedAt; self.stale = stale
    }
    enum CodingKeys: String, CodingKey {
        case taskID = "task_id", allocationID = "allocation_id", generation, title, expiresAt = "expires_at"
        case hermesSessionID = "hermes_session_id", state, capability, canView = "can_view"
        case canResumeAndView = "can_resume_and_view", observedAt = "observed_at", stale
    }
}

struct JarvisSandboxChoices: Decodable {
    let items: [JarvisSandboxTarget]
    let hasMore: Bool
    let nextCursor: String?
    init(items: [JarvisSandboxTarget], hasMore: Bool, nextCursor: String? = nil) {
        self.items = items; self.hasMore = hasMore; self.nextCursor = nextCursor
    }
    enum CodingKeys: String, CodingKey { case items, hasMore = "has_more", nextCursor = "next_cursor" }
}

struct JarvisSandboxThumbnail: Decodable {
    let allocationID: UUID
    let generation: String
    let contentType: String
    let image: Data
    let capturedAt: String
    let stale: Bool
    enum CodingKeys: String, CodingKey {
        case allocationID = "allocation_id", generation, contentType = "content_type", image
        case capturedAt = "captured_at", stale
    }
}

struct JarvisAppControlResult: Decodable {
    struct Presentation: Decodable {
        let requestID: UUID
        let state: String?
        enum CodingKeys: String, CodingKey { case requestID = "request_id", state }
    }
    let outcome: String
    let summary: String
    let presentation: Presentation?
    let deviceCommandEnqueued: Bool?
    enum CodingKeys: String, CodingKey {
        case outcome, summary, presentation, deviceCommandEnqueued = "device_command_enqueued"
    }
}

struct JarvisSandboxLease: Decodable, Equatable, Identifiable {
    let id: UUID
    let target: JarvisSandboxTarget
    let expiresAt: String
    enum CodingKeys: String, CodingKey { case id, target, expiresAt = "expires_at" }
}

@MainActor
protocol JarvisSandboxAccess: AnyObject {
    func sandboxChoices(sessionID: UUID) async throws -> JarvisSandboxChoices
    func sandboxChoices(sessionID: UUID, after: String?) async throws -> JarvisSandboxChoices
    func controlApp(sessionID: UUID, operationID: UUID, command: [String: Any]) async throws -> JarvisAppControlResult
    func openSandboxView(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxLease
    func sandboxViewStatus(sessionID: UUID, id: UUID) async throws -> JarvisSandboxLease
    func closeSandboxView(sessionID: UUID, id: UUID) async throws
    func sandboxPageRequest(sessionID: UUID, id: UUID) throws -> URLRequest
    func sandboxThumbnail(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxThumbnail
}

extension JarvisSandboxAccess {
    func sandboxThumbnail(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxThumbnail {
        throw JarvisAPIError(code: "sandbox_thumbnail_unavailable", message: "No screen preview is available.")
    }
    func sandboxChoices(sessionID: UUID, after: String?) async throws -> JarvisSandboxChoices {
        guard after == nil else { throw JarvisAPIError(code: "sandbox_page_unavailable", message: "Refresh to see available computers.") }
        return try await sandboxChoices(sessionID: sessionID)
    }
    func controlApp(sessionID: UUID, operationID: UUID, command: [String: Any]) async throws -> JarvisAppControlResult {
        throw JarvisAPIError(code: "app_controls_unavailable", message: "App controls are unavailable.")
    }
}

extension JarvisAPIClient: JarvisSandboxAccess {
    func sandboxThumbnail(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxThumbnail {
        let query = sandboxQuery(sessionID) + [URLQueryItem(name: "allocation_id", value: allocationID.uuidString.lowercased()),
            URLQueryItem(name: "generation", value: generation)]
        return try await request(path: "/sandbox-views/thumbnail", method: "GET", queryItems: query,
                                 responseType: JarvisSandboxThumbnail.self)
    }
    func sandboxChoices(sessionID: UUID) async throws -> JarvisSandboxChoices {
        try await sandboxChoices(sessionID: sessionID, after: nil)
    }
    func sandboxChoices(sessionID: UUID, after: String?) async throws -> JarvisSandboxChoices {
        var query = sandboxQuery(sessionID)
        if let after { query.append(URLQueryItem(name: "after", value: after)) }
        return try await request(path: "/sandbox-views", method: "GET", queryItems: query, responseType: JarvisSandboxChoices.self)
    }
    func controlApp(sessionID: UUID, operationID: UUID, command: [String: Any]) async throws -> JarvisAppControlResult {
        let body = try JSONSerialization.data(withJSONObject: ["session_id": sessionID.uuidString.lowercased(),
            "operation_id": operationID.uuidString.lowercased(), "command": command])
        return try await request(path: "/app-controls", method: "POST", body: body, responseType: JarvisAppControlResult.self)
    }
    func openSandboxView(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxLease {
        let body = try JSONSerialization.data(withJSONObject: ["session_id": sessionID.uuidString.lowercased(),
            "allocation_id": allocationID.uuidString.lowercased(), "generation": generation])
        return try await request(path: "/sandbox-views", method: "POST", body: body, responseType: JarvisSandboxLease.self)
    }
    func sandboxViewStatus(sessionID: UUID, id: UUID) async throws -> JarvisSandboxLease {
        try await request(path: "/sandbox-views/" + id.uuidString.lowercased(), method: "GET",
                          queryItems: sandboxQuery(sessionID), responseType: JarvisSandboxLease.self)
    }
    func closeSandboxView(sessionID: UUID, id: UUID) async throws {
        struct Response: Decodable { let closed: Bool }
        let _: Response = try await request(path: "/sandbox-views/" + id.uuidString.lowercased(), method: "DELETE",
                                            queryItems: sandboxQuery(sessionID), responseType: Response.self)
    }
    private func sandboxQuery(_ sessionID: UUID) -> [URLQueryItem] {
        [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())]
    }
}
