#if DEBUG
import Foundation

/// Proposed backend contract. No local boolean can authorize an evaluation.
/// A 404/missing or mismatched attestation is a hard stop before session creation.
struct JarvisReplayScopeExpectation {
    let runID: UUID
    let ownerID: String
    let areaID: String
    let memoryBankID: String
}

struct JarvisReplayScope: Decodable {
    static func wireRunID(_ id: UUID) -> String { id.uuidString.replacingOccurrences(of: "-", with: "").lowercased() }
    static func parseRunID(_ value: String) throws -> UUID {
        if let id = UUID(uuidString: value) { return id }
        let chars = Array(value)
        guard chars.count == 32, chars.allSatisfy({ $0.isHexDigit }) else { throw JarvisReplayError.isolationUnavailable }
        let formatted = [0..<8, 8..<12, 12..<16, 16..<20, 20..<32].map { String(chars[$0]) }.joined(separator: "-")
        guard let id = UUID(uuidString: formatted) else { throw JarvisReplayError.isolationUnavailable }
        return id
    }
    static func applyRunHeader(_ runID: UUID?, to request: inout URLRequest) {
        if let runID { request.setValue(wireRunID(runID), forHTTPHeaderField: "X-Jarvis-Evaluation-Run") }
    }

    static func boundSocketURL(_ url: URL, base: URL, sessionID: String?) throws -> URL {
        try validateSocketURL(url, base: base)
        guard let sessionID, let session = UUID(uuidString: sessionID),
              var parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { throw JarvisReplayError.isolationUnavailable }
        var items = parts.queryItems ?? []
        let existing = items.filter { $0.name == "session_id" }
        guard existing.count <= 1, existing.isEmpty || UUID(uuidString: existing[0].value ?? "") == session else {
            throw JarvisReplayError.isolationUnavailable
        }
        if existing.isEmpty { items.append(URLQueryItem(name: "session_id", value: session.uuidString.lowercased())) }
        parts.queryItems = items
        guard let result = parts.url else { throw JarvisReplayError.isolationUnavailable }
        return result
    }

    static func validateSocketURL(_ url: URL, base: URL) throws {
        let expectedScheme = base.scheme == "https" ? "wss" : "ws"
        let defaultPort = base.scheme == "https" ? 443 : 80
        guard url.scheme == expectedScheme, url.host == base.host,
              (url.port ?? defaultPort) == (base.port ?? defaultPort),
              url.user == nil, url.password == nil, url.fragment == nil else {
            throw JarvisReplayError.isolationUnavailable
        }
    }

    let runID: UUID
    let ownerID: String
    let areaID: String
    let memoryBankID: String
    let provenance: String
    let isolationEnforced: Bool
    let freshSessionRequired: Bool
    let externalNotificationsEnabled: Bool
    let expiresAt: String
    enum CodingKeys: String, CodingKey {
        case runID = "run_id", ownerID = "owner_id", areaID = "area_id", memoryBankID = "memory_bank_id"
        case provenance, isolationEnforced = "isolation_enforced", freshSessionRequired = "fresh_session_required"
        case externalNotificationsEnabled = "external_notifications_enabled", expiresAt = "expires_at"
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        runID = try Self.parseRunID(values.decode(String.self, forKey: .runID))
        ownerID = try values.decode(String.self, forKey: .ownerID)
        areaID = try values.decode(String.self, forKey: .areaID)
        memoryBankID = try values.decode(String.self, forKey: .memoryBankID)
        provenance = try values.decode(String.self, forKey: .provenance)
        isolationEnforced = try values.decode(Bool.self, forKey: .isolationEnforced)
        freshSessionRequired = try values.decode(Bool.self, forKey: .freshSessionRequired)
        externalNotificationsEnabled = try values.decode(Bool.self, forKey: .externalNotificationsEnabled)
        expiresAt = try values.decode(String.self, forKey: .expiresAt)
    }

    func validate(_ expected: JarvisReplayScopeExpectation, now: Date, remaining: TimeInterval) throws {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var expiration = formatter.date(from: expiresAt)
        if expiration == nil { formatter.formatOptions = [.withInternetDateTime]; expiration = formatter.date(from: expiresAt) }
        guard runID == expected.runID, ownerID == expected.ownerID, areaID == expected.areaID,
              memoryBankID == expected.memoryBankID, !ownerID.isEmpty, !areaID.isEmpty, !memoryBankID.isEmpty,
              provenance == "frontend_replay", isolationEnforced, freshSessionRequired,
              !externalNotificationsEnabled, let expiration, expiration.timeIntervalSince(now) >= remaining else {
            throw JarvisReplayError.isolationUnavailable
        }
    }

    static func authorize(configuration: JarvisConfiguration, expected: JarvisReplayScopeExpectation,
                          remaining: TimeInterval, session: URLSession = .shared) async throws -> Self {
        let base = configuration.backendBaseURL
        guard base.scheme == "https" || (base.scheme == "http" && ["localhost", "127.0.0.1", "::1"].contains(base.host ?? "")),
              !configuration.staticBearerToken.isEmpty else { throw JarvisReplayError.isolationUnavailable }
        let url = base.appendingPathComponent("eval/replay/\(wireRunID(expected.runID))/scope")
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 10)
        request.setValue("Bearer \(configuration.staticBearerToken)", forHTTPHeaderField: "Authorization")
        applyRunHeader(expected.runID, to: &request)
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200, data.count <= 8192 else {
            throw JarvisReplayError.isolationUnavailable
        }
        let scope = try JSONDecoder().decode(Self.self, from: data)
        try scope.validate(expected, now: Date(), remaining: remaining)
        return scope
    }
}
#endif
