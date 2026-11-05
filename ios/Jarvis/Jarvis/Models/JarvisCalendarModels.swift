import Foundation

struct GoogleOAuthStartResponse: Decodable {
    let authorizationURL: URL
    let expiresInSeconds: Int

    enum CodingKeys: String, CodingKey {
        case authorizationURL = "authorization_url"
        case expiresInSeconds = "expires_in_seconds"
    }
}

struct GoogleCalendarStatusResponse: Decodable {
    let connected: Bool
    let expiresAt: String?
    let scopes: [String]

    enum CodingKeys: String, CodingKey {
        case connected
        case expiresAt = "expires_at"
        case scopes
    }
}
