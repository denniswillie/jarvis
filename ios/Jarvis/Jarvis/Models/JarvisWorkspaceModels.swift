import Foundation

/// Uses the app's existing authenticated transport. This is not a second session/client.
protocol JarvisWorkspaceAccess {
    func listWorkspaceRequests(sessionID: UUID) async throws -> JarvisWorkspaceRequestPage
    func acknowledgeWorkspaceRequest(requestID: UUID, ack: JarvisWorkspaceAcknowledgement) async throws -> JarvisWorkspaceRequestResponse
    func getWorkspaceFileContent(fileID: UUID, sessionID: UUID) async throws -> JarvisWorkspaceFileContent
}

struct JarvisWorkspaceError: Error, LocalizedError {
    let code: String
    let message: String
    var retryAfter: TimeInterval? = nil
    var errorDescription: String? { message }
}

enum JarvisWorkspacePresentationState: String, Codable {
    case pending, claimed, displayed, savedLocal = "saved_local", dismissed, failed, expired
    var isTerminal: Bool { self != .pending && self != .claimed }
}

struct JarvisWorkspaceRequest: Codable, Equatable, Identifiable {
    enum Action: String, Codable { case open, download }
    enum MediaTransferPolicy: String, Codable {
        case capture = "mp4_capture_v1", output = "mp4_output_v1"
        var maximumByteCount: Int { self == .capture ? 1024 * 1024 * 1024 : 500_000_000 }
        // Capture alone permits a small codec/container tail beyond 30 minutes.
        var maximumDurationMilliseconds: Int { self == .capture ? 1_800_100 : 1_800_000 }
    }
    let requestID: UUID
    let action: Action
    let artifactKind: String
    let artifactID: UUID
    let versionID: String
    let format: String?
    let contentReady: Bool
    let fileID: UUID?
    let byteCount: Int?
    let contentSHA256: String?
    let state: JarvisWorkspacePresentationState
    let createdAt: String
    let expiresAt: String
    let errorCode: String?
    let localReceipt: JarvisWorkspaceLocalReceipt?
    var transferPolicy: MediaTransferPolicy? = nil
    var transferLimitBytes: Int? = nil
    var id: UUID { requestID }
    func hasSameTarget(as other: Self) -> Bool {
        requestID == other.requestID && action == other.action && artifactKind == other.artifactKind &&
        artifactID == other.artifactID && versionID == other.versionID && format == other.format &&
        contentReady == other.contentReady && fileID == other.fileID && byteCount == other.byteCount &&
        contentSHA256 == other.contentSHA256 && transferPolicy == other.transferPolicy &&
        transferLimitBytes == other.transferLimitBytes
    }
    /// Missing media fields preserve the existing document ceiling. A partial or
    /// wider policy is never inferred from MIME, file size or a UI preference.
    func validatedTransferByteLimit() throws -> Int {
        if let policy = transferPolicy {
            guard action == .download, artifactKind == "file", contentReady, fileID != nil,
                  let limit = transferLimitBytes, limit > 0, limit <= policy.maximumByteCount,
                  let count = byteCount, count > 0, count <= limit,
                  let hash = contentSHA256, hash.utf8.count == 64,
                  hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw JarvisWorkspaceError(code: "invalid_transfer_policy", message: "The saved video's transfer limit could not be verified.")
            }
            return limit
        }
        guard transferLimitBytes == nil, byteCount == nil || (0...(8 * 1024 * 1024)).contains(byteCount!) else {
            throw JarvisWorkspaceError(code: "invalid_transfer_policy", message: "The saved file exceeds the supported document limit.")
        }
        return 8 * 1024 * 1024
    }
    enum CodingKeys: String, CodingKey {
        case requestID = "request_id", action, artifactKind = "artifact_kind", artifactID = "artifact_id"
        case versionID = "version_id", format, contentReady = "content_ready", fileID = "file_id"
        case byteCount = "byte_count", contentSHA256 = "content_sha256", state
        case createdAt = "created_at", expiresAt = "expires_at", errorCode = "error_code", localReceipt = "local_receipt"
        case transferPolicy = "transfer_policy", transferLimitBytes = "transfer_limit_bytes"
    }
}

struct JarvisWorkspaceRequestPage: Codable { let items: [JarvisWorkspaceRequest] }
struct JarvisWorkspaceRequestResponse: Codable { let request: JarvisWorkspaceRequest }
struct JarvisWorkspaceLocalReceipt: Codable, Equatable {
    let fileID: UUID
    let versionID: String
    let byteCount: Int
    let contentSHA256: String
    let destination: String
    enum CodingKeys: String, CodingKey {
        case fileID = "file_id", versionID = "version_id", byteCount = "byte_count"
        case contentSHA256 = "content_sha256", destination
    }
}

struct JarvisWorkspaceAcknowledgement: Codable, Equatable {
    let sessionID: UUID
    let state: JarvisWorkspacePresentationState
    var localReceipt: JarvisWorkspaceLocalReceipt? = nil
    var errorCode: String? = nil
    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id", state, localReceipt = "local_receipt", errorCode = "error_code"
    }
}

struct JarvisWorkspaceFileContent {
    let data: Data
    let filename: String
    let contentType: String
}

/// A validated durable file location. The caller must retain the original
/// account lifetime and make a private preview copy before handing it to UI.
struct JarvisWorkspaceLocalFileContent {
    let url: URL
    let filename: String
    let contentType: String
    let byteCount: Int
    let contentSHA256: String
}

