import Foundation

/// Binds all library operations to the same independently verified data/voice
/// session. The local fence only retires this client; it grants no authority.
final class JarvisSessionFilesClient: JarvisFilesMediaAccess, JarvisResourceFileAccess, JarvisThumbnailAccess, JarvisMediaProgressAccess {
    struct MediaLimits {
        let byteCount: Int
        let durationMilliseconds: Int
    }

    private let api: JarvisAPIClient
    private let sessionID: UUID
    private let contextAccess: JarvisWorkspaceLocalAccess
    private let mediaLimits: MediaLimits?

    init(api: JarvisAPIClient, sessionID: UUID, contextAccess: JarvisWorkspaceLocalAccess,
         mediaLimits: MediaLimits? = nil) {
        self.api = api
        self.sessionID = sessionID
        self.contextAccess = contextAccess
        self.mediaLimits = mediaLimits
    }

    func getResourceFiles(query: String?) async throws -> JarvisResourceFilePage {
        try await getResourceFiles(query: query, cursor: nil)
    }

    func getThumbnail(file: JarvisRetainedFile, reference: JarvisResourceReference,
                      access: JarvisWorkspaceLocalAccess) async throws -> Data {
        try checkContext()
        try access.check()
        guard UUID(uuidString: file.id) == reference.fileID, file.mediaType == reference.mediaType,
              let thumbnail = file.thumbnail else {
            throw JarvisAPIError(code: "thumbnail_unavailable", message: "A preview is not available yet.")
        }
        try thumbnail.validate(reference: reference)
        let data = try await api.getThumbnail(thumbnail, sessionID: sessionID, access: access)
        try checkContext()
        try access.check()
        return data
    }

    func getResourceFiles(query: String?, cursor: String?) async throws -> JarvisResourceFilePage {
        try checkContext()
        var items = sessionQuery + [URLQueryItem(name: "limit", value: "8"),
            URLQueryItem(name: "media_only", value: "true")]
        if let query, !query.isEmpty { items.append(URLQueryItem(name: "query", value: query)) }
        if let cursor { items.append(URLQueryItem(name: "cursor", value: cursor)) }
        let result: JarvisResourceFilePage = try await api.request(path: "/files", method: "GET",
            queryItems: items, responseType: JarvisResourceFilePage.self)
        try checkContext()
        return result
    }

    func getResourceContent(file: JarvisRetainedFile, reference: JarvisResourceReference) async throws -> Data {
        try checkContext()
        guard UUID(uuidString: file.id) == reference.fileID, file.mediaType == reference.mediaType,
              file.mediaType != "video/mp4", file.sizeBytes >= 0, file.sizeBytes <= 8 * 1024 * 1024 else {
            throw JarvisAPIError(code: "file_identity_changed", message: "The selected file does not match its reference.")
        }
        let result = try await api.getRetainedFileContent(file: file, sessionID: sessionID.uuidString.lowercased(),
            versionID: reference.versionID)
        try checkContext()
        return result
    }

    func getResourceFile(id: UUID, versionID: String?) async throws -> JarvisResourceFile {
        try checkContext()
        var items = sessionQuery
        if let versionID {
            guard !versionID.isEmpty, versionID.utf8.count <= 256 else { throw invalidMedia() }
            items.append(URLQueryItem(name: "version_id", value: versionID))
        }
        let response: JarvisResourceFileResponse = try await api.request(
            path: "/files/" + id.uuidString.lowercased(), method: "GET", queryItems: items,
            responseType: JarvisResourceFileResponse.self)
        try checkContext()
        guard response.file.reference.fileID == id,
              versionID == nil || response.file.reference.versionID == versionID else {
            throw JarvisAPIError(code: "file_identity_changed", message: "The selected file changed. Refresh the library.")
        }
        return response.file
    }

    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage {
        try checkContext()
        var items = sessionQuery + [URLQueryItem(name: "limit", value: "8")]
        if let query, !query.isEmpty { items.append(URLQueryItem(name: "query", value: query)) }
        let result: JarvisRetainedFilePage = try await api.request(path: "/files", method: "GET",
            queryItems: items, responseType: JarvisRetainedFilePage.self)
        try checkContext()
        return result
    }

    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        try checkContext()
        let result = try await api.getRetainedFile(id: id, sessionID: sessionID.uuidString.lowercased())
        try checkContext()
        return result
    }

    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        try checkContext()
        // Media never falls back to whole-Data document delivery, even when
        // the particular movie happens to be smaller than the document cap.
        guard file.mediaType != "video/mp4", file.sizeBytes >= 0,
              file.sizeBytes <= 8 * 1024 * 1024 else {
            throw JarvisAPIError(code: "file_transfer_unavailable", message: "This file needs a supported preview.")
        }
        let result = try await api.getRetainedFileContent(file: file, sessionID: sessionID.uuidString.lowercased())
        try checkContext()
        return result
    }

    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        try checkContext()
        guard let fileID = UUID(uuidString: id) else {
            throw JarvisAPIError(code: "invalid_file_id", message: "The saved file identifier is invalid.")
        }
        let result: JarvisFileDeletion = try await api.request(path: "/files/" + fileID.uuidString.lowercased(),
            method: "DELETE", queryItems: sessionQuery, responseType: JarvisFileDeletion.self)
        try checkContext()
        return result
    }

    func getMediaFile(id: UUID, versionID: String?, access: JarvisWorkspaceLocalAccess) async throws -> JarvisMediaFile {
        try checkContext()
        try access.check()
        let limits = try qualifiedMediaLimits()
        var query = sessionQuery
        if let versionID {
            guard !versionID.isEmpty, versionID.utf8.count <= 256 else { throw invalidMedia() }
            query.append(URLQueryItem(name: "version_id", value: versionID))
        }
        let response: JarvisMediaFileResponse = try await api.request(path: "/files/" + id.uuidString.lowercased(),
            method: "GET", queryItems: query, responseType: JarvisMediaFileResponse.self)
        try checkContext()
        try access.check()
        // A fresh library selection can discover the immutable source version;
        // later export/refresh requests must keep the selected exact version.
        try response.file.validate(fileID: id, versionID: versionID ?? response.file.versionID,
            maximumByteCount: limits.byteCount, maximumDurationMilliseconds: limits.durationMilliseconds)
        return response.file
    }

    func getMediaContent(_ media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess) async throws -> JarvisVerifiedTemporaryFile {
        try await getMediaContent(media, access: access, progress: { _ in })
    }

    func getMediaContent(_ media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess,
                         progress: @escaping @Sendable (Double) -> Void) async throws -> JarvisVerifiedTemporaryFile {
        try checkContext()
        try access.check()
        let limits = try qualifiedMediaLimits()
        let file = try await api.downloadRetainedMedia(media, sessionID: sessionID,
            maximumByteCount: limits.byteCount, maximumDurationMilliseconds: limits.durationMilliseconds,
            access: access, verifier: JarvisTemporaryFileVerifier(), progress: progress)
        do {
            try checkContext()
            try access.check()
            return file
        } catch {
            file.discard()
            throw error
        }
    }

    private var sessionQuery: [URLQueryItem] {
        [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())]
    }

    func checkContext() throws {
        try Task.checkCancellation()
        try contextAccess.check()
    }

    private func qualifiedMediaLimits() throws -> MediaLimits {
        guard let mediaLimits, mediaLimits.byteCount > 0, mediaLimits.byteCount <= JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumByteCount,
              mediaLimits.durationMilliseconds > 0,
              mediaLimits.durationMilliseconds <= JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumDurationMilliseconds else {
            throw invalidMedia()
        }
        return mediaLimits
    }

    private func invalidMedia() -> JarvisAPIError {
        JarvisAPIError(code: "media_transfer_unavailable", message: "Video previews are not available for this connection.")
    }
}
