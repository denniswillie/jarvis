import Foundation
import CryptoKit

extension JarvisAPIClient: JarvisWorkspaceMediaAccess {
    func getWorkspaceMediaContent(request: JarvisWorkspaceRequest, sessionID: UUID,
                                  access: JarvisWorkspaceLocalAccess) async throws -> JarvisWorkspaceMediaFileContent {
        let preparedLimit = try request.validatedTransferByteLimit()
        let id: UUID, limit: Int, duration: Int
        if request.action == .download, let policy = request.transferPolicy, let fileID = request.fileID {
            id = fileID; limit = preparedLimit; duration = policy.maximumDurationMilliseconds
        } else if request.action == .open, request.artifactKind == "file", request.contentReady,
                  request.transferPolicy == nil, request.format == nil {
            id = request.artifactID
            // The largest qualified preview bound is narrowed by independently
            // fetched exact metadata. This does not classify the file as capture.
            limit = JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumByteCount; duration = JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumDurationMilliseconds
        } else {
            throw JarvisWorkspaceError(code: "invalid_transfer_policy", message: "An exact saved video is required.")
        }
        let media = try await getRetainedMediaFile(id: id, versionID: request.versionID, sessionID: sessionID,
            maximumByteCount: limit, maximumDurationMilliseconds: duration, access: access)
        guard request.action == .open || (media.file.sizeBytes == request.byteCount && media.contentSHA256 == request.contentSHA256) else {
            throw JarvisWorkspaceError(code: "download_mismatch", message: "The video no longer matches the prepared saved version.")
        }
        let file = try await downloadRetainedMedia(media, sessionID: sessionID, maximumByteCount: limit,
            maximumDurationMilliseconds: duration, access: access,
            verifier: JarvisTemporaryFileVerifier())
        return JarvisWorkspaceMediaFileContent(temporaryFile: file, filename: media.file.exportFilename,
            contentType: "video/mp4", maximumByteCount: min(limit, media.transferLimitBytes),
            maximumDurationMilliseconds: min(duration, media.inspection.durationMilliseconds))
    }

    func getRetainedMediaFile(id: UUID, versionID: String, sessionID: UUID,
                             maximumByteCount: Int, maximumDurationMilliseconds: Int,
                             access: JarvisWorkspaceLocalAccess) async throws -> JarvisMediaFile {
        try Task.checkCancellation()
        try access.check()
        guard !versionID.isEmpty, versionID.utf8.count <= 256, maximumByteCount > 0,
              maximumDurationMilliseconds > 0 else {
            throw JarvisAPIError(code: "invalid_media_selection", message: "Select an exact saved video version.")
        }
        let response: JarvisMediaFileResponse = try await request(path: "/files/" + id.uuidString.lowercased(),
            method: "GET", queryItems: mediaQuery(versionID: versionID, sessionID: sessionID),
            responseType: JarvisMediaFileResponse.self)
        try Task.checkCancellation()
        try access.check()
        try response.file.validate(fileID: id, versionID: versionID, maximumByteCount: maximumByteCount,
                                   maximumDurationMilliseconds: maximumDurationMilliseconds)
        return response.file
    }

    /// This produces only a private temporary copy. Durable journal adoption and
    /// any saved_local receipt remain the presentation owner's separate work.
    func downloadRetainedMedia(_ media: JarvisMediaFile, sessionID: UUID,
                               maximumByteCount: Int, maximumDurationMilliseconds: Int,
                               access: JarvisWorkspaceLocalAccess,
                               verifier: JarvisTemporaryFileVerifier,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> JarvisVerifiedTemporaryFile {
        try Task.checkCancellation()
        try access.check()
        guard let id = UUID(uuidString: media.file.id) else {
            throw JarvisAPIError(code: "invalid_file_id", message: "The saved file identifier is invalid.")
        }
        let limit = try media.validate(fileID: id, versionID: media.versionID, maximumByteCount: maximumByteCount,
                                       maximumDurationMilliseconds: maximumDurationMilliseconds)
        let delegate = JarvisMediaDownloadDelegate(media: media, access: access, progress: progress)
        do {
            let (location, response, requestedURL) = try await downloadedResponse(
                path: "/files/" + id.uuidString.lowercased() + "/content",
                queryItems: mediaQuery(versionID: media.versionID, sessionID: sessionID),
                accept: "video/mp4", delegate: delegate)
            defer { try? FileManager.default.removeItem(at: location) }
            try Task.checkCancellation()
            try access.check()
            try delegate.checkRejection()
            try media.validateResponse(response, requestedURL: requestedURL)
            let verified = try await verifier.copyVerifiedFile(from: location, expectedByteCount: media.file.sizeBytes,
                expectedSHA256: media.contentSHA256, maximumByteCount: limit, access: access, format: .mp4)
            do {
                try Task.checkCancellation()
                try access.check()
                return verified
            } catch {
                verified.discard()
                throw error
            }
        } catch {
            try access.check()
            try delegate.checkRejection()
            throw error
        }
    }

    private func mediaQuery(versionID: String, sessionID: UUID) -> [URLQueryItem] {
        [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased()),
         URLQueryItem(name: "version_id", value: versionID)]
    }
}

/// URLSession owns the partial download; no partial location is exposed to UI.
/// Its progress callbacks bound the transfer and retire stale account work.
private final class JarvisMediaDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let media: JarvisMediaFile
    private let progress: (@Sendable (Double) -> Void)?
    private let access: JarvisWorkspaceLocalAccess
    private let lock = NSLock()
    private var rejection: Error?

    init(media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess, progress: (@Sendable (Double) -> Void)?) {
        self.media = media
        self.progress = progress
        self.access = access
    }

    func checkRejection() throws {
        lock.lock(); defer { lock.unlock() }
        if let rejection { throw rejection }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        do {
            try access.check()
            guard let response = downloadTask.response as? HTTPURLResponse,
                  let url = downloadTask.originalRequest?.url else {
                throw JarvisAPIError(code: "media_transfer_mismatch", message: "The video transfer does not match the selected size.")
            }
            try media.validateResponse(response, requestedURL: url)
            guard totalBytesWritten <= Int64(media.file.sizeBytes), totalBytesExpectedToWrite == Int64(media.file.sizeBytes) else {
                throw JarvisAPIError(code: "media_transfer_mismatch", message: "The video transfer does not match the selected size.")
            }
            progress?(Double(totalBytesWritten) / Double(media.file.sizeBytes))
        } catch { reject(error, task: downloadTask) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        reject(JarvisAPIError(code: "media_transfer_redirect", message: "The video transfer changed its destination."), task: task)
        completionHandler(nil)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The async download completion owns this location and immediately verifies/copies it.
    }

    private func reject(_ error: Error, task: URLSessionTask) {
        lock.lock()
        if rejection == nil { rejection = error }
        lock.unlock()
        task.cancel()
    }
}


extension JarvisAPIClient {
    func getThumbnail(_ thumbnail: JarvisThumbnailReference, sessionID: UUID,
                      access: JarvisWorkspaceLocalAccess) async throws -> Data {
        let delegate = JarvisThumbnailDownloadDelegate(thumbnail: thumbnail, access: access)
        do {
            let (location, response, url) = try await downloadedResponse(path: thumbnail.path,
                queryItems: [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased()),
                             URLQueryItem(name: "version_id", value: thumbnail.versionID)],
                accept: "image/jpeg", delegate: delegate)
            defer { try? FileManager.default.removeItem(at: location) }
            try Task.checkCancellation()
            try access.check()
            try delegate.checkRejection()
            try delegate.validate(response, url: url)
            let size = try location.resourceValues(forKeys: [.fileSizeKey]).fileSize
            guard size == thumbnail.sizeBytes else { throw delegate.mismatch() }
            let data = try Data(contentsOf: location)
            let checksum = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard checksum == thumbnail.contentSHA256 else { throw delegate.mismatch() }
            return data
        } catch {
            try access.check()
            try delegate.checkRejection()
            throw error
        }
    }
}

private final class JarvisThumbnailDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let thumbnail: JarvisThumbnailReference
    let access: JarvisWorkspaceLocalAccess
    private let lock = NSLock()
    private var rejection: Error?

    init(thumbnail: JarvisThumbnailReference, access: JarvisWorkspaceLocalAccess) {
        self.thumbnail = thumbnail; self.access = access
    }
    func mismatch() -> JarvisAPIError {
        JarvisAPIError(code: "thumbnail_unavailable", message: "This preview could not be loaded. You can still open the original.")
    }
    func checkRejection() throws {
        lock.lock(); defer { lock.unlock() }
        if let rejection { throw rejection }
    }
    func validate(_ response: HTTPURLResponse, url: URL) throws {
        if response.statusCode == 401 || response.statusCode == 403 {
            throw JarvisAPIError(code: "thumbnail_access_denied", message: "Sign in again to view this library.",
                                 statusCode: response.statusCode)
        }
        guard response.statusCode == 200, response.url == url,
              response.mimeType == "image/jpeg", response.expectedContentLength == Int64(thumbnail.sizeBytes),
              response.value(forHTTPHeaderField: "X-Jarvis-Version-Id") == thumbnail.versionID,
              response.value(forHTTPHeaderField: "X-Jarvis-Content-Sha256") == thumbnail.contentSHA256,
              response.value(forHTTPHeaderField: "Content-Range") == nil,
              response.value(forHTTPHeaderField: "Content-Encoding") == nil else { throw mismatch() }
    }
    private func reject(_ error: Error, task: URLSessionTask) {
        lock.lock(); if rejection == nil { rejection = error }; lock.unlock()
        task.cancel()
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        do {
            try access.check()
            guard totalBytesWritten <= Int64(thumbnail.sizeBytes),
                  let response = downloadTask.response as? HTTPURLResponse,
                  let url = downloadTask.originalRequest?.url else { throw mismatch() }
            try validate(response, url: url)
        } catch { reject(error, task: downloadTask) }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        reject(mismatch(), task: task); completionHandler(nil)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}
