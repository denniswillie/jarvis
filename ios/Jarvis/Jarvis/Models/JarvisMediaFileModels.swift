import Foundation

/// Library selection/export has no C0 delivery request to acknowledge.
protocol JarvisFilesMediaAccess: JarvisFileAccess {
    func checkContext() throws
    func getMediaFile(id: UUID, versionID: String?, access: JarvisWorkspaceLocalAccess) async throws -> JarvisMediaFile
    func getMediaContent(_ media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess) async throws -> JarvisVerifiedTemporaryFile
}

protocol JarvisMediaProgressAccess: JarvisFilesMediaAccess {
    func getMediaContent(_ media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess,
                         progress: @escaping @Sendable (Double) -> Void) async throws -> JarvisVerifiedTemporaryFile
}

protocol JarvisWorkspaceMediaAccess {
    func getWorkspaceMediaContent(request: JarvisWorkspaceRequest, sessionID: UUID,
                                  access: JarvisWorkspaceLocalAccess) async throws -> JarvisWorkspaceMediaFileContent
}

struct JarvisWorkspaceMediaFileContent {
    let temporaryFile: JarvisVerifiedTemporaryFile
    let filename: String
    let contentType: String
    // The transport narrows these to the selected authenticated media metadata.
    // Default to the shared output policy, not the wider capture policy.
    var maximumByteCount: Int = JarvisWorkspaceRequest.MediaTransferPolicy.output.maximumByteCount
    var maximumDurationMilliseconds: Int = JarvisWorkspaceRequest.MediaTransferPolicy.output.maximumDurationMilliseconds
}

/// Export owns a separately verified private copy, independent of playback.
struct JarvisMediaExport: Identifiable, Sendable {
    let id = UUID()
    let filename: String
    let file: JarvisVerifiedTemporaryFile
}

/// Exact media metadata from the existing authenticated Files endpoint.
/// Kept separate from the small-document model and its unchanged 8 MiB limit.
struct JarvisMediaFile: Decodable, Equatable {
    let file: JarvisRetainedFile
    let versionID: String
    let contentSHA256: String
    let transferLimitBytes: Int
    let transferPolicy: JarvisWorkspaceRequest.MediaTransferPolicy?
    let inspection: Inspection

    struct Inspection: Decodable, Equatable {
        let format: String
        let contentSHA256: String
        let sizeBytes: Int
        let durationMilliseconds: Int
        let width: Int
        let height: Int
        let videoCodec: String
        let audioCodec: String?

        enum CodingKeys: String, CodingKey {
            case format, width, height
            case contentSHA256 = "content_sha256", sizeBytes = "size_bytes"
            case durationMilliseconds = "duration_ms", videoCodec = "video_codec", audioCodec = "audio_codec"
        }
    }

    enum CodingKeys: String, CodingKey {
        case versionID = "version_id", contentSHA256 = "content_sha256"
        case transferLimitBytes = "transfer_limit_bytes", inspection = "media_inspection"
        case transferPolicy = "transfer_policy"
    }

    init(from decoder: Decoder) throws {
        file = try JarvisRetainedFile(from: decoder)
        let values = try decoder.container(keyedBy: CodingKeys.self)
        versionID = try values.decode(String.self, forKey: .versionID)
        contentSHA256 = try values.decode(String.self, forKey: .contentSHA256)
        transferLimitBytes = try values.decode(Int.self, forKey: .transferLimitBytes)
        transferPolicy = try values.decodeIfPresent(JarvisWorkspaceRequest.MediaTransferPolicy.self, forKey: .transferPolicy)
        inspection = try values.decode(Inspection.self, forKey: .inspection)
    }

    /// The caller supplies its qualified policy; advertised capacity cannot widen it.
    @discardableResult
    func validate(fileID: UUID, versionID expectedVersion: String, maximumByteCount: Int,
                  maximumDurationMilliseconds: Int) throws -> Int {
        let limit = min(transferLimitBytes, maximumByteCount, transferPolicy?.maximumByteCount ?? maximumByteCount)
        // Older metadata can never grant the capture-only encoding allowance.
        let durationLimit = min(maximumDurationMilliseconds, transferPolicy?.maximumDurationMilliseconds ?? 300_000)
        guard UUID(uuidString: file.id) == fileID, file.memorySource.isCurrent,
              file.mediaType == "video/mp4", !expectedVersion.isEmpty, expectedVersion.utf8.count <= 256,
              versionID == expectedVersion,
              transferLimitBytes > 0, maximumByteCount > 0, file.sizeBytes > 0, file.sizeBytes <= limit,
              contentSHA256.utf8.count == 64,
              contentSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              inspection.format == "jarvis.mp4_inspection.v1",
              inspection.contentSHA256 == contentSHA256, inspection.sizeBytes == file.sizeBytes,
              inspection.durationMilliseconds > 0,
              inspection.durationMilliseconds <= durationLimit,
              (1...7680).contains(inspection.width), (1...7680).contains(inspection.height),
              inspection.videoCodec == "h264", inspection.audioCodec == nil || inspection.audioCodec == "aac" else {
            throw mismatch()
        }
        return limit
    }

    /// A full exact response is required before any temporary bytes become usable.
    /// The bytes themselves still need the independent local-file hash check.
    func validateResponse(_ response: HTTPURLResponse, requestedURL: URL) throws {
        guard response.statusCode == 200 else {
            throw JarvisAPIError(code: "media_transfer_failed", message: "The saved video could not be downloaded.",
                                 statusCode: response.statusCode)
        }
        let length = response.value(forHTTPHeaderField: "Content-Length") ?? ""
        let type = response.value(forHTTPHeaderField: "Content-Type")?
            .components(separatedBy: ";")[0].trimmingCharacters(in: .whitespaces).lowercased()
        let cache = tokens(response.value(forHTTPHeaderField: "Cache-Control"))
        let vary = tokens(response.value(forHTTPHeaderField: "Vary"))
        let encoding = response.value(forHTTPHeaderField: "Content-Encoding")?.lowercased()
        guard response.url == requestedURL,
              response.value(forHTTPHeaderField: "Content-Range") == nil,
              encoding == nil || encoding == "identity", type == "video/mp4",
              !length.isEmpty, length.utf8.allSatisfy({ (48...57).contains($0) }), Int(length) == file.sizeBytes,
              response.value(forHTTPHeaderField: "X-Jarvis-Version-Id") == versionID,
              response.value(forHTTPHeaderField: "X-Content-SHA256") == contentSHA256,
              response.value(forHTTPHeaderField: "ETag") == "\"" + contentSHA256 + "\"",
              cache.contains("private"), cache.contains("no-store"), vary.contains("authorization") else {
            throw mismatch()
        }
    }

    private func tokens(_ value: String?) -> Set<String> {
        Set((value ?? "").split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        })
    }

    private func mismatch() -> JarvisAPIError {
        JarvisAPIError(code: "media_file_mismatch", message: "The video does not match the selected saved version. Refresh before trying again.")
    }
}

struct JarvisMediaFileResponse: Decodable { let file: JarvisMediaFile }
