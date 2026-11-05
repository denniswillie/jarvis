import Foundation

protocol JarvisFileAccess {
    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion
}

protocol JarvisThumbnailAccess {
    func getThumbnail(file: JarvisRetainedFile, reference: JarvisResourceReference,
                      access: JarvisWorkspaceLocalAccess) async throws -> Data
}

struct JarvisThumbnailReference: Decodable, Equatable {
    let path: String
    let versionID: String
    let contentType: String
    let contentSHA256: String
    let sizeBytes: Int
    let width: Int
    let height: Int

    enum CodingKeys: String, CodingKey {
        case path, width, height
        case versionID = "version_id", contentType = "content_type"
        case contentSHA256 = "content_sha256", sizeBytes = "size_bytes"
    }

    func validate(reference: JarvisResourceReference) throws {
        guard path == "/files/" + reference.fileID.uuidString.lowercased() + "/thumbnail",
              versionID == reference.versionID, contentType == "image/jpeg",
              (1...65536).contains(sizeBytes), (1...480).contains(width), (1...480).contains(height),
              contentSHA256.utf8.count == 64,
              contentSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw JarvisAPIError(code: "thumbnail_unavailable", message: "This preview is unavailable. You can still open the original.")
        }
    }
}

struct JarvisRetainedFile: Decodable, Identifiable, Equatable {
    let id: String
    let filename: String
    let contentType: String
    let sizeBytes: Int
    let description: String
    let source: String
    let createdAt: String
    let memorySource: JarvisSourceEvidence
    var thumbnail: JarvisThumbnailReference? = nil

    enum CodingKeys: String, CodingKey {
        case id, filename, description, source, thumbnail
        case contentType = "content_type", sizeBytes = "size_bytes"
        case createdAt = "created_at", memorySource = "memory_source"
    }

    var mediaType: String {
        contentType.components(separatedBy: ";")[0].trimmingCharacters(in: .whitespaces).lowercased()
    }

    var exportFilename: String {
        let name = filename.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
            .components(separatedBy: .controlCharacters).joined(separator: "-")
        return name.isEmpty || name == "." || name == ".." ? "Jarvis-file" : name
    }

    func validateContent(_ data: Data, contentType receivedType: String?) throws {
        let received = receivedType?.components(separatedBy: ";")[0]
            .trimmingCharacters(in: .whitespaces).lowercased()
        guard UUID(uuidString: id) != nil, memorySource.isCurrent,
              sizeBytes >= 0, sizeBytes <= 8 * 1024 * 1024,
              data.count == sizeBytes, received == mediaType else {
            throw JarvisAPIError(code: "file_content_mismatch",
                message: "The download does not match the saved file. Refresh before trying again.")
        }
    }
}

struct JarvisRetainedFilePage: Decodable {
    let files: [JarvisRetainedFile]
    let pageFull: Bool
    enum CodingKeys: String, CodingKey { case files; case pageFull = "page_full" }
}

struct JarvisRetainedFileResponse: Decodable { let file: JarvisRetainedFile }

struct JarvisFileDeletion: Decodable {
    let deleted: Bool
    let storageCleanup: String
    let fullErasureVerified: Bool
    enum CodingKeys: String, CodingKey {
        case deleted
        case storageCleanup = "storage_cleanup", fullErasureVerified = "full_erasure_verified"
    }
}

struct JarvisFileDownload {
    let file: JarvisRetainedFile
    let data: Data
}

