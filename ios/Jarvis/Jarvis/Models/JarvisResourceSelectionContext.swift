import Foundation

/// Context for ordinary user requests, not an action or permission to operate.
/// The server must recheck source access and clarify ambiguous spoken references.
struct JarvisResourceReference: Codable, Equatable, Identifiable {
    let fileID: UUID
    let versionID: String
    let filename: String
    let mediaType: String
    var id: UUID { fileID }
    var displayID: String { String(fileID.uuidString.lowercased().prefix(8)) }

    enum CodingKeys: String, CodingKey {
        case fileID = "file_id", versionID = "version_id", filename
        case mediaType = "media_type"
    }
}

struct JarvisResourceSelectionContext: Codable, Equatable {
    let visibleResources: [JarvisResourceReference]
    let selectedResource: JarvisResourceReference?
    let pageFull: Bool

    enum CodingKeys: String, CodingKey {
        case visibleResources = "visible_resources", selectedResource = "selected_resource"
        case pageFull = "page_full"
    }

    /// Labels accompany exact identities as display context. They never replace
    /// the user's request or grant access to a resource on the server.
    static func envelope(_ context: Self?, sessionID: UUID, audioSessionID: String,
                         sequence: Int) throws -> [String: Any] {
        guard sequence > 0, !audioSessionID.isEmpty, audioSessionID.utf8.count <= 128,
              !audioSessionID.contains("\0") else { throw WireError.invalidContext }
        var value: Any = NSNull()
        if let context {
            guard context.visibleResources.count <= 32 else { throw WireError.invalidContext }
            for ref in context.visibleResources + [context.selectedResource].compactMap({ $0 }) {
                guard !ref.versionID.isEmpty, ref.versionID.utf8.count <= 256,
                      ref.filename.utf8.count <= 1_024, !ref.mediaType.isEmpty,
                      ref.mediaType.utf8.count <= 128 else { throw WireError.invalidContext }
            }
            value = try JSONSerialization.jsonObject(with: JSONEncoder().encode(context))
        }
        let result: [String: Any] = ["type": "workspace.resource_context", "version": 1,
            "session_id": sessionID.uuidString.lowercased(), "audio_session_id": audioSessionID,
            "sequence": sequence, "context": value]
        guard try JSONSerialization.data(withJSONObject: result).count <= 65_536 else {
            throw WireError.invalidContext
        }
        return result
    }

    enum WireError: Error { case invalidContext }
}

/// Reads the canonical Files version (source document ID), never a UI-generated
/// alias or a version inferred from the filename, date, or source revision.
struct JarvisResourceFile: Decodable {
    let file: JarvisRetainedFile
    let reference: JarvisResourceReference

    private enum Keys: String, CodingKey { case versionID = "version_id", memorySource = "memory_source" }
    private enum SourceKeys: String, CodingKey { case documentID = "document_id" }

    init(from decoder: Decoder) throws {
        file = try JarvisRetainedFile(from: decoder)
        let values = try decoder.container(keyedBy: Keys.self)
        let source = try values.nestedContainer(keyedBy: SourceKeys.self, forKey: .memorySource)
        let documentID = try source.decodeIfPresent(String.self, forKey: .documentID)
        let explicitVersion = try values.decodeIfPresent(String.self, forKey: .versionID)
        guard let id = UUID(uuidString: file.id), file.memorySource.isCurrent,
              let version = explicitVersion ?? documentID, !version.isEmpty, version.utf8.count <= 256,
              documentID == nil || explicitVersion == nil || documentID == explicitVersion else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                debugDescription: "The resource has no consistent canonical Files version."))
        }
        reference = JarvisResourceReference(fileID: id, versionID: version,
            filename: file.filename, mediaType: file.mediaType)
    }
}

struct JarvisResourceFilePage: Decodable {
    let files: [JarvisResourceFile]
    let pageFull: Bool
    var nextCursor: String? = nil
    enum CodingKeys: String, CodingKey {
        case files; case pageFull = "page_full", nextCursor = "next_cursor"
    }
}

struct JarvisResourceFileResponse: Decodable { let file: JarvisResourceFile }

protocol JarvisResourceFileAccess: JarvisFileAccess {
    func getResourceFiles(query: String?) async throws -> JarvisResourceFilePage
    func getResourceFiles(query: String?, cursor: String?) async throws -> JarvisResourceFilePage
    func getResourceFile(id: UUID, versionID: String?) async throws -> JarvisResourceFile
    func getResourceContent(file: JarvisRetainedFile, reference: JarvisResourceReference) async throws -> Data
}

extension JarvisResourceFileAccess {
    func getResourceFiles(query: String?, cursor: String?) async throws -> JarvisResourceFilePage {
        guard cursor == nil else { throw JarvisResourceSelectionContext.WireError.invalidContext }
        return try await getResourceFiles(query: query)
    }
}
