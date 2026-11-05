import Foundation

enum JarvisMemoryLoadState: Equatable {
    case idle
    case loading
    case loaded
    case empty
    case degraded(String)
}

enum JarvisJSONValue: Codable, Equatable {
    case string(String)
    case number(Double)
    case boolean(Bool)
    case object([String: JarvisJSONValue])
    case array([JarvisJSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JarvisJSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JarvisJSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported Jarvis JSON value."
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .boolean(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var objectValue: [String: JarvisJSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }
}

struct JarvisMemoryItem: Codable, Identifiable, Equatable {
    let id: String
    let sessionId: String?
    let memoryType: String
    let status: String
    let title: String
    var content: String
    let summary: String
    let sourceExperienceEventId: String?
    let occurredAt: String?
    let validFrom: String?
    let validTo: String?
    let importance: Double?
    let confidence: Double?
    let source: String
    let isExplicit: Bool
    let supersededByMemoryId: String?
    let createdAt: String
    let updatedAt: String
    var sourceEvidence: JarvisSourceEvidence? = nil
    var contentTruncated: Bool? = nil

    enum CodingKeys: String, CodingKey {
        case id
        case sessionId = "session_id"
        case memoryType = "memory_type"
        case status
        case title
        case content
        case summary
        case sourceExperienceEventId = "source_experience_event_id"
        case occurredAt = "occurred_at"
        case validFrom = "valid_from"
        case validTo = "valid_to"
        case importance
        case confidence
        case source
        case isExplicit = "is_explicit"
        case supersededByMemoryId = "superseded_by_memory_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case sourceEvidence = "source_evidence"
        case contentTruncated = "content_truncated"
    }

    var displaySummary: String {
        let trimmed = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? content : trimmed
    }
}

struct JarvisRecentMemoryPage: Codable, Equatable {
    let items: [JarvisMemoryItem]
    let nextCursor: String?
    var view: String? = nil

    enum CodingKeys: String, CodingKey {
        case items, view
        case nextCursor = "next_cursor"
    }
}

struct JarvisMemoryEvidence: Codable, Identifiable, Equatable {
    let id: String
    let memoryItemId: String
    let evidenceType: String
    let evidenceTime: String?
    let excerpt: String
    let confidence: Double?
    let sourceAvailable: Bool
    let retainedImageAvailable: Bool?
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case memoryItemId = "memory_item_id"
        case evidenceType = "evidence_type"
        case evidenceTime = "evidence_time"
        case excerpt
        case confidence
        case sourceAvailable = "source_available"
        case retainedImageAvailable = "retained_image_available"
        case createdAt = "created_at"
    }
}

struct JarvisMemoryEvidencePage: Codable, Equatable {
    let items: [JarvisMemoryEvidence]
    let nextCursor: String?

    enum CodingKeys: String, CodingKey {
        case items
        case nextCursor = "next_cursor"
    }
}

struct JarvisMemoryLink: Codable, Identifiable, Equatable {
    let id: String
    let sourceMemoryId: String
    let targetMemoryId: String
    let relationship: String
    let confidence: Double
    let validFrom: String
    let validTo: String?

    enum CodingKeys: String, CodingKey {
        case id
        case sourceMemoryId = "source_memory_id"
        case targetMemoryId = "target_memory_id"
        case relationship
        case confidence
        case validFrom = "valid_from"
        case validTo = "valid_to"
    }
}

struct JarvisRelatedMemory: Codable, Identifiable, Equatable {
    let id: String
    let memoryType: String
    let status: String
    let title: String
    let summary: String

    enum CodingKeys: String, CodingKey {
        case id
        case memoryType = "memory_type"
        case status
        case title
        case summary
    }
}

struct JarvisMemoryDetail: Codable, Equatable {
    var memory: JarvisMemoryItem
    let evidence: [JarvisMemoryEvidence]
    let links: [JarvisMemoryLink]
    let relatedMemories: [JarvisRelatedMemory]
    let uncertainty: String?
    var contentPage: JarvisMemoryContentPage? = nil

    enum CodingKeys: String, CodingKey {
        case memory
        case evidence
        case links
        case relatedMemories = "related_memories"
        case uncertainty
        case contentPage = "content_page"
    }

    func appendingOriginalPage(_ next: JarvisMemoryDetail) throws -> JarvisMemoryDetail {
        guard let source = memory.sourceEvidence, let incoming = next.memory.sourceEvidence,
              let currentPage = contentPage, let nextPage = next.contentPage,
              memory.id == next.memory.id, source.sourceId == incoming.sourceId,
              source.revision == incoming.revision, source.generation == incoming.generation,
              source.contentDigest == incoming.contentDigest,
              currentPage.encoding == "canonical-json-ascii-v1", nextPage.encoding == currentPage.encoding,
              currentPage.offset == 0, currentPage.nextOffset == nextPage.offset,
              memory.content.count == nextPage.offset,
              nextPage.totalCharacters == currentPage.totalCharacters,
              next.memory.content.count + nextPage.offset == (nextPage.nextOffset ?? nextPage.totalCharacters),
              nextPage.nextOffset.map({ $0 < nextPage.totalCharacters }) ?? true
        else { throw SourceMemoryPageError.changed }
        var result = self
        result.memory.content += next.memory.content
        result.memory.sourceEvidence = incoming
        result.memory.contentTruncated = nextPage.nextOffset != nil
        result.contentPage = JarvisMemoryContentPage(
            encoding: nextPage.encoding, offset: 0, totalCharacters: nextPage.totalCharacters,
            nextOffset: nextPage.nextOffset, complete: nextPage.nextOffset == nil)
        return result
    }
}

enum SourceMemoryPageError: LocalizedError {
    case changed
    var errorDescription: String? { "The original source changed. Reload it before reading more." }
}

struct JarvisSourceEvidence: Codable, Equatable {
    let sourceId: String
    let revision: Int
    let generation: Int
    let contentDigest: String
    let sourceType: String
    let trust: String
    let sourceTime: String?
    let timeBasis: String
    let recordedAt: String
    let isCurrent: Bool
    let ingestionStatus: String

    enum CodingKeys: String, CodingKey {
        case sourceId = "source_id", revision, generation
        case contentDigest = "content_digest"
        case sourceType = "source_type", trust
        case sourceTime = "source_time", timeBasis = "time_basis"
        case recordedAt = "recorded_at", isCurrent = "is_current"
        case ingestionStatus = "ingestion_status"
    }
}

struct JarvisMemoryContentPage: Codable, Equatable {
    let encoding: String
    let offset: Int
    let totalCharacters: Int
    let nextOffset: Int?
    let complete: Bool

    enum CodingKeys: String, CodingKey {
        case encoding, offset, complete
        case totalCharacters = "total_characters", nextOffset = "next_offset"
    }
}

struct JarvisSessionWorkingMemory: Codable, Equatable {
    let sessionId: String
    let revision: Int64
    let state: [String: JarvisJSONValue]
    let expiresAt: String
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case revision
        case state
        case expiresAt = "expires_at"
        case updatedAt = "updated_at"
    }
}

struct JarvisOwnerWorkingMemory: Codable, Equatable {
    let revision: Int64
    let state: [String: JarvisJSONValue]
    let expiresAt: String?
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case revision
        case state
        case expiresAt = "expires_at"
        case updatedAt = "updated_at"
    }
}

struct JarvisWorkingMemoryState: Codable, Equatable {
    let session: JarvisSessionWorkingMemory?
    let owner: JarvisOwnerWorkingMemory?
}

extension Dictionary where Key == String, Value == JarvisJSONValue {
    func jarvisLabel(for key: String) -> String? {
        guard let value = self[key] else { return nil }
        if let label = value.stringValue { return label }
        return value.objectValue?["label"]?.stringValue
    }
}
