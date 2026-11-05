import CoreMedia
import Foundation

/// Exact client descriptor, not proof of capture, synchronization or hearing.
/// The backend freezes every field on first finalization; never regenerate a
/// pending descriptor when retrying or reconciling an upload.
enum JarvisRecordingManifest {
    static let maximumDescriptorBytes = 8192
    enum Failure: Error, Equatable {
        case invalidScope, invalidCommand, invalidCompleted, invalidDescriptor
        case invalidReceipt, scopeMismatch, conflict, corruptEntry, unsafeFile
        case ioFailure, busy, capacityExceeded, unreservedFile
    }

    struct Binding: Codable, Equatable, Sendable {
        let operationID: UUID
        let recordingID: UUID
        let sessionID: UUID
        let generationID: UUID
        let audioSessionID: String
        let includeJarvisAudio: Bool
        let maxBytes: Int
        let maxDurationMilliseconds: Int
        let recordingMode: String?
        let includeMediaAudio: Bool?

        init(command: JarvisRecordingControl.Command) throws {
            guard command.action == .start else { throw Failure.invalidCommand }
            operationID = command.operationID; recordingID = command.recordingID
            sessionID = command.sessionID; generationID = command.generationID
            audioSessionID = command.audioSessionID; includeJarvisAudio = command.includeJarvisAudio
            maxBytes = command.maxBytes; maxDurationMilliseconds = command.maxDurationMilliseconds
            recordingMode = command.version == 2 ? command.recordingMode : nil
            includeMediaAudio = command.version == 2 ? command.includeMediaAudio : nil
            try validate()
        }

        func validate() throws {
            if let mode = recordingMode {
                guard ["ordinary", "voiceover"].contains(mode), let media = includeMediaAudio,
                      mode != "voiceover" || (!includeJarvisAudio && !media) else { throw Failure.invalidCommand }
            } else if includeMediaAudio != nil { throw Failure.invalidCommand }
            guard (1...1024 * 1024 * 1024).contains(maxBytes),
                  (1...1_800_000).contains(maxDurationMilliseconds),
                  !audioSessionID.isEmpty, audioSessionID.utf8.count <= 128,
                  !audioSessionID.contains("\0") else { throw Failure.invalidCommand }
        }
    }

    struct Gap: Codable, Equatable, Sendable {
        let track: String
        let startUS: Int64
        let endUS: Int64
        let reason: String
        let inferredVideoDuration: Bool
        enum CodingKeys: String, CodingKey {
            case track, reason
            case startUS = "start_us", endUS = "end_us"
            case inferredVideoDuration = "inferred_video_duration"
        }
    }

    struct Report: Codable, Equatable, Sendable {
        let format: String
        let generationID: UUID
        let timelineEndMS: Int64
        let videoFrames: Int
        let sourceAudioFrames: Int64
        let paddedAudioFrames: Int64
        var missingIntervals: [Gap]
        var omittedIntervalDetails: Int
        var recordingMode: String? = nil
        var includeMediaAudio: Bool? = nil
        var mediaCoverage: [String: UInt64]? = nil
        enum CodingKeys: String, CodingKey {
            case format
            case generationID = "generation_id", timelineEndMS = "timeline_end_ms"
            case videoFrames = "video_frames", sourceAudioFrames = "source_audio_frames"
            case paddedAudioFrames = "padded_audio_frames", missingIntervals = "missing_intervals"
            case omittedIntervalDetails = "omitted_interval_details"
            case recordingMode = "recording_mode", includeMediaAudio = "include_media_audio"
            case mediaCoverage = "media_coverage"
        }
    }

    struct Upload: Codable, Equatable, Sendable {
        let filename: String
        let description: String
        let sizeBytes: Int
        let contentSHA256: String
        var clientReport: Report
        enum CodingKeys: String, CodingKey {
            case filename, description
            case sizeBytes = "size_bytes", contentSHA256 = "content_sha256", clientReport = "client_report"
        }
    }

    /// The server's saved receipt contains no file hash. This is confirmed
    /// backend linkage only, never permission to delete the retained original.
    struct SavedLinkage: Codable, Equatable, Sendable {
        struct ReceiptScope: Codable, Equatable, Sendable {
            let recordingID: UUID
            let ownerID: String
            let sessionID: UUID
            let areaID: UUID
            let includeJarvisAudio: Bool
            let transferPolicy: String
            let maxBytes: Int
            let maxDurationMS: Int
            let admissionEventID: UUID
            let admissionPayloadID: UUID
            enum CodingKeys: String, CodingKey {
                case recordingID = "recording_id", ownerID = "owner_id", sessionID = "session_id"
                case areaID = "area_id", includeJarvisAudio = "include_jarvis_audio"
                case transferPolicy = "transfer_policy", maxBytes = "max_bytes", maxDurationMS = "max_duration_ms"
                case admissionEventID = "admission_event_id", admissionPayloadID = "admission_payload_id"
            }
        }
        let scope: ReceiptScope
        let operationID: UUID
        let outcome: String
        let receiptEventID: UUID
        let receiptPayloadID: UUID
        let replayed: Bool
        let captureAllowed: Bool
        let publicationID: UUID
        let fileID: UUID
        let sourceID: UUID
        let versionID: String
        enum CodingKeys: String, CodingKey {
            case scope, outcome, replayed
            case operationID = "operation_id", receiptEventID = "receipt_event_id"
            case receiptPayloadID = "receipt_payload_id", captureAllowed = "capture_allowed"
            case publicationID = "publication_id", fileID = "file_id", sourceID = "source_id", versionID = "version_id"
        }

        func validate(binding: Binding, ownerID: String) throws {
            guard outcome == "saved", !captureAllowed, operationID == binding.recordingID,
                  scope.recordingID == binding.recordingID, scope.ownerID == ownerID,
                  scope.sessionID == binding.sessionID,
                  scope.includeJarvisAudio == binding.includeJarvisAudio,
                  scope.transferPolicy == "mp4_capture_v1", scope.maxBytes == binding.maxBytes,
                  scope.maxDurationMS == binding.maxDurationMilliseconds,
                  !versionID.isEmpty, versionID.utf8.count <= 1024, !versionID.contains("\0") else {
                throw Failure.invalidReceipt
            }
        }

        func identifiesSamePublication(as other: Self) -> Bool {
            // A reconciliation can change replayed, not the authoritative linkage.
            scope == other.scope && operationID == other.operationID && outcome == other.outcome &&
                receiptEventID == other.receiptEventID && receiptPayloadID == other.receiptPayloadID &&
                captureAllowed == other.captureAllowed && publicationID == other.publicationID &&
                fileID == other.fileID && sourceID == other.sourceID && versionID == other.versionID
        }
    }

    static func make(binding: Binding, completed: JarvisRecordingWriter.Completed,
                     expectedWriterGenerationID: UUID, mediaCoverage: [String: UInt64]? = nil) throws -> String {
        try binding.validate()
        let evidence = completed.evidence
        let end = completed.requestedEnd
        guard validTime(end), CMTimeCompare(end, .zero) > 0,
              CMTimeCompare(end, CMTime(value: Int64(binding.maxDurationMilliseconds), timescale: 1000)) <= 0,
              completed.generationID == expectedWriterGenerationID,
              completed.generationID == completed.file.generationID,
              (1...binding.maxBytes).contains(completed.file.byteCount), validHash(completed.file.contentSHA256),
              (1...100_000).contains(evidence.video.appended),
              (0...JarvisRecordingWriter.maximumAudioTimelineFrames).contains(evidence.sourceAudioFrames),
              (0...JarvisRecordingWriter.maximumAudioTimelineFrames).contains(evidence.paddedAudioFrames),
              (0...1_000_000).contains(evidence.truncatedIntervalDetails), evidence.intervals.count <= 1024,
              validTime(evidence.video.lastAppendedEnd), validTime(evidence.audio.lastAppendedEnd),
              CMTimeCompare(evidence.video.lastAppendedEnd, end) <= 0,
              CMTimeCompare(evidence.audio.lastAppendedEnd, end) <= 0 else { throw Failure.invalidCompleted }
        let endMS = CMTimeConvertScale(end, timescale: 1000, method: .roundTowardPositiveInfinity).value
        var gaps: [Gap] = []
        var omitted = evidence.truncatedIntervalDetails
        for interval in evidence.intervals {
            // Out-of-window or nonnumeric rejected-input details cannot be
            // represented by this server schema. Count them, never clip them
            // into an invented interval within the completed media.
            guard gaps.count < 128, validTime(interval.start), validTime(interval.end),
                  CMTimeCompare(interval.end, interval.start) >= 0,
                  CMTimeCompare(interval.end, end) <= 0 else { omitted += 1; continue }
            gaps.append(Gap(track: interval.track == .video ? "video" : "audio",
                startUS: CMTimeConvertScale(interval.start, timescale: 1_000_000, method: .roundTowardNegativeInfinity).value,
                endUS: CMTimeConvertScale(interval.end, timescale: 1_000_000, method: .roundTowardPositiveInfinity).value,
                reason: interval.reason.rawValue, inferredVideoDuration: interval.inferredVideoDuration))
        }
        let selection = binding.includeJarvisAudio ? "includes rendered Jarvis audio" : "excludes the rendered Jarvis track"
        var upload = Upload(filename: "recording-\(binding.recordingID.uuidString.lowercased()).mp4",
            description: "Client-reported recording; capture configuration \(selection). Sensor origin, hearing and physical audio/video synchronization are not independently verified. Omitted gap details include truncated or out-of-window intervals.",
            sizeBytes: completed.file.byteCount, contentSHA256: completed.file.contentSHA256,
            clientReport: Report(format: "jarvis.recording_client_report.v1", generationID: binding.generationID,
                timelineEndMS: endMS, videoFrames: evidence.video.appended,
                sourceAudioFrames: evidence.sourceAudioFrames, paddedAudioFrames: evidence.paddedAudioFrames,
                missingIntervals: gaps, omittedIntervalDetails: omitted))
        upload.clientReport.recordingMode = binding.recordingMode
        upload.clientReport.includeMediaAudio = binding.includeMediaAudio
        upload.clientReport.mediaCoverage = mediaCoverage
        while true {
            try validate(upload, binding: binding)
            let data = try encode(upload)
            if data.count <= maximumDescriptorBytes {
                guard data.allSatisfy({ $0 < 128 }), let text = String(data: data, encoding: .ascii) else {
                    throw Failure.invalidDescriptor
                }
                return text
            }
            guard !upload.clientReport.missingIntervals.isEmpty else { throw Failure.invalidDescriptor }
            upload.clientReport.missingIntervals.removeLast()
            upload.clientReport.omittedIntervalDetails += 1
        }
    }

    static func decode(_ descriptor: String, binding: Binding) throws -> Upload {
        let data = Data(descriptor.utf8)
        guard !data.isEmpty, data.count <= maximumDescriptorBytes, data.allSatisfy({ $0 < 128 }) else {
            throw Failure.invalidDescriptor
        }
        let upload = try JSONDecoder().decode(Upload.self, from: data)
        try validate(upload, binding: binding)
        // Only our exact canonical descriptor is journaled. Unknown fields and
        // damaged/rewritten encodings must not silently become retry metadata.
        guard try encode(upload) == data else { throw Failure.invalidDescriptor }
        return upload
    }

    static func savedReceipt(_ data: Data, binding: Binding, ownerID: String) throws -> SavedLinkage {
        guard !data.isEmpty, data.count <= 16_384 else { throw Failure.invalidReceipt }
        let receipt = try JSONDecoder().decode(SavedLinkage.self, from: data)
        try receipt.validate(binding: binding, ownerID: ownerID)
        return receipt
    }

    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    static func validHash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func validTime(_ time: CMTime) -> Bool {
        time.isNumeric && time.epoch == 0 && CMTimeCompare(time, .zero) >= 0
    }

    private static func validate(_ upload: Upload, binding: Binding) throws {
        let report = upload.clientReport
        guard report.recordingMode == binding.recordingMode,
              report.includeMediaAudio == binding.includeMediaAudio else { throw Failure.invalidDescriptor }
        if let coverage = report.mediaCoverage {
            let keys: Set<String> = ["consumed_output_frames", "resolved_source_frames", "missing_source_frames",
                "unknown_output_frames", "receipt_dropped_output_frames", "zero_gain_source_frames",
                "admitted_source_frames", "rejected_source_frames", "recorded_source_frames", "output_capture_dropped_frames"]
            guard binding.includeMediaAudio == true, Set(coverage.keys) == keys,
                  coverage.values.allSatisfy({ $0 <= 9_007_199_254_740_991 }) else { throw Failure.invalidDescriptor }
        }
        guard upload.filename == "recording-\(binding.recordingID.uuidString.lowercased()).mp4",
              !upload.description.isEmpty, upload.description.count <= 2000, !upload.description.contains("\0"),
              (1...binding.maxBytes).contains(upload.sizeBytes), validHash(upload.contentSHA256),
              report.format == "jarvis.recording_client_report.v1", report.generationID == binding.generationID,
              (1...Int64(binding.maxDurationMilliseconds)).contains(report.timelineEndMS),
              (1...100_000).contains(report.videoFrames),
              (0...JarvisRecordingWriter.maximumAudioTimelineFrames).contains(report.sourceAudioFrames),
              (0...JarvisRecordingWriter.maximumAudioTimelineFrames).contains(report.paddedAudioFrames),
              report.sourceAudioFrames + report.paddedAudioFrames <= report.timelineEndMS * 48,
              report.missingIntervals.count <= 128, (0...1_000_000).contains(report.omittedIntervalDetails) else {
            throw Failure.invalidDescriptor
        }
        for gap in report.missingIntervals {
            guard ["audio", "video"].contains(gap.track), gap.startUS >= 0, gap.endUS >= gap.startUS,
                  gap.endUS <= report.timelineEndMS * 1000, !gap.reason.isEmpty,
                  gap.reason.count <= 64, !gap.reason.contains("\0") else { throw Failure.invalidDescriptor }
        }
    }
}
