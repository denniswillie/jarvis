#if DEBUG
import CryptoKit
import Foundation
import CoreFoundation
import Darwin

// BEGIN REPLAY FRAME LINEAGE: Foundation-only local evidence.
struct JarvisReplayImageEncoding: Sendable {
    let sourceWidth: Double
    let sourceHeight: Double
    let sourceScale: Double
    let sourceOrientation: Int
    let sourcePixelWidth: Int?
    let sourcePixelHeight: Int?
    let maximumDimension: Double
    let resizeScale: Double
    let outputWidth: Int
    let outputHeight: Int
    let rendererScale: Double
    let rendererOpaque: Bool
    let rendererPreferredRange: Int
    let jpegQuality: Double

    var json: [String: Any] {
        ["decoder": "UIImage(data:)", "renderer": "UIGraphicsImageRenderer/image.draw",
         "source_width_points": sourceWidth, "source_height_points": sourceHeight,
         "source_image_scale": sourceScale, "source_orientation_raw": sourceOrientation,
         "source_cg_width": sourcePixelWidth as Any? ?? NSNull(),
         "source_cg_height": sourcePixelHeight as Any? ?? NSNull(),
         "maximum_dimension": maximumDimension, "resize_scale": resizeScale,
         "output_width": outputWidth, "output_height": outputHeight,
         "renderer_scale": rendererScale, "renderer_opaque": rendererOpaque,
         "renderer_preferred_range_raw": rendererPreferredRange, "jpeg_quality": jpegQuality]
    }
}

struct JarvisReplayFrameLineage: Sendable {
    let runID: UUID
    let launchManifestSHA256: String?
    let preparedFrameOrdinal: Int
    let scheduledAt: TimeInterval
    let sourceSHA256: String
    let sourceByteCount: Int
    let encodedSHA256: String
    let encodedByteCount: Int
    let frameSequence: Int64
    let capturedAt: Date
    let encoding: JarvisReplayImageEncoding

    static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    init(runID: UUID, launchManifestSHA256: String?, preparedFrameOrdinal: Int,
         scheduledAt: TimeInterval, sourceJPEG: Data, encodedJPEG: Data,
         frameSequence: Int64, capturedAt: Date, encoding: JarvisReplayImageEncoding) {
        self.runID = runID; self.launchManifestSHA256 = launchManifestSHA256
        self.preparedFrameOrdinal = preparedFrameOrdinal; self.scheduledAt = scheduledAt
        sourceSHA256 = Self.hash(sourceJPEG); sourceByteCount = sourceJPEG.count
        encodedSHA256 = Self.hash(encodedJPEG); encodedByteCount = encodedJPEG.count
        self.frameSequence = frameSequence; self.capturedAt = capturedAt; self.encoding = encoding
    }

    var json: [String: Any] {
        ["run_id": runID.uuidString.lowercased(),
         "launch_manifest_sha256": launchManifestSHA256 as Any? ?? NSNull(),
         "prepared_frame_ordinal": preparedFrameOrdinal, "scheduled_at_seconds": scheduledAt,
         "source_jpeg_sha256": sourceSHA256, "source_jpeg_bytes": sourceByteCount,
         "encoded_jpeg_sha256": encodedSHA256, "encoded_jpeg_bytes": encodedByteCount,
         "frame_sequence": frameSequence, "captured_at_unix_seconds": capturedAt.timeIntervalSince1970,
         "encoding": encoding.json]
    }
}

struct JarvisReplayFrameUploadReceipt: Sendable {
    let lineage: JarvisReplayFrameLineage
    let uploadedSHA256: String
    let uploadedByteCount: Int
    let frameSequence: Int64
    let capturedAtWire: String?
    let clientEventID: String?
    let visualSessionID: String?
    let connectionID: String?
    let audioSessionID: String?

    init(lineage: JarvisReplayFrameLineage, uploadedJPEG: Data, frameSequence: Int64,
         capturedAtWire: String?, clientEventID: String?, visualSessionID: String?,
         connectionID: String?, audioSessionID: String?) {
        self.lineage = lineage; uploadedSHA256 = JarvisReplayFrameLineage.hash(uploadedJPEG)
        uploadedByteCount = uploadedJPEG.count; self.frameSequence = frameSequence
        // Only allow bounded identifiers, never payloads, URLs, credentials or images.
        func bounded(_ value: String?) -> String? {
            guard let value, !value.isEmpty, value.utf8.count <= 128 else { return nil }
            return value
        }
        self.capturedAtWire = bounded(capturedAtWire); self.clientEventID = bounded(clientEventID)
        self.visualSessionID = bounded(visualSessionID); self.connectionID = bounded(connectionID)
        self.audioSessionID = bounded(audioSessionID)
    }

    var matchesEncodedFrame: Bool {
        uploadedSHA256 == lineage.encodedSHA256 && uploadedByteCount == lineage.encodedByteCount
            && frameSequence == lineage.frameSequence
    }
    var streamIdentityPresent: Bool {
        clientEventID != nil && visualSessionID != nil && connectionID != nil && capturedAtWire != nil
    }
    var json: [String: Any] {
        ["stage": "socket_send_returned", "lineage": lineage.json,
         "uploaded_jpeg_sha256": uploadedSHA256, "uploaded_jpeg_bytes": uploadedByteCount,
         "frame_sequence": frameSequence, "captured_at": capturedAtWire as Any? ?? NSNull(),
         "client_event_id": clientEventID as Any? ?? NSNull(),
         "visual_session_id": visualSessionID as Any? ?? NSNull(),
         "connection_id": connectionID as Any? ?? NSNull(),
         "audio_session_id": audioSessionID as Any? ?? NSNull(),
         "matches_encoded_frame": matchesEncodedFrame, "stream_identity_present": streamIdentityPresent,
         "server_accepted": NSNull(), "provider_received": NSNull()]
    }
}

struct JarvisReplayFrameLineageLedger {
    let runID: UUID
    let launchManifestSHA256: String?
    private(set) var receipts: [JarvisReplayFrameUploadReceipt] = []
    private(set) var rejectedIdentityCount = 0
    private(set) var truncated = false
    static let maximumReceipts = 240

    mutating func append(_ receipt: JarvisReplayFrameUploadReceipt) {
        guard receipt.lineage.runID == runID,
              receipt.lineage.launchManifestSHA256 == launchManifestSHA256 else {
            rejectedIdentityCount += 1; return
        }
        guard receipts.count < Self.maximumReceipts else { truncated = true; return }
        receipts.append(receipt)
    }

    var json: [String: Any] {
        ["schema_version": 1, "run_id": runID.uuidString.lowercased(),
         "launch_manifest_sha256": launchManifestSHA256 as Any? ?? NSNull(),
         "manifest_bound": launchManifestSHA256 != nil,
         "maximum_receipts": Self.maximumReceipts, "receipts": receipts.map(\.json),
         "truncated": truncated, "rejected_identity_count": rejectedIdentityCount,
         "mismatch_count": receipts.filter { !$0.matchesEncodedFrame }.count,
         "missing_stream_identity_count": receipts.filter { !$0.streamIdentityPresent }.count,
         "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
         "app_bundle_id": Bundle.main.bundleIdentifier as Any? ?? NSNull(),
         "app_version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as Any? ?? NSNull(),
         "app_build": Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as Any? ?? NSNull(),
         "binary_build_sha256": NSNull(),
         "limitations": "Per-row original loaded replay JPEG -> actual UIKit encoding -> exact normal upload JPEG bytes after local socket send returned. No server acceptance, provider use, complete frame coverage, physical capture or semantic pass. Missing/dropped/coalesced/failed/in-flight writes remain unmeasured. Bind installed binary externally; bundle version is not a binary hash."]
    }
}
// END REPLAY FRAME LINEAGE

struct JarvisReplayClockAnchor {
    let monotonicBefore: TimeInterval
    let utc: Date
    let monotonicAfter: TimeInterval
    var referenceUptime: TimeInterval { (monotonicBefore + monotonicAfter) / 2 }
}

struct JarvisReplayInputBlock {
    let sourceSampleOffset: Int
    let clipSampleCount: Int
    let transportFrameCount: Int
    let callbackStart: TimeInterval
    let callbackEnd: TimeInterval
    let status: Int32
}

struct JarvisReplayUtteranceEvidence {
    let ordinal: Int
    let occurrenceID: String
    let clipID: String?
    let triggerID: String
    let trigger: JarvisReplayPlan.Trigger
    let sampleCount: Int
    let pcmSHA256: String
    let allSamplesZero: Bool
    let activityThreshold: Int
    let activityThresholdOrigin: String
    let sourceFirstActive: Int?
    let sourceLastActiveExclusive: Int?
    var status = "scheduled"
    var failureCode: String?
    var triggerArmedAt: TimeInterval?
    var triggeredAt: TimeInterval?
    var enqueueRequestedAt: TimeInterval?
    var firstCallbackStart: TimeInterval?
    var lastCallbackEnd: TimeInterval?
    var firstSuccessfulActivity: TimeInterval?
    var lastSuccessfulActivity: TimeInterval?
    var stopRequestedAt: TimeInterval?
    var attemptedSamples = 0
    var acceptedSamples = 0
    var callbacks: [JarvisReplayInputBlock] = []
    var responseTrigger: JarvisReplayResponseTriggerEvidence?
    var taskClaimSequenceAtEnqueue: Int?
    var providerInputFinalization: JarvisReplayProviderInputFinalization?

    init(plan: JarvisReplayPlan, ordinal: Int) {
        let item = plan.utterances[ordinal]
        self.ordinal = ordinal
        let run = plan.runID.uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        occurrenceID = "\(run)/utterance/\(ordinal)"
        triggerID = "\(occurrenceID)/trigger"
        clipID = item.clipID; trigger = item.trigger
        sampleCount = item.pcm16.count / 2
        pcmSHA256 = SHA256.hash(data: item.pcm16).map { String(format: "%02x", $0) }.joined()
        allSamplesZero = !item.pcm16.contains { $0 != 0 }
        activityThreshold = item.assetActivityThreshold ?? 1
        activityThresholdOrigin = item.assetActivityThreshold == nil ? "native_default" : "speech_manifest"
        sourceFirstActive = item.assetFirstActiveSample
        sourceLastActiveExclusive = item.assetLastActiveSampleExclusive
    }
}

/// Local replay-exclusive provider transaction only, never canonical input
/// authority, transport-flush proof, or a change to production VAD/commit.
struct JarvisReplayProviderInputFinalization {
    let previousOrdinal: Int
    let timeout: TimeInterval
    var reservedAt: TimeInterval?
    var cleanCount: Int?
    var validationCount: Int?
    var sessionIndex: Int?
    var connectionID: String?
    var itemID: String?
    var startedIndex: Int?
    var stoppedIndex: Int?
    var committedIndex: Int?
    var transcribedIndex: Int?
    var inputCompleteAt: TimeInterval?
    var sceneDeadline: TimeInterval?
    var deadline: TimeInterval?
    var enqueueRevalidatedAt: TimeInterval?
    var failureCode: String?
    var ready = false

    private struct Input {
        let started: Int
        var stopped: Int?
        var committed: Int?
        var transcribed: Int?
        var complete: Bool { stopped != nil && committed != nil && transcribed != nil }
    }
    private struct Lifecycle {
        var connection: String?
        var session: Int?
        var inputs: [String: Input] = [:]
        var error: String?
    }
    private static let inputEvents: Set<String> = [
        "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed", "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.failed",
    ]
    private static func lifecycle(_ events: [JarvisReplayControlEvent], count: Int) -> Lifecycle {
        var result = Lifecycle()
        guard count >= 0, count <= events.count, count <= JarvisReplayPCMState.maximumControlEvents else {
            result.error = "provider_input_evidence_limit"; return result
        }
        var seen: [String: String] = [:]
        var lastReceived: TimeInterval = -1
        var lastHandled: TimeInterval = -1
        for (index, event) in events.prefix(count).enumerated() {
            if event.stage == .connectionReset, result.session != nil {
                result.error = "provider_input_connection_reset"; return result
            }
            guard event.stage == .providerDataChannelHandled else { continue }
            let type = event.fields["event_type"] ?? ""
            if type == "replay.provider_input.invalid" {
                result.error = "provider_input_invalid_event"; return result
            }
            guard type == "session.created" || inputEvents.contains(type) else { continue }
            guard event.fields["boundary"] == "ordered_current_provider_channel",
                  let connection = event.fields["connection_id"], UUID(uuidString: connection) != nil,
                  let eventID = event.fields["provider_event_id"],
                  JarvisReplayResponseIdentity.safeToken(eventID, field: "provider_event_id"),
                  let received = event.receivedAt, received.isFinite, received >= 0,
                  event.handledAt.isFinite, received <= event.handledAt,
                  received >= lastReceived, event.handledAt >= lastHandled else {
                result.error = "provider_input_invalid_event"; return result
            }
            lastReceived = received; lastHandled = event.handledAt
            let signature = connection + "|" + type + "|" + (event.fields["item_id"] ?? "")
            if let existing = seen[eventID] {
                if existing != signature { result.error = "provider_input_event_conflict"; return result }
                continue
            }
            seen[eventID] = signature
            if type == "session.created" {
                guard result.session == nil else { result.error = "provider_input_session_replaced"; return result }
                result.connection = connection; result.session = index
                continue
            }
            guard result.connection == connection, result.session != nil,
                  let item = event.fields["item_id"],
                  JarvisReplayResponseIdentity.safeToken(item, field: "item_id") else {
                result.error = "provider_input_identity_missing_or_changed"; return result
            }
            if type == "input_audio_buffer.speech_started" {
                guard result.inputs[item] == nil, result.inputs.count < 20 else {
                    result.error = "provider_input_item_reused_or_limit"; return result
                }
                result.inputs[item] = Input(started: index)
                continue
            }
            guard var input = result.inputs[item] else {
                result.error = "provider_input_unknown_item"; return result
            }
            switch type {
            case "input_audio_buffer.speech_stopped":
                guard input.stopped == nil, input.committed == nil else {
                    result.error = "provider_input_lifecycle_conflict"; return result
                }
                input.stopped = index
            case "input_audio_buffer.committed":
                guard input.stopped != nil, input.committed == nil else {
                    result.error = "provider_input_lifecycle_conflict"; return result
                }
                input.committed = index
            case "conversation.item.input_audio_transcription.completed":
                guard input.committed != nil, input.transcribed == nil else {
                    result.error = "provider_input_lifecycle_conflict"; return result
                }
                input.transcribed = index
            default:
                result.error = "provider_input_transcription_failed"; return result
            }
            result.inputs[item] = input
        }
        return result
    }

    static func reserve(previous: Int, timeout: TimeInterval, evidence: JarvisReplayAudioEvidence,
                        at time: TimeInterval, sceneDeadline: TimeInterval?) -> Self {
        var value = Self(previousOrdinal: previous, timeout: timeout)
        value.reservedAt = time; value.cleanCount = evidence.controlEvents.count
        value.sceneDeadline = sceneDeadline
        let prefix = lifecycle(evidence.controlEvents, count: evidence.controlEvents.count)
        value.connectionID = prefix.connection; value.sessionIndex = prefix.session
        if let error = prefix.error { value.failureCode = error }
        else if !time.isFinite || sceneDeadline.map({ !$0.isFinite || $0 <= time }) ?? true
                    || prefix.session == nil || !prefix.inputs.values.allSatisfy(\.complete)
                    || !evidence.controlEvents.allSatisfy({ $0.handledAt <= time })
                    || !evidence.utterances.prefix(previous).allSatisfy({
                        $0.status == "sdk_input_complete" && $0.acceptedSamples == $0.sampleCount
                    }) {
            value.failureCode = "provider_input_clean_boundary_missing"
        }
        return value
    }

    func evaluated(_ evidence: JarvisReplayAudioEvidence) -> Self {
        var value = self
        value.ready = false
        guard value.failureCode == nil, enqueueRevalidatedAt == nil else { return value }
        guard !evidence.controlEventsTruncated, !evidence.controlEventTimingInvalid,
              !evidence.inputMappingTruncated, evidence.callbackFailure == nil,
              evidence.utterances.indices.contains(previousOrdinal),
              let reservedAt, let cleanCount, let sceneDeadline else {
            value.failureCode = "provider_input_evidence_unavailable"; return value
        }
        let input = evidence.utterances[previousOrdinal]
        if input.status == "sdk_input_complete", input.acceptedSamples == input.sampleCount,
           let end = input.lastCallbackEnd, end.isFinite {
            value.inputCompleteAt = end
            value.deadline = min(sceneDeadline, end + timeout)
        } else if ["failed", "cancelled", "not_started"].contains(input.status) {
            value.failureCode = "provider_input_sdk_incomplete"; return value
        }
        let prefix = Self.lifecycle(evidence.controlEvents, count: cleanCount)
        let current = Self.lifecycle(evidence.controlEvents, count: evidence.controlEvents.count)
        value.validationCount = evidence.controlEvents.count
        if let error = prefix.error ?? current.error { value.failureCode = error; return value }
        guard prefix.session == sessionIndex, prefix.connection == connectionID,
              current.session == sessionIndex, current.connection == connectionID,
              prefix.inputs.values.allSatisfy(\.complete) else {
            value.failureCode = "provider_input_clean_boundary_changed"; return value
        }
        let candidates = current.inputs.filter { prefix.inputs[$0.key] == nil }
        guard candidates.count <= 1 else {
            value.failureCode = "provider_input_ambiguous_items"; return value
        }
        guard let (item, lifecycle) = candidates.first else { return value }
        value.itemID = item; value.startedIndex = lifecycle.started
        value.stoppedIndex = lifecycle.stopped; value.committedIndex = lifecycle.committed
        value.transcribedIndex = lifecycle.transcribed
        guard lifecycle.started >= cleanCount, let firstSDK = input.firstCallbackStart,
              let received = evidence.controlEvents[lifecycle.started].receivedAt,
              firstSDK >= reservedAt, received >= firstSDK, received >= reservedAt,
              input.acceptedSamples > 0 else {
            value.failureCode = "provider_input_before_sdk_production"; return value
        }
        // VAD may finalize during trailing PCM/silence. Both conditions are
        // mandatory, but provider finalization need not follow SDK completion.
        value.ready = lifecycle.complete && value.inputCompleteAt != nil
        return value
    }

    func json(fallbackFailure: String?) -> [String: Any] {
        func nullable<T>(_ value: T?) -> Any { value.map { $0 as Any } ?? NSNull() }
        return [
            "schema_version": 1, "after_input_ordinal": previousOrdinal,
            "correlation_basis": "replay_exclusive_provider_transaction", "canonical_input_binding": "unknown",
            "status": enqueueRevalidatedAt == nil ? "not_reached" : "validated_provider_transaction",
            "failure_code": nullable(enqueueRevalidatedAt == nil
                ? (failureCode ?? fallbackFailure ?? "provider_input_not_reached") : nil),
            "connection_id": nullable(connectionID), "provider_item_id": nullable(itemID),
            "transaction_reserved_uptime_seconds": nullable(reservedAt),
            "clean_control_event_count": nullable(cleanCount), "validation_control_event_count": nullable(validationCount),
            "session_created_control_event_index": nullable(sessionIndex),
            "speech_started_control_event_index": nullable(startedIndex),
            "speech_stopped_control_event_index": nullable(stoppedIndex),
            "committed_control_event_index": nullable(committedIndex),
            "transcription_completed_control_event_index": nullable(transcribedIndex),
            "input_complete_uptime_seconds": nullable(inputCompleteAt),
            "scene_deadline_uptime_seconds": nullable(sceneDeadline), "deadline_uptime_seconds": nullable(deadline),
            "enqueue_revalidated_uptime_seconds": nullable(enqueueRevalidatedAt),
        ]
    }
}

struct JarvisReplayDecodedBlock {
    let sampleOffset: Int
    let sampleCount: Int
    let callbackStart: TimeInterval
    let callbackEnd: TimeInterval?
    let outputEpoch: Int
    var responseIdentity: JarvisReplayResponseIdentity? = nil
    var responseAssociation: JarvisReplayResponseAssociation? = nil
}

struct JarvisReplayDecodedSample: Equatable {
    let sampleOffset: Int
    let nominalUptime: TimeInterval
    let outputEpoch: Int
}

/// Bounded categories only; never retain the provider item or arbitrary strings.
struct JarvisReplayPlaybackItemDiagnostics {
    let typeCategory: String
    let phaseCategory: String
    let policyAllowsPlayback: Bool

    init(item: [String: Any], policyAllowsPlayback: Bool) {
        typeCategory = Self.category(item["type"], known: ["message", "function_call"])
        phaseCategory = Self.category(item["phase"], known: ["final_answer", "commentary", "analysis"])
        self.policyAllowsPlayback = policyAllowsPlayback
    }

    private static func category(_ value: Any?, known: Set<String>) -> String {
        guard let value else { return "missing" }
        if value is NSNull { return "null" }
        guard let text = value as? String else { return "invalid_type" }
        return known.contains(text) ? text : "other_string"
    }
}

/// Observation of the existing predicate, never an input to playback admission.
struct JarvisReplayPlaybackGateDiagnostics {
    enum Origin: String {
        case speakerPredicate = "speaker_predicate"
        case forcedClose = "forced_close"
    }
    let origin: Origin
    let remoteDescriptionPresent: Bool
    let permittedIdentityPresent: Bool
    let playableIdentityPresent: Bool
    let playableIdentityMatchesPermit: Bool
    let externalPlaybackInactive: Bool
    let routeOutputHeld: Bool
    let receiverPresent: Bool
    let replayItemIdentityPresent: Bool
    let itemMatchesPermittedResponse: Bool
    let item: JarvisReplayPlaybackItemDiagnostics?

    var predicateAllowsPlayback: Bool {
        remoteDescriptionPresent && permittedIdentityPresent && playableIdentityMatchesPermit
            && externalPlaybackInactive && !routeOutputHeld
    }

    var json: [String: Any] {
        ["schema_version": 1, "decision_origin": origin.rawValue,
         "remote_description_present": remoteDescriptionPresent,
         "permitted_identity_present": permittedIdentityPresent,
         "playable_identity_present": playableIdentityPresent,
         "playable_identity_matches_permit": playableIdentityMatchesPermit,
         "external_playback_inactive": externalPlaybackInactive,
         "route_output_held": routeOutputHeld, "receiver_present": receiverPresent,
         "replay_item_identity_present": replayItemIdentityPresent,
         "item_matches_permitted_response": itemMatchesPermittedResponse,
         "predicate_allows_playback": predicateAllowsPlayback,
         "item_type_category": item?.typeCategory ?? "not_observed",
         "item_phase_category": item?.phaseCategory ?? "not_observed",
         "item_policy_allows_playback": item.map { $0.policyAllowsPlayback } as Any? ?? NSNull()]
    }
}

struct JarvisReplayOutputGateEvidence {
    let eventID: String
    let enabled: Bool
    let changed: Bool
    let requestedAt: TimeInterval?
    let outputEpoch: Int
    let lastAdmittedActiveSample: JarvisReplayDecodedSample?
    var responseIdentity: JarvisReplayResponseIdentity? = nil
    var responseAssociation: JarvisReplayResponseAssociation? = nil
    var localInterruptionEventID: String? = nil
    var playbackDiagnostics: JarvisReplayPlaybackGateDiagnostics? = nil
}

struct JarvisReplayResponseTriggerEvidence {
    let selection: JarvisReplayResponseSelection
    var enqueueRevalidatedAt: TimeInterval?
    var failureCode: String?
}
enum JarvisReplayLocalInterruptionCause: String {
    case clientInterrupt = "client_interrupt_method"
    case routeFenceClose = "route_output_fence_close"
}
struct JarvisReplayLocalInterruption {
    let eventID: String
    let identity: JarvisReplayResponseIdentity
    let cause: JarvisReplayLocalInterruptionCause
    let requestedAt: TimeInterval
}

/// Preserve only actual integer generations. These are distinct namespaces,
/// never aliases for a request ID, a media epoch, or one another.
enum JarvisReplayControlGenerations {
    static func safe(_ fields: [String: Any]) -> [String: Int] {
        var result: [String: Int] = [:]
        for key in ["connection_generation", "turn_generation", "route_generation"] {
            guard let number = fields[key] as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite, number.doubleValue >= 0,
                  number.doubleValue <= 9_007_199_254_740_991,
                  number.doubleValue.rounded(.towardZero) == number.doubleValue else { continue }
            result[key] = number.intValue
        }
        return result
    }
}

struct JarvisReplayManualSessionIdentity: Equatable {
    let ownerID: String
    let sessionID: UUID
    let audioSessionID: UUID
    let controlConnectionID: UUID
    let segmentID: UUID

    init(ownerID: String, sessionID: String, audioSessionID: String,
         controlConnectionID: UUID, segmentID: String) throws {
        guard RealtimeDiagnostics.safeFields(["reason": ownerID])["reason"] as? String == ownerID,
              let session = UUID(uuidString: sessionID), let audio = UUID(uuidString: audioSessionID),
              let segment = UUID(uuidString: segmentID) else { throw JarvisReplayError.manualIdentityMismatch }
        self.ownerID = ownerID; self.sessionID = session; self.audioSessionID = audio
        self.controlConnectionID = controlConnectionID; self.segmentID = segment
    }
}

/// Created from the SAME payload that sendEvent serializes, not a second UUID.
/// A returned socket write is not a provider ACK or canonical persistence.
struct JarvisReplayManualWireObservation {
    enum Stage: String {
        case started = "send_started", written = "socket_write_returned"
        case failed = "send_failed", retired = "socket_write_returned_retired"
    }
    let runID: UUID
    let connectionID: UUID
    let segmentID: UUID
    let clientEventID: UUID
    let eventType: String
    let generations: [String: Int]
    var stage: Stage = .started
    var at: TimeInterval

    init?(runID: UUID, connectionID: UUID, payload: [String: Any],
          context: [String: Any], at: TimeInterval) {
        guard let type = payload["type"] as? String,
              ["audio.input.begin", "audio.input.commit", "audio.input.abort"].contains(type),
              let segment = payload["segment_id"] as? String, let segmentID = UUID(uuidString: segment),
              let event = payload["client_event_id"] as? String, let eventID = UUID(uuidString: event),
              at.isFinite, at >= 0 else { return nil }
        self.runID = runID; self.connectionID = connectionID; self.segmentID = segmentID
        clientEventID = eventID; eventType = type; self.at = at
        generations = JarvisReplayControlGenerations.safe(context)
    }
    var json: [String: Any] {
        ["event_type": eventType, "client_event_id": clientEventID.uuidString.lowercased(),
         "segment_id": segmentID.uuidString.lowercased(),
         "control_connection_id": connectionID.uuidString.lowercased(),
         "stage": stage.rawValue, "observed_uptime_seconds": at, "generations": generations,
         "provider_acknowledged": NSNull(), "canonical_input_id": NSNull()]
    }
}

/// Debug-only adapter used by the actual replay source. It calls the normal
/// press/release closures and NEVER constructs a provider/canonical request.
@MainActor
final class JarvisReplayManualInputController {
    static let readyTimeout: TimeInterval = 5
    static let commitTimeout: TimeInterval = 5
    static let maximumOccurrences = 20
    static let maximumWritesPerOccurrence = 8
    private enum Phase: String { case waitingReady, waitingConsume, injecting, awaitingCommit, written, failed }
    private struct Row {
        let input: JarvisReplayUtteranceEvidence
        let identity: JarvisReplayManualSessionIdentity
        let began: TimeInterval
        let responseEnqueueDeadline: TimeInterval?
        var phase = Phase.waitingReady
        var ready: TimeInterval?
        var readyGenerations: [String: Int] = [:]
        var enqueued: TimeInterval?
        var sdkEnd: TimeInterval?
        var acceptedSamples = 0
        var released: TimeInterval?
        var commitID: UUID?
        var writes: [JarvisReplayManualWireObservation] = []
    }
    let runID: UUID
    let invocationID: UUID?
    private var rows: [Row] = []
    private var active: Int?
    private(set) var retired = false
    private(set) var failureCode: String?
    private(set) var truncated = false
    var isBusy: Bool { active != nil }
    var activeOrdinal: Int? { active.map { rows[$0].input.ordinal } }
    var readyOrdinal: Int? {
        guard let active, (rows[active].phase == .waitingReady || rows[active].phase == .waitingConsume),
              rows[active].ready != nil else { return nil }
        return rows[active].input.ordinal
    }
    var memoryReadinessEvidence: [String: Any]?
    func committed(_ ordinal: Int) -> JarvisReplayManualCommit? {
        guard !retired, !truncated, failureCode == nil,
              let row = rows.first(where: { $0.input.ordinal == ordinal }),
              row.phase == .written, row.sdkEnd != nil, let commit = row.commitID else { return nil }
        return .init(ordinal: ordinal, identity: row.identity, clientEventID: commit)
    }

    init(runID: UUID, invocationID: UUID?) {
        self.runID = runID; self.invocationID = invocationID
    }
    private func fail(_ code: String, row: Int? = nil) {
        if failureCode == nil { failureCode = code }
        if let index = row ?? active { rows[index].phase = .failed }
    }
    func begin(input: JarvisReplayUtteranceEvidence, at: TimeInterval, sourceClockOrigin: TimeInterval? = nil,
               press: () throws -> JarvisReplayManualSessionIdentity) throws {
        guard !retired, failureCode == nil, active == nil,
              rows.count < Self.maximumOccurrences, !rows.contains(where: { $0.input.ordinal == input.ordinal }),
              (0..<Self.maximumOccurrences).contains(input.ordinal), input.status == "scheduled",
              input.sampleCount > 0, at.isFinite, at >= 0 else { throw JarvisReplayError.manualInputIncomplete }
        let responseDeadline: TimeInterval?
        if case .afterManualResponse(let trigger) = input.trigger {
            guard let origin = sourceClockOrigin, origin.isFinite, origin >= 0,
                  at >= origin + trigger.at, at < origin + trigger.deadline else {
                throw JarvisReplayError.manualInputIncomplete
            }
            responseDeadline = origin + trigger.deadline
        } else { responseDeadline = nil }
        let identity = try press()
        rows.append(Row(input: input, identity: identity, began: at, responseEnqueueDeadline: responseDeadline))
        active = rows.count - 1
    }
    @discardableResult
    func ready(segmentID: String, connectionID: UUID, generations: [String: Any], at: TimeInterval) -> Bool {
        guard !retired, failureCode == nil, let index = active, rows[index].phase == .waitingReady,
              rows[index].ready == nil, UUID(uuidString: segmentID) == rows[index].identity.segmentID,
              connectionID == rows[index].identity.controlConnectionID,
              at.isFinite, at >= rows[index].began, at < rows[index].began + Self.readyTimeout,
              rows[index].responseEnqueueDeadline.map({ at < $0 }) ?? true else { return false }
        rows[index].ready = at; rows[index].readyGenerations = JarvisReplayControlGenerations.safe(generations)
        if rows[index].responseEnqueueDeadline != nil { rows[index].phase = .waitingConsume }
        return true
    }
    func observe(_ value: JarvisReplayManualWireObservation) {
        guard !retired, value.runID == runID,
              let index = rows.firstIndex(where: { $0.identity.segmentID == value.segmentID
                  && $0.identity.controlConnectionID == value.connectionID }),
              value.at.isFinite, value.at >= rows[index].began else { return }
        if rows[index].writes.contains(where: { $0.clientEventID == value.clientEventID && $0.stage == value.stage }) { return }
        guard rows[index].writes.count < Self.maximumWritesPerOccurrence else {
            truncated = true; fail("manual_wire_capacity", row: index); return
        }
        rows[index].writes.append(value)
        if value.stage == .failed || value.stage == .retired {
            fail("manual_wire_not_current_or_failed", row: index); return
        }
        if value.eventType == "audio.input.abort" {
            fail("manual_input_aborted", row: index); return
        }
        guard value.eventType == "audio.input.commit" else { return }
        guard rows[index].phase == .awaitingCommit, rows[index].sdkEnd != nil,
              rows[index].released != nil else { fail("manual_commit_before_sdk_completion", row: index); return }
        if value.stage == .started {
            guard rows[index].commitID == nil || rows[index].commitID == value.clientEventID else {
                fail("manual_commit_identity_conflict", row: index); return
            }
            rows[index].commitID = value.clientEventID
        } else if value.stage == .written {
            guard rows[index].commitID == value.clientEventID else {
                fail("manual_commit_start_unobserved", row: index); return
            }
            rows[index].phase = .written
            if active == index { active = nil }
        }
    }
    func advance(inputs: [JarvisReplayUtteranceEvidence], at: TimeInterval, sceneDeadline: TimeInterval,
                 canEnqueue: (Int) throws -> Bool = { _ in true },
                 enqueue: (Int, TimeInterval) throws -> Void,
                 release: (JarvisReplayManualSessionIdentity) throws -> Void) throws {
        guard !retired, failureCode == nil, at.isFinite, at >= 0,
              sceneDeadline.isFinite, sceneDeadline >= 0 else { throw JarvisReplayError.manualInputIncomplete }
        guard let index = active else { return }
        // The caller supplies the original source-clock origin + scene duration.
        // Deadline wins over a matching ready or an already-completed SDK input.
        // Never enqueue or release/commit pending work on a late scheduler wake.
        guard at < sceneDeadline else {
            fail("manual_scene_deadline"); throw JarvisReplayError.manualInputIncomplete
        }
        if rows[index].phase == .waitingReady || rows[index].phase == .waitingConsume,
           let deadline = rows[index].responseEnqueueDeadline, at >= deadline {
            fail("manual_calendar_enqueue_deadline"); throw JarvisReplayError.outputWaitExpired
        }
        switch rows[index].phase {
        case .waitingReady, .waitingConsume:
            if rows[index].phase == .waitingReady {
                // Ordinary manual mode retains its original watchdog. In gated
                // mode this is only the actual input-ready handshake.
                guard at < rows[index].began + Self.readyTimeout else {
                    fail("manual_ready_timeout"); throw JarvisReplayError.manualReadyExpired
                }
                guard rows[index].ready != nil else { return }
            } else {
                // Matching ready was observed within five seconds. Waiting for
                // a fresh host consume now uses the ORIGINAL source-clock gate
                // deadline, never ready-time + timeout or a renewed scene.
                guard rows[index].ready != nil, rows[index].responseEnqueueDeadline != nil else {
                    fail("manual_consume_state_invalid"); throw JarvisReplayError.manualInputIncomplete
                }
            }
            guard try canEnqueue(rows[index].input.ordinal) else { return }
            try enqueue(rows[index].input.ordinal, rows[index].began)
            rows[index].enqueued = at; rows[index].phase = .injecting
        case .injecting:
            guard let input = inputs.first(where: { $0.ordinal == rows[index].input.ordinal }),
                  input.occurrenceID == rows[index].input.occurrenceID,
                  input.pcmSHA256 == rows[index].input.pcmSHA256,
                  input.sampleCount == rows[index].input.sampleCount else {
                fail("manual_input_identity_changed"); throw JarvisReplayError.manualIdentityMismatch
            }
            if input.status == "failed" || input.failureCode != nil {
                fail("manual_sdk_input_failed"); throw JarvisReplayError.manualInputIncomplete
            }
            guard input.status == "sdk_input_complete" else { return }
            var offset = 0
            var previousEnd = rows[index].enqueued ?? rows[index].began
            for block in input.callbacks {
                guard block.status == 0, block.sourceSampleOffset == offset, block.clipSampleCount > 0,
                      block.callbackStart.isFinite, block.callbackEnd.isFinite,
                      block.callbackStart >= previousEnd, block.callbackEnd >= block.callbackStart else {
                    fail("manual_sdk_receipt_invalid"); throw JarvisReplayError.manualInputIncomplete
                }
                offset += block.clipSampleCount; previousEnd = block.callbackEnd
            }
            guard !input.callbacks.isEmpty, offset == input.sampleCount,
                  input.acceptedSamples == input.sampleCount, input.attemptedSamples == input.sampleCount,
                  let ended = input.lastCallbackEnd, ended == previousEnd, ended <= at else {
                fail("manual_sdk_receipt_incomplete"); throw JarvisReplayError.manualInputIncomplete
            }
            rows[index].sdkEnd = ended; rows[index].acceptedSamples = input.acceptedSamples
            rows[index].released = at; rows[index].phase = .awaitingCommit
            try release(rows[index].identity)
        case .awaitingCommit:
            if let released = rows[index].released, at >= released + Self.commitTimeout {
                fail("manual_commit_write_timeout"); throw JarvisReplayError.manualInputIncomplete
            }
        case .written: break
        case .failed: throw JarvisReplayError.manualInputIncomplete
        }
    }
    func retire() {
        guard !retired else { return }
        if active != nil { fail("manual_retired_before_commit") }
        retired = true
    }
    var json: [String: Any] {
        ["schema_version": 1, "mode": "manual_hold_to_talk", "run_id": runID.uuidString.lowercased(),
         "replay_invocation_id": invocationID?.uuidString.lowercased() as Any? ?? NSNull(),
         "ready_timeout_seconds": Self.readyTimeout, "commit_write_timeout_seconds": Self.commitTimeout,
         "retired": retired, "truncated": truncated, "failure_code": failureCode as Any? ?? NSNull(),
         "canonical_binding_status": "unresolved", "full_chain_proven": false,
          "memory_readiness": memoryReadinessEvidence as Any? ?? NSNull(),
         "limitations": "Manual hold-to-talk only, not automatic VAD or Live. SDK completion and socket write are not provider ACK, canonical finalization, semantic correctness or hearing.",
         "occurrences": rows.map { row -> [String: Any] in
             ["ordinal": row.input.ordinal, "occurrence_id": row.input.occurrenceID,
              "clip_id": row.input.clipID as Any? ?? NSNull(), "pcm16_sha256": row.input.pcmSHA256,
              "sample_count": row.input.sampleCount, "sdk_accepted_clip_samples": row.acceptedSamples,
              "owner_id": row.identity.ownerID, "session_id": row.identity.sessionID.uuidString.lowercased(),
              "audio_session_id": row.identity.audioSessionID.uuidString.lowercased(),
              "control_connection_id": row.identity.controlConnectionID.uuidString.lowercased(),
              "segment_id": row.identity.segmentID.uuidString.lowercased(), "status": row.phase.rawValue,
              "press_uptime_seconds": row.began, "ready_uptime_seconds": row.ready as Any? ?? NSNull(),
              "ready_generations": row.readyGenerations,
               "response_enqueue_deadline_uptime_seconds": row.responseEnqueueDeadline as Any? ?? NSNull(),
              "enqueue_uptime_seconds": row.enqueued as Any? ?? NSNull(),
              "sdk_end_uptime_seconds": row.sdkEnd as Any? ?? NSNull(),
              "release_uptime_seconds": row.released as Any? ?? NSNull(),
              "commit_client_event_id": row.commitID?.uuidString.lowercased() as Any? ?? NSNull(),
              "canonical_input_id": NSNull(), "wire_observations": row.writes.map(\.json)]
         }]
    }
}

enum JarvisReplayControlStage: String {
    case clientStartupObserved = "client_startup_observed"
    case connectionReset = "connection_reset"
    case offerStarted = "offer_started"
    case offerReady = "offer_ready"
    case answerApplyStarted = "answer_apply_started"
    case answerApplied = "answer_applied"
    case mediaStarted = "media_started"
    case responsePermitted = "response_permitted"
    case backendControlHandled = "backend_control_handled"
    case providerDataChannelHandled = "provider_data_channel_handled"
    case localInterruptionRequested = "local_interruption_requested"
}

struct JarvisReplayControlEvent {
    let stage: JarvisReplayControlStage
    let handledAt: TimeInterval
    let receivedAt: TimeInterval?
    let fields: [String: String]
    var generations: [String: Int] = [:]
    var boundResponseIdentity: JarvisReplayResponseIdentity? = nil
    var jsonFields: [String: Any] {
        var result = fields.mapValues { $0 as Any }
        for (key, value) in generations { result[key] = value }
        return result
    }
}

/// Post-processing only. The caller, not this code, chooses a semantic cut or
/// annotates a useful sample. Nothing here assigns a provider response ID.
enum JarvisReplayEvidenceReport {
    static func identity(_ value: JarvisReplayResponseIdentity) -> [String: Any] {
        ["connection_id": value.connectionID.uuidString.lowercased(), "response_id": value.responseID,
         "item_id": value.itemID, "local_output_epoch": value.outputEpoch]
    }

    static func responseStopMeasurements(_ evidence: JarvisReplayAudioEvidence) -> [[String: Any]] {
        guard !evidence.truncated, !evidence.inputMappingTruncated, !evidence.decodedMappingTruncated,
              !evidence.outputGateHistoryTruncated, !evidence.responseEvidenceTruncated,
              !evidence.responseEvidenceInvalid else { return [] }
        return evidence.utterances.compactMap { utterance in
            guard let trigger = utterance.responseTrigger, trigger.enqueueRevalidatedAt != nil,
                  let onset = utterance.firstSuccessfulActivity,
                  let interruption = evidence.localInterruptions.first(where: {
                      $0.identity == trigger.selection.output.identity && $0.requestedAt >= onset
                  }), let gate = evidence.outputGates.first(where: {
                      $0.localInterruptionEventID == interruption.eventID && !$0.enabled && $0.changed
                          && $0.responseIdentity == interruption.identity
                  }), let closed = gate.requestedAt, closed >= interruption.requestedAt else { return nil }
            let tail: Any = gate.lastAdmittedActiveSample.map { $0.nominalUptime - onset } as Any? ?? NSNull()
            return ["occurrence_id": utterance.occurrenceID, "identity": identity(interruption.identity),
                "interruption_event_id": interruption.eventID, "output_gate_event_id": gate.eventID,
                "input_first_successful_activity_uptime_seconds": onset,
                "local_interruption_requested_uptime_seconds": interruption.requestedAt,
                "local_gate_closed_uptime_seconds": closed,
                "input_to_local_interruption_seconds": interruption.requestedAt - onset,
                "input_to_local_gate_close_seconds": closed - onset,
                "last_retained_active_minus_input_seconds": tail]
        }
    }
    static func locateSample(_ offset: Int, evidence: JarvisReplayAudioEvidence) -> JarvisReplayDecodedSample? {
        guard offset >= 0, let block = evidence.decodedSampleMap.first(where: {
            offset >= $0.sampleOffset && offset < $0.sampleOffset + $0.sampleCount
        }) else { return nil }
        return JarvisReplayDecodedSample(sampleOffset: offset,
            nominalUptime: block.callbackStart + Double(offset - block.sampleOffset) / 48_000,
            outputEpoch: block.outputEpoch)
    }

    static func intervalForSelectedCut(occurrenceID: String, gateEventID: String,
                                      evidence: JarvisReplayAudioEvidence) -> TimeInterval? {
        guard !evidence.truncated, !evidence.outputGateHistoryTruncated, !evidence.inputMappingTruncated,
              let occurrence = evidence.utterances.first(where: { $0.occurrenceID == occurrenceID }),
              let onset = occurrence.firstSuccessfulActivity,
              let gate = evidence.outputGates.first(where: { $0.eventID == gateEventID }), !gate.enabled, gate.changed,
              let tail = gate.lastAdmittedActiveSample, tail.outputEpoch == gate.outputEpoch else { return nil }
        // May be negative: the selected tail may have ended before speech began.
        return tail.nominalUptime - onset
    }

    static func make(evidence: JarvisReplayAudioEvidence, runID: UUID, finalUIStatus: String) -> [String: Any] {
        func nullable<T>(_ value: T?) -> Any { if let value { return value }; return NSNull() }
        func trigger(_ value: JarvisReplayPlan.Trigger) -> [String: Any] {
            switch value {
            case .atTime(let at): return ["kind": "at_time", "at_seconds": at]
            case .outputStart(let cursor, let delay, let timeout):
                return ["kind": "output_start", "after_cursor": cursor, "delay_seconds": delay, "timeout_seconds": timeout]
            case .responseOutputStart(let value):
                return ["kind": "response_output_start", "after_input_ordinal": value.afterInputOrdinal,
                    "delay_seconds": value.delay, "timeout_seconds": value.timeout,
                    "max_pcm_age_seconds": value.maxPCMAge, "response_id": nullable(value.responseID),
                    "connection_id": nullable(value.connectionID?.uuidString.lowercased()),
                    "require_task_claim": value.requireTaskClaim]
            case .afterTaskClaim(let previous, let timeout):
                return ["kind": "after_task_claim", "after_input_ordinal": previous, "timeout_seconds": timeout]
            case .afterInputComplete(let previous, let delay, let timeout):
                return ["kind": "after_input_complete", "after_input_ordinal": previous,
                    "delay_seconds": delay, "timeout_seconds": timeout]
            case .afterProviderInputFinalized(let previous, let timeout):
                return ["kind": "after_provider_input_finalized", "after_input_ordinal": previous, "timeout_seconds": timeout]
            case .afterManualResponse(let value): return value.json
            case .playbackDrained(let timeout): return ["kind": "playback_drained", "timeout_seconds": timeout]
            }
        }
        func marker(_ sample: JarvisReplayDecodedSample?) -> Any {
            guard let sample else { return NSNull() }
            return ["sample_offset": sample.sampleOffset, "nominal_uptime_seconds": sample.nominalUptime,
                    "local_output_epoch": sample.outputEpoch] as [String: Any]
        }
        func responseTrigger(_ value: JarvisReplayResponseTriggerEvidence?) -> Any {
            guard let value else { return NSNull() }
            let selection = value.selection, output = selection.output
            return ["identity": identity(output.identity), "after_input_ordinal": selection.input.ordinal,
                "input_complete_uptime_seconds": selection.input.at,
                "response_admitted_uptime_seconds": output.admittedAt,
                "first_retained_active_sample": marker(output.firstActive),
                "latest_retained_active_sample": marker(output.latestActive),
                "latest_decoded_callback_end_uptime_seconds": output.callbackEnd,
                "selected_uptime_seconds": selection.selectedAt,
                "enqueue_revalidated": value.enqueueRevalidatedAt != nil,
                "enqueue_revalidated_uptime_seconds": nullable(value.enqueueRevalidatedAt),
                "pcm_age_seconds": nullable(value.enqueueRevalidatedAt.map { $0 - output.callbackEnd }),
                "failure_code": nullable(value.failureCode), "semantic_causality": "unknown"] as [String: Any]
        }
        var anchor: Any = NSNull()
        if let value = evidence.clockAnchor {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            anchor = ["monotonic_before_seconds": value.monotonicBefore,
                      "monotonic_after_seconds": value.monotonicAfter,
                      "monotonic_reference_seconds": value.referenceUptime,
                      "utc_unix_seconds": value.utc.timeIntervalSince1970,
                      "utc_iso8601": formatter.string(from: value.utc),
                      "sampling_bracket_seconds": value.monotonicAfter - value.monotonicBefore,
                      "purpose": evidence.sourceClock == nil ? "scene_start_clock_pair_not_playback_or_network_sync"
                          : "derived_source_phase_origin_not_new_wall_clock_measurement",
                      "limitations": "Midpoint offset estimate only; no UTC accuracy, drift or wall-clock-change guarantee."] as [String: Any]
        }
        var utterances: [[String: Any]] = evidence.utterances.map { item in
            ["ordinal": item.ordinal, "occurrence_id": item.occurrenceID, "clip_id": nullable(item.clipID),
             "trigger_id": item.triggerID, "trigger": trigger(item.trigger), "status": item.status,
             "failure_code": nullable(item.failureCode), "sample_count": item.sampleCount,
             "pcm16_sha256": item.pcmSHA256, "all_samples_zero": item.allSamplesZero,
             "activity_threshold_abs_pcm16": item.activityThreshold, "activity_comparison": ">=",
             "activity_threshold_origin": item.activityThresholdOrigin,
             "source_first_active_sample": nullable(item.sourceFirstActive),
             "source_last_active_sample_exclusive": nullable(item.sourceLastActiveExclusive),
             "trigger_armed_uptime_seconds": nullable(item.triggerArmedAt),
             "triggered_uptime_seconds": nullable(item.triggeredAt),
             "enqueue_requested_uptime_seconds": nullable(item.enqueueRequestedAt),
             "response_trigger": responseTrigger(item.responseTrigger),
             "task_claim_sequence_at_enqueue": nullable(item.taskClaimSequenceAtEnqueue),
             "first_sdk_callback_start_uptime_seconds": nullable(item.firstCallbackStart),
             "last_sdk_callback_end_uptime_seconds": nullable(item.lastCallbackEnd),
             "first_successful_activity_uptime_seconds": nullable(item.firstSuccessfulActivity),
             "last_successful_activity_uptime_seconds": nullable(item.lastSuccessfulActivity),
             "stop_requested_uptime_seconds": nullable(item.stopRequestedAt),
             "attempted_clip_samples": item.attemptedSamples, "sdk_accepted_clip_samples": item.acceptedSamples,
             "callbacks": item.callbacks.map { block in
                 ["source_sample_offset": block.sourceSampleOffset, "clip_sample_count": block.clipSampleCount,
                  "transport_frame_count": block.transportFrameCount,
                  "callback_start_uptime_seconds": block.callbackStart,
                  "callback_end_uptime_seconds": block.callbackEnd, "sdk_status": block.status] as [String: Any]
             }] as [String: Any]
        }
        for (index, item) in evidence.utterances.enumerated() {
            if case .afterProviderInputFinalized(let previous, let timeout) = item.trigger {
                let proof = item.providerInputFinalization
                    ?? JarvisReplayProviderInputFinalization(previousOrdinal: previous, timeout: timeout)
                utterances[index]["provider_input_finalization"] = proof.evaluated(evidence).json(fallbackFailure: item.failureCode)
            }
        }
        return [
            "schema_version": 2, "run_id": runID.uuidString.lowercased(),
            "replay_invocation_id": nullable(evidence.invocationID?.uuidString.lowercased()),
            "replay_source_clock": nullable(evidence.sourceClock),
            "post_media_capture": evidence.outputDrain.json,
            "response_trigger_schema_version": 1,
            "task_barrier_failure_code": nullable(evidence.taskBarrierFailureCode),
            "task_barrier_failure_diagnostic": evidence.taskBarrierFailureDiagnostic?.json as Any? ?? NSNull(),
            "response_correlation_basis": "authorized_gate_interval",
            "response_semantic_causality": "unknown",
            "response_evidence_truncated": evidence.responseEvidenceTruncated,
            "response_evidence_invalid": evidence.responseEvidenceInvalid,
            "uncorrelated_decoded_blocks": evidence.uncorrelatedDecodedBlocks,
            "response_uncorrelated_decoded_blocks": evidence.responseUncorrelatedDecodedBlocks,
            "response_association_basis": "authorized_response_and_epoch_at_callback_entry_and_return_not_packet_item_or_phase_attribution",
            "local_interruptions": evidence.localInterruptions.map { event in
                ["event_id": event.eventID, "identity": identity(event.identity), "cause": event.cause.rawValue,
                 "requested_uptime_seconds": event.requestedAt] as [String: Any]
            },
            "response_stop_measurements": responseStopMeasurements(evidence),
            "media_epoch_id": evidence.mediaEpochID.uuidString.lowercased(),
            "provenance": evidence.provenance, "boundary": evidence.boundary,
            "physical_playback_measured": false, "provider_closure_confirmed": false, "useful_answer_onset": NSNull(),
            "clock_anchor": anchor, "utterances": utterances,
            "control_events": evidence.controlEvents.map { event in
                ["stage": event.stage.rawValue, "handled_monotonic_s": event.handledAt,
                 "received_monotonic_s": nullable(event.receivedAt),
                 "callback_to_handler_seconds": nullable(event.receivedAt.map { event.handledAt - $0 }),
                  "fields": event.jsonFields,
                   "bound_response_identity": event.boundResponseIdentity.map(identity) as Any? ?? NSNull()] as [String: Any]
            },
            "control_events_truncated": evidence.controlEventsTruncated,
            "manual_input": evidence.manualInput as Any? ?? NSNull(),
            "control_event_timing_invalid": evidence.controlEventTimingInvalid,
            "control_event_timing_basis": "Client monotonic receipt/handling only. Provider event IDs do not attribute individual decoded PCM samples or prove server execution, playback, or hearing. Missing event types remain unmeasured.",
            "input_threshold_abs_pcm16": 1, "decoded_output_threshold_abs_pcm16": 17,
            "legacy_input_activity_boundary": "render_preparation_before_sdk_call_not_delivery_confirmation",
            "input_blocks": evidence.inputBlocks, "decoded_blocks": evidence.decodedBlocks,
            "gated_decoded_blocks": evidence.gatedDecodedBlocks,
            "input_first_non_silent_uptime": evidence.inputFirstNonSilent,
            "input_last_non_silent_uptime": evidence.inputLastNonSilent,
            "output_starts": evidence.outputStarts.map { ["local_cursor": $0.cursor, "uptime": $0.at] as [String: Any] },
            "callback_failure": nullable(evidence.callbackFailure), "truncated": evidence.truncated,
            "input_mapping_truncated": evidence.inputMappingTruncated,
            "output_gate_history_truncated": evidence.outputGateHistoryTruncated,
            "decoded_mapping_truncated": evidence.decodedMappingTruncated,
            "decoded_pcm": ["filename": "decoded.pcm", "sample_rate_hz": 48_000, "channels": 1,
                            "encoding": "signed_pcm16_little_endian", "sample_count": evidence.decodedPCM16.count / 2,
                            "byte_count": evidence.decodedPCM16.count] as [String: Any],
            "decoded_sample_map": evidence.decodedSampleMap.map { block in
                ["sample_offset": block.sampleOffset, "sample_count": block.sampleCount,
                 "callback_start_uptime_seconds": block.callbackStart,
                 "callback_end_uptime_seconds": nullable(block.callbackEnd),
                 "local_output_epoch": block.outputEpoch,
                 "response_identity": block.responseIdentity.map(identity) as Any? ?? NSNull(),
                 "response_association": block.responseAssociation.map(\.json) as Any? ?? NSNull()] as [String: Any]
            },
            "output_gates": evidence.outputGates.map { gate in
                var value: [String: Any] = ["event_id": gate.eventID, "enabled": gate.enabled, "changed": gate.changed,
                 "requested_uptime_seconds": nullable(gate.requestedAt), "local_output_epoch": gate.outputEpoch,
                 "last_admitted_active_sample": marker(gate.lastAdmittedActiveSample),
                 "response_identity": gate.responseIdentity.map(identity) as Any? ?? NSNull(),
                 "response_association": gate.responseAssociation.map(\.json) as Any? ?? NSNull(),
                  "local_interruption_event_id": nullable(gate.localInterruptionEventID)]
                if let diagnostics = gate.playbackDiagnostics {
                    value["playback_diagnostics"] = diagnostics.json
                }
                return value
            },
            "sample_time_mapping": "Offsets are zero-based mono samples with exclusive range ends. Within a block: callback_start + relative_sample/48000. Nominal callback-relative placement, not DAC, hearing or network arrival time. Preserve inter-block gaps.",
            "cutoff_measurement": "Explicitly pair an occurrence with a local closed output epoch. Last admitted threshold-17 sample minus first SDK-accepted occurrence activity. No provider-response attribution; negative values are possible. Useful-response interval requires a separate later sample annotation.",
            "injection_boundary": "SDK callback attempt/return only; successful return is not proof of remote receipt or recognition. Clip samples exclude zero padding in the 480-frame transport block.",
            "response_correlation_limitations": "Accepted control identity at callback entry/return, not per-packet RTP attribution or proof that buffered old audio drained. Prior full SDK input completion is not a causal request-to-response join. Local explicit fence measurements are not actual semantic cuts, useful answer latency, acoustic cutoff, playback, hearing, or task cancellation.",
            "final_ui_status": finalUIStatus,
            "qualification": "native replay capture only; not correctness, useful onset or hearing acceptance"
        ]
    }
}

// Test-only readiness transport. Host receipts are authority over canonical
// sources, not a claim of model use, decoded drain, useful speech or hearing.
struct JarvisReplayManualResponseTrigger {
    let afterInputOrdinal: Int
    let at: TimeInterval
    let timeout: TimeInterval
    let canonical: String
    let playback: String
    var deadline: TimeInterval { at + timeout }
    func validate(ordinal: Int, duration: TimeInterval, pcmSeconds: TimeInterval) throws {
        guard afterInputOrdinal >= 0, afterInputOrdinal < ordinal,
              at.isFinite, at >= 0, timeout.isFinite, timeout > 0,
              timeout <= JarvisReplayPlan.maximumTriggerTimeout,
              deadline > at, deadline + pcmSeconds <= duration,
              ["finalized_input", "retain_searchable"].contains(canonical),
              ["none", "authenticated_buffer_stop"].contains(playback) else {
            throw JarvisReplayError.invalidPlan
        }
    }
    var json: [String: Any] {
        ["kind": "after_manual_response", "after_input_ordinal": afterInputOrdinal,
         "at_seconds": at, "timeout_seconds": timeout,
         "canonical_requirement": canonical, "playback_requirement": playback]
    }
}

struct JarvisReplayMemoryConfiguration {
    let scope: [String: String]
    let manifestSHA256: String
    let controlsURL: URL

    static func load(scopeData: Data, manifestSHA256: String, inputsRoot: URL,
                     runID: UUID, ownerID: String, areaID: String, bankID: String) throws -> Self {
        let value = try JarvisReplayTaskJSON.object(scopeData, limit: 8192)
        try JarvisReplayTaskJSON.keys(value, ["run_id", "owner_id", "database_schema", "area_id", "memory_bank_id"])
        var scope: [String: String] = [:]
        for (key, item) in value { scope[key] = try JarvisReplayTaskJSON.string(item, max: 200) }
        guard scope["run_id"] == runID.uuidString.replacingOccurrences(of: "-", with: "").lowercased(),
              scope["owner_id"] == ownerID, scope["area_id"] == areaID, UUID(uuidString: areaID) != nil,
              scope["memory_bank_id"] == bankID,
              let schema = scope["database_schema"],
              schema.range(of: #"^jarvis_sb_eval_[a-f0-9]{32}_[0-9]+$"#, options: .regularExpression) != nil,
              bankID.range(of: #"^jarvis_[a-f0-9]{32}$"#, options: .regularExpression) != nil,
              isHash(manifestSHA256), inputsRoot.lastPathComponent == "inputs" else {
            throw JarvisReplayError.invalidPlan
        }
        // Never synthesize a schema/bank, nor hash rebound runtime JSON.
        return .init(scope: scope, manifestSHA256: manifestSHA256,
            controlsURL: inputsRoot.deletingLastPathComponent().appendingPathComponent("controls"))
    }
    static func readScope(_ url: URL) throws -> Data {
        let file = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard file >= 0 else { throw JarvisReplayError.memoryReadinessUnavailable }
        defer { Darwin.close(file) }
        var info = stat()
        guard fstat(file, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == geteuid(), info.st_mode & 0o077 == 0, info.st_nlink == 1,
              info.st_size > 0, info.st_size <= 8192 else { throw JarvisReplayError.memoryReadinessUnavailable }
        var bytes = [UInt8](repeating: 0, count: 8193)
        let count = bytes.withUnsafeMutableBytes { Darwin.read(file, $0.baseAddress, $0.count) }
        guard count == info.st_size else { throw JarvisReplayError.memoryReadinessUnavailable }
        return Data(bytes.prefix(count))
    }
    static func isHash(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) }
    }
}

/// Private regular files, pinned directory, no-follow bounded reads and atomic
/// request replacement. Used only from the replay scheduler, never an ADM or
/// onFrame callback. No host/provider work is performed here.
final class JarvisReplayMemoryFiles {
    static let requestName = "memory-readiness-request.json"
    static let receiptName = "memory-readiness-receipt.json"
    private var fd: Int32
    private let root: URL
    private let device: dev_t
    private let inode: ino_t
    init(_ root: URL) throws {
        self.root = root
        fd = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        var info = stat()
        guard fd >= 0, fstat(fd, &info) == 0, info.st_uid == geteuid(),
              info.st_mode & 0o077 == 0, info.st_mode & S_IFMT == S_IFDIR else {
            if fd >= 0 { Darwin.close(fd) }
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        device = info.st_dev; inode = info.st_ino
    }
    deinit { close() }
    func close() { if fd >= 0 { Darwin.close(fd); fd = -1 } }
    private func check() throws {
        var info = stat()
        guard fd >= 0, lstat(root.path, &info) == 0, info.st_dev == device, info.st_ino == inode,
              info.st_mode & S_IFMT == S_IFDIR, info.st_mode & 0o077 == 0, info.st_uid == geteuid() else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
    }
    func read(_ name: String) throws -> Data? {
        try check()
        guard [Self.requestName, Self.receiptName].contains(name) else { throw JarvisReplayError.invalidPlan }
        let limit = name == Self.requestName ? 16384 : 32768
        let file = openat(fd, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if file < 0 {
            if errno == ENOENT { return nil }
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        defer { Darwin.close(file) }
        var before = stat()
        guard fstat(file, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
              before.st_uid == geteuid(), before.st_mode & 0o077 == 0, before.st_nlink == 1,
              before.st_size > 0, before.st_size <= limit else { throw JarvisReplayError.memoryReadinessUnavailable }
        var bytes = [UInt8](repeating: 0, count: limit + 1)
        let count = bytes.withUnsafeMutableBytes { Darwin.read(file, $0.baseAddress, $0.count) }
        var after = stat()
        guard count == before.st_size, count <= limit, fstat(file, &after) == 0,
              before.st_size == after.st_size, before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        try check()
        return Data(bytes.prefix(count))
    }
    func publish(_ data: Data) throws {
        try check()
        guard !data.isEmpty, data.count <= 16384 else { throw JarvisReplayError.invalidPlan }
        // Reject a hostile/invalid existing target instead of following it.
        _ = try read(Self.requestName)
        let temporary = ".memory-app-" + UUID().uuidString.lowercased()
        let file = openat(fd, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard file >= 0 else { throw JarvisReplayError.memoryReadinessUnavailable }
        defer { Darwin.close(file); unlinkat(fd, temporary, 0) }
        let written = data.withUnsafeBytes { Darwin.write(file, $0.baseAddress, $0.count) }
        guard written == data.count, fsync(file) == 0 else { throw JarvisReplayError.memoryReadinessUnavailable }
        try check()
        guard renameat(fd, temporary, fd, Self.requestName) == 0, fsync(fd) == 0 else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
    }
}

struct JarvisReplayManualCommit: Equatable {
    let ordinal: Int
    let identity: JarvisReplayManualSessionIdentity
    let clientEventID: UUID
}

struct JarvisReplayMemoryCandidate: Equatable {
    let identity: JarvisReplayResponseIdentity
    let gateEventID: String
    let stop: TimeInterval?
    static func proposals(_ evidence: JarvisReplayAudioEvidence, commit: JarvisReplayManualCommit) -> [Self] {
        guard !evidence.inputMappingTruncated, !evidence.controlEventsTruncated,
              !evidence.controlEventTimingInvalid, !evidence.outputGateHistoryTruncated,
              !evidence.responseEvidenceInvalid, !evidence.responseEvidenceTruncated,
              evidence.callbackFailure == nil else { return [] }
        var result: [Self] = []
        for gate in evidence.outputGates {
            guard gate.enabled, let identity = gate.responseIdentity,
                  let diagnostic = gate.playbackDiagnostics, diagnostic.origin == .speakerPredicate,
                  diagnostic.predicateAllowsPlayback, diagnostic.itemMatchesPermittedResponse,
                  diagnostic.replayItemIdentityPresent,
                  diagnostic.item?.typeCategory == "message",
                  diagnostic.item?.phaseCategory == "final_answer",
                  diagnostic.item?.policyAllowsPlayback == true,
                  evidence.controlEvents.contains(where: {
                      $0.stage == .backendControlHandled && $0.fields["event_type"] == "audio.webrtc.response"
                      && $0.fields["response_id"] == identity.responseID
                      && $0.fields["connection_id"] == identity.connectionID.uuidString.lowercased()
                      && $0.fields["control_connection_id"] == commit.identity.controlConnectionID.uuidString.lowercased()
                      && $0.fields["audio_session_id"] == commit.identity.audioSessionID.uuidString.lowercased()
                  }) else { continue }
            // A current connection reset invalidates historical proposals. This
            // is a local fence, NOT a time-based canonical input/output join.
            if evidence.controlEvents.contains(where: {
                $0.stage == .connectionReset && $0.handledAt >= (gate.requestedAt ?? .infinity)
            }) { continue }
            let stop = evidence.controlEvents.first(where: {
                $0.stage == .backendControlHandled && $0.fields["event_type"] == "audio.webrtc.stopped"
                    && $0.boundResponseIdentity == identity
            })?.handledAt
            if !result.contains(where: { $0.identity == identity }) {
                result.append(.init(identity: identity, gateEventID: gate.eventID, stop: stop))
            }
        }
        return result
    }
}

struct JarvisReplayMemoryRequest {
    let bytes: Data
    let fields: [String: Any]
    let candidate: JarvisReplayMemoryCandidate
    let commit: JarvisReplayManualCommit
    let issued: Date
    let expires: Date
    let nonce: String
    var sha256: String { JarvisReplayTaskJSON.hash(bytes) }
    init(configuration: JarvisReplayMemoryConfiguration, invocationID: UUID, target: Int,
         trigger: JarvisReplayManualResponseTrigger, commit: JarvisReplayManualCommit,
         candidate: JarvisReplayMemoryCandidate, issued: Date, expires: Date, nonce: UUID = UUID()) throws {
        guard commit.ordinal == trigger.afterInputOrdinal, commit.ordinal < target,
              commit.identity.ownerID == configuration.scope["owner_id"],
              expires > issued, expires.timeIntervalSince(issued) <= 120 else { throw JarvisReplayError.invalidPlan }
        self.candidate = candidate; self.commit = commit
        self.issued = try JarvisReplayTaskJSON.time(JarvisReplayTaskJSON.timestamp(issued))
        self.expires = try JarvisReplayTaskJSON.time(JarvisReplayTaskJSON.timestamp(expires))
        self.nonce = nonce.uuidString.lowercased()
        fields = [
            "scope": configuration.scope, "invocation_id": invocationID.uuidString.lowercased(),
            "manifest_sha256": configuration.manifestSHA256,
            "session_id": commit.identity.sessionID.uuidString.lowercased(),
            "audio_session_id": commit.identity.audioSessionID.uuidString.lowercased(),
            "connection_id": commit.identity.controlConnectionID.uuidString.lowercased(),
            "prior_input_ordinal": commit.ordinal, "target_input_ordinal": target,
            "segment_id": commit.identity.segmentID.uuidString.lowercased(),
            "client_event_id": commit.clientEventID.uuidString.lowercased(),
            "response_id": candidate.identity.responseID, "item_id": candidate.identity.itemID,
            "nonce": self.nonce, "issued_at": JarvisReplayTaskJSON.timestamp(issued),
            "expires_at": JarvisReplayTaskJSON.timestamp(expires), "canonical_requirement": trigger.canonical]
        bytes = try JarvisReplayTaskJSON.encode(
            ["format": "jarvis.native_memory_readiness.request.v1", "request": fields], limit: 16384)
    }
}

/// Decode only typed metadata. Inline evidence remains host-owned; no source
/// contents, transcript or arbitrary inline objects are copied into capture.
enum JarvisReplayMemoryReceipt {
    /// Additive host observation, never an authority to replace the local scene
    /// origin, extend its deadline, or refresh the pre-enqueue consume window.
    static func validatedSourceClock(_ raw: Any) throws -> [String: Any] {
        guard let value = raw as? [String: Any] else { throw JarvisReplayError.memoryReadinessUnavailable }
        try JarvisReplayTaskJSON.keys(value, ["clock", "boot_session_id", "phase_start_seconds",
            "arm_uptime_seconds", "deadline_uptime_seconds", "sampled_uptime_seconds",
            "remaining_seconds", "status", "anchor_sha256"])
        guard value["clock"] as? String == "ProcessInfo.systemUptime",
              value["status"] as? String == "open",
              let boot = value["boot_session_id"] as? String, boot.utf8.count == 36,
              UUID(uuidString: boot) != nil,
              let hash = value["anchor_sha256"] as? String,
              JarvisReplayMemoryConfiguration.isHash(hash) else { throw JarvisReplayError.memoryReadinessUnavailable }
        func number(_ key: String) throws -> Double {
            guard let number = value[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite, number.doubleValue >= 0 else {
                throw JarvisReplayError.memoryReadinessUnavailable
            }
            return number.doubleValue
        }
        let phase = try number("phase_start_seconds"), arm = try number("arm_uptime_seconds")
        let deadline = try number("deadline_uptime_seconds"), sampled = try number("sampled_uptime_seconds")
        let remaining = try number("remaining_seconds")
        guard phase <= JarvisReplayPlan.maximumSceneDuration, arm >= phase,
              deadline > arm, deadline - arm <= JarvisReplayPlan.maximumTriggerTimeout,
              sampled >= arm, sampled < deadline, remaining > 0,
              remaining <= JarvisReplayPlan.maximumTriggerTimeout,
              abs(remaining - (deadline - sampled)) <= max(deadline.ulp, sampled.ulp) * 4 else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        return value
    }
    static func boolean(_ value: Any?, _ expected: Bool) -> Bool {
        guard let value = value as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else { return false }
        return value.boolValue == expected
    }
    static func ready(_ data: Data, request: JarvisReplayMemoryRequest, now: Date,
                      notBefore: Date? = nil) throws -> [String: Any]? {
        let envelope = try JarvisReplayTaskJSON.object(data, limit: 32768)
        try JarvisReplayTaskJSON.keys(envelope, ["format", "request_sha256", "recorded_at", "readiness"])
        guard envelope["format"] as? String == "jarvis.native_memory_readiness.receipt.v1" else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        // Old receipts normally remain while a newer nonce is being observed.
        guard envelope["request_sha256"] as? String == request.sha256 else { return nil }
        let recorded = try JarvisReplayTaskJSON.time(envelope["recorded_at"])
        guard recorded >= (notBefore ?? request.issued), recorded <= now, now < request.expires,
              let value = envelope["readiness"] as? [String: Any] else { return nil }
        let allowed: Set<String> = ["format", "run_id", "owner_id", "invocation_id", "manifest_sha256",
            "session_id", "audio_session_id", "connection_id", "prior_input_ordinal", "target_input_ordinal",
            "segment_id", "client_event_id", "nonce", "canonical_requirement", "response_id", "item_id",
            "expires_at", "canonical_input", "full_chain_proven", "heard_state", "playback",
            "semantic_acceptance", "evidence_authenticity", "control_label_basis", "model_use_of_tool_result",
            "status", "reason", "response", "retained_source", "admission_evidence", "continuation_evidence",
            "observer", "inline_evidence", "continuation_basis", "final_output_policy", "source_clock"]
        guard Set(value.keys).isSubset(of: allowed),
              value["format"] as? String == "jarvis.native_memory_readiness.v1",
              value["run_id"] as? String == (request.fields["scope"] as? [String: String])?["run_id"],
              value["owner_id"] as? String == request.commit.identity.ownerID else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        for key in ["invocation_id", "manifest_sha256", "session_id", "audio_session_id", "connection_id",
                    "segment_id", "client_event_id", "nonce", "canonical_requirement", "response_id", "item_id"] {
            guard value[key] as? String == request.fields[key] as? String else { return nil }
        }
        for key in ["prior_input_ordinal", "target_input_ordinal"] {
            guard let number = value[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue == Double(request.fields[key] as? Int ?? -1) else { return nil }
        }
        guard try JarvisReplayTaskJSON.time(value["expires_at"]) == request.expires,
              boolean(value["full_chain_proven"], false), value["heard_state"] as? String == "unknown",
              value["playback"] as? String == "not_evaluated",
              value["semantic_acceptance"] as? String == "not_evaluated" else { return nil }
        guard value["status"] as? String == "ready" else { return nil }
        guard let policy = value["final_output_policy"] as? [String: Any],
              let output = policy["realtime_output"] as? [String: Any] else { return nil }
        try JarvisReplayTaskJSON.keys(policy, ["response_kind", "audio_session_id", "realtime_output"])
        try JarvisReplayTaskJSON.keys(output, ["item_id", "phase", "phase_anomaly", "delivery_disposition"])
        guard policy["response_kind"] as? String == "normal_user_turn",
              policy["audio_session_id"] as? String == request.commit.identity.audioSessionID.uuidString.lowercased(),
              output["item_id"] as? String == request.candidate.identity.itemID,
              output["phase"] as? String == "final_answer", boolean(output["phase_anomaly"], false),
              output["delivery_disposition"] as? String == "forwarded",
              let response = value["response"] as? [String: Any],
              let input = value["canonical_input"] as? [String: Any] else { return nil }
        try JarvisReplayTaskJSON.keys(response, ["event_id", "revision", "response_id", "item_id", "completion_state"])
        try JarvisReplayTaskJSON.keys(input, ["event_id", "payload_id", "revision", "source_namespace",
                                             "source_event_id", "provider_item_id"])
        _ = try JarvisReplayTaskJSON.uuid(response["event_id"])
        _ = try JarvisReplayTaskJSON.positive(response["revision"])
        _ = try JarvisReplayTaskJSON.uuid(input["event_id"])
        _ = try JarvisReplayTaskJSON.uuid(input["payload_id"])
        _ = try JarvisReplayTaskJSON.positive(input["revision"])
        for key in ["source_namespace", "source_event_id", "provider_item_id"] {
            _ = try JarvisReplayTaskJSON.string(input[key], max: 512)
        }
        guard response["response_id"] as? String == request.candidate.identity.responseID,
              response["item_id"] as? String == request.candidate.identity.itemID,
              response["completion_state"] as? String == "completed",
              let observer = value["observer"] as? [String: Any],
              boolean(observer["read_only"], true), boolean(observer["automatic_polling"], false),
              observer["isolation"] as? String == "repeatable_read" else { return nil }
        try JarvisReplayTaskJSON.keys(observer, ["queries", "elapsed_seconds", "read_only", "isolation", "automatic_polling"])
        guard try JarvisReplayTaskJSON.positive(observer["queries"]) <= 9,
              let elapsed = observer["elapsed_seconds"] as? NSNumber,
              CFGetTypeID(elapsed) != CFBooleanGetTypeID(), elapsed.doubleValue.isFinite,
              elapsed.doubleValue >= 0, elapsed.doubleValue <= 3.1 else { return nil }
        var projection: [String: Any] = [
            "request_sha256": request.sha256, "nonce": request.nonce,
            "recorded_at": envelope["recorded_at"]!, "canonical_input": input, "response": response,
            "final_output_policy": policy, "full_chain_proven": false, "heard_state": "unknown",
            "playback": "not_evaluated", "semantic_acceptance": "not_evaluated"]
        if let sourceClock = value["source_clock"] {
            projection["source_clock"] = try validatedSourceClock(sourceClock)
        }
        func reference(_ key: String, path: String) throws -> [String: Any] {
            guard let ref = value[key] as? [String: Any] else { throw JarvisReplayError.memoryReadinessUnavailable }
            try JarvisReplayTaskJSON.keys(ref, ["path", "sha256", "selector"])
            guard ref["path"] as? String == path,
                  JarvisReplayMemoryConfiguration.isHash(try JarvisReplayTaskJSON.string(ref["sha256"], max: 64)) else {
                throw JarvisReplayError.memoryReadinessUnavailable
            }
            _ = try JarvisReplayTaskJSON.string(ref["selector"], max: 256)
            return ref
        }
        projection["admission_evidence"] = try reference("admission_evidence", path: "inline/admission.json")
        if request.fields["canonical_requirement"] as? String == "retain_searchable" {
            guard value["reason"] as? String == "exact_remember_searchable_and_continued",
                  value["continuation_basis"] as? String == "admitted_response_source_dependencies_not_tool_trigger_or_model_use",
                  let source = value["retained_source"] as? [String: Any],
                  source["owner_id"] as? String == request.commit.identity.ownerID,
                  source["area_id"] as? String == (request.fields["scope"] as? [String: String])?["area_id"],
                  source["status"] as? String == "searchable" else { return nil }
            try JarvisReplayTaskJSON.keys(source, ["owner_id", "area_id", "source_id", "revision", "generation",
                "payload_id", "document_id", "content_digest", "retain_operation_id", "status"])
            for key in ["source_id", "payload_id", "retain_operation_id"] { _ = try JarvisReplayTaskJSON.uuid(source[key]) }
            for key in ["revision", "generation"] { _ = try JarvisReplayTaskJSON.positive(source[key]) }
            _ = try JarvisReplayTaskJSON.string(source["document_id"], max: 512)
            guard JarvisReplayMemoryConfiguration.isHash(try JarvisReplayTaskJSON.string(source["content_digest"], max: 64)) else { return nil }
            projection["retained_source"] = source
            projection["continuation_evidence"] = try reference("continuation_evidence", path: "inline/continuation.json")
        } else if value["reason"] as? String != "finalized_input_and_terminal_response" { return nil }
        return projection
    }
}

/// One attachment, bounded consumes. Probe permits normal press; after matching
/// ready a NEW nonce must be read and consumed immediately before actual PCM.
/// No idle wait here stops the original scene or acquires a new input authority.
@MainActor
final class JarvisReplayMemoryGate {
    private let configuration: JarvisReplayMemoryConfiguration
    private let invocationID: UUID
    private let files: JarvisReplayMemoryFiles
    private var target: Int?
    private var trigger: JarvisReplayManualResponseTrigger?
    private var commit: JarvisReplayManualCommit?
    private var candidate: JarvisReplayMemoryCandidate?
    private var request: JarvisReplayMemoryRequest?
    private var requestAt: TimeInterval = 0
    private var nextPoll: TimeInterval = 0
    private var origin: JarvisReplayClockAnchor?
    private var consumeMode = false
    private var pressedTarget: Int?
    private var readyAt: TimeInterval?
    private var readyProjection: [String: Any]?
    private var usedNonces: Set<String> = []
    private var consumedTargets: Set<Int> = []
    private(set) var attempts = 0
    private(set) var retired = false
    private(set) var status = "not_armed"
    private(set) var settlements: [[String: Any]] = []
    private var requests: [[String: Any]] = []
    var probeReady: Bool { !retired && pressedTarget != target && !consumeMode && readyProjection != nil }

    init(configuration: JarvisReplayMemoryConfiguration, invocationID: UUID) throws {
        self.configuration = configuration; self.invocationID = invocationID
        files = try JarvisReplayMemoryFiles(configuration.controlsURL)
    }
    func poll(target: Int, trigger: JarvisReplayManualResponseTrigger,
              commit: JarvisReplayManualCommit?, evidence: JarvisReplayAudioEvidence,
              clock: JarvisReplayClockAnchor, now: TimeInterval, currentScope: Bool, consume: Bool) throws {
        guard !retired, now.isFinite else { throw JarvisReplayError.memoryReadinessUnavailable }
        if consumedTargets.contains(target) { return }
        let deadline = clock.referenceUptime + trigger.deadline
        guard now < deadline else { status = "deadline"; throw JarvisReplayError.outputWaitExpired }
        guard now >= clock.referenceUptime + trigger.at else { return }
        guard let commit, currentScope else { status = "prior_commit_or_scope_unavailable"; return }
        if self.target != target {
            self.target = target; self.trigger = trigger; self.origin = clock; self.commit = commit
            request = nil; candidate = nil; readyProjection = nil; readyAt = nil; consumeMode = false; pressedTarget = nil
        }
        guard self.commit == commit else { throw JarvisReplayError.manualIdentityMismatch }
        let proposals = JarvisReplayMemoryCandidate.proposals(evidence, commit: commit)
        if candidate == nil || (!consume && pressedTarget != target
            && proposals.last(where: { trigger.playback == "none" || $0.stop != nil })?.identity != candidate?.identity) {
            // This is only a local proposal. The host must bind it to the exact
            // commit/source graph; ordering NEVER supplies the missing edge.
            candidate = proposals.last(where: { trigger.playback == "none" || $0.stop != nil })
            request = nil; readyProjection = nil; readyAt = nil
        }
        guard let candidate else { status = "final_candidate_unavailable"; return }
        guard proposals.contains(where: { $0.identity == candidate.identity
            && (trigger.playback == "none" || $0.stop == candidate.stop && $0.stop != nil) }) else {
            status = "candidate_fenced"; throw JarvisReplayError.responseCandidateChanged
        }
        if pressedTarget == target && !consume {
            // Normal press already happened. Re-probing while waiting for ready
            // cannot authorize another press or enqueue, so do not churn stale
            // 250ms probe receipts. A new consume nonce is still mandatory below.
            status = "waiting_manual_ready"
            return
        }
        guard !consume || pressedTarget == target else { throw JarvisReplayError.memoryReadinessUnavailable }
        if consume != consumeMode {
            consumeMode = consume; request = nil; readyProjection = nil; readyAt = nil
        }
        if let readyAt, now - readyAt > 0.25 {
            request = nil; readyProjection = nil; self.readyAt = nil
        }
        guard readyProjection == nil, now >= nextPoll else { return }
        nextPoll = now + 0.05
        let utc = clock.utc.addingTimeInterval(now - clock.referenceUptime)
        if request == nil {
            guard attempts < 32 else { status = "consume_budget"; throw JarvisReplayError.memoryReadinessUnavailable }
            let new = try JarvisReplayMemoryRequest(configuration: configuration, invocationID: invocationID,
                target: target, trigger: trigger, commit: commit, candidate: candidate,
                issued: clock.utc.addingTimeInterval(trigger.at),
                expires: clock.utc.addingTimeInterval(trigger.deadline))
            try files.publish(new.bytes)
            request = new; requestAt = now; attempts += 1
            requests.append(["nonce": new.nonce, "request_sha256": new.sha256,
                "target_input_ordinal": target, "stage": consume ? "enqueue_consume" : "press_probe",
                "issued_at": new.fields["issued_at"]!, "expires_at": new.fields["expires_at"]!,
                "response_id": candidate.identity.responseID, "item_id": candidate.identity.itemID])
            status = "waiting"; return
        }
        guard let request, try files.read(JarvisReplayMemoryFiles.requestName) == request.bytes else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        if let bytes = try files.read(JarvisReplayMemoryFiles.receiptName),
           let result = try JarvisReplayMemoryReceipt.ready(bytes, request: request, now: utc,
                notBefore: clock.utc.addingTimeInterval(requestAt - clock.referenceUptime)) {
            readyProjection = result; readyAt = now; status = consume ? "enqueue_ready" : "press_ready"
        } else if now - requestAt >= 3.25 {
            // A new nonce never renews issued_at, expires_at, or scene time.
            self.request = nil
        }
    }
    /// Attribute a thrown poll to its exact pending/active ordinal before teardown.
    /// Diagnostic only: no receipt, admission, clock, or gate state is changed.
    func recordPollingFailure(_ error: Error, target: Int, at time: TimeInterval,
                              record: (Int, String, TimeInterval) -> Void) {
        let code: String
        switch error as? JarvisReplayError {
        case .outputWaitExpired: code = "memory_readiness_deadline"
        case .responseCandidateChanged: code = "memory_readiness_candidate_fenced"
        case .manualIdentityMismatch: code = "memory_readiness_identity_mismatch"
        case .memoryReadinessUnavailable:
            code = status == "consume_budget" ? "memory_readiness_consume_budget" : "memory_readiness_unavailable"
        default: code = "memory_readiness_poll_failed"
        }
        // Never retain error descriptions, paths, payloads, or other arbitrary strings.
        record(target, code, time)
    }
    func mayPress(target: Int, now: TimeInterval, currentScope: Bool) -> Bool {
        guard !retired, self.target == target, pressedTarget != target, !consumeMode, currentScope, readyProjection != nil,
              let readyAt, let origin, let trigger else { return false }
        return now >= readyAt && now - readyAt <= 0.25
            && now >= origin.referenceUptime + trigger.at && now < origin.referenceUptime + trigger.deadline
    }
    func markPressed(target: Int, at now: TimeInterval) throws {
        guard mayPress(target: target, now: now, currentScope: true) else {
            throw JarvisReplayError.memoryReadinessUnavailable
        }
        pressedTarget = target
        request = nil; readyProjection = nil; readyAt = nil
        status = "waiting_manual_ready"
    }
    func consume(target: Int, now: TimeInterval, currentScope: Bool) throws -> Bool {
        if consumedTargets.contains(target) { return false }
        guard !retired, self.target == target, pressedTarget == target, consumeMode, currentScope,
              let trigger, let clock = origin, let request, let readyAt,
              readyProjection != nil, now >= readyAt, now - readyAt <= 0.25,
              now >= clock.referenceUptime + trigger.at, now < clock.referenceUptime + trigger.deadline,
              !usedNonces.contains(request.nonce) else { return false }
        // Re-read literal current bytes at the actual enqueue boundary.
        guard try files.read(JarvisReplayMemoryFiles.requestName) == request.bytes,
              let bytes = try files.read(JarvisReplayMemoryFiles.receiptName),
              let projection = try JarvisReplayMemoryReceipt.ready(bytes, request: request,
                  now: clock.utc.addingTimeInterval(now - clock.referenceUptime),
                  notBefore: clock.utc.addingTimeInterval(requestAt - clock.referenceUptime)) else { return false }
        usedNonces.insert(request.nonce); consumedTargets.insert(target); status = "consumed"
        var record = projection
        record["target_input_ordinal"] = target
        record["enqueue_consume_uptime_seconds"] = now
        record["candidate"] = JarvisReplayEvidenceReport.identity(request.candidate.identity)
        record["candidate_gate_event_id"] = request.candidate.gateEventID
        record["authenticated_buffer_stop_uptime_seconds"] = request.candidate.stop as Any? ?? NSNull()
        record["playback_boundary"] = trigger.playback
        record["decoded_drain_confirmed"] = false
        settlements.append(record)
        readyProjection = nil
        return true
    }
    func stop() { retired = true; readyProjection = nil; files.close() }
    var json: [String: Any] {
        ["schema_version": 1, "status": status, "retired": retired, "attempts": attempts,
         "maximum_consumes": 32, "requests": requests, "settlements": settlements,
         "full_chain_proven": false, "heard_state": "unknown", "decoded_drain_confirmed": false]
    }
}

#endif
