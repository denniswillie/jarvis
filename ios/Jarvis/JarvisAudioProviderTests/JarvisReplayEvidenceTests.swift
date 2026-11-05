import Foundation
import XCTest

// BEGIN REPLAY FRAME LINEAGE TESTS: portable, no UIKit/network execution.
final class JarvisReplayFrameLineageTests: XCTestCase {
    private let run = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
    private let pin = String(repeating: "a", count: 64)
    private let source = Data("original reviewed JPEG fixture".utf8)
    private let encoded = Data("actual adapter JPEG fixture".utf8)

    private func lineage(runID: UUID? = nil, manifest: String? = nil) -> JarvisReplayFrameLineage {
        .init(runID: runID ?? run, launchManifestSHA256: manifest ?? pin, preparedFrameOrdinal: 7,
              scheduledAt: 19.002, sourceJPEG: source, encodedJPEG: encoded, frameSequence: 4,
              capturedAt: Date(timeIntervalSince1970: 1_700_000_000),
              encoding: .init(sourceWidth: 1920, sourceHeight: 1440, sourceScale: 2,
                sourceOrientation: 6, sourcePixelWidth: 3840, sourcePixelHeight: 2880,
                maximumDimension: 320, resizeScale: 1.0 / 6, outputWidth: 320, outputHeight: 240,
                rendererScale: 1, rendererOpaque: false, rendererPreferredRange: 2, jpegQuality: 0.4))
    }
    private func upload(_ value: JarvisReplayFrameLineage? = nil, bytes: Data? = nil,
                        sequence: Int64 = 4, connection: String? = "connection-1") -> JarvisReplayFrameUploadReceipt {
        .init(lineage: value ?? lineage(), uploadedJPEG: bytes ?? encoded, frameSequence: sequence,
              capturedAtWire: "2023-11-14T22:13:20.000Z", clientEventID: "event-1",
              visualSessionID: "visual-1", connectionID: connection, audioSessionID: "audio-1")
    }

    func testKnownHashAndExactOriginalEncodedUploadedBytes() {
        XCTAssertEqual(JarvisReplayFrameLineage.hash(Data("abc".utf8)),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let value = upload()
        XCTAssertEqual(value.lineage.sourceSHA256, JarvisReplayFrameLineage.hash(source))
        XCTAssertEqual(value.lineage.sourceByteCount, source.count)
        XCTAssertEqual(value.lineage.encodedSHA256, JarvisReplayFrameLineage.hash(encoded))
        XCTAssertEqual(value.uploadedByteCount, encoded.count)
        XCTAssertTrue(value.matchesEncodedFrame)
        XCTAssertNotEqual(value.lineage.sourceSHA256, value.uploadedSHA256)
    }
    func testActualNonDefaultEncodingParametersAndScenarioIdentityAreRetained() {
        let value = lineage()
        XCTAssertEqual(value.launchManifestSHA256, pin)
        XCTAssertEqual(value.preparedFrameOrdinal, 7)
        XCTAssertEqual(value.scheduledAt, 19.002)
        XCTAssertEqual(value.frameSequence, 4)
        XCTAssertEqual(value.encoding.json["maximum_dimension"] as? Double, 320)
        XCTAssertEqual(value.encoding.json["jpeg_quality"] as? Double, 0.4)
        XCTAssertEqual(value.encoding.json["source_image_scale"] as? Double, 2)
        XCTAssertEqual(value.encoding.json["source_orientation_raw"] as? Int, 6)
        XCTAssertEqual(value.encoding.json["renderer_preferred_range_raw"] as? Int, 2)
    }
    func testUploadMismatchIsDiagnosticNotInventedIdentityOrMutation() {
        let changed = Data("different uploaded bytes".utf8)
        let value = upload(bytes: changed)
        XCTAssertFalse(value.matchesEncodedFrame)
        XCTAssertEqual(value.uploadedSHA256, JarvisReplayFrameLineage.hash(changed))
        XCTAssertEqual(value.lineage.encodedSHA256, JarvisReplayFrameLineage.hash(encoded))
        XCTAssertFalse(upload(sequence: 5).matchesEncodedFrame)
        var ledger = JarvisReplayFrameLineageLedger(runID: run, launchManifestSHA256: pin)
        ledger.append(value)
        XCTAssertEqual(ledger.json["mismatch_count"] as? Int, 1)
    }
    func testWrongRunAndManifestCannotContaminateLedger() {
        var ledger = JarvisReplayFrameLineageLedger(runID: run, launchManifestSHA256: pin)
        ledger.append(upload(lineage(runID: UUID())))
        ledger.append(upload(lineage(manifest: String(repeating: "b", count: 64))))
        XCTAssertTrue(ledger.receipts.isEmpty)
        XCTAssertEqual(ledger.rejectedIdentityCount, 2)
        ledger.append(upload())
        XCTAssertEqual(ledger.receipts.count, 1)
    }
    func testBoundedLedgerAndIncompleteStreamStayExplicit() throws {
        var ledger = JarvisReplayFrameLineageLedger(runID: run, launchManifestSHA256: pin)
        for _ in 0..<241 { ledger.append(upload(connection: nil)) }
        XCTAssertEqual(ledger.receipts.count, 240)
        XCTAssertTrue(ledger.truncated)
        XCTAssertEqual(ledger.json["missing_stream_identity_count"] as? Int, 240)
        XCTAssertLessThan(try JSONSerialization.data(withJSONObject: ledger.json).count, 1_500_000)
        XCTAssertFalse(upload(connection: String(repeating: "x", count: 129)).streamIdentityPresent)
    }
    func testNoUploadOrHistoricalObservationIsInferredFromEncoding() {
        _ = lineage()
        let empty = JarvisReplayFrameLineageLedger(runID: run, launchManifestSHA256: nil)
        XCTAssertTrue(empty.receipts.isEmpty)
        XCTAssertEqual(empty.json["manifest_bound"] as? Bool, false)
        XCTAssertTrue(upload().json["server_accepted"] is NSNull)
        XCTAssertTrue(upload().json["provider_received"] is NSNull)
    }
    func testExportContainsOnlyMetadataNotJPEGOrOracle() throws {
        var ledger = JarvisReplayFrameLineageLedger(runID: run, launchManifestSHA256: pin)
        ledger.append(upload())
        let text = String(decoding: try JSONSerialization.data(withJSONObject: ledger.json), as: UTF8.self)
        XCTAssertFalse(text.contains("original reviewed JPEG fixture"))
        XCTAssertFalse(text.contains("actual adapter JPEG fixture"))
        for forbidden in ["image_base64", "expected_answer", "prior_context", "bearer_token", "speech_text"] {
            XCTAssertFalse(text.contains(forbidden))
        }
        XCTAssertTrue(text.contains("socket_send_returned"))
        XCTAssertTrue(text.contains("visual-1"))
        XCTAssertTrue(text.contains("connection-1"))
    }
}
// END REPLAY FRAME LINEAGE TESTS

@MainActor
final class JarvisProviderInputFinalizationTests: XCTestCase {
    private let connection = "11111111-1111-4111-8111-111111111111"
    private let clip = Data([200, 0] + Array(repeating: UInt8(0), count: 1918))
    private func plan(timeout: Double = 2) throws -> JarvisReplayPlan {
        try .init(runID: UUID(uuidString: "22222222-2222-4222-8222-222222222222")!, duration: 10,
            frames: [.init(at: 0, image: Data([1])), .init(at: 2, image: Data([2]))],
            utterances: [.init(trigger: .atTime(1), pcm16: clip, clipID: "pause"),
                .init(trigger: .afterProviderInputFinalized(afterInputOrdinal: 0, timeout: timeout),
                      pcm16: clip, clipID: "continue")], postMediaCaptureSeconds: 0)
    }
    private func event(_ state: inout JarvisReplayPCMState, _ type: String, at: Double,
                       item: String? = "item_pause", id: String? = nil, connection other: String? = nil) {
        var fields: [String: Any] = ["event_type": type, "connection_id": other ?? connection,
            "provider_event_id": id ?? "event_\(type)", "boundary": "ordered_current_provider_channel",
            "transcript": "MUST_NOT_BE_RETAINED"]
        if let item { fields["item_id"] = item }
        state.recordControlEvent(.providerDataChannelHandled, fields: fields, receivedAt: at, at: at + 0.0001)
    }
    private func state(_ plan: JarvisReplayPlan) throws -> JarvisReplayPCMState {
        var state = JarvisReplayPCMState(maxDuration: 10)
        try state.register(plan)
        try state.recordAnchor(.init(monotonicBefore: 1000, utc: Date(timeIntervalSince1970: 1_800_000_000),
                                     monotonicAfter: 1000))
        state.gateInput(true)
        event(&state, "session.created", at: 999.5, item: nil)
        return state
    }
    private func callback(_ state: inout JarvisReplayPCMState, at: Double) {
        _ = state.renderInput(at: at)
        state.inputCallbackStarted(at: at)
        state.inputCallbackCompleted(status: 0, at: at + 0.001)
    }
    private func lifecycle(_ state: inout JarvisReplayPCMState, at: Double, item: String = "item_pause",
                           prefix: String = "event_pause") {
        for (offset, type) in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                              "input_audio_buffer.committed", "conversation.item.input_audio_transcription.completed"].enumerated() {
            event(&state, type, at: at + Double(offset) * 0.001, item: item, id: "\(prefix)_\(offset)")
        }
    }
    private func completed(_ plan: JarvisReplayPlan, early: Bool) throws -> JarvisReplayPCMState {
        var state = try state(plan)
        try state.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
        callback(&state, at: 1001.010)
        if early { lifecycle(&state, at: 1001.012) }
        callback(&state, at: 1001.020)
        if !early { lifecycle(&state, at: 1001.030) }
        return state
    }
    func testFinalizedBeforeSDKEndWaitsForAllSamples() throws {
        let plan = try plan()
        var state = try state(plan)
        var timeline = JarvisReplayTimeline(plan: plan)
        XCTAssertEqual(timeline.advance(elapsed: 1, output: nil, inputIdle: true, origin: 1000).utterance, 0)
        try state.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
        callback(&state, at: 1001.010)
        lifecycle(&state, at: 1001.012)
        var snapshot = state.responseSnapshot(afterInputOrdinal: 0)
        XCTAssertNil(snapshot.inputCompletion); XCTAssertFalse(snapshot.providerInputFinalized)
        XCTAssertNil(timeline.advance(elapsed: 1.016, output: nil, inputIdle: false,
            responseSnapshot: snapshot, origin: 1000).utterance)
        callback(&state, at: 1001.020)
        snapshot = state.responseSnapshot(afterInputOrdinal: 0)
        XCTAssertTrue(snapshot.providerInputFinalized)
        XCTAssertEqual(timeline.advance(elapsed: 1.022, output: nil, inputIdle: true,
            responseSnapshot: snapshot, origin: 1000).utterance, 1)
        try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: 1001.022)
        XCTAssertEqual(state.evidence.utterances[1].providerInputFinalization?.enqueueRevalidatedAt, 1001.022)
    }
    func testNormalFinalizationAfterSDKCompletion() throws {
        let plan = try plan()
        var state = try completed(plan, early: false)
        XCTAssertTrue(state.responseSnapshot(afterInputOrdinal: 0).providerInputFinalized)
        try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: 1001.04)
        XCTAssertEqual(state.evidence.utterances[1].providerInputFinalization?.itemID, "item_pause")
    }
    func testPreviousItemNeverSatisfiesNewTransaction() throws {
        let plan = try plan()
        var state = try state(plan)
        lifecycle(&state, at: 999.6, item: "item_old", prefix: "event_old")
        try state.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
        callback(&state, at: 1001.010); callback(&state, at: 1001.020)
        event(&state, "conversation.item.input_audio_transcription.completed", at: 1001.03,
              item: "item_old", id: "event_old_3")
        XCTAssertFalse(state.responseSnapshot(afterInputOrdinal: 0).providerInputFinalized)
        XCTAssertThrowsError(try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: 1001.04))
    }
    func testUnfinishedPrefixAndMissingSessionFailClosed() throws {
        for missingSession in [false, true] {
            let plan = try plan()
            var state = missingSession ? JarvisReplayPCMState(maxDuration: 10) : try self.state(plan)
            if missingSession {
                try state.register(plan); state.gateInput(true)
                try state.recordAnchor(.init(monotonicBefore: 1000, utc: Date(timeIntervalSince1970: 1_800_000_000),
                                            monotonicAfter: 1000))
            } else { event(&state, "input_audio_buffer.speech_started", at: 999.7, item: "item_old") }
            try state.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
            callback(&state, at: 1001.010); callback(&state, at: 1001.020)
            XCTAssertNotNil(state.responseSnapshot(afterInputOrdinal: 0).providerInputFailure)
        }
    }
    func testAmbiguousSplitAndResetRejectReadyCandidate() throws {
        for reset in [false, true] {
            let plan = try plan()
            var state = try completed(plan, early: false)
            if reset { state.recordControlEvent(.connectionReset, fields: ["connection_id": connection], at: 1001.04) }
            else { event(&state, "input_audio_buffer.speech_started", at: 1001.04, item: "item_other", id: "event_other") }
            XCTAssertFalse(state.responseSnapshot(afterInputOrdinal: 0).providerInputFinalized)
            XCTAssertNotNil(state.responseSnapshot(afterInputOrdinal: 0).providerInputFailure)
            XCTAssertThrowsError(try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: 1001.05))
        }
    }
    func testDuplicatesIdempotentButEventConflictAndForeignConnectionFail() throws {
        let plan = try plan()
        var duplicate = try completed(plan, early: false)
        event(&duplicate, "conversation.item.input_audio_transcription.completed", at: 1001.04, id: "event_pause_3")
        XCTAssertTrue(duplicate.responseSnapshot(afterInputOrdinal: 0).providerInputFinalized)
        var conflict = duplicate
        event(&conflict, "input_audio_buffer.committed", at: 1001.05, id: "event_pause_3")
        XCTAssertEqual(conflict.responseSnapshot(afterInputOrdinal: 0).providerInputFailure, "provider_input_event_conflict")
        var foreign = duplicate
        event(&foreign, "input_audio_buffer.speech_started", at: 1001.05, item: "item_other", id: "event_foreign",
              connection: "33333333-3333-4333-8333-333333333333")
        XCTAssertNotNil(foreign.responseSnapshot(afterInputOrdinal: 0).providerInputFailure)
    }
    func testMissingTranscriptionAndFailedTranscriptionDoNotRelease() throws {
        let plan = try plan()
        var state = try self.state(plan)
        try state.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
        callback(&state, at: 1001.010); callback(&state, at: 1001.020)
        event(&state, "input_audio_buffer.speech_started", at: 1001.03)
        event(&state, "input_audio_buffer.speech_stopped", at: 1001.04)
        event(&state, "input_audio_buffer.committed", at: 1001.05)
        XCTAssertFalse(state.responseSnapshot(afterInputOrdinal: 0).providerInputFinalized)
        event(&state, "conversation.item.input_audio_transcription.failed", at: 1001.06)
        XCTAssertEqual(state.responseSnapshot(afterInputOrdinal: 0).providerInputFailure, "provider_input_transcription_failed")
    }
    func testExactDeadlineLateWakeupAndFramesAdvance() throws {
        let plan = try plan()
        for elapsed in [3.021, 3.5] {
            var state = try completed(plan, early: true)
            let snapshot = state.responseSnapshot(afterInputOrdinal: 0)
            let end = try XCTUnwrap(snapshot.inputCompletion).at
            var timeline = JarvisReplayTimeline(plan: plan)
            _ = timeline.advance(elapsed: 1, output: nil, inputIdle: true, origin: 1000)
            let at = elapsed == 3.021 ? end + 2 - 1000 : elapsed
            let step = timeline.advance(elapsed: at, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 1000)
            XCTAssertEqual(step.failure, .outputWaitExpired); XCTAssertNil(step.utterance); XCTAssertEqual(step.frame, 1)
            XCTAssertThrowsError(try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: end + 2))
        }
    }
    func testResetBetweenSnapshotAndEnqueueCannotUseCachedReadiness() throws {
        let plan = try plan()
        var state = try completed(plan, early: true)
        XCTAssertTrue(state.responseSnapshot(afterInputOrdinal: 0).providerInputFinalized)
        state.recordControlEvent(.connectionReset, at: 1001.03)
        XCTAssertThrowsError(try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: 1001.04))
        XCTAssertEqual(state.evidence.utterances[1].acceptedSamples, 0)
    }
    func testBeforeProductionCancellationAndTruncationFailClosed() throws {
        let plan = try plan()
        var early = try state(plan)
        try early.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
        lifecycle(&early, at: 1001.001)
        callback(&early, at: 1001.010); callback(&early, at: 1001.020)
        XCTAssertEqual(early.responseSnapshot(afterInputOrdinal: 0).providerInputFailure, "provider_input_before_sdk_production")
        var cancelled = try state(plan)
        try cancelled.enqueue(clip, utteranceOrdinal: 0, requestedAt: 1001)
        cancelled.gateInput(false, at: 1001.005)
        XCTAssertEqual(cancelled.responseSnapshot(afterInputOrdinal: 0).providerInputFailure, "provider_input_sdk_incomplete")
        let complete = try completed(plan, early: true)
        var evidence = complete.evidence; evidence.controlEventsTruncated = true
        let proof = try XCTUnwrap(evidence.utterances[1].providerInputFinalization).evaluated(evidence)
        XCTAssertEqual(proof.failureCode, "provider_input_evidence_unavailable")
    }
    func testActualSerializerFixtureKeepsUnknownAuthorityAndNoTranscript() throws {
        let plan = try plan()
        var state = try completed(plan, early: true)
        try state.enqueue(clip, utteranceOrdinal: 1, requestedAt: 1001.022)
        callback(&state, at: 1001.03); callback(&state, at: 1001.04)
        let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: plan.runID, finalUIStatus: "controlled_fixture")
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertFalse(text.contains("MUST_NOT_BE_RETAINED"))
        let rows = try XCTUnwrap(report["utterances"] as? [[String: Any]])
        XCTAssertNil(rows[0]["provider_input_finalization"])
        let proof = try XCTUnwrap(rows[1]["provider_input_finalization"] as? [String: Any])
        XCTAssertEqual(proof["status"] as? String, "validated_provider_transaction")
        XCTAssertEqual(proof["canonical_input_binding"] as? String, "unknown")
        XCTAssertEqual(proof["transcription_completed_control_event_index"] as? Int, 4)
        XCTAssertEqual(proof["input_complete_uptime_seconds"] as? Double, 1001.021)
        let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
        attachment.name = "provider-input-finalized-capture.json"; attachment.lifetime = .keepAlways
        add(attachment)
        print("PROVIDER_INPUT_FINALIZED_FIXTURE_BASE64=" + data.base64EncodedString())
    }
}

@MainActor
final class JarvisReplayEvidenceTests: XCTestCase {
    private func pcm(_ samples: [Int16]) -> Data {
        Data(samples.flatMap { sample -> [UInt8] in
            let value = UInt16(bitPattern: sample)
            return [UInt8(truncatingIfNeeded: value), UInt8(truncatingIfNeeded: value >> 8)]
        })
    }
    private func plan(_ clips: [[Int16]], wait: Bool = false) throws -> JarvisReplayPlan {
        try JarvisReplayPlan(runID: UUID(), duration: 10, frames: [.init(at: 0, image: Data([1]))],
            utterances: clips.enumerated().map { index, samples in
                .init(trigger: wait ? .outputStart(afterCursor: 0, delay: 0, timeout: 2) : .atTime(Double(index)),
                      pcm16: pcm(samples), clipID: "same-clip", assetActivityThreshold: 128,
                      assetFirstActiveSample: samples.firstIndex { abs(Int($0)) >= 128 },
                      assetLastActiveSampleExclusive: samples.lastIndex { abs(Int($0)) >= 128 }.map { $0 + 1 })
            })
    }
    private func inject(_ state: inout JarvisReplayPCMState, _ plan: JarvisReplayPlan, ordinal: Int,
                        at time: TimeInterval, status: Int32 = 0) throws {
        try state.enqueue(plan.utterances[ordinal].pcm16, utteranceOrdinal: ordinal, requestedAt: time - 0.1, triggeredAt: time - 0.2)
        _ = state.renderInput(at: time - 0.01)
        state.inputCallbackStarted(at: time)
        state.inputCallbackCompleted(status: status, at: time + 0.001)
    }
    func testRepeatedAndSilentClipsHaveDistinctOccurrencesAndActualCallbackTimes() throws {
        let plan = try plan([[0, 200, 0], [0, 0, 0], [0, 200, 0]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true)
        for index in 0..<3 { try inject(&state, plan, ordinal: index, at: 100 + Double(index)) }
        let records = state.evidence.utterances
        XCTAssertEqual(Set(records.map(\.occurrenceID)).count, 3)
        XCTAssertEqual(Set(records.map(\.triggerID)).count, 3)
        XCTAssertEqual(Set(records.compactMap(\.clipID)), ["same-clip"])
        XCTAssertTrue(records.allSatisfy { $0.status == "sdk_input_complete" && $0.acceptedSamples == 3 })
        XCTAssertTrue(records[1].allSamplesZero); XCTAssertEqual(records[1].firstCallbackStart, 101)
        XCTAssertEqual(records[1].lastCallbackEnd, 101.001); XCTAssertNil(records[1].firstSuccessfulActivity)
        XCTAssertEqual(records[0].pcmSHA256, records[2].pcmSHA256)
        XCTAssertEqual(records[0].activityThreshold, 128)
        XCTAssertEqual(try XCTUnwrap(records[0].firstSuccessfulActivity), 100 + 1.0 / 48_000, accuracy: 0.0000001)
    }
    func testMultipleSilentOccurrencesAreNotMissingOrMislabelledAsSpeech() throws {
        let plan = try plan([[0, 0], [0, 0]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true)
        try inject(&state, plan, ordinal: 0, at: 2); try inject(&state, plan, ordinal: 1, at: 3)
        XCTAssertTrue(state.evidence.inputFirstNonSilent.isEmpty)
        XCTAssertEqual(state.evidence.utterances.count, 2)
        XCTAssertTrue(state.evidence.utterances.allSatisfy { $0.callbacks.count == 1 && $0.firstCallbackStart != nil && $0.firstSuccessfulActivity == nil })
    }
    func testFailedAndNeverStartedOccurrencesRemainInLedger() throws {
        let plan = try plan([[128], [128]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan)
        XCTAssertThrowsError(try state.enqueue(plan.utterances[0].pcm16, utteranceOrdinal: 0, requestedAt: 10, triggeredAt: 9.9))
        state.close(at: 11)
        XCTAssertEqual(state.evidence.utterances[0].status, "failed")
        XCTAssertEqual(state.evidence.utterances[0].enqueueRequestedAt, 10)
        XCTAssertNil(state.evidence.utterances[0].firstCallbackStart)
        XCTAssertEqual(state.evidence.utterances[1].status, "not_started")
        XCTAssertNil(state.evidence.utterances[1].enqueueRequestedAt)
    }
    func testFailedSDKCallbackIsAnAttemptNotSuccessfulInjection() throws {
        let plan = try plan([[0, 200]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true)
        try inject(&state, plan, ordinal: 0, at: 3.25, status: -50)
        let record = state.evidence.utterances[0]
        XCTAssertEqual(record.status, "failed"); XCTAssertEqual(record.attemptedSamples, 2)
        XCTAssertEqual(record.acceptedSamples, 0); XCTAssertNil(record.firstSuccessfulActivity)
        XCTAssertEqual(record.firstCallbackStart, 3.25); XCTAssertEqual(record.callbacks[0].status, -50)
    }
    func testInFlightCompletionAfterStopRequestIsStillAccountedBeforeFenceReceipt() throws {
        let plan = try plan([[200]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true)
        try state.enqueue(plan.utterances[0].pcm16, utteranceOrdinal: 0, requestedAt: 1)
        _ = state.renderInput(at: 2); state.inputCallbackStarted(at: 2.01)
        state.close(at: 2.02); state.inputCallbackCompleted(status: 0, at: 2.03)
        XCTAssertEqual(state.evidence.utterances[0].stopRequestedAt, 2.02)
        XCTAssertEqual(state.evidence.utterances[0].acceptedSamples, 1)
        XCTAssertEqual(state.evidence.utterances[0].lastCallbackEnd, 2.03)
    }
    func testInputBlockOffsetsExcludeTransportPaddingAndKeepCallbackGaps() throws {
        let plan = try plan([[Int16](repeating: 200, count: 483)])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true)
        try state.enqueue(plan.utterances[0].pcm16, utteranceOrdinal: 0, requestedAt: 1)
        _ = state.renderInput(at: 2); state.inputCallbackStarted(at: 2); state.inputCallbackCompleted(status: 0, at: 2.001)
        _ = state.renderInput(at: 2.04); state.inputCallbackStarted(at: 2.04); state.inputCallbackCompleted(status: 0, at: 2.041)
        let blocks = state.evidence.utterances[0].callbacks
        XCTAssertEqual(blocks.map(\.sourceSampleOffset), [0, 480]); XCTAssertEqual(blocks.map(\.clipSampleCount), [480, 3])
        XCTAssertEqual(blocks[1].transportFrameCount, 480)
        XCTAssertEqual(state.evidence.utterances[0].acceptedSamples, 483)
        XCTAssertEqual(try XCTUnwrap(state.evidence.utterances[0].lastSuccessfulActivity), 2.04 + 2.0 / 48_000, accuracy: 0.0000001)
    }
    func testAnchorAndExportUseSecondsAndDoNotManufactureUsefulOnset() throws {
        let plan = try plan([[128]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan)
        try state.recordAnchor(.init(monotonicBefore: 1000, utc: Date(timeIntervalSince1970: 1_700_000_000.5), monotonicAfter: 1000.002))
        let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: plan.runID, finalUIStatus: "fixture")
        let anchor = try XCTUnwrap(report["clock_anchor"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(anchor["monotonic_reference_seconds"] as? Double), 1000.001, accuracy: 0.0000001)
        XCTAssertEqual(anchor["utc_unix_seconds"] as? Double, 1_700_000_000.5)
        XCTAssertTrue(report["useful_answer_onset"] is NSNull)
        XCTAssertEqual(report["physical_playback_measured"] as? Bool, false)
        XCTAssertTrue(JSONSerialization.isValidJSONObject(report))
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: report))
    }
    func testDecodedOffsetsPreserveGapsAndAllowLaterSampleAnnotation() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.gateOutput(true, at: 5)
        state.observeDecoded([0, 200], at: 5.1, callbackEnd: 5.101)
        state.observeDecoded([300, 400, 0], at: 5.5, callbackEnd: 5.501)
        XCTAssertEqual(state.evidence.decodedSampleMap.map(\.sampleOffset), [0, 2])
        let sample = try XCTUnwrap(JarvisReplayEvidenceReport.locateSample(3, evidence: state.evidence))
        XCTAssertEqual(sample.nominalUptime, 5.5 + 1.0 / 48_000, accuracy: 0.0000001)
        XCTAssertNil(JarvisReplayEvidenceReport.locateSample(5, evidence: state.evidence))
        XCTAssertNil(JarvisReplayEvidenceReport.locateSample(-1, evidence: state.evidence))
    }
    func testPartialOutputBlockTruncationNeverCreatesUnmappedPCM() {
        var state = JarvisReplayPCMState(maxDuration: 3.0 / 48_000)
        state.gateOutput(true)
        state.observeDecoded([100, 200, 300, 400, 500], at: 8)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 6)
        XCTAssertEqual(state.evidence.decodedSampleMap[0].sampleCount, 3)
        XCTAssertTrue(state.evidence.truncated)
        XCTAssertNil(JarvisReplayEvidenceReport.locateSample(3, evidence: state.evidence))
    }
    func testMapCapacityIsBoundedEvenForAbnormallyTinyCallbacks() {
        var state = JarvisReplayPCMState(maxDuration: 0.01)
        state.gateOutput(true)
        state.observeDecoded([100], at: 1)
        state.observeDecoded([200], at: 1.01)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 2)
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 1)
        XCTAssertTrue(state.evidence.decodedMappingTruncated); XCTAssertTrue(state.evidence.truncated)
    }
    func testOneTwoThreeCutsOfEachResumedOutputKeepOccurrenceAndEpochIdentity() throws {
        for cuts in 1...3 {
            let plan = try plan(Array(repeating: [Int16](repeating: 200, count: 480), count: cuts))
            var state = JarvisReplayPCMState(maxDuration: 10)
            try state.register(plan); state.gateInput(true)
            for ordinal in 0..<cuts {
                let onset = 10 + Double(ordinal)
                state.gateOutput(true, at: onset - 0.5)
                state.observeDecoded([200], at: onset - 0.4)
                try inject(&state, plan, ordinal: ordinal, at: onset)
                state.observeDecoded([200, 200], at: onset + 0.02)
                state.gateOutput(false, at: onset + 0.03)
                let gate = try XCTUnwrap(state.evidence.outputGates.last)
                let id = state.evidence.utterances[ordinal].occurrenceID
                let interval = try XCTUnwrap(JarvisReplayEvidenceReport.intervalForSelectedCut(occurrenceID: id, gateEventID: gate.eventID, evidence: state.evidence))
                XCTAssertEqual(interval, 0.02 + 1.0 / 48_000, accuracy: 0.0000001)
                let bytes = state.evidence.decodedPCM16.count
                state.observeDecoded([999], at: onset + 0.04) // late pull after local gate
                XCTAssertEqual(state.evidence.decodedPCM16.count, bytes)
                state.gateOutput(true, at: onset + 0.1)
                let annotatedOffset = state.evidence.decodedPCM16.count / 2
                state.observeDecoded([300], at: onset + 0.2)
                let explicitlyAnnotated = try XCTUnwrap(JarvisReplayEvidenceReport.locateSample(annotatedOffset, evidence: state.evidence))
                let newInterval = explicitlyAnnotated.nominalUptime - (try XCTUnwrap(state.evidence.utterances[ordinal].lastSuccessfulActivity))
                XCTAssertGreaterThan(newInterval, interval)
                // A subsequent cut targets this resumed output; no task semantics are applied here.
                state.gateOutput(false, at: onset + 0.3)
            }
            XCTAssertEqual(Set(state.evidence.utterances.map(\.occurrenceID)).count, cuts)
            XCTAssertTrue(state.evidence.utterances.allSatisfy { $0.clipID == "same-clip" })
        }
    }
    func testCutBeforeAnyDecodedOutputHasNoInventedOldTail() throws {
        let plan = try plan([[200]])
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true); state.gateOutput(true, at: 1)
        try inject(&state, plan, ordinal: 0, at: 2)
        state.gateOutput(false, at: 2.01)
        let gate = try XCTUnwrap(state.evidence.outputGates.last)
        XCTAssertNil(gate.lastAdmittedActiveSample)
        XCTAssertNil(JarvisReplayEvidenceReport.intervalForSelectedCut(occurrenceID: state.evidence.utterances[0].occurrenceID,
            gateEventID: gate.eventID, evidence: state.evidence))
    }
    func testCutDuringResponseWaitKeepsPendingOccurrenceWithoutInjectingTextOrAudio() throws {
        let plan = try plan([[200]], wait: true)
        var clock = JarvisReplayTimeline(plan: plan)
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.armUtterance(0, at: 100)
        XCTAssertNil(clock.advance(elapsed: 1, output: nil, inputIdle: true).utterance)
        state.gateOutput(false, at: 101)
        state.close(at: 101.5)
        XCTAssertEqual(state.evidence.utterances[0].status, "not_started")
        XCTAssertEqual(state.evidence.utterances[0].triggerArmedAt, 100)
        XCTAssertNil(state.evidence.utterances[0].firstCallbackStart)
        state.observeDecoded([200], at: 102)
        XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
    }
    func testResponseWaitFailureNamesTheFailedOccurrence() throws {
        let plan = try plan([[200]], wait: true)
        var clock = JarvisReplayTimeline(plan: plan)
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan)
        let step = clock.advance(elapsed: 2, output: .init(cursor: 1, at: 2), inputIdle: true)
        XCTAssertEqual(step.failedUtterance, 0)
        state.failUtterance(try XCTUnwrap(step.failedUtterance), code: "trigger_outputWaitExpired", at: 102)
        state.close(at: 103)
        XCTAssertEqual(state.evidence.utterances[0].status, "failed")
        XCTAssertNil(state.evidence.utterances[0].firstCallbackStart)
    }
    func testLateOutputCannotPopulateRetiredOrReplacementMediaEvidence() {
        var old = JarvisReplayPCMState(maxDuration: 1)
        old.gateOutput(true); old.observeDecoded([200], at: 1); old.close(at: 2)
        let prior = old.evidence.decodedPCM16
        old.gateOutput(true, at: 3); old.observeDecoded([999], at: 3)
        let replacement = JarvisReplayPCMState(maxDuration: 1)
        XCTAssertEqual(old.evidence.decodedPCM16, prior)
        XCTAssertNotEqual(old.evidence.mediaEpochID, replacement.evidence.mediaEpochID)
        XCTAssertTrue(replacement.evidence.decodedPCM16.isEmpty)
    }
    func testGateHistoryAndSelectedCutClaimsAreBounded() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        for index in 0...JarvisReplayPCMState.maximumOutputGateEntries { state.gateOutput(index.isMultiple(of: 2), at: Double(index)) }
        XCTAssertEqual(state.evidence.outputGates.count, JarvisReplayPCMState.maximumOutputGateEntries)
        XCTAssertTrue(state.evidence.outputGateHistoryTruncated)
        XCTAssertNil(JarvisReplayEvidenceReport.intervalForSelectedCut(occurrenceID: "missing", gateEventID: "missing", evidence: state.evidence))
    }
    func testActualNativeReplacementStartsWithFreshDeviceAndNoInheritedDecodedOutput() async throws {
        let client = JarvisWebRTCClient()
        let old = JarvisReplayAudioDevice(maxDuration: 1), replacement = JarvisReplayAudioDevice(maxDuration: 1)
        defer { client.close(); old.close(); replacement.close(); try? client.useReplayAudioDevice(nil) }
        try client.useReplayAudioDevice(old); _ = try await client.prepareOffer()
        XCTAssertTrue(client.fenceRouteOutput(closeRoute: true)); XCTAssertTrue(old.isRetired)
        try client.useReplayAudioDevice(replacement); _ = try await client.prepareOffer()
        XCTAssertFalse(replacement.isRetired); XCTAssertFalse(client.microphoneEnabled)
        XCTAssertNotEqual(old.evidence.mediaEpochID, replacement.evidence.mediaEpochID)
        XCTAssertTrue(replacement.evidence.decodedPCM16.isEmpty)
        XCTAssertThrowsError(try old.enqueuePCM16(pcm([200])))
        XCTAssertEqual(old.evidence.controlEvents.filter { $0.stage == .offerReady }.count, 1)
        XCTAssertEqual(replacement.evidence.controlEvents.filter { $0.stage == .offerReady }.count, 1)
    }
    func testControlTimelineRetainsReceiptAndHandlingButNotPayloads() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.recordControlEvent(.providerDataChannelHandled, fields: [
            "event_type": "input_audio_buffer.speech_stopped", "item_id": "item_fixture",
            "provider_event_id": "event_fixture", "text": "private transcript",
            "authorization": "private bearer", "reason": "invalid reason with spaces",
        ], receivedAt: 10, at: 10.25)
        let event = try XCTUnwrap(state.evidence.controlEvents.first)
        XCTAssertEqual(event.receivedAt, 10)
        XCTAssertEqual(event.handledAt, 10.25)
        XCTAssertEqual(event.fields, ["event_type": "input_audio_buffer.speech_stopped",
                                      "item_id": "item_fixture", "provider_event_id": "event_fixture"])
        let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: UUID(), finalUIStatus: "fixture")
        let exported = try XCTUnwrap(report["control_events"] as? [[String: Any]])
        XCTAssertEqual(exported[0]["callback_to_handler_seconds"] as? Double, 0.25)
        XCTAssertTrue(JSONSerialization.isValidJSONObject(report))
        XCTAssertFalse(String(decoding: try JSONSerialization.data(withJSONObject: report), as: UTF8.self).contains("private"))
    }
    func testControlTimelinePreservesDistinctNumericGenerationsWithoutChangingPlayback() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        let control = UUID()
        state.recordControlEvent(.backendControlHandled, fields: [
            "event_type": "audio.webrtc.response", "response_id": "resp_fixture",
            "control_connection_id": control.uuidString, "connection_generation": 2,
            "turn_generation": 7, "route_generation": 3, "text": "PRIVATE",
        ], at: 10)
        let event = try XCTUnwrap(state.evidence.controlEvents.first)
        XCTAssertEqual(event.generations, ["connection_generation": 2, "turn_generation": 7, "route_generation": 3])
        XCTAssertEqual(event.jsonFields["turn_generation"] as? Int, 7)
        XCTAssertEqual(event.fields["control_connection_id"], control.uuidString.lowercased())
        XCTAssertFalse(state.outputEnabled)
        XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
        XCTAssertFalse(String(decoding: try JSONSerialization.data(withJSONObject: event.jsonFields), as: UTF8.self).contains("PRIVATE"))
    }
    func testControlGenerationsRejectBooleanFractionNonfiniteAndString() {
        for invalid: Any in [true, -1, 1.5, Double.nan, Double.infinity, "7", NSNull()] {
            XCTAssertTrue(JarvisReplayControlGenerations.safe(["turn_generation": invalid]).isEmpty)
        }
        XCTAssertEqual(JarvisReplayControlGenerations.safe(["turn_generation": 0]), ["turn_generation": 0])
    }
    func testControlTimelineIsBoundedAndRetirementRejectsLateEvents() {
        var state = JarvisReplayPCMState(maxDuration: 1)
        for index in 0...JarvisReplayPCMState.maximumControlEvents {
            state.recordControlEvent(.offerStarted, at: Double(index))
        }
        XCTAssertEqual(state.evidence.controlEvents.count, JarvisReplayPCMState.maximumControlEvents)
        XCTAssertTrue(state.evidence.controlEventsTruncated)
        state.close(at: 1000)
        let count = state.evidence.controlEvents.count
        state.recordControlEvent(.mediaStarted, at: 1001)
        XCTAssertEqual(state.evidence.controlEvents.count, count)
    }
    func testInvalidControlTimesAreFlaggedWithoutInventingDelay() {
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.recordControlEvent(.providerDataChannelHandled, receivedAt: 11, at: 10)
        state.recordControlEvent(.offerReady, at: .nan)
        XCTAssertTrue(state.evidence.controlEventTimingInvalid)
        XCTAssertTrue(state.evidence.controlEvents.isEmpty)
    }
    func testControlObservationNeverChangesInputOutputOrPCMState() {
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.gateInput(true)
        state.recordControlEvent(.backendControlHandled, fields: ["event_type": "audio.webrtc.started"], at: 10)
        XCTAssertTrue(state.inputEnabled)
        XCTAssertFalse(state.outputEnabled)
        XCTAssertEqual(state.evidence.inputBlocks, 0)
        XCTAssertEqual(state.evidence.decodedBlocks, 0)
        XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
    }

    private func responsePlan(cuts: Int = 1, delay: Double = 0, timeout: Double = 2,
                              responseID: String? = nil, connectionID: UUID? = nil) throws -> JarvisReplayPlan {
        let seed = JarvisReplayPlan.Utterance(trigger: .atTime(0), pcm16: pcm([200]))
        return try JarvisReplayPlan(runID: UUID(), duration: 10, frames: [.init(at: 0, image: Data([1]))],
            utterances: [seed] + (0..<cuts).map { index in
                .init(trigger: .responseOutputStart(.init(afterInputOrdinal: index, delay: delay,
                    timeout: timeout, maxPCMAge: 0.05, responseID: responseID, connectionID: connectionID)),
                    pcm16: pcm([200]), clipID: "cut-\(index)")
            })
    }
    private func decoded(_ state: inout JarvisReplayPCMState, at time: Double, samples: [Int16] = [200]) {
        let identity = state.currentResponseIdentity
        state.observeDecoded(samples, at: time, callbackEnd: time + 0.001, responseIdentity: identity)
    }
    private func admitted(_ state: inout JarvisReplayPCMState, connection: UUID, response: String, at time: Double) {
        state.gateOutput(true, at: time, responseID: response, itemID: "item_\(response)", connectionID: connection)
    }

    func testResponseTriggerOneTwoThreeDistinctControlCorrelatedStops() throws {
        for cuts in 1...3 {
            let plan = try responsePlan(cuts: cuts)
            var state = JarvisReplayPCMState(maxDuration: 10)
            var timeline = JarvisReplayTimeline(plan: plan)
            try state.register(plan); state.gateInput(true)
            XCTAssertEqual(timeline.advance(elapsed: 0, output: nil, inputIdle: true).utterance, 0)
            try inject(&state, plan, ordinal: 0, at: 0.01)
            let connection = UUID()
            for index in 0..<cuts {
                let at = 0.5 + Double(index)
                admitted(&state, connection: connection, response: "resp_\(index)", at: at)
                decoded(&state, at: at + 0.01)
                let step = timeline.advance(elapsed: at + 0.02, output: state.latestOutput, inputIdle: state.inputIdle,
                    responseSnapshot: state.responseSnapshot(afterInputOrdinal: index))
                XCTAssertEqual(step.utterance, index + 1)
                let selected = try XCTUnwrap(step.responseSelection)
                try state.enqueue(plan.utterances[index + 1].pcm16, utteranceOrdinal: index + 1,
                    requestedAt: at + 0.021, responseSelection: selected)
                _ = state.renderInput(at: at + 0.03)
                state.inputCallbackStarted(at: at + 0.03); state.inputCallbackCompleted(status: 0, at: at + 0.031)
                decoded(&state, at: at + 0.04)
                state.recordLocalInterruption(.clientInterrupt, at: at + 0.05)
                state.gateOutput(false, at: at + 0.06)
                let bytes = state.evidence.decodedPCM16.count
                decoded(&state, at: at + 0.07) // Retired tail cannot create another trigger.
                XCTAssertEqual(state.evidence.decodedPCM16.count, bytes)
                XCTAssertNil(state.responseSnapshot(afterInputOrdinal: index + 1).output)
            }
            let measurements = JarvisReplayEvidenceReport.responseStopMeasurements(state.evidence)
            XCTAssertEqual(measurements.count, cuts)
            XCTAssertEqual(Set(state.evidence.localInterruptions.map { $0.identity.responseKey }).count, cuts)
            XCTAssertEqual(Set(state.evidence.localInterruptions.map { $0.identity.outputEpoch }).count, cuts)
            for item in measurements {
                XCTAssertEqual(try XCTUnwrap(item["input_to_local_gate_close_seconds"] as? Double), 0.03, accuracy: 0.000001)
                XCTAssertTrue(state.evidence.outputGates.contains { $0.eventID == item["output_gate_event_id"] as? String })
            }
            let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: plan.runID, finalUIStatus: "fixture")
            XCTAssertEqual(report["response_semantic_causality"] as? String, "unknown")
            XCTAssertNil(report["actual_cuts"])
            XCTAssertTrue(JSONSerialization.isValidJSONObject(report))
        }
    }

    func testResponseTriggerRequiresSDKCompletionAndRetainedActivePCM() throws {
        let plan = try responsePlan()
        var state = JarvisReplayPCMState(maxDuration: 1)
        try state.register(plan); state.gateInput(true)
        admitted(&state, connection: UUID(), response: "resp_a", at: 1)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).output)
        decoded(&state, at: 1.01, samples: [0, 0])
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).output)
        decoded(&state, at: 1.02)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).inputCompletion)
        try state.enqueue(plan.utterances[0].pcm16, utteranceOrdinal: 0, requestedAt: 1.03)
        _ = state.renderInput(at: 1.04); state.inputCallbackStarted(at: 1.04)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).inputCompletion)
        state.inputCallbackCompleted(status: 0, at: 1.041)
        XCTAssertEqual(state.responseSnapshot(afterInputOrdinal: 0).inputCompletion?.at, 1.041)
        XCTAssertNotNil(state.responseSnapshot(afterInputOrdinal: 0).output)
    }

    func testResponseTimeoutStartsAtActualFullSDKCompletionNotNominalStart() throws {
        let plan = try responsePlan(timeout: 2)
        var state = JarvisReplayPCMState(maxDuration: 10)
        var timeline = JarvisReplayTimeline(plan: plan)
        try state.register(plan); state.gateInput(true)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        XCTAssertNil(timeline.advance(elapsed: 3, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0)).failure)
        try inject(&state, plan, ordinal: 0, at: 4)
        let completion = try XCTUnwrap(state.responseSnapshot(afterInputOrdinal: 0).inputCompletion)
        XCTAssertNil(timeline.advance(elapsed: completion.at + 1.99, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0)).failure)
        admitted(&state, connection: UUID(), response: "resp_late", at: 5)
        decoded(&state, at: completion.at + 1.999)
        let step = timeline.advance(elapsed: completion.at + 2, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0))
        XCTAssertEqual(step.failure, .outputWaitExpired); XCTAssertNil(step.utterance)
    }

    func testResponseDelayUsesFirstRetainedPCMAndEnqueueRechecksFreshness() throws {
        let plan = try responsePlan(delay: 0.3)
        var state = JarvisReplayPCMState(maxDuration: 10)
        var timeline = JarvisReplayTimeline(plan: plan)
        try state.register(plan); state.gateInput(true)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        try inject(&state, plan, ordinal: 0, at: 0.01)
        admitted(&state, connection: UUID(), response: "resp_a", at: 0.1)
        decoded(&state, at: 0.2)
        XCTAssertNil(timeline.advance(elapsed: 0.49, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0)).utterance)
        decoded(&state, at: 0.5)
        let step = timeline.advance(elapsed: 0.51, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0))
        XCTAssertEqual(step.utterance, 1)
        XCTAssertThrowsError(try state.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1,
            requestedAt: 0.7, responseSelection: step.responseSelection)) {
            XCTAssertEqual($0 as? JarvisReplayError, .responseCandidateChanged)
        }
        XCTAssertNil(state.evidence.utterances[1].responseTrigger?.enqueueRevalidatedAt)
        XCTAssertEqual(state.evidence.utterances[1].acceptedSamples, 0)
    }

    func testResponseRetirementAndInFlightOldIdentityCannotLabelNewPCM() throws {
        let plan = try responsePlan()
        var state = JarvisReplayPCMState(maxDuration: 10)
        try state.register(plan); state.gateInput(true); try inject(&state, plan, ordinal: 0, at: 0.01)
        let connection = UUID()
        admitted(&state, connection: connection, response: "resp_old", at: 0.1)
        let old = try XCTUnwrap(state.currentResponseIdentity)
        state.gateOutput(false, at: 0.2)
        admitted(&state, connection: connection, response: "resp_new", at: 0.3)
        state.observeDecoded([200], at: 0.31, callbackEnd: 0.311, responseIdentity: old)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).output)
        XCTAssertNil(state.evidence.decodedSampleMap.last?.responseIdentity)
        decoded(&state, at: 0.32)
        XCTAssertEqual(state.responseSnapshot(afterInputOrdinal: 0).output?.identity.responseID, "resp_new")
        state.gateOutput(false, at: 0.4)
        admitted(&state, connection: connection, response: "resp_old", at: 0.5)
        XCTAssertNil(state.currentResponseIdentity)
    }

    func testResponseBindingChangeBetweenSelectionAndEnqueueFailsClosed() throws {
        let plan = try responsePlan()
        var state = JarvisReplayPCMState(maxDuration: 10)
        var timeline = JarvisReplayTimeline(plan: plan)
        try state.register(plan); state.gateInput(true)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        try inject(&state, plan, ordinal: 0, at: 0.01)
        let connection = UUID()
        admitted(&state, connection: connection, response: "resp_a", at: 0.1); decoded(&state, at: 0.2)
        let step = timeline.advance(elapsed: 0.21, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0))
        state.gateOutput(false, at: 0.22)
        admitted(&state, connection: connection, response: "resp_b", at: 0.23); decoded(&state, at: 0.24)
        XCTAssertThrowsError(try state.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1,
            requestedAt: 0.25, responseSelection: step.responseSelection))
        XCTAssertEqual(state.evidence.utterances[1].acceptedSamples, 0)
    }

    func testNaturalOutputClosureNeverBecomesAControlCorrelatedStop() throws {
        let plan = try responsePlan()
        var state = JarvisReplayPCMState(maxDuration: 10)
        var timeline = JarvisReplayTimeline(plan: plan)
        try state.register(plan); state.gateInput(true)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        try inject(&state, plan, ordinal: 0, at: 0.01)
        admitted(&state, connection: UUID(), response: "resp_a", at: 0.1); decoded(&state, at: 0.2)
        let step = timeline.advance(elapsed: 0.21, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0))
        try state.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1, requestedAt: 0.22,
                          responseSelection: step.responseSelection)
        _ = state.renderInput(at: 0.23); state.inputCallbackStarted(at: 0.23)
        state.inputCallbackCompleted(status: 0, at: 0.231)
        state.gateOutput(false, at: 0.24)
        XCTAssertTrue(state.evidence.localInterruptions.isEmpty)
        XCTAssertTrue(JarvisReplayEvidenceReport.responseStopMeasurements(state.evidence).isEmpty)
    }

    func testResponseExactConstraintsAndClosedDeviceRejectCandidates() throws {
        let connection = UUID(), foreign = UUID()
        let plan = try responsePlan(responseID: "resp_expected", connectionID: connection)
        var state = JarvisReplayPCMState(maxDuration: 10)
        var timeline = JarvisReplayTimeline(plan: plan)
        try state.register(plan); state.gateInput(true)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        try inject(&state, plan, ordinal: 0, at: 0.01)
        admitted(&state, connection: foreign, response: "resp_expected", at: 0.1); decoded(&state, at: 0.2)
        XCTAssertNil(timeline.advance(elapsed: 0.21, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0)).utterance)
        state.close(at: 0.22); decoded(&state, at: 0.23)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).output)
        XCTAssertThrowsError(try state.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1, requestedAt: 0.24))
    }

    func testResponseEvidenceCapacityAndPrivacyFailClosed() throws {
        var state = JarvisReplayPCMState(maxDuration: 0.01)
        state.gateOutput(true, at: 0, responseID: "private transcript", itemID: "Bearer secret", connectionID: UUID())
        XCTAssertNil(state.currentResponseIdentity)
        let connection = UUID()
        for index in 0...JarvisReplayPCMState.maximumResponseIdentities {
            admitted(&state, connection: connection, response: "resp_\(index)", at: Double(index))
            state.gateOutput(false, at: Double(index) + 0.1)
        }
        XCTAssertTrue(state.evidence.responseEvidenceTruncated)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).output)
        let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: UUID(), finalUIStatus: "fixture")
        let text = String(decoding: try JSONSerialization.data(withJSONObject: report), as: UTF8.self)
        XCTAssertFalse(text.contains("private transcript")); XCTAssertFalse(text.contains("Bearer secret"))
        var bounded = JarvisReplayPCMState(maxDuration: 0.01)
        admitted(&bounded, connection: UUID(), response: "resp_capacity", at: 0)
        decoded(&bounded, at: 0.1, samples: Array(repeating: 200, count: 960))
        XCTAssertTrue(bounded.evidence.truncated)
        XCTAssertNil(bounded.responseSnapshot(afterInputOrdinal: 0).output)
    }

    func testStrongRunLimitReportsMissingFullSDKInputInsteadOfNormalCompletion() throws {
        var timeline = JarvisReplayTimeline(plan: try responsePlan())
        XCTAssertEqual(timeline.advance(elapsed: 0, output: nil, inputIdle: true).utterance, 0)
        let end = timeline.advance(elapsed: 10, output: nil, inputIdle: true,
            responseSnapshot: JarvisReplayResponseSnapshot())
        XCTAssertEqual(end.failure, .inputCompletionUnavailable)
        XCTAssertEqual(end.failedUtterance, 1)
        XCTAssertNil(end.utterance)
        XCTAssertFalse(end.finished)
        XCTAssertTrue(timeline.advance(elapsed: 10.1, output: nil, inputIdle: true).finished)
    }

    func testStrongRunLimitAfterSDKCompletionReportsExpiryAndCannotRevive() throws {
        let plan = try responsePlan(timeout: 2)
        var timeline = JarvisReplayTimeline(plan: plan)
        var state = JarvisReplayPCMState(maxDuration: 10)
        try state.register(plan); state.gateInput(true)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        try inject(&state, plan, ordinal: 0, at: 9.2)
        let end = timeline.advance(elapsed: 10, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0))
        XCTAssertEqual(end.failure, .expired)
        XCTAssertEqual(end.failedUtterance, 1)
        XCTAssertFalse(end.finished)
        admitted(&state, connection: UUID(), response: "resp_after_run_limit", at: 10.1)
        decoded(&state, at: 10.11)
        let late = timeline.advance(elapsed: 10.12, output: nil, inputIdle: true,
            responseSnapshot: state.responseSnapshot(afterInputOrdinal: 0))
        XCTAssertTrue(late.finished)
        XCTAssertNil(late.utterance)
        XCTAssertNil(late.responseSelection)
    }
}

@MainActor
final class JarvisReplayManualInputTests: XCTestCase {
    private func fixture() throws -> (JarvisReplayPlan, JarvisReplayPCMState, JarvisReplayManualInputController, JarvisReplayManualSessionIdentity) {
        let plan = try JarvisReplayPlan(runID: UUID(), duration: 10, frames: [.init(at: 0, image: Data([1]))],
            utterances: [.init(trigger: .atTime(0), pcm16: Data([200, 0, 100, 0]))], invocationID: UUID())
        var state = JarvisReplayPCMState(maxDuration: 10); try state.register(plan)
        let identity = try JarvisReplayManualSessionIdentity(ownerID: "isolated-fixture", sessionID: UUID().uuidString,
            audioSessionID: UUID().uuidString, controlConnectionID: UUID(), segmentID: UUID().uuidString)
        return (plan, state, JarvisReplayManualInputController(runID: plan.runID, invocationID: plan.invocationID), identity)
    }
    private func write(_ plan: JarvisReplayPlan, _ identity: JarvisReplayManualSessionIdentity,
                       id: UUID, at: Double, type: String = "audio.input.commit") throws -> JarvisReplayManualWireObservation {
        try XCTUnwrap(.init(runID: plan.runID, connectionID: identity.controlConnectionID,
            payload: ["type": type, "segment_id": identity.segmentID.uuidString, "client_event_id": id.uuidString,
                      "text": "PRIVATE_TEXT", "authorization": "PRIVATE_TOKEN"],
            context: ["connection_generation": 4], at: at))
    }
    func testManualRequiresMatchingReadyThenActualSDKCompletionThenExactCommitWrite() throws {
        var (plan, state, driver, identity) = try fixture()
        var presses = 0, releases = 0, enqueues = 0
        try driver.begin(input: state.evidence.utterances[0], at: 10) { presses += 1; return identity }
        try driver.advance(inputs: state.evidence.utterances, at: 10.1, sceneDeadline: 20,
            enqueue: { _, _ in enqueues += 1 }, release: { _ in releases += 1 })
        XCTAssertEqual(presses, 1); XCTAssertEqual(enqueues, 0)
        XCTAssertFalse(driver.ready(segmentID: UUID().uuidString, connectionID: identity.controlConnectionID, generations: [:], at: 10.2))
        XCTAssertFalse(driver.ready(segmentID: identity.segmentID.uuidString, connectionID: UUID(), generations: [:], at: 10.2))
        XCTAssertTrue(driver.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                   generations: ["turn_generation": 7], at: 10.2))
        state.gateInput(true)
        try driver.advance(inputs: state.evidence.utterances, at: 10.3, sceneDeadline: 20, enqueue: { ordinal, triggered in
            enqueues += 1; try state.enqueue(plan.utterances[ordinal].pcm16, utteranceOrdinal: ordinal,
                                            requestedAt: 10.3, triggeredAt: triggered)
        }, release: { _ in releases += 1 })
        _ = state.renderInput(at: 10.4); state.inputCallbackStarted(at: 10.4)
        try driver.advance(inputs: state.evidence.utterances, at: 10.401, sceneDeadline: 20,
            enqueue: { _, _ in XCTFail("duplicate enqueue") }, release: { _ in releases += 1 })
        XCTAssertEqual(releases, 0)
        state.inputCallbackCompleted(status: 0, at: 10.402)
        try driver.advance(inputs: state.evidence.utterances, at: 10.41, sceneDeadline: 20,
            enqueue: { _, _ in XCTFail("duplicate enqueue") }, release: { value in
                XCTAssertEqual(value, identity); releases += 1; state.gateInput(false)
            })
        XCTAssertEqual(enqueues, 1); XCTAssertEqual(releases, 1); XCTAssertTrue(driver.isBusy)
        let clientID = UUID()
        var observation = try write(plan, identity, id: clientID, at: 10.42)
        driver.observe(observation); driver.observe(observation)
        observation.stage = .written; observation.at = 10.43
        driver.observe(observation); driver.observe(observation)
        XCTAssertFalse(driver.isBusy); XCTAssertNil(driver.failureCode)
        let rows = try XCTUnwrap(driver.json["occurrences"] as? [[String: Any]])
        XCTAssertEqual(rows[0]["commit_client_event_id"] as? String, clientID.uuidString.lowercased())
        XCTAssertEqual((rows[0]["wire_observations"] as? [[String: Any]])?.count, 2)
        XCTAssertTrue(rows[0]["canonical_input_id"] is NSNull)
        XCTAssertEqual(driver.json["full_chain_proven"] as? Bool, false)
        // Completed work may enter the existing media-end/tail path normally.
        try driver.advance(inputs: state.evidence.utterances, at: 20, sceneDeadline: 20,
            enqueue: { _, _ in XCTFail("enqueue after completion") },
            release: { _ in XCTFail("release after completion") })
        XCTAssertNil(driver.failureCode)
    }
    func testReadyTimeoutNeverEnqueuesOrReleasesAndRetirementCannotRevive() throws {
        let (_, state, driver, identity) = try fixture()
        try driver.begin(input: state.evidence.utterances[0], at: 10) { identity }
        XCTAssertFalse(driver.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                    generations: [:], at: 15))
        XCTAssertThrowsError(try driver.advance(inputs: state.evidence.utterances, at: 15, sceneDeadline: 20,
            enqueue: { _, _ in XCTFail("timeout enqueue") }, release: { _ in XCTFail("timeout release") }))
        driver.retire()
        XCTAssertFalse(driver.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                    generations: [:], at: 15.1))
    }
    func testCommitBeforeSDKCompletionDoesNotBecomeAProof() throws {
        let (plan, state, driver, identity) = try fixture()
        try driver.begin(input: state.evidence.utterances[0], at: 10) { identity }
        driver.observe(try write(plan, identity, id: UUID(), at: 10.1))
        XCTAssertEqual(driver.failureCode, "manual_commit_before_sdk_completion")
        XCTAssertEqual(driver.json["full_chain_proven"] as? Bool, false)
    }
    func testFailedSDKInputNeverCallsNormalRelease() throws {
        var (plan, state, driver, identity) = try fixture()
        try driver.begin(input: state.evidence.utterances[0], at: 10) { identity }
        XCTAssertTrue(driver.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID, generations: [:], at: 10.1))
        state.gateInput(true)
        try driver.advance(inputs: state.evidence.utterances, at: 10.2, sceneDeadline: 20, enqueue: { ordinal, triggered in
            try state.enqueue(plan.utterances[ordinal].pcm16, utteranceOrdinal: ordinal, requestedAt: 10.2, triggeredAt: triggered)
        }, release: { _ in XCTFail("early release") })
        _ = state.renderInput(at: 10.3); state.inputCallbackStarted(at: 10.3); state.inputCallbackCompleted(status: -1, at: 10.301)
        XCTAssertThrowsError(try driver.advance(inputs: state.evidence.utterances, at: 10.31, sceneDeadline: 20,
            enqueue: { _, _ in XCTFail("duplicate") }, release: { _ in XCTFail("failed input release") }))
    }
    func testExactSceneDeadlineAndDelayedWakeCannotEnqueueReadyManualInput() throws {
        for wake in [12.0, 12.75] {
            let (_, state, driver, identity) = try fixture()
            try driver.begin(input: state.evidence.utterances[0], at: 10) { identity }
            XCTAssertTrue(driver.ready(segmentID: identity.segmentID.uuidString,
                connectionID: identity.controlConnectionID, generations: [:], at: 10.1))
            var enqueues = 0, releases = 0
            XCTAssertThrowsError(try driver.advance(inputs: state.evidence.utterances, at: wake,
                sceneDeadline: 12, enqueue: { _, _ in enqueues += 1 }, release: { _ in releases += 1 })) {
                    XCTAssertEqual($0 as? JarvisReplayError, .manualInputIncomplete)
                }
            XCTAssertEqual(enqueues, 0); XCTAssertEqual(releases, 0)
            XCTAssertEqual(driver.failureCode, "manual_scene_deadline")
            XCTAssertEqual(state.evidence.utterances[0].acceptedSamples, 0)
            let row = try XCTUnwrap((driver.json["occurrences"] as? [[String: Any]])?.first)
            XCTAssertTrue(row["enqueue_uptime_seconds"] is NSNull)
            XCTAssertTrue(row["release_uptime_seconds"] is NSNull)
            XCTAssertTrue(row["commit_client_event_id"] is NSNull)
            XCTAssertEqual(driver.json["full_chain_proven"] as? Bool, false)
        }
    }
    func testExactSceneDeadlineAndDelayedWakeCannotReleaseCompletedSDKInput() throws {
        for wake in [12.0, 12.75] {
            var (plan, state, driver, identity) = try fixture()
            try driver.begin(input: state.evidence.utterances[0], at: 10) { identity }
            XCTAssertTrue(driver.ready(segmentID: identity.segmentID.uuidString,
                connectionID: identity.controlConnectionID, generations: [:], at: 10.1))
            state.gateInput(true)
            try driver.advance(inputs: state.evidence.utterances, at: 10.2, sceneDeadline: 12,
                enqueue: { ordinal, triggered in
                    try state.enqueue(plan.utterances[ordinal].pcm16, utteranceOrdinal: ordinal,
                                      requestedAt: 10.2, triggeredAt: triggered)
                }, release: { _ in XCTFail("release before SDK completion") })
            _ = state.renderInput(at: 11.9)
            state.inputCallbackStarted(at: 11.9)
            state.inputCallbackCompleted(status: 0, at: 11.901)
            XCTAssertEqual(state.evidence.utterances[0].status, "sdk_input_complete")
            var enqueues = 0, releases = 0
            XCTAssertThrowsError(try driver.advance(inputs: state.evidence.utterances, at: wake,
                sceneDeadline: 12, enqueue: { _, _ in enqueues += 1 }, release: { _ in releases += 1 })) {
                    XCTAssertEqual($0 as? JarvisReplayError, .manualInputIncomplete)
                }
            XCTAssertEqual(enqueues, 0); XCTAssertEqual(releases, 0)
            XCTAssertEqual(driver.failureCode, "manual_scene_deadline")
            // Keep the actual SDK evidence; do not erase it or invent a commit.
            XCTAssertEqual(state.evidence.utterances[0].acceptedSamples, 2)
            XCTAssertEqual(state.evidence.utterances[0].lastCallbackEnd, 11.901)
            let row = try XCTUnwrap((driver.json["occurrences"] as? [[String: Any]])?.first)
            XCTAssertTrue(row["release_uptime_seconds"] is NSNull)
            XCTAssertTrue(row["commit_client_event_id"] is NSNull)
        }
    }
    func testActualWireProjectionRetainsGeneratedIDAndDropsContents() throws {
        let (plan, _, _, identity) = try fixture()
        let id = UUID(), value = try write(plan, identity, id: id, at: 10)
        XCTAssertEqual(value.clientEventID, id); XCTAssertEqual(value.generations, ["connection_generation": 4])
        let text = String(decoding: try JSONSerialization.data(withJSONObject: value.json), as: UTF8.self)
        XCTAssertFalse(text.contains("PRIVATE")); XCTAssertTrue(value.json["canonical_input_id"] is NSNull)
        XCTAssertNil(JarvisReplayManualWireObservation(runID: plan.runID, connectionID: identity.controlConnectionID,
            payload: ["type": "audio.input.append", "audio_base64": "PRIVATE"], context: [:], at: 10))
    }
    func testWrongConnectionAndRetiredObservationsCannotSettlePendingInput() throws {
        let (plan, state, driver, identity) = try fixture()
        try driver.begin(input: state.evidence.utterances[0], at: 10) { identity }
        let wrong = try JarvisReplayManualSessionIdentity(ownerID: identity.ownerID, sessionID: identity.sessionID.uuidString,
            audioSessionID: identity.audioSessionID.uuidString, controlConnectionID: UUID(), segmentID: identity.segmentID.uuidString)
        driver.observe(try write(plan, wrong, id: UUID(), at: 10.1))
        XCTAssertNil(driver.failureCode); XCTAssertTrue(driver.isBusy)
        driver.retire()
        driver.observe(try write(plan, identity, id: UUID(), at: 10.2))
        XCTAssertEqual(driver.failureCode, "manual_retired_before_commit")
    }
}


// Controlled, offline frontend contract tests. Host-ready objects below are
// fixtures, NOT actual SQL/source authority or native/provider acceptance.
@MainActor
final class JarvisReplayMemoryReadinessTests: XCTestCase {
    private var roots: [URL] = []
    private let run = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
    private let fixtureInvocationID = UUID(uuidString: "22222222-2222-4222-8222-222222222222")!
    private let area = "33333333-3333-4333-8333-333333333333"
    private let fixtureManifestSHA256 = String(repeating: "a", count: 64)
    private var runHex: String { run.uuidString.replacingOccurrences(of: "-", with: "").lowercased() }
    private var owner: String { "direct-image-eval-second-brain-" + runHex + "-a" }
    private var scope: [String: String] {
        ["run_id": runHex, "owner_id": owner, "database_schema": "jarvis_sb_eval_" + runHex + "_0",
         "area_id": area, "memory_bank_id": "jarvis_" + String(repeating: "b", count: 32)]
    }
    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots = []
        super.tearDown()
    }
    private func root() throws -> URL {
        let value = FileManager.default.temporaryDirectory.appendingPathComponent("memory-gate-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: value, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        for name in ["inputs", "controls"] {
            try FileManager.default.createDirectory(at: value.appendingPathComponent(name),
                withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        }
        roots.append(value); return value
    }
    private func write(_ data: Data, to url: URL) throws {
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
    private func config(_ root: URL) throws -> JarvisReplayMemoryConfiguration {
        try .load(scopeData: JarvisReplayTaskJSON.encode(scope), manifestSHA256: fixtureManifestSHA256,
            inputsRoot: root.appendingPathComponent("inputs"), runID: run, ownerID: owner,
            areaID: area, bankID: scope["memory_bank_id"]!)
    }
    private func trigger(_ canonical: String = "finalized_input", playback: String = "authenticated_buffer_stop") -> JarvisReplayManualResponseTrigger {
        .init(afterInputOrdinal: 0, at: 15, timeout: 22, canonical: canonical, playback: playback)
    }
    private func fixture(_ root: URL) throws -> (JarvisReplayMemoryConfiguration, JarvisReplayManualCommit,
                                                JarvisReplayMemoryCandidate, JarvisReplayAudioEvidence) {
        let configuration = try config(root)
        let identity = try JarvisReplayManualSessionIdentity(ownerID: owner, sessionID: UUID().uuidString,
            audioSessionID: UUID().uuidString, controlConnectionID: UUID(), segmentID: UUID().uuidString)
        let commit = JarvisReplayManualCommit(ordinal: 0, identity: identity, clientEventID: UUID())
        var state = JarvisReplayPCMState(maxDuration: 95)
        let media = UUID()
        state.recordControlEvent(.backendControlHandled, fields: [
            "event_type": "audio.webrtc.response", "response_id": "resp_final", "connection_id": media.uuidString.lowercased(),
            "control_connection_id": identity.controlConnectionID.uuidString.lowercased(),
            "audio_session_id": identity.audioSessionID.uuidString.lowercased(), "turn_generation": 1], at: 110)
        state.gateOutput(true, at: 111, responseID: "resp_final", itemID: "item_final", connectionID: media,
            diagnostics: diagnostic("final_answer"))
        state.recordControlEvent(.backendControlHandled, fields: ["event_type": "audio.webrtc.stopped",
            "response_id": "resp_final", "connection_id": media.uuidString.lowercased()], at: 112)
        state.gateOutput(false, at: 112.1)
        let candidate = try XCTUnwrap(JarvisReplayMemoryCandidate.proposals(state.evidence, commit: commit).first)
        return (configuration, commit, candidate, state.evidence)
    }
    private func diagnostic(_ phase: String, type: String = "message") -> JarvisReplayPlaybackGateDiagnostics {
        .init(origin: .speakerPredicate, remoteDescriptionPresent: true, permittedIdentityPresent: true,
            playableIdentityPresent: true, playableIdentityMatchesPermit: true, externalPlaybackInactive: true,
            routeOutputHeld: false, receiverPresent: true, replayItemIdentityPresent: true,
            itemMatchesPermittedResponse: true,
            item: .init(item: ["type": type, "phase": phase], policyAllowsPlayback: type == "message" && phase != "analysis"))
    }
    private func request(_ root: URL, requirement: String = "finalized_input") throws -> JarvisReplayMemoryRequest {
        let (configuration, commit, candidate, _) = try fixture(root)
        return try .init(configuration: configuration, invocationID: fixtureInvocationID, target: 1,
            trigger: trigger(requirement), commit: commit, candidate: candidate,
            issued: Date(timeIntervalSince1970: 1_700_000_015), expires: Date(timeIntervalSince1970: 1_700_000_037))
    }
    // Structural copy of the scoped host's additive receipt object. All
    // identities, hashes, and times are synthetic; no native data or secrets.
    private var hostSourceClock: [String: Any] {
        let deadline = 124876.80179525001, sampled = 124864.861197125
        return ["clock": "ProcessInfo.systemUptime", "boot_session_id": fixtureInvocationID.uuidString.lowercased(),
            "phase_start_seconds": 0, "arm_uptime_seconds": 124854.80179525001,
            "deadline_uptime_seconds": deadline, "sampled_uptime_seconds": sampled,
            "remaining_seconds": deadline - sampled, "status": "open", "anchor_sha256": fixtureManifestSHA256]
    }
    private func receipt(_ request: JarvisReplayMemoryRequest) -> [String: Any] {
        var value = request.fields
        value.removeValue(forKey: "scope"); value.removeValue(forKey: "issued_at")
        value["format"] = "jarvis.native_memory_readiness.v1"; value["run_id"] = runHex; value["owner_id"] = owner
        value["full_chain_proven"] = false; value["heard_state"] = "unknown"; value["playback"] = "not_evaluated"
        value["semantic_acceptance"] = "not_evaluated"; value["status"] = "ready"
        value["source_clock"] = hostSourceClock
        value["reason"] = "finalized_input_and_terminal_response"
        value["canonical_input"] = ["event_id": UUID().uuidString, "payload_id": UUID().uuidString, "revision": 2,
            "source_namespace": "voice:actual_scope", "source_event_id": "input:actual_source",
            "provider_item_id": "item_input"] as [String: Any]
        value["response"] = ["event_id": UUID().uuidString, "revision": 3,
            "response_id": request.candidate.identity.responseID, "item_id": request.candidate.identity.itemID,
            "completion_state": "completed"] as [String: Any]
        value["final_output_policy"] = ["response_kind": "normal_user_turn",
            "audio_session_id": request.commit.identity.audioSessionID.uuidString.lowercased(),
            "realtime_output": ["item_id": request.candidate.identity.itemID, "phase": "final_answer",
                               "phase_anomaly": false, "delivery_disposition": "forwarded"]] as [String: Any]
        value["observer"] = ["queries": 9, "elapsed_seconds": 0.1, "read_only": true,
                            "isolation": "repeatable_read", "automatic_polling": false] as [String: Any]
        value["admission_evidence"] = ["path": "inline/admission.json", "sha256": fixtureManifestSHA256, "selector": "$"]
        if request.fields["canonical_requirement"] as? String == "retain_searchable" {
            value["reason"] = "exact_remember_searchable_and_continued"
            value["continuation_basis"] = "admitted_response_source_dependencies_not_tool_trigger_or_model_use"
            value["continuation_evidence"] = ["path": "inline/continuation.json", "sha256": fixtureManifestSHA256, "selector": "$"]
            value["retained_source"] = ["owner_id": owner, "area_id": area, "source_id": UUID().uuidString,
                "revision": 1, "generation": 1, "payload_id": UUID().uuidString, "document_id": "source-doc",
                "content_digest": fixtureManifestSHA256, "retain_operation_id": UUID().uuidString, "status": "searchable"] as [String: Any]
        }
        return ["format": "jarvis.native_memory_readiness.receipt.v1", "request_sha256": request.sha256,
            "recorded_at": JarvisReplayTaskJSON.timestamp(request.issued.addingTimeInterval(0.1)), "readiness": value]
    }
    private func parse(_ value: [String: Any], request: JarvisReplayMemoryRequest,
                       now: Date? = nil) throws -> [String: Any]? {
        try JarvisReplayMemoryReceipt.ready(JarvisReplayTaskJSON.encode(value, limit: 32768),
            request: request, now: now ?? request.issued.addingTimeInterval(1))
    }

    func testCanonicalFinalReceiptPreservesActualEventRevisionAndNonProof() throws {
        let req = try request(root())
        let output = try XCTUnwrap(parse(receipt(req), request: req))
        XCTAssertEqual((output["canonical_input"] as? [String: Any])?["revision"] as? Int, 2)
        XCTAssertEqual((output["response"] as? [String: Any])?["revision"] as? Int, 3)
        XCTAssertEqual(output["full_chain_proven"] as? Bool, false)
        XCTAssertEqual(output["heard_state"] as? String, "unknown")
        XCTAssertNil(output["inline_evidence"])
    }
    func testHostSourceClockShapeAcceptedWithoutChangingCanonicalPolicyOrProof() throws {
        let req = try request(root(), requirement: "retain_searchable")
        for phase in [0.0, 115.0] {
            var returned = receipt(req), body = returned["readiness"] as! [String: Any]
            var clock = hostSourceClock; clock["phase_start_seconds"] = phase
            body["source_clock"] = clock
            // Python's equivalent UTC serialization is accepted as an instant,
            // not used to replace the original local monotonic deadline.
            body["expires_at"] = "2023-11-14T22:13:57.000000+00:00"
            returned["readiness"] = body
            let output = try XCTUnwrap(parse(returned, request: req))
            let projected = try XCTUnwrap(output["source_clock"] as? [String: Any])
            XCTAssertEqual(try JarvisReplayTaskJSON.encode(projected), try JarvisReplayTaskJSON.encode(clock))
            XCTAssertEqual((output["canonical_input"] as? [String: Any])?["revision"] as? Int, 2)
            XCTAssertEqual((output["response"] as? [String: Any])?["revision"] as? Int, 3)
            XCTAssertEqual((output["final_output_policy"] as? [String: Any])?["response_kind"] as? String, "normal_user_turn")
            XCTAssertEqual(output["full_chain_proven"] as? Bool, false)
            XCTAssertEqual(output["heard_state"] as? String, "unknown")
            XCTAssertNil(output["inline_evidence"])
        }
    }
    func testHostSourceClockMissingExtraAndMalformedFieldsFailClosed() throws {
        let req = try request(root())
        func rejected(_ clock: Any) throws {
            var returned = receipt(req), body = returned["readiness"] as! [String: Any]
            body["source_clock"] = clock; returned["readiness"] = body
            XCTAssertThrowsError(try parse(returned, request: req))
        }
        for key in hostSourceClock.keys {
            var clock = hostSourceClock; clock.removeValue(forKey: key); try rejected(clock)
        }
        var extra = hostSourceClock; extra["arbitrary_payload"] = "must-not-be-retained"; try rejected(extra)
        for (key, replacement) in [("clock", "wall_clock"), ("status", "expired"),
                                    ("boot_session_id", "not-a-uuid"), ("anchor_sha256", "not-a-hash")] {
            var clock = hostSourceClock; clock[key] = replacement; try rejected(clock)
        }
        try rejected(NSNull()); try rejected("not-an-object")
        var returned = receipt(req), body = returned["readiness"] as! [String: Any]
        body["another_unknown_field"] = true; returned["readiness"] = body
        XCTAssertThrowsError(try parse(returned, request: req))
    }
    func testHostSourceClockRejectsInvalidNumbersAndInconsistentOpenInterval() throws {
        let numeric = ["phase_start_seconds", "arm_uptime_seconds", "deadline_uptime_seconds",
                       "sampled_uptime_seconds", "remaining_seconds"]
        for key in numeric {
            for replacement: Any in [true, "115", -1.0, Double.nan, Double.infinity, NSNull()] {
                var clock = hostSourceClock; clock[key] = replacement
                XCTAssertThrowsError(try JarvisReplayMemoryReceipt.validatedSourceClock(clock))
            }
        }
        let arm = hostSourceClock["arm_uptime_seconds"] as! Double
        let deadline = hostSourceClock["deadline_uptime_seconds"] as! Double
        for (key, replacement) in [("phase_start_seconds", 180.001), ("deadline_uptime_seconds", arm),
                                    ("deadline_uptime_seconds", arm + 120.01), ("sampled_uptime_seconds", arm - 0.01),
                                    ("sampled_uptime_seconds", deadline), ("remaining_seconds", 0.0),
                                    ("remaining_seconds", 121.0), ("remaining_seconds", 1.0)] {
            var clock = hostSourceClock; clock[key] = replacement
            XCTAssertThrowsError(try JarvisReplayMemoryReceipt.validatedSourceClock(clock))
        }
    }
    func testHostSourceClockCannotRenewLocalDeadlineOrConsumeFreshness() throws {
        let (config, prior, candidate, evidence) = try fixture(root())
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100,
            utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        func poll(_ now: Double, consume: Bool) throws {
            try gate.poll(target: 1, trigger: trigger(), commit: prior, evidence: evidence,
                          clock: clock, now: now, currentScope: true, consume: consume)
        }
        try poll(115, consume: false)
        let probe = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 115.1)
        try poll(115.2, consume: false); try gate.markPressed(target: 1, at: 115.21)
        try poll(116, consume: true)
        let fresh = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 116.1)
        XCTAssertNotEqual(probe.nonce, fresh.nonce)
        XCTAssertEqual(probe.issued, fresh.issued); XCTAssertEqual(probe.expires, fresh.expires)
        try poll(116.2, consume: true)
        XCTAssertFalse(try gate.consume(target: 1, now: 116.451, currentScope: true))
        // Host observation has a much larger uptime; it cannot renew local 137.
        XCTAssertThrowsError(try poll(137, consume: true)) {
            XCTAssertEqual($0 as? JarvisReplayError, .outputWaitExpired)
        }
        XCTAssertFalse(try gate.consume(target: 1, now: 137, currentScope: true))
        XCTAssertTrue(gate.settlements.isEmpty)
        XCTAssertEqual(gate.status, "deadline"); gate.stop()
    }
    func testRetainSearchableRequiresActualSourceAndContinuation() throws {
        let req = try request(root(), requirement: "retain_searchable")
        let original = receipt(req)
        XCTAssertNotNil(try parse(original, request: req))
        for field in ["retained_source", "continuation_basis", "continuation_evidence"] {
            var bad = original, body = original["readiness"] as! [String: Any]
            body.removeValue(forKey: field); bad["readiness"] = body
            if field == "continuation_evidence" {
                XCTAssertThrowsError(try parse(bad, request: req)) {
                    XCTAssertEqual($0 as? JarvisReplayError, .memoryReadinessUnavailable)
                }
            } else {
                XCTAssertNil(try parse(bad, request: req))
            }
        }
    }
    func testMissingCommentarySuppressedAnomalousAndWrongPolicyCannotReady() throws {
        let req = try request(root())
        let original = receipt(req)
        for (key, replacement) in [("phase", "commentary" as Any), ("phase", "analysis"), ("item_id", "item_other"),
                                    ("delivery_disposition", "suppressed"), ("phase_anomaly", true), ("phase_anomaly", 0)] {
            var bad = original, body = original["readiness"] as! [String: Any]
            var policy = body["final_output_policy"] as! [String: Any]
            var item = policy["realtime_output"] as! [String: Any]
            item[key] = replacement; policy["realtime_output"] = item; body["final_output_policy"] = policy
            bad["readiness"] = body
            XCTAssertNil(try parse(bad, request: req))
        }
        for key in ["final_output_policy", "canonical_input", "response"] {
            var bad = original, body = original["readiness"] as! [String: Any]
            body.removeValue(forKey: key); bad["readiness"] = body
            XCTAssertNil(try parse(bad, request: req))
        }
    }
    func testStaleNonceScopeHashAndDeadlineCannotReady() throws {
        let req = try request(root())
        let original = receipt(req)
        for key in ["nonce", "connection_id", "session_id", "segment_id", "client_event_id", "item_id", "response_id", "manifest_sha256"] {
            var bad = original, body = original["readiness"] as! [String: Any]
            body[key] = "wrong"; bad["readiness"] = body
            XCTAssertNil(try parse(bad, request: req))
        }
        var bad = original; bad["request_sha256"] = String(repeating: "f", count: 64)
        XCTAssertNil(try parse(bad, request: req))
        XCTAssertNil(try parse(original, request: req, now: req.expires))
        XCTAssertNil(try parse(original, request: req, now: req.expires.addingTimeInterval(0.01)))
    }
    func testLiteralRequestBytesHashIsNotReserializedReceipt() throws {
        let req = try request(root())
        let value = try JarvisReplayTaskJSON.object(req.bytes, limit: 16384)
        let pretty = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        XCTAssertNotEqual(JarvisReplayTaskJSON.hash(pretty), req.sha256)
        var wrong = receipt(req); wrong["request_sha256"] = JarvisReplayTaskJSON.hash(pretty)
        XCTAssertNil(try parse(wrong, request: req))
    }
    func testDuplicateUnknownAndNonfiniteReceiptRejects() throws {
        let req = try request(root())
        var value = receipt(req); value["arbitrary_payload"] = "DO_NOT_RETAIN"
        XCTAssertThrowsError(try parse(value, request: req))
        let data = try JarvisReplayTaskJSON.encode(receipt(req), limit: 32768)
        let text = String(decoding: data, as: UTF8.self)
        let duplicate = Data(("{\"format\":\"duplicate\"," + text.dropFirst()).utf8)
        XCTAssertThrowsError(try JarvisReplayMemoryReceipt.ready(duplicate, request: req, now: req.issued))
        XCTAssertThrowsError(try JarvisReplayTaskJSON.object(Data("{\"x\":NaN}".utf8), limit: 32768))
    }
    func testActualScopeRequiredNoDerivationAndAuthorizedPreparedHashPreserved() throws {
        let directory = try root()
        let good = try config(directory)
        XCTAssertEqual(good.scope["database_schema"], scope["database_schema"])
        XCTAssertEqual(good.manifestSHA256, fixtureManifestSHA256)
        for key in ["run_id", "owner_id", "database_schema", "area_id", "memory_bank_id"] {
            var bad = scope; bad.removeValue(forKey: key)
            XCTAssertThrowsError(try JarvisReplayMemoryConfiguration.load(scopeData: JarvisReplayTaskJSON.encode(bad),
                manifestSHA256: fixtureManifestSHA256, inputsRoot: directory.appendingPathComponent("inputs"),
                runID: run, ownerID: owner, areaID: area, bankID: scope["memory_bank_id"]!))
        }
    }
    func testPrivateControlsMissingSymlinkOversizeReplacementAndRetirement() throws {
        let directory = try root().appendingPathComponent("controls")
        let io = try JarvisReplayMemoryFiles(directory)
        XCTAssertNil(try io.read(JarvisReplayMemoryFiles.receiptName))
        try io.publish(Data("first".utf8)); try io.publish(Data("replacement".utf8))
        XCTAssertEqual(try io.read(JarvisReplayMemoryFiles.requestName), Data("replacement".utf8))
        let receiptURL = directory.appendingPathComponent(JarvisReplayMemoryFiles.receiptName)
        try FileManager.default.createSymbolicLink(at: receiptURL,
            withDestinationURL: directory.appendingPathComponent(JarvisReplayMemoryFiles.requestName))
        XCTAssertThrowsError(try io.read(JarvisReplayMemoryFiles.receiptName))
        try FileManager.default.removeItem(at: receiptURL)
        try write(Data(repeating: 0, count: 32769), to: receiptURL)
        XCTAssertThrowsError(try io.read(JarvisReplayMemoryFiles.receiptName))
        io.close(); XCTAssertThrowsError(try io.publish(Data("retired".utf8)))
    }
    func testAuthenticatedStopRequiresActualCurrentItemNotResponseOnlyHistory() throws {
        let directory = try root()
        let (_, commit, candidate, evidence) = try fixture(directory)
        XCTAssertNotNil(candidate.stop)
        var absent = evidence
        absent.controlEvents = evidence.controlEvents.filter { $0.fields["event_type"] != "audio.webrtc.stopped" }
        XCTAssertNil(JarvisReplayMemoryCandidate.proposals(absent, commit: commit).first?.stop)
        var wrong = evidence
        wrong.controlEvents = evidence.controlEvents.map { event in
            var copy = event; copy.boundResponseIdentity = nil; return copy
        }
        XCTAssertNil(JarvisReplayMemoryCandidate.proposals(wrong, commit: commit).first?.stop)
        var changed = JarvisReplayPCMState(maxDuration: 95)
        let media = candidate.identity.connectionID
        changed.gateOutput(true, at: 1, responseID: "resp_final", itemID: "item_commentary", connectionID: media)
        changed.gateOutput(true, at: 2, responseID: "resp_final", itemID: "item_final", connectionID: media)
        changed.recordControlEvent(.backendControlHandled, fields: ["event_type": "audio.webrtc.stopped",
            "response_id": "resp_final", "connection_id": media.uuidString.lowercased()], at: 3)
        XCTAssertNil(changed.evidence.controlEvents.last?.boundResponseIdentity)
    }
    func testCommentaryNeverProposesFinalAndConnectionResetFences() throws {
        let (_, commit, _, evidence) = try fixture(root())
        var commentary = evidence
        commentary.outputGates = evidence.outputGates.map { value in
            var copy = value; copy.playbackDiagnostics = diagnostic("commentary"); return copy
        }
        XCTAssertTrue(JarvisReplayMemoryCandidate.proposals(commentary, commit: commit).isEmpty)
        var retired = evidence
        retired.controlEvents.append(.init(stage: .connectionReset, handledAt: 120, receivedAt: nil, fields: [:]))
        XCTAssertTrue(JarvisReplayMemoryCandidate.proposals(retired, commit: commit).isEmpty)
    }
    func testProbeThenNewConsumeNonceWithSameTimesAndOneEnqueueOnly() throws {
        let directory = try root()
        let (config, commit, candidate, evidence) = try fixture(directory)
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        func poll(_ time: Double, consume: Bool) throws {
            try gate.poll(target: 1, trigger: trigger(), commit: commit, evidence: evidence,
                          clock: clock, now: time, currentScope: true, consume: consume)
        }
        func answer(recorded: Double) throws -> [String: Any] {
            let io = try JarvisReplayMemoryFiles(config.controlsURL)
            let bytes = try XCTUnwrap(io.read(JarvisReplayMemoryFiles.requestName)); io.close()
            let body = try JarvisReplayTaskJSON.object(bytes, limit: 16384)["request"] as! [String: Any]
            let req = try JarvisReplayMemoryRequest(configuration: config, invocationID: fixtureInvocationID, target: 1,
                trigger: trigger(), commit: commit, candidate: candidate,
                issued: JarvisReplayTaskJSON.time(body["issued_at"]), expires: JarvisReplayTaskJSON.time(body["expires_at"]),
                nonce: JarvisReplayTaskJSON.uuid(body["nonce"]))
            XCTAssertEqual(bytes, req.bytes)
            var returned = receipt(req)
            returned["recorded_at"] = JarvisReplayTaskJSON.timestamp(clock.utc.addingTimeInterval(recorded - 100))
            try write(JarvisReplayTaskJSON.encode(returned, limit: 32768),
                      to: config.controlsURL.appendingPathComponent(JarvisReplayMemoryFiles.receiptName))
            return body
        }
        try poll(115, consume: false)
        XCTAssertFalse(gate.probeReady)
        let first = try answer(recorded: 115.1)
        try poll(115.2, consume: false)
        XCTAssertTrue(gate.mayPress(target: 1, now: 115.21, currentScope: true))
        XCTAssertFalse(try gate.consume(target: 1, now: 115.21, currentScope: true))
        try gate.markPressed(target: 1, at: 115.22)
        try poll(115.3, consume: true)
        XCTAssertFalse(gate.probeReady)
        let second = try answer(recorded: 115.4)
        XCTAssertNotEqual(first["nonce"] as? String, second["nonce"] as? String)
        XCTAssertEqual(first["issued_at"] as? String, second["issued_at"] as? String)
        XCTAssertEqual(first["expires_at"] as? String, second["expires_at"] as? String)
        try poll(115.5, consume: true)
        XCTAssertFalse(try gate.consume(target: 1, now: 115.51, currentScope: false))
        XCTAssertTrue(try gate.consume(target: 1, now: 115.51, currentScope: true))
        XCTAssertFalse(try gate.consume(target: 1, now: 115.52, currentScope: true))
        XCTAssertEqual(gate.settlements.count, 1)
        XCTAssertEqual(gate.settlements[0]["decoded_drain_confirmed"] as? Bool, false)
        gate.stop()
        XCTAssertThrowsError(try poll(116, consume: true))
    }
    func testMissingHostReceiptNeverEnqueuesAndDeadlineWinsDelayedWake() throws {
        let directory = try root()
        let (config, commit, _, evidence) = try fixture(directory)
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        try gate.poll(target: 1, trigger: trigger(), commit: commit, evidence: evidence, clock: clock,
                      now: 115, currentScope: true, consume: false)
        XCTAssertFalse(gate.probeReady)
        for time in [137.0, 139.0] {
            XCTAssertThrowsError(try gate.poll(target: 1, trigger: trigger(), commit: commit, evidence: evidence,
                clock: clock, now: time, currentScope: true, consume: false))
            XCTAssertFalse(try gate.consume(target: 1, now: time, currentScope: true))
        }
        gate.stop()
    }
    func testPollingDeadlineRecordsOnlyPendingOrdinalBeforeTeardown() throws {
        for time in [137.0, 139.0] {
            let directory = try root()
            let (configuration, commit, _, evidence) = try fixture(directory)
            let gate = try JarvisReplayMemoryGate(configuration: configuration, invocationID: fixtureInvocationID)
            let clock = JarvisReplayClockAnchor(monotonicBefore: 100,
                utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
            let plan = try JarvisReplayPlan(runID: run, duration: 95,
                frames: [.init(at: 0, image: Data([1]))], utterances: [
                    .init(trigger: .atTime(5), pcm16: Data([1, 0])),
                    .init(trigger: .afterManualResponse(trigger()), pcm16: Data([1, 0])),
                    .init(trigger: .afterManualResponse(.init(afterInputOrdinal: 1, at: 53, timeout: 20,
                        canonical: "retain_searchable", playback: "authenticated_buffer_stop")), pcm16: Data([1, 0]))
                ], invocationID: fixtureInvocationID)
            var state = JarvisReplayPCMState(maxDuration: 95)
            try state.register(plan); state.gateInput(true)
            try state.enqueue(plan.utterances[0].pcm16, utteranceOrdinal: 0, requestedAt: 105, triggeredAt: 105)
            _ = state.renderInput(at: 105.01)
            state.inputCallbackStarted(at: 105.02)
            state.inputCallbackCompleted(status: 0, at: 105.03)
            var timeline = JarvisReplayTimeline(plan: plan)
            XCTAssertEqual(timeline.advance(elapsed: 5, output: nil, inputIdle: true).utterance, 0)
            let target = try XCTUnwrap(timeline.pendingUtteranceOrdinal)
            XCTAssertEqual(target, 1)
            state.armUtterance(target, at: 115)
            try gate.poll(target: target, trigger: trigger(), commit: commit, evidence: evidence,
                          clock: clock, now: 115, currentScope: true, consume: false)
            XCTAssertThrowsError(try gate.poll(target: target, trigger: trigger(), commit: commit,
                evidence: evidence, clock: clock, now: time, currentScope: true, consume: false)) { error in
                XCTAssertEqual(error as? JarvisReplayError, .outputWaitExpired)
                gate.recordPollingFailure(error, target: target, at: time) { ordinal, code, observed in
                    state.failUtterance(ordinal, code: code, at: observed)
                }
            }
            XCTAssertEqual(gate.status, "deadline")
            XCTAssertEqual(gate.attempts, 1); XCTAssertTrue(gate.settlements.isEmpty)
            XCTAssertFalse(gate.mayPress(target: target, now: time, currentScope: true))
            XCTAssertFalse(try gate.consume(target: target, now: time, currentScope: true))
            gate.stop(); state.close(at: time)
            let inputs = state.evidence.utterances
            XCTAssertEqual(inputs[0].status, "sdk_input_complete"); XCTAssertNil(inputs[0].failureCode)
            XCTAssertEqual(inputs[0].acceptedSamples, 1)
            XCTAssertEqual(inputs[1].status, "failed")
            XCTAssertEqual(inputs[1].failureCode, "memory_readiness_deadline")
            XCTAssertEqual(inputs[1].acceptedSamples, 0); XCTAssertTrue(inputs[1].callbacks.isEmpty)
            XCTAssertNil(inputs[1].enqueueRequestedAt)
            XCTAssertEqual(inputs[2].status, "not_started")
            XCTAssertEqual(inputs[2].failureCode, "run_closed_before_injection")
            XCTAssertNil(inputs[2].triggerArmedAt); XCTAssertNil(inputs[2].enqueueRequestedAt)
            XCTAssertTrue(inputs[2].callbacks.isEmpty)
        }
    }
    func testPollingFailureProjectionNeverRetainsArbitraryErrorDetails() throws {
        let gate = try JarvisReplayMemoryGate(configuration: config(root()), invocationID: fixtureInvocationID)
        let failures: [(Error, String)] = [
            (JarvisReplayError.memoryReadinessUnavailable, "memory_readiness_unavailable"),
            (JarvisReplayError.responseCandidateChanged, "memory_readiness_candidate_fenced"),
            (JarvisReplayError.manualIdentityMismatch, "memory_readiness_identity_mismatch"),
            (NSError(domain: "private-fixture-path", code: 7,
                userInfo: [NSLocalizedDescriptionKey: "private-token-and-payload"]), "memory_readiness_poll_failed")
        ]
        for (error, expected) in failures {
            var records = 0
            gate.recordPollingFailure(error, target: 1, at: 120) { ordinal, code, time in
                records += 1
                XCTAssertEqual(ordinal, 1); XCTAssertEqual(code, expected); XCTAssertEqual(time, 120)
            }
            XCTAssertEqual(records, 1)
            XCTAssertEqual(gate.status, "not_armed"); XCTAssertEqual(gate.attempts, 0)
            XCTAssertTrue(gate.settlements.isEmpty); XCTAssertFalse(gate.probeReady)
        }
        gate.stop()
    }
    func testTimelineKeepsVideoWhileWaitingAndExactBoundaryCannotTrigger() throws {
        let plan = try JarvisReplayPlan(runID: run, duration: 95,
            frames: [.init(at: 0, image: Data([1])), .init(at: 20, image: Data([2]))],
            utterances: [.init(trigger: .atTime(5), pcm16: Data([1, 0])),
                         .init(trigger: .afterManualResponse(trigger()), pcm16: Data([1, 0]))], invocationID: fixtureInvocationID)
        var timeline = JarvisReplayTimeline(plan: plan)
        XCTAssertEqual(timeline.advance(elapsed: 5, output: nil, inputIdle: true).utterance, 0)
        let waiting = timeline.advance(elapsed: 20, output: nil, inputIdle: true)
        XCTAssertEqual(waiting.frame, 1); XCTAssertNil(waiting.utterance)
        let late = timeline.advance(elapsed: 37, output: nil, inputIdle: true, manualResponseReady: true)
        XCTAssertEqual(late.failure, .outputWaitExpired); XCTAssertNil(late.utterance)
    }
    func testManualReadyDoesNotBypassPreEnqueuePredicate() throws {
        let plan = try JarvisReplayPlan(runID: run, duration: 20, frames: [.init(at: 0, image: Data([1]))],
            utterances: [.init(trigger: .atTime(0), pcm16: Data([1, 0]))], invocationID: fixtureInvocationID)
        let input = JarvisReplayUtteranceEvidence(plan: plan, ordinal: 0)
        let controller = JarvisReplayManualInputController(runID: run, invocationID: fixtureInvocationID)
        let identity = try JarvisReplayManualSessionIdentity(ownerID: owner, sessionID: UUID().uuidString,
            audioSessionID: UUID().uuidString, controlConnectionID: UUID(), segmentID: UUID().uuidString)
        try controller.begin(input: input, at: 1) { identity }
        XCTAssertTrue(controller.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                       generations: [:], at: 1.1))
        var count = 0
        try controller.advance(inputs: [input], at: 1.2, sceneDeadline: 20, canEnqueue: { _ in false },
                               enqueue: { _, _ in count += 1 }, release: { _ in XCTFail("early release") })
        XCTAssertEqual(count, 0)
        try controller.advance(inputs: [input], at: 1.3, sceneDeadline: 20, canEnqueue: { _ in true },
                               enqueue: { _, _ in count += 1 }, release: { _ in XCTFail("early release") })
        XCTAssertEqual(count, 1)
    }
    func testDeclarationRequiredManualOnlyAndFinitePriorOrdinal() throws {
        let input = JarvisReplayPlan.Utterance(trigger: .afterManualResponse(trigger()), pcm16: Data([1, 0]))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
            invocationID: fixtureInvocationID, taskMode: false, utterances: [input]))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("auto", route: .realtime,
            invocationID: fixtureInvocationID, taskMode: false, utterances: [input], memoryReadiness: true))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .live,
            invocationID: fixtureInvocationID, taskMode: false, utterances: [input], memoryReadiness: true))
        XCTAssertThrowsError(try trigger().validate(ordinal: 0, duration: 95, pcmSeconds: 1))
        for timeout in [Double.nan, .infinity, 0, 121] {
            let invalid = JarvisReplayManualResponseTrigger(afterInputOrdinal: 0, at: 15, timeout: timeout,
                canonical: "finalized_input", playback: "none")
            XCTAssertThrowsError(try invalid.validate(ordinal: 1, duration: 180, pcmSeconds: 1))
        }
    }


    private func gatedManualFixture(ordinary: Bool = false) throws ->
        (JarvisReplayPlan, JarvisReplayPCMState, JarvisReplayManualInputController, JarvisReplayManualSessionIdentity) {
        let plan = try JarvisReplayPlan(runID: run, duration: 95, frames: [.init(at: 0, image: Data([1]))],
            utterances: [.init(trigger: .atTime(5), pcm16: Data([1, 0])),
                .init(trigger: ordinary ? .atTime(15) : .afterManualResponse(trigger()), pcm16: Data([200, 0]))],
            invocationID: fixtureInvocationID)
        var state = JarvisReplayPCMState(maxDuration: 115); try state.register(plan)
        let controller = JarvisReplayManualInputController(runID: run, invocationID: fixtureInvocationID)
        let identity = try JarvisReplayManualSessionIdentity(ownerID: owner, sessionID: UUID().uuidString,
            audioSessionID: UUID().uuidString, controlConnectionID: UUID(), segmentID: UUID().uuidString)
        return (plan, state, controller, identity)
    }
    @discardableResult
    private func answerCurrentGate(_ config: JarvisReplayMemoryConfiguration, commit: JarvisReplayManualCommit,
                                   candidate: JarvisReplayMemoryCandidate, recordedAt: Double) throws -> JarvisReplayMemoryRequest {
        let io = try JarvisReplayMemoryFiles(config.controlsURL)
        let bytes = try XCTUnwrap(io.read(JarvisReplayMemoryFiles.requestName)); io.close()
        let body = try JarvisReplayTaskJSON.object(bytes, limit: 16384)["request"] as! [String: Any]
        let req = try JarvisReplayMemoryRequest(configuration: config, invocationID: fixtureInvocationID, target: 1,
            trigger: trigger(), commit: commit, candidate: candidate,
            issued: JarvisReplayTaskJSON.time(body["issued_at"]), expires: JarvisReplayTaskJSON.time(body["expires_at"]),
            nonce: JarvisReplayTaskJSON.uuid(body["nonce"]))
        XCTAssertEqual(bytes, req.bytes)
        var returned = receipt(req)
        returned["recorded_at"] = JarvisReplayTaskJSON.timestamp(Date(timeIntervalSince1970: 1_700_000_000 + recordedAt - 100))
        try write(JarvisReplayTaskJSON.encode(returned, limit: 32768),
                  to: config.controlsURL.appendingPathComponent(JarvisReplayMemoryFiles.receiptName))
        return req
    }

    func testSlowFreshConsumeAfterMatchingReadyUsesOriginalDeadlineAndRealSDK() throws {
        let (config, prior, candidate, evidence) = try fixture(root())
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        func poll(_ at: Double, consume: Bool) throws {
            try gate.poll(target: 1, trigger: trigger(), commit: prior, evidence: evidence, clock: clock,
                          now: at, currentScope: true, consume: consume)
        }
        try poll(115, consume: false)
        let probe = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 115.1)
        try poll(115.2, consume: false)
        var (plan, state, manual, identity) = try gatedManualFixture()
        identity = try JarvisReplayManualSessionIdentity(ownerID: prior.identity.ownerID,
            sessionID: prior.identity.sessionID.uuidString, audioSessionID: prior.identity.audioSessionID.uuidString,
            controlConnectionID: prior.identity.controlConnectionID, segmentID: UUID().uuidString)
        var enqueues = 0, releases = 0
        try manual.begin(input: state.evidence.utterances[1], at: 115.21, sourceClockOrigin: 100) { identity }
        try gate.markPressed(target: 1, at: 115.22)
        XCTAssertTrue(manual.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                   generations: [:], at: 116))
        state.gateInput(true)
        try poll(116, consume: true)
        func advance(_ at: Double) throws {
            try manual.advance(inputs: state.evidence.utterances, at: at, sceneDeadline: 195,
                canEnqueue: { ordinal in try gate.consume(target: ordinal, now: at, currentScope: true) },
                enqueue: { ordinal, triggered in
                    enqueues += 1
                    try state.enqueue(plan.utterances[ordinal].pcm16, utteranceOrdinal: ordinal,
                                      requestedAt: at, triggeredAt: triggered)
                }, release: { _ in releases += 1 })
        }
        try advance(116.1)
        try poll(119.3, consume: true); try poll(119.4, consume: true)
        try advance(121)
        XCTAssertNil(manual.failureCode); XCTAssertEqual(enqueues, 0); XCTAssertEqual(releases, 0)
        let final = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 124.9)
        XCTAssertNotEqual(probe.nonce, final.nonce); XCTAssertEqual(probe.expires, final.expires)
        try poll(125, consume: true)
        try advance(125.01) // Almost ten seconds after press, still before source deadline 137.
        XCTAssertEqual(enqueues, 1); XCTAssertEqual(releases, 0)
        _ = state.renderInput(at: 125.1); state.inputCallbackStarted(at: 125.1)
        try advance(125.101); XCTAssertEqual(releases, 0)
        state.inputCallbackCompleted(status: 0, at: 125.102)
        try advance(125.11); XCTAssertEqual(releases, 1)
        let rows = try XCTUnwrap(manual.json["occurrences"] as? [[String: Any]])
        XCTAssertEqual(rows[0]["response_enqueue_deadline_uptime_seconds"] as? Double, 137)
        XCTAssertEqual(rows[0]["sdk_accepted_clip_samples"] as? Int, 1)
        XCTAssertEqual(gate.settlements.count, 1)
        manual.retire(); gate.stop()
    }

    func testGatedReadyHandshakeStillRejectsFiveSecondsAndMissingSourceOrigin() throws {
        let (_, state, manual, identity) = try gatedManualFixture()
        var presses = 0
        XCTAssertThrowsError(try manual.begin(input: state.evidence.utterances[1], at: 115) { presses += 1; return identity })
        XCTAssertEqual(presses, 0)
        try manual.begin(input: state.evidence.utterances[1], at: 115, sourceClockOrigin: 100) { identity }
        XCTAssertFalse(manual.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                    generations: [:], at: 120))
        XCTAssertThrowsError(try manual.advance(inputs: state.evidence.utterances, at: 120, sceneDeadline: 195,
            canEnqueue: { _ in XCTFail("no ready"); return true },
            enqueue: { _, _ in XCTFail("no input") }, release: { _ in XCTFail("no empty commit") })) {
            XCTAssertEqual($0 as? JarvisReplayError, .manualReadyExpired)
        }
        XCTAssertEqual(manual.failureCode, "manual_ready_timeout")
        manual.retire()
    }

    func testGatedConsumeExactDeadlineAndDelayedWakeFailWithoutEmptyCommit() throws {
        for time in [137.0, 140.0] {
            let (_, state, manual, identity) = try gatedManualFixture()
            try manual.begin(input: state.evidence.utterances[1], at: 115, sourceClockOrigin: 100) { identity }
            XCTAssertTrue(manual.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                       generations: [:], at: 116))
            XCTAssertThrowsError(try manual.advance(inputs: state.evidence.utterances, at: time, sceneDeadline: 195,
                canEnqueue: { _ in XCTFail("deadline precedes consume"); return true },
                enqueue: { _, _ in XCTFail("late input") }, release: { _ in XCTFail("empty commit") })) {
                XCTAssertEqual($0 as? JarvisReplayError, .outputWaitExpired)
            }
            XCTAssertEqual(manual.failureCode, "manual_calendar_enqueue_deadline")
            manual.retire()
            XCTAssertThrowsError(try manual.advance(inputs: state.evidence.utterances, at: time + 1, sceneDeadline: 195,
                enqueue: { _, _ in XCTFail("revived input") }, release: { _ in XCTFail("revived commit") }))
        }
    }

    func testOrdinaryManualReadyEnqueueWatchdogRemainsUnchanged() throws {
        let (_, state, manual, identity) = try gatedManualFixture(ordinary: true)
        try manual.begin(input: state.evidence.utterances[1], at: 115, sourceClockOrigin: 100) { identity }
        XCTAssertTrue(manual.ready(segmentID: identity.segmentID.uuidString, connectionID: identity.controlConnectionID,
                                   generations: [:], at: 116))
        XCTAssertThrowsError(try manual.advance(inputs: state.evidence.utterances, at: 120, sceneDeadline: 195,
            enqueue: { _, _ in XCTFail("ordinary watchdog changed") }, release: { _ in XCTFail("empty commit") })) {
            XCTAssertEqual($0 as? JarvisReplayError, .manualReadyExpired)
        }
        manual.retire()
    }

    func testWaitingForMatchingReadyDoesNotChurnExpiredProbeNonce() throws {
        let (config, prior, candidate, evidence) = try fixture(root())
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        func poll(_ at: Double, _ consume: Bool) throws {
            try gate.poll(target: 1, trigger: trigger(), commit: prior, evidence: evidence, clock: clock,
                          now: at, currentScope: true, consume: consume)
        }
        try poll(115, false)
        let probe = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 115.1)
        try poll(115.2, false); try gate.markPressed(target: 1, at: 115.21)
        for time in [115.6, 116, 117, 118, 119.9] {
            try poll(time, false)
            XCTAssertEqual(gate.attempts, 1); XCTAssertFalse(gate.probeReady)
            XCTAssertFalse(try gate.consume(target: 1, now: time, currentScope: true))
        }
        try poll(120, true)
        XCTAssertEqual(gate.attempts, 2)
        let consume = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 120.1)
        XCTAssertNotEqual(probe.nonce, consume.nonce)
        XCTAssertEqual(probe.issued, consume.issued); XCTAssertEqual(probe.expires, consume.expires)
        gate.stop()
    }

    func testConsumeFreshnessStillRejectsOlderThan250Milliseconds() throws {
        let (config, prior, candidate, evidence) = try fixture(root())
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        func poll(_ at: Double, _ consume: Bool) throws {
            try gate.poll(target: 1, trigger: trigger(), commit: prior, evidence: evidence, clock: clock,
                          now: at, currentScope: true, consume: consume)
        }
        try poll(115, false)
        try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 115.1)
        try poll(115.2, false); try gate.markPressed(target: 1, at: 115.21)
        try poll(116, true)
        let first = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 116.1)
        try poll(116.2, true)
        XCTAssertFalse(try gate.consume(target: 1, now: 116.451, currentScope: true))
        try poll(116.5, true)
        let fresh = try answerCurrentGate(config, commit: prior, candidate: candidate, recordedAt: 116.6)
        XCTAssertNotEqual(first.nonce, fresh.nonce); XCTAssertEqual(first.expires, fresh.expires)
        try poll(116.7, true)
        XCTAssertTrue(try gate.consume(target: 1, now: 116.71, currentScope: true))
        gate.stop()
    }

    func testCandidateChurnHasExplicitBoundedFailureNotUnlimitedRetry() throws {
        let (config, prior, candidate, evidence) = try fixture(root())
        let gate = try JarvisReplayMemoryGate(configuration: config, invocationID: fixtureInvocationID)
        let clock = JarvisReplayClockAnchor(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100)
        func changed(_ n: Int) -> JarvisReplayAudioEvidence {
            var value = evidence
            let id = JarvisReplayResponseIdentity(connectionID: candidate.identity.connectionID,
                responseID: "resp_candidate_\(n)", itemID: "item_candidate_\(n)", outputEpoch: n + 1)
            value.outputGates = [.init(eventID: "gate_\(n)", enabled: true, changed: true,
                requestedAt: 110, outputEpoch: n + 1, lastAdmittedActiveSample: nil,
                responseIdentity: id, playbackDiagnostics: diagnostic("final_answer"))]
            value.controlEvents = [
                .init(stage: .backendControlHandled, handledAt: 110, receivedAt: nil, fields: [
                    "event_type": "audio.webrtc.response", "response_id": id.responseID,
                    "connection_id": id.connectionID.uuidString.lowercased(),
                    "control_connection_id": prior.identity.controlConnectionID.uuidString.lowercased(),
                    "audio_session_id": prior.identity.audioSessionID.uuidString.lowercased()]),
                .init(stage: .backendControlHandled, handledAt: 112, receivedAt: nil,
                    fields: ["event_type": "audio.webrtc.stopped"], boundResponseIdentity: id)]
            return value
        }
        for n in 0..<32 {
            try gate.poll(target: 1, trigger: trigger(), commit: prior, evidence: changed(n), clock: clock,
                          now: 115 + Double(n) * 0.1, currentScope: true, consume: false)
        }
        XCTAssertEqual(gate.attempts, 32)
        XCTAssertThrowsError(try gate.poll(target: 1, trigger: trigger(), commit: prior, evidence: changed(32),
            clock: clock, now: 118.3, currentScope: true, consume: false)) {
            XCTAssertEqual($0 as? JarvisReplayError, .memoryReadinessUnavailable)
        }
        XCTAssertEqual(gate.status, "consume_budget")
        XCTAssertTrue(gate.settlements.isEmpty)
        gate.stop()
    }

    private func wav() -> Data {
        let pcm = Data((0..<480).flatMap { _ in [UInt8(200), UInt8(0)] })
        var data = Data("RIFF".utf8)
        func u32(_ n: UInt32) { var n = n.littleEndian; withUnsafeBytes(of: &n) { data.append(contentsOf: $0) } }
        func u16(_ n: UInt16) { var n = n.littleEndian; withUnsafeBytes(of: &n) { data.append(contentsOf: $0) } }
        u32(UInt32(36 + pcm.count)); data.append(Data("WAVEfmt ".utf8)); u32(16)
        u16(1); u16(1); u32(48000); u32(96000); u16(2); u16(16)
        data.append(Data("data".utf8)); u32(UInt32(pcm.count)); data.append(pcm); return data
    }
    private func qualifyPrepared(_ prepared: Data, offset: Double, tail: Double, expectedSHA: String) throws {
        XCTAssertEqual(JarvisReplayTaskJSON.hash(prepared), expectedSHA)
        let decoded = try JSONDecoder().decode(JarvisReplayLaunchPlan.self, from: prepared)
        XCTAssertEqual(decoded.memory_readiness?.scope_file, "memory-readiness-scope.json")
        XCTAssertNil(decoded.memory_readiness?.manifest_sha256)
        XCTAssertEqual(decoded.duration_seconds, 95); XCTAssertEqual(decoded.post_media_capture_seconds, tail)
        for utterance in decoded.utterances.dropFirst() {
            guard case .afterManualResponse(let value) = try utterance.trigger.value() else { return XCTFail("actual trigger form") }
            XCTAssertEqual(value.deadline, value.at + value.timeout)
        }
        let directory = try root(), inputs = directory.appendingPathComponent("inputs")
        let speech = inputs.appendingPathComponent("speech")
        try FileManager.default.createDirectory(at: speech, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let audio = wav()
        var clips: [[String: Any]] = []
        for utterance in decoded.utterances {
            let path = utterance.clip_id + ".wav"
            try write(audio, to: speech.appendingPathComponent(path))
            clips.append(["id": utterance.clip_id, "relative_path": path, "sha256": JarvisReplayTaskJSON.hash(audio),
                "sample_rate_hz": 48000, "channels": 1, "sample_width_bytes": 2, "sample_count": 480,
                "first_active_sample": 0, "last_active_sample_exclusive": 480, "energy_threshold_abs_pcm16": 128])
        }
        try write(JarvisReplayTaskJSON.encode(["format": "jarvis.speech_assets.v1", "clips": clips]),
                  to: speech.appendingPathComponent("speech.json"))
        try write(Data("fixture-token".utf8), to: inputs.appendingPathComponent("isolated-token"))
        try write(JarvisReplayTaskJSON.encode(scope), to: inputs.appendingPathComponent("memory-readiness-scope.json"))
        var runtime = try JarvisReplayTaskJSON.object(prepared, limit: 512 * 1024)
        runtime["run_id"] = runHex; runtime["owner_id"] = owner; runtime["area_id"] = area
        runtime["memory_bank_id"] = scope["memory_bank_id"]; runtime["bearer_token_file"] = "isolated-token"
        runtime["replay_invocation_id"] = fixtureInvocationID.uuidString.lowercased()
        runtime["memory_readiness"] = ["version": 1, "scope_file": "memory-readiness-scope.json",
                                       "manifest_sha256": expectedSHA] as [String: Any]
        var clock: [String: Any] = ["version": 1, "phase_start_seconds": offset]
        let boot = UUID()
        if offset > 0 {
            let anchor = JarvisReplaySourceAnchor(runID: run, invocationID: UUID(),
                clock: .init(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_700_000_000), monotonicAfter: 100), bootID: boot)
            let bytes = try JSONEncoder().encode(anchor)
            try write(bytes, to: inputs.appendingPathComponent("source-clock.json"))
            clock["anchor_file"] = "source-clock.json"; clock["anchor_sha256"] = JarvisReplayTaskJSON.hash(bytes)
        }
        runtime["replay_source_clock"] = clock
        let bytes = try JarvisReplayTaskJSON.encode(runtime, limit: 512 * 1024)
        let path = inputs.appendingPathComponent("replay.json")
        try write(bytes, to: path)
        let loaded = try JarvisReplayLaunchPlan.load(path)
        XCTAssertTrue(loaded.manualInput); XCTAssertTrue(loaded.plan.frames.isEmpty)
        XCTAssertEqual(loaded.memoryReadiness?.manifestSHA256, expectedSHA)
        XCTAssertNotEqual(loaded.memoryReadiness?.manifestSHA256, JarvisReplayTaskJSON.hash(bytes))
        XCTAssertEqual(loaded.memoryReadiness?.controlsURL, directory.appendingPathComponent("controls"))
        XCTAssertEqual(loaded.plan.duration, 95); XCTAssertEqual(loaded.plan.postMediaCaptureSeconds, tail)
        if offset > 0 {
            let resolved = try XCTUnwrap(loaded.plan.sourceClock).resolve(plan: loaded.plan,
                measured: .init(monotonicBefore: 215, utc: Date(timeIntervalSince1970: 1_700_000_115), monotonicAfter: 215),
                bootID: boot, now: 215)
            XCTAssertEqual(resolved.phase.referenceUptime, 100 + offset)
            XCTAssertEqual(resolved.phase.utc.timeIntervalSince1970, 1_700_000_000 + offset)
        }
        // No runtime authorization fixtureManifestSHA256 may be invented from this rebound JSON.
        runtime["memory_readiness"] = ["version": 1, "scope_file": "memory-readiness-scope.json"]
        try write(JarvisReplayTaskJSON.encode(runtime, limit: 512 * 1024), to: path)
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.load(path))
    }
    func testActualPreparedPhaseAAndRuntimeEquivalent() throws {
        try qualifyPrepared(Self.preparedA, offset: 0, tail: 10, expectedSHA: "be7637f3363940031bce025bc9ce52a83dcd3e576bbbd8c20649abecb41f2ffd")
    }
    func testActualPreparedPhaseBAndRuntimeEquivalentNoSourceClockRebase() throws {
        try qualifyPrepared(Self.preparedB, offset: 115, tail: 20, expectedSHA: "350f7a3f2c2f7a01679bac0dc3925660c307b5fd55699b9de1c3d92f91c7d1b1")
    }
    func testFormerBoundaryAliasesAreNotAccepted() throws {
        let bytes = Data(#"{"kind":"after_manual_response","after_input_ordinal":0,"at_seconds":15,"timeout_seconds":22,"canonical_boundary":"retain_searchable","playback_boundary":"authenticated_buffer_stop"}"#.utf8)
        let trigger = try JSONDecoder().decode(JarvisReplayLaunchPlan.Trigger.self, from: bytes)
        XCTAssertThrowsError(try trigger.value())
    }
    func testScopeFileSymlinkAndNonprivateFileRejected() throws {
        let directory = try root(), source = directory.appendingPathComponent("scope.json")
        try write(JarvisReplayTaskJSON.encode(scope), to: source)
        let link = directory.appendingPathComponent("scope-link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        XCTAssertThrowsError(try JarvisReplayMemoryConfiguration.readScope(link))
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: source.path)
        XCTAssertThrowsError(try JarvisReplayMemoryConfiguration.readScope(source))
    }

    // Original: calendar-response-manual-v2.4tpytC/phase-a/manifest.json
    private static let preparedA = Data((#"""
        {
          "version": 1,
          "run_id": "00000000-0000-0000-0000-000000000000",
          "backend_base_url": "http://127.0.0.1:1",
          "owner_id": "compiler-placeholder",
          "area_id": "00000000-0000-0000-0000-000000000000",
          "memory_bank_id": "compiler-placeholder",
          "voice_route": "realtime",
          "bearer_token_file": "compiler-placeholder",
          "duration_seconds": 95,
          "post_media_capture_seconds": 10,
          "capture_mode": "manual",
          "memory_readiness": {
            "version": 1,
            "scope_file": "memory-readiness-scope.json"
          },
          "speech_manifest": "speech/speech.json",
          "frames": [],
          "utterances": [
            {
              "clip_id": "cal_a1",
              "trigger": {
                "kind": "at_time",
                "at_seconds": 5
              }
            },
            {
              "clip_id": "cal_a2",
              "trigger": {
                "kind": "after_manual_response",
                "after_input_ordinal": 0,
                "at_seconds": 15,
                "timeout_seconds": 22,
                "canonical_requirement": "retain_searchable",
                "playback_requirement": "authenticated_buffer_stop"
              }
            },
            {
              "clip_id": "cal_a3",
              "trigger": {
                "kind": "after_manual_response",
                "after_input_ordinal": 1,
                "at_seconds": 53,
                "timeout_seconds": 20,
                "canonical_requirement": "retain_searchable",
                "playback_requirement": "authenticated_buffer_stop"
              }
            }
          ]
        }
        """# + "\n").utf8)

    // Original: calendar-response-manual-v2.4tpytC/phase-b/manifest.json
    private static let preparedB = Data((#"""
        {
          "version": 1,
          "run_id": "00000000-0000-0000-0000-000000000000",
          "backend_base_url": "http://127.0.0.1:1",
          "owner_id": "compiler-placeholder",
          "area_id": "00000000-0000-0000-0000-000000000000",
          "memory_bank_id": "compiler-placeholder",
          "voice_route": "realtime",
          "bearer_token_file": "compiler-placeholder",
          "duration_seconds": 95,
          "post_media_capture_seconds": 20,
          "capture_mode": "manual",
          "memory_readiness": {
            "version": 1,
            "scope_file": "memory-readiness-scope.json"
          },
          "speech_manifest": "speech/speech.json",
          "frames": [],
          "utterances": [
            {
              "clip_id": "cal_b1",
              "trigger": {
                "kind": "at_time",
                "at_seconds": 20
              }
            },
            {
              "clip_id": "cal_b2",
              "trigger": {
                "kind": "after_manual_response",
                "after_input_ordinal": 0,
                "at_seconds": 37,
                "timeout_seconds": 18,
                "canonical_requirement": "finalized_input",
                "playback_requirement": "authenticated_buffer_stop"
              }
            },
            {
              "clip_id": "cal_b3",
              "trigger": {
                "kind": "after_manual_response",
                "after_input_ordinal": 1,
                "at_seconds": 68,
                "timeout_seconds": 16,
                "canonical_requirement": "finalized_input",
                "playback_requirement": "authenticated_buffer_stop"
              }
            }
          ]
        }
        """# + "\n").utf8)
}
