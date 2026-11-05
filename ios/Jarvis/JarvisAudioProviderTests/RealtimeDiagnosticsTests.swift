import Foundation
import XCTest

final class RealtimeDiagnosticsTests: XCTestCase {
    func testLiveProtocolFailureOnlyRetainsClosedCategoriesNotPayloadOrErrorDescriptions() throws {
        let event = JarvisRealtimeServerEvent(type: "SECRET_TOKEN", values: [
            "type": "SECRET_TOKEN", "payload": ["kind": "PRIVATE_CAPTION", "text": "PRIVATE_TEXT"],
            "authorization": "Bearer PRIVATE_KEY", "url": "https://private.invalid",
        ])
        let failure = JarvisLiveProtocolFailure(stage: .payload, field: .observedAt, event: event,
            error: NSError(domain: "PRIVATE_DOMAIN", code: 1, userInfo: [NSLocalizedDescriptionKey: "PRIVATE_ERROR"]))
        let line = try XCTUnwrap(RealtimeDiagnostics.record("live_protocol_failure", fields: failure.fields))
        let record = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        XCTAssertEqual(record["outer_type"] as? String, "unknown")
        XCTAssertEqual(record["nested_type"] as? String, "unknown")
        XCTAssertEqual(record["protocol_reason"] as? String, "unexpected_error")
        XCTAssertEqual(record["protocol_field"] as? String, "observed_at")
        XCTAssertFalse(line.contains("PRIVATE"))
        XCTAssertFalse(line.contains("SECRET"))
        XCTAssertFalse(line.contains("https"))
        XCTAssertNil(record["text"])
        XCTAssertNil(record["authorization"])
    }

    func testProtocolFieldSanitizerRejectsArbitrarySafeLookingTokensAndWrongTypes() {
        XCTAssertTrue(RealtimeDiagnostics.safeFields([
            "protocol_stage": "secret_token", "protocol_field": "private_field",
            "protocol_reason": "PRIVATE", "outer_type": "secret", "nested_type": 42,
        ]).isEmpty)
        let event = JarvisRealtimeServerEvent(type: "", values: ["type": 42, "payload": ["kind": NSNull()]])
        let failure = JarvisLiveProtocolFailure(stage: .envelope, field: .protocolVersion,
            event: event, error: JarvisVoiceRouteError.invalidWire)
        XCTAssertEqual(failure.fields["outer_type"] as? String, "non_string")
        XCTAssertEqual(failure.fields["nested_type"] as? String, "non_string")
        let missing = JarvisLiveProtocolFailure(stage: .mediaSignaling, field: .mediaOperation,
                                               error: CancellationError())
        XCTAssertEqual(missing.fields["outer_type"] as? String, "missing")
        XCTAssertEqual(missing.fields["nested_type"] as? String, "missing")
        XCTAssertEqual(missing.fields["protocol_reason"] as? String, "cancelled")
    }

    func testFieldsExcludePayloadsAndCredentials() {
        let fields = RealtimeDiagnostics.safeFields([
            "response_id": "fixture-response", "audio_base64": "SECRET", "text": "SECRET",
            "authorization": "Bearer SECRET", "frame_id": "https://private/image?token=SECRET",
            "reason": "private content", "audio_end_ms": Double.infinity,
            "connection_generation": true, "image": ["url": "SECRET"],
        ])
        XCTAssertEqual(Set(fields.keys), ["response_id"])
    }

    func testFirstAudioIsCorrelatedDeduplicatedAndBounded() throws {
        var diagnostics = RealtimeDiagnostics()
        let fields = ["response_id": "r", "item_id": "i", "audio_session_id": "a", "provider": "gemini"]
        let first = try XCTUnwrap(diagnostics.firstAudio(fields: fields))
        XCTAssertNil(diagnostics.firstAudio(fields: fields))
        let record = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(first.utf8)) as? [String: Any])
        XCTAssertEqual(record["event"] as? String, "first_audio_received")
        XCTAssertEqual(record["audio_session_id"] as? String, "a")
        XCTAssertNotNil(record["monotonic_s"])
        XCTAssertNil(record["audible_at"])
        for index in 0..<(RealtimeDiagnostics.recentItemLimit + 3) {
            XCTAssertNotNil(diagnostics.firstAudio(fields: ["response_id": "r\(index)"]))
        }
        XCTAssertEqual(diagnostics.retainedItemCount, RealtimeDiagnostics.recentItemLimit)
        XCTAssertNil(diagnostics.firstAudio(fields: [:]))
    }

    @MainActor
    func testPlaybackDiagnosticsDoNotRequireOrStartAnAudioEngine() {
        let playback = AudioPlaybackService()
        XCTAssertFalse(playback.isPlaying)
        XCTAssertNil(playback.truncationSnapshot())
        let record = RealtimeDiagnostics.record("playback_requested", fields: ["boundary": "player_play_called_not_hardware_ack"])
        XCTAssertTrue(record?.contains("not_hardware_ack") == true)
        XCTAssertFalse(playback.isPlaying)
    }
}

final class JarvisStartupDiagnosticsTests: XCTestCase {
    func testActualCaptureSerializerExportsStartupReaderCompatibilityFixtures() throws {
        let runID = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
        let attemptID = UUID(uuidString: "22222222-2222-4222-8222-222222222222")!
        let sessionID = UUID(uuidString: "33333333-3333-4333-8333-333333333333")!
        let audioSessionID = UUID(uuidString: "44444444-4444-4444-8444-444444444444")!
        let connectionID = UUID(uuidString: "55555555-5555-4555-8555-555555555555")!
        for capped in [false, true] {
            var state = JarvisReplayPCMState(maxDuration: 45)
            var startup = JarvisStartupDiagnostics(attemptID: attemptID, origin: .replayAutostart)
            func append(_ marker: JarvisStartupDiagnostics.Marker?) {
                guard let marker else { return }
                var fields = marker.fields
                fields["unknown_private_payload"] = "DO_NOT_RETAIN_PAYLOAD_OR_SECRET"
                state.recordControlEvent(.clientStartupObserved, fields: fields,
                                         receivedAt: marker.at, at: marker.at + 0.001)
            }
            append(startup.start(at: 100))
            append(startup.bindSession(sessionID, at: 100.1))
            append(startup.bindAudioSession(audioSessionID, at: 100.2))
            append(startup.bindConnection(connectionID, at: 100.3))
            append(startup.observedOpen(connection: connectionID, at: 101))
            append(startup.predicate(.sessionInputReady, enabled: true, at: 101.1))
            append(startup.predicate(.manualBeginEnabled, enabled: true, at: 101.2))
            if capped {
                for index in 0..<80 {
                    append(startup.predicate(.manualBeginEnabled, enabled: index % 2 == 1,
                                             at: 102 + Double(index) / 100))
                }
            }
            try state.recordAnchor(.init(monotonicBefore: 105,
                                         utc: Date(timeIntervalSince1970: 1_800_000_000),
                                         monotonicAfter: 105))
            state.close(at: 110)
            let capture = JarvisReplayEvidenceReport.make(evidence: state.evidence,
                runID: runID, finalUIStatus: "offline startup compatibility fixture")
            let events = try XCTUnwrap(capture["control_events"] as? [[String: Any]])
            XCTAssertEqual(events.count, capped ? JarvisStartupDiagnostics.maximumMarkers : 7)
            XCTAssertEqual(events.first?["received_monotonic_s"] as? Double, 100)
            XCTAssertEqual((capture["clock_anchor"] as? [String: Any])?["monotonic_reference_seconds"] as? Double, 105)
            if capped {
                let lastFields = try XCTUnwrap(events.last?["fields"] as? [String: Any])
                XCTAssertEqual(lastFields["event_type"] as? String, "client.startup.diagnostics_limit")
                XCTAssertNil(lastFields["status"])
            }
            let bytes = try JSONSerialization.data(withJSONObject: capture, options: [.sortedKeys])
            XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("DO_NOT_RETAIN_PAYLOAD_OR_SECRET"))
            XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("unknown_private_payload"))
            XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
            let attachment = XCTAttachment(data: bytes, uniformTypeIdentifier: "public.json")
            attachment.name = capped ? "startup-capped.capture.json" : "startup-normal.capture.json"
            attachment.lifetime = .keepAlways
            add(attachment)
        }
    }

    func testStartOriginsAndAttemptIdentityAreExplicitAndDeduplicated() throws {
        for origin in JarvisStartupDiagnostics.Origin.allCases {
            let attempt = UUID()
            var trace = JarvisStartupDiagnostics(attemptID: attempt, origin: origin)
            let marker = try XCTUnwrap(trace.start(at: 10))
            XCTAssertEqual(marker.at, 10)
            XCTAssertEqual(marker.fields["startup_attempt_id"] as? String, attempt.uuidString.lowercased())
            XCTAssertEqual(marker.fields["reason"] as? String, origin.rawValue)
            XCTAssertEqual(marker.fields["event_type"] as? String, "client.startup.start_requested")
            XCTAssertNil(trace.start(at: 11))
            XCTAssertNil(marker.fields["respond_ready"])
        }
    }

    func testObservedOpenRequiresCurrentConnectionAndCannotBeInferredFromBinding() throws {
        var trace = JarvisStartupDiagnostics(attemptID: UUID(), origin: .replayAutostart)
        _ = trace.start(at: 10)
        let old = UUID(), current = UUID()
        XCTAssertNil(trace.observedOpen(connection: old, at: 10.1))
        let bound = try XCTUnwrap(trace.bindConnection(old, at: 10.2))
        XCTAssertEqual(bound.fields["event_type"] as? String, "client.startup.control_connect_requested")
        XCTAssertNil(trace.observedOpen(connection: current, at: 10.3))
        XCTAssertNotNil(trace.observedOpen(connection: old, at: 10.4))
        XCTAssertNil(trace.observedOpen(connection: old, at: 10.5))
        _ = trace.bindConnection(current, at: 11)
        XCTAssertNil(trace.observedOpen(connection: old, at: 11.1))
        let opened = try XCTUnwrap(trace.observedOpen(connection: current, at: 11.2))
        XCTAssertEqual(opened.fields["control_connection_id"] as? String, current.uuidString.lowercased())
    }

    func testCancelledAttemptRejectsLateOpenAndInputEnabledObservations() throws {
        let connection = UUID()
        var trace = JarvisStartupDiagnostics(attemptID: UUID(), origin: .humanStartButton)
        _ = trace.start(at: 10); _ = trace.bindConnection(connection, at: 11)
        XCTAssertNotNil(trace.retire(at: 12))
        XCTAssertNil(trace.observedOpen(connection: connection, at: 13))
        XCTAssertNil(trace.predicate(.sessionInputReady, enabled: true, at: 13))
        XCTAssertNil(trace.retire(at: 14))
        var replacement = JarvisStartupDiagnostics(attemptID: UUID(), origin: .humanStartButton)
        let next = try XCTUnwrap(replacement.start(at: 15))
        XCTAssertNotEqual(next.fields["startup_attempt_id"] as? String, trace.attemptID.uuidString.lowercased())
        XCTAssertNil(replacement.observedOpen(connection: connection, at: 16))
    }

    func testOwnedSessionIdentitiesCannotBeOverwrittenWithinAttempt() throws {
        var trace = JarvisStartupDiagnostics(attemptID: UUID(), origin: .unspecified)
        _ = trace.start(at: 10)
        let session = UUID(), audio = UUID()
        _ = trace.bindSession(session, at: 11); _ = trace.bindAudioSession(audio, at: 12)
        XCTAssertNil(trace.bindSession(UUID(), at: 13))
        XCTAssertNil(trace.bindAudioSession(UUID(), at: 13))
        let marker = try XCTUnwrap(trace.predicate(.sessionInputReady, enabled: true, at: 14))
        XCTAssertEqual(marker.fields["session_id"] as? String, session.uuidString.lowercased())
        XCTAssertEqual(marker.fields["audio_session_id"] as? String, audio.uuidString.lowercased())
    }

    func testLocalAdmissionAndMicrophoneTransitionsRemainDistinctAndDoNotAuthorizeOutput() throws {
        var trace = JarvisStartupDiagnostics(attemptID: UUID(), origin: .replayAutostart)
        _ = trace.start(at: 10)
        XCTAssertNotNil(trace.predicate(.sessionInputReady, enabled: false, at: 10))
        XCTAssertNil(trace.predicate(.sessionInputReady, enabled: false, at: 11))
        let ready = try XCTUnwrap(trace.predicate(.sessionInputReady, enabled: true, at: 12))
        let muted = try XCTUnwrap(trace.predicate(.nativeMicrophoneEnabled, enabled: false, at: 12))
        XCTAssertEqual(ready.fields["status"] as? String, "enabled")
        XCTAssertEqual(muted.fields["status"] as? String, "disabled")
        let mic = try XCTUnwrap(trace.predicate(.nativeMicrophoneEnabled, enabled: true, at: 14))
        XCTAssertEqual(mic.at - ready.at, 2)
        var state = JarvisReplayPCMState(maxDuration: 5)
        state.recordControlEvent(.clientStartupObserved, fields: ready.fields, receivedAt: ready.at, at: 12.1)
        XCTAssertFalse(state.inputEnabled); XCTAssertFalse(state.outputEnabled)
        XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
    }

    func testInvalidTimesAndBoundedChurnNeverInventAnEarliestReadyMarker() throws {
        var trace = JarvisStartupDiagnostics(attemptID: UUID(), origin: .unspecified)
        XCTAssertNil(trace.start(at: .nan)); XCTAssertNil(trace.start(at: -1))
        _ = trace.start(at: 10)
        XCTAssertNil(trace.predicate(.sessionInputReady, enabled: true, at: 9))
        XCTAssertNil(trace.predicate(.sessionInputReady, enabled: true, at: .infinity))
        var last: JarvisStartupDiagnostics.Marker?
        for index in 0..<(JarvisStartupDiagnostics.maximumMarkers * 2) {
            if let marker = trace.predicate(.manualBeginEnabled, enabled: index.isMultiple(of: 2), at: 11 + Double(index)) {
                last = marker
            }
        }
        XCTAssertEqual(trace.markerCount, JarvisStartupDiagnostics.maximumMarkers)
        XCTAssertEqual(last?.fields["event_type"] as? String, "client.startup.diagnostics_limit")
        XCTAssertNil(last?.fields["status"])
        _ = trace.retire(at: 200)
        XCTAssertTrue(trace.retired)
    }

    func testPreSceneStartupMarkersSurviveExistingBoundedCaptureAndKeepOriginalTimes() throws {
        var trace = JarvisStartupDiagnostics(attemptID: UUID(), origin: .replayAutostart)
        let start = try XCTUnwrap(trace.start(at: 100))
        let connection = UUID()
        _ = trace.bindConnection(connection, at: 101)
        let opened = try XCTUnwrap(trace.observedOpen(connection: connection, at: 102))
        var state = JarvisReplayPCMState(maxDuration: 5)
        for marker in [start, opened] {
            state.recordControlEvent(.clientStartupObserved, fields: marker.fields,
                                     receivedAt: marker.at, at: marker.at + 0.01)
        }
        try state.recordAnchor(.init(monotonicBefore: 104, utc: Date(timeIntervalSince1970: 1_700_000_000),
                                     monotonicAfter: 104))
        let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: UUID(), finalUIStatus: "fixture")
        let events = try XCTUnwrap(report["control_events"] as? [[String: Any]])
        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0]["stage"] as? String, "client_startup_observed")
        XCTAssertEqual(events[0]["received_monotonic_s"] as? Double, 100)
        XCTAssertEqual(events[1]["received_monotonic_s"] as? Double, 102)
        XCTAssertEqual((events[0]["fields"] as? [String: Any])?["startup_attempt_id"] as? String,
                       trace.attemptID.uuidString.lowercased())
        XCTAssertFalse(state.evidence.controlEventsTruncated)
        XCTAssertFalse(state.evidence.controlEventTimingInvalid)
        state.close(at: 110)
        state.recordControlEvent(.clientStartupObserved, fields: start.fields, at: 111)
        XCTAssertEqual(state.evidence.controlEvents.count, 2)
    }

    func testStartupFieldsRejectPrivatePayloadAndMalformedAttemptIdentity() {
        XCTAssertTrue(RealtimeDiagnostics.safeFields(["startup_attempt_id": "PRIVATE", "text": "PRIVATE",
            "authorization": "PRIVATE", "protocol": "PRIVATE"]).isEmpty)
        let id = UUID()
        XCTAssertEqual(RealtimeDiagnostics.safeFields(["startup_attempt_id": id.uuidString])["startup_attempt_id"] as? String,
                       id.uuidString.lowercased())
    }

    func testFoundationOpenObservationDoesNotResumeTaskOrRetainNegotiatedProtocol() {
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        let socket = session.webSocketTask(with: URL(string: "ws://127.0.0.1:1/offline-not-opened")!)
        let observed = expectation(description: "Explicit delegate callback only")
        let observer = JarvisControlSocketOpenObserver { task, at in
            XCTAssertEqual(task.state, .suspended)
            XCTAssertTrue(at.isFinite); XCTAssertGreaterThan(at, 0)
            observed.fulfill()
        }
        XCTAssertEqual(socket.state, .suspended)
        observer.urlSession(session, webSocketTask: socket, didOpenWithProtocol: "PRIVATE_NEVER_RETAINED")
        wait(for: [observed], timeout: 1)
        XCTAssertEqual(socket.state, .suspended)
    }
}

final class AudioPlaybackLedgerTests: XCTestCase {
    func testGenerationIsNotPlaybackAndCompletionIsDeduplicated() throws {
        var ledger = AudioPlaybackLedger()
        let token = try ledger.schedule(responseId: "r", itemId: "i", samples: 2400)
        XCTAssertTrue(ledger.generationFinished(responseId: "r", itemId: "i").isEmpty)
        let receipts = ledger.playedBack(token)
        XCTAssertEqual(receipts.map(\.state), [.started, .completed])
        XCTAssertEqual(receipts.last?.playedSamples, 2400)
        XCTAssertTrue(ledger.playedBack(token).isEmpty)
        XCTAssertTrue(ledger.generationFinished(responseId: "r", itemId: "i").isEmpty)
        XCTAssertFalse(ledger.hasPendingAudio)
    }

    func testLateGenerationDonePreservesTheAlreadyPlayedItem() throws {
        var ledger = AudioPlaybackLedger()
        let token = try ledger.schedule(responseId: "r", itemId: "i", samples: 2400)
        XCTAssertEqual(ledger.playedBack(token).map(\.state), [.started])
        XCTAssertEqual(ledger.generationFinished(responseId: "r", itemId: "i").map(\.state), [.completed])
    }

    func testOutOfOrderBufferCallbacksNeverInventAPlayedPrefix() throws {
        var ledger = AudioPlaybackLedger()
        let first = try ledger.schedule(responseId: "r", itemId: "i", samples: 2400)
        let second = try ledger.schedule(responseId: "r", itemId: "i", samples: 4800)
        _ = ledger.generationFinished(responseId: "r", itemId: "i")
        XCTAssertTrue(ledger.playedBack(second).isEmpty)
        XCTAssertEqual(ledger.truncationSnapshot(sampleRate: 24000)?.audioEndMs, 0)
        XCTAssertEqual(ledger.playedBack(first).last?.playedSamples, 7200)
    }

    func testNewItemDoesNotRetireBuffersFromThePreviousItem() throws {
        var ledger = AudioPlaybackLedger()
        let first = try ledger.schedule(responseId: "r", itemId: "one", samples: 2400)
        let second = try ledger.schedule(responseId: "r", itemId: "two", samples: 4800)
        _ = ledger.generationFinished(responseId: "r", itemId: "one")
        _ = ledger.generationFinished(responseId: "r", itemId: "two")
        XCTAssertEqual(ledger.playedBack(first).last?.itemId, "one")
        XCTAssertTrue(ledger.hasPendingAudio)
        XCTAssertEqual(ledger.playedBack(second).last?.itemId, "two")
        XCTAssertFalse(ledger.hasPendingAudio)
    }

    func testStopFencesCallbacksAndLateMediaButConnectionResetCanReuseIDs() throws {
        var ledger = AudioPlaybackLedger()
        let old = try ledger.schedule(responseId: "r", itemId: "i", samples: 2400)
        XCTAssertEqual(ledger.stop().first?.playedSamples, 0)
        XCTAssertTrue(ledger.playedBack(old).isEmpty)
        XCTAssertThrowsError(try ledger.schedule(responseId: "r", itemId: "i", samples: 2400))
        ledger.resetConnection()
        let current = try ledger.schedule(responseId: "r", itemId: "i", samples: 2400)
        XCTAssertTrue(ledger.playedBack(old).isEmpty)
        XCTAssertEqual(ledger.playedBack(current).first?.state, .started)
    }

    func testInterruptedBoundaryUsesPlayedBuffersNotQueuedDuration() throws {
        var ledger = AudioPlaybackLedger()
        let first = try ledger.schedule(responseId: "r", itemId: "i", samples: 1200)
        _ = try ledger.schedule(responseId: "r", itemId: "i", samples: 24000)
        _ = ledger.playedBack(first)
        XCTAssertEqual(ledger.truncationSnapshot(sampleRate: 24000)?.audioEndMs, 50)
        XCTAssertEqual(ledger.stop().first?.playedSamples, 1200)
    }

    func testIdentityAndCompletedItemRejectAdditionalData() throws {
        var ledger = AudioPlaybackLedger()
        _ = try ledger.schedule(responseId: "r", itemId: "i", samples: 2400)
        XCTAssertThrowsError(try ledger.schedule(responseId: "different", itemId: "i", samples: 1))
        _ = ledger.generationFinished(responseId: "r", itemId: "i")
        XCTAssertThrowsError(try ledger.schedule(responseId: "r", itemId: "i", samples: 1))
    }

    func testRetainedCompletedItemsAreBounded() throws {
        var ledger = AudioPlaybackLedger()
        for index in 0..<270 {
            let token = try ledger.schedule(responseId: "r", itemId: "\(index)", samples: 1)
            _ = ledger.generationFinished(responseId: "r", itemId: "\(index)")
            _ = ledger.playedBack(token)
        }
        XCTAssertEqual(ledger.retainedItemCount, 256)
    }

    @MainActor
    func testSimulatorUsesActualPlayedBackCallbacksForTwoQueuedItems() async throws {
        let playback = AudioPlaybackService()
        defer { playback.stop() }
        let completed = expectation(description: "Two actual played-back items")
        completed.expectedFulfillmentCount = 2
        var counts: [String: Int] = [:]
        playback.onPlaybackReceipt = { receipt in
            if receipt.state == .completed {
                counts[receipt.itemId] = receipt.playedSamples
                completed.fulfill()
            }
        }
        let silence = Data(repeating: 0, count: 4800).base64EncodedString()
        try playback.enqueuePCM16(base64: silence, responseId: "sim-r", itemId: "one")
        try playback.enqueuePCM16(base64: silence, responseId: "sim-r", itemId: "two")
        playback.generationFinished(responseId: "sim-r", itemId: "one")
        playback.generationFinished(responseId: "sim-r", itemId: "two")
        await fulfillment(of: [completed], timeout: 15)
        XCTAssertEqual(counts, ["one": 2400, "two": 2400])
    }

    @MainActor
    func testSimulatorStopCannotBecomeAFalsePlayedBackCompletion() async throws {
        let playback = AudioPlaybackService()
        var receipts: [AudioPlaybackReceipt] = []
        playback.onPlaybackReceipt = { receipts.append($0) }
        try playback.enqueuePCM16(base64: Data(repeating: 0, count: 48000).base64EncodedString(),
                                  responseId: "sim-stop", itemId: "item")
        playback.generationFinished(responseId: "sim-stop", itemId: "item")
        playback.stop()
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertFalse(receipts.contains { $0.state == .completed })
        XCTAssertEqual(receipts.last?.state, .interrupted)
        XCTAssertFalse(playback.isPlaying)
    }
}
