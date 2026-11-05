import Foundation
import XCTest

final class JarvisVoiceRouteTests: XCTestCase {
    private let sessionID = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
    private let routeID = UUID(uuidString: "22222222-2222-4222-8222-222222222222")!
    private let audioID = "fixture-audio"

    private func route(generation: Int = 1) -> [String: Any] {
        ["scope": ["owner_id": "fixture-owner", "session_id": sessionID.uuidString, "area_id": NSNull()],
         "route_id": routeID.uuidString, "route_generation": generation]
    }

    private func event(_ values: [String: Any]) -> JarvisRealtimeServerEvent {
        JarvisRealtimeServerEvent(type: values["type"] as? String ?? "", values: values)
    }

    private func selection(_ kind: JarvisVoiceMediaKind = .nativeWebRTC, generation: Int = 1) -> [String: Any] {
        ["type": "voice_route.selected", "protocol_version": 1, "route": route(generation: generation),
         "audio_session_id": audioID, "capabilities": [
            "transport": kind.transport, "media_topology": kind.topology, "control_owner": "backend",
            "credential_mode": "server_only", "input_codec": kind.codec, "output_codec": kind.codec,
            "vad_owner": "provider", "explicit_input_commit": "supported", "explicit_response_create": "supported",
            "full_duplex": "supported", "image_input": "unsupported", "provider_context_receipts": "unsupported",
            "notification_output_attribution": "supported",
            "client_playback_evidence": kind.supportsRenderReceipts ? "supported" : "unsupported",
            "provider_interruption": "supported",
            "operations": ["start", "publish_context", "offer_notification", "retire_context", "interrupt_speech", "close"].map {
                ["operation": $0, "support": "supported"]
            },
         ]]
    }

    @MainActor
    private func ready(_ kind: JarvisVoiceMediaKind = .nativeWebRTC,
                       fence: @escaping (Bool) -> Bool = { _ in true }) throws -> JarvisVoiceRouteDriver {
        let driver = JarvisVoiceRouteDriver(sessionID: sessionID, connectionID: UUID(), mediaKind: kind, fenceMedia: fence)
        _ = try driver.select(event(selection(kind)))
        try driver.didWriteReady(connectionID: driver.connectionID)
        XCTAssertTrue(try driver.observeSessionReady(audioSessionID: audioID))
        return driver
    }

    private func command(_ type: String, id: UUID = UUID(), generation: Int = 1) -> [String: Any] {
        ["type": type, "protocol_version": 1, "audio_session_id": audioID,
         "payload": ["route": route(generation: generation), "command_id": id.uuidString,
                     "requested_at": "2026-10-30T16:00:00Z", "reason": "fixture_interrupt"]]
    }

    private func binding(output: UUID = UUID(), attempt: UUID = UUID(), offer: UUID = UUID(), generation: Int = 1) -> [String: Any] {
        ["type": "voice_route.output_bound", "protocol_version": 1, "audio_session_id": audioID,
         "route": route(generation: generation), "offer_id": offer.uuidString, "attempt_id": attempt.uuidString,
         "attempt_generation": 4, "output": ["output_id": output.uuidString,
            "provider_response_ref": "response", "provider_item_ref": "item"]]
    }

    func testNativeCapabilitiesNeverInventRenderEvidence() throws {
        let native = JarvisVoiceMediaKind.nativeWebRTC.clientCapabilities
        XCTAssertEqual(native["client_playback_boundaries"] as? [String], [])
        XCTAssertEqual(native["supported_versions"] as? [Int], [1])
        XCTAssertEqual(native["supported_transports"] as? [String], ["native_voice"])
        XCTAssertNil(native["api_key"])
        var selected = selection()
        var capabilities = try XCTUnwrap(selected["capabilities"] as? [String: Any])
        capabilities["client_playback_evidence"] = "supported"
        selected["capabilities"] = capabilities
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selected), mediaKind: .nativeWebRTC))
    }

    func testStrictSelectionRejectsUnknownFieldsBooleansAndUnsupportedOperations() throws {
        var selected = selection()
        selected["api_key"] = "must_not_be_accepted"
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selected), mediaKind: .nativeWebRTC))
        selected = selection()
        selected["protocol_version"] = true
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selected), mediaKind: .nativeWebRTC))
        selected = selection()
        var capabilities = try XCTUnwrap(selected["capabilities"] as? [String: Any])
        var operations = try XCTUnwrap(capabilities["operations"] as? [[String: Any]])
        operations[0] = ["operation": "start", "support": "unsupported", "reason": "not_available"]
        capabilities["operations"] = operations
        selected["capabilities"] = capabilities
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selected), mediaKind: .nativeWebRTC))
        operations[0] = operations[1]
        capabilities["operations"] = operations
        selected["capabilities"] = capabilities
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selected), mediaKind: .nativeWebRTC))
    }

    func testDriverSelectionRejectsImplicitTransportSwitch() {
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selection(.streamedPCM)), mediaKind: .nativeWebRTC))
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(selection()), mediaKind: .streamedPCM))
    }

    @MainActor
    func testReadinessRequiresNegotiationWriteAndMatchingSession() throws {
        var fences = 0
        let driver = JarvisVoiceRouteDriver(sessionID: sessionID, connectionID: UUID(), mediaKind: .nativeWebRTC) { _ in
            fences += 1
            return true
        }
        XCTAssertFalse(try driver.observeSessionReady(audioSessionID: audioID))
        let message = try driver.select(event(selection()))
        XCTAssertEqual(message["type"] as? String, "voice_route.ready")
        XCTAssertEqual(fences, 1)
        XCTAssertFalse(driver.isReady)
        XCTAssertThrowsError(try driver.didWriteReady(connectionID: UUID()))
        try driver.didWriteReady(connectionID: driver.connectionID)
        XCTAssertTrue(driver.isReady)
        XCTAssertThrowsError(try driver.observeSessionReady(audioSessionID: "another"))
        _ = try driver.select(event(selection()))
        XCTAssertEqual(fences, 1)
        XCTAssertThrowsError(try driver.select(event(selection(generation: 2))))
    }

    @MainActor
    func testScopeAndUnconfirmedFenceCannotAdmitRoute() throws {
        let wrongScope = JarvisVoiceRouteDriver(sessionID: UUID(), connectionID: UUID(), mediaKind: .nativeWebRTC) { _ in true }
        XCTAssertThrowsError(try wrongScope.select(event(selection())))
        let unfenced = JarvisVoiceRouteDriver(sessionID: sessionID, connectionID: UUID(), mediaKind: .nativeWebRTC) { _ in false }
        XCTAssertThrowsError(try unfenced.select(event(selection())))
        XCTAssertNil(unfenced.selection)
        XCTAssertFalse(unfenced.isReady)
    }

    @MainActor
    func testNativeOutputRejectsOtherRouteAndSecondPCMSpeaker() throws {
        let driver = try ready()
        var media: [String: Any] = ["type": "audio.webrtc.response", "protocol_version": 1,
                                  "audio_session_id": audioID, "route": route(), "response_id": "response"]
        XCTAssertTrue(driver.acceptsMedia(event(media)))
        media["type"] = "audio.output.delta"
        XCTAssertFalse(driver.acceptsMedia(event(media)))
        media["type"] = "audio.webrtc.response"
        media["route"] = route(generation: 2)
        XCTAssertFalse(driver.acceptsMedia(event(media)))
        media.removeValue(forKey: "route")
        XCTAssertFalse(driver.acceptsMedia(event(media)))
    }

    @MainActor
    func testInterruptionFencesLocallyBeforeAnyNetworkCallAndRequiresMatchingAcknowledgment() throws {
        var fences = 0
        let driver = try ready { _ in fences += 1; return true }
        let message = try XCTUnwrap(driver.interruptLocally())
        XCTAssertEqual(fences, 2)
        XCTAssertNil(try driver.interruptLocally())
        let media = event(["type": "audio.webrtc.response", "protocol_version": 1, "audio_session_id": audioID,
                           "route": route(), "response_id": "response"])
        XCTAssertFalse(driver.acceptsMedia(media))
        let payload = try XCTUnwrap(message["payload"] as? [String: Any])
        let receipt: [String: Any] = ["type": "voice_route.event", "protocol_version": 1, "audio_session_id": audioID,
            "payload": ["kind": "command_receipt", "route": route(), "operation": "interrupt_speech",
                        "command_id": payload["command_id"]!, "status": "completed", "output_fenced": true]]
        try driver.receiveEvent(event(receipt))
        XCTAssertTrue(driver.acceptsMedia(media))
    }

    @MainActor
    func testCloseReceiptReportsLocalFenceNotProviderStopOrConversationClosure() throws {
        var closes = 0
        let driver = try ready { close in if close { closes += 1 }; return true }
        let close = command("voice_route.close")
        let receipt = try driver.handleCommand(event(close))
        let payload = try XCTUnwrap(receipt["payload"] as? [String: Any])
        XCTAssertEqual(payload["status"] as? String, "completed")
        XCTAssertEqual(payload["output_fenced"] as? Bool, true)
        XCTAssertTrue(payload["provider_stopped"] is NSNull)
        XCTAssertEqual(payload["operation"] as? String, "close")
        XCTAssertTrue(driver.isClosed)
        XCTAssertFalse(driver.isReady)
        XCTAssertEqual(closes, 1)
        _ = try driver.handleCommand(event(close))
        XCTAssertEqual(closes, 1)
        XCTAssertThrowsError(try driver.select(event(selection(generation: 2))))
    }

    @MainActor
    func testFailedFenceIsUnknownNotCompleted() throws {
        var canFence = true
        let driver = try ready { _ in canFence }
        canFence = false
        let receipt = try driver.handleCommand(event(command("voice_route.close")))
        let payload = try XCTUnwrap(receipt["payload"] as? [String: Any])
        XCTAssertEqual(payload["status"] as? String, "unknown")
        XCTAssertEqual(payload["output_fenced"] as? Bool, false)
        XCTAssertTrue(driver.isClosed)
    }

    @MainActor
    func testOldGenerationCommandsAndChangedCommandReplayAreRejected() throws {
        let driver = try ready()
        XCTAssertThrowsError(try driver.handleCommand(event(command("voice_route.interrupt", generation: 2))))
        let original = command("voice_route.interrupt")
        _ = try driver.handleCommand(event(original))
        var changed = original
        var payload = try XCTUnwrap(changed["payload"] as? [String: Any])
        payload["reason"] = "different_reason"
        changed["payload"] = payload
        XCTAssertThrowsError(try driver.handleCommand(event(changed)))
    }

    @MainActor
    func testPCMPlaybackRequiresActualContiguousRenderAndExplicitAttribution() throws {
        let driver = try ready(.streamedPCM)
        var ledger = AudioPlaybackLedger()
        let token = try ledger.schedule(responseId: "response", itemId: "item", samples: 2400)
        XCTAssertTrue(ledger.generationFinished(responseId: "response", itemId: "item").isEmpty)
        let rendered = try XCTUnwrap(ledger.playedBack(token).first)
        XCTAssertTrue(driver.playbackMessages(rendered).isEmpty)
        try driver.bindOutput(event(binding()))
        let message = try XCTUnwrap(driver.playbackMessages(rendered).first)
        let playback = try XCTUnwrap(message["playback"] as? [String: Any])
        XCTAssertEqual(playback["boundary"] as? String, "client_render_lower_bound")
        XCTAssertEqual(playback["rendered_ms"] as? Int, 100)
        XCTAssertNil(playback["human_heard"])
        XCTAssertNil(playback["completed"])
        XCTAssertEqual(message["attempt_generation"] as? Int, 4)
        XCTAssertNil(message["played_samples"])
        XCTAssertTrue(driver.playbackMessages(rendered).isEmpty)
        driver.close()
        XCTAssertTrue(driver.playbackMessages(rendered).isEmpty)
    }

    @MainActor
    func testCombinedOutputRequiresSeparateBindingsAndNeverUsesTimingAttribution() throws {
        let driver = try ready(.streamedPCM)
        let output = UUID()
        try driver.bindOutput(event(binding(output: output)))
        try driver.bindOutput(event(binding(output: output)))
        let missing = AudioPlaybackReceipt(responseId: "other", itemId: "item", playedSamples: 2400, state: .progress)
        XCTAssertTrue(driver.playbackMessages(missing).isEmpty)
        let actual = AudioPlaybackReceipt(responseId: "response", itemId: "item", playedSamples: 2400, state: .progress)
        XCTAssertEqual(driver.playbackMessages(actual).count, 2)
        XCTAssertTrue(driver.playbackMessages(actual).isEmpty)
        XCTAssertThrowsError(try driver.bindOutput(event(binding(generation: 2))))
    }

    @MainActor
    func testRetiredOutputCannotBeReboundByLateCallback() throws {
        let driver = try ready(.streamedPCM)
        let bound = binding()
        try driver.bindOutput(event(bound))
        _ = try driver.handleCommand(event(command("voice_route.interrupt")))
        XCTAssertThrowsError(try driver.bindOutput(event(bound)))
        let actual = AudioPlaybackReceipt(responseId: "response", itemId: "item", playedSamples: 2400, state: .completed)
        XCTAssertTrue(driver.playbackMessages(actual).isEmpty)
    }

    private func statusEvent(_ kind: String = "route_error", generation: Int = 1,
                             fields: [String: Any] = ["code": "voice_route_unavailable", "message": "private provider detail", "recoverable": false]) -> JarvisRealtimeServerEvent {
        var payload: [String: Any] = ["kind": kind, "route": route(generation: generation), "observed_at": "2026-10-31T16:00:00Z"]
        payload.merge(fields) { _, new in new }
        return event(["type": "voice_route.event", "protocol_version": 1, "audio_session_id": audioID, "payload": payload])
    }

    @MainActor
    func testNonrecoverableRouteErrorIsVisibleWithoutChangingMediaOrTranscript() throws {
        var fences = 0
        let driver = try ready { _ in fences += 1; return true }
        let before = fences
        try driver.receiveEvent(statusEvent())
        XCTAssertEqual(driver.failure, .nonrecoverableRouteError)
        XCTAssertEqual(JarvisVoiceRouteFailure.displayText(driver.failure, transcript: "question"), driver.failure?.message)
        XCTAssertTrue(driver.isReady)
        XCTAssertFalse(driver.isClosed)
        XCTAssertEqual(fences, before)
        XCTAssertTrue(driver.acceptsMedia(event(["type": "audio.webrtc.response", "protocol_version": 1,
            "audio_session_id": audioID, "route": route(), "response_id": "already-authorized"])))
    }

    @MainActor
    func testFailedConnectionProjectsStatusWithoutInventingProviderDetails() throws {
        let driver = try ready()
        try driver.receiveEvent(statusEvent("connection_state", fields: ["state": "failed", "reason": "private detail"]))
        XCTAssertEqual(driver.failure, .connectionFailed)
        let evidence = JarvisVoiceRouteFailure.evidence(driver.failure)
        XCTAssertEqual(evidence["status"] as? String, "unavailable")
        XCTAssertEqual(evidence["failure_source"] as? String, "connection_failed")
        XCTAssertEqual(Set(evidence.keys), ["schema_version", "status", "failure_source"])
        let encoded = String(data: try JSONSerialization.data(withJSONObject: evidence), encoding: .utf8)!
        XCTAssertFalse(encoded.contains("private"))
        XCTAssertFalse(driver.failure!.message.contains("private"))
    }

    @MainActor
    func testRecoverableRefreshAndOrdinaryStartupNeverBecomeFatalNotice() throws {
        let driver = try ready()
        for state in ["starting", "ready", "closing", "closed"] {
            try driver.receiveEvent(statusEvent("connection_state", fields: ["state": state, "reason": NSNull()]))
        }
        try driver.receiveEvent(statusEvent(fields: ["code": "context_refresh_pending", "message": "refreshing", "recoverable": true]))
        try driver.receiveEvent(statusEvent("context_receipt", fields: ["status": "unconfirmed"]))
        XCTAssertNil(driver.failure)
        XCTAssertEqual(JarvisVoiceRouteFailure.displayText(nil, transcript: "question"), "question")
        XCTAssertEqual(JarvisVoiceRouteFailure.evidence(nil)["status"] as? String, "not_reported")
        XCTAssertTrue(JarvisVoiceRouteFailure.evidence(nil)["failure_source"] is NSNull)
    }

    @MainActor
    func testMalformedFailureCannotPromoteRefreshToFatalOrChangeProtocolPolicy() throws {
        let driver = try ready()
        for invalid: Any in [0, "false", NSNull()] {
            try driver.receiveEvent(statusEvent(fields: ["code": "x", "message": "x", "recoverable": invalid]))
        }
        try driver.receiveEvent(statusEvent(fields: ["code": "x", "message": "x", "recoverable": false, "extra": "secret"]))
        try driver.receiveEvent(statusEvent(fields: ["code": "x", "message": String(repeating: "x", count: 513), "recoverable": false]))
        try driver.receiveEvent(statusEvent(fields: ["code": "x", "message": "x", "recoverable": false, "observed_at": "invalid"]))
        XCTAssertNil(driver.failure)
        XCTAssertTrue(driver.isReady)
    }

    @MainActor
    func testLateWrongGenerationOrClosedDriverCannotMutateStatus() throws {
        let driver = try ready()
        XCTAssertThrowsError(try driver.receiveEvent(statusEvent(generation: 2)))
        XCTAssertNil(driver.failure)
        driver.close()
        XCTAssertThrowsError(try driver.receiveEvent(statusEvent()))
        XCTAssertNil(driver.failure)
    }

    @MainActor
    func testDuplicateFailureIsLatchedUntilNewDriverWithoutExtraFence() throws {
        var fences = 0
        let driver = try ready { _ in fences += 1; return true }
        let before = fences
        try driver.receiveEvent(statusEvent())
        try driver.receiveEvent(statusEvent())
        try driver.receiveEvent(statusEvent("connection_state", fields: ["state": "ready"]))
        try driver.receiveEvent(statusEvent("connection_state", fields: ["state": "failed"]))
        XCTAssertEqual(driver.failure, .nonrecoverableRouteError)
        XCTAssertEqual(fences, before)
        driver.close()
        XCTAssertEqual(driver.failure, .nonrecoverableRouteError)
        XCTAssertNil(try ready().failure)
    }

    func testTimingBoundariesRemainSeparateAndContentFree() {
        let fields = RealtimeDiagnostics.safeFields([
            "event_type": "audio.input.started", "route_generation": 4, "actor_wait_ms": 19_000,
            "handler_elapsed_ms": 5, "decode_elapsed_ms": 0.2, "receive_monotonic_s": 30,
            "write_elapsed_ms": 12, "text": "private conversation", "route_id": "route-fixture",
        ])
        XCTAssertEqual(fields["actor_wait_ms"] as? Int, 19_000)
        XCTAssertEqual(fields["handler_elapsed_ms"] as? Int, 5)
        XCTAssertNil(fields["text"])
        XCTAssertNil(fields["network_latency_ms"])
        XCTAssertNil(fields["heard_at"])
    }
}
