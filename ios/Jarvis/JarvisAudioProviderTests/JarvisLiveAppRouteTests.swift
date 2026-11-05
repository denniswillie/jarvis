import Foundation
import XCTest

@MainActor
final class JarvisLiveAppRouteTests: XCTestCase {
    private let session = UUID()
    private let routeID = UUID()
    private let command = UUID()
    private let audio = "app-live-audio"
    private let provider = "live_app_fixture"

    private func route(generation: Int = 1) -> [String: Any] {
        ["scope": ["owner_id": "fixture", "session_id": session.uuidString, "area_id": NSNull()],
         "route_id": routeID.uuidString, "route_generation": generation]
    }
    private func event(_ values: [String: Any]) -> JarvisRealtimeServerEvent {
        JarvisRealtimeServerEvent(type: values["type"] as? String ?? "", values: values)
    }
    private func selected() -> JarvisRealtimeServerEvent {
        event(["type": "voice_route.selected", "protocol_version": 1, "route": route(),
               "audio_session_id": audio, "capabilities": JarvisLiveBackendCapabilityFixture.capabilities])
    }
    private func mediaCommand(_ phase: String) -> JarvisRealtimeServerEvent {
        var value: [String: Any] = ["type": "live.media." + phase, "protocol_version": 1,
                                    "route": route(), "audio_session_id": audio,
                                    "command_id": command.uuidString, "provider_session_id": provider]
        if phase == "prepare" { value["answer_sdp"] = "v=0\r\n" }
        return event(value)
    }
    private func routeEvent(_ payload: [String: Any], generation: Int = 1) -> JarvisRealtimeServerEvent {
        var body = payload
        body["route"] = route(generation: generation)
        body["observed_at"] = "2026-10-31T16:06:43.709174Z"
        return event(["type": "voice_route.event", "protocol_version": 1, "audio_session_id": audio, "payload": body])
    }
    private func serialized(_ value: JarvisRealtimeServerEvent) throws -> JarvisRealtimeServerEvent {
        let data = try JSONSerialization.data(withJSONObject: value.values, options: .sortedKeys)
        return event(try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }
    private var canonicalRouteError: [String: Any] {
        // Controlled canonical Python serialization, not a retained native event.
        ["kind": "route_error", "code": "live_reader_failed",
         "message": "PRIVATE_FAILURE_TEXT", "recoverable": false]
    }
    private var canonicalCaption: [String: Any] {
        ["kind": "transcript_revision", "transcript_id": "77777777-7777-4777-8777-777777777777",
         "revision": 1, "role": "assistant", "text": "PRIVATE_SYNTHETIC_CAPTION",
         "final": false, "output": NSNull()]
    }
    private func canonicalContext(_ status: String) -> [String: Any] {
        ["kind": "context_receipt", "update_id": "44444444-4444-4444-8444-444444444444",
         "cursor": ["epoch": "33333333-3333-4333-8333-333333333333", "revision": 4],
         "status": status, "provider_receipt_ref": status == "provider_acknowledged" ? "PRIVATE_PROVIDER_REFERENCE" : NSNull(),
         "reason": NSNull()]
    }
    private func stopCommand(_ type: String) -> JarvisRealtimeServerEvent {
        event(["type": type, "protocol_version": 1, "audio_session_id": audio,
               "payload": ["route": route(), "command_id": UUID().uuidString,
                           "requested_at": "2026-10-30T21:00:00Z", "reason": "fixture"]])
    }
    private func make(_ mode: JarvisAudioCaptureMode = .autoDetect) throws -> AppRouteRig {
        let rig = AppRouteRig()
        rig.driver = JarvisLiveVoiceRouteDriver(sessionID: session, connectionID: rig.connection,
            media: rig.media, ownership: JarvisLiveMediaOwnership(), fencePrevious: { rig.previousFences += 1; return true })
        rig.owner = try JarvisLiveAppRoute(driver: rig.driver, captureMode: mode,
            currentConnection: { rig.currentConnection }, write: { payload, connection, current in
                try Task.checkCancellation()
                guard connection == rig.currentConnection, current() else { throw JarvisVoiceRouteError.identityMismatch }
                let text = try JarvisLiveAppWire.encode(payload)
                rig.messages.append(try JSONSerialization.jsonObject(with: Data(text.utf8)) as! [String: Any])
            })
        rig.owner.onReady = { rig.ready.append($0) }
        rig.owner.onCaption = { rig.captions.append(($0, $1)) }
        rig.owner.onFailure = { rig.failures.append($0) }
        rig.owner.onStopped = { rig.stops.append($0) }
        rig.owner.onProtocolFailure = { rig.protocolFailures.append($0) }
        return rig
    }
    private func until(_ predicate: @MainActor () -> Bool, file: StaticString = #filePath, line: UInt = #line) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !predicate() {
            guard ContinuousClock.now < deadline else { XCTFail("Condition timed out", file: file, line: line); throw JarvisVoiceRouteError.notReady }
            await Task.yield()
        }
    }
    private func prepare(_ rig: AppRouteRig) async throws {
        try await rig.owner.start()
        XCTAssertTrue(try rig.owner.receive(selected(), connectionID: rig.connection))
        try await until { rig.messages.contains { $0["type"] as? String == "voice_route.ready" } }
        _ = try rig.owner.receive(mediaCommand("prepare"), connectionID: rig.connection)
        try await until { rig.media.hasRemoteDescription }
        rig.media.onProviderStarted?(event(["type": "session.started", "session": ["id": provider]]))
        try await until { rig.messages.contains { $0["phase"] as? String == "prepare" } }
    }
    private func activate(_ rig: AppRouteRig) async throws {
        _ = try rig.owner.receive(mediaCommand("activate"), connectionID: rig.connection)
        try await until { rig.messages.contains { $0["phase"] as? String == "activate" } }
    }
    private func ready(_ rig: AppRouteRig) throws {
        _ = try rig.owner.receive(routeEvent(["kind": "connection_state", "state": "ready"]), connectionID: rig.connection)
    }

    func testRecordingMediaRequiresReadyCurrentConnectionAndRetiresOnClose() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        XCTAssertNil(rig.owner.activeMediaForRecording)
        try await prepare(rig); try await activate(rig)
        XCTAssertNil(rig.owner.activeMediaForRecording)
        try ready(rig)
        try await until { rig.owner.isReady }
        XCTAssertNil(rig.owner.activeMediaForRecording)
        _ = try rig.owner.receive(event(["type": "session.created", "session_id": UUID().uuidString,
            "audio_session_id": audio, "recording_protocol_version": 1]), connectionID: rig.connection)
        XCTAssertNil(rig.owner.activeMediaForRecording)
        _ = try rig.owner.receive(event(["type": "session.created", "session_id": session.uuidString,
            "audio_session_id": audio, "recording_protocol_version": 1]), connectionID: rig.connection)
        XCTAssertTrue(rig.owner.activeMediaForRecording === rig.media)
        let original = rig.currentConnection
        rig.currentConnection = UUID()
        XCTAssertNil(rig.owner.activeMediaForRecording)
        rig.currentConnection = original
        XCTAssertTrue(rig.owner.activeMediaForRecording === rig.media)
        _ = rig.owner.close()
        XCTAssertNil(rig.owner.activeMediaForRecording)
        XCTAssertNil(rig.owner.recordingProtocolVersion)
    }

    func testRecordingMediaSupportsVersion2AndRejectsUnknownVersions() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        for version in [2, 99, 1] {
            _ = try rig.owner.receive(event(["type": "session.created", "session_id": session.uuidString,
                "audio_session_id": audio, "recording_protocol_version": version]), connectionID: rig.connection)
            if version == 99 {
                XCTAssertNil(rig.owner.recordingProtocolVersion)
                XCTAssertNil(rig.owner.activeMediaForRecording)
            } else {
                XCTAssertEqual(rig.owner.recordingProtocolVersion, version)
                XCTAssertTrue(rig.owner.activeMediaForRecording === rig.media)
            }
        }
    }

    func testCanonicalTimestampedRouteErrorStopsStartupWithoutLeakingMessage() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        let failure = try serialized(routeEvent(canonicalRouteError))
        let payload = try XCTUnwrap(failure.values["payload"] as? [String: Any])
        XCTAssertEqual(Set(payload.keys), Set(["kind", "route", "observed_at", "code", "message", "recoverable"]))
        XCTAssertEqual(payload["observed_at"] as? String, "2026-10-31T16:06:43.709174Z")
        XCTAssertTrue(try rig.owner.receive(failure, connectionID: rig.connection))
        XCTAssertEqual(rig.failures, ["GPT Live reported a route error. Restart to recover."])
        XCTAssertTrue(rig.owner.isClosed)
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertTrue(rig.media.closed)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertEqual(rig.media.activations, 0)
        XCTAssertTrue(rig.ready.isEmpty)
        XCTAssertThrowsError(try rig.owner.receive(mediaCommand("activate"), connectionID: rig.connection))
    }

    func testCanonicalRouteErrorFencesActiveMediaWithoutLeakingItsMessage() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        XCTAssertTrue(rig.media.microphoneEnabled)
        XCTAssertTrue(try rig.owner.receive(try serialized(routeEvent(canonicalRouteError)), connectionID: rig.connection))
        XCTAssertEqual(rig.failures, ["GPT Live reported a route error. Restart to recover."])
        XCTAssertFalse(rig.failures.contains { $0.contains("PRIVATE_FAILURE_TEXT") })
        XCTAssertFalse(rig.media.microphoneEnabled)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertEqual(rig.media.activations, 1)
        XCTAssertEqual(rig.ready, [audio])
        XCTAssertThrowsError(try rig.owner.receive(routeEvent(["kind": "connection_state", "state": "ready"]), connectionID: rig.connection))
    }

    func testCanonicalRouteErrorStillRequiresCurrentRouteAudioAndConnection() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        XCTAssertThrowsError(try rig.owner.receive(try serialized(routeEvent(canonicalRouteError, generation: 2)), connectionID: rig.connection))
        var wrongAudio = routeEvent(canonicalRouteError).values
        wrongAudio["audio_session_id"] = "retired-audio"
        XCTAssertThrowsError(try rig.owner.receive(try serialized(event(wrongAudio)), connectionID: rig.connection))
        XCTAssertThrowsError(try rig.owner.receive(try serialized(routeEvent(canonicalRouteError)), connectionID: UUID()))
        var wrongProtocol = routeEvent(canonicalRouteError).values
        wrongProtocol["protocol_version"] = true
        XCTAssertThrowsError(try rig.owner.receive(try serialized(event(wrongProtocol)), connectionID: rig.connection))
        XCTAssertFalse(rig.owner.isClosed)
        XCTAssertTrue(rig.failures.isEmpty)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertEqual(rig.media.activations, 0)
    }

    func testTimedRouteEventsStillRejectMissingAndMalformedTimestamps() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        let payloads: [[String: Any]] = [
            ["kind": "connection_state", "state": "ready", "reason": NSNull()],
            canonicalContext("written"), canonicalCaption, canonicalRouteError,
            ["kind": "command_receipt", "command_id": command.uuidString, "operation": "start",
             "status": "completed", "output_fenced": NSNull(), "provider_stopped": NSNull(), "reason": NSNull()],
        ]
        for body in payloads {
            for timestamp in [nil, "not-a-timestamp"] as [String?] {
                var values = routeEvent(body).values
                var payload = try XCTUnwrap(values["payload"] as? [String: Any])
                payload.removeValue(forKey: "observed_at")
                if let timestamp { payload["observed_at"] = timestamp }
                values["payload"] = payload
                XCTAssertThrowsError(try rig.owner.receive(try serialized(event(values)), connectionID: rig.connection))
            }
        }
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertTrue(rig.ready.isEmpty)
        XCTAssertTrue(rig.failures.isEmpty)
    }

    func testCanonicalStartupContextReceiptsDoNotReplaceMediaActivation() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await rig.owner.start()
        _ = try rig.owner.receive(try serialized(selected()), connectionID: rig.connection)
        try await until { rig.messages.contains { $0["type"] as? String == "voice_route.ready" } }
        _ = try rig.owner.receive(try serialized(routeEvent(["kind": "connection_state", "state": "starting"])), connectionID: rig.connection)
        _ = try rig.owner.receive(try serialized(mediaCommand("prepare")), connectionID: rig.connection)
        try await until { rig.media.hasRemoteDescription }
        rig.media.onProviderStarted?(event(["type": "session.started", "session": ["id": provider]]))
        try await until { rig.messages.contains { $0["phase"] as? String == "prepare" } }
        for (status, timestamp) in [("queued", "2026-10-31T16:06:43.709174+00:00"),
                                    ("written", "2026-10-31T16:06:43.786603+00:00"),
                                    ("provider_acknowledged", "2026-10-31T16:06:44.000000+00:00")] {
            var values = routeEvent(canonicalContext(status)).values
            var payload = try XCTUnwrap(values["payload"] as? [String: Any])
            let cursor = try XCTUnwrap(payload["cursor"] as? [String: Any])
            XCTAssertEqual(cursor["revision"] as? Int, 4)
            XCTAssertNotNil(UUID(uuidString: try XCTUnwrap(cursor["epoch"] as? String)))
            payload["observed_at"] = timestamp
            values["payload"] = payload
            XCTAssertTrue(try rig.owner.receive(try serialized(event(values)), connectionID: rig.connection))
            XCTAssertFalse(rig.owner.isReady)
            XCTAssertFalse(rig.media.outputEnabled)
            XCTAssertFalse(rig.media.microphoneEnabled)
            XCTAssertEqual(rig.media.activations, 0)
        }
        try await activate(rig)
        XCTAssertFalse(rig.owner.isReady)
        try ready(rig)
        XCTAssertTrue(rig.owner.isReady)
        XCTAssertEqual(rig.ready, [audio])
        XCTAssertEqual(rig.media.activations, 1)
        XCTAssertTrue(rig.failures.isEmpty)
    }

    func testCanonicalProvisionalCaptionBeforeReadinessIsValidatedWithoutMediaOrDisplay() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await rig.owner.start()
        _ = try rig.owner.receive(try serialized(selected()), connectionID: rig.connection)
        try await until { rig.messages.contains { $0["type"] as? String == "voice_route.ready" } }
        let caption = try serialized(routeEvent(canonicalCaption))
        XCTAssertTrue(try rig.owner.receive(caption, connectionID: rig.connection))
        XCTAssertFalse(rig.media.hasRemoteDescription)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertFalse(rig.media.microphoneEnabled)
        XCTAssertEqual(rig.media.activations, 0)
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertTrue(rig.captions.isEmpty)

        _ = try rig.owner.receive(try serialized(mediaCommand("prepare")), connectionID: rig.connection)
        try await until { rig.media.hasRemoteDescription }
        rig.media.onProviderStarted?(event(["type": "session.started", "session": ["id": provider]]))
        try await until { rig.messages.contains { $0["phase"] as? String == "prepare" } }
        XCTAssertTrue(try rig.owner.receive(caption, connectionID: rig.connection))
        try ready(rig)
        XCTAssertTrue(try rig.owner.receive(caption, connectionID: rig.connection))
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertFalse(rig.media.microphoneEnabled)
        XCTAssertEqual(rig.media.activations, 0)
        XCTAssertTrue(rig.captions.isEmpty)

        try await activate(rig)
        XCTAssertTrue(rig.owner.isReady)
        XCTAssertEqual(rig.ready, [audio])
        XCTAssertEqual(rig.media.activations, 1)
        XCTAssertTrue(rig.captions.isEmpty) // Early fragments were not buffered.
        _ = try rig.owner.receive(caption, connectionID: rig.connection)
        _ = try rig.owner.receive(caption, connectionID: rig.connection)
        XCTAssertEqual(rig.captions.count, 1) // Early fragments did not poison dedup.
        XCTAssertEqual(rig.captions.first?.1, "PRIVATE_SYNTHETIC_CAPTION")
        XCTAssertTrue(rig.failures.isEmpty)
        XCTAssertTrue(rig.protocolFailures.isEmpty)
        XCTAssertFalse(rig.messages.contains { ($0["type"] as? String)?.contains("transcript") == true })
    }

    func testEarlyCaptionStillRejectsMalformedFieldsStaleIdentityAndRetiredRoute() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        let invalid: [(String, Any)] = [("role", "tool"), ("text", String(repeating: "x", count: 4097)),
                                      ("transcript_id", "not-a-uuid"), ("revision", true), ("revision", 0)]
        for (field, value) in invalid {
            var payload = canonicalCaption
            payload[field] = value
            XCTAssertThrowsError(try rig.owner.receive(try serialized(routeEvent(payload)), connectionID: rig.connection))
            XCTAssertEqual(rig.protocolFailures.last?.fields["protocol_stage"] as? String, "transcript")
            XCTAssertEqual(rig.protocolFailures.last?.fields["protocol_field"] as? String, field)
        }
        XCTAssertThrowsError(try rig.owner.receive(routeEvent(canonicalCaption, generation: 2), connectionID: rig.connection))
        XCTAssertEqual(rig.protocolFailures.last?.fields["protocol_field"] as? String, "route")
        XCTAssertThrowsError(try rig.owner.receive(routeEvent(canonicalCaption), connectionID: UUID()))
        XCTAssertEqual(rig.protocolFailures.last?.fields["protocol_field"] as? String, "connection_id")
        var wrongAudio = routeEvent(canonicalCaption).values
        wrongAudio["audio_session_id"] = "retired-audio"
        XCTAssertThrowsError(try rig.owner.receive(event(wrongAudio), connectionID: rig.connection))
        XCTAssertEqual(rig.protocolFailures.last?.fields["protocol_field"] as? String, "audio_session_id")
        XCTAssertTrue(try rig.owner.receive(routeEvent(canonicalCaption), connectionID: rig.connection))
        XCTAssertTrue(rig.captions.isEmpty)
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertEqual(rig.media.activations, 0)
        try rig.owner.interruptLocally()
        XCTAssertThrowsError(try rig.owner.receive(routeEvent(canonicalCaption), connectionID: rig.connection))
        XCTAssertEqual(rig.protocolFailures.last?.fields["protocol_field"] as? String, "route_state")
        XCTAssertTrue(rig.captions.isEmpty)
        XCTAssertFalse(rig.media.outputEnabled)
    }

    func testProtocolFailureDiagnosticIsBoundedAndRetainedBeforeCaptureClosure() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        var capture = JarvisReplayPCMState(maxDuration: 45)
        rig.owner.onProtocolFailure = { failure in
            rig.protocolFailures.append(failure)
            capture.recordControlEvent(.backendControlHandled, fields: failure.fields, at: 1)
        }
        var value = routeEvent(canonicalRouteError).values
        var payload = try XCTUnwrap(value["payload"] as? [String: Any])
        payload.removeValue(forKey: "observed_at")
        value["payload"] = payload
        for _ in 0..<40 {
            XCTAssertThrowsError(try rig.owner.receive(event(value), connectionID: rig.connection))
        }
        XCTAssertEqual(rig.protocolFailures.count, 16)
        XCTAssertEqual(capture.evidence.controlEvents.count, 16)
        let fields = try XCTUnwrap(capture.evidence.controlEvents.first?.fields)
        XCTAssertEqual(fields, ["boundary": "live_protocol_failure", "protocol_stage": "payload",
                                "protocol_field": "observed_at", "protocol_reason": "invalid_wire",
                                "outer_type": "voice_route.event", "nested_type": "route_error"])
        XCTAssertFalse(fields.values.contains { $0.contains("PRIVATE") })
        XCTAssertFalse(capture.evidence.controlEventTimingInvalid)
        XCTAssertFalse(capture.evidence.controlEventsTruncated)
        XCTAssertTrue(rig.failures.isEmpty) // Original thrown error remains the caller's failure boundary.
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertFalse(rig.media.outputEnabled)
    }

    func testPreferenceDefaultsToLiveAndPreservesExplicitChoices() throws {
        let name = "JarvisLiveAppRouteTests." + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        XCTAssertNil(defaults.object(forKey: JarvisVoiceRouteChoice.preferenceKey))
        XCTAssertEqual(JarvisVoiceRouteChoice.load(from: defaults), .live)
        XCTAssertNil(defaults.object(forKey: JarvisVoiceRouteChoice.preferenceKey))
        JarvisVoiceRouteChoice.live.save(to: defaults)
        XCTAssertEqual(JarvisVoiceRouteChoice.load(from: defaults), .live)
        JarvisVoiceRouteChoice.realtime.save(to: defaults)
        XCTAssertEqual(JarvisVoiceRouteChoice.load(from: defaults), .realtime)
        XCTAssertEqual(defaults.string(forKey: JarvisVoiceRouteChoice.preferenceKey), "realtime")
        defaults.removeObject(forKey: JarvisVoiceRouteChoice.preferenceKey)
        XCTAssertEqual(JarvisVoiceRouteChoice.load(from: defaults), .live)
    }

    func testExplicitSelectorPreservesEndpointAndRollbackRemovesLive() throws {
        let raw = "wss://example.invalid/realtime/audio/ws?session_id=fixture&audio_provider=gemini&voice_route=old&voice_route=live"
        let live = try JarvisVoiceRouteChoice.live.audioURL(from: raw, directAudio: true)
        let query = try XCTUnwrap(URLComponents(url: live, resolvingAgainstBaseURL: false)?.queryItems)
        XCTAssertEqual(live.path, "/realtime/audio/ws")
        XCTAssertEqual(query.filter { $0.name == "voice_route" }.map(\.value), ["live"])
        XCTAssertEqual(query.first { $0.name == "audio_provider" }?.value, "openai")
        XCTAssertEqual(query.first { $0.name == "audio_transport" }?.value, "webrtc")
        XCTAssertEqual(query.first { $0.name == "session_id" }?.value, "fixture")
        let rollback = try JarvisVoiceRouteChoice.realtime.audioURL(from: live.absoluteString, directAudio: true)
        XCTAssertFalse(rollback.absoluteString.contains("voice_route="))
        XCTAssertThrowsError(try JarvisVoiceRouteChoice.live.audioURL(from: raw, directAudio: false))
        XCTAssertThrowsError(try JarvisVoiceRouteChoice.live.audioURL(from: "https://example.invalid", directAudio: true))
    }

    func testProductionOwnerFirstMessageIsAutoSDPOfferWithLiveCapabilities() async throws {
        let rig = try make(.holdToTalk)
        defer { rig.owner.close() }
        try await rig.owner.start()
        XCTAssertEqual(rig.previousFences, 1)
        XCTAssertEqual(rig.messages.count, 1)
        let offer = try XCTUnwrap(rig.messages.first)
        XCTAssertEqual(Set(offer.keys), Set(["type", "sdp", "capture_mode", "voice_route"]))
        XCTAssertEqual(offer["type"] as? String, "audio.webrtc.offer")
        XCTAssertEqual(offer["capture_mode"] as? String, "auto")
        let caps = try XCTUnwrap(offer["voice_route"] as? [String: Any])
        XCTAssertEqual(caps["input_codecs"] as? [String], ["webrtc_negotiated"])
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertFalse(rig.media.microphoneEnabled)
        XCTAssertFalse(rig.owner.isReady)
    }

    func testConnectedNeedsBothActualMediaActivationAndBackendReady() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        XCTAssertFalse(rig.owner.isReady)
        try await activate(rig)
        XCTAssertFalse(rig.owner.isReady)
        try ready(rig)
        XCTAssertTrue(rig.owner.isReady)
        XCTAssertEqual(rig.ready, [audio])
        try ready(rig)
        XCTAssertEqual(rig.ready.count, 1)
    }

    func testEarlyBackendReadyCannotOpenMicrophoneOrReportConnected() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        try ready(rig)
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertFalse(rig.media.microphoneEnabled)
        try await activate(rig)
        try await until { rig.owner.isReady }
    }

    func testLegacySessionCreatedCannotMakeLiveReady() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await rig.owner.start()
        XCTAssertTrue(try rig.owner.receive(event(["type": "session.created", "audio_session_id": audio]), connectionID: rig.connection))
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertEqual(rig.media.activations, 0)
    }

    func testRealtimeSelectionIsRejectedWithoutFallbackMedia() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await rig.owner.start()
        var values = selected().values
        var caps = JarvisLiveBackendCapabilityFixture.capabilities
        caps["input_codec"] = "opus"; caps["output_codec"] = "opus"
        values["capabilities"] = caps
        XCTAssertThrowsError(try rig.owner.receive(event(values), connectionID: rig.connection))
        XCTAssertFalse(rig.media.outputEnabled)
        XCTAssertEqual(rig.messages.count, 1)
    }

    func testRetiredConnectionCannotActivateOrWriteOnReplacement() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        rig.currentConnection = UUID()
        XCTAssertThrowsError(try rig.owner.receive(mediaCommand("activate"), connectionID: rig.connection))
        XCTAssertFalse(rig.owner.isReady)
        XCTAssertEqual(rig.media.activations, 0)
    }

    func testCloseDuringStartupFencesAndRejectsLateMessages() async throws {
        let rig = try make()
        try await prepare(rig)
        XCTAssertTrue(rig.owner.close())
        XCTAssertThrowsError(try rig.owner.receive(mediaCommand("activate"), connectionID: rig.connection))
        XCTAssertTrue(rig.media.closed)
        XCTAssertTrue(rig.ready.isEmpty)
    }

    func testBackendInterruptReturnsFenceReceiptBeforeStoppedCallback() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        _ = try rig.owner.receive(stopCommand("voice_route.interrupt"), connectionID: rig.connection)
        XCTAssertFalse(rig.media.outputEnabled)
        try await until { rig.stops.count == 1 }
        let last = try XCTUnwrap(rig.messages.last)
        XCTAssertEqual(last["type"] as? String, "voice_route.event")
        let payload = try XCTUnwrap(last["payload"] as? [String: Any])
        XCTAssertEqual(payload["output_fenced"] as? Bool, true)
        XCTAssertEqual(payload["operation"] as? String, "interrupt_speech")
        XCTAssertTrue(rig.failures.isEmpty)
        XCTAssertFalse(rig.owner.isReady)
    }

    func testLocalInterruptionDoesNotCancelWorkOrCreateAnotherSpeaker() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        try rig.owner.interruptLocally()
        XCTAssertTrue(rig.media.closed)
        try await until { rig.stops.count == 1 }
        XCTAssertEqual(rig.messages.last?["type"] as? String, "voice_route.interrupt")
        XCTAssertFalse(rig.messages.contains { $0["type"] as? String == "task.cancel" })
        XCTAssertThrowsError(try rig.owner.setCaptureMode(.autoDetect))
    }

    func testManualMicrophoneGateNeverSendsTurnCommitOrCreate() async throws {
        let rig = try make(.holdToTalk)
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        XCTAssertFalse(rig.media.microphoneEnabled)
        let count = rig.messages.count
        try rig.owner.setManualRecording(true)
        XCTAssertTrue(rig.media.microphoneEnabled)
        try rig.owner.setManualRecording(false)
        XCTAssertFalse(rig.media.microphoneEnabled)
        try rig.owner.setCaptureMode(.autoDetect)
        XCTAssertTrue(rig.media.microphoneEnabled)
        XCTAssertEqual(rig.messages.count, count)
    }

    func testProvisionalCaptionUsesBackendIdentityWithoutInventingFinalTurn() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        let id = UUID()
        let caption = routeEvent(["kind": "transcript_revision", "transcript_id": id.uuidString,
                                  "revision": 2, "role": "assistant", "text": "provisional", "final": false])
        _ = try rig.owner.receive(caption, connectionID: rig.connection)
        _ = try rig.owner.receive(caption, connectionID: rig.connection)
        XCTAssertEqual(rig.captions.count, 1)
        XCTAssertEqual(rig.captions.first?.1, "provisional")
        XCTAssertFalse(rig.messages.contains { ($0["type"] as? String)?.contains("transcript") == true })
        XCTAssertThrowsError(try rig.owner.receive(routeEvent(["kind": "connection_state", "state": "ready"], generation: 2), connectionID: rig.connection))
    }

    func testLegacyPCMAndResponseEventsAreConsumedWithoutPlaybackClaims() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig); try ready(rig)
        let count = rig.messages.count
        for type in ["audio.output.delta", "audio.output.done", "audio.webrtc.response", "response.done", "transcript.done", "user.transcript.done"] {
            XCTAssertTrue(try rig.owner.receive(event(["type": type, "text": "not_live_caption"]), connectionID: rig.connection))
        }
        XCTAssertEqual(rig.messages.count, count)
        XCTAssertTrue(rig.captions.isEmpty)
        XCTAssertFalse(try rig.owner.receive(event(["type": "memory.updated"]), connectionID: rig.connection))
    }

    func testUnsupportedNotificationAttributionDoesNotBecomePlaybackReceipt() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig)
        XCTAssertThrowsError(try rig.owner.receive(event(["type": "voice_route.output_bound"]), connectionID: rig.connection))
        XCTAssertFalse(rig.messages.contains { $0["type"] as? String == "voice_route.playback" })
    }

    func testExactMediaReceiptsSurviveRealAppEncoderWithoutLegacyFields() async throws {
        let rig = try make()
        defer { rig.owner.close() }
        try await prepare(rig); try await activate(rig)
        let receipts = rig.messages.filter { $0["type"] as? String == "live.media.receipt" }
        XCTAssertEqual(receipts.count, 2)
        for receipt in receipts {
            XCTAssertEqual(Set(receipt.keys), Set(["type", "protocol_version", "route", "audio_session_id", "command_id",
                                                  "provider_session_id", "phase", "receipt_id", "provider_started", "output_gated"]))
        }
        XCTAssertNotEqual(receipts[0]["receipt_id"] as? String, receipts[1]["receipt_id"] as? String)
        for type in ["session.start", "audio.input.commit", "response.create", "session.input_audio.append"] {
            XCTAssertThrowsError(try JarvisLiveAppWire.encode(["type": type]))
        }
    }

    func testActualClientRefusesUnconnectedOrRetiredLiveWriter() async throws {
        let client = JarvisRealtimeClient()
        do {
            try await client.sendLiveAppEvent(["type": "voice_route.ready"], connectionID: UUID(), current: { true })
            XCTFail("Unconnected writer accepted")
        } catch {}
        let task = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            try await client.sendLiveAppEvent(["type": "voice_route.ready"], connectionID: client.connectionID, current: { true })
        }
        do { try await task.value; XCTFail("Cancelled writer accepted") } catch is CancellationError {} catch { XCTFail("Wrong cancellation boundary") }
    }
}

@MainActor
private final class AppRouteRig {
    let connection = UUID()
    lazy var currentConnection: UUID? = connection
    let media = AppRouteMedia()
    var driver: JarvisLiveVoiceRouteDriver!
    var owner: JarvisLiveAppRoute!
    var previousFences = 0
    var messages: [[String: Any]] = []
    var ready: [String] = []
    var captions: [(String, String)] = []
    var failures: [String] = []
    var stops: [String] = []
    var protocolFailures: [JarvisLiveProtocolFailure] = []
}

@MainActor
private final class AppRouteMedia: JarvisLiveMediaTransport {
    var hasRemoteDescription = false
    var isConnected = true
    var outputEnabled = false
    var microphoneEnabled = false
    var onConnectivityChanged: (() -> Void)?
    var onFailure: (() -> Void)?
    var onProviderStarted: ((JarvisRealtimeServerEvent) -> Void)?
    var activations = 0
    var closed = false
    func prepareOffer() async throws -> String { "v=0\r\n" }
    func applyAnswer(_ sdp: String) async throws { hasRemoteDescription = true }
    func activate(microphoneEnabled: Bool) throws {
        activations += 1; outputEnabled = true; self.microphoneEnabled = microphoneEnabled
    }
    func setMicrophoneEnabled(_ enabled: Bool) { microphoneEnabled = enabled }
    func fenceAndClose() -> Bool { closed = true; outputEnabled = false; microphoneEnabled = false; return true }
}
