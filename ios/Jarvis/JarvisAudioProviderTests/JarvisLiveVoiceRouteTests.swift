import Foundation
import XCTest

@MainActor
final class JarvisLiveVoiceRouteTests: XCTestCase {
    private let session = UUID()
    private let routeID = UUID()
    private let audio = "live-audio"
    private let provider = "live-provider"

    private func route(_ generation: Int = 1) -> [String: Any] {
        ["scope": ["owner_id": "fixture", "session_id": session.uuidString],
         "route_id": routeID.uuidString, "route_generation": generation]
    }
    private func event(_ values: [String: Any]) -> JarvisRealtimeServerEvent {
        JarvisRealtimeServerEvent(type: values["type"] as? String ?? "", values: values)
    }
    private func selected(_ generation: Int = 1) -> JarvisRealtimeServerEvent {
        event(["type": "voice_route.selected", "protocol_version": 1, "route": route(generation),
         "audio_session_id": audio, "capabilities": JarvisLiveBackendCapabilityFixture.capabilities])
    }
    private func command(_ name: String, id: UUID = UUID(), generation: Int = 1) -> JarvisRealtimeServerEvent {
        event(["type": name, "protocol_version": 1, "audio_session_id": audio,
               "payload": ["route": route(generation), "command_id": id.uuidString,
                           "requested_at": "2026-10-30T12:00:00Z", "reason": "fixture"]])
    }
    private func driver(_ media: Media, ownership: JarvisLiveMediaOwnership? = nil,
                        fence: @escaping () -> Bool = { true }) -> JarvisLiveVoiceRouteDriver {
        JarvisLiveVoiceRouteDriver(sessionID: session, connectionID: UUID(), media: media,
                                  ownership: ownership ?? JarvisLiveMediaOwnership(), fencePrevious: fence)
    }
    private func prepare(_ driver: JarvisLiveVoiceRouteDriver) async throws {
        _ = try await driver.prepareOffer()
        XCTAssertEqual(try driver.select(selected())["type"] as? String, "voice_route.ready")
        try driver.didWriteReady(connectionID: driver.connectionID)
    }
    private func answer(_ driver: JarvisLiveVoiceRouteDriver) async throws {
        try await driver.applyAnswer("v=0\r\n", providerSessionID: provider, route: XCTUnwrap(driver.selection).route,
                                     audioSessionID: audio, connectionID: driver.connectionID)
    }
    private func started(_ driver: JarvisLiveVoiceRouteDriver, type: String = "session.started", providerID: String? = nil) throws {
        try driver.observeProviderStarted(event(["type": type, "session": ["id": providerID ?? provider]]),
                                          route: XCTUnwrap(driver.selection).route, audioSessionID: audio, connectionID: driver.connectionID)
    }
    private func activate(_ driver: JarvisLiveVoiceRouteDriver, id: UUID = UUID()) throws -> [String: Any] {
        try driver.activate(commandID: id, route: XCTUnwrap(driver.selection).route, audioSessionID: audio,
                            providerSessionID: provider, connectionID: driver.connectionID)
    }

    func testInitializationRequiresEveryBoundaryAndExplicitActivation() async throws {
        let media = Media(), driver = driver(Media())
        driver.close()
        let current = self.driver(media)
        defer { current.close() }
        try await prepare(current)
        XCTAssertFalse(current.canActivate)
        XCTAssertThrowsError(try activate(current))
        try await answer(current)
        XCTAssertFalse(media.outputEnabled)
        XCTAssertThrowsError(try activate(current))
        try started(current)
        media.isConnected = false
        XCTAssertFalse(current.canActivate)
        XCTAssertThrowsError(try activate(current))
        media.isConnected = true
        XCTAssertTrue(current.canActivate)
        XCTAssertFalse(current.outputEnabled)
        let receipt = try activate(current)
        XCTAssertTrue(current.outputEnabled)
        XCTAssertTrue(media.microphoneEnabled)
        let payload = try XCTUnwrap(receipt["payload"] as? [String: Any])
        XCTAssertEqual(payload["operation"] as? String, "start")
        XCTAssertEqual(payload["status"] as? String, "completed")
        XCTAssertFalse(try XCTUnwrap(payload["output_fenced"] as? Bool))
        XCTAssertTrue(payload["provider_stopped"] is NSNull)
        XCTAssertNil(payload["output_id"])
    }

    func testRecordingMediaIsOnlyExposedByActiveConnectedOwner() async throws {
        let media = Media(), current = driver(media)
        defer { current.close() }
        XCTAssertNil(current.activeMediaForRecording)
        try await prepare(current); try await answer(current); try started(current)
        XCTAssertNil(current.activeMediaForRecording)
        _ = try activate(current)
        XCTAssertTrue(current.activeMediaForRecording === media)
        media.isConnected = false
        XCTAssertNil(current.activeMediaForRecording)
        media.isConnected = true
        XCTAssertTrue(current.activeMediaForRecording === media)
        _ = current.close()
        XCTAssertNil(current.activeMediaForRecording)
    }

    func testInstructionAndContextACKsNeverQualifyAsSessionStarted() async throws {
        let current = driver(Media())
        defer { current.close() }
        try await prepare(current); try await answer(current)
        for type in ["session.ready", "session.created", "session.instructions.appended", "session.thinking.appended", "session.commentary.appended"] {
            XCTAssertThrowsError(try started(current, type: type))
        }
        XCTAssertThrowsError(try started(current, providerID: "old-provider"))
        XCTAssertFalse(current.canActivate)
    }

    func testReadyWriteAndIdentityMustMatch() async throws {
        let current = driver(Media())
        defer { current.close() }
        _ = try await current.prepareOffer()
        _ = try current.select(selected())
        try await answer(current); try started(current)
        XCTAssertFalse(current.canActivate)
        XCTAssertThrowsError(try current.didWriteReady(connectionID: UUID()))
        XCTAssertThrowsError(try current.select(selected(2)))
        try current.didWriteReady(connectionID: current.connectionID)
        XCTAssertTrue(current.canActivate)
        XCTAssertThrowsError(try current.activate(commandID: UUID(), route: JarvisVoiceRouteIdentity(wire: route(2)),
                                                  audioSessionID: audio, providerSessionID: provider, connectionID: current.connectionID))
    }

    func testRejectsRealtimeCapabilitiesAndInventedPlaybackAttribution() async throws {
        let current = driver(Media())
        defer { current.close() }
        _ = try await current.prepareOffer()
        for key in ["explicit_input_commit", "explicit_response_create", "image_input", "notification_output_attribution", "client_playback_evidence"] {
            var value = selected().values
            var capabilities = try XCTUnwrap(value["capabilities"] as? [String: Any])
            capabilities[key] = "supported"; value["capabilities"] = capabilities
            XCTAssertThrowsError(try current.select(event(value)))
        }
        XCTAssertNil(current.selection)
        XCTAssertEqual(JarvisLiveVoiceRouteDriver.clientCapabilities["client_playback_boundaries"] as? [String], [])
    }

    func testActualBackendLiveCapabilitiesRequireExplicitLiveProfile() async throws {
        let current = driver(Media())
        defer { current.close() }
        _ = try await current.prepareOffer()
        let message = try current.select(selected())
        XCTAssertEqual(message["type"] as? String, "voice_route.ready")
        let live = try JarvisVoiceRouteSelection(event: selected(), mediaKind: .nativeWebRTC, profile: .liveWebRTC)
        XCTAssertEqual(current.selection, live)
        XCTAssertFalse(live.playbackEvidence)
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: selected(), mediaKind: .nativeWebRTC))
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: selected(), mediaKind: .streamedPCM, profile: .liveWebRTC))
    }

    func testUnchangedRealtimeDriverAcceptsOpusAndRejectsLiveMetadata() throws {
        let realtime = JarvisVoiceRouteDriver(sessionID: session, connectionID: UUID(), mediaKind: .nativeWebRTC) { _ in true }
        defer { realtime.close() }
        XCTAssertThrowsError(try realtime.select(selected()))
        var values = selected().values
        var capabilities = JarvisLiveBackendCapabilityFixture.capabilities
        capabilities["input_codec"] = "opus"
        capabilities["output_codec"] = "opus"
        capabilities["explicit_input_commit"] = "supported"
        capabilities["explicit_response_create"] = "supported"
        capabilities["notification_output_attribution"] = "supported"
        values["capabilities"] = capabilities
        XCTAssertEqual(try realtime.select(event(values))["type"] as? String, "voice_route.ready")
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(values), mediaKind: .nativeWebRTC, profile: .liveWebRTC))
        XCTAssertThrowsError(try JarvisVoiceRouteSelection(event: event(values), mediaKind: .streamedPCM))
    }

    func testLiveManifestDoesNotRelabelOtherClientsOrClaimPlaybackEvidence() {
        let live = JarvisLiveVoiceRouteDriver.clientCapabilities
        XCTAssertEqual(live["input_codecs"] as? [String], ["webrtc_negotiated"])
        XCTAssertEqual(live["output_codecs"] as? [String], ["webrtc_negotiated"])
        XCTAssertEqual(live["supported_transports"] as? [String], ["native_voice"])
        XCTAssertEqual(live["supported_media_topologies"] as? [String], ["phone_to_provider"])
        XCTAssertEqual(live["client_playback_boundaries"] as? [String], [])
        XCTAssertEqual(JarvisVoiceMediaKind.nativeWebRTC.clientCapabilities["input_codecs"] as? [String], ["opus"])
        XCTAssertEqual(JarvisVoiceMediaKind.nativeWebRTC.clientCapabilities["output_codecs"] as? [String], ["opus"])
        XCTAssertEqual(JarvisVoiceMediaKind.streamedPCM.clientCapabilities["input_codecs"] as? [String], ["pcm16"])
    }

    func testLiveProfileStillRejectsWrongCodecTopologyAndAuthority() async throws {
        let current = driver(Media())
        defer { current.close() }
        _ = try await current.prepareOffer()
        let invalid = [("input_codec", "opus"), ("output_codec", "opus"), ("input_codec", "pcm16"),
                       ("transport", "chained_voice"), ("media_topology", "phone_to_backend"),
                       ("control_owner", "phone"), ("credential_mode", "ephemeral"), ("full_duplex", "unsupported")]
        for (key, value) in invalid {
            var selected = self.selected().values
            var capabilities = JarvisLiveBackendCapabilityFixture.capabilities
            capabilities[key] = value
            selected["capabilities"] = capabilities
            XCTAssertThrowsError(try current.select(event(selected)), key)
        }
        XCTAssertNil(current.selection)
    }

    func testActivationReplayIsIdempotentOnlyForTheSameCommand() async throws {
        let media = Media(), current = driver(Media())
        current.close()
        let live = driver(media)
        defer { live.close() }
        try await prepare(live); try await answer(live); try started(live)
        let id = UUID()
        let first = try activate(live, id: id)
        let second = try activate(live, id: id)
        XCTAssertEqual(try JSONSerialization.data(withJSONObject: first, options: .sortedKeys),
                       try JSONSerialization.data(withJSONObject: second, options: .sortedKeys))
        XCTAssertEqual(media.activations, 1)
        XCTAssertThrowsError(try activate(live))
    }

    func testInterruptPermanentlyFencesBeforeReceiptAndCannotReactivate() async throws {
        let media = Media(), live = driver(Media())
        live.close()
        let current = driver(media)
        defer { current.close() }
        try await prepare(current); try await answer(current); try started(current)
        let id = UUID(); _ = try activate(current, id: id)
        let interrupt = command("voice_route.interrupt")
        let response = try current.handleCommand(interrupt)
        XCTAssertFalse(media.outputEnabled)
        XCTAssertTrue(media.closed)
        XCTAssertTrue(current.requiresReplacement)
        let payload = try XCTUnwrap(response["payload"] as? [String: Any])
        XCTAssertEqual(payload["output_fenced"] as? Bool, true)
        XCTAssertEqual(payload["operation"] as? String, "interrupt_speech")
        XCTAssertTrue(payload["provider_stopped"] is NSNull)
        XCTAssertThrowsError(try activate(current, id: id))
        XCTAssertThrowsError(try started(current))
        _ = try current.handleCommand(interrupt)
        XCTAssertEqual(media.fences, 1)
    }

    func testLocalInterruptDoesNotWaitForBackendOrAllowUnmute() async throws {
        let media = Media(), current = driver(Media())
        current.close()
        let live = driver(media)
        defer { live.close() }
        try await prepare(live); try await answer(live); try started(live); _ = try activate(live)
        let message = try XCTUnwrap(live.interruptLocally())
        XCTAssertEqual(message["type"] as? String, "voice_route.interrupt")
        XCTAssertFalse(media.microphoneEnabled)
        XCTAssertFalse(media.outputEnabled)
        XCTAssertThrowsError(try live.setMicrophoneEnabled(true))
        XCTAssertThrowsError(try live.handleCommand(event(["type": "session.instructions.appended"])))
        XCTAssertEqual(media.activations, 1)
    }

    func testOldGenerationCannotFenceCurrentOutput() async throws {
        let media = Media(), current = driver(media)
        defer { current.close() }
        try await prepare(current); try await answer(current); try started(current); _ = try activate(current)
        XCTAssertThrowsError(try current.handleCommand(command("voice_route.interrupt", generation: 2)))
        XCTAssertTrue(current.outputEnabled)
    }

    func testChangedCommandReplayIsRejected() async throws {
        let current = driver(Media())
        defer { current.close() }
        try await prepare(current)
        let id = UUID()
        _ = try current.handleCommand(command("voice_route.interrupt", id: id))
        XCTAssertThrowsError(try current.handleCommand(command("voice_route.close", id: id)))
    }

    func testFailedFenceBlocksReplacementAndReportsUnknown() async throws {
        let ownership = JarvisLiveMediaOwnership(), media = Media()
        let first = driver(media, ownership: ownership)
        try await prepare(first)
        media.fenceSucceeds = false
        let result = try first.handleCommand(command("voice_route.close"))
        let payload = try XCTUnwrap(result["payload"] as? [String: Any])
        XCTAssertEqual(payload["status"] as? String, "unknown")
        XCTAssertEqual(payload["output_fenced"] as? Bool, false)
        let second = driver(Media(), ownership: ownership)
        do { _ = try await second.prepareOffer(); XCTFail("Unfenced owner replaced") } catch {}
        media.fenceSucceeds = true
        first.close(); second.close()
    }

    func testExclusiveOwnerAndOldCleanupCannotReleaseReplacement() async throws {
        let ownership = JarvisLiveMediaOwnership()
        let first = driver(Media(), ownership: ownership), second = driver(Media(), ownership: ownership)
        _ = try await first.prepareOffer()
        do { _ = try await second.prepareOffer(); XCTFail("Two owners") } catch {}
        first.close()
        _ = try await second.prepareOffer()
        first.close()
        let third = driver(Media(), ownership: ownership)
        do { _ = try await third.prepareOffer(); XCTFail("Old cleanup released new lease") } catch {}
        second.close(); third.close()
    }

    func testPreviousLegacyMediaMustBeFencedBeforePreparing() async throws {
        let media = Media(), current = driver(media, fence: { false })
        do { _ = try await current.prepareOffer(); XCTFail("Missing old-media fence") } catch {}
        XCTAssertEqual(media.prepares, 0)
        current.close()
    }

    func testShutdownDuringAnswerCannotRestoreReadiness() async throws {
        let media = Media(), current = driver(media)
        try await prepare(current)
        media.suspendAnswer = true
        let task = Task { try await self.answer(current) }
        for _ in 0..<100 where media.answerContinuation == nil { await Task.yield() }
        XCTAssertNotNil(media.answerContinuation)
        current.close()
        media.answerContinuation?.resume(); media.answerContinuation = nil
        do { try await task.value; XCTFail("Closed answer revived") } catch {}
        XCTAssertEqual(current.state, .closed)
        XCTAssertFalse(current.canActivate)
    }

    func testPreparationFailureReleasesOnlyItsOwnLease() async throws {
        let ownership = JarvisLiveMediaOwnership(), media = Media()
        media.prepareFails = true
        let first = driver(media, ownership: ownership)
        do { _ = try await first.prepareOffer(); XCTFail("Failed offer accepted") } catch {}
        XCTAssertEqual(first.state, .failed)
        let next = driver(Media(), ownership: ownership)
        _ = try await next.prepareOffer()
        next.close()
    }

    func testActivationFailureNeverReportsSuccess() async throws {
        let media = Media(), current = driver(media)
        try await prepare(current); try await answer(current); try started(current)
        media.activateFails = true
        XCTAssertThrowsError(try activate(current))
        XCTAssertEqual(current.state, .failed)
        XCTAssertFalse(current.outputEnabled)
        XCTAssertTrue(media.closed)
    }

    func testTransportFailureFencesAndRejectsDelayedStartup() async throws {
        let media = Media(), current = driver(media)
        try await prepare(current); try await answer(current)
        media.onFailure?()
        XCTAssertEqual(current.state, .failed)
        XCTAssertTrue(media.closed)
        XCTAssertThrowsError(try started(current))
    }

    func testRealMediaDeniedPermissionCannotProduceOffer() async throws {
        let media = JarvisLiveWebRTCMedia(microphonePermission: { false })
        do { _ = try await media.prepareOffer(); XCTFail("Denied microphone accepted") }
        catch { XCTAssertTrue(error is JarvisLiveWebRTCMedia.MediaError) }
        XCTAssertFalse(media.outputEnabled)
        XCTAssertFalse(media.hasRemoteDescription)
        XCTAssertTrue(media.fenceAndClose())
    }

    func testRealMediaBuildsNativeAudioOfferAndPermanentlyCloses() async throws {
        // Local peer/ICE only. No provider call, API key or real microphone activation.
        let media = JarvisLiveWebRTCMedia(microphonePermission: { true })
        defer { media.fenceAndClose() }
        let sdp = try await media.prepareOffer()
        XCTAssertTrue(sdp.hasPrefix("v=0"))
        XCTAssertTrue(sdp.contains("m=audio"))
        XCTAssertTrue(sdp.contains("m=application"))
        XCTAssertTrue(sdp.contains("sctp-port"))
        XCTAssertFalse(sdp.contains("m=video"))
        XCTAssertFalse(media.outputEnabled)
        XCTAssertFalse(media.isConnected)
        XCTAssertThrowsError(try media.activate(microphoneEnabled: true))
        XCTAssertTrue(media.fenceAndClose())
        do { _ = try await media.prepareOffer(); XCTFail("Retired peer reused") } catch {}
        XCTAssertFalse(media.outputEnabled)
    }
}

@MainActor
private final class Media: JarvisLiveMediaTransport {
    var hasRemoteDescription = false
    var isConnected = true
    var outputEnabled = false
    var onConnectivityChanged: (() -> Void)?
    var onFailure: (() -> Void)?
    var onProviderStarted: ((JarvisRealtimeServerEvent) -> Void)?
    var microphoneEnabled = false
    var closed = false
    var fences = 0
    var prepares = 0
    var activations = 0
    var fenceSucceeds = true
    var prepareFails = false
    var activateFails = false
    var suspendAnswer = false
    var answerContinuation: CheckedContinuation<Void, Never>?
    func prepareOffer() async throws -> String {
        prepares += 1
        if prepareFails { throw JarvisVoiceRouteError.notReady }
        return "v=0\r\n"
    }
    func applyAnswer(_ sdp: String) async throws {
        if suspendAnswer { await withCheckedContinuation { answerContinuation = $0 } }
        hasRemoteDescription = true
    }
    func activate(microphoneEnabled: Bool) throws {
        if activateFails { throw JarvisVoiceRouteError.notReady }
        activations += 1; outputEnabled = true; self.microphoneEnabled = microphoneEnabled
    }
    func setMicrophoneEnabled(_ enabled: Bool) { microphoneEnabled = enabled }
    func fenceAndClose() -> Bool {
        fences += 1
        guard fenceSucceeds else { return false }
        closed = true; outputEnabled = false; microphoneEnabled = false; return true
    }
}

/// Exact LiveVoiceRoute.capabilities.model_dump_json() captured on 2026-10-30
/// from the real backend constructor, without starting a route or connecting to
/// any service. Shared by driver and signaling tests, including null reasons.
enum JarvisLiveBackendCapabilityFixture {
    static let capabilities: [String: Any] = {
        let json = """
        {
          "transport": "native_voice",
          "media_topology": "phone_to_provider",
          "control_owner": "backend",
          "credential_mode": "server_only",
          "input_codec": "webrtc_negotiated",
          "output_codec": "webrtc_negotiated",
          "vad_owner": "provider",
          "explicit_input_commit": "unsupported",
          "explicit_response_create": "unsupported",
          "full_duplex": "supported",
          "image_input": "unsupported",
          "provider_context_receipts": "supported",
          "notification_output_attribution": "unsupported",
          "client_playback_evidence": "unsupported",
          "provider_interruption": "unsupported",
          "operations": [
            {"operation": "start", "support": "supported", "reason": null},
            {"operation": "publish_context", "support": "supported", "reason": null},
            {"operation": "offer_notification", "support": "supported", "reason": null},
            {"operation": "retire_context", "support": "supported", "reason": null},
            {"operation": "interrupt_speech", "support": "supported", "reason": null},
            {"operation": "close", "support": "supported", "reason": null}
          ]
        }
        """
        return try! JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
    }()
}
