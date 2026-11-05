import Foundation
import XCTest

final class JarvisRealtimeClientTests: XCTestCase {
    private func url(_ path: String, provider: JarvisAudioProvider, audioID: String = "fixture-audio") throws -> URL {
        guard let base = ProcessInfo.processInfo.environment["JARVIS_CLIENT_FIXTURE_URL"],
              base.hasPrefix("ws://127.0.0.1:"), !base.contains("$(") else {
            throw XCTSkip("Set JARVIS_CLIENT_FIXTURE_URL to the isolated loopback fixture server.")
        }
        return try provider.audioWebSocketURL(from: "\(base)\(path)?session_id=fixture-session&audio_session_id=\(audioID)")
    }

    @MainActor
    func testActualLegacyGeminiResponseIsRejectedBeforeMediaStartup() async throws {
        let client = JarvisRealtimeClient()
        defer { client.disconnect() }
        var run = JarvisAudioProviderRun(requested: .gemini)
        let rejected = expectation(description: "Legacy Gemini rejected")
        var captureCount = 0
        client.onEvent = { event in
            guard event.type == "session.created" else { return }
            do {
                try run.markReady(audioSessionID: event.audioSessionId)
                try run.startMedia(captureMode: .autoDetect, unifiedSession: true,
                                   startMicrophone: { captureCount += 1 }, startCamera: { captureCount += 1 })
                XCTFail("Unsupported Gemini was accepted")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("did not confirm Gemini"))
            }
            client.disconnect()
            rejected.fulfill()
        }
        client.connect(audioWebSocketURL: try url("/legacy", provider: .gemini), bearerToken: "fixture-token")
        await fulfillment(of: [rejected], timeout: 5)
        XCTAssertEqual(captureCount, 0)
    }

    @MainActor
    func testDisabledGeminiThenManualOpenAIStartKeepsServerErrorAndWorks() async throws {
        let client = JarvisRealtimeClient()
        defer { client.disconnect() }
        let disabled = expectation(description: "Disabled Gemini error")
        var message: String?
        client.onEvent = { event in
            guard event.type == "error" else { return }
            message = event.message
            client.disconnect()
            disabled.fulfill()
        }
        client.connect(audioWebSocketURL: try url("/disabled", provider: .gemini), bearerToken: "fixture-token")
        await fulfillment(of: [disabled], timeout: 5)
        XCTAssertEqual(message, "Gemini Live audio is disabled.")
        let ready = expectation(description: "Manual OpenAI start")
        var run = JarvisAudioProviderRun(requested: .openai)
        var order: [String] = []
        client.onEvent = { event in
            guard event.type == "session.created" else { return }
            do {
                try run.markReady(audioSessionID: event.audioSessionId)
                try run.startMedia(captureMode: .holdToTalk, unifiedSession: true,
                                   startMicrophone: { order.append("microphone") }, startCamera: { order.append("camera") })
            } catch { XCTFail(error.localizedDescription) }
            ready.fulfill()
        }
        client.connect(audioWebSocketURL: try url("/legacy", provider: .openai), bearerToken: "fixture-token")
        await fulfillment(of: [ready], timeout: 5)
        XCTAssertEqual(run.activeProvider, .openai)
        XCTAssertEqual(order, ["microphone", "camera"])
    }

    @MainActor
    func testRetiredSocketCannotDisconnectOrRelabelReplacement() async throws {
        let client = JarvisRealtimeClient()
        defer { client.disconnect() }
        let nextURL = try url("/select", provider: .openai, audioID: "new-audio")
        let ready = expectation(description: "Replacement connection ready")
        let unexpectedDisconnect = expectation(description: "Old socket disconnect suppressed")
        unexpectedDisconnect.isInverted = true
        var replaced = false
        var run = JarvisAudioProviderRun(requested: .gemini)
        client.onDisconnect = { _ in unexpectedDisconnect.fulfill() }
        client.onEvent = { event in
            if event.type == "audio.provider.selected", !replaced {
                replaced = true
                run = JarvisAudioProviderRun(requested: .openai)
                client.connect(audioWebSocketURL: nextURL, bearerToken: "fixture-token")
                return
            }
            do {
                if event.type == "audio.provider.selected" {
                    try run.acknowledge(event)
                } else if event.type == "session.created" {
                    XCTAssertEqual(event.audioSessionId, "new-audio")
                    try run.markReady(audioSessionID: event.audioSessionId)
                    ready.fulfill()
                }
            } catch { XCTFail(error.localizedDescription) }
        }
        client.connect(audioWebSocketURL: try url("/select", provider: .gemini, audioID: "old-audio"), bearerToken: "fixture-token")
        await fulfillment(of: [ready], timeout: 5)
        await fulfillment(of: [unexpectedDisconnect], timeout: 0.2)
        XCTAssertEqual(run.activeProvider, .openai)
        XCTAssertEqual(run.acknowledgement?.audioSessionID, "new-audio")
        XCTAssertTrue(client.isConnected)
    }

    @MainActor
    func testActualClientRetainsBearerAndExistingInputEventSequence() async throws {
        let client = JarvisRealtimeClient()
        defer { client.disconnect() }
        let ready = expectation(description: "Connected")
        let received = expectation(description: "Wire sequence received")
        var trace: [[String: Any]] = []
        client.onEvent = { event in
            if event.type == "session.created" { ready.fulfill() }
            if event.type == "fixture.received" {
                XCTAssertEqual(event.values["authorization_verified"] as? Bool, true)
                trace = event.values["events"] as? [[String: Any]] ?? []
                received.fulfill()
            }
        }
        client.connect(audioWebSocketURL: try url("/select", provider: .openai), bearerToken: "fixture-token")
        await fulfillment(of: [ready], timeout: 5)
        try await client.sendAudioAppend(Data([1, 0, 2, 0]), segmentId: "fixture-segment")
        try await client.sendAudioCommit(segmentId: "fixture-segment", startedAt: nil, endedAt: nil)
        try await client.sendResponseCreate()
        try await client.sendResponseCancel()
        try await client.sendPlaybackTruncated(itemId: "fixture-item", audioEndMs: 10)
        try await client.sendSessionClose()
        await fulfillment(of: [received], timeout: 5)
        XCTAssertEqual(trace.compactMap { $0["type"] as? String }, [
            "audio.input.append", "audio.input.commit", "response.create", "response.cancel", "playback.truncated", "session.close",
        ])
        XCTAssertEqual(trace.first?["audio_base64"] as? String, "AQACAA==")
        XCTAssertEqual(trace[1]["segment_id"] as? String, "fixture-segment")
        XCTAssertEqual(Set(trace.compactMap { $0["client_event_id"] as? String }).count, trace.count)
    }

    @MainActor
    func testActualPlayerReceiptsTravelOnTheNegotiatedCurrentSocket() async throws {
        let client = JarvisRealtimeClient()
        let playback = AudioPlaybackService()
        defer { playback.stop(); client.disconnect() }
        let received = expectation(description: "Actual playback receipts returned by wire fixture")
        playback.onPlaybackReceipt = { receipt in
            let connectionID = client.connectionID
            Task { @MainActor in
                do {
                    let sent = try await client.sendPlaybackReceipt(receipt, audioSessionID: "fixture-audio", connectionID: connectionID)
                    XCTAssertTrue(sent)
                    if receipt.state == .completed { try await client.sendSessionClose() }
                } catch { XCTFail(error.localizedDescription) }
            }
        }
        client.onEvent = { event in
            do {
                if event.type == "audio.output.delta", let audio = event.audioBase64 {
                    try playback.enqueuePCM16(base64: audio, responseId: event.responseId, itemId: event.itemId)
                } else if event.type == "audio.output.done" {
                    playback.generationFinished(responseId: event.responseId, itemId: event.itemId)
                } else if event.type == "fixture.received" {
                    let events = event.values["events"] as? [[String: Any]] ?? []
                    let receipt = events.first { $0["type"] as? String == "playback.completed" }
                    XCTAssertEqual(receipt?["played_samples"] as? Int, 2400)
                    XCTAssertEqual(receipt?["audio_session_id"] as? String, "fixture-audio")
                    XCTAssertEqual(receipt?["response_id"] as? String, "fixture-response")
                    XCTAssertEqual(receipt?["boundary"] as? String, "av_audio_data_played_back_lower_bound")
                    received.fulfill()
                }
            } catch { XCTFail(error.localizedDescription) }
        }
        client.connect(audioWebSocketURL: try url("/playback", provider: .openai), bearerToken: "fixture-token")
        await fulfillment(of: [received], timeout: 15)
    }

    @MainActor
    func testOldBackendAndRetiredConnectionDoNotReceiveNewReceipts() async throws {
        let client = JarvisRealtimeClient()
        defer { client.disconnect() }
        let ready = expectation(description: "Legacy ready")
        client.onEvent = { event in if event.type == "session.created" { ready.fulfill() } }
        client.connect(audioWebSocketURL: try url("/legacy", provider: .openai), bearerToken: "fixture-token")
        await fulfillment(of: [ready], timeout: 5)
        let receipt = AudioPlaybackReceipt(responseId: "r", itemId: "i", playedSamples: 2400, state: .completed)
        let unsupported = try await client.sendPlaybackReceipt(receipt, audioSessionID: "fixture-audio", connectionID: client.connectionID)
        XCTAssertFalse(unsupported)
        let oldConnection = client.connectionID
        client.disconnect()
        let stale = try await client.sendPlaybackReceipt(receipt, audioSessionID: "fixture-audio", connectionID: oldConnection)
        XCTAssertFalse(stale)
    }

    // These replace paid expert-authored TTS probes. They qualify current-route
    // control/owner behavior with synthetic transport and in-memory repositories,
    // not task execution, memory correction semantics, RTP playback or hearing.
    @MainActor
    func testSingleVoiceNotificationSettlesWithoutPlaybackClaim() async throws {
        try await runSingleVoiceFixture("completion")
    }

    @MainActor
    func testSingleVoiceNotificationHonorsUserPriorityAndDeduplicates() async throws {
        try await runSingleVoiceFixture("priority_deduplication")
    }

    @MainActor
    func testSingleVoiceInterruptionAndReconnectFenceOldAttempt() async throws {
        try await runSingleVoiceFixture("cancel_reconnect")
    }

    @MainActor
    func testSingleVoiceSourceResetRejectsWithdrawnResult() async throws {
        try await runSingleVoiceFixture("source_reset")
    }

    @MainActor
    private func runSingleVoiceFixture(_ scenario: String) async throws {
        guard let address = ProcessInfo.processInfo.environment["JARVIS_DELIVERY_FIXTURE_URL"],
              address.hasPrefix("ws://127.0.0.1:"), !address.contains("$("),
              var components = URLComponents(string: address), components.path == "/single-voice",
              components.queryItems == nil else {
            throw XCTSkip("Start provider-free evals.second_brain.ios_single_voice and pass its exact ready.url.")
        }
        components.queryItems = [
            URLQueryItem(name: "case", value: scenario),
            URLQueryItem(name: "probe_id", value: UUID().uuidString.lowercased()),
            URLQueryItem(name: "audio_provider", value: "openai")
        ]
        let initialURL = try XCTUnwrap(components.url)
        let client = JarvisRealtimeClient()
        var controlTask: Task<Void, Never>?
        var reconnectTask: Task<Void, Never>?
        var run = JarvisAudioProviderRun(requested: .openai)
        var sessionID: String?
        var audioID: String?
        var audioIDs = Set<String>()
        var connectionIDs = Set<UUID>()
        var protocolSeen = false
        var connections = 0
        var submissions = 0
        var cancellations = 0
        var resetAcknowledgments = 0
        var pendingReset: JarvisContextResetNotice?
        var resetConnectionID: UUID?
        var reconnectBudget = JarvisContextReconnectBudget()
        var terminal = false
        let finished = expectation(description: "Single-voice fixture owner/control result, not hearing")
        defer { controlTask?.cancel(); reconnectTask?.cancel(); client.disconnect() }

        func failAndFinish(_ message: String) {
            guard !terminal else { return }
            terminal = true
            XCTFail(message)
            finished.fulfill()
        }
        func reconnect() throws {
            guard connections == 1, let sessionID,
                  reconnectBudget.admit(now: ProcessInfo.processInfo.systemUptime) else {
                throw NSError(domain: "SingleVoiceFixture", code: 1)
            }
            var next = components
            next.queryItems?.append(URLQueryItem(name: "session_id", value: sessionID))
            let url = try XCTUnwrap(next.url)
            let precedingControl = controlTask
            client.disconnect()
            run = JarvisAudioProviderRun(requested: .openai)
            protocolSeen = false
            audioID = nil
            reconnectTask = Task { @MainActor in
                await precedingControl?.value
                guard !Task.isCancelled, !terminal else { return }
                client.connect(audioWebSocketURL: url, bearerToken: "fixture-token")
            }
        }
        client.onEvent = { event in
            guard !terminal else { return }
            do {
                switch event.type {
                case "audio.provider.selected":
                    try run.acknowledge(event)
                case "fixture.protocol":
                    guard event.values["fixture_protocol"] as? String == "jarvis.ios_single_voice_control.v1",
                          event.values["case"] as? String == scenario,
                          event.values["native_model"] as? Bool == false,
                          event.values["native_tts"] as? Bool == false,
                          event.values["canonical_database"] as? Bool == false else {
                        failAndFinish("Wrong or paid legacy delivery fixture"); return
                    }
                    protocolSeen = true
                case "session.created":
                    guard protocolSeen, let identity = event.audioSessionId,
                          let conversation = event.values["session_id"] as? String,
                          UUID(uuidString: conversation) != nil else {
                        failAndFinish("Missing fixture/conversation binding"); return
                    }
                    try run.markReady(audioSessionID: identity)
                    XCTAssertEqual(run.activeProvider, .openai)
                    if let expected = sessionID { XCTAssertEqual(conversation, expected) }
                    else { sessionID = conversation }
                    audioID = identity
                    connections += 1
                    XCTAssertTrue(audioIDs.insert(identity).inserted)
                    XCTAssertTrue(connectionIDs.insert(client.connectionID).inserted)
                    if connections == 2, let notice = pendingReset, let oldConnection = resetConnectionID {
                        let previous = controlTask
                        controlTask = Task { @MainActor in
                            await previous?.value
                            do {
                                let sent = try await client.sendContextResetCleared(notice, connectionID: oldConnection)
                                XCTAssertFalse(sent, "Old reset ACK escaped onto the replacement connection")
                            } catch { failAndFinish("Retired reset ACK failed unexpectedly") }
                        }
                    }
                case "fixture.notification_submitted":
                    XCTAssertEqual(event.values["fixture_protocol"] as? String, "jarvis.ios_single_voice_control.v1")
                    submissions += 1
                    XCTAssertEqual(submissions, 1)
                    if scenario == "cancel_reconnect" {
                        cancellations += 1
                        controlTask = Task { @MainActor in
                            do { try await client.sendResponseCancel() }
                            catch { failAndFinish("Single-voice cancel write failed") }
                        }
                    }
                case "fixture.disconnect":
                    guard scenario == "cancel_reconnect", cancellations == 1 else {
                        failAndFinish("Unexpected reconnect"); return
                    }
                    try reconnect()
                case "session.context_invalidated":
                    guard scenario == "source_reset", let sessionID else {
                        failAndFinish("Unexpected source reset"); return
                    }
                    let notice = try XCTUnwrap(JarvisContextResetNotice(event: event, audioSessionID: audioID, sessionID: sessionID))
                    pendingReset = notice
                    let connection = client.connectionID
                    resetConnectionID = connection
                    controlTask = Task { @MainActor in
                        do {
                            guard try await client.sendContextResetCleared(notice, connectionID: connection) else {
                                failAndFinish("Source reset ACK refused"); return
                            }
                            resetAcknowledgments += 1
                        } catch { failAndFinish("Source reset ACK write failed") }
                    }
                case "session.closed":
                    guard scenario == "source_reset", let sessionID, let pendingReset,
                          JarvisContextResetNotice(event: event, audioSessionID: audioID, sessionID: sessionID) == pendingReset else {
                        failAndFinish("Unmatched closure"); return
                    }
                    try reconnect()
                case "audio.output.delta", "audio.output.done", "audio.output.clear":
                    failAndFinish("Obsolete separate PCM/TTS delivery reached the native-route fixture")
                case "fixture.result":
                    guard event.values["ok"] as? Bool == true,
                          event.values["fixture_protocol"] as? String == "jarvis.ios_single_voice_control.v1",
                          event.values["case"] as? String == scenario else {
                        failAndFinish("Single-voice fixture reported failure"); return
                    }
                    let checks = try XCTUnwrap(event.values["checks"] as? [String: Bool])
                    let common = ["one_submission", "native_voice_plan", "no_legacy_chunks",
                                  "no_played_claim", "no_manufactured_user_turn", "no_legacy_speech_provider"]
                    for key in common { XCTAssertEqual(checks[key], true, key) }
                    XCTAssertTrue(checks.values.allSatisfy { $0 })
                    XCTAssertEqual(event.values["state"] as? String, "unknown")
                    XCTAssertEqual(event.values["human_hearing"] as? String, "unknown")
                    XCTAssertEqual(event.values["task_execution"] as? String, "not_exercised")
                    XCTAssertEqual(event.values["memory_effect"] as? String, "not_exercised")
                    XCTAssertEqual(event.values["native_model"] as? Bool, false)
                    XCTAssertEqual(event.values["native_tts"] as? Bool, false)
                    XCTAssertEqual(event.values["canonical_database"] as? Bool, false)
                    XCTAssertEqual(event.values["physical_playback_measured"] as? Bool, false)
                    XCTAssertEqual(connections, event.values["connections"] as? Int)
                    XCTAssertEqual(submissions, scenario == "source_reset" ? 0 : 1)
                    XCTAssertEqual(event.values["external_response_requests"] as? Int, scenario == "source_reset" ? 0 : 1)
                    if scenario == "priority_deduplication" {
                        XCTAssertEqual(checks["user_turn_priority"], true)
                        XCTAssertEqual(checks["duplicate_evidence_no_resubmission"], true)
                    }
                    if scenario == "cancel_reconnect" {
                        XCTAssertEqual(connections, 2); XCTAssertEqual(cancellations, 1)
                        XCTAssertEqual(checks["cancel_settled_interrupted"], true)
                        XCTAssertEqual(checks["owner_retired_after_fence"], true)
                        XCTAssertEqual(checks["retired_route_cannot_affect_replacement"], true)
                    }
                    if scenario == "source_reset" {
                        XCTAssertEqual(connections, 2); XCTAssertEqual(resetAcknowledgments, 1)
                        XCTAssertEqual(checks["withdrawn_source_no_external_write"], true)
                        XCTAssertEqual(checks["matching_reset_ack"], true)
                        XCTAssertEqual(checks["withdrawn_result_not_replayed"], true)
                        XCTAssertEqual(checks["retired_route_cannot_affect_replacement"], true)
                    }
                    terminal = true
                    finished.fulfill()
                case "error": failAndFinish("Unexpected single-voice fixture error")
                default: break
                }
            } catch { failAndFinish("Single-voice fixture binding/assertion failed") }
        }
        client.connect(audioWebSocketURL: initialURL, bearerToken: "fixture-token")
        await fulfillment(of: [finished], timeout: 20)
        controlTask?.cancel(); reconnectTask?.cancel()
        await controlTask?.value; await reconnectTask?.value
    }

}


final class JarvisContextResetTests: XCTestCase {

    @MainActor
    func testBackendResetClearsPlayerAndRejoinsSameConversation() async throws {
        guard let address = ProcessInfo.processInfo.environment["JARVIS_CONTEXT_RESET_FIXTURE_URL"],
              address.hasPrefix("ws://127.0.0.1:"), !address.contains("$("),
              let url = URL(string: address),
              let sessionID = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?
                .first(where: { $0.name == "session_id" })?.value else {
            throw XCTSkip("Start the isolated source-context-reset fixture.")
        }
        let client = JarvisRealtimeClient()
        let playback = AudioPlaybackService()
        defer { playback.stop(); client.disconnect() }
        let finished = expectation(description: "Real backend rebuilt source-safe conversation")
        var connections = 0
        var audioID: String?
        var pendingReset: JarvisContextResetNotice?
        var receivedAudio = false
        var cleared = false
        var acknowledged = false
        var reconnectBudget = JarvisContextReconnectBudget()
        client.onEvent = { event in
            do {
                switch event.type {
                case "session.created":
                    connections += 1
                    audioID = event.audioSessionId
                    pendingReset = nil
                    if connections == 1 {
                        Task { @MainActor in
                            do {
                                try await client.sendAudioAppend(Data(repeating: 0, count: 4_800), segmentId: "synthetic-input")
                                try await client.sendAudioCommit(segmentId: "synthetic-input", startedAt: nil, endedAt: nil)
                                try await client.sendResponseCreate()
                            } catch { XCTFail(error.localizedDescription) }
                        }
                    }
                case "audio.output.delta":
                    XCTAssertNil(pendingReset, "Old output arrived after reset")
                    receivedAudio = true
                    try playback.enqueuePCM16(base64: event.audioBase64 ?? "",
                                              responseId: event.responseId, itemId: event.itemId)
                case "session.context_invalidated":
                    let notice = try XCTUnwrap(JarvisContextResetNotice(event: event,
                        audioSessionID: audioID, sessionID: sessionID))
                    pendingReset = notice
                    playback.resetConnection()
                    cleared = true
                    let connectionID = client.connectionID
                    Task { @MainActor in
                        do {
                            acknowledged = try await client.sendContextResetCleared(notice, connectionID: connectionID)
                            XCTAssertTrue(acknowledged)
                        } catch { XCTFail(error.localizedDescription) }
                    }
                case "session.closed":
                    guard connections == 1 else { return }
                    let notice = try XCTUnwrap(JarvisContextResetNotice(event: event,
                        audioSessionID: audioID, sessionID: sessionID))
                    XCTAssertEqual(notice, pendingReset)
                    XCTAssertTrue(cleared)
                    XCTAssertTrue(reconnectBudget.admit(now: ProcessInfo.processInfo.systemUptime))
                    client.connect(audioWebSocketURL: url, bearerToken: "fixture-token")
                case "fixture.result":
                    XCTAssertEqual(event.values["ok"] as? Bool, true, "\(event.values)")
                    XCTAssertEqual(event.values["native_model"] as? Bool, false)
                    XCTAssertEqual(connections, 2)
                    XCTAssertTrue(receivedAudio)
                    XCTAssertTrue(cleared)
                    XCTAssertTrue(acknowledged)
                    finished.fulfill()
                case "error":
                    XCTFail(event.message ?? "Unexpected server error")
                default: break
                }
            } catch {
                XCTFail(error.localizedDescription)
            }
        }
        client.connect(audioWebSocketURL: url, bearerToken: "fixture-token")
        await fulfillment(of: [finished], timeout: 70)
    }

    private let sessionID = UUID()
    private let resetID = UUID()

    private func event(_ type: String = "session.context_invalidated", audioID: String = "audio",
                       changes: [String: Any] = [:]) -> JarvisRealtimeServerEvent {
        JarvisRealtimeServerEvent(type: type, values: [
            "session_id": sessionID.uuidString, "audio_session_id": audioID,
            "context_reset": ["protocol_version": 1, "id": resetID.uuidString, "preserve_session": true]
        ].merging(changes) { _, new in new })
    }

    func testNoticeRequiresExactConversationConnectionAndProtocol() throws {
        let valid = try XCTUnwrap(JarvisContextResetNotice(event: event(), audioSessionID: "audio",
                                                         sessionID: sessionID.uuidString))
        XCTAssertEqual(valid.id, resetID)
        XCTAssertNil(JarvisContextResetNotice(event: event(audioID: "old"), audioSessionID: "audio",
                                             sessionID: sessionID.uuidString))
        XCTAssertNil(JarvisContextResetNotice(event: event(), audioSessionID: "audio", sessionID: UUID().uuidString))
        XCTAssertNil(JarvisContextResetNotice(event: event("error"), audioSessionID: "audio", sessionID: sessionID.uuidString))
        for version: Any in [true, 2, "1"] {
            XCTAssertNil(JarvisContextResetNotice(event: event(changes: [
                "context_reset":["protocol_version":version, "id":resetID.uuidString, "preserve_session":true]
            ]), audioSessionID: "audio", sessionID: sessionID.uuidString))
        }
    }

    func testReconnectBurstIsBoundedAndDoesNotResetForEachNewConnection() {
        var budget = JarvisContextReconnectBudget()
        XCTAssertTrue(budget.admit(now: 10))
        XCTAssertTrue(budget.admit(now: 11))
        XCTAssertTrue(budget.admit(now: 12))
        XCTAssertFalse(budget.admit(now: 13))
        XCTAssertFalse(budget.admit(now: 9))
        XCTAssertTrue(budget.admit(now: 71))
    }

    @MainActor
    func testRetiredConnectionCannotAcknowledgeClearOnReplacement() async throws {
        let client = JarvisRealtimeClient()
        let notice = try XCTUnwrap(JarvisContextResetNotice(event: event(), audioSessionID: "audio",
                                                           sessionID: sessionID.uuidString))
        let sent = try await client.sendContextResetCleared(notice, connectionID: UUID())
        XCTAssertFalse(sent)
    }
}


final class JarvisSpeechSegmentStateTests: XCTestCase {
    func testOldCommitCannotClearSegmentStartedWhileItsWriteWasPending() throws {
        var state = JarvisSpeechSegmentState()
        state.begin(at: Date(timeIntervalSince1970: 1))
        let oldID = try XCTUnwrap(state.id)
        let newerStart = Date(timeIntervalSince1970: 2)
        state.begin(at: newerStart)
        let newerID = state.id
        XCTAssertFalse(state.finishCommit(segmentID: oldID))
        XCTAssertEqual(state.id, newerID)
        XCTAssertEqual(state.startedAt, newerStart)
    }

    func testCurrentCommitClearsItsOwnSegmentExactlyOnce() throws {
        var state = JarvisSpeechSegmentState()
        state.begin(at: Date())
        let id = try XCTUnwrap(state.id)
        XCTAssertTrue(state.finishCommit(segmentID: id))
        XCTAssertNil(state.id)
        XCTAssertNil(state.startedAt)
        XCTAssertFalse(state.finishCommit(segmentID: id))
    }

    func testResetInvalidatesAnOlderCommitCompletion() throws {
        var state = JarvisSpeechSegmentState()
        state.begin(at: Date())
        let id = try XCTUnwrap(state.id)
        state.reset()
        XCTAssertFalse(state.finishCommit(segmentID: id))
        XCTAssertNil(state.id)
        XCTAssertNil(state.startedAt)
    }
}

@MainActor
final class JarvisAudioSendQueueTests: XCTestCase {
    private final class Gate {
        var continuation: CheckedContinuation<Void, Never>?
        func wait() async {
            await withCheckedContinuation { continuation = $0 }
        }
        func release() {
            continuation?.resume()
            continuation = nil
        }
    }

    func testCommitAndCreateStayTogetherAcrossASuspendedWrite() async throws {
        let queue = JarvisAudioSendQueue()
        let gate = Gate()
        defer { gate.release(); queue.reset() }
        let started = expectation(description: "Commit entered")
        let finished = expectation(description: "New audio sent")
        var order: [String] = []
        try queue.enqueue({
            order.append("commit")
            started.fulfill()
            await gate.wait()
            order.append("create")
        }, onFailure: { _ in XCTFail("Unexpected failure") })
        await fulfillment(of: [started], timeout: 1)
        try queue.enqueue({
            order.append("new-audio")
            finished.fulfill()
        }, onFailure: { _ in XCTFail("Unexpected failure") })
        await Task.yield()
        XCTAssertEqual(order, ["commit"])
        gate.release()
        await fulfillment(of: [finished], timeout: 1)
        XCTAssertEqual(order, ["commit", "create", "new-audio"])
    }

    func testResetDropsQueuedOldInputAndCannotClearANewerWorker() async throws {
        let queue = JarvisAudioSendQueue()
        let oldGate = Gate()
        let newGate = Gate()
        defer { oldGate.release(); newGate.release(); queue.reset() }
        let oldStarted = expectation(description: "Old send entered")
        let newStarted = expectation(description: "New send entered")
        let newFinished = expectation(description: "New sequence completed")
        var order: [String] = []
        try queue.enqueue({
            oldStarted.fulfill()
            await oldGate.wait()
            throw JarvisAudioSessionError.realtimeNotConnected
        }, onFailure: { _ in XCTFail("Retired failure must not affect a new session") })
        await fulfillment(of: [oldStarted], timeout: 1)
        try queue.enqueue({ XCTFail("Retired queued audio was sent") }, onFailure: { _ in })
        queue.reset()
        try queue.enqueue({
            newStarted.fulfill()
            await newGate.wait()
            order.append("new-first")
        }, onFailure: { _ in XCTFail("Unexpected failure") })
        await fulfillment(of: [newStarted], timeout: 1)
        oldGate.release()
        await Task.yield()
        try queue.enqueue({
            order.append("new-second")
            newFinished.fulfill()
        }, onFailure: { _ in XCTFail("Unexpected failure") })
        XCTAssertTrue(order.isEmpty)
        newGate.release()
        await fulfillment(of: [newFinished], timeout: 1)
        XCTAssertEqual(order, ["new-first", "new-second"])
    }

    func testBackpressureIsExplicitRatherThanDroppingAcceptedAudio() async throws {
        let queue = JarvisAudioSendQueue(maxPendingOperations: 1)
        defer { queue.reset() }
        try queue.enqueue({}, onFailure: { _ in })
        XCTAssertThrowsError(try queue.enqueue({}, onFailure: { _ in }))
    }

    func testFailureStopsTheRemainingBatchWithoutReplayingIt() async throws {
        let queue = JarvisAudioSendQueue()
        defer { queue.reset() }
        let failed = expectation(description: "Failure surfaced")
        var attempts = 0
        try queue.enqueue({
            attempts += 1
            throw JarvisAudioSessionError.realtimeNotConnected
        }, onFailure: { _ in failed.fulfill() })
        try queue.enqueue({ XCTFail("Input after a failed batch was sent") }, onFailure: { _ in })
        await fulfillment(of: [failed], timeout: 1)
        XCTAssertEqual(attempts, 1)
    }

    func testCancelledWriterStopsBeforeSelectingATransport() async throws {
        let client = JarvisRealtimeClient()
        let gate = Gate()
        defer { gate.release() }
        let started = expectation(description: "Retired writer entered")
        let finished = expectation(description: "Cancellation observed")
        let writer = Task {
            started.fulfill()
            await gate.wait()
            do {
                try await client.sendResponseCreate()
                XCTFail("Cancelled writer continued")
            } catch is CancellationError {
                // No socket was needed or selected.
            } catch {
                XCTFail("Cancellation must precede transport lookup: \(error)")
            }
            finished.fulfill()
        }
        await fulfillment(of: [started], timeout: 1)
        writer.cancel()
        gate.release()
        await fulfillment(of: [finished], timeout: 1)
    }
}
