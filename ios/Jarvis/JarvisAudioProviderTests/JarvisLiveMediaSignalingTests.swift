import Foundation
import XCTest

@MainActor
final class JarvisLiveMediaSignalingTests: XCTestCase {
    private let session = UUID()
    private let routeID = UUID()
    private let audio = "live-audio-fixture"
    private let provider = "live_provider_fixture"
    private let commandID = UUID()

    private func route(generation: Int = 1) -> [String: Any] {
        ["scope": ["owner_id": "fixture", "session_id": session.uuidString],
         "route_id": routeID.uuidString, "route_generation": generation]
    }
    private func event(_ values: [String: Any]) -> JarvisRealtimeServerEvent {
        JarvisRealtimeServerEvent(type: values["type"] as? String ?? "", values: values)
    }
    private func selected() -> JarvisRealtimeServerEvent {
        event(["type": "voice_route.selected", "protocol_version": 1, "route": route(),
               "audio_session_id": audio, "capabilities": JarvisLiveBackendCapabilityFixture.capabilities])
    }
    private func command(_ phase: String, id: UUID? = nil) -> JarvisRealtimeServerEvent {
        var value: [String: Any] = ["type": "live.media." + phase, "protocol_version": 1, "route": route(),
                                    "audio_session_id": audio, "command_id": (id ?? commandID).uuidString,
                                    "provider_session_id": provider]
        if phase == "prepare" { value["answer_sdp"] = "v=0\r\nfixture" }
        return event(value)
    }
    private func make(ready: Bool = true, clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
                      timeout: TimeInterval = 20) async throws
        -> (JarvisLiveMediaSignaling, JarvisLiveVoiceRouteDriver, LiveSignalingTestMedia, LiveSignalingWriter) {
        let media = LiveSignalingTestMedia(), writer = LiveSignalingWriter()
        let driver = JarvisLiveVoiceRouteDriver(sessionID: session, connectionID: UUID(), media: media,
                                                ownership: JarvisLiveMediaOwnership(), fencePrevious: { true })
        _ = try await driver.prepareOffer()
        _ = try driver.select(selected())
        if ready { try driver.didWriteReady(connectionID: driver.connectionID) }
        let bridge = try JarvisLiveMediaSignaling(driver: driver, timeout: timeout, clock: clock) { payload, connection, current in
            try await writer.send(payload, connection: connection, current: current)
        }
        return (bridge, driver, media, writer)
    }
    private func until(_ predicate: @MainActor () -> Bool, file: StaticString = #filePath, line: UInt = #line) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !predicate() {
            guard ContinuousClock.now < deadline else { XCTFail("Condition timed out", file: file, line: line); throw JarvisVoiceRouteError.notReady }
            await Task.yield()
        }
    }
    private func prepared(_ bridge: JarvisLiveMediaSignaling, _ driver: JarvisLiveVoiceRouteDriver,
                          _ media: LiveSignalingTestMedia) async throws {
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.answers == 1 && media.hasRemoteDescription }
        media.started(provider)
        try await until { bridge.state == .prepared }
    }
    private func receipt(_ value: [String: Any], phase: String) throws {
        XCTAssertEqual(Set(value.keys), Set(["type", "protocol_version", "route", "audio_session_id", "command_id",
                                            "provider_session_id", "phase", "receipt_id", "provider_started", "output_gated"]))
        XCTAssertEqual(value["type"] as? String, "live.media.receipt")
        XCTAssertEqual(value["protocol_version"] as? Int, 1)
        XCTAssertEqual(try JarvisVoiceRouteIdentity(wire: XCTUnwrap(value["route"] as? [String: Any])),
                       try JarvisVoiceRouteIdentity(wire: route()))
        XCTAssertEqual(value["audio_session_id"] as? String, audio)
        XCTAssertEqual(value["provider_session_id"] as? String, provider)
        XCTAssertEqual(value["command_id"] as? String, commandID.uuidString.lowercased())
        XCTAssertEqual(value["phase"] as? String, phase)
        XCTAssertNotNil(UUID(uuidString: try XCTUnwrap(value["receipt_id"] as? String)))
        XCTAssertEqual(value["provider_started"] as? Bool, true)
        XCTAssertEqual(value["output_gated"] as? Bool, phase == "prepare")
    }

    func testExactTwoPhaseWireUsesSameCommandAndDistinctReceiptIDs() async throws {
        let (bridge, driver, media, writer) = try await make()
        defer { bridge.close() }
        try await prepared(bridge, driver, media)
        XCTAssertFalse(driver.outputEnabled)
        try receipt(XCTUnwrap(writer.messages.first), phase: "prepare")
        try bridge.receive(command("activate"), connectionID: driver.connectionID)
        try await until { bridge.state == .active }
        XCTAssertTrue(driver.outputEnabled)
        XCTAssertEqual(writer.messages.count, 2)
        try receipt(writer.messages[1], phase: "activate")
        XCTAssertNotEqual(writer.messages[0]["receipt_id"] as? String, writer.messages[1]["receipt_id"] as? String)
        XCTAssertEqual(writer.connections, [driver.connectionID, driver.connectionID])
    }

    func testPrepareWaitsForRealStartedAndReadyWrite() async throws {
        let (bridge, driver, media, writer) = try await make(ready: false)
        defer { bridge.close() }
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.hasRemoteDescription }
        XCTAssertTrue(writer.messages.isEmpty)
        media.started(provider)
        XCTAssertTrue(writer.messages.isEmpty)
        try driver.didWriteReady(connectionID: driver.connectionID)
        try await until { bridge.state == .prepared }
        XCTAssertEqual(writer.messages.count, 1)
    }

    func testDuplicatePendingPrepareDoesNotReapplySDP() async throws {
        let (bridge, driver, media, writer) = try await make()
        defer { bridge.close() }
        media.holdAnswer = true
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.answerContinuation != nil }
        for _ in 0..<20 { try bridge.receive(command("prepare"), connectionID: driver.connectionID) }
        XCTAssertEqual(media.answers, 1)
        XCTAssertTrue(writer.messages.isEmpty)
        media.releaseAnswer()
        try await until { media.hasRemoteDescription }
        media.started(provider)
        try await until { bridge.state == .prepared }
        XCTAssertEqual(writer.messages.count, 1)
    }

    func testCompletedPhaseDuplicatesReuseReceiptsWithoutReactivation() async throws {
        let (bridge, driver, media, writer) = try await make()
        defer { bridge.close() }
        try await prepared(bridge, driver, media)
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { writer.messages.count == 2 }
        XCTAssertEqual(writer.messages[0]["receipt_id"] as? String, writer.messages[1]["receipt_id"] as? String)
        try bridge.receive(command("activate"), connectionID: driver.connectionID)
        try await until { bridge.state == .active }
        try bridge.receive(command("activate"), connectionID: driver.connectionID)
        try await until { writer.messages.count == 4 }
        XCTAssertEqual(writer.messages[2]["receipt_id"] as? String, writer.messages[3]["receipt_id"] as? String)
        XCTAssertEqual(media.activations, 1)
        XCTAssertThrowsError(try bridge.receive(command("prepare"), connectionID: driver.connectionID))
    }

    func testActivationQueuesUntilPreparationAndICEReadiness() async throws {
        let (bridge, driver, media, writer) = try await make()
        defer { bridge.close() }
        media.isConnected = false
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try bridge.receive(command("activate"), connectionID: driver.connectionID)
        try await until { media.hasRemoteDescription }
        XCTAssertEqual(media.activations, 0)
        media.started(provider)
        try await until { bridge.state == .prepared }
        XCTAssertEqual(media.activations, 0)
        media.isConnected = true
        media.onConnectivityChanged?()
        try await until { bridge.state == .active }
        XCTAssertEqual(writer.messages.count, 2)
        XCTAssertEqual(media.activations, 1)
    }

    func testICEOrDuplicateStartedCannotActivateWithoutBackendCommand() async throws {
        let (bridge, driver, media, _) = try await make()
        defer { bridge.close() }
        try await prepared(bridge, driver, media)
        media.onConnectivityChanged?(); media.started(provider)
        XCTAssertEqual(media.activations, 0)
        XCTAssertFalse(driver.outputEnabled)
    }

    func testWrongConnectionRouteAndProviderAreRejected() async throws {
        let (bridge, driver, media, _) = try await make()
        defer { bridge.close() }
        XCTAssertThrowsError(try bridge.receive(command("prepare"), connectionID: UUID()))
        var values = command("prepare").values
        values["route"] = route(generation: 2)
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
        try await prepared(bridge, driver, media)
        values = command("activate").values
        values["provider_session_id"] = "old-provider"
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
        XCTAssertThrowsError(try bridge.receive(command("activate", id: UUID()), connectionID: driver.connectionID))
        XCTAssertEqual(media.activations, 0)
    }

    func testStrictCommandsRejectAdditionalFieldsTypesAndChangedSDP() async throws {
        let (bridge, driver, media, _) = try await make()
        defer { bridge.close() }
        var values = command("prepare").values
        values["api_key"] = "not_allowed"
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
        values = command("prepare").values; values["protocol_version"] = true
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
        values = command("prepare").values; values["answer_sdp"] = "invalid"
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
        try await prepared(bridge, driver, media)
        values = command("prepare").values; values["answer_sdp"] = "v=0\r\nchanged"
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
        values = command("activate").values; values["answer_sdp"] = "v=0"
        XCTAssertThrowsError(try bridge.receive(event(values), connectionID: driver.connectionID))
    }

    func testWrongNativeProviderStartedFailsClosed() async throws {
        let (bridge, driver, media, writer) = try await make()
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.hasRemoteDescription }
        media.started("another-provider")
        XCTAssertEqual(bridge.state, .failed)
        XCTAssertTrue(media.closed)
        XCTAssertTrue(writer.messages.isEmpty)
    }

    func testDeadlineFencesMissingStartupWithoutInventingReceipt() async throws {
        var now: TimeInterval = 10
        let (bridge, driver, media, writer) = try await make(clock: { now })
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.hasRemoteDescription }
        now = 31; bridge.checkDeadline()
        XCTAssertEqual(bridge.state, .failed)
        XCTAssertTrue(media.closed)
        XCTAssertTrue(writer.messages.isEmpty)
        media.started(provider)
        XCTAssertThrowsError(try bridge.receive(command("activate"), connectionID: driver.connectionID))
    }

    func testActualWatchdogBoundsStartup() async throws {
        let (bridge, driver, media, writer) = try await make(timeout: 0.05)
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { bridge.state == .failed }
        XCTAssertTrue(media.closed)
        XCTAssertTrue(writer.messages.isEmpty)
    }

    func testCloseDuringSuspendedSDPRejectsLateCompletion() async throws {
        let (bridge, driver, media, writer) = try await make()
        media.holdAnswer = true
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.answerContinuation != nil }
        bridge.close()
        media.releaseAnswer()
        try await until { media.hasRemoteDescription }
        media.started(provider)
        XCTAssertEqual(bridge.state, .closed)
        XCTAssertTrue(writer.messages.isEmpty)
        XCTAssertFalse(driver.outputEnabled)
    }

    func testReceiptWriteGuardRejectsAfterLocalInterruption() async throws {
        let (bridge, driver, media, writer) = try await make()
        writer.hold = true
        try bridge.receive(command("prepare"), connectionID: driver.connectionID)
        try await until { media.hasRemoteDescription }
        media.started(provider)
        try await until { writer.continuation != nil }
        _ = try driver.interruptLocally()
        writer.release()
        try await until { writer.finished == 1 }
        XCTAssertTrue(writer.messages.isEmpty)
        XCTAssertEqual(bridge.state, .failed)
        XCTAssertThrowsError(try bridge.receive(command("activate"), connectionID: driver.connectionID))
    }

    func testLostActivationReceiptFencesAndDoesNotReplaySpeech() async throws {
        let (bridge, driver, media, writer) = try await make()
        try await prepared(bridge, driver, media)
        writer.failWrites = true
        try bridge.receive(command("activate"), connectionID: driver.connectionID)
        try await until { bridge.state == .failed }
        XCTAssertEqual(media.activations, 1)
        XCTAssertFalse(driver.outputEnabled)
        XCTAssertTrue(media.closed)
        XCTAssertEqual(writer.messages.count, 1)
        XCTAssertThrowsError(try bridge.receive(command("activate"), connectionID: driver.connectionID))
    }

    func testOnlyNativeSessionStartedReachesStartupInbox() throws {
        let inbox = JarvisLiveStartupInbox(), source = NSObject()
        inbox.bind(source)
        for type in ["session.instructions.appended", "session.thinking.appended", "session.commentary.appended", "session.created", "session.ready", "session.input_transcript.delta"] {
            let data = try JSONSerialization.data(withJSONObject: ["type": type, "session": ["id": provider]])
            XCTAssertFalse(inbox.receive(data, binary: false, source: source))
        }
        let data = try JSONSerialization.data(withJSONObject: ["type": "session.started", "session": ["id": provider]])
        XCTAssertFalse(inbox.receive(data, binary: true, source: source))
        XCTAssertFalse(inbox.receive(data, binary: false, source: NSObject()))
        XCTAssertTrue(inbox.receive(data, binary: false, source: source))
        for _ in 0..<100 { XCTAssertFalse(inbox.receive(data, binary: false, source: source)) }
        guard case .started(let actual)? = inbox.take(source: source) else { return XCTFail("Missing genuine startup event") }
        XCTAssertEqual(actual.type, "session.started")
        XCTAssertEqual((actual.values["session"] as? [String: Any])?["id"] as? String, provider)
        XCTAssertNil(inbox.take(source: source))
        XCTAssertFalse(inbox.receive(data, binary: false, source: source))
    }

    func testStartupInboxBoundsPayloadAndRejectsLateRetiredSource() throws {
        let inbox = JarvisLiveStartupInbox(), source = NSObject()
        inbox.bind(source)
        XCTAssertFalse(inbox.receive(Data(repeating: 65, count: 65_537), binary: false, source: source))
        let data = try JSONSerialization.data(withJSONObject: ["type": "session.started", "session": ["id": provider]])
        XCTAssertTrue(inbox.receive(data, binary: false, source: source))
        inbox.close()
        XCTAssertNil(inbox.take(source: source))
        XCTAssertFalse(inbox.receive(data, binary: false, source: source))
        XCTAssertFalse(inbox.fail(source: source))
    }

    func testConflictingStartupCoalescesToFailureNotTwoCallbacks() throws {
        let inbox = JarvisLiveStartupInbox(), source = NSObject()
        inbox.bind(source)
        let first = try JSONSerialization.data(withJSONObject: ["type": "session.started", "session": ["id": provider]])
        let second = try JSONSerialization.data(withJSONObject: ["type": "session.started", "session": ["id": "other"]])
        XCTAssertTrue(inbox.receive(first, binary: false, source: source))
        XCTAssertFalse(inbox.receive(second, binary: false, source: source))
        guard case .failed? = inbox.take(source: source) else { return XCTFail("Conflicting evidence accepted") }
    }

    func testMalformedStartupAndChannelClosureFailClosed() throws {
        let inbox = JarvisLiveStartupInbox(), source = NSObject()
        inbox.bind(source)
        let data = try JSONSerialization.data(withJSONObject: ["type": "session.started"])
        XCTAssertTrue(inbox.receive(data, binary: false, source: source))
        guard case .failed? = inbox.take(source: source) else { return XCTFail("Malformed startup accepted") }
        let closing = JarvisLiveStartupInbox()
        closing.bind(source)
        XCTAssertTrue(closing.fail(source: source))
        XCTAssertFalse(closing.fail(source: source))
        guard case .failed? = closing.take(source: source) else { return XCTFail("Channel closure ignored") }
    }
}

@MainActor
private final class LiveSignalingTestMedia: JarvisLiveMediaTransport {
    var hasRemoteDescription = false
    var isConnected = true
    var outputEnabled = false
    var onConnectivityChanged: (() -> Void)?
    var onFailure: (() -> Void)?
    var onProviderStarted: ((JarvisRealtimeServerEvent) -> Void)?
    var answers = 0
    var activations = 0
    var closed = false
    var holdAnswer = false
    var answerContinuation: CheckedContinuation<Void, Never>?
    func prepareOffer() async throws -> String { "v=0\r\n" }
    func applyAnswer(_ sdp: String) async throws {
        answers += 1
        if holdAnswer { await withCheckedContinuation { answerContinuation = $0 } }
        hasRemoteDescription = true
    }
    func activate(microphoneEnabled: Bool) throws { activations += 1; outputEnabled = true }
    func setMicrophoneEnabled(_ enabled: Bool) {}
    func fenceAndClose() -> Bool { closed = true; outputEnabled = false; return true }
    func releaseAnswer() { answerContinuation?.resume(); answerContinuation = nil }
    func started(_ provider: String) {
        onProviderStarted?(JarvisRealtimeServerEvent(type: "session.started", values: ["type": "session.started", "session": ["id": provider]]))
    }
}

@MainActor
private final class LiveSignalingWriter {
    var messages: [[String: Any]] = []
    var connections: [UUID] = []
    var hold = false
    var failWrites = false
    var finished = 0
    var continuation: CheckedContinuation<Void, Never>?
    func send(_ payload: [String: Any], connection: UUID, current: @escaping @MainActor () -> Bool) async throws {
        defer { finished += 1 }
        if hold { await withCheckedContinuation { continuation = $0 } }
        try Task.checkCancellation()
        guard !failWrites, current() else { throw JarvisVoiceRouteError.notReady }
        messages.append(payload); connections.append(connection)
    }
    func release() { continuation?.resume(); continuation = nil; hold = false }
}
