import CryptoKit
import Foundation
import UIKit
import XCTest

@MainActor
final class JarvisReplayIntegrationTests: XCTestCase {
    func testProviderFinalizedTriggerWireAndRouteBoundaries() throws {
        let wire: [String: Any] = ["kind": "after_provider_input_finalized", "after_input_ordinal": 0, "timeout_seconds": 10]
        func decode(_ object: [String: Any]) throws -> JarvisReplayPlan.Trigger {
            try JSONDecoder().decode(JarvisReplayLaunchPlan.Trigger.self,
                from: JSONSerialization.data(withJSONObject: object)).value()
        }
        let trigger = try decode(wire)
        guard case .afterProviderInputFinalized(let previous, let timeout) = trigger else { return XCTFail("wrong trigger") }
        XCTAssertEqual(previous, 0); XCTAssertEqual(timeout, 10)
        let inputs: [JarvisReplayPlan.Utterance] = [.init(trigger: .atTime(0), pcm16: Data([200, 0])),
                                                   .init(trigger: trigger, pcm16: Data([200, 0]))]
        XCTAssertNoThrow(try JarvisReplayLaunchPlan.validateTriggerRoute(.realtime, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateTriggerRoute(.live, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
            invocationID: UUID(), taskMode: false, utterances: inputs))
        for field in ["delay_seconds", "response_id", "text", "canonical_requirement"] {
            var invalid = wire; invalid[field] = field == "delay_seconds" ? 1 : "not_allowed"
            XCTAssertThrowsError(try decode(invalid))
        }
        var missing = wire; missing.removeValue(forKey: "timeout_seconds")
        XCTAssertThrowsError(try decode(missing))
    }
    func testProviderFinalizedBoundsAndLegacyTaskContinuation() throws {
        for timeout in [0.0, -1, .infinity, .nan, 121] {
            XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 180,
                frames: [.init(at: 0, image: Data([1]))],
                utterances: [.init(trigger: .atTime(0), pcm16: Data([200, 0])),
                    .init(trigger: .afterProviderInputFinalized(afterInputOrdinal: 0, timeout: timeout), pcm16: Data([200, 0]))]))
        }
        let legacy = Data(#"{"kind":"after_input_complete","after_input_ordinal":0,"delay_seconds":1,"timeout_seconds":10}"#.utf8)
        let value = try JSONDecoder().decode(JarvisReplayLaunchPlan.Trigger.self, from: legacy).value()
        guard case .afterInputComplete(let previous, let delay, let timeout) = value else { return XCTFail("legacy changed") }
        XCTAssertEqual(previous, 0); XCTAssertEqual(delay, 1); XCTAssertEqual(timeout, 10)
    }
    private func waitingPlan() throws -> JarvisReplayPlan {
        try JarvisReplayPlan(runID: UUID(), duration: 10,
            frames: [.init(at: 0, image: Data([1])), .init(at: 2, image: Data([2]))],
            utterances: [.init(trigger: .outputStart(afterCursor: 0, delay: 0, timeout: 2), pcm16: Data([128, 0]))])
    }
    func testMatchingOutputAtDeadlineIsRejectedWhileSceneAdvances() throws {
        var clock = JarvisReplayTimeline(plan: try waitingPlan())
        let step = clock.advance(elapsed: 2, output: .init(cursor: 1, at: 2), inputIdle: true)
        XCTAssertEqual(step.failure, .outputWaitExpired); XCTAssertNil(step.utterance); XCTAssertEqual(step.frame, 1)
    }
    func testLateWakeupCannotReviveAnEarlierOutputEvent() throws {
        var clock = JarvisReplayTimeline(plan: try waitingPlan())
        let step = clock.advance(elapsed: 2.1, output: .init(cursor: 1, at: 1), inputIdle: true)
        XCTAssertEqual(step.failure, .outputWaitExpired); XCTAssertNil(step.utterance)
    }
    func testOutputStrictlyBeforeDeadlineStillTriggers() throws {
        var clock = JarvisReplayTimeline(plan: try waitingPlan())
        let step = clock.advance(elapsed: 1.9, output: .init(cursor: 1, at: 1.8), inputIdle: true)
        XCTAssertEqual(step.utterance, 0); XCTAssertNil(step.failure)
    }
    func testFenceTimeoutIsNotProofAndRetirementRejectsNewCallbacks() {
        let fence = JarvisReplayCallbackFence()
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            fence.perform { entered.signal(); _ = release.wait(timeout: .now() + 2) }
            done.signal()
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 1), .success)
        XCTAssertFalse(fence.close(timeout: 0.01))
        XCTAssertFalse(fence.perform { XCTFail("Retired callback entered") })
        release.signal()
        XCTAssertEqual(done.wait(timeout: .now() + 1), .success)
        XCTAssertTrue(fence.close(timeout: 0.1)); XCTAssertTrue(fence.close(timeout: 0))
    }
    func testConfirmedFenceWaitsForCopiedInFlightBlockToFinish() {
        let fence = JarvisReplayCallbackFence()
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0), closed = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            fence.perform { entered.signal(); _ = release.wait(timeout: .now() + 2) }
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 1), .success)
        DispatchQueue.global().async { if fence.close(timeout: 1) { closed.signal() } }
        XCTAssertEqual(closed.wait(timeout: .now() + 0.02), .timedOut)
        release.signal()
        XCTAssertEqual(closed.wait(timeout: .now() + 1), .success)
        XCTAssertFalse(fence.perform { XCTFail("Callback after confirmed close") })
    }
    func testReentrantFenceIsBoundedRatherThanInventingProof() {
        let fence = JarvisReplayCallbackFence()
        fence.perform { XCTAssertFalse(fence.close(timeout: 0.01)) }
        XCTAssertTrue(fence.close(timeout: 0))
    }
    func testActualRealtimeDriverAcceptsCustomDeviceAndRetiresIt() async throws {
        let client = JarvisWebRTCClient()
        let device = JarvisReplayAudioDevice(maxDuration: 45, postMediaCaptureSeconds: 20)
        XCTAssertEqual(device.captureDuration, 65)
        XCTAssertEqual(device.recordingCapacityDuration, 95)
        try client.useReplayAudioDevice(device)
        defer { client.close(); device.close(); try? client.useReplayAudioDevice(nil) }
        let offer = try await client.prepareOffer()
        XCTAssertTrue(offer.contains("m=audio")); XCTAssertTrue(offer.contains("oai-events") || offer.contains("m=application"))
        XCTAssertFalse(device.isRetired); XCTAssertFalse(client.microphoneEnabled)
        XCTAssertTrue(client.fenceRouteOutput(closeRoute: true)); XCTAssertTrue(device.isRetired)
        do { _ = try await client.prepareOffer(); XCTFail("Retired replay device reused") } catch {}
    }
    func testActualLiveMediaUsesCustomDeviceWithoutRequestingMicrophone() async throws {
        var permissionRequested = false
        let media = JarvisLiveWebRTCMedia(microphonePermission: { permissionRequested = true; return false })
        let device = JarvisReplayAudioDevice(maxDuration: 45, postMediaCaptureSeconds: 20)
        XCTAssertEqual(device.captureDuration, 65)
        XCTAssertEqual(device.recordingCapacityDuration, 95)
        try media.useReplayAudioDevice(device)
        defer { media.fenceAndClose(); device.close() }
        let offer = try await media.prepareOffer()
        XCTAssertTrue(offer.contains("m=audio")); XCTAssertFalse(permissionRequested)
        XCTAssertFalse(media.outputEnabled); XCTAssertFalse(device.isRetired)
        XCTAssertThrowsError(try media.activate(microphoneEnabled: true))
        XCTAssertTrue(media.fenceAndClose()); XCTAssertTrue(device.isRetired)
        XCTAssertTrue(media.fenceAndClose())
    }
    func testHeaderUsesBackendHexRunIdentityAndNormalRequestIsUnchanged() throws {
        let run = UUID(uuidString: "01234567-89ab-cdef-0123-456789abcdef")!
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8765/sessions/start")!)
        JarvisReplayScope.applyRunHeader(nil, to: &request)
        XCTAssertNil(request.value(forHTTPHeaderField: "X-Jarvis-Evaluation-Run"))
        JarvisReplayScope.applyRunHeader(run, to: &request)
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-Jarvis-Evaluation-Run"), "0123456789abcdef0123456789abcdef")
        XCTAssertEqual(try JarvisReplayScope.parseRunID("0123456789abcdef0123456789abcdef"), run)
    }
    func testSocketsRequireIsolatedOriginAndExactFreshSessionIdentity() throws {
        let base = URL(string: "http://127.0.0.1:8765")!, session = UUID().uuidString
        let url = try JarvisReplayScope.boundSocketURL(URL(string: "ws://127.0.0.1:8765/realtime/audio/ws?voice_route=live")!,
            base: base, sessionID: session)
        XCTAssertEqual(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.filter { $0.name == "session_id" }.count, 1)
        XCTAssertThrowsError(try JarvisReplayScope.boundSocketURL(URL(string: "ws://127.0.0.1:8766/realtime/audio/ws")!, base: base, sessionID: session))
        XCTAssertThrowsError(try JarvisReplayScope.boundSocketURL(URL(string: "ws://127.0.0.1:8765/realtime/audio/ws?session_id=\(UUID())")!, base: base, sessionID: session))
        XCTAssertThrowsError(try JarvisReplayScope.boundSocketURL(URL(string: "ws://127.0.0.1:8765/realtime/audio/ws?session_id=\(session)&session_id=\(session)")!, base: base, sessionID: session))
    }
    func testLocalAssetPathsCannotEscapeManifestRoot() {
        let root = URL(fileURLWithPath: "/tmp/replay-fixture")
        for path in ["../token", "/etc/passwd", "https://example.com/audio.wav"] {
            XCTAssertThrowsError(try JarvisReplayLaunchPlan.localFile(path, under: root))
        }
    }
    func testLaunchManifestLoadsWAVAndJPEGButNeverInjectsScriptText() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        func writeJSON(_ object: Any, _ name: String) throws {
            try JSONSerialization.data(withJSONObject: object).write(to: root.appendingPathComponent(name))
        }
        func le(_ value: Int, _ n: Int) -> Data { Data((0..<n).map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) }) }
        let pcm = Data([0, 0, 128, 0, 0, 0])
        var wav = Data("RIFF".utf8); wav += le(42, 4); wav += Data("WAVEfmt ".utf8)
        wav += le(16, 4); wav += le(1, 2); wav += le(1, 2); wav += le(48_000, 4)
        wav += le(96_000, 4); wav += le(2, 2); wav += le(16, 2); wav += Data("data".utf8); wav += le(6, 4); wav += pcm
        try wav.write(to: root.appendingPathComponent("speech.wav"))
        let jpeg = try XCTUnwrap(UIGraphicsImageRenderer(size: CGSize(width: 8, height: 8)).image { context in
            UIColor.gray.setFill(); context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        }.jpegData(compressionQuality: 0.7))
        try jpeg.write(to: root.appendingPathComponent("frame.jpg"))
        try Data("fixture-private-token-not-a-provider-key".utf8).write(to: root.appendingPathComponent("token"))
        try writeJSON(["format": "jarvis.speech_assets.v1", "clips": [[
            "id": "local-selector", "text": "Never inject this text", "relative_path": "speech.wav", "sha256": hash(wav),
            "sample_rate_hz": 48_000, "channels": 1, "sample_width_bytes": 2, "sample_count": 3,
            "first_active_sample": 1, "last_active_sample_exclusive": 2, "energy_threshold_abs_pcm16": 128
        ]]], "speech.json")
        let run = UUID()
        try writeJSON(["version": 1, "run_id": JarvisReplayScope.wireRunID(run),
            "backend_base_url": "http://127.0.0.1:8765", "bearer_token_file": "token",
            "owner_id": "isolated-fixture", "area_id": UUID().uuidString, "memory_bank_id": "isolated-bank",
            "voice_route": "live", "duration_seconds": 5, "speech_manifest": "speech.json",
            "frames": [["at_seconds": 0, "relative_path": "frame.jpg", "sha256": hash(jpeg)]],
            "utterances": [["clip_id": "local-selector", "trigger": ["kind": "at_time", "at_seconds": 1]]]
        ], "launch.json")
        let loaded = try JarvisReplayLaunchPlan.load(root.appendingPathComponent("launch.json"))
        XCTAssertEqual(loaded.plan.runID, run); XCTAssertEqual(loaded.route, .live)
        XCTAssertEqual(loaded.plan.utterances[0].pcm16, pcm); XCTAssertEqual(loaded.plan.frames[0].image, jpeg)
        XCTAssertEqual(loaded.scope.ownerID, "isolated-fixture")
        XCTAssertFalse(loaded.manualInput)
    }
    func testManualReplayIsExplicitRealtimeAtTimeOnlyAndKeepsAutomaticDefault() throws {
        let inputs = [JarvisReplayPlan.Utterance(trigger: .atTime(1), pcm16: Data([200, 0]))]
        XCTAssertFalse(try JarvisReplayLaunchPlan.validateManualInput(nil, route: .realtime,
            invocationID: nil, taskMode: false, utterances: inputs))
        XCTAssertFalse(try JarvisReplayLaunchPlan.validateManualInput("auto", route: .live,
            invocationID: nil, taskMode: false, utterances: inputs))
        XCTAssertTrue(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
            invocationID: UUID(), taskMode: false, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .live,
            invocationID: UUID(), taskMode: false, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
            invocationID: nil, taskMode: false, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
            invocationID: UUID(), taskMode: true, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("unknown", route: .realtime,
            invocationID: UUID(), taskMode: false, utterances: inputs))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
            invocationID: UUID(), taskMode: false, utterances: [
                .init(trigger: .outputStart(afterCursor: 0, delay: 0, timeout: 1), pcm16: Data([200, 0]))
            ]))
    }
    func testSourceClockWireRequiresExactOriginalBytesAndBoundedLocalPath() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let anchor = JarvisReplaySourceAnchor(runID: UUID(), invocationID: UUID(),
            clock: .init(monotonicBefore: 100, utc: Date(timeIntervalSince1970: 1_793_456_000), monotonicAfter: 100.001), bootID: UUID())
        let bytes = try JSONEncoder().encode(anchor)
        try bytes.write(to: root.appendingPathComponent("source-clock.json"))
        let hash = JarvisReplayFrameLineage.hash(bytes)
        func decode(_ fields: [String: Any]) throws -> JarvisReplayLaunchPlan.SourceClock {
            try JSONDecoder().decode(JarvisReplayLaunchPlan.SourceClock.self, from: JSONSerialization.data(withJSONObject: fields))
        }
        let fields: [String: Any] = ["version": 1, "phase_start_seconds": 67, "anchor_file": "source-clock.json", "anchor_sha256": hash]
        let loaded = try decode(fields).load(under: root)
        XCTAssertEqual(loaded.anchorData, bytes); XCTAssertEqual(loaded.phaseStart, 67)
        for (key, bad) in [("anchor_file", "../source-clock.json"), ("anchor_sha256", String(repeating: "0", count: 64))] {
            var changed = fields; changed[key] = bad
            XCTAssertThrowsError(try decode(changed).load(under: root))
        }
        var changed = fields; changed["phase_start_seconds"] = 0
        XCTAssertThrowsError(try decode(changed).load(under: root))
        let first = try decode(["version": 1, "phase_start_seconds": 0, "anchor_file": NSNull(), "anchor_sha256": NSNull()]).load(under: root)
        XCTAssertNil(first.anchor); XCTAssertNil(first.anchorData)
    }

    func testPreparedSpeechAssetPackWhenExplicitlyProvided() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["JARVIS_REPLAY_SPEECH_MANIFEST"] ?? env["TEST_RUNNER_JARVIS_REPLAY_SPEECH_MANIFEST"] else {
            throw XCTSkip("No explicit prepared speech manifest supplied; no provider calls are made by this test.")
        }
        let manifestURL = URL(fileURLWithPath: path)
        let speech = try JSONDecoder().decode(JarvisReplayLaunchPlan.Speech.self,
            from: JarvisReplayLaunchPlan.read(manifestURL, limit: 512 * 1024))
        XCTAssertEqual(speech.format, "jarvis.speech_assets.v1"); XCTAssertFalse(speech.clips.isEmpty)
        for clip in speech.clips {
            let url = try JarvisReplayLaunchPlan.localFile(clip.relative_path, under: manifestURL.deletingLastPathComponent())
            let asset = JarvisReplayAudioAsset(fileURL: url, format: .wavPCM16Mono48000, sha256: clip.sha256,
                sampleCount: clip.sample_count, thresholdPCM16: clip.energy_threshold_abs_pcm16 - 1,
                lastAboveThresholdSampleIndex: clip.last_active_sample_exclusive.map { $0 - 1 })
            XCTAssertEqual(try asset.loadPCM16().count, clip.sample_count * 2)
        }
        print("Prepared replay WAV assets validated: \(speech.clips.count). Offline bytes only, not frontend acceptance.")
    }

    func testResponseTriggerWireContractAndLiveUnsupportedBoundary() throws {
        let wire: [String: Any] = ["kind": "response_output_start", "after_input_ordinal": 0,
            "delay_seconds": 0.3, "timeout_seconds": 5, "max_pcm_age_seconds": 0.05,
            "response_id": "resp_fixture", "connection_id": "01234567-89ab-cdef-0123-456789abcdef",
            "text": "Never exported or sent to a model"]
        let decoded = try JSONDecoder().decode(JarvisReplayLaunchPlan.Trigger.self,
            from: JSONSerialization.data(withJSONObject: wire))
        let trigger = try decoded.value()
        guard case .responseOutputStart(let value) = trigger else { return XCTFail("Wrong trigger") }
        XCTAssertEqual(value.afterInputOrdinal, 0); XCTAssertEqual(value.delay, 0.3)
        XCTAssertEqual(value.maxPCMAge, 0.05); XCTAssertEqual(value.responseID, "resp_fixture")
        let utterances: [JarvisReplayPlan.Utterance] = [
            .init(trigger: .atTime(0), pcm16: Data([200, 0])), .init(trigger: trigger, pcm16: Data([200, 0]))]
        XCTAssertNoThrow(try JarvisReplayPlan(runID: UUID(), duration: 10,
            frames: [.init(at: 0, image: Data([1]))], utterances: utterances))
        XCTAssertNoThrow(try JarvisReplayLaunchPlan.validateTriggerRoute(.realtime, utterances: utterances))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateTriggerRoute(.live, utterances: utterances)) {
            XCTAssertEqual($0 as? JarvisReplayError, .unsupportedResponseIdentity)
        }
        var missing = wire; missing.removeValue(forKey: "max_pcm_age_seconds")
        XCTAssertThrowsError(try JSONDecoder().decode(JarvisReplayLaunchPlan.Trigger.self,
            from: JSONSerialization.data(withJSONObject: missing)).value())
    }

    func testResponseTriggerRejectsInvalidDependencyBoundsAndSilentCut() {
        for input in [-1, 1] {
            let clips: [JarvisReplayPlan.Utterance] = [.init(trigger: .atTime(0), pcm16: Data([1, 0])),
                .init(trigger: .responseOutputStart(.init(afterInputOrdinal: input, delay: 0,
                    timeout: 2, maxPCMAge: 0.05)), pcm16: Data([1, 0]))]
            XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 10,
                frames: [.init(at: 0, image: Data([1]))], utterances: clips))
        }
        for age in [Double(0), -Double.infinity, Double.nan, Double(3)] {
            let clips: [JarvisReplayPlan.Utterance] = [.init(trigger: .atTime(0), pcm16: Data([1, 0])),
                .init(trigger: .responseOutputStart(.init(afterInputOrdinal: 0, delay: 0,
                    timeout: 2, maxPCMAge: age)), pcm16: Data([1, 0]))]
            XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 10,
                frames: [.init(at: 0, image: Data([1]))], utterances: clips))
        }
        let silent: [JarvisReplayPlan.Utterance] = [.init(trigger: .atTime(0), pcm16: Data([1, 0])),
            .init(trigger: .responseOutputStart(.init(afterInputOrdinal: 0, delay: 0,
                timeout: 2, maxPCMAge: 0.05)), pcm16: Data([0, 0]))]
        XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 10,
            frames: [.init(at: 0, image: Data([1]))], utterances: silent))
    }

    func testLegacyOutputWaitKeepsItsExistingRunLimitBehavior() throws {
        var timeline = JarvisReplayTimeline(plan: try waitingPlan())
        let end = timeline.advance(elapsed: 10, output: nil, inputIdle: true)
        XCTAssertTrue(end.finished)
        XCTAssertNil(end.failure)
        XCTAssertNil(end.utterance)
    }

    func testDirectSpeechHandlerDoesNotRelabelProviderEvidenceAsBackendControl() throws {
        let client = JarvisWebRTCClient()
        let device = JarvisReplayAudioDevice(maxDuration: 1)
        try client.useReplayAudioDevice(device)
        defer { client.close(); device.close(); try? client.useReplayAudioDevice(nil) }
        client.handleProviderEvent(.init(type: "input_audio_buffer.speech_started",
            values: ["item_id": "item_direct_fixture"]), replayFromProviderDataChannel: true)
        XCTAssertTrue(device.evidence.controlEvents.filter { $0.stage == .backendControlHandled }.isEmpty)
        client.handleProviderEvent(.init(type: "input_audio_buffer.speech_stopped",
            values: ["item_id": "item_backend_fixture"]))
        let backend = device.evidence.controlEvents.filter { $0.stage == .backendControlHandled }
        XCTAssertEqual(backend.count, 1)
        XCTAssertEqual(backend.first?.fields["item_id"], "item_backend_fixture")
        // The direct receipt belongs to the existing data-channel observer;
        // calling the shared handler alone must not manufacture that receipt.
        XCTAssertTrue(device.evidence.controlEvents.filter { $0.stage == .providerDataChannelHandled }.isEmpty)
    }
}
