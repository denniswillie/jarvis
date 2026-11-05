import CryptoKit
import Darwin
import Foundation
import UIKit
import XCTest

final class JarvisReplayTaskBarrierTests: XCTestCase {
    private final class QueuedFailure: @unchecked Sendable {
        private let lock = NSLock()
        private var deliveries: [JarvisReplayTaskBarrier.FailureDelivery] = []
        func append(_ delivery: @escaping JarvisReplayTaskBarrier.FailureDelivery) {
            lock.withLock { deliveries.append(delivery) }
        }
        var count: Int { lock.withLock { deliveries.count } }
        func take() -> JarvisReplayTaskBarrier.FailureDelivery? {
            lock.withLock { deliveries.isEmpty ? nil : deliveries.removeFirst() }
        }
    }

    @MainActor
    func testQueuedFailureThenRetirementCannotNotifyOrMutateCapture() async throws {
        let root = try root(), plan = try plan(), config = try configuration(root)
        let pending = evidence(plan, complete: false), queued = QueuedFailure()
        let controller = JarvisReplayTaskBarrier(configuration: config, plan: plan,
            evidence: { pending }, scheduleFailure: { queued.append($0) })
        defer { controller.stop() }
        var notifications = 0
        controller.onFailure = { _ in notifications += 1 }
        try await controller.bind(sessionID: session.uuidString)
        var invalid = wire(); invalid["reason"] = "sensitive_value_must_not_escape"
        try put(bytes(invalid), at: root)
        for _ in 0..<100 {
            if queued.count > 0 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        let delivery = try XCTUnwrap(queued.take())
        controller.stop()
        delivery()
        XCTAssertEqual(notifications, 0)
        XCTAssertNil(controller.failureDiagnostic)
        XCTAssertThrowsError(try controller.enqueue(requiresClaim: false) { _ in XCTFail("retired") })
    }

    @MainActor
    func testLiveFailureReportsOnceAndActualCaptureContainsOnlyStaticDiagnostic() async throws {
        let root = try root(), plan = try plan(), config = try configuration(root)
        let pending = evidence(plan, complete: false), queued = QueuedFailure()
        let controller = JarvisReplayTaskBarrier(configuration: config, plan: plan,
            evidence: { pending }, scheduleFailure: { queued.append($0) })
        defer { controller.stop() }
        var notifications = 0
        var capture = JarvisReplayPCMState(maxDuration: 30)
        controller.onFailure = { [weak controller] error in
            notifications += 1
            XCTAssertEqual(error as? JarvisReplayTaskError, .invalidControl)
            capture.recordTaskBarrierFailure("invalidControl", diagnostic: controller?.failureDiagnostic)
        }
        try await controller.bind(sessionID: session.uuidString)
        var invalid = wire(); invalid["reason"] = "sensitive_value_must_not_escape"
        try put(bytes(invalid), at: root)
        for _ in 0..<100 {
            if queued.count > 0 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        let delivery = try XCTUnwrap(queued.take())
        delivery(); delivery()
        XCTAssertEqual(notifications, 1)
        let diagnostic = try XCTUnwrap(controller.failureDiagnostic)
        XCTAssertEqual(diagnostic.stage, .receiptConsume)
        XCTAssertEqual(diagnostic.predicate, .receiptReason)
        XCTAssertTrue(diagnostic.isValid)
        XCTAssertEqual(capture.evidence.taskBarrierFailureCode, "invalidControl")
        let report = JarvisReplayEvidenceReport.make(evidence: capture.evidence, runID: run, finalUIStatus: "controlled_fixture")
        let retained = try XCTUnwrap(report["task_barrier_failure_diagnostic"] as? [String: Any])
        XCTAssertEqual(Set(retained.keys), Set(["schema_version", "error_code", "stage", "predicate",
            "barrier_generation", "observed_monotonic_s", "delivered_monotonic_s"]))
        XCTAssertEqual(retained["predicate"] as? String, "receipt_reason")
        XCTAssertLessThan(try bytes(retained).count, 1024)
        let data = try bytes(report), text = String(decoding: data, as: UTF8.self)
        XCTAssertFalse(text.contains("sensitive_value_must_not_escape"))
        XCTAssertFalse(text.contains(root.path))
        XCTAssertFalse(text.contains("fixture-worker"))
        let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
        attachment.name = "task-control-failure-capture.json"; attachment.lifetime = .keepAlways
        add(attachment)
        print("TASK_CONTROL_FAILURE_FIXTURE_BASE64=" + data.base64EncodedString())
    }

    func testClosedCaptureAndDuplicateFailureCannotOverwriteEvidence() throws {
        let diagnostic = JarvisReplayTaskFailureDiagnostic(stage: .controlRead, predicate: .readMetadata,
            generation: UUID(), observedAt: 10, deliveredAt: 11)
        var closed = JarvisReplayPCMState(maxDuration: 30)
        closed.close(at: 12)
        closed.recordTaskBarrierFailure("invalidControl", diagnostic: diagnostic)
        XCTAssertNil(closed.evidence.taskBarrierFailureCode)
        XCTAssertNil(closed.evidence.taskBarrierFailureDiagnostic)
        var once = JarvisReplayPCMState(maxDuration: 30)
        once.recordTaskBarrierFailure("invalidControl", diagnostic: diagnostic)
        once.recordTaskBarrierFailure("staleReceipt")
        XCTAssertEqual(once.evidence.taskBarrierFailureCode, "invalidControl")
        XCTAssertEqual(once.evidence.taskBarrierFailureDiagnostic?.predicate, .readMetadata)
        var invalidTime = JarvisReplayPCMState(maxDuration: 30)
        invalidTime.recordTaskBarrierFailure("invalidControl", diagnostic: .init(stage: .pollState,
            predicate: .directoryAvailable, generation: UUID(), observedAt: .nan, deliveredAt: 11))
        XCTAssertNil(invalidTime.evidence.taskBarrierFailureDiagnostic)
    }

    func testTerminalBlockedReceiptsRemainValidWithoutClaimOrFailure() throws {
        var state = try consumer()
        try state.consume(bytes(wire()), now: now, uptime: 100)
        let identity = state.claimIdentity
        for (offset, reason) in ["task_not_active", "canonical_unavailable"].enumerated() {
            var value = wire(sequence: offset + 2, blocked: true); value["reason"] = reason
            XCTAssertTrue(try state.consume(bytes(value), now: now, uptime: 100.1))
            XCTAssertEqual(state.claimIdentity, identity)
            XCTAssertNil(state.eligibleSequence(now: now, uptime: 100.1))
        }
    }

    func testReadFailureKeepsOriginalEnumAndStaticGuardPredicate() throws {
        let root = try root(), directory = try JarvisReplayTaskControlDirectory(root)
        defer { directory.close() }
        try put(Data("private_contents_must_not_escape".utf8), at: root)
        try FileManager.default.setAttributes([.posixPermissions: 0o644],
            ofItemAtPath: root.appendingPathComponent("task-barrier.json").path)
        XCTAssertThrowsError(try directory.readBarrier()) {
            XCTAssertEqual($0 as? JarvisReplayTaskError, .invalidControl)
        }
        XCTAssertEqual(directory.failurePredicate, .readMetadata)
    }

    private let run = UUID()
    private let barrier = UUID()
    private let session = UUID()
    private let taskID = UUID()
    private let attempt = UUID()
    private let request = UUID()
    private let event = UUID()
    private var now = Date()
    override func setUp() { super.setUp(); now = Date() }
    private typealias J = JarvisReplayTaskJSON

    private func scopeWire() -> [String: Any] {
        ["run_id": run.uuidString, "barrier_id": barrier.uuidString, "owner_id": "isolated-owner",
         "database_schema": "jarvis_sb_eval_\(run.uuidString.replacingOccurrences(of: "-", with: "").lowercased())_0",
         "worker_id": "fixture-worker", "runtime_host_id": "fixture-host", "model": "approved-fixture-model",
         "host_expires_at": J.timestamp(now.addingTimeInterval(300))]
    }
    private func scope() throws -> JarvisReplayTaskScope { try .init(scopeWire()) }
    private func consumer() throws -> JarvisReplayTaskReceiptConsumer { try .init(scope: scope(), sessionID: session) }
    private func wire(sequence: Int = 1, blocked: Bool = false) -> [String: Any] {
        var bound = scopeWire(); bound["session_id"] = session.uuidString
        let admission: [String: Any] = ["owner_id": "isolated-owner", "session_id": session.uuidString,
            "request_event_id": event.uuidString, "request_revision": 1, "request_id": request.uuidString,
            "task_id": taskID.uuidString, "accepted_revision": 1, "parameters_digest": String(repeating: "a", count: 64),
            "provider_call_id": "call_fixture", "operation": "create"]
        let claim: [String: Any] = ["task_id": taskID.uuidString, "task_revision": 1, "attempt_id": attempt.uuidString,
            "attempt_number": 1, "fence": 1, "worker_id": "fixture-worker", "runtime_host_id": "fixture-host",
            "lease_until": J.timestamp(now.addingTimeInterval(20)), "worker_expires_at": J.timestamp(now.addingTimeInterval(20)),
            "task_status": "running", "attempt_status": "running", "provider_state": "terminal"]
        return ["format": "jarvis.native_task_barrier.v1", "scope": bound, "admission": admission, "sequence": sequence,
            "observed_at": J.timestamp(now.addingTimeInterval(-0.1)), "expires_at": J.timestamp(now.addingTimeInterval(3)),
            "status": blocked ? "blocked" : "eligible", "reason": blocked ? "task_not_active" : "active_claim_observed",
            "claim": blocked ? NSNull() : claim]
    }
    private func bytes(_ value: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) }
    private func plan(cuts: Int = 1) throws -> JarvisReplayPlan {
        var inputs: [JarvisReplayPlan.Utterance] = [
            .init(trigger: .atTime(0), pcm16: Data([100, 0])),
            .init(trigger: .afterTaskClaim(afterInputOrdinal: 0, timeout: 10), pcm16: Data([101, 0]))]
        for _ in 0..<cuts {
            inputs.append(.init(trigger: .responseOutputStart(.init(afterInputOrdinal: inputs.count - 1,
                delay: 0.3, timeout: 10, maxPCMAge: 0.05, requireTaskClaim: true)), pcm16: Data([102, 0])))
            inputs.append(.init(trigger: .afterInputComplete(afterInputOrdinal: inputs.count - 1,
                delay: 1, timeout: 10), pcm16: Data([103, 0])))
        }
        return try .init(runID: run, duration: 30, frames: [0.0, 1, 2, 3, 4].map { .init(at: $0, image: Data([1])) }, utterances: inputs)
    }
    private func evidence(_ plan: JarvisReplayPlan, complete: Bool = true) -> JarvisReplayAudioEvidence {
        var result = JarvisReplayAudioEvidence()
        result.utterances = plan.utterances.indices.map {
            var item = JarvisReplayUtteranceEvidence(plan: plan, ordinal: $0)
            if complete {
                item.status = "sdk_input_complete"; item.acceptedSamples = item.sampleCount
                item.firstCallbackStart = Double($0 + 10); item.lastCallbackEnd = Double($0 + 10) + 0.1
                if item.trigger.requiresTaskClaim { item.taskClaimSequenceAtEnqueue = 1 }
            }
            return item
        }
        return result
    }
    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("task-barrier-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func configuration(_ root: URL) throws -> JarvisReplayTaskConfiguration {
        try .init(scope: scope(), manifestHash: String(repeating: "b", count: 64),
                  scopeHash: String(repeating: "c", count: 64), controls: root)
    }
    private func put(_ data: Data, at root: URL, name: String = "task-barrier.json") throws {
        let temporary = root.appendingPathComponent(UUID().uuidString)
        try data.write(to: temporary)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temporary.path)
        guard rename(temporary.path, root.appendingPathComponent(name).path) == 0 else { throw JarvisReplayTaskError.invalidControl }
    }

    /// Synthetic local bytes only. This exercises Main's exact staging layout, not voice/task acceptance.
    private func stagedManifest(taskCuts: Int?, duration: TimeInterval = 45) throws -> (url: URL, data: Data, controls: URL, scopeData: Data?) {
        let parent = try root().appendingPathComponent("Documents/ReplayInputs/" +
            run.uuidString.replacingOccurrences(of: "-", with: "").lowercased())
        let inputs = parent.appendingPathComponent("inputs"), controls = parent.appendingPathComponent("controls")
        try FileManager.default.createDirectory(at: inputs, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        if taskCuts != nil {
            try FileManager.default.createDirectory(at: controls, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        }
        func write(_ data: Data, _ name: String) throws {
            let url = inputs.appendingPathComponent(name)
            try data.write(to: url)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
        func le(_ value: Int, _ count: Int) -> Data {
            Data((0..<count).map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) })
        }
        let pcm = Data([0, 0, 128, 0, 0, 0])
        var wav = Data("RIFF".utf8); wav += le(36 + pcm.count, 4); wav += Data("WAVEfmt ".utf8)
        wav += le(16, 4); wav += le(1, 2); wav += le(1, 2); wav += le(48_000, 4)
        wav += le(96_000, 4); wav += le(2, 2); wav += le(16, 2); wav += Data("data".utf8)
        wav += le(pcm.count, 4); wav += pcm
        try write(wav, "fixture.wav")
        let jpeg = try XCTUnwrap(UIGraphicsImageRenderer(size: CGSize(width: 8, height: 8)).image { context in
            UIColor.gray.setFill(); context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        }.jpegData(compressionQuality: 0.7))
        try write(jpeg, "fixture.jpg")
        try write(Data("synthetic-test-token".utf8), "isolated-token")
        try write(bytes(["format": "jarvis.speech_assets.v1", "clips": [[
            "id": "fixture_clip", "relative_path": "fixture.wav", "sha256": J.hash(wav),
            "sample_rate_hz": 48_000, "channels": 1, "sample_width_bytes": 2, "sample_count": 3,
            "first_active_sample": 1, "last_active_sample_exclusive": 2, "energy_threshold_abs_pcm16": 128
        ]]]), "speech.json")
        var utterances: [[String: Any]] = []
        func input(_ trigger: [String: Any]) { utterances.append(["clip_id": "fixture_clip", "trigger": trigger]) }
        if let cuts = taskCuts {
            input(["kind": "at_time", "at_seconds": 7])
            input(["kind": "after_task_claim", "after_input_ordinal": 0, "timeout_seconds": 30])
            for _ in 0..<cuts {
                input(["kind": "response_output_start", "after_input_ordinal": utterances.count - 1,
                       "delay_seconds": 0.3, "timeout_seconds": 10, "max_pcm_age_seconds": 0.05, "require_task_claim": true])
                input(["kind": "after_input_complete", "after_input_ordinal": utterances.count - 1,
                       "delay_seconds": 1, "timeout_seconds": 10])
            }
        } else { input(["kind": "at_time", "at_seconds": 25]) }
        var manifest: [String: Any] = [
            "version": 1, "run_id": run.uuidString.replacingOccurrences(of: "-", with: "").lowercased(),
            "backend_base_url": "http://127.0.0.1:8765", "bearer_token_file": "isolated-token",
            "owner_id": "isolated-owner", "area_id": UUID().uuidString,
            "memory_bank_id": "jarvis_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased(),
            "voice_route": "realtime", "duration_seconds": duration, "speech_manifest": "speech.json",
            "frames": [["at_seconds": 20, "relative_path": "fixture.jpg", "sha256": J.hash(jpeg)]],
            "utterances": utterances
        ]
        var scopeData: Data?
        if taskCuts != nil {
            manifest["task_barrier"] = ["version": 1, "scope_file": "task-scope.json", "request_input_ordinal": 0]
            var scope = scopeWire(); scope["format"] = "jarvis.native_task_scope.v1"
            scopeData = try bytes(scope)
            try write(XCTUnwrap(scopeData), "task-scope.json")
        }
        // Deliberately pretty printed with a trailing newline: hash these actual launched bytes.
        var data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        data.append(10)
        try write(data, "replay.json")
        return (inputs.appendingPathComponent("replay.json"), data, controls, scopeData)
    }

    func testDefaultStagedReplayJSONLoadsIsolatedTokenWithoutTaskControls() throws {
        let fixture = try stagedManifest(taskCuts: nil)
        let loaded = try JarvisReplayLaunchPlan.load(fixture.url)
        XCTAssertEqual(fixture.url.lastPathComponent, "replay.json")
        XCTAssertEqual(loaded.plan.runID, run); XCTAssertEqual(loaded.route, .realtime)
        XCTAssertEqual(loaded.plan.duration, 45)
        XCTAssertEqual(loaded.plan.utterances.count, 1); XCTAssertEqual(loaded.plan.frames.first?.at, 20)
        XCTAssertNil(loaded.taskBarrier)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.controls.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.url.deletingLastPathComponent().appendingPathComponent("task-scope.json").path))
        try FileManager.default.removeItem(at: fixture.url.deletingLastPathComponent().appendingPathComponent("isolated-token"))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.load(fixture.url))
    }

    @MainActor
    func testSelectedStagedSceneDurationReachesSilentADMAndRuntimeCaptureValidation() throws {
        for duration in [45.0, 120, 165, 180] {
            let fixture = try stagedManifest(taskCuts: nil, duration: duration)
            let loaded = try JarvisReplayLaunchPlan.load(fixture.url)
            XCTAssertEqual(loaded.plan.duration, duration); XCTAssertNil(loaded.taskBarrier)
            let audio = JarvisReplayAudioDevice(plan: loaded.plan)
            defer { audio.close() }
            XCTAssertEqual(audio.sceneDuration, duration)
            XCTAssertNoThrow(try loaded.plan.validateAudioCapture(duration: audio.sceneDuration, retired: audio.isRetired))
        }
    }

    @MainActor
    func testRuntimeCaptureValidationRejectsMismatchedRetiredOrInvalidADMBounds() throws {
        let fixture = try stagedManifest(taskCuts: nil, duration: 180)
        let loaded = try JarvisReplayLaunchPlan.load(fixture.url)
        let audio = JarvisReplayAudioDevice(maxDuration: 45)
        defer { audio.close() }
        XCTAssertThrowsError(try loaded.plan.validateAudioCapture(duration: audio.sceneDuration, retired: audio.isRetired)) {
            XCTAssertEqual($0 as? JarvisReplayError, .invalidPlan)
        }
        let matching = JarvisReplayAudioDevice(plan: loaded.plan)
        XCTAssertNoThrow(try loaded.plan.validateAudioCapture(duration: matching.sceneDuration, retired: matching.isRetired))
        XCTAssertTrue(matching.close())
        XCTAssertThrowsError(try loaded.plan.validateAudioCapture(duration: matching.sceneDuration, retired: matching.isRetired))
        for duration in [0.0, -1, 180.000_001, Double.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try loaded.plan.validateAudioCapture(duration: duration, retired: false))
        }
    }

    func testStagedTaskModeAcceptsLongSceneAndRejectsOver180Seconds() throws {
        let fixture = try stagedManifest(taskCuts: 1, duration: 180)
        let loaded = try JarvisReplayLaunchPlan.load(fixture.url)
        XCTAssertEqual(loaded.plan.duration, 180); XCTAssertNotNil(loaded.taskBarrier)
        XCTAssertEqual(loaded.plan.utterances.count, 4)
        let oversized = try stagedManifest(taskCuts: nil, duration: 180.000_001)
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.load(oversized.url))
    }

    func testTaskStagedReplayJSONLoadsOneTwoThreeCutsAndExactLaunchedByteHashes() throws {
        for cuts in 1...3 {
            let fixture = try stagedManifest(taskCuts: cuts)
            let loaded = try JarvisReplayLaunchPlan.load(fixture.url)
            let task = try XCTUnwrap(loaded.taskBarrier)
            XCTAssertEqual(loaded.plan.utterances.count, 2 + 2 * cuts)
            XCTAssertEqual(loaded.route, .realtime); XCTAssertEqual(loaded.plan.frames.first?.at, 20)
            XCTAssertEqual(task.controls.standardizedFileURL, fixture.controls.standardizedFileURL)
            XCTAssertEqual(task.manifestHash, J.hash(fixture.data))
            XCTAssertEqual(task.scopeHash, J.hash(try XCTUnwrap(fixture.scopeData)))
            let normalized = try bytes(J.object(fixture.data, limit: 512 * 1024))
            XCTAssertNotEqual(task.manifestHash, J.hash(normalized))
            XCTAssertNil(try J.object(XCTUnwrap(fixture.scopeData), limit: 8192)["session_id"])
        }
    }

    func testLoadedTaskPathBindsActualSessionAndLaunchedHashesBeforeAnyMedia() async throws {
        let fixture = try stagedManifest(taskCuts: 1)
        let loaded = try JarvisReplayLaunchPlan.load(fixture.url)
        let config = try XCTUnwrap(loaded.taskBarrier)
        let pending = evidence(loaded.plan, complete: false)
        let controller = JarvisReplayTaskBarrier(configuration: config, plan: loaded.plan, evidence: { pending })
        defer { controller.stop() }
        let actualSession = session // Controlled normal REST result boundary; no API or media is opened.
        XCTAssertNotEqual(actualSession, loaded.plan.runID)
        try await controller.bind(sessionID: actualSession.uuidString)
        let directory = try JarvisReplayTaskControlDirectory(fixture.controls); defer { directory.close() }
        let binding = try J.object(XCTUnwrap(directory.read("session-binding.json", limit: 8192)), limit: 8192)
        XCTAssertEqual(binding["session_id"] as? String, actualSession.uuidString.lowercased())
        XCTAssertEqual(binding["manifest_file_sha256"] as? String, J.hash(fixture.data))
        XCTAssertEqual(binding["scope_file_sha256"] as? String, J.hash(try XCTUnwrap(fixture.scopeData)))
        XCTAssertNil(controller.eligibleSequence())
        XCTAssertNil(try directory.read("stimulus-completion.json", limit: 8192))
        XCTAssertThrowsError(try controller.enqueue(requiresClaim: true) { _ in XCTFail("No canonical claim supplied") })
    }

    func testStrictJSONRejectsDuplicateEscapedKeysAndSize() throws {
        for text in [#"{"a":1,"a":2}"#, #"{"a":1,"\u0061":2}"#, #"{"nested":{"a":1,"a":2}}"#] {
            XCTAssertThrowsError(try J.object(Data(text.utf8), limit: 16384))
        }
        XCTAssertThrowsError(try J.object(Data(repeating: 32, count: 16385), limit: 16384))
        XCTAssertThrowsError(try J.positive(true))
        XCTAssertThrowsError(try J.positive(1.5))
        XCTAssertThrowsError(try J.object(Data(#"{"a":NaN}"#.utf8), limit: 16384))
    }
    func testScopeLoadsExactRawHashesAndSiblingNotInputEscape() throws {
        let parent = try root()
        let inputs = parent.appendingPathComponent("inputs"), controls = parent.appendingPathComponent("controls")
        for directory in [inputs, controls] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        }
        var wire = scopeWire(); wire["format"] = "jarvis.native_task_scope.v1"
        let data = try bytes(wire), manifest = Data("{ \"version\": 1 }\n".utf8)
        let value = try JarvisReplayTaskConfiguration.load(scopeData: data, manifestData: manifest,
            inputsRoot: inputs, runID: run, ownerID: "isolated-owner")
        XCTAssertEqual(value.manifestHash, J.hash(manifest)); XCTAssertEqual(value.scopeHash, J.hash(data))
        XCTAssertEqual(value.controls.standardizedFileURL, controls.standardizedFileURL)
        XCTAssertThrowsError(try JarvisReplayTaskConfiguration.load(scopeData: data, manifestData: manifest,
            inputsRoot: inputs, runID: run, ownerID: "other"))
        wire["database_schema"] = "public"
        XCTAssertThrowsError(try JarvisReplayTaskScope(wire))
    }
    func testReceiptBindsActualSessionAndEveryStaticIdentity() throws {
        for key in ["run_id", "barrier_id", "owner_id", "session_id", "database_schema", "worker_id", "runtime_host_id", "model"] {
            var value = wire(), bound = value["scope"] as! [String: Any]
            bound[key] = key.hasSuffix("_id") && ["run_id", "barrier_id", "session_id"].contains(key) ? UUID().uuidString : "foreign"
            value["scope"] = bound
            var state = try consumer()
            XCTAssertThrowsError(try state.consume(bytes(value), now: now, uptime: 100), key)
            XCTAssertNil(state.eligibleSequence(now: now, uptime: 100))
        }
    }
    func testDuplicatePollIsNoEventAndSequenceNeverRewinds() throws {
        var state = try consumer()
        let data = try bytes(wire())
        XCTAssertTrue(try state.consume(data, now: now, uptime: 100))
        XCTAssertFalse(try state.consume(data, now: now, uptime: 100.1))
        XCTAssertEqual(state.sequence, 1)
        var rewritten = wire(); rewritten["reason"] = "canonical_unavailable"
        XCTAssertThrowsError(try state.consume(bytes(rewritten), now: now, uptime: 100.2))
        XCTAssertNil(state.eligibleSequence(now: now, uptime: 100.2))
    }
    func testBlockedReceiptClearsEligibilityButRetainsClaimAndWatermark() throws {
        var state = try consumer()
        try state.consume(bytes(wire()), now: now, uptime: 100)
        let identity = state.claimIdentity
        try state.consume(bytes(wire(sequence: 2, blocked: true)), now: now, uptime: 100.1)
        XCTAssertNil(state.eligibleSequence(now: now, uptime: 100.1))
        XCTAssertEqual(state.sequence, 2); XCTAssertEqual(state.claimIdentity, identity)
        try state.consume(bytes(wire(sequence: 3)), now: now, uptime: 100.2)
        XCTAssertEqual(state.eligibleSequence(now: now, uptime: 100.2), 3)
        XCTAssertEqual(state.claimIdentity, identity)
    }
    func testAdmissionReplacementCannotRebindTask() throws {
        var state = try consumer()
        try state.consume(bytes(wire()), now: now, uptime: 100)
        var replacement = wire(sequence: 2), admission = replacement["admission"] as! [String: Any]
        admission["request_id"] = UUID().uuidString; replacement["admission"] = admission
        XCTAssertThrowsError(try state.consume(bytes(replacement), now: now, uptime: 100.1))
        XCTAssertNil(state.eligibleSequence(now: now, uptime: 100.1))
    }
    func testAttemptFenceAndRevisionReplacementCannotRebindClaim() throws {
        for key in ["attempt_id", "attempt_number", "fence", "task_revision", "worker_id", "runtime_host_id"] {
            var state = try consumer()
            try state.consume(bytes(wire()), now: now, uptime: 100)
            var replacement = wire(sequence: 2), claim = replacement["claim"] as! [String: Any]
            claim[key] = ["attempt_number", "fence", "task_revision"].contains(key) ? 2 : UUID().uuidString as Any
            replacement["claim"] = claim
            XCTAssertThrowsError(try state.consume(bytes(replacement), now: now, uptime: 100.1), key)
            XCTAssertNil(state.eligibleSequence(now: now, uptime: 100.1))
        }
    }
    func testExpiryUsesWallAndMonotonicClocksAndRejectsInvalidTTL() throws {
        var state = try consumer()
        try state.consume(bytes(wire()), now: now, uptime: 100)
        XCTAssertNotNil(state.eligibleSequence(now: now, uptime: 100))
        XCTAssertNil(state.eligibleSequence(now: now.addingTimeInterval(4), uptime: 100.1))
        XCTAssertNil(state.eligibleSequence(now: now, uptime: 104))
        for delta in [-1.0, 6] {
            var value = wire(); value["expires_at"] = J.timestamp(now.addingTimeInterval(delta))
            var fresh = try consumer()
            XCTAssertThrowsError(try fresh.consume(bytes(value), now: now, uptime: 100))
        }
        var value = wire(); value["observed_at"] = J.timestamp(now.addingTimeInterval(1))
        var fresh = try consumer()
        XCTAssertThrowsError(try fresh.consume(bytes(value), now: now, uptime: 100))
    }
    func testBarrierRequiresRunningClaimAndExpiryInsideLease() throws {
        for key in ["task_status", "attempt_status", "provider_state", "lease_until", "worker_expires_at"] {
            var value = wire(), claim = value["claim"] as! [String: Any]
            claim[key] = key.hasSuffix("until") || key.hasSuffix("_at") ? J.timestamp(now) : "invalid"
            value["claim"] = claim
            var state = try consumer()
            XCTAssertThrowsError(try state.consume(bytes(value), now: now, uptime: 100))
        }
    }
    func testPrivateFilesRejectSymlinkPublicModeOversizeAndDirectoryReplacement() throws {
        let root = try root(), directory = try JarvisReplayTaskControlDirectory(root)
        defer { directory.close() }
        XCTAssertNil(try directory.readBarrier())
        try put(Data(repeating: 32, count: 16385), at: root)
        XCTAssertThrowsError(try directory.readBarrier())
        try put(bytes(wire()), at: root)
        XCTAssertNotNil(try directory.readBarrier())
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: root.appendingPathComponent("task-barrier.json").path)
        XCTAssertThrowsError(try directory.readBarrier())
        try FileManager.default.removeItem(at: root.appendingPathComponent("task-barrier.json"))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("task-barrier.json"), withDestinationURL: root.appendingPathComponent("missing"))
        XCTAssertThrowsError(try directory.readBarrier())
        let moved = root.appendingPathExtension("retired")
        try FileManager.default.moveItem(at: root, to: moved)
        addTeardownBlock { try? FileManager.default.removeItem(at: moved) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        XCTAssertThrowsError(try directory.readBarrier())
    }
    func testAtomicReplacementAfterOpenReturnsNoOldBytesThenExactFreshReceipt() throws {
        let root = try root(), old = try bytes(wire()), fresh = try bytes(wire(sequence: 2))
        try put(old, at: root)
        var opens = 0
        let directory = try JarvisReplayTaskControlDirectory(root, afterReadOpen: { name in
            XCTAssertEqual(name, "task-barrier.json")
            opens += 1
            guard opens == 1 else { return }
            do { try self.put(fresh, at: root) }
            catch { XCTFail("Atomic replacement failed: \(error)") }
        })
        defer { directory.close() }

        XCTAssertNil(try directory.readBarrier(), "The unlinked open inode must not supply old receipt bytes")
        let current = try XCTUnwrap(directory.readBarrier())
        XCTAssertEqual(opens, 2)
        XCTAssertEqual(current, fresh)
        XCTAssertNotEqual(current, old)
        var state = try consumer()
        XCTAssertTrue(try state.consume(current, now: now, uptime: 100))
        XCTAssertEqual(state.eligibleSequence(now: now, uptime: 100), 2)
    }

    func testUnlinkAfterOpenReturnsUnavailableUntilExactFreshReceiptAppears() throws {
        let root = try root(), fresh = try bytes(wire(sequence: 2))
        try put(bytes(wire()), at: root)
        var opens = 0
        let directory = try JarvisReplayTaskControlDirectory(root, afterReadOpen: { name in
            XCTAssertEqual(name, "task-barrier.json")
            opens += 1
            guard opens == 1 else { return }
            XCTAssertEqual(unlink(root.appendingPathComponent(name).path), 0)
        })
        defer { directory.close() }

        XCTAssertNil(try directory.readBarrier())
        XCTAssertNil(try directory.readBarrier(), "A missing current name cannot fall back to the old inode")
        XCTAssertEqual(opens, 1)
        try put(fresh, at: root)
        XCTAssertEqual(try directory.readBarrier(), fresh)
        XCTAssertEqual(opens, 2)
    }

    func testUnlinkedOpenReceiptStillRejectsWrongModeEmptyAndOversizeMetadata() throws {
        let valid = try bytes(wire())
        for (data, mode) in [(valid, 0o644), (Data(), 0o600), (Data(repeating: 32, count: 16385), 0o600)] {
            let root = try root()
            try put(data, at: root)
            try FileManager.default.setAttributes([.posixPermissions: mode],
                ofItemAtPath: root.appendingPathComponent("task-barrier.json").path)
            var opens = 0
            let directory = try JarvisReplayTaskControlDirectory(root, afterReadOpen: { name in
                opens += 1
                XCTAssertEqual(unlink(root.appendingPathComponent(name).path), 0)
            })
            defer { directory.close() }
            XCTAssertThrowsError(try directory.readBarrier()) {
                XCTAssertEqual($0 as? JarvisReplayTaskError, .invalidControl)
            }
            XCTAssertEqual(opens, 1)
            XCTAssertEqual(directory.failurePredicate, .readMetadata)
        }
    }

    func testHardLinkCreatedAfterOpenRemainsFatal() throws {
        let root = try root()
        try put(bytes(wire()), at: root)
        var opens = 0
        let directory = try JarvisReplayTaskControlDirectory(root, afterReadOpen: { name in
            opens += 1
            XCTAssertEqual(link(root.appendingPathComponent(name).path,
                root.appendingPathComponent("receipt-hardlink").path), 0)
        })
        defer { directory.close() }
        XCTAssertThrowsError(try directory.readBarrier()) {
            XCTAssertEqual($0 as? JarvisReplayTaskError, .invalidControl)
        }
        XCTAssertEqual(opens, 1)
        XCTAssertEqual(directory.failurePredicate, .readMetadata)
    }

    func testUnlinkedReceiptCannotHideControlDirectoryReplacement() throws {
        let root = try root(), moved = root.appendingPathExtension("retired")
        addTeardownBlock { try? FileManager.default.removeItem(at: moved) }
        try put(bytes(wire()), at: root)
        var opens = 0
        let directory = try JarvisReplayTaskControlDirectory(root, afterReadOpen: { name in
            opens += 1
            XCTAssertEqual(unlink(root.appendingPathComponent(name).path), 0)
            do {
                try FileManager.default.moveItem(at: root, to: moved)
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false,
                    attributes: [.posixPermissions: 0o700])
            } catch { XCTFail("Control directory replacement failed: \(error)") }
        })
        defer { directory.close() }
        XCTAssertThrowsError(try directory.readBarrier()) {
            XCTAssertEqual($0 as? JarvisReplayTaskError, .invalidControl)
        }
        XCTAssertEqual(opens, 1)
        XCTAssertEqual(directory.failurePredicate, .directoryIdentity)
    }

    func testSymlinkReplacementAfterOpenIsUnavailableThenRejectedWithoutFollowing() throws {
        let root = try root(), fresh = try bytes(wire(sequence: 2))
        try put(bytes(wire()), at: root)
        try put(fresh, at: root, name: "fresh-target.json")
        var opens = 0
        let directory = try JarvisReplayTaskControlDirectory(root, afterReadOpen: { name in
            opens += 1
            XCTAssertEqual(unlink(root.appendingPathComponent(name).path), 0)
            do {
                try FileManager.default.createSymbolicLink(at: root.appendingPathComponent(name),
                    withDestinationURL: root.appendingPathComponent("fresh-target.json"))
            } catch { XCTFail("Symlink replacement failed: \(error)") }
        })
        defer { directory.close() }
        XCTAssertNil(try directory.readBarrier())
        XCTAssertThrowsError(try directory.readBarrier()) {
            XCTAssertEqual($0 as? JarvisReplayTaskError, .invalidControl)
        }
        XCTAssertEqual(opens, 1, "O_NOFOLLOW must reject before the post-open hook")
        XCTAssertEqual(directory.failurePredicate, .readOpen)
    }

    func testOrdinaryPollClearsMissingReceiptEligibilityThenAdmitsOnlyFreshSequence() async throws {
        let root = try root(), plan = try plan(), config = try configuration(root)
        let pending = evidence(plan, complete: false)
        let controller = JarvisReplayTaskBarrier(configuration: config, plan: plan, evidence: { pending })
        defer { controller.stop() }
        try await controller.bind(sessionID: session.uuidString)
        try put(bytes(wire()), at: root)
        for _ in 0..<50 {
            if controller.eligibleSequence() == 1 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(controller.eligibleSequence(), 1)
        try FileManager.default.removeItem(at: root.appendingPathComponent("task-barrier.json"))
        for _ in 0..<50 {
            if controller.eligibleSequence() == nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertNil(controller.eligibleSequence())
        XCTAssertThrowsError(try controller.enqueue(requiresClaim: true) { _ in XCTFail("No stale claim fallback") })
        try put(bytes(wire(sequence: 2)), at: root)
        for _ in 0..<50 {
            if controller.eligibleSequence() == 2 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(controller.eligibleSequence(), 2)
        var accepted: Int?
        try controller.enqueue(requiresClaim: true) { accepted = $0 }
        XCTAssertEqual(accepted, 2)
        XCTAssertNil(controller.failureDiagnostic)
        XCTAssertFalse(controller.completionPublished)
    }

    func testAppObservationPublicationIsPrivateBoundedAndOneShot() throws {
        let directory = try JarvisReplayTaskControlDirectory(root())
        defer { directory.close() }
        let payload = Data("{}".utf8)
        try directory.publish("session-binding.json", data: payload)
        XCTAssertEqual(try directory.read("session-binding.json", limit: 8192), payload)
        XCTAssertThrowsError(try directory.publish("session-binding.json", data: payload))
        XCTAssertThrowsError(try directory.publish("stimulus-completion.json", data: Data(repeating: 32, count: 8193)))
        XCTAssertThrowsError(try directory.publish("release.json", data: payload))
    }
    func testTaskSequenceSupportsOneTwoThreeCutsAndRejectsOtherShapes() throws {
        let config = JarvisReplayLaunchPlan.TaskBarrier(version: 1, scope_file: "task-scope.json", request_input_ordinal: 0)
        for cuts in 1...3 {
            XCTAssertNoThrow(try JarvisReplayLaunchPlan.validateTaskMode(config, route: .realtime, utterances: plan(cuts: cuts).utterances))
        }
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateTaskMode(config, route: .live, utterances: plan().utterances))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateTaskMode(nil, route: .realtime, utterances: plan().utterances))
        XCTAssertThrowsError(try JarvisReplayLaunchPlan.validateTaskMode(config, route: .realtime, utterances: Array(plan().utterances.dropLast())))
        let ordinary = [JarvisReplayPlan.Utterance(trigger: .atTime(0), pcm16: Data([1, 0]))]
        XCTAssertNoThrow(try JarvisReplayLaunchPlan.validateTaskMode(nil, route: .live, utterances: ordinary))
        XCTAssertNoThrow(try JarvisReplayLaunchPlan.validateTaskMode(nil, route: .realtime, utterances: ordinary))
    }
    func testTriggerWireRejectsMisplacedClaimFlagAndFutureDependency() throws {
        let decoder = JSONDecoder()
        for kind in ["at_time", "after_task_claim", "after_input_complete"] {
            let wire: [String: Any] = ["kind": kind, "at_seconds": 0, "after_input_ordinal": 0,
                "timeout_seconds": 10, "delay_seconds": 1, "require_task_claim": false]
            XCTAssertThrowsError(try decoder.decode(JarvisReplayLaunchPlan.Trigger.self, from: bytes(wire)).value())
        }
        for previous in [-1, 1] {
            XCTAssertThrowsError(try JarvisReplayPlan(runID: run, duration: 30, frames: [.init(at: 0, image: Data([1]))],
                utterances: [.init(trigger: .atTime(0), pcm16: Data([1, 0])),
                             .init(trigger: .afterTaskClaim(afterInputOrdinal: previous, timeout: 10), pcm16: Data([1, 0]))]))
        }
    }
    func testClaimWaitUsesActualSDKCompletionWhileFramesContinue() throws {
        var timeline = JarvisReplayTimeline(plan: try plan())
        XCTAssertEqual(timeline.advance(elapsed: 0, output: nil, inputIdle: true, origin: 100).utterance, 0)
        let waiting = timeline.advance(elapsed: 4, output: nil, inputIdle: true, origin: 100, taskClaimSequence: 1)
        XCTAssertEqual(waiting.frame, 4); XCTAssertNil(waiting.utterance)
        let snapshot = JarvisReplayResponseSnapshot(inputCompletion: .init(ordinal: 0, at: 105))
        XCTAssertNil(timeline.advance(elapsed: 5, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100).utterance)
        let ready = timeline.advance(elapsed: 5.1, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100, taskClaimSequence: 1)
        XCTAssertEqual(ready.utterance, 1); XCTAssertEqual(ready.taskClaimSequence, 1)
        XCTAssertNil(timeline.advance(elapsed: 5.2, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100, taskClaimSequence: 1).utterance)
    }
    func testClaimTimeoutCannotPassWithoutReachingPrerequisite() throws {
        var timeline = JarvisReplayTimeline(plan: try plan())
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true, origin: 100)
        let snapshot = JarvisReplayResponseSnapshot(inputCompletion: .init(ordinal: 0, at: 105))
        let result = timeline.advance(elapsed: 15, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100, taskClaimSequence: 1)
        XCTAssertEqual(result.failure, .outputWaitExpired); XCTAssertNil(result.utterance)
    }
    func testContinueUsesActualCallbackEndAndPreservesInputIdleFence() throws {
        let inputs: [JarvisReplayPlan.Utterance] = [.init(trigger: .atTime(0), pcm16: Data([1, 0])),
            .init(trigger: .afterInputComplete(afterInputOrdinal: 0, delay: 1, timeout: 5), pcm16: Data([1, 0]))]
        let plan = try JarvisReplayPlan(runID: run, duration: 30, frames: [.init(at: 0, image: Data([1]))], utterances: inputs)
        var timeline = JarvisReplayTimeline(plan: plan)
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true, origin: 100)
        let snapshot = JarvisReplayResponseSnapshot(inputCompletion: .init(ordinal: 0, at: 119))
        XCTAssertNil(timeline.advance(elapsed: 19.9, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100).utterance)
        XCTAssertEqual(timeline.advance(elapsed: 20, output: nil, inputIdle: false, responseSnapshot: snapshot, origin: 100).failure, .overlappingInput)
    }
    func testResponsePauseRequiresBothCanonicalClaimAndExistingFreshPCM() throws {
        var timeline = JarvisReplayTimeline(plan: try plan())
        _ = timeline.advance(elapsed: 0, output: nil, inputIdle: true, origin: 100)
        _ = timeline.advance(elapsed: 1, output: nil, inputIdle: true,
            responseSnapshot: .init(inputCompletion: .init(ordinal: 0, at: 100.5)), origin: 100, taskClaimSequence: 1)
        let identity = JarvisReplayResponseIdentity(connectionID: UUID(), responseID: "resp_fixture", itemID: "item_fixture", outputEpoch: 1)
        let active = JarvisReplayDecodedSample(sampleOffset: 0, nominalUptime: 108, outputEpoch: 1)
        let output = JarvisReplayResponseObservation(identity: identity, admittedAt: 107.5,
            firstActive: active, latestActive: active, callbackEnd: 108.3)
        let snapshot = JarvisReplayResponseSnapshot(inputCompletion: .init(ordinal: 1, at: 107), output: output)
        XCTAssertNil(timeline.advance(elapsed: 8.31, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100).utterance)
        let cut = timeline.advance(elapsed: 8.32, output: nil, inputIdle: true, responseSnapshot: snapshot, origin: 100, taskClaimSequence: 2)
        XCTAssertEqual(cut.utterance, 2); XCTAssertEqual(cut.taskClaimSequence, 2); XCTAssertNotNil(cut.responseSelection)
    }
    func testActualPCMStateRequiresClaimAndSDKCompletionAtEnqueue() throws {
        let plan = try plan()
        var state = JarvisReplayPCMState(maxDuration: 30)
        try state.register(plan); state.gateInput(true)
        try state.enqueue(plan.utterances[0].pcm16, utteranceOrdinal: 0, requestedAt: 1)
        _ = state.renderInput(at: 1); state.inputCallbackStarted(at: 1)
        state.inputCallbackCompleted(status: 0, at: 1.01)
        XCTAssertEqual(state.evidence.utterances[0].status, "sdk_input_complete")
        var missingClaim = state
        XCTAssertThrowsError(try missingClaim.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1, requestedAt: 2))
        try state.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1, requestedAt: 2, taskClaimSequence: 3)
        XCTAssertEqual(state.evidence.utterances[1].taskClaimSequenceAtEnqueue, 3)
        var refused = JarvisReplayPCMState(maxDuration: 30)
        try refused.register(plan); refused.gateInput(true)
        XCTAssertThrowsError(try refused.enqueue(plan.utterances[1].pcm16, utteranceOrdinal: 1, requestedAt: 2, taskClaimSequence: 3))
    }
    func testCompletionHasEveryOrdinalCountsHashesAndNoReleaseOrAudio() throws {
        let plan = try plan(cuts: 3), configuration = try configuration(root())
        var state = try consumer()
        try state.consume(bytes(wire()), now: now, uptime: 100)
        let data = try XCTUnwrap(JarvisReplayTaskCompletion.make(configuration: configuration, sessionID: session, plan: plan,
            evidence: evidence(plan), consumer: state, now: now, uptime: 100))
        XCTAssertLessThanOrEqual(data.count, 8192)
        let value = try J.object(data, limit: 8192), inputs = try XCTUnwrap(value["inputs"] as? [[String: Any]])
        XCTAssertEqual(inputs.count, 8); XCTAssertEqual(inputs.compactMap { $0["input_ordinal"] as? Int }, Array(0..<8))
        XCTAssertEqual(value["manifest_file_sha256"] as? String, configuration.manifestHash)
        XCTAssertEqual(value["scope_file_sha256"] as? String, configuration.scopeHash)
        XCTAssertNil(value["release"]); XCTAssertNil(value["audio"]); XCTAssertNil(value["text"])
    }
    func testCompletionRejectsMissingFailedCountOrStaleClaim() throws {
        let plan = try plan(), configuration = try configuration(root())
        var state = try consumer()
        try state.consume(bytes(wire()), now: now, uptime: 100)
        XCTAssertNil(try JarvisReplayTaskCompletion.make(configuration: configuration, sessionID: session, plan: plan,
            evidence: evidence(plan, complete: false), consumer: state, now: now, uptime: 100))
        for index in 0..<4 {
            var evidence = evidence(plan); evidence.utterances[index].acceptedSamples = 0
            XCTAssertThrowsError(try JarvisReplayTaskCompletion.make(configuration: configuration, sessionID: session, plan: plan,
                evidence: evidence, consumer: state, now: now, uptime: 100))
        }
        var failed = evidence(plan); failed.callbackFailure = -1
        XCTAssertThrowsError(try JarvisReplayTaskCompletion.make(configuration: configuration, sessionID: session, plan: plan,
            evidence: failed, consumer: state, now: now, uptime: 100))
        try state.consume(bytes(wire(sequence: 2, blocked: true)), now: now, uptime: 100)
        XCTAssertNil(try JarvisReplayTaskCompletion.make(configuration: configuration, sessionID: session, plan: plan,
            evidence: evidence(plan), consumer: state, now: now, uptime: 100))
    }
    func testIndependentPollPublishesCompletionOnlyAfterBoundFreshClaim() async throws {
        let root = try root(), plan = try plan(), config = try configuration(root)
        let completed = evidence(plan)
        let controller = JarvisReplayTaskBarrier(configuration: config, plan: plan, evidence: { completed })
        defer { controller.stop() }
        try await controller.bind(sessionID: session.uuidString)
        XCTAssertFalse(controller.completionPublished)
        try put(bytes(wire()), at: root)
        for _ in 0..<50 {
            if controller.completionPublished { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(controller.completionPublished)
        let directory = try JarvisReplayTaskControlDirectory(root); defer { directory.close() }
        let receipt = try J.object(XCTUnwrap(directory.read("stimulus-completion.json", limit: 8192)), limit: 8192)
        XCTAssertEqual((receipt["inputs"] as? [[String: Any]])?.count, 4)
        XCTAssertEqual(receipt["session_id"] as? String, session.uuidString.lowercased())
        controller.stop()
        XCTAssertTrue(controller.completionPublished)
    }

    func testIndependentPollBindsBeforeEnqueueAndStopJoinsWithoutRelease() async throws {
        let root = try root(), plan = try plan(), config = try configuration(root)
        let pending = evidence(plan, complete: false)
        let controller = JarvisReplayTaskBarrier(configuration: config, plan: plan, evidence: { pending })
        defer { controller.stop() }
        XCTAssertThrowsError(try controller.enqueue(requiresClaim: false) { _ in XCTFail("Before binding") })
        try await controller.bind(sessionID: session.uuidString)
        let directory = try JarvisReplayTaskControlDirectory(root); defer { directory.close() }
        let binding = try J.object(XCTUnwrap(directory.read("session-binding.json", limit: 8192)), limit: 8192)
        XCTAssertEqual(binding["session_id"] as? String, session.uuidString.lowercased())
        XCTAssertNil(controller.eligibleSequence())
        try put(bytes(wire()), at: root)
        for _ in 0..<50 {
            if controller.eligibleSequence() != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(controller.eligibleSequence(), 1)
        var accepted: Int?
        try controller.enqueue(requiresClaim: true) { accepted = $0 }
        XCTAssertEqual(accepted, 1)
        try put(bytes(wire(sequence: 2, blocked: true)), at: root)
        for _ in 0..<50 {
            if controller.eligibleSequence() == nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertThrowsError(try controller.enqueue(requiresClaim: true) { _ in XCTFail("Blocked claim") })
        XCTAssertNil(try directory.read("stimulus-completion.json", limit: 8192))
        controller.stop()
        XCTAssertThrowsError(try controller.enqueue(requiresClaim: false) { _ in XCTFail("Retired") })
        do { try await controller.bind(sessionID: UUID().uuidString); XCTFail("Rebound retired run") } catch {}
    }
}
