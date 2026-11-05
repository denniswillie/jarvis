import Foundation

private struct CheckFailure: Error { let message: String }
private func require(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw CheckFailure(message: message) }
}

@MainActor
private final class Fixture {
    typealias Owner = JarvisRecordingCoordinator<String>
    let context = Owner.Context(sessionID: UUID(), audioSessionID: "audio-test", generationID: UUID(), connectionID: UUID())
    var valid = true
    var starts = 0, finishes = 0
    var delayed = false, failRetain = false, failWrite = false
    var pending: CheckedContinuation<Owner.Capture, Error>?
    var automatic: (@Sendable () -> Void)?
    var retained: [(UUID, String)] = []
    var reports: [[String: Any]] = []
    var states: [Owner.State] = []
    var issues = 0
    lazy var owner = Owner(context: context, current: { [unowned self] in valid },
        start: { [unowned self] command, stopped in
            starts += 1; automatic = stopped
            if delayed { return try await withCheckedThrowingContinuation { pending = $0 } }
            return handle()
        }, retain: { [unowned self] command, completed in
            if failRetain { throw CheckFailure(message: "retention failed") }
            retained.append((command.recordingID, completed))
        }, write: { [unowned self] event, context in
            try require(context == self.context, "wrong connection")
            if failWrite { throw CheckFailure(message: "socket failed") }
            let text = try JarvisRecordingControl.encode(event)
            reports.append(try JSONSerialization.jsonObject(with: Data(text.utf8)) as! [String: Any])
        }, changed: { [unowned self] _, state in states.append(state) },
        issue: { [unowned self] _, _ in issues += 1 })

    func handle() -> Owner.Capture {
        Owner.Capture(captureID: UUID(), finish: { [weak self] in
            await MainActor.run { self?.finishes += 1 }
            return "original-file"
        })
    }

    func command(_ action: String, id: UUID, operation: UUID = UUID(), include: Bool = true,
                 generation: UUID? = nil) throws -> JarvisRecordingControl.Command {
        try .init(values: ["type": "recording.command", "version": 1, "action": action,
            "operation_id": operation.uuidString.lowercased(), "recording_id": id.uuidString.lowercased(),
            "session_id": context.sessionID.uuidString.lowercased(),
            "generation_id": (generation ?? context.generationID).uuidString.lowercased(),
            "audio_session_id": context.audioSessionID, "include_jarvis_audio": include,
            "max_bytes": 1_000_000, "max_duration_ms": 5_000])
    }

    func wait(_ condition: () -> Bool) async throws {
        for _ in 0..<100 {
            if condition() { return }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        throw CheckFailure(message: "fixture timed out")
    }
}

@main
private struct Checks {
    @MainActor static func main() async throws {
        do {
            let f = Fixture(), id = UUID()
            let start = try f.command("start", id: id)
            try f.owner.receive(start); try f.owner.receive(start)
            try await f.wait { f.states.contains(.recording) }
            try f.owner.receive(f.command("stop", id: id))
            try f.owner.receive(f.command("stop", id: id))
            await f.owner.drained()
            try require(f.starts == 1 && f.finishes == 1 && f.retained.count == 1, "duplicate effect")
            try require(f.reports.compactMap { $0["state"] as? String } == ["started", "stopped"], "report order")
            try require(f.reports.compactMap { $0["sequence"] as? Int } == [1, 2], "report sequence")
            try f.owner.receive(f.command("start", id: id))
            try require(f.starts == 1, "new call resurrected recording")
        }
        print("PASS exact replay, ordered reports and one retained original")
        do {
            let f = Fixture(), id = UUID(); f.delayed = true
            try f.owner.receive(f.command("start", id: id))
            try await f.wait { f.pending != nil }
            try f.owner.receive(f.command("stop", id: id))
            f.pending?.resume(returning: f.handle()); f.pending = nil
            await f.owner.drained()
            try require(f.finishes == 1 && f.retained.count == 1, "late attach was lost")
            try require(!f.states.contains(.recording), "start reported after stop")
            try require(f.reports.compactMap { $0["state"] as? String } == ["stopped"], "late started report")
        }
        print("PASS stop during attachment fences and retains once")
        do {
            let f = Fixture(), id = UUID()
            try f.owner.receive(f.command("stop", id: id))
            try f.owner.receive(f.command("start", id: id)); await f.owner.drained()
            try require(f.starts == 0, "stop tombstone revived")
            do { try f.owner.receive(f.command("start", id: id, include: false)); throw CheckFailure(message: "scope conflict accepted") }
            catch Fixture.Owner.Failure.conflictingRecording { }
            do { try f.owner.receive(f.command("start", id: UUID(), generation: UUID())); throw CheckFailure(message: "stale generation accepted") }
            catch is CheckFailure { throw CheckFailure(message: "stale generation accepted") }
            catch { }
            try require(f.starts == 0, "invalid command started capture")
        }
        print("PASS stop-before-start and generation/scope rejection")
        do {
            let f = Fixture(), id = UUID(); f.delayed = true
            try f.owner.receive(f.command("start", id: id)); try await f.wait { f.pending != nil }
            f.owner.retire(); f.valid = false
            f.pending?.resume(returning: f.handle()); f.pending = nil
            await f.owner.drained()
            try require(f.retained.count == 1 && f.reports.isEmpty, "retired result lost or sent on old socket")
            do { try f.owner.receive(f.command("start", id: UUID())); throw CheckFailure(message: "retired accepted start") }
            catch Fixture.Owner.Failure.retired { }
        }
        print("PASS retirement preserves late original and suppresses stale reports")
        do {
            let f = Fixture(), id = UUID(); f.failRetain = true
            try f.owner.receive(f.command("start", id: id)); try await f.wait { f.states.contains(.recording) }
            f.automatic?(); f.automatic?()
            try await f.wait { f.states.contains(.retentionFailed) }; await f.owner.drained()
            try require(f.finishes == 1 && !f.owner.isCapturing, "automatic stop duplicated or still capturing")
            try require(f.reports.compactMap { $0["state"] as? String } == ["started", "stopped"], "retention error misreported capture")
        }
        print("PASS repeated automatic stop and truthful retention failure")
        do {
            let f = Fixture(), id = UUID(); f.failWrite = true
            let start = try f.command("start", id: id)
            try f.owner.receive(start); try await f.wait { f.states.contains(.recording) }
            do { try f.owner.receive(f.command("start", id: UUID())); throw CheckFailure(message: "parallel capture accepted") }
            catch Fixture.Owner.Failure.busy { }
            do { try f.owner.receive(f.command("stop", id: id, operation: start.operationID)); throw CheckFailure(message: "operation collision accepted") }
            catch Fixture.Owner.Failure.conflictingOperation { }
            f.owner.stopLocally(recordingID: id); await f.owner.drained()
            try require(f.retained.count == 1 && f.issues == 2, "report failure lost accepted work")
        }
        print("PASS busy/conflicting commands and socket failure isolation")
    }
}
