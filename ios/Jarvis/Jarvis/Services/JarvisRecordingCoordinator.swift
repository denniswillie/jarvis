import Foundation

/// Owns recording commands for one control connection, never voice/camera I/O.
/// Slow attachment, fencing and retention run outside the event receive path.
/// Domain admission and a client report are not a server save receipt.
@MainActor
final class JarvisRecordingCoordinator<Completed: Sendable> {
    struct Context: Equatable, Sendable {
        let sessionID: UUID
        let audioSessionID: String
        let generationID: UUID
        let connectionID: UUID
        var protocolVersion: Int = 1
    }

    struct Capture: Sendable {
        let captureID: UUID
        let finish: @Sendable () async throws -> Completed
    }

    enum State: String, Sendable {
        case starting, recording, stopping, retaining, retained, captureFailed, retentionFailed
        case stoppedBeforeStart
    }

    enum Failure: Error { case retired, conflictingOperation, conflictingRecording, capacity, busy }
    enum Issue { case reportNotDelivered }
    typealias Command = JarvisRecordingControl.Command
    typealias Start = @MainActor (Command, @escaping @Sendable () -> Void) async throws -> Capture
    typealias Retain = @MainActor (Command, Completed) async throws -> Void

    private final class Run {
        let command: Command
        var state: State
        var stopRequested = false
        var sequence = 0
        var capture: Capture?
        var startTask: Task<Void, Never>?
        var finishTask: Task<Void, Never>?
        init(command: Command, state: State) { self.command = command; self.state = state }
    }

    let context: Context
    private let startCapture: Start
    private let retain: Retain
    private let write: @MainActor (JarvisRecordingControl.Outbound, Context) async throws -> Void
    private let current: @MainActor () -> Bool
    private let changed: @MainActor (UUID, State) -> Void
    private let issue: @MainActor (UUID, Issue) -> Void
    private var accepting = true
    private var operations: [UUID: Command] = [:]
    private var runs: [UUID: Run] = [:]
    private var reportTail: Task<Void, Never>?

    init(context: Context, current: @escaping @MainActor () -> Bool,
         start: @escaping Start, retain: @escaping Retain,
         write: @escaping @MainActor (JarvisRecordingControl.Outbound, Context) async throws -> Void,
         changed: @escaping @MainActor (UUID, State) -> Void = { _, _ in },
         issue: @escaping @MainActor (UUID, Issue) -> Void = { _, _ in }) {
        self.context = context; self.current = current; startCapture = start
        self.retain = retain; self.write = write; self.changed = changed; self.issue = issue
    }

    var isCapturing: Bool {
        runs.values.contains { [.starting, .recording, .stopping].contains($0.state) }
    }

    /// Decode and scope-check before admitting any effect. Operation replay is
    /// exact; stop-before-start tombstones cannot be revived by delayed delivery.
    func receive(_ command: Command) throws {
        guard accepting, current() else { throw Failure.retired }
        guard command.version <= context.protocolVersion else { throw Failure.conflictingOperation }
        try command.requireCurrent(sessionID: context.sessionID, audioSessionID: context.audioSessionID,
                                   generationID: context.generationID)
        if let previous = operations[command.operationID] {
            guard previous == command else { throw Failure.conflictingOperation }
            return
        }
        guard operations.count < 256 else { throw Failure.capacity }
        let existing = runs[command.recordingID]
        if let existing, !Self.sameRecording(existing.command, command) {
            throw Failure.conflictingRecording
        }
        if existing == nil {
            guard runs.count < 128 else { throw Failure.capacity }
            if command.action == .start && isCapturing { throw Failure.busy }
        }
        operations[command.operationID] = command
        if command.action == .stop {
            if let existing { requestStop(existing) }
            else {
                let run = Run(command: command, state: .stoppedBeforeStart)
                run.stopRequested = true; runs[command.recordingID] = run
                changed(command.recordingID, run.state)
                report(run, .stopped)
            }
            return
        }
        // A different call ID cannot restart a recording already known locally.
        guard existing == nil else { return }
        let run = Run(command: command, state: .starting)
        runs[command.recordingID] = run; changed(command.recordingID, run.state)
        run.startTask = Task { [self, run] in
            defer { run.startTask = nil }
            do {
                let handle = try await startCapture(command) { [weak self] in
                    Task { @MainActor in self?.sourceStopped(command.recordingID) }
                }
                run.capture = handle
                if run.stopRequested || !accepting || !current() { requestStop(run) }
                else { transition(run, .recording); report(run, .started) }
            } catch {
                transition(run, .captureFailed)
                report(run, .failed)
            }
        }
    }

    /// Explicit phone stop only. Merely interrupting speech never calls this.
    /// It fences capture; it does not assert domain cancellation or a saved file.
    func stopLocally(recordingID: UUID) {
        guard let run = runs[recordingID] else { return }
        requestStop(run)
    }

    /// Preserve already captured media on reconnect. Do not cancel a pending
    /// attach, discard its result, or send its reports on the replacement socket.
    func retire() {
        guard accepting else { return }
        accepting = false
        for run in runs.values { requestStop(run) }
    }

    /// Used for explicit lifecycle draining, not on the foreground receive path.
    func drained() async {
        for run in runs.values {
            await run.startTask?.value
            await run.finishTask?.value
        }
        await reportTail?.value
    }

    private func sourceStopped(_ id: UUID) {
        guard let run = runs[id] else { return }
        requestStop(run)
    }

    private func requestStop(_ run: Run) {
        guard [.starting, .recording, .stopping].contains(run.state) else { return }
        run.stopRequested = true
        if run.state != .stopping { transition(run, .stopping) }
        // Start completion retains responsibility for a late successful attach.
        guard let handle = run.capture, run.finishTask == nil else { return }
        run.finishTask = Task { [self, run] in
            defer { run.capture = nil; run.finishTask = nil }
            let completed: Completed
            do { completed = try await handle.finish() }
            catch { transition(run, .captureFailed); report(run, .failed); return }
            transition(run, .retaining)
            report(run, .stopped)
            do {
                try await retain(run.command, completed)
                transition(run, .retained)
            } catch {
                // A failure to retain/upload must not erase the original or
                // misreport that capture itself failed. Retain owns its bytes.
                transition(run, .retentionFailed)
            }
        }
    }

    private func transition(_ run: Run, _ state: State) {
        run.state = state; changed(run.command.recordingID, state)
    }

    private func report(_ run: Run, _ state: JarvisRecordingControl.ReportState) {
        guard accepting, current() else { return }
        run.sequence += 1
        let event = JarvisRecordingControl.Outbound.report(generationID: context.generationID,
            recordingID: run.command.recordingID, sequence: run.sequence, state: state)
        let prior = reportTail, id = run.command.recordingID
        reportTail = Task { [weak self] in
            await prior?.value
            guard let self, self.accepting, self.current() else { return }
            do { try await self.write(event, self.context) }
            catch { self.issue(id, .reportNotDelivered) }
        }
    }

    private static func sameRecording(_ lhs: Command, _ rhs: Command) -> Bool {
        lhs.recordingID == rhs.recordingID && lhs.sessionID == rhs.sessionID
            && lhs.generationID == rhs.generationID && lhs.audioSessionID == rhs.audioSessionID
            && lhs.includeJarvisAudio == rhs.includeJarvisAudio
            && lhs.includeMediaAudio == rhs.includeMediaAudio && lhs.recordingMode == rhs.recordingMode
            && lhs.maxBytes == rhs.maxBytes && lhs.maxDurationMilliseconds == rhs.maxDurationMilliseconds
    }
}
