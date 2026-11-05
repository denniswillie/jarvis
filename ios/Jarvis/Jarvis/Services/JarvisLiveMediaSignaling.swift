import Foundation

/// Exact LiveMediaCommand.payload wire, after app authentication. No credentials
/// or provider commands are accepted here. One instance serves one connection.
private struct JarvisLiveMediaCommand: Equatable {
    enum Phase: String { case prepare, activate }
    let phase: Phase
    let route: JarvisVoiceRouteIdentity
    let audioSessionID: String
    let commandID: UUID
    let providerSessionID: String
    let answer: String?

    init(_ event: JarvisRealtimeServerEvent) throws {
        switch event.type {
        case "live.media.prepare": phase = .prepare
        case "live.media.activate": phase = .activate
        default: throw JarvisVoiceRouteError.invalidWire
        }
        let required: Set<String> = ["type", "protocol_version", "route", "audio_session_id", "command_id", "provider_session_id"]
        try JarvisVoiceWire.keys(event.values, required: phase == .prepare ? required.union(["answer_sdp"]) : required)
        guard event.values["type"] as? String == event.type,
              try JarvisVoiceWire.integer(event.values["protocol_version"]) == 1 else { throw JarvisVoiceRouteError.invalidWire }
        route = try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(event.values["route"]))
        audioSessionID = try JarvisVoiceWire.string(event.values["audio_session_id"])
        commandID = try JarvisVoiceWire.uuid(event.values["command_id"])
        providerSessionID = try JarvisVoiceWire.string(event.values["provider_session_id"])
        if phase == .prepare {
            let sdp = try JarvisVoiceWire.string(event.values["answer_sdp"], limit: 65_536)
            guard sdp.hasPrefix("v=0"), sdp.utf8.count <= 65_536 else { throw JarvisVoiceRouteError.invalidWire }
            answer = sdp
        } else { answer = nil }
    }

    func sameStartup(as other: Self) -> Bool {
        route == other.route && audioSessionID == other.audioSessionID
            && commandID == other.commandID && providerSessionID == other.providerSessionID
    }

    func receipt() -> [String: Any] {
        ["type": "live.media.receipt", "protocol_version": 1, "route": route.wire,
         "audio_session_id": audioSessionID, "command_id": commandID.uuidString.lowercased(),
         "provider_session_id": providerSessionID, "phase": phase.rawValue,
         "receipt_id": UUID().uuidString.lowercased(), "provider_started": true,
         "output_gated": phase == .prepare]
    }
}

/// Nonblocking app-signaling receiver. Media work, receipts and the startup
/// watchdog are owned tasks, not waits inside the authenticated socket reader.
/// The sender MUST check `current` immediately at its actual write boundary.
@MainActor
final class JarvisLiveMediaSignaling {
    typealias Writer = @MainActor ([String: Any], UUID, @escaping @MainActor () -> Bool) async throws -> Void
    enum State: Equatable { case idle, preparing, prepared, activating, active, failed, closed }
    enum Failure: Error { case deadline, retired, transport }
    private(set) var state: State = .idle
    var onFailure: ((Error) -> Void)?
    var onReceiptWritten: ((String) -> Void)?

    private let driver: JarvisLiveVoiceRouteDriver
    private let write: Writer
    private let timeout: TimeInterval
    private let clock: () -> TimeInterval
    private var deadline: TimeInterval?
    private var prepareCommand: JarvisLiveMediaCommand?
    private var activateCommand: JarvisLiveMediaCommand?
    private var prepareReceipt: [String: Any]?
    private var activateReceipt: [String: Any]?
    private var prepareWritten = false
    private var operation: Task<Void, Never>?
    private var receiptWrite: Task<Void, Never>?
    private var watchdog: Task<Void, Never>?
    private var advancing = false
    private var terminal: Bool { state == .failed || state == .closed }

    init(driver: JarvisLiveVoiceRouteDriver, timeout: TimeInterval = 20,
         clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
         write: @escaping Writer) throws {
        guard timeout.isFinite, timeout > 0, timeout <= 120 else { throw JarvisVoiceRouteError.invalidWire }
        self.driver = driver; self.timeout = timeout; self.clock = clock; self.write = write
        let previous = driver.onStateChanged
        driver.onStateChanged = { [weak self] in previous?(); self?.advance() }
    }

    /// Concrete writer over the host's EXISTING authenticated app socket. This
    /// does not create another connection or change the Realtime client's API.
    convenience init(driver: JarvisLiveVoiceRouteDriver, socket: URLSessionWebSocketTask,
                     currentConnection: @escaping @MainActor () -> UUID?, timeout: TimeInterval = 20) throws {
        try self.init(driver: driver, timeout: timeout) { payload, connection, current in
            try Task.checkCancellation()
            guard currentConnection() == connection, current() else { throw Failure.retired }
            let data = try JSONSerialization.data(withJSONObject: payload, options: .sortedKeys)
            guard let text = String(data: data, encoding: .utf8) else { throw JarvisVoiceRouteError.invalidWire }
            try await socket.send(.string(text))
            try Task.checkCancellation()
            guard currentConnection() == connection, current() else { throw Failure.retired }
        }
    }

    /// Returns immediately. Caller supplies the actual authenticated connection
    /// token, not a connection ID parsed from untrusted JSON.
    func receive(_ event: JarvisRealtimeServerEvent, connectionID: UUID) throws {
        checkDeadline()
        guard !terminal, connectionID == driver.connectionID, let selection = driver.selection else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        let command = try JarvisLiveMediaCommand(event)
        guard command.route == selection.route, command.audioSessionID == selection.audioSessionID else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        guard !driver.requiresReplacement else { fail(Failure.retired); throw Failure.retired }
        switch command.phase {
        case .prepare:
            if let prepareCommand {
                guard command == prepareCommand else { throw JarvisVoiceRouteError.identityMismatch }
                guard activateReceipt == nil else { throw JarvisVoiceRouteError.notReady }
                if let prepareReceipt { send(prepareReceipt, for: command) }
                return
            }
            guard state == .idle, driver.state == .prepared, let answer = command.answer else { throw JarvisVoiceRouteError.notReady }
            prepareCommand = command
            state = .preparing
            deadline = clock() + timeout
            watchdog = Task { [weak self, timeout] in
                do { try await Task.sleep(for: .seconds(timeout)) } catch { return }
                self?.checkDeadline()
            }
            operation = Task { [weak self, driver] in
                do {
                    try await driver.applyAnswer(answer, providerSessionID: command.providerSessionID,
                                                 route: command.route, audioSessionID: command.audioSessionID,
                                                 connectionID: connectionID)
                    try Task.checkCancellation()
                    guard let self, !self.terminal else { return }
                    self.operation = nil
                    self.advance()
                } catch { self?.fail(error) }
            }
        case .activate:
            guard let prepareCommand, command.sameStartup(as: prepareCommand) else { throw JarvisVoiceRouteError.identityMismatch }
            if let activateCommand {
                guard command == activateCommand else { throw JarvisVoiceRouteError.identityMismatch }
                if let activateReceipt { send(activateReceipt, for: command) }
                return
            }
            activateCommand = command
            advance()
        }
    }

    /// Watchdog entry is also injectable through the monotonic clock in native
    /// tests; wall-clock changes cannot extend startup in production.
    func checkDeadline() {
        guard !terminal, state != .active, let deadline, clock() >= deadline else { return }
        fail(Failure.deadline)
    }

    /// Synchronous local fence first. Cancellation never claims provider stop.
    func close() {
        guard !terminal else { return }
        state = .closed
        cancelOwnedTasks()
        driver.close()
    }

    private func advance() {
        guard !terminal, !advancing else { return }
        advancing = true
        defer { advancing = false }
        checkDeadline()
        guard !terminal else { return }
        guard !driver.requiresReplacement else { fail(Failure.retired); return }
        if prepareReceipt == nil, let prepareCommand, driver.canConfirmPreparation {
            let receipt = prepareCommand.receipt()
            prepareReceipt = receipt
            send(receipt, for: prepareCommand)
        }
        if prepareWritten, receiptWrite == nil, activateReceipt == nil,
           let activateCommand, driver.canActivate {
            do {
                // This existing start CommandReceipt stays internal. The wire
                // receives the separate strict LiveMediaReceipt below.
                _ = try driver.activate(commandID: activateCommand.commandID, route: activateCommand.route,
                                        audioSessionID: activateCommand.audioSessionID,
                                        providerSessionID: activateCommand.providerSessionID,
                                        connectionID: driver.connectionID)
                state = .activating
                let receipt = activateCommand.receipt()
                activateReceipt = receipt
                send(receipt, for: activateCommand)
            } catch { fail(error) }
        }
    }

    private func isCurrent(_ command: JarvisLiveMediaCommand) -> Bool {
        guard !terminal, !driver.requiresReplacement, let selection = driver.selection,
              selection.route == command.route, selection.audioSessionID == command.audioSessionID,
              driver.providerSessionID == command.providerSessionID else { return false }
        if state != .active, let deadline, clock() >= deadline { return false }
        switch command.phase {
        case .prepare: return activateReceipt == nil && driver.canConfirmPreparation
        case .activate: return driver.state == .active && driver.outputEnabled
        }
    }

    private func send(_ payload: [String: Any], for command: JarvisLiveMediaCommand) {
        guard !terminal, receiptWrite == nil, isCurrent(command) else { return }
        receiptWrite = Task { [weak self] in
            guard let self else { return }
            do {
                try Task.checkCancellation()
                guard self.isCurrent(command) else { throw Failure.retired }
                try await self.write(payload, self.driver.connectionID, { [weak self] in self?.isCurrent(command) == true })
                try Task.checkCancellation()
                guard self.isCurrent(command) else { throw Failure.retired }
                self.receiptWrite = nil
                switch command.phase {
                case .prepare: self.prepareWritten = true; self.state = .prepared
                case .activate: self.state = .active; self.watchdog?.cancel(); self.watchdog = nil
                }
                self.onReceiptWritten?(command.phase.rawValue)
                self.advance()
            } catch { self.fail(error) }
        }
    }

    private func cancelOwnedTasks() {
        watchdog?.cancel(); watchdog = nil
        operation?.cancel(); operation = nil
        receiptWrite?.cancel(); receiptWrite = nil
    }

    private func fail(_ error: Error) {
        guard !terminal else { return }
        state = .failed
        cancelOwnedTasks()
        driver.close()
        onFailure?(error)
    }
}
