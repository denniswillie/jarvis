import Foundation

enum JarvisVoiceRouteChoice: String, CaseIterable, Identifiable {
    case realtime, live
    static let preferenceKey = "jarvis.voiceRouteChoice"
    var id: String { rawValue }
    var label: String { self == .realtime ? "Realtime" : "GPT Live (default)" }
    static func load(from defaults: UserDefaults) -> Self {
        defaults.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .live
    }
    func save(to defaults: UserDefaults) { defaults.set(rawValue, forKey: Self.preferenceKey) }

    func audioURL(from rawURL: String, directAudio: Bool) throws -> URL {
        guard self != .live || directAudio else { throw JarvisVoiceRouteError.incompatibleDriver }
        let url = try JarvisAudioProvider.openai.audioWebSocketURL(from: rawURL, directAudio: directAudio)
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { throw JarvisAudioProviderError.invalidURL }
        var query = (components.queryItems ?? []).filter { $0.name != "voice_route" }
        if self == .live { query.append(URLQueryItem(name: "voice_route", value: "live")) }
        components.queryItems = query
        guard let result = components.url else { throw JarvisAudioProviderError.invalidURL }
        return result
    }
}

/// Exact app-control JSON. In particular, do not add the legacy client_event_id
/// to strict LiveMediaReceipt or route-control envelopes.
enum JarvisLiveAppWire {
    static func encode(_ value: [String: Any]) throws -> String {
        let allowed = ["audio.webrtc.offer", "voice_route.ready", "voice_route.event", "voice_route.interrupt", "live.media.receipt"]
        guard let type = value["type"] as? String, allowed.contains(type), value["client_event_id"] == nil else {
            throw JarvisVoiceRouteError.invalidWire
        }
        let data = try JSONSerialization.data(withJSONObject: value, options: .sortedKeys)
        guard data.count <= 131_072, let text = String(data: data, encoding: .utf8) else { throw JarvisVoiceRouteError.invalidWire }
        return text
    }
}

/// Production app owner, used by JarvisAudioSessionViewModel. One selected route
/// and authenticated socket own both media phases. Shared reminders/visual events
/// can pass through; legacy PCM/Realtime media cannot become a second speaker.
@MainActor
final class JarvisLiveAppRoute {
    typealias Writer = JarvisLiveMediaSignaling.Writer
    let driver: JarvisLiveVoiceRouteDriver
    private(set) var isReady = false
    private(set) var isClosed = false
    private(set) var recordingProtocolVersion: Int?
    var onRecordingCapabilityChanged: (() -> Void)?
    var onReady: ((String) -> Void)?
    var onCaption: ((String, String) -> Void)?
    var onFailure: ((String) -> Void)?
    var onStopped: ((String) -> Void)?
    var onProtocolFailure: ((JarvisLiveProtocolFailure) -> Void)?
    private var protocolFailureCount = 0
    private let write: Writer
    private let currentConnection: () -> UUID?
    private var signaling: JarvisLiveMediaSignaling!
    private var tasks: [UUID: Task<Void, Never>] = [:]
    private var startupTimeout: Task<Void, Never>?
    private var backendReady = false
    private var mediaReady = false
    private var fencing = false
    private var fenced = false
    private var started = false
    private var captureMode: JarvisAudioCaptureMode
    private var captions: [String: (id: UUID, revision: Int)] = [:]

    init(driver: JarvisLiveVoiceRouteDriver, captureMode: JarvisAudioCaptureMode,
         currentConnection: @escaping () -> UUID?, write: @escaping Writer) throws {
        self.driver = driver; self.captureMode = captureMode
        self.currentConnection = currentConnection; self.write = write
        try driver.setMicrophoneIntent(captureMode == .autoDetect)
        signaling = try JarvisLiveMediaSignaling(driver: driver) { [weak self] value, connection, current in
            guard let self, self.current, current() else { throw JarvisVoiceRouteError.notReady }
            try await self.write(value, connection, { [weak self] in self?.current == true && current() })
        }
        signaling.onReceiptWritten = { [weak self] phase in
            guard let self, self.current, phase == "activate" else { return }
            self.mediaReady = true
            self.finishReadiness()
        }
        signaling.onFailure = { [weak self] error in
            guard let self, !self.fencing else { return }
            self.reportProtocolFailure(stage: .mediaSignaling, field: .mediaOperation, error: error)
            self.fail("GPT Live media setup stopped. Start a new route; no fallback speaker was started.")
        }
    }

    private var current: Bool { !isClosed && !fenced && currentConnection() == driver.connectionID }
    var activeMediaForRecording: JarvisLiveMediaTransport? {
        guard current, isReady, recordingProtocolVersion == 1 || recordingProtocolVersion == 2 else { return nil }
        return driver.activeMediaForRecording
    }

    func start() async throws {
        guard !started, current else { throw JarvisVoiceRouteError.notReady }
        started = true
        startupTimeout = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(30)) } catch { return }
            guard let self, !self.isReady else { return }
            self.fail("GPT Live startup timed out. This backend may not have the Live route enabled.")
        }
        do {
            let offer = try await driver.prepareOffer()
            try Task.checkCancellation()
            guard current else { throw JarvisVoiceRouteError.notReady }
            // Live always owns turn detection. Hold-to-talk changes only the
            // local microphone gate, never backend input commits/create calls.
            try await write(["type": "audio.webrtc.offer", "sdp": offer, "capture_mode": "auto",
                             "voice_route": JarvisLiveVoiceRouteDriver.clientCapabilities], driver.connectionID,
                            { [weak self] in self?.current == true })
            guard current else { throw JarvisVoiceRouteError.notReady }
        } catch {
            reportProtocolFailure(stage: .startup, field: .offer, error: error)
            fail("GPT Live connection failed: \(error.localizedDescription)")
            throw error
        }
    }

    /// True means consumed. False preserves existing non-audio app handling.
    func receive(_ event: JarvisRealtimeServerEvent, connectionID: UUID) throws -> Bool {
        var stage = JarvisLiveProtocolFailure.Stage.admission
        var field = JarvisLiveProtocolFailure.Field.routeState
        do {
            return try receiveValidated(event, connectionID: connectionID, stage: &stage, field: &field)
        } catch {
            reportProtocolFailure(stage: stage, field: field, event: event, error: error)
            throw error
        }
    }

    private func receiveValidated(_ event: JarvisRealtimeServerEvent, connectionID: UUID,
                                  stage: inout JarvisLiveProtocolFailure.Stage,
                                  field: inout JarvisLiveProtocolFailure.Field) throws -> Bool {
        guard !isClosed else { throw JarvisVoiceRouteError.identityMismatch }
        field = .connectionID
        guard connectionID == driver.connectionID, currentConnection() == connectionID else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        switch event.type {
        case "voice_route.selected":
            stage = .selection; field = .selection
            guard !fenced else { throw JarvisVoiceRouteError.notReady }
            let message = try driver.select(event)
            send(message) { [weak self] in
                guard let self, self.current else { return }
                try self.driver.didWriteReady(connectionID: connectionID)
            }
        case "live.media.prepare", "live.media.activate":
            stage = .mediaSignaling; field = .mediaCommand
            guard current else { throw JarvisVoiceRouteError.notReady }
            try signaling.receive(event, connectionID: connectionID)
        case "voice_route.interrupt", "voice_route.close":
            stage = .routeControl; field = .controlCommand
            fencing = true
            defer { fencing = false }
            let receipt = try driver.handleCommand(event)
            fenced = true; isReady = false; recordingProtocolVersion = nil; startupTimeout?.cancel()
            send(receipt, allowFenced: true) { [weak self] in
                self?.onStopped?("GPT Live output is fenced. Restart to speak again; background work is retained.")
            }
        case "voice_route.event":
            stage = .envelope; field = .selection
            guard let selection = driver.selection else { throw JarvisVoiceRouteError.identityMismatch }
            field = .audioSessionID
            guard event.audioSessionId == selection.audioSessionID else { throw JarvisVoiceRouteError.identityMismatch }
            field = .protocolVersion
            guard try JarvisVoiceWire.integer(event.values["protocol_version"]) == 1 else { throw JarvisVoiceRouteError.identityMismatch }
            field = .envelope
            try JarvisVoiceWire.keys(event.values, required: ["type", "protocol_version", "audio_session_id", "payload"])
            stage = .payload; field = .payload
            let payload = try JarvisVoiceWire.object(event.values["payload"])
            field = .route
            guard try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(payload["route"])) == selection.route else {
                throw JarvisVoiceRouteError.identityMismatch
            }
            field = .observedAt
            try JarvisVoiceWire.timestamp(payload["observed_at"])
            switch payload["kind"] as? String {
            case "connection_state":
                switch payload["state"] as? String {
                case "ready":
                    field = .routeState
                    guard current else { throw JarvisVoiceRouteError.notReady }
                    backendReady = true; finishReadiness()
                case "failed", "closed": fail("GPT Live route ended. Restart or explicitly choose Realtime.")
                default: break
                }
            case "transcript_revision":
                stage = .transcript; field = .routeState
                guard current else { throw JarvisVoiceRouteError.invalidWire }
                field = .role
                guard let role = payload["role"] as? String, ["user", "assistant"].contains(role) else {
                    throw JarvisVoiceRouteError.invalidWire
                }
                field = .text
                guard let text = payload["text"] as? String, text.count <= 4096 else { throw JarvisVoiceRouteError.invalidWire }
                field = .transcriptID
                let id = try JarvisVoiceWire.uuid(payload["transcript_id"])
                field = .revision
                let revision = try JarvisVoiceWire.integer(payload["revision"])
                // The sideband may publish a valid provisional caption before
                // media activation. Validate and consume it, without display,
                // buffering, dedup mutation, or permission to open a speaker.
                guard isReady else { return true }
                if let previous = captions[role], previous.id == id, previous.revision >= revision { return true }
                captions[role] = (id, revision)
                // Display only. No canonical/final utterance or playback identity
                // is invented from the provider's provisional caption fragments.
                onCaption?(role, text)
            case "route_error": fail("GPT Live reported a route error. Restart to recover.")
            default: break // Context/command ACKs never reopen native playback.
            }
        case "voice_route.output_bound":
            stage = .selection; field = .selection
            throw JarvisVoiceRouteError.incompatibleDriver // Attribution unsupported.
        case "audio.provider.selected":
            stage = .providerAcknowledgement; field = .providerAcknowledgement
            let acknowledgement = try JarvisAudioProviderAcknowledgement(event: event)
            guard acknowledgement.provider == .openai,
                  acknowledgement.model?.hasPrefix("gpt-live-") == true,
                  driver.selection == nil || acknowledgement.audioSessionID == driver.selection?.audioSessionID else {
                throw JarvisVoiceRouteError.incompatibleDriver
            }
            return false
        case "session.created":
            // Backend emits this only after binding real recording dispatch.
            // This capability is not media readiness and never opens playback.
            guard current, let selection = driver.selection,
                  event.audioSessionId == selection.audioSessionID,
                  event.sessionId.flatMap(UUID.init(uuidString:)) == driver.sessionID else { return true }
            let version = try? JarvisVoiceWire.integer(event.values["recording_protocol_version"])
            recordingProtocolVersion = (version == 1 || version == 2) ? version : nil
            onRecordingCapabilityChanged?()
            return true
        default:
            if event.type.hasPrefix("audio.") || event.type.hasPrefix("response.")
                || event.type.hasPrefix("transcript.") || event.type == "user.transcript.done" { return true }
            return false
        }
        return true
    }

    func setCaptureMode(_ mode: JarvisAudioCaptureMode) throws {
        guard current else { throw JarvisVoiceRouteError.notReady }
        captureMode = mode
        try driver.setMicrophoneIntent(mode == .autoDetect)
    }
    func setManualRecording(_ recording: Bool) throws {
        guard current, isReady, captureMode == .holdToTalk else { throw JarvisVoiceRouteError.notReady }
        try driver.setMicrophoneIntent(recording)
    }

    func interruptLocally() throws {
        guard current else { return }
        fencing = true
        defer { fencing = false }
        let command = try driver.interruptLocally()
        fenced = true; isReady = false; recordingProtocolVersion = nil; startupTimeout?.cancel()
        if let command {
            send(command, allowFenced: true) { [weak self] in
                self?.onStopped?("GPT Live was interrupted locally. Restart for a fresh audio route.")
            }
        }
    }

    @discardableResult func close() -> Bool {
        isClosed = true; isReady = false; recordingProtocolVersion = nil
        startupTimeout?.cancel(); startupTimeout = nil
        for task in tasks.values { task.cancel() }
        tasks.removeAll()
        fencing = true
        signaling.close()
        let confirmed = driver.close()
        fencing = false
        return confirmed
    }

    private func finishReadiness() {
        guard current, !isReady, mediaReady, backendReady, driver.state == .active,
              let selection = driver.selection else { return }
        isReady = true
        startupTimeout?.cancel(); startupTimeout = nil
        onReady?(selection.audioSessionID)
    }

    private func send(_ payload: [String: Any], allowFenced: Bool = false,
                      afterWrite: @escaping @MainActor () throws -> Void = {}) {
        guard tasks.count < 16 else { fail("Live control queue capacity reached."); return }
        let id = UUID()
        tasks[id] = Task { [weak self] in
            guard let self else { return }
            defer { self.tasks.removeValue(forKey: id) }
            let eligible = { [weak self] in
                guard let self else { return false }
                return !self.isClosed && (allowFenced || !self.fenced) && self.currentConnection() == self.driver.connectionID
            }
            do {
                try Task.checkCancellation()
                guard eligible() else { return }
                try await self.write(payload, self.driver.connectionID, eligible)
                try Task.checkCancellation()
                guard eligible() else { return }
                try afterWrite()
            } catch {
                if !Task.isCancelled {
                    self.reportProtocolFailure(stage: .controlWrite, field: .controlWrite, error: error)
                    self.fail("Live control write failed: \(error.localizedDescription)")
                }
            }
        }
    }

    private func reportProtocolFailure(stage: JarvisLiveProtocolFailure.Stage,
                                       field: JarvisLiveProtocolFailure.Field,
                                       event: JarvisRealtimeServerEvent? = nil, error: Error) {
        guard protocolFailureCount < 16 else { return }
        protocolFailureCount += 1
        onProtocolFailure?(JarvisLiveProtocolFailure(stage: stage, field: field, event: event, error: error))
    }

    private func fail(_ message: String) {
        guard !isClosed else { return }
        close()
        onFailure?(message)
    }
}
