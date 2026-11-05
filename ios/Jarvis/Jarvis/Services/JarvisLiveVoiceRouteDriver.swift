import Foundation

/// Media only: no task policy, provider credentials, PCM forwarding, input
/// commits or response creation. Each instance is single-use, even after failure.
@MainActor
protocol JarvisLiveMediaTransport: AnyObject {
    var hasRemoteDescription: Bool { get }
    var isConnected: Bool { get }
    var outputEnabled: Bool { get }
    var onConnectivityChanged: (() -> Void)? { get set }
    var onFailure: (() -> Void)? { get set }
    var onProviderStarted: ((JarvisRealtimeServerEvent) -> Void)? { get set }
    func prepareOffer() async throws -> String
    func applyAnswer(_ sdp: String) async throws
    func activate(microphoneEnabled: Bool) throws
    func setMicrophoneEnabled(_ enabled: Bool)
    /// Permanently fences queued output and destroys the peer. Never reopens it.
    func fenceAndClose() -> Bool
}

/// Shared among Live drivers. The host must also synchronously fence its previous
/// non-Live media owner before claiming this lease. This does not replace the
/// application's existing route handover coordinator.
@MainActor
final class JarvisLiveMediaOwnership {
    static let shared = JarvisLiveMediaOwnership()
    private var owner: UUID?

    func claim(_ token: UUID, fencePrevious: () -> Bool) throws {
        guard owner == nil else { throw JarvisVoiceRouteError.alreadySelected }
        guard fencePrevious() else { throw JarvisVoiceRouteError.outputNotFenced }
        owner = token
    }

    func release(_ token: UUID) {
        if owner == token { owner = nil }
    }
}

/// Provider-specific lifecycle over the frozen v1 route-control wire. The host
/// maps authenticated backend media/activation messages to the typed methods
/// below; no speculative new wire namespace is introduced here.
@MainActor
final class JarvisLiveVoiceRouteDriver {
    enum State: Equatable { case idle, preparing, prepared, applyingAnswer, awaitingActivation, active, fenced, closed, failed }

    let sessionID: UUID
    let connectionID: UUID
    private(set) var state: State = .idle
    private(set) var selection: JarvisVoiceRouteSelection?
    private(set) var providerSessionID: String?
    var onStateChanged: (() -> Void)?

    static var clientCapabilities: [String: Any] {
        var capabilities = JarvisVoiceMediaKind.nativeWebRTC.clientCapabilities
        capabilities["input_codecs"] = ["webrtc_negotiated"]
        capabilities["output_codecs"] = ["webrtc_negotiated"]
        return capabilities
    }
    var requiresReplacement: Bool { [.fenced, .closed, .failed].contains(state) }
    var canConfirmPreparation: Bool {
        state == .awaitingActivation && readyWritten && providerStarted
            && media.hasRemoteDescription && !media.outputEnabled
    }
    var canActivate: Bool {
        canConfirmPreparation && media.isConnected
    }
    // This means only an open local gate, not audible output or user hearing.
    var outputEnabled: Bool { state == .active && media.outputEnabled }
    /// Borrowed only while this selected route owns active connected media.
    /// The concrete native adapter exposes its existing duplex device; this
    /// protocol-level gate creates no microphone or second audio owner.
    var activeMediaForRecording: JarvisLiveMediaTransport? {
        guard ownsLease, state == .active, selection != nil,
              media.outputEnabled, media.isConnected else { return nil }
        return media
    }

    private let media: JarvisLiveMediaTransport
    private let ownership: JarvisLiveMediaOwnership
    private let lease = UUID()
    private let fencePrevious: () -> Bool
    private var ownsLease = false
    private var readyWritten = false
    private var providerStarted = false
    private var microphoneIntent = true
    private var activationID: UUID?
    private var activationReceipt: [String: Any]?
    private var localInterrupt: [String: Any]?
    private var receipts: [UUID: [String: Any]] = [:]
    private var fingerprints: [UUID: Data] = [:]

    init(sessionID: UUID, connectionID: UUID, media: JarvisLiveMediaTransport,
         ownership: JarvisLiveMediaOwnership, fencePrevious: @escaping () -> Bool) {
        self.sessionID = sessionID
        self.connectionID = connectionID
        self.media = media
        self.ownership = ownership
        self.fencePrevious = fencePrevious
        media.onConnectivityChanged = { [weak self] in self?.onStateChanged?() }
        media.onFailure = { [weak self] in
            guard let self, !self.requiresReplacement else { return }
            self.retire(as: .failed)
        }
        media.onProviderStarted = { [weak self] event in
            guard let self, !self.requiresReplacement, let selection = self.selection else { return }
            do {
                try self.observeProviderStarted(event, route: selection.route,
                                                audioSessionID: selection.audioSessionID,
                                                connectionID: self.connectionID)
            } catch { self.retire(as: .failed) }
        }
    }

    func prepareOffer() async throws -> String {
        guard state == .idle else { throw JarvisVoiceRouteError.alreadySelected }
        try ownership.claim(lease, fencePrevious: fencePrevious)
        ownsLease = true
        transition(.preparing)
        do {
            let offer = try await media.prepareOffer()
            try Task.checkCancellation()
            guard state == .preparing else { throw CancellationError() }
            transition(.prepared)
            return offer
        } catch {
            if !requiresReplacement { retire(as: .failed) }
            throw error
        }
    }

    func select(_ event: JarvisRealtimeServerEvent) throws -> [String: Any] {
        guard !requiresReplacement, state != .idle, state != .preparing else { throw JarvisVoiceRouteError.notReady }
        let proposed = try JarvisVoiceRouteSelection(event: event, mediaKind: .nativeWebRTC, profile: .liveWebRTC)
        guard proposed.route.sessionID == sessionID else { throw JarvisVoiceRouteError.identityMismatch }
        let capabilities = try JarvisVoiceWire.object(event.values["capabilities"])
        for key in ["explicit_input_commit", "explicit_response_create", "image_input",
                    "notification_output_attribution", "client_playback_evidence"] {
            guard try !JarvisVoiceWire.support(capabilities[key]) else { throw JarvisVoiceRouteError.incompatibleDriver }
        }
        guard try JarvisVoiceWire.support(capabilities["full_duplex"]) else { throw JarvisVoiceRouteError.incompatibleDriver }
        if let selection {
            guard selection == proposed else { throw JarvisVoiceRouteError.alreadySelected }
        } else {
            guard !media.outputEnabled else { throw JarvisVoiceRouteError.outputNotFenced }
            selection = proposed
        }
        return proposed.envelope(type: "voice_route.ready")
    }

    func didWriteReady(connectionID: UUID) throws {
        guard self.connectionID == connectionID, selection != nil, !requiresReplacement else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        readyWritten = true
        onStateChanged?()
    }

    func applyAnswer(_ sdp: String, providerSessionID: String, route: JarvisVoiceRouteIdentity,
                     audioSessionID: String, connectionID: UUID) async throws {
        try validate(route, audioSessionID: audioSessionID, connectionID: connectionID)
        guard state == .prepared else { throw JarvisVoiceRouteError.notReady }
        self.providerSessionID = try JarvisVoiceWire.string(providerSessionID)
        transition(.applyingAnswer)
        do {
            try await media.applyAnswer(sdp)
            try Task.checkCancellation()
            guard state == .applyingAnswer else { throw CancellationError() }
            transition(.awaitingActivation)
        } catch {
            if !requiresReplacement { retire(as: .failed) }
            throw error
        }
    }

    /// Called by the owned native data channel. An authenticated backend may also
    /// forward actual evidence through this seam, but no such forwarder is assumed.
    /// Session creation, SDP success, context ACKs and generic
    /// Jarvis session.ready are deliberately not accepted as this evidence.
    func observeProviderStarted(_ event: JarvisRealtimeServerEvent, route: JarvisVoiceRouteIdentity,
                                audioSessionID: String, connectionID: UUID) throws {
        try validate(route, audioSessionID: audioSessionID, connectionID: connectionID)
        guard !requiresReplacement, let providerSessionID,
              event.type == "session.started", event.values["type"] as? String == "session.started",
              let session = event.values["session"] as? [String: Any],
              session["id"] as? String == providerSessionID else { throw JarvisVoiceRouteError.identityMismatch }
        providerStarted = true
        onStateChanged?()
    }

    /// Explicit backend initialization completion, NOT a context/instruction ACK.
    /// Returns the existing CommandReceipt wire shape only after real media setup.
    /// Integration must retry a notReady activation after local ICE readiness,
    /// using the same command ID, rather than claiming success from a fake peer.
    func activate(commandID: UUID, route: JarvisVoiceRouteIdentity, audioSessionID: String,
                  providerSessionID: String, connectionID: UUID, now: Date = Date()) throws -> [String: Any] {
        try validate(route, audioSessionID: audioSessionID, connectionID: connectionID)
        guard !requiresReplacement, self.providerSessionID == providerSessionID else { throw JarvisVoiceRouteError.identityMismatch }
        if let activationID {
            guard activationID == commandID, state == .active, let activationReceipt else { throw JarvisVoiceRouteError.identityMismatch }
            return activationReceipt
        }
        guard canActivate, receipts[commandID] == nil else { throw JarvisVoiceRouteError.notReady }
        do {
            try media.activate(microphoneEnabled: microphoneIntent)
            guard media.outputEnabled else { throw JarvisVoiceRouteError.notReady }
        } catch {
            retire(as: .failed)
            throw error
        }
        activationID = commandID
        let value = receipt(commandID: commandID, operation: "start", fenced: false, completed: true, now: now)
        activationReceipt = value
        transition(.active)
        return value
    }

    func setMicrophoneEnabled(_ enabled: Bool) throws {
        guard state == .active else { throw JarvisVoiceRouteError.notReady }
        microphoneIntent = enabled
        media.setMicrophoneEnabled(enabled)
    }

    /// App capture preference may arrive during startup. It cannot open an
    /// unactivated or fenced peer; activation applies this intent atomically.
    func setMicrophoneIntent(_ enabled: Bool) throws {
        guard !requiresReplacement else { throw JarvisVoiceRouteError.notReady }
        microphoneIntent = enabled
        if state == .active { media.setMicrophoneEnabled(enabled) }
    }

    func handleCommand(_ event: JarvisRealtimeServerEvent, now: Date = Date()) throws -> [String: Any] {
        guard let selection, event.audioSessionId == selection.audioSessionID,
              try JarvisVoiceWire.integer(event.values["protocol_version"]) == 1 else { throw JarvisVoiceRouteError.identityMismatch }
        let closing = event.type == "voice_route.close"
        guard closing || event.type == "voice_route.interrupt", event.values["type"] as? String == event.type else {
            throw JarvisVoiceRouteError.invalidWire
        }
        try JarvisVoiceWire.keys(event.values, required: ["type", "protocol_version", "audio_session_id", "payload"])
        let payload = try JarvisVoiceWire.object(event.values["payload"])
        try JarvisVoiceWire.keys(payload, required: ["route", "command_id", "requested_at", "reason"],
                                optional: closing ? [] : ["attempt", "output"])
        guard try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(payload["route"])) == selection.route else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        let id = try JarvisVoiceWire.uuid(payload["command_id"])
        try JarvisVoiceWire.timestamp(payload["requested_at"])
        _ = try JarvisVoiceWire.string(payload["reason"], limit: 512)
        guard id != activationID else { throw JarvisVoiceRouteError.identityMismatch }
        let fingerprint = try JSONSerialization.data(withJSONObject: event.values, options: [.sortedKeys])
        if let previous = receipts[id] {
            guard fingerprints[id] == fingerprint else { throw JarvisVoiceRouteError.identityMismatch }
            return previous
        }
        // Even capacity exhaustion must fence first, never leave audio running.
        let fenced = retire(as: closing ? .closed : .fenced)
        guard receipts.count < 128 else { throw JarvisVoiceRouteError.capacity }
        let value = receipt(commandID: id, operation: closing ? "close" : "interrupt_speech",
                            fenced: fenced, completed: fenced, now: now)
        receipts[id] = value
        fingerprints[id] = fingerprint
        return value
    }

    /// No awaiting networking. Receipt/context delivery cannot reopen this peer.
    func interruptLocally(now: Date = Date()) throws -> [String: Any]? {
        if let localInterrupt { return localInterrupt }
        guard retire(as: .fenced) else { throw JarvisVoiceRouteError.outputNotFenced }
        guard let selection else { return nil }
        var value = selection.envelope(type: "voice_route.interrupt")
        value.removeValue(forKey: "route")
        value["payload"] = ["route": selection.route.wire, "command_id": UUID().uuidString.lowercased(),
                            "requested_at": now.jarvisISO8601String(), "reason": "local_user_interruption"]
        localInterrupt = value
        return value
    }

    @discardableResult func close() -> Bool { retire(as: .closed) }

    private func validate(_ route: JarvisVoiceRouteIdentity, audioSessionID: String, connectionID: UUID) throws {
        guard self.connectionID == connectionID, let selection,
              selection.route == route, selection.audioSessionID == audioSessionID else { throw JarvisVoiceRouteError.identityMismatch }
    }

    @discardableResult private func retire(as finalState: State) -> Bool {
        // Invalidate in-flight operations BEFORE closing media or notifying host.
        state = finalState
        let fenced = media.fenceAndClose()
        if fenced, ownsLease { ownership.release(lease); ownsLease = false }
        if !fenced { state = .failed }
        onStateChanged?()
        return fenced
    }

    private func transition(_ value: State) { state = value; onStateChanged?() }

    private func receipt(commandID: UUID, operation: String, fenced: Bool, completed: Bool, now: Date) -> [String: Any] {
        guard let selection else { preconditionFailure("Receipt requires selected route") }
        var value = selection.envelope(type: "voice_route.event")
        value.removeValue(forKey: "route")
        value["payload"] = ["kind": "command_receipt", "route": selection.route.wire,
                            "command_id": commandID.uuidString.lowercased(), "operation": operation,
                            "observed_at": now.jarvisISO8601String(), "status": completed ? "completed" : "unknown",
                            "output_fenced": fenced, "provider_stopped": NSNull()]
        return value
    }
}
