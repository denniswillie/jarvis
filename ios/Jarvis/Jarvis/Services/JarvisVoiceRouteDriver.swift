import Foundation

/// Presentation only. Never retain provider text or use this as a media gate.
enum JarvisVoiceRouteFailure: String, Equatable {
    case nonrecoverableRouteError = "nonrecoverable_route_error"
    case connectionFailed = "connection_failed"

    var message: String {
        "Voice is unavailable for this connection. Stop and start a new voice session."
    }

    static func displayText(_ failure: Self?, transcript: String) -> String {
        failure?.message ?? transcript
    }

    static func evidence(_ failure: Self?) -> [String: Any] {
        ["schema_version": 1, "status": failure == nil ? "not_reported" : "unavailable",
         "failure_source": failure.map { $0.rawValue } as Any? ?? NSNull()]
    }

    /// Identity is checked by the driver first. Invalid/unknown status payloads
    /// remain ignored, rather than introducing a new protocol teardown policy.
    static func reported(in payload: [String: Any]) -> Self? {
        do {
            switch payload["kind"] as? String {
            case "route_error":
                try JarvisVoiceWire.keys(payload, required: ["kind", "route", "observed_at", "code", "message", "recoverable"])
                try JarvisVoiceWire.timestamp(payload["observed_at"])
                _ = try JarvisVoiceWire.string(payload["code"])
                _ = try JarvisVoiceWire.string(payload["message"], limit: 512)
                guard let recoverable = payload["recoverable"] as? NSNumber,
                      CFGetTypeID(recoverable) == CFBooleanGetTypeID(), !recoverable.boolValue else { return nil }
                return .nonrecoverableRouteError
            case "connection_state":
                try JarvisVoiceWire.keys(payload, required: ["kind", "route", "observed_at", "state"], optional: ["reason"])
                try JarvisVoiceWire.timestamp(payload["observed_at"])
                if let reason = payload["reason"], !(reason is NSNull) {
                    _ = try JarvisVoiceWire.string(reason, limit: 512)
                }
                return payload["state"] as? String == "failed" ? .connectionFailed : nil
            default:
                return nil
            }
        } catch { return nil }
    }
}

/// One connection owns one local media driver. Closures synchronously fence the
/// real media implementation; a timeout or backend acknowledgment is not a fence.
@MainActor
final class JarvisVoiceRouteDriver {
    let sessionID: UUID
    let connectionID: UUID
    let mediaKind: JarvisVoiceMediaKind
    private(set) var selection: JarvisVoiceRouteSelection?
    private(set) var isReady = false
    private(set) var isClosed = false
    private(set) var failure: JarvisVoiceRouteFailure?
    private var readyWritten = false
    private var observedAudioSessionID: String?
    private let fenceMedia: (_ close: Bool) -> Bool
    private var bindings: [JarvisVoiceOutputBinding] = []
    private var playedSamples: [UUID: Int] = [:]
    private var retiredOutputs = Set<UUID>()
    private var receipts: [UUID: [String: Any]] = [:]
    private var commandFingerprints: [UUID: Data] = [:]
    private var pendingInterrupt: UUID?

    init(sessionID: UUID, connectionID: UUID, mediaKind: JarvisVoiceMediaKind,
         fenceMedia: @escaping (_ close: Bool) -> Bool) {
        self.sessionID = sessionID
        self.connectionID = connectionID
        self.mediaKind = mediaKind
        self.fenceMedia = fenceMedia
    }

    func select(_ event: JarvisRealtimeServerEvent) throws -> [String: Any] {
        guard !isClosed else { throw JarvisVoiceRouteError.notReady }
        let proposed = try JarvisVoiceRouteSelection(event: event, mediaKind: mediaKind)
        guard proposed.route.sessionID == sessionID,
              observedAudioSessionID == nil || observedAudioSessionID == proposed.audioSessionID else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        if let selection {
            guard selection == proposed else { throw JarvisVoiceRouteError.alreadySelected }
        } else {
            guard fenceMedia(false) else { throw JarvisVoiceRouteError.outputNotFenced }
            selection = proposed
        }
        return proposed.envelope(type: "voice_route.ready")
    }

    func didWriteReady(connectionID: UUID) throws {
        guard self.connectionID == connectionID, selection != nil, !isClosed else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        readyWritten = true
        isReady = observedAudioSessionID != nil
    }

    func observeSessionReady(audioSessionID: String) throws -> Bool {
        guard !isClosed, !audioSessionID.isEmpty, audioSessionID.count <= 256,
              observedAudioSessionID == nil || observedAudioSessionID == audioSessionID,
              selection == nil || selection?.audioSessionID == audioSessionID else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        observedAudioSessionID = audioSessionID
        isReady = selection != nil && readyWritten
        return isReady
    }

    func acceptsMedia(_ event: JarvisRealtimeServerEvent) -> Bool {
        guard isReady, !isClosed, pendingInterrupt == nil, let selection,
              (try? selection.validateEnvelope(event)) != nil else { return false }
        if mediaKind == .nativeWebRTC && ["audio.output.delta", "audio.output.done", "audio.output.clear"].contains(event.type) {
            return false // No independently speaking PCM/TTS worker in native mode.
        }
        return true
    }

    func bindOutput(_ event: JarvisRealtimeServerEvent) throws {
        guard isReady, !isClosed, pendingInterrupt == nil, let selection else { throw JarvisVoiceRouteError.notReady }
        let binding = try JarvisVoiceOutputBinding(event: event, selection: selection)
        guard !retiredOutputs.contains(binding.outputID) else { throw JarvisVoiceRouteError.identityMismatch }
        if bindings.contains(binding) { return }
        guard !bindings.contains(where: {
            ($0.outputID == binding.outputID && ($0.responseID != binding.responseID || $0.itemID != binding.itemID)) ||
            ($0.offerID == binding.offerID && $0.outputID == binding.outputID && $0 != binding)
        }) else { throw JarvisVoiceRouteError.identityMismatch }
        guard bindings.count < 128 else { throw JarvisVoiceRouteError.capacity }
        bindings.append(binding)
    }

    /// Called only from AudioPlaybackLedger's contiguous dataPlayedBack result.
    /// Completion here is a lower bound, never a human-heard or whole-output claim.
    func playbackMessages(_ receipt: AudioPlaybackReceipt, now: Date = Date()) -> [[String: Any]] {
        guard isReady, !isClosed, pendingInterrupt == nil, let selection, selection.playbackEvidence,
              mediaKind.supportsRenderReceipts, receipt.playedSamples > 0,
              receipt.playedSamples <= 43_200_000 else { return [] }
        let matched = bindings.filter { $0.responseID == receipt.responseId && $0.itemID == receipt.itemId }
        var result: [[String: Any]] = []
        let milliseconds = receipt.playedSamples / 24
        guard milliseconds > 0 else { return [] }
        let outputs = Set(matched.map(\.outputID))
        for output in outputs where !retiredOutputs.contains(output) && receipt.playedSamples > (playedSamples[output] ?? 0) {
            for binding in matched where binding.outputID == output {
                var value = selection.envelope(type: "voice_route.playback")
                value["offer_id"] = binding.offerID.uuidString.lowercased()
                value["attempt_id"] = binding.attemptID.uuidString.lowercased()
                value["attempt_generation"] = binding.attemptGeneration
                value["output_id"] = output.uuidString.lowercased()
                value["playback"] = ["receipt_id": UUID().uuidString.lowercased(), "observed_at": now.jarvisISO8601String(),
                                     "boundary": "client_render_lower_bound", "rendered_ms": milliseconds]
                result.append(value)
            }
            playedSamples[output] = receipt.playedSamples
        }
        return result
    }

    /// Local mute happens BEFORE network I/O. Await the backend's correlated
    /// command receipt before accepting further media from the control stream.
    func interruptLocally(now: Date = Date()) throws -> [String: Any]? {
        guard isReady, !isClosed, let selection else { return nil }
        guard fenceMedia(false) else { throw JarvisVoiceRouteError.outputNotFenced }
        try retireBindings()
        if pendingInterrupt != nil { return nil }
        let commandID = UUID()
        pendingInterrupt = commandID
        var value = selection.envelope(type: "voice_route.interrupt")
        value.removeValue(forKey: "route")
        value["payload"] = ["route": selection.route.wire, "command_id": commandID.uuidString.lowercased(),
                            "requested_at": now.jarvisISO8601String(), "reason": "local_user_interruption"]
        return value
    }

    func receiveEvent(_ event: JarvisRealtimeServerEvent) throws {
        guard let selection, !isClosed,
              try JarvisVoiceWire.integer(event.values["protocol_version"]) == 1,
              event.audioSessionId == selection.audioSessionID else { throw JarvisVoiceRouteError.identityMismatch }
        try JarvisVoiceWire.keys(event.values, required: ["type", "protocol_version", "audio_session_id", "payload"])
        let payload = try JarvisVoiceWire.object(event.values["payload"])
        guard try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(payload["route"])) == selection.route else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        if payload["kind"] as? String == "command_receipt", payload["operation"] as? String == "interrupt_speech",
           let pendingInterrupt, try JarvisVoiceWire.uuid(payload["command_id"]) == pendingInterrupt {
            guard payload["status"] as? String == "completed",
                  let fenced = payload["output_fenced"] as? NSNumber,
                  CFGetTypeID(fenced) == CFBooleanGetTypeID(), fenced.boolValue else {
                throw JarvisVoiceRouteError.outputNotFenced
            }
            self.pendingInterrupt = nil
        }
        if failure == nil { failure = JarvisVoiceRouteFailure.reported(in: payload) }
        // Typed status is presentation only, not microphone/playback/task control.
    }

    func handleCommand(_ event: JarvisRealtimeServerEvent, now: Date = Date()) throws -> [String: Any] {
        guard let selection, try JarvisVoiceWire.integer(event.values["protocol_version"]) == 1,
              event.audioSessionId == selection.audioSessionID else { throw JarvisVoiceRouteError.identityMismatch }
        try JarvisVoiceWire.keys(event.values, required: ["type", "protocol_version", "audio_session_id", "payload"])
        let closing = event.type == "voice_route.close"
        guard closing || event.type == "voice_route.interrupt" else { throw JarvisVoiceRouteError.invalidWire }
        let payload = try JarvisVoiceWire.object(event.values["payload"])
        try JarvisVoiceWire.keys(payload, required: ["route", "command_id", "requested_at", "reason"],
                                optional: closing ? [] : ["attempt", "output"])
        guard try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(payload["route"])) == selection.route else {
            throw JarvisVoiceRouteError.identityMismatch
        }
        let command = try JarvisVoiceWire.uuid(payload["command_id"])
        try JarvisVoiceWire.timestamp(payload["requested_at"])
        _ = try JarvisVoiceWire.string(payload["reason"], limit: 512)
        let fingerprint = try JSONSerialization.data(withJSONObject: event.values, options: [.sortedKeys])
        if let saved = receipts[command] {
            guard commandFingerprints[command] == fingerprint else { throw JarvisVoiceRouteError.identityMismatch }
            return saved
        }
        guard !isClosed else { throw JarvisVoiceRouteError.notReady }
        guard receipts.count < 128 else { throw JarvisVoiceRouteError.capacity }
        // The backend owns provider cancellation. This receipt proves only the
        // local driver gate, including queued callbacks and the inactive driver.
        let fenced = fenceMedia(closing)
        if closing || !fenced { isClosed = true; isReady = false }
        if fenced { try retireBindings() }
        var value = selection.envelope(type: "voice_route.event")
        value.removeValue(forKey: "route")
        value["payload"] = ["kind": "command_receipt", "route": selection.route.wire,
                            "command_id": command.uuidString.lowercased(), "operation": closing ? "close" : "interrupt_speech",
                            "observed_at": now.jarvisISO8601String(), "status": fenced ? "completed" : "unknown",
                            "output_fenced": fenced, "provider_stopped": NSNull()]
        receipts[command] = value
        commandFingerprints[command] = fingerprint
        return value
    }

    @discardableResult
    func close() -> Bool {
        isReady = false
        isClosed = true
        bindings.removeAll()
        pendingInterrupt = nil
        return fenceMedia(true)
    }

    private func retireBindings() throws {
        guard retiredOutputs.count + bindings.count <= 512 else { throw JarvisVoiceRouteError.capacity }
        retiredOutputs.formUnion(bindings.map(\.outputID))
        bindings.removeAll()
        playedSamples.removeAll()
    }
}
