import AVFoundation
import Foundation
import WebRTC
import os

/// Carries native microphone/playout media only. Jarvis's authenticated socket
/// remains the authority for responses, tools, context changes and expert TTS.
@MainActor
final class JarvisWebRTCClient: NSObject {
    var onSpeechStarted: (() -> Void)?
    var onSpeechStopped: (() -> Void)?
    var onPlaybackChanged: ((Bool) -> Void)?
    var onFailure: ((String) -> Void)?
    var onMicrophoneEnabledChanged: ((Bool) -> Void)?
    var onNativeResponseChanged: ((Bool) -> Void)?

    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "webrtc-audio")
    private static let sharedFactory: RTCPeerConnectionFactory = {
        RTCInitializeSSL()
        return RTCPeerConnectionFactory()
    }()
    private var factory = JarvisWebRTCClient.sharedFactory
    private var duplexRoute: JarvisDuplexAudioRoute?
    private var duplexFenceConfirmed = true
    var recordingAudioDevice: JarvisDuplexAudioDevice? { duplexRoute?.device }
    private var replayFenceConfirmed = true
    private var usesSystemAudio: Bool {
        if duplexRoute != nil { return false }
#if DEBUG
        return replayAudioDevice == nil
#else
        return true
#endif
    }
#if DEBUG
    private var replayAudioDevice: JarvisReplayAudioDevice?
    private var replayPlayableItemID: String?
    private var replayObservedItemResponseID: String?
    private var replayObservedItemDiagnostics: JarvisReplayPlaybackItemDiagnostics?
    private static let replayProviderEventTypes: Set<String> = [
        "session.created", "session.updated", "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped", "input_audio_buffer.committed", "response.created", "response.done",
        "response.output_item.added", "response.output_item.done", "response.output_audio.done",
        "output_audio_buffer.started", "output_audio_buffer.stopped", "output_audio_buffer.cleared",
        "conversation.item.created", "conversation.item.added", "conversation.item.deleted", "error",
        "conversation.item.input_audio_transcription.completed", "conversation.item.input_audio_transcription.failed",
    ]
    private func recordReplayControl(_ stage: JarvisReplayControlStage, fields: [String: Any] = [:],
                                     receivedAt: TimeInterval? = nil) {
        var bound = fields
        bound["connection_id"] = generation.uuidString.lowercased()
        replayAudioDevice?.recordControlEvent(stage, fields: bound, receivedAt: receivedAt)
    }
    /// Observation only. The existing permitResponse and speaker predicates
    /// still decide admission. Backend turn generation is not a media epoch.
    func recordReplayResponsePermission(_ event: JarvisRealtimeServerEvent, controlConnectionID: UUID) {
        guard event.type == "audio.webrtc.response", let response = event.responseId else { return }
        var fields: [String: Any] = ["event_type": event.type, "response_id": response,
                                    "control_connection_id": controlConnectionID.uuidString.lowercased()]
        for key in ["session_id", "audio_session_id", "connection_generation", "turn_generation", "route_generation"] {
            if let value = event.values[key] { fields[key] = value }
        }
        if let route = event.values["route"] as? [String: Any], let value = route["route_generation"] {
            fields["route_generation"] = value
        }
        recordReplayControl(.backendControlHandled, fields: fields)
    }
    func useReplayAudioDevice(_ device: JarvisReplayAudioDevice?) throws {
        close()
        guard duplexFenceConfirmed else { throw JarvisReplayError.retired }
        if let previous = replayAudioDevice, !previous.close() { throw JarvisReplayError.retired }
        replayAudioDevice = device
        replayFenceConfirmed = true
        if let device {
            factory = RTCPeerConnectionFactory(encoderFactory: nil, decoderFactory: nil, audioDevice: device)
        } else { factory = Self.sharedFactory }
    }
#endif
    private var peer: RTCPeerConnection?
    private var channel: RTCDataChannel?
    private var microphone: RTCAudioTrack?
    private var speaker: RTCAudioTrack?
    private var generation = UUID()
    private var disconnectTimeout: Task<Void, Never>?
    private var ownsAudioActivation = false
    private var externalPlaybackActive = false
    private var nativeTurnsRequested = false
    private var nativeTurnsEnabled = false
    private var permittedResponseID: String?
    private var playableResponseID: String?
    private var observedResponseID: String?
    private var lastPlaybackGate: Bool?
    private var routeOutputHeld = true
    private var retiredResponses: [String] = []
    private var startedSpeech: [String] = []
    private var stoppedSpeech: [String] = []
    private(set) var isPlaying = false
    private(set) var microphoneEnabled = false {
        didSet {
            if microphoneEnabled != oldValue { onMicrophoneEnabledChanged?(microphoneEnabled) }
        }
    }
    private(set) var hasRemoteDescription = false

    func prepareOffer(nativeTurns: Bool = false) async throws -> String {
        close()
        guard duplexFenceConfirmed else { throw TransportError.connection }
        nativeTurnsRequested = nativeTurns
        let token = generation
#if DEBUG
        guard replayAudioDevice?.isRetired != true, replayFenceConfirmed else { throw JarvisReplayError.retired }
        recordReplayControl(.offerStarted)
#endif
        var allowDuplex = JarvisDuplexAudioRoute.isSelected
#if DEBUG
        if replayAudioDevice != nil { allowDuplex = false }
#endif
        if allowDuplex {
            let route = JarvisDuplexAudioRoute(generation: token) { [weak self] in
                self?.onFailure?("Shared microphone/playback route failed.")
            }
            duplexRoute = route; factory = route.factory
        }
        if usesSystemAudio || duplexRoute != nil {
            let granted = await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
            }
            guard granted else { throw TransportError.microphonePermission }
            guard token == generation else { throw CancellationError() }
            let audio = RTCAudioSession.sharedInstance()
            audio.useManualAudio = true
            audio.isAudioEnabled = false
        }
        try Task.checkCancellation()
        let configuration = RTCConfiguration()
        configuration.sdpSemantics = .unifiedPlan
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        guard let connection = factory.peerConnection(with: configuration, constraints: constraints, delegate: self) else {
            throw TransportError.connection
        }
        peer = connection
        let source = factory.audioSource(with: constraints)
        let track = factory.audioTrack(with: source, trackId: "jarvis-microphone")
        track.isEnabled = false
        microphone = track
        connection.add(track, streamIds: ["jarvis-audio"])
        let dataConfiguration = RTCDataChannelConfiguration()
        guard let events = connection.dataChannel(forLabel: "oai-events", configuration: dataConfiguration) else {
            throw TransportError.connection
        }
        channel = events
        events.delegate = self
        let offer: RTCSessionDescription = try await withCheckedThrowingContinuation { continuation in
            connection.offer(for: constraints) { description, error in
                if let error { continuation.resume(throwing: error) }
                else if let description { continuation.resume(returning: description) }
                else { continuation.resume(throwing: TransportError.connection) }
            }
        }
        guard generation == token, peer === connection else { throw CancellationError() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.setLocalDescription(offer) { error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
        }
        // The HTTP SDP exchange is non-trickle; include gathered candidates.
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while connection.iceGatheringState != .complete {
            try Task.checkCancellation()
            guard generation == token, peer === connection else { throw CancellationError() }
            guard ContinuousClock.now < deadline else { throw TransportError.iceTimeout }
            try await Task.sleep(for: .milliseconds(40))
        }
        guard generation == token, let sdp = connection.localDescription?.sdp,
              sdp.utf8.count <= 96_000 else { throw TransportError.connection }
#if DEBUG
        recordReplayControl(.offerReady)
#endif
        return sdp
    }

    func applyAnswer(_ sdp: String) async throws {
#if DEBUG
        recordReplayControl(.answerApplyStarted)
#endif
        NSLog("[JarvisWebRTC] Applying remote answer")
        guard let connection = peer, !hasRemoteDescription,
              sdp.hasPrefix("v=0"), sdp.utf8.count <= 96_000 else { throw TransportError.answer }
        let token = generation
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.setRemoteDescription(RTCSessionDescription(type: .answer, sdp: sdp)) { error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
        }
        guard generation == token, peer === connection else { throw CancellationError() }
        hasRemoteDescription = true
#if DEBUG
        recordReplayControl(.answerApplied)
#endif
        // Reconcile the receiver even if its delegate callback is still queued.
        if let track = connection.receivers.compactMap({ $0.track as? RTCAudioTrack }).first {
            speaker = track
            NSLog("[JarvisWebRTC] Remote audio receiver attached")
        }
        updateSpeaker()
        NSLog("[JarvisWebRTC] Remote answer applied")
    }

    func startAudio(microphoneEnabled: Bool) throws {
        if let duplexRoute {
            guard peer != nil, hasRemoteDescription else { throw TransportError.answer }
            try duplexRoute.activate()
            setMicrophoneEnabled(microphoneEnabled); updateSpeaker()
            return
        }
#if DEBUG
        if replayAudioDevice != nil {
            guard peer != nil, hasRemoteDescription, replayAudioDevice?.isRetired == false else { throw TransportError.answer }
            setMicrophoneEnabled(microphoneEnabled)
            updateSpeaker()
            recordReplayControl(.mediaStarted)
            return
        }
#endif
        NSLog("[JarvisWebRTC] Starting native audio")
        guard peer != nil, hasRemoteDescription else { throw TransportError.answer }
        let audio = RTCAudioSession.sharedInstance()
        do {
            audio.lockForConfiguration()
            NSLog("[JarvisWebRTC] Audio configuration lock acquired")
            defer { audio.unlockForConfiguration() }
            try audio.setCategory(.playAndRecord,
                                  with: [.defaultToSpeaker, .allowBluetoothA2DP])
            NSLog("[JarvisWebRTC] Audio category set")
            try audio.setMode(.voiceChat)
            let session = AVAudioSession.sharedInstance()
            if let builtInMic = session.availableInputs?.first(where: { $0.portType == .builtInMic }),
               !session.currentRoute.inputs.contains(where: { $0.portType == .headsetMic || $0.portType == .usbAudio }) {
                try? session.setPreferredInput(builtInMic)
            }
            NSLog("[JarvisWebRTC] Audio mode set")
            if !ownsAudioActivation {
                try audio.setActive(true)
                NSLog("[JarvisWebRTC] Audio session activated")
                ownsAudioActivation = true
            }
            audio.isAudioEnabled = true
            NSLog("[JarvisWebRTC] Native audio enabled")
        }
        // Track setters may synchronously dispatch to WebRTC's worker, which
        // also uses the audio-session lock. Never hold it across those calls.
        setMicrophoneEnabled(microphoneEnabled)
        NSLog("[JarvisWebRTC] Microphone state applied")
        updateSpeaker()
        NSLog("[JarvisWebRTC] Native audio setup complete")
    }

    func setMicrophoneEnabled(_ enabled: Bool) {
#if DEBUG
        replayAudioDevice?.setInputEnabled(enabled && hasRemoteDescription)
#endif
        microphoneEnabled = enabled && hasRemoteDescription
        microphone?.isEnabled = microphoneEnabled
        duplexRoute?.setMicrophoneEnabled(microphoneEnabled)
    }

    func setExternalPlaybackActive(_ active: Bool) {
        if active && !externalPlaybackActive { interrupt() }
        externalPlaybackActive = active
        updateSpeaker()
    }

    func permitResponse(_ responseID: String) {
        guard peer != nil, !retiredResponses.contains(responseID) else { return }
        routeOutputHeld = false
        permittedResponseID = responseID
        observedResponseID = responseID
        record("client_webrtc_response_permitted", responseID: responseID)
#if DEBUG
        recordReplayControl(.responsePermitted, fields: ["response_id": responseID])
#endif
        updateSpeaker()
    }

    func activateNativeTurns() -> Bool {
        guard nativeTurnsRequested, peer != nil, hasRemoteDescription else { return false }
        nativeTurnsEnabled = true
        return true
    }

    private func handleNativeProviderEvent(_ event: JarvisRealtimeServerEvent) {
        guard nativeTurnsEnabled else { return }
        let response = event.values["response"] as? [String: Any]
        let responseID = event.responseId ?? response?["id"] as? String
        switch event.type {
        case "response.created":
            guard let responseID else { return }
            permitResponse(responseID)
            onNativeResponseChanged?(true)
        case "response.output_item.added", "response.output_item.created",
             "response.output_item.done", "response.output_item.completed":
            var values = event.values
            if let responseID { values["response_id"] = responseID }
            handleProviderEvent(.init(type: "audio.webrtc.output_item", values: values),
                                replayFromProviderDataChannel: true)
        case "output_audio_buffer.started", "output_audio_buffer.stopped", "output_audio_buffer.cleared":
            var values = event.values
            if let responseID { values["response_id"] = responseID }
            let kind = "audio.webrtc." + String(event.type.split(separator: ".").last ?? "")
            handleProviderEvent(.init(type: kind, values: values), replayFromProviderDataChannel: true)
        case "response.done":
            if let responseID, response?["status"] as? String != "completed" {
                interrupt(responseID: responseID)
            }
            if responseID == observedResponseID || observedResponseID == nil {
                onNativeResponseChanged?(false)
            }
        default:
            break
        }
    }

    func interrupt(responseID: String? = nil) {
        if let responseID, responseID != permittedResponseID, responseID != observedResponseID { return }
#if DEBUG
        replayAudioDevice?.recordLocalInterruption(.clientInterrupt)
        replayPlayableItemID = nil
#endif
        routeOutputHeld = true
        for id in [permittedResponseID, observedResponseID].compactMap({ $0 }) { retire(id) }
        permittedResponseID = nil
        playableResponseID = nil
        observedResponseID = nil
        setPlaying(false)
        updateSpeaker()
    }

    /// A synchronous local track gate, not provider cancellation or proof that
    /// the listener heard any samples. Late receiver callbacks respect this gate.
func fenceRouteOutput(closeRoute: Bool) -> Bool {
#if DEBUG
    if closeRoute { replayAudioDevice?.recordLocalInterruption(.routeFenceClose) }
#endif
    routeOutputHeld = true
    if closeRoute { close() } else { interrupt() }
    return replayFenceConfirmed && duplexFenceConfirmed && speaker?.isEnabled != true && (!closeRoute || peer == nil)
}

    /// Output decisions come only from the authenticated control connection.
    /// Buffer hints are not proof of audible output.
    func handleProviderEvent(_ event: JarvisRealtimeServerEvent, replayFromProviderDataChannel: Bool = false) {
#if DEBUG
        // Direct speech was already observed at its data-channel receipt. This
        // diagnostic-only flag never changes the event's admission or effects.
        if !replayFromProviderDataChannel,
           ["audio.webrtc.output_item", "input_audio_buffer.speech_started", "audio.input.started",
            "input_audio_buffer.speech_stopped", "audio.input.stopped", "audio.webrtc.started",
            "audio.webrtc.stopped", "audio.webrtc.cleared"].contains(event.type) {
            var fields: [String: Any] = ["event_type": event.type]
            if let id = event.responseId { fields["response_id"] = id }
            if let id = event.itemId { fields["item_id"] = id }
            recordReplayControl(.backendControlHandled, fields: fields)
        }
#endif
        switch event.type {
        case "audio.webrtc.output_item":
            guard let responseID = event.responseId,
                  let item = event.values["item"] as? [String: Any] else { return }
            let action = JarvisDirectOutputPolicy.bindingAction(
                item: item, responseID: responseID, permittedResponseID: permittedResponseID,
                isRetired: retiredResponses.contains(responseID), eventItemID: event.itemId)
            switch action {
            case .ignore:
                return
            case .retainMessage:
                break
            case .playMessage(let itemID):
                playableResponseID = responseID
#if DEBUG
                replayPlayableItemID = itemID
#endif
            case .clear:
                playableResponseID = nil
#if DEBUG
                replayPlayableItemID = nil
#endif
            }
#if DEBUG
            // Observe the already accepted item only. This does not admit audio.
            replayObservedItemResponseID = responseID
            replayObservedItemDiagnostics = JarvisReplayPlaybackItemDiagnostics(
                item: item, policyAllowsPlayback: JarvisDirectOutputPolicy.allowsPlayback(item: item))
#endif
            record("client_webrtc_output_item_received", responseID: responseID)
            updateSpeaker()
        case "input_audio_buffer.speech_started", "audio.input.started":
            guard let id = event.itemId, !startedSpeech.contains(id), !stoppedSpeech.contains(id) else { return }
            startedSpeech.append(id)
            if startedSpeech.count > 256 { startedSpeech.removeFirst() }
            recordingAudioDevice?.setMediaUserSpeaking(true)
            interrupt()
            record("client_webrtc_speech_started", responseID: nil)
            onSpeechStarted?()
        case "input_audio_buffer.speech_stopped", "audio.input.stopped":
            guard let id = event.itemId, !stoppedSpeech.contains(id) else { return }
            stoppedSpeech.append(id)
            if stoppedSpeech.count > 256 { stoppedSpeech.removeFirst() }
            recordingAudioDevice?.setMediaUserSpeaking(startedSpeech.contains { !stoppedSpeech.contains($0) })
            record("client_webrtc_speech_stopped", responseID: nil)
            onSpeechStopped?()
        case "audio.webrtc.started":
            guard let id = event.responseId, id == permittedResponseID, !retiredResponses.contains(id) else { return }
            if !isPlaying { record("client_webrtc_output_started", responseID: id) }
            setPlaying(true)
        case "audio.webrtc.stopped", "audio.webrtc.cleared":
            guard let id = event.responseId else { return }
            retire(id)
            if permittedResponseID == id { permittedResponseID = nil; setPlaying(false) }
            if playableResponseID == id { playableResponseID = nil }
            if observedResponseID == id { observedResponseID = nil }
            updateSpeaker()
            record("client_webrtc_output_stopped", responseID: id)
        default:
            break
        }
    }

    func close() {
        let closingDuplex = duplexRoute
        if let closingDuplex { duplexFenceConfirmed = closingDuplex.fenceAndClose() }
#if DEBUG
        recordReplayControl(.connectionReset, fields: ["reason": peer == nil ? "pre_offer_reset" : "close_peer"])
        let closingDiagnostics = replayPlaybackGateDiagnostics(origin: .forcedClose)
        replayPlayableItemID = nil
        replayAudioDevice?.setInputEnabled(false)
        replayAudioDevice?.setOutputEnabled(false, diagnostics: closingDiagnostics)
        replayObservedItemResponseID = nil
        replayObservedItemDiagnostics = nil
        // prepareOffer calls close before its first peer exists. Do not retire
        // the newly installed device there; an existing peer is terminal.
        if peer != nil, let replayAudioDevice { replayFenceConfirmed = replayAudioDevice.close() }
#endif
        generation = UUID()
        routeOutputHeld = true
        disconnectTimeout?.cancel()
        disconnectTimeout = nil
        setMicrophoneEnabled(false)
        speaker?.isEnabled = false
        channel?.delegate = nil
        channel?.close()
        channel = nil
        let oldPeer = peer
        peer = nil
        oldPeer?.delegate = nil
        oldPeer?.close()
        if closingDuplex != nil, duplexFenceConfirmed {
            duplexRoute = nil; factory = Self.sharedFactory
        }
        microphone = nil
        speaker = nil
        hasRemoteDescription = false
        externalPlaybackActive = false
        nativeTurnsRequested = false
        nativeTurnsEnabled = false
        permittedResponseID = nil
        playableResponseID = nil
        observedResponseID = nil
        lastPlaybackGate = nil
        retiredResponses.removeAll()
        startedSpeech.removeAll()
        stoppedSpeech.removeAll()
        setPlaying(false)
        if usesSystemAudio {
            let audio = RTCAudioSession.sharedInstance()
            audio.isAudioEnabled = false
            if ownsAudioActivation {
                audio.lockForConfiguration()
                try? audio.setActive(false)
                audio.unlockForConfiguration()
                ownsAudioActivation = false
            }
        }
    }

#if DEBUG
    private func replayPlaybackGateDiagnostics(
        origin: JarvisReplayPlaybackGateDiagnostics.Origin
    ) -> JarvisReplayPlaybackGateDiagnostics {
        .init(origin: origin, remoteDescriptionPresent: hasRemoteDescription,
              permittedIdentityPresent: permittedResponseID != nil,
              playableIdentityPresent: playableResponseID != nil,
              playableIdentityMatchesPermit: permittedResponseID != nil && permittedResponseID == playableResponseID,
              externalPlaybackInactive: !externalPlaybackActive, routeOutputHeld: routeOutputHeld,
              receiverPresent: speaker != nil, replayItemIdentityPresent: replayPlayableItemID != nil,
              itemMatchesPermittedResponse: replayObservedItemResponseID != nil
                  && replayObservedItemResponseID == permittedResponseID,
              item: replayObservedItemDiagnostics)
    }
#endif

    private func updateSpeaker() {
        let allowed = hasRemoteDescription && permittedResponseID != nil
            && permittedResponseID == playableResponseID && !externalPlaybackActive && !routeOutputHeld
        speaker?.isEnabled = allowed
#if DEBUG
        replayAudioDevice?.setOutputEnabled(allowed, responseID: permittedResponseID,
                                           itemID: replayPlayableItemID, connectionID: generation,
                                           diagnostics: replayPlaybackGateDiagnostics(origin: .speakerPredicate))
#endif
        let enabled = speaker != nil && allowed
        duplexRoute?.setOutputEnabled(enabled)
        if lastPlaybackGate != enabled {
            lastPlaybackGate = enabled
            let route = AVAudioSession.sharedInstance().currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: ",")
            NSLog("[JarvisWebRTC] Playback gate enabled=%d receiver=%d admitted=%d final=%d external=%d output=%@",
                  enabled ? 1 : 0, speaker == nil ? 0 : 1, permittedResponseID == nil ? 0 : 1,
                  playableResponseID == nil ? 0 : 1, externalPlaybackActive ? 1 : 0, route)
        }
    }

    private func setPlaying(_ playing: Bool) {
        guard isPlaying != playing else { return }
        isPlaying = playing
        onPlaybackChanged?(playing)
    }

    private func retire(_ id: String) {
        guard !retiredResponses.contains(id) else { return }
        retiredResponses.append(id)
        if retiredResponses.count > 256 { retiredResponses.removeFirst() }
    }

    private func record(_ stage: String, responseID: String?) {
        var fields: [String: Any] = ["transport": "webrtc", "boundary": "provider_buffer_event_received"]
        if let responseID { fields["response_id"] = responseID }
        if let trace = RealtimeDiagnostics.record(stage, fields: fields) {
            logger.info("realtime_trace \(trace, privacy: .public)")
            NSLog("[JarvisWebRTC] %@", trace)
        }
    }

    private enum TransportError: LocalizedError {
        case microphonePermission, connection, answer, iceTimeout
        var errorDescription: String? {
            switch self {
            case .microphonePermission: return "Microphone permission is required for Jarvis voice."
            case .connection: return "Could not prepare the direct audio connection."
            case .answer: return "The direct audio handshake was not accepted."
            case .iceTimeout: return "Direct audio network setup timed out. Check your connection."
            }
        }
    }
}

extension JarvisWebRTCClient: RTCPeerConnectionDelegate {
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {}
    nonisolated func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {}

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {
        Task { @MainActor [weak self] in
            guard let self, self.peer === peerConnection else { return }
            self.channel = dataChannel
            dataChannel.delegate = self
        }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didAdd rtpReceiver: RTCRtpReceiver,
                                    streams: [RTCMediaStream]) {
        guard let track = rtpReceiver.track as? RTCAudioTrack else { return }
        track.isEnabled = false
        Task { @MainActor [weak self] in
            guard let self, self.peer === peerConnection else { return }
            self.speaker = track
            NSLog("[JarvisWebRTC] Remote audio receiver callback")
            self.updateSpeaker()
        }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {
        Task { @MainActor [weak self] in
            guard let self, self.peer === peerConnection else { return }
            self.disconnectTimeout?.cancel()
            self.disconnectTimeout = nil
            switch newState {
            case .failed:
                self.onFailure?("Direct audio connection failed. Check your network and restart Jarvis.")
            case .disconnected:
                self.disconnectTimeout = Task { @MainActor [weak self] in
                    do { try await Task.sleep(for: .seconds(8)) } catch { return }
                    guard let self, self.peer === peerConnection,
                          peerConnection.iceConnectionState == .disconnected else { return }
                    self.onFailure?("Direct audio connection was lost. Restart Jarvis to reconnect.")
                }
            default:
                break
            }
        }
    }
}

extension JarvisWebRTCClient: RTCDataChannelDelegate {
    nonisolated func dataChannelDidChangeState(_ dataChannel: RTCDataChannel) {}

    nonisolated func dataChannel(_ dataChannel: RTCDataChannel, didReceiveMessageWith buffer: RTCDataBuffer) {
        guard !buffer.isBinary, buffer.data.count <= 262_144 else {
#if DEBUG
            Task { @MainActor [weak self] in
                guard let self, self.channel === dataChannel else { return }
                self.recordReplayControl(.providerDataChannelHandled,
                    fields: ["event_type": "replay.provider_input.invalid", "reason": "invalid_envelope"])
            }
#endif
            return
        }
        let received = ProcessInfo.processInfo.systemUptime
        let data = buffer.data
        Task { @MainActor [weak self] in
            guard let self, self.channel === dataChannel else { return }
            guard let values = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let type = values["type"] as? String else {
#if DEBUG
                self.recordReplayControl(.providerDataChannelHandled,
                    fields: ["event_type": "replay.provider_input.invalid", "reason": "invalid_envelope"], receivedAt: received)
#endif
                return
            }
#if DEBUG
            if Self.replayProviderEventTypes.contains(type) {
                var fields: [String: Any] = ["event_type": type,
                    "boundary": dataChannel.isOrdered ? "ordered_current_provider_channel" : "unordered_provider_channel"]
                let response = values["response"] as? [String: Any]
                let item = values["item"] as? [String: Any]
                let error = values["error"] as? [String: Any]
                if let id = values["response_id"] as? String ?? response?["id"] as? String { fields["response_id"] = id }
                if let id = values["item_id"] as? String ?? item?["id"] as? String { fields["item_id"] = id }
                if let id = values["event_id"] as? String { fields["provider_event_id"] = id }
                if let code = error?["code"] as? String { fields["reason"] = code }
                if let id = error?["event_id"] as? String { fields["client_event_id"] = id }
                self.recordReplayControl(.providerDataChannelHandled, fields: fields, receivedAt: received)
            }
#endif
            if self.nativeTurnsEnabled {
                self.handleNativeProviderEvent(JarvisRealtimeServerEvent(type: type, values: values))
            }
            // Legacy retains the direct path for prompt local barge-in only. Output
            // metadata and retirement must not race the backend control stream.
            guard ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"].contains(type) else { return }
            if let trace = RealtimeDiagnostics.record("client_native_control_handling", fields: [
                "event_type": type, "boundary": "data_channel_callback_to_main_actor",
                "receive_monotonic_s": received,
                "actor_wait_ms": (ProcessInfo.processInfo.systemUptime - received) * 1_000,
            ]) { self.logger.info("realtime_trace \(trace, privacy: .public)") }
            self.handleProviderEvent(JarvisRealtimeServerEvent(type: type, values: values),
                                     replayFromProviderDataChannel: true)
        }
    }
}
