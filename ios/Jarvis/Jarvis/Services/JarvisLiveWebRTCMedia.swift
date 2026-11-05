import AVFoundation
import Foundation
import WebRTC

/// Real native RTP media. Backend sideband owns ALL Live protocol commands.
/// No Realtime output IDs, output_finished synthesis, data-channel commands or
/// backend PCM paths exist in this implementation.
@MainActor
final class JarvisLiveWebRTCMedia: NSObject, JarvisLiveMediaTransport {
    enum MediaError: Error { case microphonePermission, connection, answer, iceTimeout, retired }
    var onConnectivityChanged: (() -> Void)?
    var onFailure: (() -> Void)?
    var onProviderStarted: ((JarvisRealtimeServerEvent) -> Void)?
    private(set) var hasRemoteDescription = false
    private(set) var outputEnabled = false
    var isConnected: Bool {
        guard !retired, let peer else { return false }
        return peer.iceConnectionState == .connected || peer.iceConnectionState == .completed
    }

    private static let factory: RTCPeerConnectionFactory = {
        RTCInitializeSSL()
        return RTCPeerConnectionFactory()
    }()
    private var replayFenceConfirmed = true
    private var duplexRoute: JarvisDuplexAudioRoute?
    private var duplexFenceConfirmed = true
    var recordingAudioDevice: JarvisDuplexAudioDevice? {
        guard !retired, outputEnabled, isConnected else { return nil }
        return duplexRoute?.device
    }
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
    private var replayFactory: RTCPeerConnectionFactory?
    private var diagnosticTask: Task<Void, Never>?
    private let diagnosticID = UUID().uuidString
    func useReplayAudioDevice(_ device: JarvisReplayAudioDevice) throws {
        guard !prepared, !retired, replayFactory == nil, !device.isRetired else { throw MediaError.retired }
        RTCInitializeSSL()
        replayAudioDevice = device
        replayFactory = RTCPeerConnectionFactory(encoderFactory: nil, decoderFactory: nil, audioDevice: device)
    }
#endif
    private var mediaFactory: RTCPeerConnectionFactory {
        if let duplexRoute { return duplexRoute.factory }
#if DEBUG
        if let replayFactory { return replayFactory }
#endif
        return Self.factory
    }
    private var peer: RTCPeerConnection?
    private var channel: RTCDataChannel?
    private nonisolated let startupInbox = JarvisLiveStartupInbox()
    private var microphone: RTCAudioTrack?
    private var prepared = false
    private var retired = false
    private var ownsAudioConfiguration = false
    private var ownsAudioActivation = false
    private let microphonePermission: () async -> Bool

    init(microphonePermission: @escaping () async -> Bool = {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
        }
    }) {
        self.microphonePermission = microphonePermission
        super.init()
    }

    func prepareOffer() async throws -> String {
        guard !retired, !prepared else { throw MediaError.retired }
        prepared = true
        do {
            var allowDuplex = JarvisDuplexAudioRoute.isSelected
#if DEBUG
            if replayAudioDevice != nil { allowDuplex = false }
#endif
            if allowDuplex {
                duplexRoute = JarvisDuplexAudioRoute(generation: UUID()) { [weak self] in self?.onFailure?() }
            }
            if usesSystemAudio || duplexRoute != nil {
            guard await microphonePermission() else { throw MediaError.microphonePermission }
            try Task.checkCancellation()
            guard !retired else { throw CancellationError() }
            let audio = RTCAudioSession.sharedInstance()
            audio.useManualAudio = true
            audio.isAudioEnabled = false
            ownsAudioConfiguration = duplexRoute == nil
            }
            try Task.checkCancellation()
            guard !retired else { throw CancellationError() }
            let configuration = RTCConfiguration()
            configuration.sdpSemantics = .unifiedPlan
            let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
            guard let connection = mediaFactory.peerConnection(with: configuration, constraints: constraints, delegate: self) else {
                throw MediaError.connection
            }
            peer = connection
            let track = mediaFactory.audioTrack(with: mediaFactory.audioSource(with: constraints), trackId: "jarvis-live-microphone")
            track.isEnabled = false
            microphone = track
            connection.add(track, streamIds: ["jarvis-live-audio"])
            // Live requires this channel in the SDP offer. It is receive-only:
            // the authenticated backend remains the sole command authority.
            guard let events = connection.dataChannel(forLabel: "oai-events", configuration: RTCDataChannelConfiguration()) else {
                throw MediaError.connection
            }
            channel = events
            startupInbox.bind(events)
            events.delegate = self
            let offer: RTCSessionDescription = try await withCheckedThrowingContinuation { continuation in
                connection.offer(for: constraints) { description, error in
                    if let error { continuation.resume(throwing: error) }
                    else if let description { continuation.resume(returning: description) }
                    else { continuation.resume(throwing: MediaError.connection) }
                }
            }
            try ensureCurrent(connection)
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                connection.setLocalDescription(offer) { error in
                    if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                }
            }
            let deadline = ContinuousClock.now.advanced(by: .seconds(10))
            while connection.iceGatheringState != .complete {
                try ensureCurrent(connection)
                guard ContinuousClock.now < deadline else { throw MediaError.iceTimeout }
                try await Task.sleep(for: .milliseconds(40))
            }
            try ensureCurrent(connection)
            guard let sdp = connection.localDescription?.sdp, sdp.utf8.count <= 65_536 else { throw MediaError.connection }
            return sdp
        } catch {
            fenceAndClose()
            throw error
        }
    }

    func applyAnswer(_ sdp: String) async throws {
        guard !retired, let connection = peer, !hasRemoteDescription,
              sdp.hasPrefix("v=0"), sdp.utf8.count <= 65_536 else { throw MediaError.answer }
        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                connection.setRemoteDescription(RTCSessionDescription(type: .answer, sdp: sdp)) { error in
                    if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                }
            }
            try ensureCurrent(connection)
            hasRemoteDescription = true
            updateReceivers()
        } catch {
            fenceAndClose()
            throw error
        }
    }

    func activate(microphoneEnabled: Bool) throws {
        guard !retired, hasRemoteDescription, isConnected else { throw MediaError.connection }
#if DEBUG
        defer { if outputEnabled { startAudioDiagnostics() } }
#endif
        if let duplexRoute {
            try duplexRoute.activate()
            outputEnabled = true; setMicrophoneEnabled(microphoneEnabled); updateReceivers()
            return
        }
#if DEBUG
        if replayAudioDevice != nil {
            guard replayAudioDevice?.isRetired == false else { throw MediaError.retired }
            outputEnabled = true
            setMicrophoneEnabled(microphoneEnabled)
            updateReceivers()
            return
        }
#endif
        let audio = RTCAudioSession.sharedInstance()
        do {
            audio.lockForConfiguration()
            defer { audio.unlockForConfiguration() }
            try audio.setCategory(.playAndRecord, with: [.defaultToSpeaker, .allowBluetoothA2DP])
            try audio.setMode(.voiceChat)
            let session = AVAudioSession.sharedInstance()
            if let builtInMic = session.availableInputs?.first(where: { $0.portType == .builtInMic }),
               !session.currentRoute.inputs.contains(where: { $0.portType == .headsetMic || $0.portType == .usbAudio }) {
                try? session.setPreferredInput(builtInMic)
            }
            if !ownsAudioActivation {
                try audio.setActive(true)
                ownsAudioActivation = true
            }
            audio.isAudioEnabled = true
        }
        // Track setters can dispatch to WebRTC's worker; do not hold the audio
        // session lock while invoking them.
        outputEnabled = true
        setMicrophoneEnabled(microphoneEnabled)
        updateReceivers()
    }

    func setMicrophoneEnabled(_ enabled: Bool) {
#if DEBUG
        replayAudioDevice?.setInputEnabled(enabled && !retired && outputEnabled && hasRemoteDescription)
#endif
        microphone?.isEnabled = enabled && !retired && outputEnabled && hasRemoteDescription
        duplexRoute?.setMicrophoneEnabled(enabled && !retired && outputEnabled && hasRemoteDescription)
    }

    @discardableResult func fenceAndClose() -> Bool {
#if DEBUG
        diagnosticTask?.cancel()
        diagnosticTask = nil
#endif
        if let duplexRoute { duplexFenceConfirmed = duplexRoute.fenceAndClose() }
        guard !retired else {
#if DEBUG
            if let replayAudioDevice { replayFenceConfirmed = replayAudioDevice.close() }
#endif
            return replayFenceConfirmed && duplexFenceConfirmed
        }
        retired = true
#if DEBUG
        if let replayAudioDevice { replayFenceConfirmed = replayAudioDevice.close() }
#endif
        outputEnabled = false
        microphone?.isEnabled = false
        updateReceivers()
        startupInbox.close()
        channel?.delegate = nil
        channel?.close()
        channel = nil
        let previous = peer
        peer = nil
        previous?.delegate = nil
        previous?.close() // Destroys the jitter/receiver queues; never reused.
        microphone = nil
        hasRemoteDescription = false
        if ownsAudioConfiguration {
            let audio = RTCAudioSession.sharedInstance()
            audio.isAudioEnabled = false
            if ownsAudioActivation {
                audio.lockForConfiguration()
                try? audio.setActive(false)
                audio.unlockForConfiguration()
                ownsAudioActivation = false
            }
            ownsAudioConfiguration = false
        }
        return replayFenceConfirmed && duplexFenceConfirmed // Local fence, not provider shutdown/hearing.
    }

    private func ensureCurrent(_ connection: RTCPeerConnection) throws {
        try Task.checkCancellation()
        guard !retired, peer === connection else { throw CancellationError() }
    }

#if DEBUG
    private func startAudioDiagnostics() {
        guard diagnosticTask == nil else { return }
        diagnosticTask = Task { [weak self] in
            // Five samples through 30 seconds, not continuous logging or recording.
            for delay in [2, 3, 5, 10, 10] {
                do { try await Task.sleep(for: .seconds(delay)) } catch { return }
                guard let self, !self.retired, let connection = self.peer else { return }
                let base: [String: Any] = [
                    "captured_at": Date().ISO8601Format(),
                    "media_id": self.diagnosticID,
                    "microphone_track_enabled": self.microphone?.isEnabled == true,
                    "output_enabled": self.outputEnabled,
                    "ice_state": connection.iceConnectionState.rawValue,
                    "native_input": self.duplexRoute?.device.inputDiagnosticSnapshot() ?? [:]]
                connection.statistics { report in
                    var payload = base
                    payload["rtp"] = report.statistics.values.compactMap { stat -> [String: Any]? in
                        guard ["media-source", "outbound-rtp", "inbound-rtp"].contains(stat.type) else { return nil }
                        var fields: [String: Any] = ["type": stat.type]
                        for key in ["kind", "mediaType", "packetsSent", "bytesSent", "packetsReceived",
                                    "bytesReceived", "audioLevel", "totalAudioEnergy", "totalSamplesDuration",
                                    "totalSamplesSent", "totalSamplesReceived"] {
                            if let value = stat.values[key] { fields[key] = value }
                        }
                        return fields
                    }
                    guard JSONSerialization.isValidJSONObject(payload) else {
                        NSLog("[JarvisLiveAudio] diagnostic_payload_invalid")
                        return
                    }
                    if let data = try? JSONSerialization.data(withJSONObject: payload, options: .sortedKeys),
                       let text = String(data: data, encoding: .utf8) {
                        NSLog("[JarvisLiveAudio] %@", text)
                        // Keep only the latest counter snapshot across app relaunches.
                        // No audio contents or conversation are written here.
                        if let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
                            try? data.write(to: cache.appendingPathComponent("JarvisLiveAudioDiagnostic.json"),
                                            options: .atomic)
                        }
                    }
                }
            }
        }
    }
#endif

    private func updateReceivers() {
        duplexRoute?.setOutputEnabled(outputEnabled && !retired)
#if DEBUG
        replayAudioDevice?.setOutputEnabled(outputEnabled && !retired)
#endif
        for receiver in peer?.receivers ?? [] {
            (receiver.track as? RTCAudioTrack)?.isEnabled = outputEnabled && !retired
        }
    }
}

extension JarvisLiveWebRTCMedia: RTCPeerConnectionDelegate {
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {}
    nonisolated func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {}
    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {
        Task { @MainActor [weak self] in
            guard let self, !self.retired, self.peer === peerConnection, self.channel === dataChannel else {
                dataChannel.close()
                return
            }
            dataChannel.delegate = self
        }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didAdd rtpReceiver: RTCRtpReceiver,
                                    streams: [RTCMediaStream]) {
        guard let track = rtpReceiver.track as? RTCAudioTrack else { return }
        track.isEnabled = false
        Task { @MainActor [weak self] in
            guard let self, !self.retired, self.peer === peerConnection else { return }
            track.isEnabled = self.outputEnabled
        }
    }

    nonisolated func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {
        Task { @MainActor [weak self] in
            guard let self, !self.retired, self.peer === peerConnection else { return }
            switch newState {
            case .failed, .disconnected, .closed:
                self.fenceAndClose()
                self.onFailure?()
            default:
                self.onConnectivityChanged?()
            }
        }
    }
}

extension JarvisLiveWebRTCMedia: RTCDataChannelDelegate {
    nonisolated func dataChannelDidChangeState(_ dataChannel: RTCDataChannel) {
        guard dataChannel.readyState == .closed, startupInbox.fail(source: dataChannel) else { return }
        drainStartup(dataChannel)
    }

    nonisolated func dataChannel(_ dataChannel: RTCDataChannel, didReceiveMessageWith buffer: RTCDataBuffer) {
        guard startupInbox.receive(buffer.data, binary: buffer.isBinary, source: dataChannel) else { return }
        drainStartup(dataChannel)
    }

    private nonisolated func drainStartup(_ source: RTCDataChannel) {
        Task { @MainActor [weak self] in
            guard let self, !self.retired, self.channel === source,
                  let result = self.startupInbox.take(source: source) else { return }
            switch result {
            case .started(let event): self.onProviderStarted?(event)
            case .failed:
                self.fenceAndClose()
                self.onFailure?()
            }
        }
    }
}

/// Bounded cross-thread ingress: one pending startup event and one scheduled
/// drain, never a task per transcript. Only an actual session.started is retained.
/// Source identity and permanent closure reject late callbacks from old channels.
final class JarvisLiveStartupInbox: @unchecked Sendable {
    enum Result { case started(JarvisRealtimeServerEvent), failed }
    private let lock = NSLock()
    private var source: AnyObject?
    private var pending: Data?
    private var providerID: String?
    private var scheduled = false
    private var failed = false
    private var closed = false

    func bind(_ source: AnyObject) {
        lock.lock(); defer { lock.unlock() }
        guard self.source == nil, !closed else { return }
        self.source = source
    }

    /// True schedules the sole drain. Unknown, binary and oversized events are
    /// dropped without claiming startup; the signaling deadline remains active.
    func receive(_ data: Data, binary: Bool, source: AnyObject) -> Bool {
        guard !binary, data.count <= 65_536 else { return false }
        lock.lock(); defer { lock.unlock() }
        guard !closed, !failed, self.source === source,
              let values = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let type = values["type"] as? String else { return false }
        if type == "session.closed" { return markFailed() }
        guard type == "session.started" else { return false }
        guard let session = values["session"] as? [String: Any],
              let id = session["id"] as? String, !id.isEmpty, id.count <= 256 else { return markFailed() }
        if let providerID {
            return providerID == id ? false : markFailed()
        }
        providerID = id
        pending = data
        if scheduled { return false }
        scheduled = true
        return true
    }

    func fail(source: AnyObject) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !closed, self.source === source else { return false }
        return markFailed()
    }

    private func markFailed() -> Bool {
        failed = true
        pending = nil
        if scheduled { return false }
        scheduled = true
        return true
    }

    func take(source: AnyObject) -> Result? {
        lock.lock(); defer { lock.unlock() }
        guard !closed, self.source === source, scheduled else { return nil }
        scheduled = false
        if failed { closed = true; return .failed }
        guard let pending,
              let values = (try? JSONSerialization.jsonObject(with: pending)) as? [String: Any] else { return nil }
        self.pending = nil
        return .started(JarvisRealtimeServerEvent(type: "session.started", values: values))
    }

    func close() {
        lock.lock(); defer { lock.unlock() }
        closed = true; source = nil; pending = nil; scheduled = false
    }
}

extension JarvisLiveVoiceRouteDriver {
    /// Production entry point. The explicit previous-owner fence must close the
    /// application's existing PCM/Realtime driver before this native peer starts.
    convenience init(sessionID: UUID, connectionID: UUID, fencePrevious: @escaping () -> Bool) {
        self.init(sessionID: sessionID, connectionID: connectionID, media: JarvisLiveWebRTCMedia(),
                  ownership: .shared, fencePrevious: fencePrevious)
    }
}

// Keep native audio types out of the independently tested route protocol.
extension JarvisLiveVoiceRouteDriver {
    var recordingAudioDevice: JarvisDuplexAudioDevice? {
        (activeMediaForRecording as? JarvisLiveWebRTCMedia)?.recordingAudioDevice
    }
}
extension JarvisLiveAppRoute {
    var recordingAudioDevice: JarvisDuplexAudioDevice? {
        (activeMediaForRecording as? JarvisLiveWebRTCMedia)?.recordingAudioDevice
    }
}
