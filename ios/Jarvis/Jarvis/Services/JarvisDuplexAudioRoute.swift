import AVFoundation
import Foundation
import WebRTC

/// One owner per peer lifetime. Recording only attaches a sink to device;
/// it must never replace this route, factory, or shared audio-session lease.
@MainActor
final class JarvisDuplexAudioRoute {
    static let preferenceKey = "jarvis.recordingCaptureEnabled"
    static var isSelected: Bool {
        UserDefaults.standard.object(forKey: preferenceKey) == nil
            || UserDefaults.standard.bool(forKey: preferenceKey)
    }
    let device: JarvisDuplexAudioDevice
    let factory: RTCPeerConnectionFactory
    private let onFailure: () -> Void
    private var observers: [NSObjectProtocol] = []
    private var active = false
    private var suspended = false
    private var retired = false
    private var microphoneWanted = false
    private var outputWanted = false
    private var routeSignature: String?

    init(generation: UUID, onFailure: @escaping () -> Void) {
        self.onFailure = onFailure
        let lease = JarvisPhysicalDuplexLease()
        device = JarvisDuplexAudioDevice(voiceGeneration: generation,
            sessionLease: .init(acquire: { try lease.acquire(rate: $0, buffer: $1) }, release: { lease.release() }),
            onStatus: { value in
                NSLog("[JarvisDuplex] %@ route=%llu status=%d", value.reason, value.routeGeneration, value.osStatus ?? 0)
            })
        RTCInitializeSSL()
        factory = RTCPeerConnectionFactory(encoderFactory: nil, decoderFactory: nil, audioDevice: device)
    }

    func activate() throws {
        guard !retired else { throw RouteError.retired }
        if !active {
            guard device.controlSynchronously(.authorize) else { throw RouteError.unavailable }
            active = true; routeSignature = Self.currentSignature(); observeSession()
        }
        applyGates()
    }
    func setMicrophoneEnabled(_ enabled: Bool) { microphoneWanted = enabled; applyGates() }
    func setOutputEnabled(_ enabled: Bool) { outputWanted = enabled; applyGates() }

    /// Hardware callback retirement, not remote shutdown or acoustic delivery.
    @discardableResult func fenceAndClose() -> Bool {
        retired = true; active = false; microphoneWanted = false; outputWanted = false; applyGates()
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
        observers.removeAll()
        return device.controlSynchronously(.retire)
    }
    private func applyGates() {
        let eligible = active && !retired && !suspended
        device.setGates(microphone: eligible && microphoneWanted, output: eligible && outputWanted)
    }
    private func observeSession() {
        guard observers.isEmpty else { return }
        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: AVAudioSession.interruptionNotification,
                                            object: nil, queue: .main) { [weak self] notification in
            let raw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            let options = notification.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
            Task { @MainActor [weak self] in self?.interruption(raw: raw, options: options) }
        })
        for name in [AVAudioSession.routeChangeNotification, AVAudioSession.mediaServicesWereResetNotification] {
            let force = name == AVAudioSession.mediaServicesWereResetNotification
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor [weak self] in self?.refreshRoute(force: force) }
            })
        }
    }
    private func interruption(raw: UInt?, options: UInt) {
        guard active, !retired, let raw, let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
        switch type {
        case .began:
            suspended = true; applyGates()
            if !device.controlSynchronously(.interruptionBegan) { fail() }
        case .ended:
            guard suspended else { return }
            guard AVAudioSession.InterruptionOptions(rawValue: options).contains(.shouldResume),
                  device.controlSynchronously(.interruptionEnded) else { fail(); return }
            suspended = false; routeSignature = Self.currentSignature(); applyGates()
        @unknown default: fail()
        }
    }
    private func refreshRoute(force: Bool) {
        guard active, !retired, !suspended else { return }
        let signature = Self.currentSignature()
        // Ignore notifications caused by our own unchanged category/activation.
        guard force || signature != routeSignature else { return }
        device.setGates(microphone: false, output: false)
        guard device.controlSynchronously(.routeChanged) else { fail(); return }
        routeSignature = Self.currentSignature(); applyGates()
    }
    private func fail() { _ = fenceAndClose(); onFailure() }
    private static func currentSignature() -> String {
        let session = AVAudioSession.sharedInstance()
        // Local equality only: never log or publish physical device identifiers.
        return "\(session.sampleRate):\(session.inputNumberOfChannels):\(session.outputNumberOfChannels):"
            + session.currentRoute.inputs.map(\.uid).joined(separator: ",") + ":"
            + session.currentRoute.outputs.map(\.uid).joined(separator: ",")
    }
    private enum RouteError: Error { case retired, unavailable }
}

/// SDK ADM-thread only. Reuses RTCAudioSession's actual shared lock/activation
/// accounting; no separate engine, microphone tap, or forced hardware format.
private final class JarvisPhysicalDuplexLease: @unchecked Sendable {
    private var activationOwned = false
    func acquire(rate: Double, buffer: TimeInterval) throws -> JarvisDuplexAudioDevice.Route {
        let audio = RTCAudioSession.sharedInstance()
        audio.lockForConfiguration(); defer { audio.unlockForConfiguration() }
        audio.useManualAudio = true
        audio.isAudioEnabled = false
        // Route output to Ray-Ban glasses over 48 kHz A2DP instead of HFP so
        // MWDATCamera streaming never mutes the Bluetooth HFP uplink. Real
        // glasses mic audio arrives via MWDAT 1.0.1 audioFramePublisher with
        // seamless built-in mic fallback while the camera stream is offline.
        // The glasses mute A2DP during DAT camera streaming. HFP carries the
        // reply while DAT PCM supplies microphone input when available.
        try audio.setCategory(.playAndRecord, mode: .default,
                              options: [.defaultToSpeaker, .allowBluetooth])
        try? audio.setPreferredSampleRate(rate)
        try? audio.setPreferredIOBufferDuration(buffer)
        let session = AVAudioSession.sharedInstance()
        let headsets = session.availableInputs?.filter { $0.portType == .bluetoothHFP } ?? []
        let wiredInput = session.currentRoute.inputs.contains {
            $0.portType == .headsetMic || $0.portType == .usbAudio
        }
        let headset = session.currentRoute.inputs.first(where: { $0.portType == .bluetoothHFP })
            ?? (headsets.count == 1 ? headsets.first : nil)
        if !wiredInput, let headset {
            try session.setPreferredInput(headset)
        } else if headsets.isEmpty,
                  let builtInMic = session.availableInputs?.first(where: { $0.portType == .builtInMic }),
           !session.currentRoute.inputs.contains(where: { $0.portType == .headsetMic || $0.portType == .usbAudio }) {
            try? session.setPreferredInput(builtInMic)
        }
        if !activationOwned { try audio.setActive(true); activationOwned = true }
        // Native WebRTC reconciles a Bluetooth-selected rate after activation.
        // Leaving a 48 kHz preference on a 16 kHz HFP route can leave Core Audio
        // inconsistent with the active I/O timing (upstream webrtc:6004).
        if session.sampleRate != rate {
            do { try audio.setPreferredSampleRate(session.sampleRate) }
            catch { NSLog("[JarvisDuplex] Active sample-rate reconciliation failed: %@", String(describing: error)) }
        }
        NSLog("[JarvisDuplex] session_rate requested=%f preferred=%f actual=%f buffer=%f",
              rate, session.preferredSampleRate, session.sampleRate, session.ioBufferDuration)
        return .init(identity: UUID().uuidString.lowercased(), sampleRate: session.sampleRate,
                     inputChannels: session.inputNumberOfChannels, outputChannels: session.outputNumberOfChannels,
                     ioBufferDuration: session.ioBufferDuration,
                     inputLatency: session.inputLatency, outputLatency: session.outputLatency)
    }
    func release() {
        guard activationOwned else { return }
        let audio = RTCAudioSession.sharedInstance()
        audio.lockForConfiguration(); defer { audio.unlockForConfiguration() }
        do { try audio.setActive(false); activationOwned = false }
        catch { NSLog("[JarvisDuplex] Audio session deactivation failed") }
    }
}
