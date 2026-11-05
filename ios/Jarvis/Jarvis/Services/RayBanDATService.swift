import AVFAudio
import Combine
import Darwin
import Foundation
import MWDATCamera
import MWDATCore
import os

final class RayBanDATService: ObservableObject {
    @Published private(set) var diagnostics: RayBanDATDiagnostics
    let recordingFrames = JarvisRecordingVideoSource()

    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "rayban-dat")
    private var configuration: RayBanDATConfiguration
    private var wearables: WearablesInterface?
    private var deviceSelector: AutoDeviceSelector?
    private var deviceSession: DeviceSession?
    private var camera: MWDATCamera.Camera?
    private var stream: MWDATCamera.Stream?
    private var registrationTask: Task<Void, Never>?
    private var deviceStreamTask: Task<Void, Never>?
    private var deviceSessionStateTask: Task<Void, Never>?
    private var streamStateListenerToken: AnyListenerToken?
    private var videoFrameListenerToken: AnyListenerToken?
    private var audioFrameListenerToken: AnyListenerToken?
    private var streamErrorListenerToken: AnyListenerToken?
    private var activeFrameCallback: ((VideoFrame) -> Void)?
    private var retryingWithoutAudioCodec = false
    private var isConfigured = false

    init(configuration: RayBanDATConfiguration = .load()) {
        self.configuration = configuration
        self.diagnostics = .initial(configuration: configuration)
    }

#if DEBUG
    /// Read-only probe support; the production service still owns the lifecycle.
    @MainActor
    var captureProbeState: (hasSelectedDevice: Bool, session: String, stream: String, fullyStopped: Bool) {
        (
            deviceSelector?.activeDevice != nil,
            deviceSession.map { String(describing: $0.state) } ?? "absent",
            stream.map { String(describing: $0.state) } ?? "absent",
            stream == nil && (deviceSession == nil || deviceSession?.state == .stopped)
        )
    }
#endif

    deinit {
        registrationTask?.cancel()
        deviceStreamTask?.cancel()
        deviceSessionStateTask?.cancel()
        deviceSession?.stop()
    }

    @MainActor
    func configureIfNeeded() {
        configuration = .load()
        diagnostics.configuration = configuration
        diagnostics.configured = configuration.isConfigured
        diagnostics.updatedAt = Date()
        logger.info("Configuring Ray-Ban DAT. configured=\(self.configuration.isConfigured, privacy: .public)")

        guard configuration.isConfigured else {
            diagnostics.lastError = "Ray-Ban DAT is not configured."
            diagnostics.lastRefreshedAt = Date()
            logger.error("Ray-Ban DAT is not configured")
            return
        }

        guard !isConfigured else {
            refreshDiagnostics()
            return
        }

        do {
            try Wearables.configure()
            wearables = Wearables.shared
            deviceSelector = AutoDeviceSelector(wearables: Wearables.shared)
            isConfigured = true
            diagnostics.sdkAvailable = true
            diagnostics.lastError = nil
            startRegistrationListener()
            startDeviceStreamListener()
            refreshDiagnostics()
            logger.info("Ray-Ban DAT SDK configured")
        } catch {
            diagnostics.sdkAvailable = false
            diagnostics.registrationState = .failed
            diagnostics.lastError = "Failed to configure Meta DAT SDK: \(error.localizedDescription)"
            diagnostics.lastRefreshedAt = Date()
            diagnostics.updatedAt = Date()
            logger.error("Failed to configure Meta DAT SDK: \(error.localizedDescription, privacy: .public)")
        }
    }

    @MainActor
    func handleOpenURL(_ url: URL) -> Bool {
        guard isDATCallback(url) else { return false }
        configureIfNeeded()
        guard isConfigured else { return true }

        Task { [weak self] in
            do {
                _ = try await Wearables.shared.handleUrl(url)
                await self?.refreshDiagnosticsAsync()
            } catch {
                await self?.setError("DAT callback failed: \(error.localizedDescription)")
            }
        }
        return true
    }

    @MainActor
    func refreshDiagnostics() {
        refreshDiagnosticsSnapshot()
        guard wearables != nil else { return }

        Task { [weak self] in
            await self?.refreshPermissionState()
        }
    }

    @MainActor
    func refreshDiagnosticsAsync() async {
        refreshDiagnosticsSnapshot()
        await refreshPermissionState()
    }

    @MainActor
    private func refreshDiagnosticsSnapshot() {
        guard let wearables else {
            diagnostics.lastRefreshedAt = Date()
            diagnostics.updatedAt = Date()
            return
        }

        diagnostics.registrationState = RayBanRegistrationState.fromDATState(wearables.registrationState)
        diagnostics.deviceCount = wearables.devices.count
        diagnostics.deviceState = wearables.devices.isEmpty ? .unavailable : .connected
        if stream == nil {
            diagnostics.streamState = .stopped
        }
        diagnostics.lastError = nil
        diagnostics.lastRefreshedAt = Date()
        diagnostics.updatedAt = Date()
    }

    @MainActor
    func requestRegistrationOrPermission() {
        configureIfNeeded()
        guard let wearables, isConfigured else { return }
        logger.info("Requesting Ray-Ban registration or camera permission")

        Task { [weak self] in
            guard let self else { return }
            do {
                if wearables.registrationState != .registered {
                    self.diagnostics.registrationState = .registering
                    self.diagnostics.lastError = nil
                    self.diagnostics.updatedAt = Date()
                    try await wearables.startRegistration()
                    await self.refreshDiagnosticsAsync()
                    return
                }

                if self.diagnostics.permissionState != .granted {
                    self.diagnostics.permissionState = .requesting
                    self.diagnostics.lastError = nil
                    self.diagnostics.updatedAt = Date()
                    let status = try await wearables.requestPermission(.camera)
                    self.diagnostics.permissionState = RayBanCameraPermissionState.fromDATStatus(status)
                    guard status == .granted else {
                        self.diagnostics.lastRefreshedAt = Date()
                        self.diagnostics.updatedAt = Date()
                        return
                    }
                }

                self.diagnostics.micPermissionState = .requesting
                self.diagnostics.lastError = nil
                self.diagnostics.updatedAt = Date()
                let micStatus = try await wearables.requestPermission(.microphone)
                self.diagnostics.micPermissionState = RayBanCameraPermissionState.fromDATStatus(micStatus)
                self.diagnostics.lastError = nil
                self.diagnostics.lastRefreshedAt = Date()
                self.diagnostics.updatedAt = Date()
            } catch {
                self.logger.error("Ray-Ban registration or permission failed: \(error.localizedDescription, privacy: .public)")
                self.setError("Ray-Ban registration or permission failed: \(error.localizedDescription)")
            }
        }
    }

    @MainActor
    func startGlassesFrameStream(onFrame: @escaping (VideoFrame) -> Void) async throws {
        retryingWithoutAudioCodec = false
        try await startGlassesFrameStreamInternal(allowInStreamAudio: true, onFrame: onFrame)
    }

    @MainActor
    private func startGlassesFrameStreamInternal(allowInStreamAudio: Bool, onFrame: @escaping (VideoFrame) -> Void) async throws {
        configureIfNeeded()
        guard let wearables, isConfigured else {
            throw RayBanDATStreamError.notConfigured
        }
        guard wearables.registrationState == .registered else {
            diagnostics.registrationState = RayBanRegistrationState.fromDATState(wearables.registrationState)
            diagnostics.updatedAt = Date()
            throw RayBanDATStreamError.notRegistered
        }

        let status = try await wearables.checkPermissionStatus(.camera)
        diagnostics.permissionState = RayBanCameraPermissionState.fromDATStatus(status)
        let micStatus = (try? await wearables.checkPermissionStatus(.microphone)) ?? .denied
        diagnostics.micPermissionState = RayBanCameraPermissionState.fromDATStatus(micStatus)
        diagnostics.updatedAt = Date()
        guard status == .granted else {
            throw RayBanDATStreamError.cameraPermissionDenied
        }

        guard !wearables.devices.isEmpty else {
            diagnostics.deviceState = .unavailable
            diagnostics.deviceCount = 0
            diagnostics.updatedAt = Date()
            throw RayBanDATStreamError.noDevice
        }

        await stopGlassesFrameStream()

        activeFrameCallback = onFrame
        diagnostics.streamState = .starting
        diagnostics.lastError = nil
        diagnostics.updatedAt = Date()
        let enableAudioCodec = allowInStreamAudio && micStatus == .granted
        logger.info("Starting Ray-Ban frame stream (inStreamAudio=\(enableAudioCodec, privacy: .public))")

        if deviceSession?.state == .stopping {
            throw NSError(
                domain: "JarvisRayBanSession",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "The previous glasses session is still stopping. Please try again shortly."]
            )
        }
        let session = try await startedDeviceSession()
        guard session.state == .started else {
            diagnostics.streamState = .failed
            diagnostics.updatedAt = Date()
            throw RayBanDATStreamError.sessionNotReady
        }

        let streamConfiguration = StreamConfiguration(
            videoCodec: VideoCodec.raw,
            audioCodec: enableAudioCodec ? .pcm(sampleRate: .rate16000, numberOfChannels: 1) : nil,
            resolution: StreamingResolution.medium,
            frameRate: 24
        )

        let createdCamera: MWDATCamera.Camera
        do {
            guard let addedCamera = try session.addCamera(config: streamConfiguration) else {
                diagnostics.streamState = .failed
                diagnostics.lastError = "Ray-Ban camera creation returned no camera."
                diagnostics.updatedAt = Date()
                throw RayBanDATStreamError.streamCreationFailed("No camera was returned by the Meta DAT SDK.")
            }
            createdCamera = addedCamera
        } catch {
            diagnostics.streamState = .failed
            diagnostics.lastError = "Ray-Ban stream creation failed: \(error.localizedDescription)"
            diagnostics.updatedAt = Date()
            throw RayBanDATStreamError.streamCreationFailed(error.localizedDescription)
        }

        let newStream = createdCamera.stream
        camera = createdCamera
        stream = newStream
        setupStreamListeners(for: newStream, inStreamAudioEnabled: enableAudioCodec, onFrame: onFrame)
        newStream.start()
        logger.info("Ray-Ban frame stream start requested")
    }

    @MainActor
    func stopGlassesFrameStream() async {
        activeFrameCallback = nil
        JarvisDuplexAudioDevice.clearActiveExternalMicrophoneSamples()
        guard stream != nil || camera != nil else {
            clearStreamListeners()
            diagnostics.streamState = .stopped
            diagnostics.updatedAt = Date()
            return
        }
        let activeStream = stream
        let activeCamera = camera
        stream = nil
        camera = nil
        clearStreamListeners()
        diagnostics.streamState = .stopped
        diagnostics.updatedAt = Date()
        let endingSession = deviceSession
        activeStream?.stop()
        activeCamera?.stop()
        guard let endingSession, deviceSession === endingSession else { return }

        // A Jarvis stop ends the glasses session, not just its preview stream.
        let endingStates = endingSession.stateStream()
        endingSession.stop()
        let ended = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                if endingSession.state == .stopped { return true }
                for await state in endingStates {
                    if state == .stopped { return true }
                    if Task.isCancelled { return false }
                }
                return endingSession.state == .stopped
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                return false
            }
            let stopped = await group.next() ?? false
            group.cancelAll()
            return stopped
        }
        if ended, deviceSession === endingSession {
            deviceSession = nil
            NSLog("[camera-audio-lifecycle] Glasses session fully stopped")
        } else if !ended {
            NSLog("[camera-audio-lifecycle] Glasses session stop not confirmed; retaining its identity")
        }
        logger.info("Stopped Ray-Ban frame stream")
    }

    @MainActor
    func recordGlassesFrame(_ frame: CapturedVisualFrame) {
        diagnostics.lastFrameReceivedAt = frame.capturedAt
        diagnostics.lastFrameSequence = frame.sequence
        diagnostics.lastFrameSizeDescription = frame.sizeDescription
        diagnostics.updatedAt = Date()
    }

    @MainActor
    func recordGlassesStreamError(_ message: String, streamFailed: Bool = true) {
        if streamFailed {
            diagnostics.streamState = .failed
        }
        diagnostics.lastError = message
        diagnostics.updatedAt = Date()
    }

    @MainActor
    private func refreshPermissionState() async {
        guard let wearables, isConfigured else { return }
        guard wearables.registrationState == .registered else {
            diagnostics.permissionState = .notRequested
            diagnostics.micPermissionState = .notRequested
            diagnostics.lastRefreshedAt = Date()
            diagnostics.updatedAt = Date()
            return
        }

        do {
            let status = try await wearables.checkPermissionStatus(.camera)
            diagnostics.permissionState = RayBanCameraPermissionState.fromDATStatus(status)
            if let micStatus = try? await wearables.checkPermissionStatus(.microphone) {
                diagnostics.micPermissionState = RayBanCameraPermissionState.fromDATStatus(micStatus)
            }
            diagnostics.lastError = nil
            diagnostics.lastRefreshedAt = Date()
            diagnostics.updatedAt = Date()
        } catch {
            let message = "Ray-Ban camera permission check failed: \(error.localizedDescription)"
            logger.error("\(message, privacy: .public)")

            // DAT can finish an older permission query after a newer query has
            // already confirmed access. Keep the useful console diagnostic, but
            // do not replace an authoritative granted state with a stale error.
            guard diagnostics.permissionState != .granted else {
                if diagnostics.lastError?.hasPrefix("Ray-Ban camera permission check failed:") == true {
                    diagnostics.lastError = nil
                }
                diagnostics.lastRefreshedAt = Date()
                diagnostics.updatedAt = Date()
                return
            }

            setError(message)
        }
    }

    @MainActor
    private func startRegistrationListener() {
        guard let wearables else { return }
        registrationTask?.cancel()
        registrationTask = Task { [weak self] in
            for await state in wearables.registrationStateStream() {
                await MainActor.run {
                    self?.diagnostics.registrationState = RayBanRegistrationState.fromDATState(state)
                    self?.diagnostics.lastError = nil
                    self?.diagnostics.lastRefreshedAt = Date()
                    self?.diagnostics.updatedAt = Date()
                }
            }
        }
    }

    @MainActor
    private func startDeviceStreamListener() {
        guard let wearables else { return }
        deviceStreamTask?.cancel()
        deviceStreamTask = Task { [weak self] in
            for await devices in wearables.devicesStream() {
                await MainActor.run {
                    self?.diagnostics.deviceCount = devices.count
                    self?.diagnostics.deviceState = devices.isEmpty ? .unavailable : .connected
                    self?.diagnostics.lastRefreshedAt = Date()
                    self?.diagnostics.updatedAt = Date()
                }
            }
        }
    }

    @MainActor
    private func startedDeviceSession() async throws -> DeviceSession {
        if let session = deviceSession, session.state == .started {
            diagnostics.deviceState = .connected
            diagnostics.updatedAt = Date()
            startDeviceSessionStateObserver(for: session)
            return session
        }

        if deviceSession?.state == .stopped {
            deviceSession = nil
        }

        if let session = deviceSession {
            try await waitForDeviceSessionStart(
                stateStream: session.stateStream(),
                errorStream: session.errorStream()
            )
            diagnostics.deviceState = .connected
            diagnostics.updatedAt = Date()
            startDeviceSessionStateObserver(for: session)
            return session
        }

        guard let wearables, let deviceSelector else {
            throw RayBanDATStreamError.notConfigured
        }

        do {
            let session = try wearables.createSession(deviceSelector: deviceSelector)
            deviceSession = session
            diagnostics.deviceState = .connecting
            diagnostics.updatedAt = Date()

            let stateStream = session.stateStream()
            let errorStream = session.errorStream()
            try session.start()

            if session.state == .started {
                diagnostics.deviceState = .connected
                diagnostics.updatedAt = Date()
                startDeviceSessionStateObserver(for: session)
                return session
            }

            try await waitForDeviceSessionStart(stateStream: stateStream, errorStream: errorStream)
            diagnostics.deviceState = .connected
            diagnostics.updatedAt = Date()
            startDeviceSessionStateObserver(for: session)
            return session
        } catch {
            deviceSession = nil
            diagnostics.deviceState = .failed
            diagnostics.lastError = "Ray-Ban device session failed: \(error.localizedDescription)"
            diagnostics.updatedAt = Date()
            throw error
        }
    }

    private func waitForDeviceSessionStart(
        stateStream: AsyncStream<DeviceSessionState>,
        errorStream: AsyncStream<DeviceSessionError>
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                for await state in stateStream {
                    if state == .started {
                        return
                    }
                    if state == .stopped {
                        throw RayBanDATStreamError.sessionStartFailed
                    }
                }
                guard !Task.isCancelled else { return }
                throw RayBanDATStreamError.sessionStartFailed
            }

            group.addTask {
                for await error in errorStream {
                    throw error
                }
                guard !Task.isCancelled else { return }
                throw RayBanDATStreamError.sessionStartFailed
            }

            guard try await group.next() != nil else {
                throw RayBanDATStreamError.sessionStartFailed
            }
            group.cancelAll()
        }
    }

    @MainActor
    private func startDeviceSessionStateObserver(for session: DeviceSession) {
        deviceSessionStateTask?.cancel()
        deviceSessionStateTask = Task { [weak self] in
            for await state in session.stateStream() {
                await MainActor.run {
                    switch state {
                    case .started:
                        self?.diagnostics.deviceState = .connected
                    case .stopped:
                        let deviceCount = self?.diagnostics.deviceCount ?? 0
                        self?.diagnostics.deviceState = deviceCount > 0 ? .disconnected : .unavailable
                        self?.deviceSession = nil
                        self?.deviceSessionStateTask = nil
                    default:
                        self?.diagnostics.deviceState = .unknown
                    }
                    self?.diagnostics.updatedAt = Date()
                }
            }
        }
    }

    @MainActor
    private func setupStreamListeners(
        for stream: MWDATCamera.Stream,
        inStreamAudioEnabled: Bool,
        onFrame: @escaping (VideoFrame) -> Void
    ) {
        clearStreamListeners()
        let recordingSource = recordingFrames
        let recordingGeneration = recordingSource.beginGeneration()

        streamStateListenerToken = stream.statePublisher.listen { [weak self] state in
            Task { @MainActor in
                self?.handleStreamStateChange(state)
            }
        }

        videoFrameListenerToken = stream.videoFramePublisher.listen { frame in
            let arrival = mach_absolute_time()
            recordingSource.offer(frame.sampleBuffer, sourceGeneration: recordingGeneration, receivedHostTime: arrival)
            Task { @MainActor in
                onFrame(frame)
            }
        }

        if inStreamAudioEnabled {
            audioFrameListenerToken = stream.audioFramePublisher.listen { audioFrame in
                Self.forwardDATGlassesAudioFrame(audioFrame)
            }
        }

        streamErrorListenerToken = stream.errorPublisher.listen { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                if inStreamAudioEnabled && !self.retryingWithoutAudioCodec
                    && (error == .audioStreamingError || error == .permissionDenied) {
                    self.retryingWithoutAudioCodec = true
                    self.logger.warning("In-stream glasses audio failed (\(error.localizedDescription, privacy: .public)); falling back to video-only + phone mic")
                    if let callback = self.activeFrameCallback {
                        try? await self.startGlassesFrameStreamInternal(allowInStreamAudio: false, onFrame: callback)
                        return
                    }
                }
                self.recordGlassesStreamError("Ray-Ban stream failed: \(error.localizedDescription)")
            }
        }
    }

    private nonisolated static func forwardDATGlassesAudioFrame(_ audioFrame: AudioFrame) {
        let pcmBuffer = audioFrame.pcmBuffer
        let frameLength = Int(pcmBuffer.frameLength)
        let sampleRate = pcmBuffer.format.sampleRate
        guard frameLength > 0, sampleRate >= 8_000 else { return }

        if let int16Channels = pcmBuffer.int16ChannelData {
            JarvisDuplexAudioDevice.pushActiveExternalMicrophoneSamples(
                int16Channels[0],
                count: UInt32(frameLength),
                sampleRate: sampleRate
            )
            return
        }

        if let floatChannels = pcmBuffer.floatChannelData {
            let src = floatChannels[0]
            var converted = [Int16](repeating: 0, count: frameLength)
            for i in 0..<frameLength {
                let clamped = max(-1.0, min(1.0, src[i]))
                converted[i] = Int16((clamped * 32767.0).rounded())
            }
            converted.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                JarvisDuplexAudioDevice.pushActiveExternalMicrophoneSamples(
                    base,
                    count: UInt32(frameLength),
                    sampleRate: sampleRate
                )
            }
            return
        }

        let audioBuffer = pcmBuffer.audioBufferList.pointee.mBuffers
        let channels = max(1, Int(audioBuffer.mNumberChannels))
        if pcmBuffer.format.streamDescription.pointee.mBitsPerChannel == 16,
           let raw = audioBuffer.mData?.assumingMemoryBound(to: Int16.self) {
            if channels == 1 {
                JarvisDuplexAudioDevice.pushActiveExternalMicrophoneSamples(
                    raw,
                    count: UInt32(frameLength),
                    sampleRate: sampleRate
                )
            } else {
                var mono = [Int16](repeating: 0, count: frameLength)
                for i in 0..<frameLength {
                    mono[i] = raw[i * channels]
                }
                mono.withUnsafeBufferPointer { ptr in
                    guard let base = ptr.baseAddress else { return }
                    JarvisDuplexAudioDevice.pushActiveExternalMicrophoneSamples(
                        base,
                        count: UInt32(frameLength),
                        sampleRate: sampleRate
                    )
                }
            }
        }
    }

    @MainActor
    private func clearStreamListeners() {
        recordingFrames.endGeneration()
        streamStateListenerToken = nil
        videoFrameListenerToken = nil
        audioFrameListenerToken = nil
        streamErrorListenerToken = nil
    }

    @MainActor
    private func handleStreamStateChange(_ state: StreamState) {
        switch state {
        case .stopped:
            diagnostics.streamState = .stopped
        case .waitingForDevice, .starting, .stopping:
            diagnostics.streamState = .starting
        case .paused:
            diagnostics.streamState = .paused
        case .streaming:
            diagnostics.streamState = .active
        default:
            diagnostics.streamState = .failed
        }
        diagnostics.lastRefreshedAt = Date()
        diagnostics.updatedAt = Date()
    }

    private func isDATCallback(_ url: URL) -> Bool {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return false
        }
        if components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true {
            return true
        }
        let expectedScheme = configuration.appLinkURLScheme?.replacingOccurrences(of: "://", with: "")
        return expectedScheme != nil && url.scheme == expectedScheme
    }

    @MainActor
    private func setError(_ message: String) {
        diagnostics.lastError = message
        diagnostics.updatedAt = Date()
        logger.error("\(message, privacy: .public)")
    }
}

enum RayBanDATStreamError: LocalizedError {
    case notConfigured
    case notRegistered
    case cameraPermissionDenied
    case noDevice
    case sessionNotReady
    case sessionStartFailed
    case streamCreationFailed(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "Ray-Ban DAT is not configured."
        case .notRegistered:
            return "Ray-Ban DAT is not registered. Register/request permission first."
        case .cameraPermissionDenied:
            return "Ray-Ban camera permission is not granted."
        case .noDevice:
            return "No Ray-Ban device is connected."
        case .sessionNotReady:
            return "Ray-Ban device session is not ready."
        case .sessionStartFailed:
            return "Ray-Ban device session failed to start."
        case .streamCreationFailed(let message):
            return "Ray-Ban stream creation failed: \(message)"
        }
    }
}
