import AVFoundation
import Combine
import Foundation
import os

@MainActor
final class JarvisAudioSessionViewModel: ObservableObject {
    var onReminderStateChanged: (() -> Void)?
    var onMemoryStateChanged: ((JarvisMemoryRefreshHint) -> Void)?
    var onWorkingMemoryStateChanged: ((JarvisWorkingMemoryRefreshHint) -> Void)?
    var onRealtimeConnectionStateChanged: ((JarvisAudioConnectionState) -> Void)?
    var onWorkspacePresentationIntent: ((UUID, UUID) -> Void)?
    var onAppLinkRequested: ((JarvisAppLinkRequest) -> Void)?
    var onSandboxViewStopped: ((UUID, UUID) -> Void)?
    var onRecordingSaved: ((JarvisRecordingAppController.Saved) -> Void)?
    var onBehaviorProfileCommitted: ((UUID, JarvisBehaviorProfileCommitted) -> Void)?
    @Published var captureMode: JarvisAudioCaptureMode = .autoDetect {
        didSet { recordStartupInputPredicates() }
    }
    @Published private(set) var connectionState: JarvisAudioConnectionState = .disconnected
    @Published private(set) var turnState: JarvisAudioTurnState = .idle
    @Published private(set) var voiceRouteFailure: JarvisVoiceRouteFailure?
    @Published private(set) var transcriptPreview: String = "Start a Jarvis session, then speak through your glasses."
    @Published private(set) var visualStatus: String = "Glasses vision standby."
    @Published private(set) var visualBackendStatus: String = "Backend visual memory disconnected."
    @Published private(set) var visualFrameAcceptedCount = 0
    @Published private(set) var visualFrameDroppedCount = 0
    @Published private(set) var visualStreamState: JarvisVisualStreamState = .stopped
    @Published private(set) var latestVisualFrame: CapturedVisualFrame?
    @Published private(set) var rayBanDiagnostics: RayBanDATDiagnostics
    @Published private(set) var activeSession: JarvisSession?
    @Published private(set) var audioSessionId: String?
    @Published private(set) var routeSummary: String = "Audio route unknown"
    @Published private(set) var routeUsesGlasses = false
    @Published private(set) var diagnostics: [String] = []
    @Published private(set) var isManualRecording = false {
        didSet { recordStartupInputPredicates() }
    }
    @Published private(set) var speakingReminderMessage: String?
    @Published private(set) var speakingReminderEventId: String?
    @Published private(set) var preferredAudioProvider: JarvisAudioProvider = .openai
    @Published private(set) var providerRun: JarvisAudioProviderRun?
    @Published private(set) var preferredVoiceRoute = JarvisVoiceRouteChoice.load(from: .standard)
    @Published private(set) var recordingStatus = "Recording is not active."
    @Published private(set) var isRecording = false
    @Published private(set) var recordingPhase: JarvisRecordingSession.State?
    @Published private(set) var recordingCommandInFlight = false
    @Published private(set) var recordingCommandMessage: String?
    private var recordingID: UUID?
    private var recordingProtocolVersion = 1
    @Published private(set) var mediaStatus = "Music is not active."
    private var mediaID: UUID?
    private var mediaTask: Task<Void, Never>?

    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "audio-session")
    private let realtimeClient = JarvisRealtimeClient()
    private let directAudioClient = JarvisWebRTCClient()
    private let visualIngestClient = JarvisVisualIngestClient()
    private let captureService = AudioCaptureService()
    private let playbackService = AudioPlaybackService()
    private let segmenter = AudioSegmenter()
    private let routeDiagnostics = AudioRouteDiagnostics()
    private let rayBanDATService: RayBanDATService
    private let visualCoordinator: VisualFrameCaptureCoordinator
    private lazy var recordings = JarvisRecordingAppController(write: { [weak self] event, bound in
        guard let self else { throw CancellationError() }
        try await self.realtimeClient.sendRecordingEvent(event, connectionID: bound.connectionID,
            current: { [weak self] in
                guard let self else { return false }
                return self.isSessionReady && !self.isStopping
                    && self.realtimeClient.connectionID == bound.connectionID
                    && self.audioSessionId == bound.audioSessionID
                    && self.activeSession.flatMap { UUID(uuidString: $0.id) } == bound.sessionID
            })
    }, changed: { [weak self] status in
        self?.handleRecordingStatus(status)
    }, saved: { [weak self] saved in
        self?.onRecordingSaved?(saved)
    })

#if DEBUG
    @Published private(set) var replayRunID: UUID?
    private(set) var lastReplayAudioEvidence: JarvisReplayAudioEvidence?
    private(set) var lastReplayFrameLineage: [String: Any]?
    private var replayFrameLineageLedger: JarvisReplayFrameLineageLedger?
    private var replayStartPending = false
    private var replayCleanupFailed = false
    private var replayAudioDevice: JarvisReplayAudioDevice?
    private var replayCapture: JarvisReplayCaptureSource?
    private var replayVoiceRoute: JarvisVoiceRouteChoice?
    private var replayPreviousConfiguration: JarvisConfiguration?
    private var replayPreviousCaptureMode: JarvisAudioCaptureMode?
    private var replayWatchdog: Task<Void, Never>?

    func startReplay(plan: JarvisReplayPlan, route: JarvisVoiceRouteChoice,
                     evaluationConfiguration: JarvisConfiguration,
                     expectedScope: JarvisReplayScopeExpectation,
                     taskConfiguration: JarvisReplayTaskConfiguration? = nil,
                      launchManifestSHA256: String? = nil, manualInput: Bool = false,
                      memoryConfiguration: JarvisReplayMemoryConfiguration? = nil) async throws {
        guard !shouldShowStopJarvis, !replayStartPending, replayRunID == nil, !replayCleanupFailed,
              expectedScope.runID == plan.runID else { throw JarvisReplayError.retired }
        _ = try JarvisReplayLaunchPlan.validateManualInput(manualInput ? "manual" : nil, route: route,
            invocationID: plan.invocationID, taskMode: taskConfiguration != nil, utterances: plan.utterances,
            memoryReadiness: memoryConfiguration != nil)
        if let taskConfiguration {
            guard route == .realtime, taskConfiguration.scope.runID == plan.runID,
                  taskConfiguration.scope.ownerID == expectedScope.ownerID else { throw JarvisReplayError.invalidPlan }
        }
        replayStartPending = true
        defer { replayStartPending = false }
        _ = try await JarvisReplayScope.authorize(configuration: evaluationConfiguration,
            expected: expectedScope, remaining: plan.recordingCapacityDuration)
        try Task.checkCancellation()
        guard replayStartPending, !shouldShowStopJarvis else { throw CancellationError() }
        let audio = JarvisReplayAudioDevice(plan: plan)
        let source: JarvisReplayCaptureSource
        do { source = try JarvisReplayCaptureSource(plan: plan, audio: audio,
            taskConfiguration: taskConfiguration, launchManifestSHA256: launchManifestSHA256, manualInput: manualInput,
            memoryConfiguration: memoryConfiguration) }
        catch { audio.close(); throw error }
        do { try directAudioClient.useReplayAudioDevice(route == .realtime ? audio : nil) }
        catch { audio.close(); throw error }
        replayPreviousConfiguration = configuration
        replayPreviousCaptureMode = captureMode
        configuration = evaluationConfiguration
        apiClient = JarvisAPIClient(configuration: evaluationConfiguration)
        apiClient?.evaluationRunID = plan.runID
        realtimeClient.evaluationRunID = plan.runID
        visualIngestClient.evaluationRunID = plan.runID
        replayRunID = plan.runID
        lastReplayAudioEvidence = nil
        lastReplayFrameLineage = nil
        replayFrameLineageLedger = JarvisReplayFrameLineageLedger(runID: plan.runID,
            launchManifestSHA256: launchManifestSHA256)
        visualIngestClient.onReplayFrameSocketWritten = { [weak self] receipt in
            guard self?.replayRunID == receipt.lineage.runID else { return }
            self?.replayFrameLineageLedger?.append(receipt)
        }
        replayAudioDevice = audio; replayCapture = source; replayVoiceRoute = route
        captureMode = manualInput ? .holdToTalk : .autoDetect
        if manualInput {
            source.onManualScopeIsCurrent = { [weak self] identity in
                guard let self else { return false }
                return self.replayRunID == plan.runID && self.replayVoiceRoute == .realtime
                    && !self.isStopping && self.isSessionReady && self.captureMode == .holdToTalk
                    && identity.ownerID == expectedScope.ownerID
                    && self.activeSession?.id.lowercased() == identity.sessionID.uuidString.lowercased()
                    && self.audioSessionId?.lowercased() == identity.audioSessionID.uuidString.lowercased()
                    && self.realtimeClient.connectionID == identity.controlConnectionID
            }
            source.onManualBegin = { [weak self] in
                guard let self, self.replayRunID == plan.runID, self.replayVoiceRoute == .realtime,
                      self.isSessionReady, !self.isManualRecording, self.captureMode == .holdToTalk,
                      let session = self.activeSession, let audioSession = self.audioSessionId else {
                    throw JarvisReplayError.manualInputIncomplete
                }
                self.beginManualSpeech()
                guard self.isManualRecording, let segment = self.activeSegmentId else {
                    throw JarvisReplayError.manualInputIncomplete
                }
                return try JarvisReplayManualSessionIdentity(ownerID: expectedScope.ownerID,
                    sessionID: session.id, audioSessionID: audioSession,
                    controlConnectionID: self.realtimeClient.connectionID, segmentID: segment)
            }
            source.onManualRelease = { [weak self] identity in
                guard let self, self.replayRunID == plan.runID, self.isManualRecording,
                      self.captureMode == .holdToTalk, self.replayVoiceRoute == .realtime,
                      self.activeSession?.id.lowercased() == identity.sessionID.uuidString.lowercased(),
                      self.audioSessionId?.lowercased() == identity.audioSessionID.uuidString.lowercased(),
                      self.activeSegmentId.flatMap(UUID.init(uuidString:)) == identity.segmentID,
                      self.realtimeClient.connectionID == identity.controlConnectionID else {
                    throw JarvisReplayError.manualIdentityMismatch
                }
                self.endManualSpeech()
            }
            realtimeClient.onReplayManualWrite = { [weak source] value in source?.manualWrite(value) }
        }
        source.onFailure = { [weak self] _ in self?.failSession("Evaluation replay capture failed.") }
        source.onFinished = { [weak self] in self?.stopJarvis() }
        source.onMediaEnded = { [weak self] in
            self?.visualStatus = "Replay media ended; bounded output capture only. No new frames."
        }
        replayStartPending = false
        replayWatchdog = Task { @MainActor [weak self] in
            do { try await Task.sleep(for: .seconds(plan.recordingCapacityDuration)) } catch { return }
            guard self?.replayRunID == plan.runID else { return }
            self?.failSession("Evaluation startup/run deadline exceeded.")
        }
        startJarvis(startupOrigin: .replayAutostart)
    }

    private func finishReplay() {
        guard replayRunID != nil else { return }
        replayWatchdog?.cancel(); replayWatchdog = nil
        replayCapture?.stop()
        realtimeClient.onReplayManualWrite = nil
        guard replayAudioDevice?.close() != false else {
            replayCleanupFailed = true
            appendDiagnostic("Replay callback fence unconfirmed; new sessions blocked.")
            return
        }
        lastReplayAudioEvidence = replayAudioDevice?.evidence
        lastReplayFrameLineage = replayFrameLineageLedger?.json
        replayFrameLineageLedger = nil
        visualIngestClient.onReplayFrameSocketWritten = nil
        do { try directAudioClient.useReplayAudioDevice(nil) }
        catch { replayCleanupFailed = true; return }
        replayCapture = nil; replayAudioDevice = nil
        realtimeClient.evaluationRunID = nil; visualIngestClient.evaluationRunID = nil
        configuration = replayPreviousConfiguration
        apiClient = configuration.map { JarvisAPIClient(configuration: $0) }
        if let replayPreviousCaptureMode { captureMode = replayPreviousCaptureMode }
        replayPreviousConfiguration = nil; replayPreviousCaptureMode = nil
        replayVoiceRoute = nil; replayRunID = nil
    }
#endif
    private var configuration: JarvisConfiguration?
    private var apiClient: JarvisAPIClient?
    private var activeRealtimeConfig: JarvisRealtimeConfig?
    private var visualSessionId: String?
    private var visualBackendUploadsDisabled = false
    private var speechSegment = JarvisSpeechSegmentState()
    private var activeSegmentId: String? { speechSegment.id }
    private var activeSegmentStartedAt: Date? { speechSegment.startedAt }
    private var hasAssistantResponseInProgress = false
    private var isSessionReady = false {
        didSet {
            recordStartupInputPredicates()
            if isSessionReady { sendResourceContext() }
            else { resourceContextWrite?.cancel(); resourceContextWrite = nil }
        }
    }
    private var resourceSelectionContext: JarvisResourceSelectionContext?
    private var resourceSelectionSessionID: UUID?
    private var resourceContextConnectionID: UUID?
    private var resourceContextSequence = 0
    private var resourceContextRevision = 0
    private var resourceContextWrite: Task<Void, Never>?
    private var startupDiagnostics: JarvisStartupDiagnostics?
    private var isStopping = false
    private var isUnifiedJarvisSession = false
    private var usesDirectAudio = false
    private var voiceRouteEnabled = false
    private var usesLeanVoice = false
    private var voiceRouteDriver: JarvisVoiceRouteDriver?
    private var liveAppRoute: JarvisLiveAppRoute?
    private var activeVoiceRouteChoice: JarvisVoiceRouteChoice = .realtime
    private var usesLiveAudio: Bool { activeVoiceRouteChoice == .live }
    private var voiceRouteTimeout: Task<Void, Never>?
    private var voiceControlTasks: [UUID: Task<Void, Never>] = [:]
    private var pendingDirectReadyEvent: JarvisRealtimeServerEvent?
    private var sessionStartTask: Task<Void, Never>?
    private var contextReconnectTask: Task<Void, Never>?
    private var contextReconnectBudget = JarvisContextReconnectBudget()
    private var isContextReconnecting = false
    private var pendingContextReset: JarvisContextResetNotice?
    private var sessionStopTask: Task<Void, Never>?
    private var visualStartTask: Task<Void, Never>?
    private var visualStreamGeneration: UUID?
    private let audioSendQueue = JarvisAudioSendQueue()
    private var cancellables = Set<AnyCancellable>()
    private var localVisualFrameCount = 0

    init(rayBanDATService: RayBanDATService = RayBanDATService()) {
        self.rayBanDATService = rayBanDATService
        self.visualCoordinator = VisualFrameCaptureCoordinator(
            glassesSource: GlassesFrameCaptureService(datService: rayBanDATService)
        )
        self.rayBanDiagnostics = rayBanDATService.diagnostics

        captureService.onAudioChunk = { [weak self] chunk in
            Task { @MainActor in
                self?.handleCapturedAudio(chunk)
            }
        }
        playbackService.onPlaybackStarted = { [weak self] in
            Task { @MainActor in
                self?.directAudioClient.setExternalPlaybackActive(true)
                self?.turnState = .speaking
            }
        }
        playbackService.onPlaybackStopped = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.directAudioClient.setExternalPlaybackActive(false)
                if !self.hasAssistantResponseInProgress && !self.directAudioClient.isPlaying {
                    self.turnState = .idle
                }
            }
        }
        playbackService.onPlaybackReceipt = { [weak self] receipt in
            guard let self, !self.usesLiveAudio, let runID = self.providerRun?.id, let audioID = self.audioSessionId else { return }
            if let driver = self.voiceRouteDriver {
                for message in driver.playbackMessages(receipt) { self.sendVoiceControl(message, driver: driver) }
                return
            }
            let connectionID = self.realtimeClient.connectionID
            self.enqueueAudioSend {
                guard self.providerRun?.id == runID, self.audioSessionId == audioID else { return }
                _ = try await self.realtimeClient.sendPlaybackReceipt(receipt, audioSessionID: audioID,
                                                                      connectionID: connectionID)
            }
        }
        realtimeClient.onEvent = { [weak self] event in
            self?.handleServerEvent(event)
        }
        realtimeClient.onSocketOpened = { [weak self] connection, observedAt in
            guard let self else { return }
            self.recordStartup(self.startupDiagnostics?.observedOpen(connection: connection, at: observedAt))
        }
        directAudioClient.onMicrophoneEnabledChanged = { [weak self] enabled in
            guard let self else { return }
            self.recordStartup(self.startupDiagnostics?.predicate(.nativeMicrophoneEnabled,
                enabled: enabled, at: ProcessInfo.processInfo.systemUptime))
        }
        realtimeClient.onDisconnect = { [weak self] error in
            guard let self, !self.isStopping else { return }
            self.failSession("Realtime disconnected: \(error?.localizedDescription ?? "unknown")")
        }
        directAudioClient.onSpeechStarted = { [weak self] in
            guard let self, !self.usesLiveAudio, self.usesDirectAudio, self.isSessionReady else { return }
            // The native client has already fenced local playback, and the
            // backend's matching provider event owns the new user turn. Sending
            // a second route cancel here can settle that turn before its audio
            // commit arrives. Explicit user stop controls still use the driver.
            self.playbackService.stop()
            self.hasAssistantResponseInProgress = false
            self.turnState = .listening
        }
        directAudioClient.onSpeechStopped = { [weak self] in
            guard let self, !self.usesLiveAudio, self.usesDirectAudio, self.isSessionReady else { return }
            self.turnState = .processingInput
        }
        directAudioClient.onPlaybackChanged = { [weak self] playing in
            guard let self, !self.usesLiveAudio, self.usesDirectAudio, self.isSessionReady else { return }
            if playing { self.turnState = .speaking }
            else if !self.playbackService.isPlaying && !self.hasAssistantResponseInProgress { self.turnState = .idle }
        }
        directAudioClient.onNativeResponseChanged = { [weak self] active in
            guard let self, self.usesLeanVoice, self.isSessionReady else { return }
            self.hasAssistantResponseInProgress = active
            if !active && !self.directAudioClient.isPlaying { self.turnState = .idle }
        }
        directAudioClient.onFailure = { [weak self] message in
            guard let self, !self.isStopping, !self.usesLiveAudio, self.usesDirectAudio else { return }
            self.failSession(message)
        }
        visualIngestClient.onEvent = { [weak self] event in
            self?.handleVisualServerEvent(event)
        }
        visualIngestClient.onDisconnect = { [weak self] error in
            guard let self, !self.isStopping else { return }
            self.visualBackendStatus = "Backend visual memory disconnected."
            self.appendDiagnostic("Visual websocket disconnected: \(error?.localizedDescription ?? "unknown")")
        }
        visualCoordinator.onFrame = { [weak self] frame, source in
            self?.handleCapturedVisualFrame(frame, source: source)
        }
        rayBanDATService.$diagnostics
            .sink { [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    // A queued notification may belong to a retired stream.
                    let diagnostics = self.rayBanDATService.diagnostics
                    self.rayBanDiagnostics = diagnostics
                    self.syncVisualState(with: diagnostics)
                }
            }
            .store(in: &cancellables)
        observeRouteChanges()
    }

    var isConnected: Bool {
        connectionState.isConnected
    }

    var isBusy: Bool {
        connectionState == .connecting
    }

    var isJarvisActive: Bool {
        // Voice remains usable when glasses vision is unavailable.
        isConnected
    }

    var isJarvisStarting: Bool {
        isBusy || visualStreamState.isStarting
    }

    var shouldShowStopJarvis: Bool {
        sessionStopTask != nil || connectionState != .disconnected || visualStreamState != .stopped
    }

    var visualBackendSummary: String {
        "\(visualBackendStatus) · accepted \(visualFrameAcceptedCount) · dropped \(visualFrameDroppedCount)"
    }

    var primaryStatus: String {
        switch connectionState {
        case .connected:
            return turnState.label
        case .connecting:
            return "Connecting"
        case .disconnected:
            return "Ready"
        case .failed:
            return "Needs attention"
        }
    }

    func configure(configuration: JarvisConfiguration) {
#if DEBUG
        guard replayRunID == nil, !replayStartPending, !replayCleanupFailed else { return }
#endif
        if self.configuration != configuration { recordings.retire(); isRecording = false }
        self.configuration = configuration
        self.apiClient = JarvisAPIClient(configuration: configuration)
    }

#if DEBUG && targetEnvironment(simulator)
    /// Prerecorded input for an explicitly authorized, ordinary-account native test.
    /// This does not create or authorize an isolated evaluation session.
    func useServingTestAudioDevice(_ device: JarvisReplayAudioDevice?) throws {
        guard !shouldShowStopJarvis, !isSessionReady, sessionStartTask == nil,
              sessionStopTask == nil, replayRunID == nil, !replayStartPending,
              !replayCleanupFailed, replayCapture == nil else {
            throw JarvisReplayError.invalidPlan
        }
        if let device {
            guard replayAudioDevice == nil, !device.isRetired else {
                throw JarvisReplayError.invalidPlan
            }
            replayAudioDevice = device
        } else {
            guard replayAudioDevice?.close() != false else { throw JarvisReplayError.invalidPlan }
            replayAudioDevice = nil
        }
    }
#endif

    func setPreferredAudioProvider(_ provider: JarvisAudioProvider) {
        preferredAudioProvider = provider.supportedSelection
    }

    func setPreferredVoiceRoute(_ choice: JarvisVoiceRouteChoice) {
        guard !shouldShowStopJarvis else { return }
        preferredVoiceRoute = choice
        choice.save(to: .standard)
    }

    var voiceStatusText: String {
        JarvisVoiceRouteFailure.displayText(voiceRouteFailure, transcript: transcriptPreview)
    }

    var voiceStatusEvidence: [String: Any] {
        JarvisVoiceRouteFailure.evidence(voiceRouteFailure)
    }

    var audioProviderStatus: String {
        if usesLiveAudio, providerRun != nil {
            return isSessionReady ? "Active: GPT Live native audio. Playback attribution unavailable."
                : "Requested: GPT Live; waiting for matching backend and native media readiness."
        }
        guard let run = providerRun else { return "No active audio session." }
        if let active = run.activeProvider {
            return "Active: \(active.label) | \(run.activeModel ?? "model not reported by backend")"
        }
        if let acknowledgement = run.acknowledgement {
            return "Backend selected \(acknowledgement.provider.label); waiting for session readiness."
        }
        return "Requested: \(run.requested.label); waiting for backend acknowledgement."
    }

    func startJarvis(startupOrigin: JarvisStartupDiagnostics.Origin = .unspecified) {
        guard !shouldShowStopJarvis else { return }
        isUnifiedJarvisSession = true
        startSession(startupOrigin: startupOrigin)
    }

    func stopJarvis() {
        isUnifiedJarvisSession = false
        stopLocalVisualStream()
        stopSession()
    }

    func startSession(startupOrigin: JarvisStartupDiagnostics.Origin = .unspecified) {
        let requestedAt = ProcessInfo.processInfo.systemUptime
#if DEBUG
        guard !replayStartPending, !replayCleanupFailed else { return }
#endif
        guard connectionState != .connecting, !isSessionReady, sessionStopTask == nil else { return }
        guard let configuration, let apiClient else {
            failSession(JarvisAudioSessionError.missingConfiguration.localizedDescription)
            return
        }

        let run = JarvisAudioProviderRun(requested: preferredAudioProvider)
        recordStartup(startupDiagnostics?.retire(at: requestedAt))
        startupDiagnostics = JarvisStartupDiagnostics(attemptID: run.id, origin: startupOrigin)
        recordStartup(startupDiagnostics?.start(at: requestedAt))
        recordStartupInputPredicates()
        voiceRouteFailure = nil
        contextReconnectBudget = JarvisContextReconnectBudget()
        isContextReconnecting = false
        pendingContextReset = nil
        providerRun = run
        activeVoiceRouteChoice = preferredVoiceRoute
#if DEBUG
        if let replayVoiceRoute { activeVoiceRouteChoice = replayVoiceRoute }
#endif
        // Explicit development rollback: set jarvis.directAudioEnabled=false.
        // Do not silently change transports after a partially established call.
        usesDirectAudio = usesLiveAudio || (run.requested == .openai
            && (UserDefaults.standard.object(forKey: "jarvis.directAudioEnabled") as? Bool ?? true))
        // Explicit opt-in; absence of v1 negotiation is never a silent fallback.
        voiceRouteEnabled = usesLiveAudio || UserDefaults.standard.bool(forKey: "jarvis.voiceRouteEnabled")
        // Explicit Live selection uses its existing legacy-mode negotiation.
        // Keep the saved Lean preference for the next Realtime session.
        usesLeanVoice = !usesLiveAudio && UserDefaults.standard.bool(forKey: "jarvis.leanVoiceEnabled")
        if usesLeanVoice {
            guard usesDirectAudio, !usesLiveAudio, run.requested == .openai,
                  captureMode == .autoDetect else {
                failSession("Lean voice requires direct Realtime audio and Auto detect speech.")
                return
            }
            voiceRouteEnabled = false
        }
#if DEBUG
        if replayRunID != nil { usesDirectAudio = true; voiceRouteEnabled = !usesLeanVoice }
#endif
        pendingDirectReadyEvent = nil
        isStopping = false
        connectionState = .connecting
        turnState = .idle
        transcriptPreview = "Opening a private Jarvis channel..."
        visualFrameAcceptedCount = 0
        visualFrameDroppedCount = 0
        visualBackendStatus = "Backend visual memory disconnected."
        visualBackendUploadsDisabled = false
        appendDiagnostic("Starting Jarvis audio session. Requested provider: \(run.requested.rawValue).")

        sessionStartTask?.cancel()
        sessionStartTask = Task {
            do {
                var resumeActive = true
#if DEBUG
                resumeActive = replayRunID == nil
#endif
                let response = try await apiClient.startSession(title: nil, resumeActive: resumeActive)
                guard !Task.isCancelled, !isStopping, providerRun?.id == run.id else {
                    try? await apiClient.closeSession(id: response.session.id)
                    return
                }
                activeSession = response.session
                if let sessionID = UUID(uuidString: response.session.id) {
                    recordStartup(startupDiagnostics?.bindSession(sessionID, at: ProcessInfo.processInfo.systemUptime))
                }
#if DEBUG
                try await replayCapture?.bindTaskSession(response.session.id)
                try Task.checkCancellation()
                guard !isStopping, providerRun?.id == run.id else { throw CancellationError() }
#endif
                activeRealtimeConfig = response.realtime
                try await connectRealtime(config: response.realtime, run: run, configuration: configuration)
                sessionStartTask = nil
            } catch {
                guard !Task.isCancelled, providerRun?.id == run.id else { return }
                sessionStartTask = nil
                failSession("Start failed: \(error.localizedDescription)")
            }
        }
    }

    func stopSession() {
#if DEBUG
        replayStartPending = false
#endif
        guard sessionStopTask == nil else { return }
        isStopping = true
        contextReconnectTask?.cancel()
        contextReconnectTask = nil
        isContextReconnecting = false
        pendingContextReset = nil
        let startingTask = sessionStartTask
        startingTask?.cancel()
        sessionStartTask = nil
        let closingSession = activeSession
        let closingAPIClient = apiClient
        appendDiagnostic("Stopping Jarvis audio session.")
        resetLocalAudioState()

        sessionStopTask = Task {
            // Finish cancelled REST-start cleanup before another Start can
            // resume the same durable session on a different provider.
            await startingTask?.value
            if realtimeClient.isConnected {
                try? await realtimeClient.sendSessionClose()
            }
            realtimeClient.disconnect()
            if let closingAPIClient, let closingSession {
                try? await closingAPIClient.closeSession(id: closingSession.id)
            }
            self.activeSession = nil
            self.activeRealtimeConfig = nil
            self.connectionState = .disconnected
            self.turnState = .idle
            self.transcriptPreview = "Session closed."
            self.updateRouteDiagnostics(reason: "Session stopped")
            self.onRealtimeConnectionStateChanged?(.disconnected)
            self.sessionStopTask = nil
        }
    }

    func beginManualSpeech() {
        guard manualBeginEnabled else { return }
        if usesLiveAudio {
            do {
                guard let liveAppRoute else { throw JarvisVoiceRouteError.notReady }
                try liveAppRoute.setManualRecording(true)
                isManualRecording = true; turnState = .listening
            } catch { failSession(error.localizedDescription) }
            return
        }
        isManualRecording = true
        do {
            try startCaptureIfNeeded()
            beginSpeechSegment(startedAt: Date())
            if usesDirectAudio, let segmentID = activeSegmentId {
                enqueueAudioSend { try await self.realtimeClient.sendAudioInputBegin(segmentId: segmentID) }
            }
            appendDiagnostic("Manual speech started.")
        } catch {
            isManualRecording = false
            failSession("Manual capture failed: \(error.localizedDescription)")
        }
    }

    func endManualSpeech() {
        guard captureMode == .holdToTalk, isManualRecording else { return }
        isManualRecording = false
        if usesLiveAudio {
            do { try liveAppRoute?.setManualRecording(false); turnState = .idle }
            catch { failSession(error.localizedDescription) }
            return
        }
        if usesDirectAudio {
            let hadAudio = directAudioClient.microphoneEnabled
            directAudioClient.setMicrophoneEnabled(false)
            guard hadAudio else {
                if let segmentID = activeSegmentId {
                    enqueueAudioSend { try await self.realtimeClient.sendAudioInputAbort(segmentId: segmentID) }
                }
                speechSegment.reset()
                turnState = .idle
                appendDiagnostic("Manual speech released before capture was ready; no empty turn submitted.")
                return
            }
        }
        let segmentId = activeSegmentId
        let startedAt = activeSegmentStartedAt
        let endedAt = Date()
        turnState = .processingInput
        appendDiagnostic("Manual speech ended.")
        // Keep the microphone configured for the unified glasses stream.
        // In hold-to-talk mode, handleCapturedAudio discards idle audio.
        if !isUnifiedJarvisSession {
            captureService.stop()
        }
        enqueueAudioSend {
            try await self.commitSpeechSegment(segmentId: segmentId, startedAt: startedAt, endedAt: endedAt)
        }
    }

    func captureModeDidChange() {
        if usesLeanVoice && captureMode != .autoDetect && isSessionReady {
            failSession("Stop lean voice before selecting Hold to talk.")
            return
        }
        segmenter.reset()
        speechSegment.reset()
        isManualRecording = false
        if usesLiveAudio {
            do { try liveAppRoute?.setCaptureMode(captureMode); turnState = .idle }
            catch { failSession(error.localizedDescription) }
            return
        }
        if usesDirectAudio {
            directAudioClient.setMicrophoneEnabled(false)
            guard isSessionReady else { return }
            let mode = captureMode
            enqueueAudioSend { try await self.realtimeClient.sendCaptureMode(mode) }
            turnState = .idle
            return
        }
        switch captureMode {
        case .autoDetect:
            if isSessionReady {
                do {
                    try startCaptureIfNeeded()
                } catch {
                    failSession("Audio capture failed: \(error.localizedDescription)")
                }
            }
        case .holdToTalk:
            if !isUnifiedJarvisSession {
                captureService.stop()
            }
            turnState = .idle
        }
    }

    func refreshRouteDiagnostics() {
        updateRouteDiagnostics(reason: "Manual route refresh")
    }

    func configureRayBanIfNeeded() {
#if DEBUG
        guard replayRunID == nil, !replayStartPending else { return }
#endif
        rayBanDATService.configureIfNeeded()
    }

    func handleOpenURL(_ url: URL) -> Bool {
        rayBanDATService.handleOpenURL(url)
    }

    func refreshRayBanDiagnostics() {
        rayBanDATService.refreshDiagnostics()
        rayBanDiagnostics = rayBanDATService.diagnostics
        syncVisualState(with: rayBanDATService.diagnostics)
        appendDiagnostic("Ray-Ban diagnostics refreshed.")
    }

    func requestRayBanRegistrationOrPermission() {
        rayBanDATService.requestRegistrationOrPermission()
        appendDiagnostic("Ray-Ban registration or permission requested.")
    }

    func startLocalVisualStream() {
#if DEBUG
        if let replayCapture, !visualCoordinator.isReplaySelected {
            do { try visualCoordinator.selectReplay(replayCapture) }
            catch { failSession("Replay capture selection failed."); return }
        }
#endif
        guard !visualStreamState.isStarting, !visualStreamState.isActive else { return }
        let generation = UUID()
        visualStreamGeneration = generation
        configureRayBanIfNeeded()
        localVisualFrameCount = 0
        latestVisualFrame = nil
        visualStreamState = .starting
        visualStatus = "Starting Ray-Ban vision..."
        appendDiagnostic("Starting local Ray-Ban visual stream.")

        visualStartTask?.cancel()
        visualStartTask = Task {
            guard !Task.isCancelled, visualStreamGeneration == generation else { return }
            do {
                try await visualCoordinator.startGlasses()
                guard !Task.isCancelled, visualStreamGeneration == generation else { return }
                visualStartTask = nil
                // DAT can report a failure through diagnostics without throwing
                // from start. Reconcile the latest state after startup settles.
                let diagnostics = rayBanDATService.diagnostics
                rayBanDiagnostics = diagnostics
                syncVisualState(with: diagnostics)
                appendDiagnostic("Local Ray-Ban visual stream startup returned: \(diagnostics.streamState.label).")
            } catch {
                guard !Task.isCancelled, visualStreamGeneration == generation else { return }
                visualStartTask = nil
                failLocalVisualStream(error.localizedDescription)
            }
        }
    }

    func stopLocalVisualStream() {
        recordings.refresh(visionReady: false)
        let needsStop = visualStartTask != nil || visualStreamGeneration != nil
            || visualStreamState != .stopped || visualCoordinator.isRunning
        visualStreamGeneration = nil
        visualStartTask?.cancel()
        visualStartTask = nil
        guard needsStop else { return }
        visualCoordinator.stop()
        visualIngestClient.disconnect()
        visualSessionId = nil
        visualBackendStatus = "Backend visual memory disconnected."
        visualStreamState = .stopped
        visualStatus = "Glasses vision stopped."
        appendDiagnostic("Local Ray-Ban visual stream stopped.")
    }

    private func failLocalVisualStream(_ message: String) {
        stopLocalVisualStream()
        visualStreamState = .failed(message)
        visualStatus = "Ray-Ban vision failed: \(message)"
        appendDiagnostic(visualStatus)
        // Vision failure must not cancel audio startup or a usable audio session.
    }

    private func handleCapturedAudio(_ chunk: CapturedAudioChunk) {
        guard isSessionReady, !usesDirectAudio else { return }
        switch captureMode {
        case .autoDetect:
            for event in segmenter.process(chunk) {
                handleSegmenterEvent(event)
            }
        case .holdToTalk:
            guard isManualRecording else { return }
            sendSpeechChunk(chunk)
        }
    }

    private func handleSegmenterEvent(_ event: AudioSegmenter.Event) {
        switch event {
        case .speechStarted(let startedAt):
            beginSpeechSegment(startedAt: startedAt)
        case .speechChunk(let chunk):
            sendSpeechChunk(chunk)
        case .speechEnded(let startedAt, let endedAt):
            let segmentId = activeSegmentId
            turnState = .processingInput
            appendDiagnostic("Speech ended; committing segment.")
            enqueueAudioSend {
                try await self.commitSpeechSegment(segmentId: segmentId, startedAt: startedAt, endedAt: endedAt)
            }
        }
    }

    private func beginSpeechSegment(startedAt: Date) {
        if playbackService.isPlaying || directAudioClient.isPlaying || hasAssistantResponseInProgress {
            bargeInIfNeeded()
        }
        speechSegment.begin(at: startedAt)
        turnState = .listening
        appendDiagnostic("Speech segment started.")
    }

    private func sendSpeechChunk(_ chunk: CapturedAudioChunk) {
        guard let segmentId = activeSegmentId else { return }
        enqueueAudioSend {
            try await self.realtimeClient.sendAudioAppend(chunk.pcm16Data, segmentId: segmentId)
        }
    }

    private func commitSpeechSegment(segmentId: String?, startedAt: Date?, endedAt: Date?) async throws {
        guard let segmentId else {
            throw JarvisAudioSessionError.serverError("No active speech segment to commit.")
        }
        try await realtimeClient.sendAudioCommit(segmentId: segmentId, startedAt: startedAt, endedAt: endedAt)
        try await realtimeClient.sendResponseCreate()
        if speechSegment.finishCommit(segmentID: segmentId) {
            turnState = .processingInput
        }
    }

    private func bargeInIfNeeded() {
        if usesLiveAudio { interruptSelectedRoute(); return }
        if voiceRouteDriver != nil {
            interruptSelectedRoute()
            hasAssistantResponseInProgress = false
            return
        }
        if usesDirectAudio { directAudioClient.interrupt() }
        let truncation = playbackService.truncationSnapshot()
        playbackService.stop()
        hasAssistantResponseInProgress = false
        enqueueAudioSend {
            try await self.realtimeClient.sendResponseCancel()
            if let truncation {
                try await self.realtimeClient.sendPlaybackTruncated(
                    itemId: truncation.itemId,
                    audioEndMs: truncation.audioEndMs
                )
            }
            self.appendDiagnostic("Barge-in sent.")
        }
    }

    private func enqueueAudioSend(_ send: @escaping @MainActor () async throws -> Void) {
        do {
            try audioSendQueue.enqueue(send, onFailure: { [weak self] error in
                self?.failSession("Audio send failed: \(error.localizedDescription)")
            })
        } catch {
            failSession("Audio send failed: \(error.localizedDescription)")
        }
    }

    private func receiveMediaCommand(_ event: JarvisRealtimeServerEvent) {
        guard !isStopping, isSessionReady, event.audioSessionId == audioSessionId,
              let rawSession = event.sessionId, let sessionID = UUID(uuidString: rawSession),
              activeSession.flatMap({ UUID(uuidString: $0.id) }) == sessionID,
              let rawID = event.values["media_id"] as? String, let id = UUID(uuidString: rawID),
              let action = event.values["action"] as? String else { return }
        if action == "stop" {
            guard mediaID == id else { return }
            mediaTask?.cancel(); mediaTask = nil; mediaID = nil
            mediaStatus = "Music stopped."
            return
        }
        let volume = Float((event.values["volume"] as? NSNumber)?.doubleValue ?? 0.65)
        if action == "volume" {
            guard mediaID == id else { return }
            _ = selectedRecordingAudioDevice?.setMediaVolume(volume, muted: volume == 0)
            return
        }
        guard action == "start", mediaID != id, let api = apiClient else { return }
        guard let device = selectedRecordingAudioDevice else {
            mediaStatus = "Music needs the active recording-capable audio route."
            Task { try? await api.reportMediaState(sessionID: sessionID, id: id, state: "failed") }
            return
        }
        mediaTask?.cancel()
        mediaID = id; mediaStatus = "Connecting music."
        mediaTask = Task { [weak self] in
            do {
                let request = try api.mediaStreamRequest(sessionID: sessionID, id: id)
                try await JarvisMediaStream().run(request: request, device: device, volume: volume, receiving: { [weak self] in
                    Task { @MainActor [weak self] in
                        guard let self, self.mediaID == id else { return }
                        self.mediaStatus = "Receiving music."
                        try? await api.reportMediaState(sessionID: sessionID, id: id, state: "receiving")
                    }
                }, playing: { [weak self] in
                    Task { @MainActor [weak self] in
                        guard let self, self.mediaID == id else { return }
                        self.mediaStatus = "Playing music."
                        try? await api.reportMediaState(sessionID: sessionID, id: id, state: "playing")
                    }
                })
            } catch {
                guard !Task.isCancelled, let self, self.mediaID == id else { return }
                self.mediaStatus = "Music connection stopped. Ask Jarvis to play again."
                self.appendDiagnostic("Media stream ended: \(type(of: error)).")
                self.mediaID = nil
                try? await api.reportMediaState(sessionID: sessionID, id: id, state: "failed")
            }
        }
    }

    private func handleServerEvent(_ event: JarvisRealtimeServerEvent) {
        if let version = event.values["recording_protocol_version"] as? Int, version == 1 || version == 2 {
            recordingProtocolVersion = version
        }
        if usesLiveAudio {
            do {
                if let liveAppRoute, try liveAppRoute.receive(event, connectionID: realtimeClient.connectionID) { return }
            } catch { failSession(error.localizedDescription); return }
            // No fallback media or legacy readiness during partial Live startup.
            if event.type.hasPrefix("live.media.") || event.type.hasPrefix("voice_route.")
                || (event.type.hasPrefix("audio.") && event.type != "audio.provider.selected")
                || event.type.hasPrefix("response.") || event.type.hasPrefix("transcript.")
                || ["session.created", "user.transcript.done"].contains(event.type) { return }
        }
        if event.type.hasPrefix("voice_route.") {
            handleVoiceRouteEvent(event)
            return
        }
        if let driver = voiceRouteDriver,
           ["audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started", "audio.webrtc.stopped",
            "audio.webrtc.cleared", "audio.output.delta", "audio.output.done", "audio.output.clear",
            "audio.input.ready", "audio.capture_mode.ready"].contains(event.type),
           !driver.acceptsMedia(event) { return }
        if pendingContextReset != nil,
           ["audio.output.delta", "audio.output.done", "transcript.delta", "transcript.done", "response.done",
            "audio.webrtc.answer", "audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started",
            "audio.webrtc.stopped", "audio.webrtc.cleared", "audio.input.started", "audio.input.stopped"].contains(event.type) {
            return
        }
        switch event.type {
        case "app.link":
            guard !isStopping, isSessionReady, pendingContextReset == nil,
                  let link = JarvisAppLinkRequest(event: event), link.audioSessionID == audioSessionId,
                  activeSession.flatMap({ UUID(uuidString: $0.id) }) == link.sessionID else { return }
            onAppLinkRequested?(link)
        case "media.command":
            receiveMediaCommand(event)
        case "workspace.sandbox.closed":
            guard !isStopping, isSessionReady, event.audioSessionId == audioSessionId,
                  let rawSession = event.sessionId, let sessionID = UUID(uuidString: rawSession),
                  activeSession.flatMap({ UUID(uuidString: $0.id) }) == sessionID,
                  let rawView = event.values["view_id"] as? String, let viewID = UUID(uuidString: rawView) else { return }
            onSandboxViewStopped?(sessionID, viewID)
        case "recording.command":
            recordings.receive(event.values)
        case "behavior_profile.committed":
            guard !isStopping, isSessionReady, pendingContextReset == nil,
                  let expectedAudioSessionID = audioSessionId, event.audioSessionId == expectedAudioSessionID,
                  let rawSession = event.sessionId, let sessionID = UUID(uuidString: rawSession),
                  let activeID = activeSession?.id, UUID(uuidString: activeID) == sessionID else { return }
            if let committed = event.behaviorProfileCommitted {
                onBehaviorProfileCommitted?(sessionID, committed)
            }
        case "workspace.presentation.intent":
            guard !isStopping, isSessionReady, pendingContextReset == nil,
                  let expectedAudioSessionID = audioSessionId, event.audioSessionId == expectedAudioSessionID,
                  let rawSession = event.sessionId, let sessionID = UUID(uuidString: rawSession),
                  let activeID = activeSession?.id, UUID(uuidString: activeID) == sessionID,
                  let rawRequest = event.values["request_id"] as? String,
                  let requestID = UUID(uuidString: rawRequest) else { return }
            onWorkspacePresentationIntent?(sessionID, requestID)
        case "audio.webrtc.answer":
            guard usesDirectAudio, !isStopping, let sdp = event.values["sdp"] as? String,
                  let runID = providerRun?.id else { return }
            guard (event.values["conversation_mode"] as? String ?? "legacy") == (usesLeanVoice ? "lean_v1" : "legacy") else {
                failSession("The backend did not confirm the selected conversation mode.")
                return
            }
            Task { @MainActor in
                do {
                    try await self.directAudioClient.applyAnswer(sdp)
                    guard self.providerRun?.id == runID, !self.isStopping else { return }
                    // Warm the native audio route/unit while server context is
                    // still preparing. Input stays muted until session.created.
                    try self.directAudioClient.startAudio(microphoneEnabled: false)
                    if let ready = self.pendingDirectReadyEvent {
                        self.pendingDirectReadyEvent = nil
                        self.handleServerEvent(ready)
                    }
                } catch {
                    guard self.providerRun?.id == runID, !self.isStopping else { return }
                    self.failSession("Direct audio setup failed: \(error.localizedDescription)")
                }
            }
        case "audio.webrtc.response":
            guard usesDirectAudio, !usesLeanVoice, isSessionReady, let responseID = event.responseId else { return }
#if DEBUG
            directAudioClient.recordReplayResponsePermission(event, controlConnectionID: realtimeClient.connectionID)
#endif
            directAudioClient.permitResponse(responseID)
            hasAssistantResponseInProgress = true
        case "audio.webrtc.output_item", "audio.webrtc.started", "audio.webrtc.stopped", "audio.webrtc.cleared",
             "audio.input.started", "audio.input.stopped":
            guard usesDirectAudio, !usesLeanVoice, isSessionReady else { return }
            directAudioClient.handleProviderEvent(event)
        case "audio.input.ready":
            guard usesDirectAudio, isSessionReady, captureMode == .holdToTalk, isManualRecording,
                  event.values["segment_id"] as? String == activeSegmentId else { return }
            directAudioClient.setMicrophoneEnabled(true)
#if DEBUG
            if let segment = event.values["segment_id"] as? String {
                replayCapture?.manualReady(segmentID: segment, connectionID: realtimeClient.connectionID, fields: event.values)
            }
#endif
        case "audio.capture_mode.ready":
            guard usesDirectAudio, isSessionReady,
                  event.values["capture_mode"] as? String == (captureMode == .autoDetect ? "auto" : "manual") else { return }
            directAudioClient.setMicrophoneEnabled(captureMode == .autoDetect)
        case "audio.provider.selected":
            guard !isStopping, var run = providerRun else { return }
            do {
                try run.acknowledge(event)
                providerRun = run
                if let value = event.audioSessionId, let id = UUID(uuidString: value) {
                    recordStartup(startupDiagnostics?.bindAudioSession(id, at: ProcessInfo.processInfo.systemUptime))
                }
                appendDiagnostic("Backend selected audio provider: \(run.requested.rawValue).")
            } catch {
                failSession(error.localizedDescription)
            }
        case "session.created":
            guard !isStopping, connectionState == .connecting else { return }
            guard (event.values["conversation_mode"] as? String ?? "legacy") == (usesLeanVoice ? "lean_v1" : "legacy") else {
                failSession("Conversation mode changed during startup.")
                return
            }
            if let driver = voiceRouteDriver {
                do {
                    guard let identity = event.audioSessionId else { throw JarvisVoiceRouteError.identityMismatch }
                    if try !driver.observeSessionReady(audioSessionID: identity) {
                        pendingDirectReadyEvent = event
                        return
                    }
                } catch { failSession(error.localizedDescription); return }
            }
            if usesDirectAudio {
                guard event.values["audio_transport"] as? String == "webrtc" else {
                    failSession("The backend did not confirm direct audio support.")
                    return
                }
                guard directAudioClient.hasRemoteDescription else {
                    pendingDirectReadyEvent = event
                    return
                }
                if usesLeanVoice && !directAudioClient.activateNativeTurns() {
                    failSession("Native turn ownership could not be activated.")
                    return
                }
            }
            guard var run = providerRun else { return }
            do {
                try run.markReady(audioSessionID: event.audioSessionId)
            } catch {
                failSession(error.localizedDescription)
                return
            }
            providerRun = run
            if let value = event.audioSessionId, let id = UUID(uuidString: value) {
                recordStartup(startupDiagnostics?.bindAudioSession(id, at: ProcessInfo.processInfo.systemUptime))
            }
            let resumedContext = isContextReconnecting
            isContextReconnecting = false
            pendingContextReset = nil
            isSessionReady = true
            voiceRouteTimeout?.cancel()
            voiceRouteTimeout = nil
            audioSessionId = event.audioSessionId
            connectionState = .connected
            onRealtimeConnectionStateChanged?(.connected)
            turnState = .idle
            transcriptPreview = "Session active. Speak naturally through the selected audio route."
            if resumedContext {
                transcriptPreview = "Context refreshed. Please repeat anything said during the pause."
            }
            updateRouteDiagnostics(reason: "Realtime session created")
            do {
                try run.startMedia(
                    captureMode: captureMode, unifiedSession: isUnifiedJarvisSession,
                    startMicrophone: { try self.startCaptureIfNeeded() },
                    startCamera: {
                        // Including hold-to-talk: configure HFP before DAT.
                        self.appendDiagnostic("Audio capture ready; starting glasses vision.")
                        self.startLocalVisualStream()
                    }
                )
            } catch {
                failSession("Audio capture failed: \(error.localizedDescription)")
                return
            }
            connectVisualIngestIfPossible()
            appendDiagnostic("Realtime session created.")
            refreshRecordingReadiness()
        case "session.context_invalidated":
            guard !isStopping, let session = activeSession,
                  let notice = JarvisContextResetNotice(event: event,
                    audioSessionID: audioSessionId ?? providerRun?.acknowledgement?.audioSessionID,
                    sessionID: session.id) else { return }
            if let pendingContextReset, pendingContextReset != notice { return }
            pendingContextReset = notice
            pauseAudioForContextReset()
            let connectionID = realtimeClient.connectionID
            Task { @MainActor in
                _ = try? await self.realtimeClient.sendContextResetCleared(notice, connectionID: connectionID)
            }
        case "session.closed":
            if !isStopping, let session = activeSession,
               let notice = JarvisContextResetNotice(event: event,
                    audioSessionID: audioSessionId ?? providerRun?.acknowledgement?.audioSessionID,
                    sessionID: session.id),
               pendingContextReset == nil || pendingContextReset == notice {
                pendingContextReset = notice
                scheduleContextReconnect()
                return
            }
            isStopping = true
            resetLocalAudioState()
            realtimeClient.disconnect()
            activeSession = nil
            activeRealtimeConfig = nil
            connectionState = .disconnected
            turnState = .idle
            onRealtimeConnectionStateChanged?(.disconnected)
            appendDiagnostic("Realtime session closed.")
        case "user.transcript.done":
            appendTranscript(role: .user, text: event.text ?? "")
            turnState = .thinking
        case "transcript.delta":
            if event.role == "assistant", let text = event.text, !text.isEmpty {
                transcriptPreview = text
            }
        case "transcript.done":
            if event.role == "assistant" {
                appendTranscript(role: .assistant, text: event.text ?? "")
            }
        case "audio.output.delta":
            hasAssistantResponseInProgress = true
            turnState = .speaking
            handleAudioOutput(event)
        case "audio.output.clear":
            if usesDirectAudio, let responseID = event.responseId {
                directAudioClient.interrupt(responseID: responseID)
            }
            guard providerRun?.shouldClearPlayback(event, snapshot: playbackService.truncationSnapshot()) == true else {
                return
            }
            playbackService.stop()
            hasAssistantResponseInProgress = false
            appendDiagnostic("The backend cleared the matching playback response.")
        case "audio.output.done":
            playbackService.generationFinished(responseId: event.responseId, itemId: event.itemId)
            appendDiagnostic("Assistant audio done.")
        case "response.done":
            // Native provider events own lean response state. The backend's
            // transcript/persistence queue can finish an older response later.
            guard !usesLeanVoice else { return }
            hasAssistantResponseInProgress = false
            if !playbackService.isPlaying && !directAudioClient.isPlaying {
                turnState = .idle
            }
        case "reminder.started":
            if let message = event.message, !message.isEmpty {
                transcriptPreview = message
                speakingReminderMessage = message
            }
            speakingReminderEventId = event.reminderEventId
            appendDiagnostic("Scheduled reminder started: \(event.reminderId ?? "-")")
            onReminderStateChanged?()
        case "reminder.response_created":
            appendDiagnostic("Scheduled reminder response created: \(event.reminderEventId ?? "-")")
            onReminderStateChanged?()
        case "reminder.done":
            speakingReminderMessage = nil
            speakingReminderEventId = nil
            appendDiagnostic("Reminder delivery completed.")
            onReminderStateChanged?()
        case "reminders.changed":
            appendDiagnostic("Reminder state changed.")
            onReminderStateChanged?()
        case "reminder.interrupted", "reminder.failed":
            speakingReminderMessage = nil
            speakingReminderEventId = nil
            appendDiagnostic(event.message ?? "Scheduled reminder delivery failed.")
            onReminderStateChanged?()
        case "memory.updated":
            if let hint = event.memoryRefreshHint {
                onMemoryStateChanged?(hint)
            }
            appendDiagnostic("Authoritative memory changed.")
        case "working_memory.updated":
            if let hint = event.workingMemoryRefreshHint {
                onWorkingMemoryStateChanged?(hint)
            }
            appendDiagnostic("Working memory revision changed.")
        case "error":
            let message = event.message ?? event.code ?? "Realtime error"
            if isContextReconnecting, event.code == "audio_session_already_active" {
                scheduleContextReconnect()
                return
            }
            if connectionState == .connecting || (providerRun?.activeProvider == .gemini && event.recoverable == false) {
                failSession(message)
                return
            }
            turnState = .failed(message)
            appendTranscript(role: .error, text: message)
            appendDiagnostic(message)
        default:
            appendDiagnostic("Realtime event: \(event.type)")
        }
    }

    private func pauseAudioForContextReset() {
        // Keep the durable session and glasses camera; pause only local voice.
        isSessionReady = false
        recordStartup(startupDiagnostics?.retire(at: ProcessInfo.processInfo.systemUptime))
        retireLocalVoiceRoute()
        captureService.stop()
        playbackService.resetConnection()
        directAudioClient.close()
        pendingDirectReadyEvent = nil
        segmenter.reset()
        audioSendQueue.reset()
        speechSegment.reset()
        isManualRecording = false
        hasAssistantResponseInProgress = false
        speakingReminderMessage = nil
        speakingReminderEventId = nil
        transcriptPreview = "Refreshing changed context. Please wait before speaking."
        turnState = .thinking
    }

    private func scheduleContextReconnect() {
#if DEBUG
        if replayRunID != nil { failSession("Replay context reset requires a new explicit evaluation run."); return }
#endif
        guard !isStopping, pendingContextReset != nil,
              let configuration, let realtimeConfig = activeRealtimeConfig,
              let requested = providerRun?.requested else { return }
        guard contextReconnectBudget.admit() else {
            failSession("Context refresh failed repeatedly. Start Jarvis again to retry.")
            return
        }
        pauseAudioForContextReset()
        realtimeClient.disconnect()
        audioSessionId = nil
        isContextReconnecting = true
        connectionState = .connecting
        onRealtimeConnectionStateChanged?(.connecting)
        let run = JarvisAudioProviderRun(requested: requested)
        providerRun = run
        contextReconnectTask?.cancel()
        contextReconnectTask = Task {
            do {
                // Allow the backend to unregister the retired connection.
                try await Task.sleep(for: .milliseconds(250))
                guard !Task.isCancelled, !isStopping, providerRun?.id == run.id else { return }
                try await connectRealtime(config: realtimeConfig, run: run, configuration: configuration)
                contextReconnectTask = nil
                appendDiagnostic("Reconnecting the same conversation with refreshed context.")
            } catch {
                guard !Task.isCancelled, providerRun?.id == run.id else { return }
                contextReconnectTask = nil
                failSession("Context reconnect failed: \(error.localizedDescription)")
            }
        }
    }

    private func handleAudioOutput(_ event: JarvisRealtimeServerEvent) {
#if DEBUG
        if replayRunID != nil {
            failSession("External PCM output is unsupported during silent native replay.")
            return
        }
#endif
        guard let audioBase64 = event.audioBase64 else { return }
        if usesDirectAudio { directAudioClient.setExternalPlaybackActive(true) }
        do {
            try playbackService.enqueuePCM16(
                base64: audioBase64,
                responseId: event.responseId,
                itemId: event.itemId
            )
        } catch {
            failSession("Playback failed: \(error.localizedDescription)")
        }
    }

    private func startCaptureIfNeeded() throws {
        if usesLiveAudio {
            guard let liveAppRoute, liveAppRoute.isReady else { throw JarvisVoiceRouteError.notReady }
            try liveAppRoute.setCaptureMode(captureMode)
            return
        }
        if usesDirectAudio {
            try directAudioClient.startAudio(microphoneEnabled: captureMode == .autoDetect)
            updateRouteDiagnostics(reason: "Direct audio ready")
            return
        }
        guard !captureService.isRunning else { return }
        try captureService.start()
        updateRouteDiagnostics(reason: "Audio capture started")
        appendDiagnostic("Audio capture started.")
    }

    private func connectRealtime(config: JarvisRealtimeConfig, run: JarvisAudioProviderRun,
                                 configuration: JarvisConfiguration) async throws {
#if DEBUG
        if replayRunID != nil {
            guard let audioURL = URL(string: config.audioWebsocketURL), let visualURL = URL(string: config.visualWebsocketURL) else {
                throw JarvisReplayError.isolationUnavailable
            }
            try JarvisReplayScope.validateSocketURL(audioURL, base: configuration.backendBaseURL)
            try JarvisReplayScope.validateSocketURL(visualURL, base: configuration.backendBaseURL)
        }
#endif
        if usesLiveAudio {
            try await connectLive(config: config, run: run, configuration: configuration)
            return
        }
        let offer = usesDirectAudio ? try await directAudioClient.prepareOffer(nativeTurns: usesLeanVoice) : nil
        try Task.checkCancellation()
        guard !isStopping, providerRun?.id == run.id else { throw CancellationError() }
        var url = try JarvisVoiceRouteChoice.realtime.audioURL(from: config.audioWebsocketURL, directAudio: usesDirectAudio)
        if usesLeanVoice {
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { throw URLError(.badURL) }
            var query = (components.queryItems ?? []).filter { $0.name != "conversation_mode" }
            query.append(URLQueryItem(name: "conversation_mode", value: "lean_v1"))
            components.queryItems = query
            guard let selectedURL = components.url else { throw URLError(.badURL) }
            url = selectedURL
        }
#if DEBUG
        if replayRunID != nil {
            url = try JarvisReplayScope.boundSocketURL(url, base: configuration.backendBaseURL, sessionID: activeSession?.id)
        }
#endif
        realtimeClient.connect(audioWebSocketURL: url, bearerToken: configuration.staticBearerToken)
        recordStartup(startupDiagnostics?.bindConnection(realtimeClient.connectionID,
            at: ProcessInfo.processInfo.systemUptime))

        let kind: JarvisVoiceMediaKind = usesDirectAudio ? .nativeWebRTC : .streamedPCM
        if voiceRouteEnabled {
            guard let rawSessionID = activeSession?.id, let sessionID = UUID(uuidString: rawSessionID) else {
                throw JarvisVoiceRouteError.identityMismatch
            }
            let driver = JarvisVoiceRouteDriver(sessionID: sessionID, connectionID: realtimeClient.connectionID,
                                               mediaKind: kind, fenceMedia: { [weak self] close in
                guard let self else { return false }
                if close { self.captureService.stop() }
                self.playbackService.stop()
                return self.directAudioClient.fenceRouteOutput(closeRoute: close) && !self.playbackService.isPlaying
            })
            voiceRouteDriver = driver
            voiceRouteFailure = nil
            voiceRouteTimeout?.cancel()
            voiceRouteTimeout = Task { @MainActor [weak self, weak driver] in
                do { try await Task.sleep(for: .seconds(15)) } catch { return }
                guard let self, let driver, self.voiceRouteDriver === driver, !self.isSessionReady else { return }
                self.failSession("Voice-route negotiation timed out. No fallback speaker was started.")
            }
        }
        if let offer {
            try await realtimeClient.sendWebRTCOffer(offer, captureMode: captureMode,
                                                     voiceRoute: voiceRouteEnabled ? kind.clientCapabilities : nil,
                                                     conversationMode: usesLeanVoice ? "lean_v1" : "legacy")
        } else if let driver = voiceRouteDriver {
            try await realtimeClient.sendVoiceRouteEvent([
                "type": "voice_route.client_capabilities", "voice_route": kind.clientCapabilities,
            ], connectionID: driver.connectionID)
        }
        appendDiagnostic(usesDirectAudio ? "Direct audio negotiating; backend controls connected." : "Audio relay connecting.")
    }

    private func connectLive(config: JarvisRealtimeConfig, run: JarvisAudioProviderRun,
                             configuration: JarvisConfiguration) async throws {
        guard let rawID = activeSession?.id, let sessionID = UUID(uuidString: rawID),
              !isStopping, providerRun?.id == run.id else { throw JarvisVoiceRouteError.identityMismatch }
        var url = try JarvisVoiceRouteChoice.live.audioURL(from: config.audioWebsocketURL, directAudio: true)
#if DEBUG
        if replayRunID != nil {
            url = try JarvisReplayScope.boundSocketURL(url, base: configuration.backendBaseURL, sessionID: activeSession?.id)
        }
#endif
        realtimeClient.connect(audioWebSocketURL: url, bearerToken: configuration.staticBearerToken)
        recordStartup(startupDiagnostics?.bindConnection(realtimeClient.connectionID,
            at: ProcessInfo.processInfo.systemUptime))

        let fencePrevious: () -> Bool = { [weak self] in
            guard let self else { return false }
            self.captureService.stop()
            self.playbackService.resetConnection()
            return self.directAudioClient.fenceRouteOutput(closeRoute: true)
                && !self.captureService.isRunning && !self.playbackService.isPlaying
        }
#if DEBUG
        let media = JarvisLiveWebRTCMedia()
        if let replayAudioDevice { try media.useReplayAudioDevice(replayAudioDevice) }
        let driver = JarvisLiveVoiceRouteDriver(sessionID: sessionID, connectionID: realtimeClient.connectionID,
            media: media, ownership: .shared, fencePrevious: fencePrevious)
#else
        let driver = JarvisLiveVoiceRouteDriver(sessionID: sessionID, connectionID: realtimeClient.connectionID,
            fencePrevious: fencePrevious)
#endif
        let owner = try JarvisLiveAppRoute(driver: driver, captureMode: captureMode,
            currentConnection: { [weak self] in self?.realtimeClient.connectionID },
            write: { [weak self] payload, connection, current in
                guard let self, !self.isStopping, self.providerRun?.id == run.id else { throw CancellationError() }
                try await self.realtimeClient.sendLiveAppEvent(payload, connectionID: connection, current: current)
            })
        liveAppRoute = owner
        owner.onProtocolFailure = { [weak self, weak owner] failure in
            guard let self, self.liveAppRoute === owner, !self.isStopping else { return }
            if let record = RealtimeDiagnostics.record("live_protocol_failure", fields: failure.fields) {
                self.appendDiagnostic(record)
            }
#if DEBUG
            self.replayAudioDevice?.recordControlEvent(.backendControlHandled, fields: failure.fields)
#endif
        }
        owner.onFailure = { [weak self, weak owner] message in
            guard let self, self.liveAppRoute === owner, !self.isStopping else { return }
            self.failSession(message)
        }
        owner.onStopped = owner.onFailure
        owner.onRecordingCapabilityChanged = { [weak self, weak owner] in
            guard let self, self.liveAppRoute === owner, !self.isStopping else { return }
            self.refreshRecordingReadiness()
        }
        owner.onCaption = { [weak self, weak owner] _, text in
            guard let self, self.liveAppRoute === owner, self.isSessionReady else { return }
            self.transcriptPreview = text
        }
        owner.onReady = { [weak self, weak owner] audioID in
            guard let self, let owner, self.liveAppRoute === owner, !self.isStopping,
                  self.providerRun?.id == run.id else { return }
            self.finishLiveReadiness(audioSessionID: audioID)
        }
        appendDiagnostic("GPT Live negotiating; Realtime rollback remains an explicit next-session choice.")
        try await owner.start()
    }

    private func finishLiveReadiness(audioSessionID: String) {
        guard let liveAppRoute, liveAppRoute.isReady, var run = providerRun else { return }
        do { try run.markReady(audioSessionID: audioSessionID) }
        catch { failSession(error.localizedDescription); return }
        providerRun = run
        audioSessionId = audioSessionID
        if let id = UUID(uuidString: audioSessionID) {
            recordStartup(startupDiagnostics?.bindAudioSession(id, at: ProcessInfo.processInfo.systemUptime))
        }
        isSessionReady = true
        isContextReconnecting = false; pendingContextReset = nil
        connectionState = .connected; turnState = .idle
        transcriptPreview = "GPT Live is connected. Captions are provisional; playback attribution is unavailable."
        onRealtimeConnectionStateChanged?(.connected)
        updateRouteDiagnostics(reason: "Live backend and native activation confirmed")
        if isUnifiedJarvisSession { startLocalVisualStream() }
        connectVisualIngestIfPossible()
        refreshRecordingReadiness()
    }

    private func failSession(_ message: String) {
#if DEBUG
        let replayClosingClient = replayRunID == nil ? nil : apiClient
        let replayClosingSession = replayRunID == nil ? nil : activeSession
        if let replayClosingClient, let replayClosingSession {
            Task { try? await replayClosingClient.closeSession(id: replayClosingSession.id) }
        }
#endif
        if isUnifiedJarvisSession {
            isUnifiedJarvisSession = false
            stopLocalVisualStream()
        }
        logger.error("\(message, privacy: .public)")
        resetLocalAudioState()
        realtimeClient.disconnect()
        connectionState = .failed(message)
        onRealtimeConnectionStateChanged?(.failed(message))
        turnState = .failed(message)
        transcriptPreview = message
        appendDiagnostic(message)
        updateRouteDiagnostics(reason: "Session failed")
    }

    private func resetLocalAudioState() {
#if DEBUG
        defer { finishReplay() }
#endif
        retireLocalVoiceRoute()
        contextReconnectTask?.cancel()
        contextReconnectTask = nil
        isContextReconnecting = false
        pendingContextReset = nil
        providerRun = nil
        captureService.stop()
        playbackService.resetConnection()
        directAudioClient.close()
        pendingDirectReadyEvent = nil
        stopLocalVisualStream()
        segmenter.reset()
        audioSendQueue.reset()
        isSessionReady = false
        isManualRecording = false
        recordStartup(startupDiagnostics?.retire(at: ProcessInfo.processInfo.systemUptime))
        speechSegment.reset()
        hasAssistantResponseInProgress = false
        speakingReminderMessage = nil
        speakingReminderEventId = nil
        audioSessionId = nil
        activeRealtimeConfig = nil
        visualIngestClient.disconnect()
        visualBackendStatus = "Backend visual memory disconnected."
        visualSessionId = nil
        visualBackendUploadsDisabled = false
    }

    private func retireLocalVoiceRoute() {
        recordingProtocolVersion = 1
        mediaTask?.cancel(); mediaTask = nil; mediaID = nil
        mediaStatus = "Music is not active."
        recordings.retire()
        isRecording = false
        voiceRouteTimeout?.cancel()
        voiceRouteTimeout = nil
        let live = liveAppRoute
        liveAppRoute = nil
        _ = live?.close()
        // Remove the owner before stopping PCM, which may synchronously emit a receipt.
        let driver = voiceRouteDriver
        voiceRouteDriver = nil
        _ = driver?.close()
        for task in voiceControlTasks.values { task.cancel() }
        voiceControlTasks.removeAll()
    }

    private func interruptSelectedRoute() {
        if let liveAppRoute {
            do { try liveAppRoute.interruptLocally() }
            catch { failSession(error.localizedDescription) }
            return
        }
        guard let driver = voiceRouteDriver else { return }
        do {
            if let command = try driver.interruptLocally() { sendVoiceControl(command, driver: driver) }
        } catch { failSession(error.localizedDescription) }
    }

    private func sendVoiceControl(_ message: [String: Any], driver: JarvisVoiceRouteDriver,
                                  afterWrite: (@MainActor () throws -> Void)? = nil) {
        guard voiceControlTasks.count < 32 else {
            failSession("Voice control queue capacity reached.")
            return
        }
        let id = UUID()
        voiceControlTasks[id] = Task { @MainActor [weak self, weak driver] in
            guard let self, let driver else { return }
            defer { self.voiceControlTasks.removeValue(forKey: id) }
            guard self.voiceRouteDriver === driver, !self.isStopping else { return }
            do {
                // Independent of microphone uploads, database work and context updates.
                try await self.realtimeClient.sendVoiceRouteEvent(message, connectionID: driver.connectionID)
                guard !Task.isCancelled, self.voiceRouteDriver === driver else { return }
                try afterWrite?()
            } catch {
                guard !Task.isCancelled, self.voiceRouteDriver === driver else { return }
                self.failSession("Voice control failed: \(error.localizedDescription)")
            }
        }
    }

    private func handleVoiceRouteEvent(_ event: JarvisRealtimeServerEvent) {
        guard !isStopping, let driver = voiceRouteDriver else { return }
        do {
            switch event.type {
            case "voice_route.selected":
                let ready = try driver.select(event)
                sendVoiceControl(ready, driver: driver) { [weak self, weak driver] in
                    guard let self, let driver, self.voiceRouteDriver === driver else { return }
                    try driver.didWriteReady(connectionID: self.realtimeClient.connectionID)
                    if let pending = self.pendingDirectReadyEvent {
                        self.pendingDirectReadyEvent = nil
                        self.handleServerEvent(pending)
                    }
                }
            case "voice_route.output_bound":
                try driver.bindOutput(event)
            case "voice_route.interrupt", "voice_route.close":
                let receipt = try driver.handleCommand(event)
                if event.type == "voice_route.close" {
                    isSessionReady = false
                    isManualRecording = false
                    hasAssistantResponseInProgress = false
                    turnState = .idle
                    connectionState = .disconnected
                    onRealtimeConnectionStateChanged?(.disconnected)
                    transcriptPreview = "Voice route closed. Your conversation and background work are retained."
                }
                sendVoiceControl(receipt, driver: driver)
            case "voice_route.event":
                try driver.receiveEvent(event)
                if voiceRouteFailure == nil, let failure = driver.failure {
                    voiceRouteFailure = failure
                    appendDiagnostic("Voice route unavailable (\(failure.rawValue)).")
                }
            default:
                throw JarvisVoiceRouteError.invalidWire
            }
        } catch { failSession(error.localizedDescription) }
    }

    private func handleCapturedVisualFrame(_ frame: CapturedVisualFrame, source: JarvisVisualCaptureSource) {
        guard visualStreamGeneration != nil, visualCoordinator.isRunning else { return }
        latestVisualFrame = frame
        visualStreamState = .active
        refreshRecordingReadiness()
        visualStatus = "\(source.label) frame #\(frame.sequence) captured at \(frame.capturedAt.formatted(date: .omitted, time: .standard))."
        localVisualFrameCount += 1
        connectVisualIngestIfPossible()
        visualIngestClient.enqueueLatestVisualFrame(frame)
#if DEBUG
        if replayRunID != nil { visualStatus = "Evaluation replay vision (not live glasses)." }
#endif
        if localVisualFrameCount == 1 || localVisualFrameCount % 10 == 0 {
            appendDiagnostic("Local visual frame #\(frame.sequence): \(frame.sizeDescription)")
        }
    }

    /// Stops only the recording consumers. Conversation/vision remain active.
    var canRequestRecording: Bool { isSessionReady && !isStopping && activeSession != nil }

    func startRecording(includeJarvisAudio: Bool) {
        guard !recordingCommandInFlight else { return }
        recordingCommand(["action": "recording.start", "include_jarvis_audio": includeJarvisAudio], stopping: nil)
    }

    func stopRecording() {
        guard let recordingID else { recordings.stop(); return }
        recordingCommand(["action": "recording.stop", "recording_id": recordingID.uuidString.lowercased()], stopping: recordingID)
    }

    private func recordingCommand(_ command: [String: Any], stopping: UUID?) {
        guard let apiClient, let sessionID = activeSession.flatMap({ UUID(uuidString: $0.id) }), canRequestRecording else {
            if stopping != nil { recordings.stop() }
            recordingCommandMessage = "Phone control is unavailable; no server command was confirmed."
            return
        }
        recordingCommandInFlight = true
        recordingCommandMessage = stopping == nil ? "Requesting recording..." : "Requesting stop..."
        Task { [weak self] in
            guard let self else { return }
            defer { self.recordingCommandInFlight = false }
            do {
                let result = try await apiClient.controlApp(sessionID: sessionID, operationID: UUID(), command: command)
                guard self.activeSession.flatMap({ UUID(uuidString: $0.id) }) == sessionID else { return }
                self.recordingCommandMessage = result.summary
                if let stopping, self.recordingID == stopping, result.deviceCommandEnqueued != true {
                    self.recordings.stop()
                }
            } catch {
                guard self.activeSession.flatMap({ UUID(uuidString: $0.id) }) == sessionID else { return }
                if let stopping, self.recordingID == stopping { self.recordings.stop() }
                self.recordingCommandMessage = stopping == nil ? error.localizedDescription
                    : "Stopped locally; the server stop is unconfirmed."
            }
        }
    }

    func handleRecordingStatus(_ status: JarvisRecordingAppController.Status) {
        recordingStatus = status.message
        isRecording = status.capturing
        recordingPhase = status.phase
        recordingID = status.recordingID
    }

    func setResourceContext(_ context: JarvisResourceSelectionContext?, sessionID: UUID?) {
        resourceSelectionContext = context
        resourceSelectionSessionID = sessionID
        resourceContextRevision += 1
        sendResourceContext()
    }

    private func sendResourceContext() {
        guard isSessionReady, !isStopping,
              let sessionID = activeSession.flatMap({ UUID(uuidString: $0.id) }),
              let audioID = audioSessionId else { return }
        let connection = realtimeClient.connectionID
        if resourceContextConnectionID != connection {
            resourceContextWrite?.cancel()
            resourceContextWrite = nil
            resourceContextConnectionID = connection
            resourceContextSequence = 0
        }
        let context = resourceSelectionSessionID == sessionID ? resourceSelectionContext : nil
        let revision = resourceContextRevision, previous = resourceContextWrite
        resourceContextWrite = Task { [weak self] in
            await previous?.value
            guard let self, !Task.isCancelled, self.isSessionReady, !self.isStopping,
                  self.realtimeClient.connectionID == connection, self.audioSessionId == audioID,
                  self.resourceContextRevision == revision else { return }
            self.resourceContextSequence += 1
            do {
                let payload = try JarvisResourceSelectionContext.envelope(context, sessionID: sessionID,
                    audioSessionID: audioID, sequence: self.resourceContextSequence)
                try await self.realtimeClient.sendWorkspaceResourceContext(payload, connectionID: connection) { [weak self] in
                    guard let self else { return false }
                    return self.isSessionReady && !self.isStopping && self.audioSessionId == audioID
                        && self.resourceContextRevision == revision
                        && self.activeSession.flatMap({ UUID(uuidString: $0.id) }) == sessionID
                }
            } catch {
                if !Task.isCancelled, self.realtimeClient.connectionID == connection {
                    self.appendDiagnostic("Workspace selection context could not be sent.")
                }
            }
        }
    }

    private var selectedRecordingAudioDevice: JarvisDuplexAudioDevice? {
        guard isSessionReady, !isStopping else { return nil }
        if usesLiveAudio {
            guard let owner = liveAppRoute, owner.driver.connectionID == realtimeClient.connectionID else { return nil }
            return owner.recordingAudioDevice
        }
        return usesLeanVoice ? directAudioClient.recordingAudioDevice : nil
    }

    private func refreshRecordingReadiness() {
        guard let device = selectedRecordingAudioDevice,
              let sessionID = activeSession.flatMap({ UUID(uuidString: $0.id) }),
              let audioID = audioSessionId, let apiClient else { recordings.retire(); return }
        let connection = realtimeClient.connectionID
        let bound = JarvisRecordingSession.Context(sessionID: sessionID, audioSessionID: audioID,
            generationID: device.voiceGeneration, connectionID: connection, protocolVersion: recordingProtocolVersion)
        recordings.configure(context: bound, audio: device, video: rayBanDATService.recordingFrames,
            api: apiClient, current: { [weak self] in
                guard let self else { return false }
                return self.isSessionReady && !self.isStopping
                    && self.realtimeClient.connectionID == connection && self.audioSessionId == audioID
                    && self.activeSession.flatMap { UUID(uuidString: $0.id) } == sessionID
                    && self.selectedRecordingAudioDevice?.voiceGeneration == bound.generationID
            })
        recordings.refresh(visionReady: visualStreamState.isActive)
    }

    private var manualBeginEnabled: Bool {
        isSessionReady && captureMode == .holdToTalk && !isManualRecording
    }

    private func recordStartupInputPredicates() {
        let at = ProcessInfo.processInfo.systemUptime
        recordStartup(startupDiagnostics?.predicate(.sessionInputReady, enabled: isSessionReady, at: at))
        recordStartup(startupDiagnostics?.predicate(.manualBeginEnabled, enabled: manualBeginEnabled, at: at))
        recordStartup(startupDiagnostics?.predicate(.automaticInputEligible,
            enabled: isSessionReady && captureMode == .autoDetect, at: at))
        if usesDirectAudio && !usesLiveAudio {
            recordStartup(startupDiagnostics?.predicate(.nativeMicrophoneEnabled,
                enabled: directAudioClient.microphoneEnabled, at: at))
        }
    }

    private func recordStartup(_ marker: JarvisStartupDiagnostics.Marker?) {
        guard let marker else { return }
        var fields = marker.fields
        fields["receive_monotonic_s"] = marker.at
        if let record = RealtimeDiagnostics.record("client_startup_observed", fields: fields) {
            logger.info("realtime_trace \(record, privacy: .public)")
        }
#if DEBUG
        // This buffer exists before the scene anchor. Never rebase the marker
        // onto scene+0 or infer response authority from local input eligibility.
        replayAudioDevice?.recordControlEvent(.clientStartupObserved, fields: marker.fields, receivedAt: marker.at)
#endif
    }

    private func connectVisualIngestIfPossible() {
        guard visualStreamGeneration != nil else { return }
        var selectedSourceReady = rayBanDATService.diagnostics.streamState == .active
#if DEBUG
        if replayRunID != nil { selectedSourceReady = visualCoordinator.isReplaySelected && visualCoordinator.isRunning }
#endif
        guard visualStartTask == nil, selectedSourceReady else { return }
        guard !visualIngestClient.isConnected else { return }
        guard !visualBackendUploadsDisabled else { return }
        guard isSessionReady else { return }
        guard visualStreamState.isActive else { return }
        guard let configuration, let realtimeConfig = activeRealtimeConfig else { return }
        guard var visualWebsocketURL = URL(string: realtimeConfig.visualWebsocketURL) else {
            visualBackendStatus = "Backend visual memory URL is invalid."
            appendDiagnostic("Invalid visual websocket URL: \(realtimeConfig.visualWebsocketURL)")
            return
        }

#if DEBUG
        if replayRunID != nil {
            do { visualWebsocketURL = try JarvisReplayScope.boundSocketURL(visualWebsocketURL,
                base: configuration.backendBaseURL, sessionID: activeSession?.id) }
            catch { failSession("Replay visual socket identity mismatch."); return }
        }
#endif
        let visualSessionId = self.visualSessionId ?? UUID().uuidString

        self.visualSessionId = visualSessionId
        visualBackendStatus = "Backend visual memory connecting."
        visualIngestClient.connect(
            visualWebSocketURL: visualWebsocketURL,
            bearerToken: configuration.staticBearerToken,
            audioSessionId: audioSessionId,
            visualSessionId: visualSessionId
        )
        appendDiagnostic("Visual websocket connecting.")
    }

    private func handleVisualServerEvent(_ event: JarvisRealtimeServerEvent) {
        switch event.type {
        case "visual.session.created":
#if DEBUG
            replayCapture?.beginPlayback()
#endif
            visualBackendStatus = "Backend visual memory ready."
            appendDiagnostic("Visual websocket created.")
        case "visual.frame.accepted":
            visualFrameAcceptedCount += 1
            let sequence = event.frameSequence.map { "#\($0)" } ?? "-"
            visualBackendStatus = "Backend accepted visual frame \(sequence)."
            if visualFrameAcceptedCount == 1 || visualFrameAcceptedCount % 10 == 0 {
                appendDiagnostic("Backend accepted visual frame \(sequence).")
            }
        case "visual.frame.dropped":
            visualFrameDroppedCount += 1
            let message = event.message ?? event.code ?? "Visual frame dropped."
            visualBackendStatus = "Backend dropped visual frame: \(message)"
            if event.code == "source_not_allowed" || event.code == "storage_not_configured" {
                visualBackendUploadsDisabled = true
                visualIngestClient.disconnect()
            }
            appendDiagnostic(visualBackendStatus)
        case "error":
            let message = event.message ?? event.code ?? "Visual backend error."
            visualBackendStatus = "Backend visual memory error: \(message)"
            if event.recoverable == false {
                visualBackendUploadsDisabled = true
                visualIngestClient.disconnect()
            }
            appendDiagnostic(visualBackendStatus)
        default:
            appendDiagnostic("Visual event: \(event.type)")
        }
    }

    private func syncVisualState(with diagnostics: RayBanDATDiagnostics) {
#if DEBUG
        if replayRunID != nil {
            guard visualStreamGeneration != nil, visualStartTask == nil else { return }
            visualStreamState = visualCoordinator.isRunning ? .active : .stopped
            visualStatus = "Evaluation replay vision (not live glasses)."
            connectVisualIngestIfPossible()
            return
        }
#endif
        // Startup reconciles the latest diagnostics when its task settles.
        // Notifications cannot revive a stopped or failed generation.
        guard visualStreamGeneration != nil, visualStartTask == nil else { return }
        switch diagnostics.streamState {
        case .stopped:
            stopLocalVisualStream()
            return
        case .starting:
            visualStreamState = .starting
            visualStatus = "Starting Ray-Ban vision..."
        case .active:
            visualStreamState = .active
            connectVisualIngestIfPossible()
        case .paused:
            visualStatus = "Ray-Ban vision paused."
        case .failed:
            failLocalVisualStream(diagnostics.lastError ?? "Ray-Ban vision failed.")
            return
        }

        if let lastError = diagnostics.lastError {
            visualStatus = lastError
        } else if diagnostics.streamState == .active {
            visualStatus = "Ray-Ban vision active."
        }
    }

    private func updateRouteDiagnostics(reason: String) {
#if DEBUG
        if replayRunID != nil {
            routeSummary = "Evaluation custom ADM: no physical microphone or speaker."
            routeUsesGlasses = false
            return
        }
#endif
        let snapshot = routeDiagnostics.logCurrentRoute(reason: reason)
        routeSummary = snapshot.summary
        routeUsesGlasses = snapshot.appearsToUseGlasses
        appendDiagnostic("\(reason): \(snapshot.summary)")
    }

    private func observeRouteChanges() {
        NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.updateRouteDiagnostics(reason: "Audio route changed")
            }
        }
    }

    private func appendTranscript(role: JarvisTranscriptEntry.Role, text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        transcriptPreview = trimmed
        appendDiagnostic("\(role.rawValue): \(trimmed)")
    }

    private func appendDiagnostic(_ message: String) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        logger.info("\(trimmed, privacy: .public)")
        diagnostics.append(trimmed)
        if diagnostics.count > 40 {
            diagnostics.removeFirst(diagnostics.count - 40)
        }
    }
}
