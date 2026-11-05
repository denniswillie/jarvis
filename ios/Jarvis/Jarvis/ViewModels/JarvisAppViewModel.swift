import Foundation

enum JarvisAppSheet: String, Identifiable {
    case settings, preferences, reminders, work, files, offlineCopies
    var id: String { rawValue }
}

enum JarvisProductTab: Hashable { case home, workspace, audiovisual, plugins }

struct JarvisProductWorkLink: Identifiable, Equatable {
    enum Destination: Equatable { case task(UUID), inbox(UUID) }
    let id = UUID()
    let destination: Destination
}

@MainActor
final class JarvisAppViewModel: ObservableObject {
    @Published private(set) var state: JarvisAppState = .launching
    @Published private(set) var configuration: JarvisConfiguration?
    @Published private(set) var activeSession: JarvisSession?
    @Published private(set) var transcriptPreview: String = "Jarvis is ready for a new session."
    @Published private(set) var visualStatus: String = "Glasses standby"
    @Published private(set) var phoneCameraDebugEnabled: Bool
    @Published private(set) var audioProvider: JarvisAudioProvider
    @Published private(set) var setupErrorMessage: String?
    @Published private(set) var activeReminders: [JarvisReminderSummary] = []
    @Published private(set) var reminderEvents: [JarvisReminderEvent] = []
    @Published private(set) var reminderRefreshError: String?
    @Published private(set) var reminderLastRefreshedAt: Date?
    @Published private(set) var googleCalendarConnected = false
    @Published private(set) var googleCalendarStatusError: String?
    @Published private(set) var isGoogleCalendarStatusLoading = false
    @Published private(set) var isGoogleCalendarConnecting = false
    @Published private(set) var memoryLoadState: JarvisMemoryLoadState = .idle
    @Published private(set) var recentMemories: [JarvisMemoryItem] = []
    @Published private(set) var memoryNextCursor: String?
    @Published private(set) var memoryView: String?
    @Published private(set) var isLoadingMoreMemories = false
    @Published private(set) var memoryLastRefreshedAt: Date?
    @Published private(set) var workingMemoryLoadState: JarvisMemoryLoadState = .idle
    @Published private(set) var workingMemory: JarvisWorkingMemoryState?
    @Published private(set) var selectedMemoryID: String?
    @Published private(set) var selectedMemoryDetail: JarvisMemoryDetail?
    @Published private(set) var memoryDetailLoadState: JarvisMemoryLoadState = .idle
    @Published private(set) var isLoadingMemoryContent = false
    @Published private(set) var workInbox: JarvisWorkInboxViewModel?
    @Published private(set) var pushNotifications: JarvisPushViewModel?
    @Published private(set) var notificationInboxLink: JarvisNotificationLink?
    @Published var presentedSheet: JarvisAppSheet?
    @Published var productTab: JarvisProductTab = .home
    @Published private(set) var workspace: JarvisWorkspaceViewModel?
    @Published private(set) var savedFiles: JarvisFilesViewModel?
    @Published private(set) var audiovisual: JarvisFilesViewModel?
    @Published private(set) var audiovisualTransferMessage: String?
    @Published private(set) var preferences: JarvisPreferencesViewModel?
    @Published private(set) var plugins: JarvisPluginsViewModel?
    @Published private(set) var isPreparingSavedCopies = false
    @Published private(set) var savedCopiesPreparationError: String?
    @Published private(set) var workProductLink: JarvisProductWorkLink?
    @Published private(set) var productConfigurationGeneration = UUID()
    var isShowingWorkInbox: Bool {
        get { presentedSheet == .work }
        set {
            if newValue { presentedSheet = .work }
            else if presentedSheet == .work { presentedSheet = nil }
        }
    }
    var isShowingSettings: Bool {
        get { presentedSheet == .settings }
        set {
            if newValue { presentedSheet = .settings }
            else if presentedSheet == .settings { presentedSheet = nil }
        }
    }

    let audioSessionViewModel: JarvisAudioSessionViewModel
    let appLinkPresenter = JarvisAppLinkPresenter()
    let locationService = JarvisLocationService()

    private let store: JarvisSessionStore
    private let googleCalendarAuthorizationSession: GoogleCalendarAuthorizationSession
    private let workspaceJournalFactory: (JarvisAuthenticatedContext) throws -> JarvisWorkspaceLocalStore
    private var apiClient: JarvisAPIClient?
    private var activeSessionConfigurationGeneration: UUID?
    private var replacedProductSessionID: String?
    private var verifiedWorkspaceContext: JarvisAuthenticatedContext?
    private var workspaceContextTask: Task<Void, Never>?
    private var workspaceContextTaskID: UUID?
    private var recordingRefreshTask: Task<Void, Never>?
    private var audiovisualAccess: JarvisWorkspaceLocalAccess?
    private var workspaceContextRetryUntil: Date?
    private var savedCopiesPreparationTask: Task<Void, Never>?
    private var savedCopiesPreparationID: UUID?
    private var preparingSavedFileLibrary = false
    private var savedFilesAccess: JarvisWorkspaceLocalAccess?
    private var productForeground = false
    private var freshPresentationTask: Task<Void, Never>?
    private var freshPresentationTaskID: UUID?
    private var recentPresentationIntents: [UUID] = []
    private var awaitingSheetDismissal = false
    private var reminderReadGeneration = UUID()
    private var memoryRefreshTask: Task<Void, Never>?
    private var memoryFallbackPollTask: Task<Void, Never>?
    private var pendingRecentMemoryRefresh = false
    private var pendingWorkingMemoryRefresh = false
    private var pendingDetailMemoryID: String?
    private var memoryListGeneration = 0
    private var workingMemoryRequestGeneration = 0
    private var memoryDetailRequestGeneration = 0
    private var highestSessionWorkingRevisionBySession: [String: Int64] = [:]
    private var highestOwnerWorkingRevision: Int64 = -1

#if DEBUG
    private var productEvaluationRunID: UUID?

    /// Bind the existing product HTTP client to the same isolated replay host.
    /// Must precede configuration, so even startup reads carry its admission header.
    func bindProductEvaluationRun(_ runID: UUID) throws {
        guard configuration == nil, apiClient == nil, productEvaluationRunID == nil else {
            throw JarvisReplayError.invalidPlan
        }
        productEvaluationRunID = runID
    }
#endif

    init(
        store: JarvisSessionStore = .shared,
        audioSessionViewModel: JarvisAudioSessionViewModel? = nil,
        googleCalendarAuthorizationSession: GoogleCalendarAuthorizationSession? = nil,
        workspaceJournalFactory: @escaping (JarvisAuthenticatedContext) throws -> JarvisWorkspaceLocalStore = { context in
            try JarvisWorkspaceLocalStore.verifiedAccountJournal(for: context)
        }
    ) {
        let resolvedAudioSessionViewModel = audioSessionViewModel ?? JarvisAudioSessionViewModel()
        self.store = store
        self.audioSessionViewModel = resolvedAudioSessionViewModel
        self.googleCalendarAuthorizationSession = googleCalendarAuthorizationSession ?? GoogleCalendarAuthorizationSession()
        self.workspaceJournalFactory = workspaceJournalFactory
        self.phoneCameraDebugEnabled = store.isPhoneCameraDebugEnabled
        self.audioProvider = store.audioProvider
        resolvedAudioSessionViewModel.setPreferredAudioProvider(store.audioProvider)
        resolvedAudioSessionViewModel.onReminderStateChanged = { [weak self] in
            self?.refreshReminderState()
        }
        resolvedAudioSessionViewModel.onMemoryStateChanged = { [weak self] hint in
            self?.handleMemoryRefreshHint(hint)
        }
        resolvedAudioSessionViewModel.onWorkingMemoryStateChanged = { [weak self] hint in
            self?.handleWorkingMemoryRefreshHint(hint)
        }
        resolvedAudioSessionViewModel.onRealtimeConnectionStateChanged = { [weak self] state in
            self?.handleRealtimeConnectionStateChanged(state)
        }
        resolvedAudioSessionViewModel.onWorkspacePresentationIntent = { [weak self] sessionID, requestID in
            self?.handleWorkspacePresentationIntent(sessionID: sessionID, requestID: requestID)
        }
        resolvedAudioSessionViewModel.onAppLinkRequested = { [weak self] link in
            guard let self, let api = self.apiClient, self.configuration != nil else { return }
            let generation = self.productConfigurationGeneration
            self.appLinkPresenter.receive(link, api: api) { [weak self] in
                guard let self else { return false }
                return self.productConfigurationGeneration == generation
                    && self.replacedProductSessionID != link.sessionID.uuidString.lowercased()
                    && self.audioSessionViewModel.connectionState.isConnected
                    && self.audioSessionViewModel.activeSession.flatMap({ UUID(uuidString: $0.id) }) == link.sessionID
                    && self.audioSessionViewModel.audioSessionId == link.audioSessionID
            }
        }
        resolvedAudioSessionViewModel.onSandboxViewStopped = { [weak self] sessionID, viewID in
            guard let self, self.workspace?.sessionID == sessionID,
                  self.verifiedWorkspaceContext?.sessionID == sessionID else { return }
            self.workspace?.sandbox.receiveStop(viewID: viewID)
        }
        resolvedAudioSessionViewModel.onRecordingSaved = { [weak self] saved in
            self?.refreshConfirmedRecording(saved)
        }
        resolvedAudioSessionViewModel.onBehaviorProfileCommitted = { [weak self] sessionID, event in
            guard let self, self.verifiedWorkspaceContext?.sessionID == sessionID,
                  self.verifiedWorkspaceContext?.ownerID == event.ownerID,
                  let model = self.preferences else { return }
            Task { await model.committed(event) }
        }
        JarvisNotificationService.shared.onToken = { [weak self] token in
            self?.pushNotifications?.receivedToken(token)
        }
        JarvisNotificationService.shared.onRegistrationFailure = { [weak self] in
            self?.pushNotifications?.registrationFailed()
        }
        JarvisNotificationService.shared.onOpenInbox = { [weak self] in
            self?.openPendingNotification()
        }
    }

    func bootstrap() async {
        guard state == .launching else { return }
        guard let configuration = store.loadConfiguration() else {
            state = .needsSetup
            return
        }
        applyConfiguration(configuration)
        state = .ready
        await loadReminderState()
        await loadGoogleCalendarStatus()
        await reloadMemoryState()
    }

    func saveSetup(backendBaseURL: String, staticBearerToken: String) {
        guard canReplaceNotificationConfiguration() else { return }
        do {
            let configuration = try store.saveConfiguration(
                backendBaseURL: backendBaseURL,
                staticBearerToken: staticBearerToken
            )
            setupErrorMessage = nil
            applyConfiguration(configuration)
            state = .ready
            refreshGoogleCalendarStatus()
            refreshMemoryState()
        } catch {
            setupErrorMessage = error.localizedDescription
            state = .needsSetup
        }
    }

    func useBundledConfiguration() {
        guard canReplaceNotificationConfiguration() else { return }
        guard let configuration = store.loadConfiguration() else {
            setupErrorMessage = "Bundled Jarvis configuration is missing."
            state = .needsSetup
            return
        }
        setupErrorMessage = nil
        applyConfiguration(configuration)
        state = .ready
        refreshMemoryState()
    }

    func startSession() {
        guard state != .connecting else { return }
        guard let apiClient else {
            state = .needsSetup
            return
        }

        state = .connecting
        transcriptPreview = "Opening a private Jarvis session..."
        let generation = productConfigurationGeneration
        Task {
            guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
            do {
                let response = try await apiClient.startSession(title: nil, resumeActive: true)
                guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
                activeSessionConfigurationGeneration = generation
                activeSession = response.session
                visualStatus = "Glasses ready"
                transcriptPreview = "Session active. Jarvis will listen through the glasses in a later phase."
                state = .active
                await loadReminderState()
            } catch {
                guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
                transcriptPreview = "Could not start Jarvis."
                state = .error(error.localizedDescription)
            }
        }
    }

    func stopSession() {
        guard let apiClient, let activeSession else {
            activeSessionConfigurationGeneration = nil
            self.activeSession = nil
            state = .ready
            return
        }

        transcriptPreview = "Closing Jarvis session..."
        let generation = productConfigurationGeneration
        Task {
            guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
            do {
                try await apiClient.closeSession(id: activeSession.id)
                guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
                activeSessionConfigurationGeneration = nil
                self.activeSession = nil
                visualStatus = "Glasses standby"
                transcriptPreview = "Session closed."
                state = .ready
            } catch {
                guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
                state = .error(error.localizedDescription)
            }
        }
    }

    /// Resolve existing account work for a cold phone. This is GET-only: no
    /// voice, Jarvis session creation, sandbox allocation or resumption.
    func openExistingSandbox() async -> JarvisSandboxViewModel.Notice? {
        guard let apiClient else { return nil }
        let generation = productConfigurationGeneration
        do {
            if workspace == nil {
                let existing = try await apiClient.getActiveSession()
                guard generation == productConfigurationGeneration, !Task.isCancelled,
                      productForeground, productTab == .workspace, presentedSheet == nil else { return nil }
                if audioSessionViewModel.activeSession == nil, productFallbackSessionID == nil {
                    activeSessionConfigurationGeneration = generation
                    activeSession = existing
                }
                let raw = audioSessionViewModel.activeSession?.id ?? productFallbackSessionID
                synchronizeProductSession(raw)
            }
            guard generation == productConfigurationGeneration, !Task.isCancelled,
                  productForeground, productTab == .workspace, presentedSheet == nil else { return nil }
            guard let workspace else {
                return .init(title: "No sandbox to view", message: "Ask Jarvis to start a sandbox, then view its screen here.")
            }
            await workspace.sandbox.choose()
            return nil
        } catch {
            guard generation == productConfigurationGeneration, !Task.isCancelled else { return nil }
            return .init(title: "Can’t connect", message: "Your active sandbox couldn't be checked. Try again when connected.")
        }
    }

    func refreshActiveSession() {
        guard let apiClient else { return }
        let generation = productConfigurationGeneration
        Task {
            guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
            do {
                let session = try await apiClient.getActiveSession()
                guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
                activeSessionConfigurationGeneration = generation
                activeSession = session
                if activeSession != nil, state == .ready {
                    transcriptPreview = "An active Jarvis session is available."
                }
            } catch {
                guard generation == productConfigurationGeneration, !Task.isCancelled else { return }
                transcriptPreview = "Could not refresh session state."
            }
        }
    }

    @discardableResult
    func refreshReminderState() -> Task<Void, Never> {
        Task {
            await loadReminderState()
        }
    }

    func refreshMemoryState() {
        scheduleMemoryRefresh(recent: true, working: true)
    }

    func refreshForegroundState() {
        refreshWorkspaceContext()
        refreshAccountProducts()
        Task { await pushNotifications?.refresh() }
        openPendingNotification()
        refreshReminderState()
        refreshMemoryState()
    }

    func reloadMemoryState() async {
        await loadRecentMemories(replacing: true)
        await loadWorkingMemory()
    }

    func loadMoreMemories() async {
        await loadRecentMemories(replacing: false)
    }

    func loadMemoryDetail(id: String) async {
        guard let apiClient else { return }
        memoryDetailRequestGeneration += 1
        isLoadingMemoryContent = false
        let generation = memoryDetailRequestGeneration
        pendingDetailMemoryID = nil
        selectedMemoryID = id
        selectedMemoryDetail = nil
        memoryDetailLoadState = .loading
        do {
            let detail = try await apiClient.getMemoryDetail(id: id)
            guard
                generation == memoryDetailRequestGeneration,
                selectedMemoryID == id,
                detail.memory.id == id
            else { return }
            selectedMemoryDetail = detail
            memoryDetailLoadState = .loaded
        } catch {
            guard generation == memoryDetailRequestGeneration, selectedMemoryID == id else {
                return
            }
            memoryDetailLoadState = .degraded(error.localizedDescription)
        }
    }

    func loadMoreMemoryContent() async {
        guard let apiClient, !isLoadingMemoryContent, let detail = selectedMemoryDetail,
              let source = detail.memory.sourceEvidence, let offset = detail.contentPage?.nextOffset,
              selectedMemoryID == detail.memory.id else { return }
        let generation = memoryDetailRequestGeneration
        isLoadingMemoryContent = true
        defer {
            if generation == memoryDetailRequestGeneration { isLoadingMemoryContent = false }
        }
        do {
            let next = try await apiClient.getMemoryDetail(id: detail.memory.id,
                revision: source.revision, offset: offset, expectedDigest: source.contentDigest)
            guard generation == memoryDetailRequestGeneration, selectedMemoryID == detail.memory.id else { return }
            selectedMemoryDetail = try detail.appendingOriginalPage(next)
        } catch {
            guard generation == memoryDetailRequestGeneration, selectedMemoryID == detail.memory.id else { return }
            // A removed or changed original must not leave its old body on screen.
            selectedMemoryDetail = nil
            memoryDetailLoadState = .degraded(error.localizedDescription)
        }
    }

    func clearMemoryDetail() {
        memoryDetailRequestGeneration += 1
        isLoadingMemoryContent = false
        selectedMemoryID = nil
        selectedMemoryDetail = nil
        memoryDetailLoadState = .idle
        pendingDetailMemoryID = nil
    }

    func resetConfiguration(forgetWithoutServerRevocation: Bool = false) {
        if forgetWithoutServerRevocation {
            pushNotifications?.prepareForLocalReset()
        } else {
            guard canReplaceNotificationConfiguration() else { return }
        }
        appLinkPresenter.reset()
        pushNotifications?.deactivate()
        pushNotifications = nil
        presentedSheet = nil
        cancelSavedCopiesPreparation()
        retireWorkspaceContext()
        preferences?.deactivate(); preferences = nil
        plugins?.deactivate(); plugins = nil
        productForeground = false
        productConfigurationGeneration = UUID()
        notificationInboxLink = nil
        workProductLink = nil
        store.clearConfiguration()
        audioProvider = .openai
        audioSessionViewModel.setPreferredAudioProvider(.openai)
        workInbox?.deactivate()
        workInbox = nil
        workspace?.deactivate()
        workspace = nil
        productTab = .home
        // Core clears its audio session only after asynchronous stop completes.
        // A new configuration must not attach that still-published old ID.
        replacedProductSessionID = (audioSessionViewModel.activeSession?.id ?? activeSession?.id)?.lowercased()
        configuration = nil
        apiClient = nil
        activeSessionConfigurationGeneration = nil
        activeSession = nil
        setupErrorMessage = nil
        clearReminderState()
        googleCalendarConnected = false
        googleCalendarStatusError = nil
        memoryRefreshTask?.cancel()
        memoryRefreshTask = nil
        memoryFallbackPollTask?.cancel()
        memoryFallbackPollTask = nil
        pendingRecentMemoryRefresh = false
        pendingWorkingMemoryRefresh = false
        pendingDetailMemoryID = nil
        memoryListGeneration += 1
        workingMemoryRequestGeneration += 1
        memoryDetailRequestGeneration += 1
        isLoadingMemoryContent = false
        isLoadingMoreMemories = false
        recentMemories = []
        memoryNextCursor = nil
        memoryView = nil
        memoryLoadState = .idle
        memoryLastRefreshedAt = nil
        workingMemory = nil
        workingMemoryLoadState = .idle
        selectedMemoryID = nil
        selectedMemoryDetail = nil
        memoryDetailLoadState = .idle
        highestSessionWorkingRevisionBySession = [:]
        highestOwnerWorkingRevision = -1
        audioSessionViewModel.stopSession()
        state = .needsSetup
    }

    func setPhoneCameraDebugEnabled(_ enabled: Bool) {
        store.isPhoneCameraDebugEnabled = enabled
        phoneCameraDebugEnabled = enabled
    }

    func setAudioProvider(_ provider: JarvisAudioProvider) {
        store.audioProvider = provider
        audioProvider = store.audioProvider
        audioSessionViewModel.setPreferredAudioProvider(audioProvider)
    }

    func refreshGoogleCalendarStatus() {
        Task { await loadGoogleCalendarStatus() }
    }

    func connectGoogleCalendar() {
        guard let apiClient, !isGoogleCalendarConnecting else { return }
        isGoogleCalendarConnecting = true
        googleCalendarStatusError = nil
        Task {
            defer { isGoogleCalendarConnecting = false }
            do {
                let response = try await apiClient.startGoogleCalendarAuthorization()
                try await googleCalendarAuthorizationSession.authorize(at: response.authorizationURL)
                await loadGoogleCalendarStatus()
            } catch is CancellationError {
                return
            } catch {
                googleCalendarStatusError = error.localizedDescription
            }
        }
    }

    var googleCalendarStatusLabel: String {
        if isGoogleCalendarStatusLoading { return "Checking" }
        if googleCalendarConnected { return "Connected" }
        if googleCalendarStatusError != nil { return "Needs attention" }
        return "Not connected"
    }

    var backendDisplayText: String {
        configuration?.backendBaseURL.absoluteString ?? "Not configured"
    }

    var hasBundledConfiguration: Bool {
        store.bundledConfigurationAvailable()
    }

    var stateLabel: String {
        switch state {
        case .launching: return "Launching"
        case .needsSetup: return "Setup needed"
        case .ready: return "Ready"
        case .connecting: return "Connecting"
        case .active: return "Active"
        case .error: return "Needs attention"
        }
    }

    var isSessionActive: Bool {
        if case .active = state { return true }
        return false
    }

    var reminderSyncStatusText: String {
        if let reminderRefreshError {
            return "Sync failed: \(reminderRefreshError)"
        }
        guard let reminderLastRefreshedAt else { return "Not synced yet" }
        return "Synced \(reminderLastRefreshedAt.formatted(date: .omitted, time: .shortened))"
    }

    var sourceBackedMemory: Bool { memoryView == "original_sources" }

    var recentMemoryHighlights: [JarvisMemoryItem] {
        recentMemories.filter {
            $0.status == "active" && ($0.sourceEvidence != nil || $0.memoryType == "episode")
        }
    }

    var currentSceneSummary: String? {
        workingMemory?.session?.state.jarvisLabel(for: "scene")
            ?? workingMemory?.owner?.state.jarvisLabel(for: "scene")
    }

    var currentActivitySummary: String? {
        workingMemory?.session?.state.jarvisLabel(for: "activity")
            ?? workingMemory?.owner?.state.jarvisLabel(for: "activity")
    }

    var errorMessage: String? {
        if case let .error(message) = state { return message }
        return nil
    }

    private func applyConfiguration(_ configuration: JarvisConfiguration) {
        appLinkPresenter.reset()
        if let previous = self.configuration, previous != configuration {
            // Existing audio/session ownership stays with Core. Do not attach a
            // prior account's session to the newly configured product client.
            replacedProductSessionID = (audioSessionViewModel.activeSession?.id ?? activeSession?.id)?.lowercased()
        }
        cancelSavedCopiesPreparation()
        retireWorkspaceContext()
        self.configuration = configuration
        productConfigurationGeneration = UUID()
        clearReminderState()
        activeSessionConfigurationGeneration = nil
        activeSession = nil
        self.pushNotifications?.deactivate()
        self.workInbox?.deactivate()
        self.workspace?.deactivate()
        self.workspace = nil
        self.preferences?.deactivate()
        self.plugins?.deactivate()
        self.locationService.deactivate()
        let client = JarvisAPIClient(configuration: configuration)
#if DEBUG
        client.evaluationRunID = productEvaluationRunID
#endif
        self.apiClient = client
        self.preferences = makePreferences(api: client)
        let pluginsModel = makePlugins(api: client)
        self.plugins = pluginsModel
        self.workInbox = JarvisWorkInboxViewModel(api: client)
        // Remote/APNs push is out of scope. Do not load an old opt-in, register,
        // or contact push endpoints. In-app updates and reminders stay separate.
        self.pushNotifications = nil
        self.locationService.activate(api: client)
        Task { await pluginsModel.refresh() }
        self.audioSessionViewModel.configure(configuration: configuration)
    }

    private func makePreferences(api: JarvisAPIClient) -> JarvisPreferencesViewModel {
        let profile = JarvisPreferencesViewModel(api: api)
        profile.onAuthorizationFailure = { [weak self, weak profile] in
            guard let self, let profile, self.preferences === profile else { return }
            self.plugins?.lockForSignIn()
            self.workspace?.lockJournal(clearRemoteContent: true)
            self.retireWorkspaceContext()
        }
        return profile
    }


    private func makePlugins(api: JarvisAPIClient) -> JarvisPluginsViewModel {
        let plugins = JarvisPluginsViewModel(api: api)
        plugins.onAuthorizationFailure = { [weak self, weak plugins] in
            guard let self, let plugins, self.plugins === plugins else { return }
            self.preferences?.lockForSignIn()
            self.workspace?.lockJournal(clearRemoteContent: true)
            self.retireWorkspaceContext()
        }
        return plugins
    }

    /// Session identity comes from the existing session owner. Browsing does
    /// not construct a persistent store until Core supplies a trusted owner key.
    var productFallbackSessionID: String? {
        guard activeSessionConfigurationGeneration == productConfigurationGeneration else { return nil }
        return activeSession?.id
    }

    func synchronizeProductSession(_ rawID: String?) {
        if rawID == nil { replacedProductSessionID = nil }
        let id = rawID.flatMap(UUID.init(uuidString:))
        let isReplacedSession = rawID?.lowercased() == replacedProductSessionID && replacedProductSessionID != nil
        if isReplacedSession {
            retireWorkspaceContext()
            workspace?.deactivate(); workspace = nil
            return
        }
        guard workspace?.sessionID != id else { return }
        retireWorkspaceContext()
        workspace?.deactivate()
        workspace = nil
        guard let id, let apiClient else { return }
        workspace = makeWorkspace(api: apiClient, sessionID: id)
        refreshWorkspaceContext()
    }

    private func makeWorkspace(api: JarvisAPIClient, sessionID: UUID) -> JarvisWorkspaceViewModel {
        let model = JarvisWorkspaceViewModel(api: api, sessionID: sessionID)
        model.onAuthorizationFailure = { [weak self, weak model] in
            guard let self, let model, self.workspace === model else { return }
            self.preferences?.lockForSignIn()
            self.plugins?.lockForSignIn()
            self.retireWorkspaceContext()
        }
        return model
    }

    /// Core's session/audio startup never awaits this independent context read.
    /// Only a current, verified return may unlock the existing owner journal.
    func refreshWorkspaceContext() {
        guard let model = workspace, let api = apiClient, workspaceContextTaskID == nil else { return }
        if verifiedWorkspaceContext?.sessionID == model.sessionID, model.canHandleDeliveries { return }
        if let deadline = workspaceContextRetryUntil, deadline > Date() { return }
        let generation = productConfigurationGeneration, taskID = UUID(), sessionID = model.sessionID
        workspaceContextTaskID = taskID
        workspaceContextTask = Task { [weak self, weak model] in
            guard let self, let model, !Task.isCancelled,
                  self.productConfigurationGeneration == generation, self.workspace === model else { return }
            defer {
                if self.workspaceContextTaskID == taskID {
                    self.workspaceContextTaskID = nil; self.workspaceContextTask = nil
                }
            }
            do {
                let context = try await api.fetchAuthenticatedContext(sessionID: sessionID)
                guard !Task.isCancelled, self.productConfigurationGeneration == generation,
                      self.workspaceContextTaskID == taskID, self.workspace === model,
                      context.sessionID == sessionID, context.hasValidIdentity else { return }
                // Construct synchronously only after the verified context and
                // lifetime checks; the account journal retains its access fence.
                let journal = try self.workspaceJournalFactory(context)
                if !model.attachJournal(journal) {
                    // A revoked lifetime is never unlocked in place. Reverification
                    // creates a fresh model/access fence, without changing audio.
                    model.deactivate()
                    let replacement = self.makeWorkspace(api: api, sessionID: sessionID)
                    guard replacement.attachJournal(journal) else { return }
                    self.workspace = replacement
                }
                self.verifiedWorkspaceContext = context
                if self.preferences?.isLocked == true {
                    self.preferences?.deactivate()
                    self.preferences = self.makePreferences(api: api)
                }
                self.preferences?.verifyOwner(context.ownerID)
                if self.plugins?.isLocked == true {
                    self.plugins?.deactivate()
                    self.plugins = self.makePlugins(api: api)
                }
                self.workspaceContextRetryUntil = nil
                // Startup reads can predate commit hints that were correctly
                // ignored until this owner was verified. Catch up from the
                // authoritative account routes after closing that window.
                if self.verifiedWorkspaceContext?.sessionID == sessionID {
                    self.refreshAccountProducts()
                }
            } catch {
                guard !Task.isCancelled, self.productConfigurationGeneration == generation,
                      self.workspaceContextTaskID == taskID, self.workspace === model else { return }
                self.verifiedWorkspaceContext = nil
                let status = (error as? JarvisAPIError)?.statusCode
                let requiresSignIn = status == 401 || status == 403
                if requiresSignIn {
                    self.preferences?.lockForSignIn()
                    self.plugins?.lockForSignIn()
                    model.lockJournal(clearRemoteContent: true)
                } else if model.canHandleDeliveries {
                    model.lockJournal()
                }
                if let delay = (error as? JarvisAPIError)?.retryAfter, delay.isFinite, delay >= 0 {
                    self.workspaceContextRetryUntil = Date().addingTimeInterval(delay)
                }
                model.deliveryContextUnavailable(requiresSignIn: requiresSignIn)
            }
        }
    }

    private func retireWorkspaceContext() {
        audioSessionViewModel.setResourceContext(nil, sessionID: nil)
        audiovisualAccess?.invalidate(); audiovisualAccess = nil
        audiovisual?.deactivate(); audiovisual = nil
        audiovisualTransferMessage = nil
        recordingRefreshTask?.cancel(); recordingRefreshTask = nil
        retireSavedFiles()
        workspaceContextTask?.cancel(); workspaceContextTask = nil; workspaceContextTaskID = nil
        verifiedWorkspaceContext = nil; workspaceContextRetryUntil = nil
        freshPresentationTask?.cancel(); freshPresentationTask = nil; freshPresentationTaskID = nil
        recentPresentationIntents = []
    }

    private func refreshConfirmedRecording(_ saved: JarvisRecordingAppController.Saved) {
        recordingRefreshTask?.cancel()
        let generation = productConfigurationGeneration
        recordingRefreshTask = Task { [weak self] in
            guard let self else { return }
            self.refreshWorkspaceContext()
            await self.workspaceContextTask?.value
            guard !Task.isCancelled, self.productConfigurationGeneration == generation,
                  let context = self.verifiedWorkspaceContext, context.sessionID == saved.sessionID,
                  context.ownerID == saved.ownerID, context.backendNamespace == saved.backendNamespace else { return }
            self.audiovisualTransferMessage = nil
            await self.audiovisual?.reload()
        }
    }

    private func refreshAccountProducts() {
        let profile = preferences
        let pluginsModel = plugins
        locationService.syncNow()
        Task {
            await profile?.refresh()
            await pluginsModel?.refresh(clearError: false)
        }
    }

    /// Root is the existing single scene/visibility owner. A user navigation or
    /// sheet/background transition cancels an unrendered fresh voice request.
    func updateProductScene(foreground: Bool) {
        productForeground = foreground
        if !foreground || presentedSheet != nil || awaitingSheetDismissal || productTab != .workspace {
            freshPresentationTask?.cancel(); freshPresentationTask = nil; freshPresentationTaskID = nil
        }
    }

    private func handleWorkspacePresentationIntent(sessionID: UUID, requestID: UUID) {
        guard productForeground, presentedSheet == nil, !awaitingSheetDismissal, let model = workspace,
              model.sessionID == sessionID, verifiedWorkspaceContext?.sessionID == sessionID,
              model.canHandleDeliveries, !recentPresentationIntents.contains(requestID) else { return }
        recentPresentationIntents.append(requestID)
        if recentPresentationIntents.count > 64 { recentPresentationIntents.removeFirst() }
        freshPresentationTask?.cancel()
        let generation = productConfigurationGeneration, taskID = UUID()
        freshPresentationTaskID = taskID
        productTab = .workspace
        model.setVisibility(true, foreground: true, workVisible: true)
        freshPresentationTask = Task { [weak self, weak model] in
            guard let self, let model, !Task.isCancelled,
                  self.productConfigurationGeneration == generation, self.workspace === model,
                  self.productForeground, self.presentedSheet == nil, !self.awaitingSheetDismissal,
                  self.productTab == .workspace else { return }
            defer {
                if self.freshPresentationTaskID == taskID {
                    self.freshPresentationTaskID = nil; self.freshPresentationTask = nil
                }
            }
            await model.presentFreshRequest(requestID)
        }
    }

    func showWorkInbox() {
        guard workInbox != nil else { return }
        notificationInboxLink = nil
        workProductLink = nil
        isShowingWorkInbox = true
    }

    func showSavedFiles() {
        guard configuration != nil else { return }
        presentedSheet = .files
    }

    func prepareSavedFiles() {
        prepareAccountFiles(forLibrary: true)
    }

    func prepareAudiovisual() {
        guard audiovisual == nil else { return }
        prepareAccountFiles(forLibrary: true)
    }

    func closeSavedFiles() {
        if preparingSavedFileLibrary { cancelSavedCopiesPreparation() }
        retireSavedFiles()
    }

    private func retireSavedFiles() {
        savedFilesAccess?.invalidate()
        savedFilesAccess = nil
        savedFiles?.deactivate()
        savedFiles = nil
    }

    func showSavedCopies() {
        guard configuration != nil else { return }
        presentedSheet = .offlineCopies
        prepareSavedCopies()
    }

    /// Explicit local-copy navigation may prepare an existing data session.
    /// It never changes voice, capture, listening copy or the app's active state.
    func prepareSavedCopies() {
        prepareAccountFiles(forLibrary: false)
    }

    private func prepareAccountFiles(forLibrary: Bool) {
        guard let api = apiClient else { return }
        preparingSavedFileLibrary = forLibrary
        guard savedCopiesPreparationTask == nil else { return }
        savedCopiesPreparationError = nil
        let generation = productConfigurationGeneration, taskID = UUID()
        savedCopiesPreparationID = taskID
        isPreparingSavedCopies = true
        savedCopiesPreparationTask = Task { [weak self] in
            guard let self, !Task.isCancelled, self.productConfigurationGeneration == generation,
                  self.savedCopiesPreparationID == taskID else { return }
            defer {
                if self.savedCopiesPreparationID == taskID {
                    self.savedCopiesPreparationTask = nil
                    self.savedCopiesPreparationID = nil
                    self.isPreparingSavedCopies = false
                }
            }
            do {
                if let model = self.workspace, model.canHandleDeliveries,
                   self.verifiedWorkspaceContext?.sessionID == model.sessionID {
                    await self.finishAccountFilesPreparation(model: model, api: api)
                    return
                }
                var rawSession = self.savedCopiesDataSessionID
                if rawSession == nil {
                    let response = try await api.startSession(title: nil, resumeActive: true)
                    guard !Task.isCancelled, self.productConfigurationGeneration == generation,
                          self.savedCopiesPreparationID == taskID else { return }
                    // Voice may have established its own current session while
                    // the data-only request was in flight. Preserve that owner.
                    rawSession = self.savedCopiesDataSessionID
                    if rawSession == nil {
                        self.activeSessionConfigurationGeneration = generation
                        self.activeSession = response.session
                        rawSession = response.session.id
                    }
                }
                guard let rawSession, let sessionID = UUID(uuidString: rawSession) else {
                    throw JarvisWorkspaceError(code: "invalid_copy_context", message: "Your account could not be verified. Try again while online.")
                }
                self.synchronizeProductSession(rawSession)
                self.refreshWorkspaceContext()
                await self.workspaceContextTask?.value
                guard !Task.isCancelled, self.productConfigurationGeneration == generation,
                      self.savedCopiesPreparationID == taskID else { return }
                guard let model = self.workspace, model.sessionID == sessionID else {
                    self.savedCopiesPreparationError = "Your session changed. Try opening your files again."
                    return
                }
                await self.finishAccountFilesPreparation(model: model, api: api)
            } catch {
                guard !Task.isCancelled, self.productConfigurationGeneration == generation,
                      self.savedCopiesPreparationID == taskID else { return }
                let status = (error as? JarvisAPIError)?.statusCode
                if status == 401 || status == 403 {
                    self.preferences?.lockForSignIn()
                    self.plugins?.lockForSignIn()
                    self.workspace?.lockJournal(clearRemoteContent: true)
                    self.retireWorkspaceContext()
                    self.savedCopiesPreparationError = "Sign in to this account again to open its files."
                } else {
                    self.savedCopiesPreparationError = "Connect online to verify this account and open its files."
                }
            }
        }
    }

    private func finishAccountFilesPreparation(model: JarvisWorkspaceViewModel, api: JarvisAPIClient) async {
        if preparingSavedFileLibrary {
            guard model.canHandleDeliveries, verifiedWorkspaceContext?.sessionID == model.sessionID else {
                savedCopiesPreparationError = model.deliveryMessage ?? "Connect online to verify this account and open saved files."
                return
            }
            if savedFiles == nil {
                let access = JarvisWorkspaceLocalAccess()
                let limits: JarvisSessionFilesClient.MediaLimits? = model.canSaveOffline
                    ? .init(byteCount: JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumByteCount, durationMilliseconds: JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumDurationMilliseconds) : nil
                let client = JarvisSessionFilesClient(api: api, sessionID: model.sessionID,
                    contextAccess: access, mediaLimits: limits)
                let files = JarvisFilesViewModel(api: client)
                files.onAuthorizationFailure = { [weak self, weak files] in
                    guard let self, let files, self.savedFiles === files else { return }
                    self.preferences?.lockForSignIn()
                    self.plugins?.lockForSignIn()
                    self.workspace?.lockJournal(clearRemoteContent: true)
                    self.savedCopiesPreparationError = "Sign in to this account again to open saved files."
                    self.retireWorkspaceContext()
                }
                savedFilesAccess = access
                savedFiles = files
            }
            if audiovisual == nil {
                let access = JarvisWorkspaceLocalAccess()
                let limits: JarvisSessionFilesClient.MediaLimits? = model.canSaveOffline
                    ? .init(byteCount: JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumByteCount, durationMilliseconds: JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumDurationMilliseconds) : nil
                let client = JarvisSessionFilesClient(api: api, sessionID: model.sessionID,
                    contextAccess: access, mediaLimits: limits)
                let files = JarvisFilesViewModel(api: client, localMediaLookup: { [weak model] reference, viewAccess in
                    try access.check()
                    guard let model else { throw CancellationError() }
                    return try await model.localMediaFile(fileID: reference.fileID,
                        versionID: reference.versionID, access: viewAccess)
                })
                files.onAuthorizationFailure = { [weak self, weak files] in
                    guard let self, let files, self.audiovisual === files else { return }
                    self.preferences?.lockForSignIn(); self.plugins?.lockForSignIn()
                    self.workspace?.lockJournal(clearRemoteContent: true)
                    self.retireWorkspaceContext()
                }
                files.onResourceContextChange = { [weak self, weak files] context in
                    guard let self, let files, self.audiovisual === files else { return }
                    self.audioSessionViewModel.setResourceContext(context, sessionID: model.sessionID)
                }
                audiovisualAccess = access; audiovisual = files
            }
        } else if model.canSaveOffline {
            await model.refreshLocalEntries()
        } else if model.journalReady {
            savedCopiesPreparationError = "Saving copies on this phone is not available yet."
        } else {
            savedCopiesPreparationError = model.deliveryMessage
                ?? "Connect online to verify this account and open its saved copies."
        }
    }

    private var savedCopiesDataSessionID: String? {
        [audioSessionViewModel.activeSession?.id, productFallbackSessionID]
            .compactMap { $0 }
            .first { $0.lowercased() != replacedProductSessionID }
    }

    private func cancelSavedCopiesPreparation() {
        savedCopiesPreparationTask?.cancel()
        savedCopiesPreparationTask = nil
        savedCopiesPreparationID = nil
        isPreparingSavedCopies = false
        savedCopiesPreparationError = nil
        preparingSavedFileLibrary = false
    }

    func showWorkspaceTask(_ id: UUID) {
        guard workInbox != nil else { return }
        notificationInboxLink = nil
        workProductLink = JarvisProductWorkLink(destination: .task(id))
        isShowingWorkInbox = true
    }
    func showWorkspaceInboxItem(_ id: UUID) {
        guard workInbox != nil else { return }
        notificationInboxLink = nil
        workProductLink = JarvisProductWorkLink(destination: .inbox(id))
        isShowingWorkInbox = true
    }

    func sheetDidDismiss() {
        awaitingSheetDismissal = false
        workInbox?.hide()
        notificationInboxLink = nil
        workProductLink = nil
        openPendingNotification()
    }

    func openPendingNotification() {
        guard !awaitingSheetDismissal, workInbox != nil,
              JarvisNotificationService.shared.pendingLink != nil else { return }
        if presentedSheet != nil, presentedSheet != .work {
            awaitingSheetDismissal = true
            presentedSheet = nil
            return // The existing product sheet's dismissal completes navigation.
        }
        notificationInboxLink = JarvisNotificationService.shared.takePendingLink()
        workProductLink = nil
        isShowingWorkInbox = true
    }

    private func canReplaceNotificationConfiguration() -> Bool {
        guard pushNotifications?.requiresDisconnectCleanup != true else {
            setupErrorMessage = "Turn phone notifications off and resolve the server registration before changing backend credentials."
            return false
        }
        return true
    }

    private func handleMemoryRefreshHint(_ hint: JarvisMemoryRefreshHint) {
        Task { await workspace?.acceptedCommandOrReconnect() }
        let selectedID = selectedMemoryID
        let shouldRefreshDetail = selectedID.map(hint.affectedMemoryIds.contains) ?? false
        scheduleMemoryRefresh(
            recent: true,
            working: false,
            detailMemoryID: shouldRefreshDetail ? selectedID : nil
        )
    }

    private func handleWorkingMemoryRefreshHint(_ hint: JarvisWorkingMemoryRefreshHint) {
        let sessionID = hint.sessionId
            ?? audioSessionViewModel.activeSession?.id
            ?? activeSession?.id
        var containsNewerRevision = sessionID == nil
        if let sessionID, let revision = hint.revision {
            let knownRevision = highestSessionWorkingRevisionBySession[sessionID] ?? -1
            if revision > knownRevision {
                highestSessionWorkingRevisionBySession[sessionID] = revision
                containsNewerRevision = true
            }
        }
        if let ownerRevision = hint.ownerRevision, ownerRevision > highestOwnerWorkingRevision {
            highestOwnerWorkingRevision = ownerRevision
            containsNewerRevision = true
        }
        if let sessionAvailable = hint.sessionAvailable {
            let currentSessionAvailable = workingMemory?.session != nil
            let hasSessionWatermark = sessionID.flatMap {
                highestSessionWorkingRevisionBySession[$0]
            } != nil
            if sessionAvailable != currentSessionAvailable
                || (!sessionAvailable && hasSessionWatermark) {
                containsNewerRevision = true
            }
        }
        if let ownerAvailable = hint.ownerAvailable {
            let currentOwnerAvailable = workingMemory?.owner != nil
            if ownerAvailable != currentOwnerAvailable
                || (!ownerAvailable && highestOwnerWorkingRevision >= 0) {
                containsNewerRevision = true
            }
        }
        guard containsNewerRevision else { return }
        scheduleMemoryRefresh(recent: false, working: true)
    }

    private func handleRealtimeConnectionStateChanged(_ connectionState: JarvisAudioConnectionState) {
        if !connectionState.isConnected { appLinkPresenter.dismiss() }
        switch connectionState {
        case .connected:
            refreshAccountProducts()
            workInbox?.setVoiceActive(true)
            stopMemoryFallbackPolling()
            scheduleMemoryRefresh(recent: true, working: true)
            Task { await workspace?.acceptedCommandOrReconnect() }
        case .failed:
            workInbox?.setVoiceActive(false)
            startMemoryFallbackPolling()
        case .disconnected:
            workInbox?.setVoiceActive(false)
            stopMemoryFallbackPolling()
            scheduleMemoryRefresh(recent: true, working: true)
        case .connecting:
            break
        }
    }

    private func scheduleMemoryRefresh(
        recent: Bool,
        working: Bool,
        detailMemoryID: String? = nil
    ) {
        pendingRecentMemoryRefresh = pendingRecentMemoryRefresh || recent
        pendingWorkingMemoryRefresh = pendingWorkingMemoryRefresh || working
        if working {
            workingMemoryRequestGeneration += 1
        }
        if let detailMemoryID {
            pendingDetailMemoryID = detailMemoryID
        }
        guard memoryRefreshTask == nil else { return }
        memoryRefreshTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            await self?.flushMemoryRefresh()
        }
    }

    private func flushMemoryRefresh() async {
        let refreshRecent = pendingRecentMemoryRefresh
        let refreshWorking = pendingWorkingMemoryRefresh
        let detailMemoryID = pendingDetailMemoryID
        pendingRecentMemoryRefresh = false
        pendingWorkingMemoryRefresh = false
        pendingDetailMemoryID = nil
        memoryRefreshTask = nil

        if refreshRecent {
            await loadRecentMemories(replacing: true)
        }
        if refreshWorking {
            await loadWorkingMemory()
        }
        if let detailMemoryID, selectedMemoryID == detailMemoryID {
            await loadMemoryDetail(id: detailMemoryID)
        }
    }

    private func loadRecentMemories(replacing: Bool) async {
        guard let apiClient else { return }
        let cursor: String?
        let generation: Int
        if replacing {
            memoryListGeneration += 1
            generation = memoryListGeneration
            cursor = nil
            memoryLoadState = .loading
        } else {
            guard !isLoadingMoreMemories, let memoryNextCursor else { return }
            generation = memoryListGeneration
            cursor = memoryNextCursor
            isLoadingMoreMemories = true
        }
        defer {
            if !replacing { isLoadingMoreMemories = false }
        }

        do {
            let page = try await apiClient.getRecentMemory(limit: 30, cursor: cursor)
            guard generation == memoryListGeneration else { return }
            recentMemories = replacing
                ? Self.stablyDeduplicated(page.items)
                : Self.stablyDeduplicated(recentMemories + page.items)
            memoryNextCursor = page.nextCursor
            memoryView = page.view
            memoryLoadState = recentMemories.isEmpty ? .empty : .loaded
            memoryLastRefreshedAt = Date()
        } catch {
            guard generation == memoryListGeneration else { return }
            memoryLoadState = .degraded(error.localizedDescription)
        }
    }

    private func loadWorkingMemory() async {
        guard let apiClient else { return }
        workingMemoryRequestGeneration += 1
        let generation = workingMemoryRequestGeneration
        workingMemoryLoadState = .loading
        let sessionID = audioSessionViewModel.activeSession?.id ?? activeSession?.id
        do {
            let state = try await apiClient.getWorkingMemoryState(sessionId: sessionID)
            guard generation == workingMemoryRequestGeneration else { return }
            if let session = state.session {
                let knownRevision = highestSessionWorkingRevisionBySession[session.sessionId] ?? -1
                guard session.revision >= knownRevision else { return }
            }
            if let owner = state.owner {
                guard owner.revision >= highestOwnerWorkingRevision else { return }
            }
            workingMemory = state
            if let session = state.session {
                highestSessionWorkingRevisionBySession[session.sessionId] = session.revision
            } else if let sessionID {
                highestSessionWorkingRevisionBySession.removeValue(forKey: sessionID)
            }
            if let owner = state.owner {
                highestOwnerWorkingRevision = owner.revision
            } else {
                highestOwnerWorkingRevision = -1
            }
            workingMemoryLoadState = state.session == nil && state.owner == nil ? .empty : .loaded
        } catch {
            guard generation == workingMemoryRequestGeneration else { return }
            workingMemoryLoadState = .degraded(error.localizedDescription)
        }
    }

    private func startMemoryFallbackPolling() {
        guard memoryFallbackPollTask == nil else { return }
        memoryFallbackPollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(20))
                guard !Task.isCancelled, let self else { return }
                await self.reloadMemoryState()
            }
        }
    }

    private func stopMemoryFallbackPolling() {
        memoryFallbackPollTask?.cancel()
        memoryFallbackPollTask = nil
    }

    private static func stablyDeduplicated(_ items: [JarvisMemoryItem]) -> [JarvisMemoryItem] {
        var seen = Set<String>()
        return items.filter { seen.insert($0.id).inserted }
    }

    private func loadReminderState() async {
        guard let apiClient, !Task.isCancelled else { return }
        let configurationGeneration = productConfigurationGeneration, requestID = UUID()
        reminderReadGeneration = requestID
        do {
            let reminders = try await apiClient.getActiveReminders()
            guard !Task.isCancelled, productConfigurationGeneration == configurationGeneration,
                  reminderReadGeneration == requestID else { return }
            let events = try await apiClient.getReminderEvents(limit: 30)
            guard !Task.isCancelled, productConfigurationGeneration == configurationGeneration,
                  reminderReadGeneration == requestID else { return }
            activeReminders = reminders
            reminderEvents = events
            reminderRefreshError = nil
            reminderLastRefreshedAt = Date()
        } catch {
            guard !Task.isCancelled, productConfigurationGeneration == configurationGeneration,
                  reminderReadGeneration == requestID else { return }
            reminderRefreshError = error.localizedDescription
        }
    }

    private func clearReminderState() {
        reminderReadGeneration = UUID()
        activeReminders = []
        reminderEvents = []
        reminderRefreshError = nil
        reminderLastRefreshedAt = nil
    }

    private func loadGoogleCalendarStatus() async {
        guard let apiClient else { return }
        isGoogleCalendarStatusLoading = true
        defer { isGoogleCalendarStatusLoading = false }
        do {
            let status = try await apiClient.getGoogleCalendarStatus()
            googleCalendarConnected = status.connected
            googleCalendarStatusError = nil
        } catch {
            googleCalendarConnected = false
            googleCalendarStatusError = error.localizedDescription
        }
    }
}
