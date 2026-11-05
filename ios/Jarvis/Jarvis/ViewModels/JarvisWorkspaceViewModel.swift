import Combine
import Foundation

/// One request per resource. Identical readers join; a changed selection waits for
/// the previous transport to finish, then starts its own read with a fresh fence.
@MainActor
private final class JarvisWorkspaceReadGate<Value> {
    private var pending: (id: UUID, key: String, task: Task<Value, Error>)?
    func cancel() { pending?.task.cancel(); pending = nil }
    func read(key: String, operation: @escaping () async throws -> Value) async throws -> Value {
        while let current = pending {
            if current.key == key { return try await current.task.value }
            _ = try? await current.task.value
            if pending?.id == current.id { pending = nil }
        }
        let id = UUID(), task = Task { try await operation() }
        pending = (id, key, task)
        defer { if pending?.id == id { pending = nil } }
        return try await task.value
    }
}

@MainActor
final class JarvisWorkspaceViewModel: ObservableObject {
    struct MediaPresentation: Equatable, Identifiable {
        let id: UUID
        let requestID: UUID
        let artifactID: UUID
        let versionID: String
    }
    private enum ReadResource: CaseIterable { case offers, tasks, inbox }
    private struct PollState {
        let key: String
        var failures = 0
        var completedAt: Date?
        var retryUntil: Date?
    }
    @Published private(set) var offers: [JarvisWorkspaceRequest] = []
    @Published private(set) var deliveries: [JarvisWorkspaceLocalStore.Entry] = []
    @Published private(set) var filePreview: JarvisWorkspaceFileContent?
    @Published private(set) var offlinePreview: JarvisWorkspaceFileContent?
    @Published private(set) var isShowingOfflineMedia = false
    let offlineMediaPreview = JarvisMediaPreviewViewModel()
    @Published private(set) var offlineMediaExport: JarvisMediaExport?
    @Published private(set) var isPreparingOfflineExport = false
    @Published private(set) var offlineExportMessage: String?
    private var offlineMediaEntryID: UUID?
    private var offlineExportAccess: JarvisWorkspaceLocalAccess?
    private var offlineExportAttempt: UUID?
    @Published private(set) var mediaPresentation: MediaPresentation?
    let mediaPreview = JarvisMediaPreviewViewModel()
    let sandbox: JarvisSandboxViewModel
    private var sandboxChanges: AnyCancellable?
    @Published private(set) var deliveryMessage: String?
    @Published private(set) var workingRequestID: UUID?
    @Published private(set) var displayRequestID: UUID?
    @Published private(set) var tasks: [JarvisWorkTask] = []
    @Published private(set) var inbox: [JarvisInboxItem] = []
    @Published private(set) var tasksError: String?
    @Published private(set) var inboxError: String?
    @Published private(set) var tasksRefreshedAt: Date?
    @Published private(set) var inboxRefreshedAt: Date?
    @Published private(set) var journalReady = false
    var onAuthorizationFailure: (() -> Void)?

    let sessionID: UUID
    private let api: any JarvisWorkspaceAccess
    private let workAPI: (any JarvisWorkAccess)?
    private var store: JarvisWorkspaceLocalStore?
    private var journalRevoked = false
    private var authorizationBlocked = false
    private let localAccess = JarvisWorkspaceLocalAccess()
    private let requestGate = JarvisWorkspaceReadGate<JarvisWorkspaceRequestPage>()
    private var selectionGeneration = 0, visibilityGeneration = 0
    private var offlineGeneration = 0
    private var localEntriesGeneration = 0
    private var visible = false, foreground = true, active = true
    private var workVisible = true
    private var voiceActive = false, selectedWorkRunning = false
    private var pollStates: [ReadResource: PollState] = [:]
    private var pollTasks: [ReadResource: Task<Void, Never>] = [:]
    private var refreshGates: [ReadResource: JarvisWorkspaceReadGate<Bool>] = [:]
    private let now: () -> Date
    private let automaticPolling: Bool
    private var pendingDisplay: (requestID: UUID, selectionGeneration: Int)?
    private var mediaReadyPresentationID: UUID?
    private var pendingSandbox: (requestID: UUID, allocationID: UUID, version: String, leaseID: UUID)?
    private var completionTasks: [UUID: (id: UUID, acknowledgement: JarvisWorkspaceAcknowledgement, task: Task<Void, Error>)] = [:]

    init(api: any JarvisWorkspaceAccess, sessionID: UUID, store: JarvisWorkspaceLocalStore? = nil,
         automaticPolling: Bool = true, now: @escaping () -> Date = Date.init) {
        self.api = api; self.sessionID = sessionID; self.store = store; self.automaticPolling = automaticPolling
        self.journalReady = store != nil
        self.workAPI = api as? any JarvisWorkAccess
        self.now = now
        self.sandbox = JarvisSandboxViewModel(api: api as? any JarvisSandboxAccess, sessionID: sessionID)
        sandboxChanges = sandbox.objectWillChange.sink { [weak self] in self?.objectWillChange.send() }
        sandbox.onAuthorizationFailure = { [weak self] error in self?.handleAuthorizationFailure(error) }
        sandbox.onPresentationIntent = { [weak self] requestID in
            await self?.presentFreshRequest(requestID)
        }
        sandbox.onFirstFrame = { [weak self] lease in
            Task { await self?.didDisplaySandbox(lease) }
        }
    }
    isolated deinit {
        offlineExportAccess?.invalidate()
        offlineMediaExport?.file.discard()
        localAccess.invalidate()
        pollTasks.values.forEach { $0.cancel() }
        completionTasks.values.forEach { $0.task.cancel() }
    }

    var canRead: Bool { active && visible && foreground && !authorizationBlocked }
    var canHandleDeliveries: Bool { active && journalReady && !journalRevoked && store != nil }
    var canSaveOffline: Bool { canHandleDeliveries && store?.allowsFileBytes == true }
    /// Borrow a verified exact saved copy for a thumbnail/player. Both the
    /// account lifetime and the caller's view lifetime must still be current.
    func localMediaFile(fileID: UUID, versionID: String, access: JarvisWorkspaceLocalAccess) async throws -> JarvisWorkspaceLocalFileContent? {
        try access.check(); try localAccess.check()
        guard active, !authorizationBlocked, canSaveOffline, let store else { return nil }
        let entries = try await store.entries(access: localAccess)
        guard active, !authorizationBlocked else { throw CancellationError() }
        guard let entry = entries.first(where: {
            $0.request.fileID == fileID && $0.request.versionID == versionID &&
            $0.acknowledgement?.state == .savedLocal && $0.request.transferPolicy != nil
        }) else { return nil }
        let file = try await store.offlineFile(id: entry.id, access: localAccess)
        try access.check(); try localAccess.check()
        return file
    }
    func canPerform(_ request: JarvisWorkspaceRequest) -> Bool {
        canHandleDeliveries && (try? request.validatedTransferByteLimit()) != nil &&
        (request.transferPolicy == nil || api is any JarvisWorkspaceMediaAccess) &&
        (request.action == .download ? (request.artifactKind == "file" && canSaveOffline) :
            ((request.artifactKind == "sandbox" && api is any JarvisSandboxAccess)
             || (request.artifactKind == "file" && canSaveOffline && api is any JarvisWorkspaceMediaAccess)))
    }
    var nextPollDelay: TimeInterval {
        enabledResources.map { delay(for: $0) }.min() ?? normalPollDelay
    }
    var offlineFiles: [JarvisWorkspaceLocalStore.Entry] { canSaveOffline ? deliveries.filter { $0.acknowledgement?.state == .savedLocal } : [] }
    var unresolvedDeliveries: [JarvisWorkspaceLocalStore.Entry] {
        deliveries.filter {
            $0.sessionID == sessionID && !$0.acknowledged &&
            $0.id != workingRequestID && $0.id != displayRequestID && completionTasks[$0.id] == nil &&
            (canSaveOffline || $0.request.artifactKind == "sandbox")
        }
    }

    /// Called only after AppViewModel verifies the current Core context. Initial
    /// attachment preserves the current presentation; a retired lifetime never unlocks.
    @discardableResult
    func attachJournal(_ journal: JarvisWorkspaceLocalStore) -> Bool {
        guard active, !journalRevoked, store == nil else { return false }
        store = journal; journalReady = true; deliveryMessage = nil
        if canRead {
            if automaticPolling { startPolling() }
            Task { [weak self] in await self?.refreshLocalEntries() }
        }
        return true
    }
    func deliveryContextUnavailable(requiresSignIn: Bool = false) {
        deliveryMessage = requiresSignIn
            ? "Sign in again to access this account's phone presentation."
            : "Phone presentation could not be verified. Try reconnecting."
    }
    func lockJournal(clearRemoteContent: Bool = false) {
        guard !journalRevoked || (clearRemoteContent && !authorizationBlocked) else { return }
        journalRevoked = true; journalReady = false
        sandbox.close()
        cancelOfflineExport(); offlineMediaEntryID = nil
        localAccess.invalidate(); store = nil
        offlineMediaPreview.deactivate(); isShowingOfflineMedia = false
        mediaPreview.deactivate(); mediaPresentation = nil; mediaReadyPresentationID = nil
        localEntriesGeneration += 1; offlineGeneration += 1
        offers = []; deliveries = []; offlinePreview = nil; filePreview = nil
        pendingDisplay = nil; displayRequestID = nil
        requestGate.cancel(); refreshGates[.offers]?.cancel(); pollTasks.removeValue(forKey: .offers)?.cancel()
        completionTasks.values.forEach { $0.task.cancel() }; completionTasks = [:]
        if clearRemoteContent {
            authorizationBlocked = true
            tasks = []; inbox = []
            visibilityGeneration += 1; selectionGeneration += 1
            stopPolling(); refreshGates.values.forEach { $0.cancel() }
        }
        deliveryContextUnavailable(requiresSignIn: clearRemoteContent)
    }

    /// Only the Core fresh-live intent callback may call this automatically.
    /// Discovery/reconnect never invokes it. The exact current request is read
    /// before the existing claim/render/ACK path, with no new claim state machine.
    func presentFreshRequest(_ requestID: UUID) async {
        guard canRead, canHandleDeliveries, !Task.isCancelled else { return }
        let visibility = visibilityGeneration, navigation = selectionGeneration
        await refreshResource(.offers, force: true)
        guard readable(visibility), navigation == selectionGeneration, canHandleDeliveries,
              let request = offers.first(where: { $0.id == requestID }),
              request.action == .open, canPerform(request), request.state == .pending else { return }
        await perform(request)
    }

    /// Explicit post-cloud-save policy: download the exact confirmed recording
    /// without opening a page or borrowing foreground display permission.
    func downloadConfirmedRecording(requestID: UUID, fileID: UUID, versionID: String) async -> Bool {
        guard active, !authorizationBlocked, canSaveOffline, !Task.isCancelled else { return false }
        do {
            let page = try await api.listWorkspaceRequests(sessionID: sessionID)
            guard active, !authorizationBlocked, canSaveOffline, !Task.isCancelled,
                  let request = page.items.first(where: { $0.id == requestID }),
                  request.action == .download, request.artifactKind == "file", request.fileID == fileID,
                  request.versionID == versionID, request.transferPolicy != nil, canPerform(request) else { return false }
            // A foreground export/download keeps its existing ownership. The
            // confirmed recording waits for that transfer instead of losing
            // its automatic download merely because the UI was busy.
            if workingRequestID != nil {
                for await current in $workingRequestID.values {
                    guard active, !authorizationBlocked, !Task.isCancelled else { return false }
                    if current == nil { break }
                }
            }
            guard active, !authorizationBlocked, canSaveOffline, !Task.isCancelled else { return false }
            await perform(request, backgroundDownload: true)
            guard active, !Task.isCancelled, let entry = try await store?.entry(id: requestID, access: localAccess),
                  entry.request.fileID == fileID, entry.request.versionID == versionID else { return false }
            return entry.acknowledgement?.state == .savedLocal
        } catch {
            if active { deliveryMessage = "Recording download is unconfirmed. " + error.localizedDescription }
            handleAuthorizationFailure(error)
            return false
        }
    }

    func setActivity(voiceActive: Bool, selectedWorkRunning: Bool = false) {
        let changed = self.voiceActive != voiceActive || self.selectedWorkRunning != selectedWorkRunning
        self.voiceActive = voiceActive; self.selectedWorkRunning = selectedWorkRunning
        if changed, canRead, automaticPolling { stopPolling(); startPolling() }
    }
    func setVisibility(_ visible: Bool, foreground: Bool, workVisible: Bool = true) {
        let wasReadable = canRead
        let workBecameVisible = !self.workVisible && workVisible
        self.workVisible = workVisible
        if workBecameVisible {
            for resource in [ReadResource.tasks, .inbox] { pollStates[resource]?.completedAt = nil }
        }
        if !workVisible {
            for resource in [ReadResource.tasks, .inbox] {
                pollTasks.removeValue(forKey: resource)?.cancel()
                refreshGates[resource]?.cancel()
            }
        }
        if self.foreground && !foreground { offlineMediaPreview.pause() }
        self.visible = visible; self.foreground = foreground
        if !canRead {
            sandbox.close()
            visibilityGeneration += 1; selectionGeneration += 1
            stopPolling()
            refreshGates.values.forEach { $0.cancel() }
            requestGate.cancel()
            pendingDisplay = nil
            displayRequestID = nil
            if mediaPreview.player == nil {
                mediaPreview.close(); mediaPresentation = nil
            } else { mediaPreview.pause() }
            mediaReadyPresentationID = nil
        } else if !wasReadable {
            // Reconnect refreshes promptly, while preserving server embargoes.
            for resource in enabledResources { pollStates[resource]?.completedAt = nil }
            Task { [weak self] in await self?.refreshLocalEntries() }
            if automaticPolling { startPolling() }
        }
        if canRead, automaticPolling { startPolling() }
    }
    func deactivate() {
        sandbox.deactivate()
        cancelOfflineExport(); offlineMediaEntryID = nil
        journalReady = false
        localAccess.invalidate()
        offlineMediaPreview.deactivate(); isShowingOfflineMedia = false
        mediaPreview.deactivate(); mediaPresentation = nil; mediaReadyPresentationID = nil
        active = false; setVisibility(false, foreground: false)
        completionTasks.values.forEach { $0.task.cancel() }
        completionTasks = [:]
        offlineGeneration += 1
        offers = []; deliveries = []; filePreview = nil; offlinePreview = nil
        tasks = []; inbox = []; tasksError = nil; inboxError = nil
        tasksRefreshedAt = nil; inboxRefreshedAt = nil
    }
    func closeDocument() {
        sandbox.close()
        pendingSandbox = nil
        selectionGeneration += 1; filePreview = nil; pendingDisplay = nil
        displayRequestID = nil
        mediaPreview.close(); mediaPresentation = nil; mediaReadyPresentationID = nil
    }
    func closeOfflinePreview() {
        cancelOfflineExport(); offlineMediaEntryID = nil; offlineExportMessage = nil
        offlineGeneration += 1; offlinePreview = nil; isShowingOfflineMedia = false
        offlineMediaPreview.close()
    }


    func acceptedCommandOrReconnect() async {
        // Reconnect discovers offers. It never claims, reopens or re-exports them automatically.
        await refresh()
    }
    func refresh(force: Bool = true) async {
        guard canRead else { return }
        async let requests: Void = refreshResource(.offers, force: force)
        async let tasks: Void = refreshResource(.tasks, force: force)
        async let inbox: Void = refreshResource(.inbox, force: force)
        _ = await (requests, tasks, inbox)
        await refreshLocalEntries()
    }

    @discardableResult
    private func refreshOffers() async -> Bool {
        let fence = visibilityGeneration
        do {
            let api = self.api, sessionID = self.sessionID
            let page = try await requestGate.read(key: sessionID.uuidString) { try await api.listWorkspaceRequests(sessionID: sessionID) }
            guard readable(fence) else { return false }
            offers = page.items.filter { !$0.state.isTerminal && $0.artifactKind != "note" }
            return true
        } catch {
            guard readable(fence) else { return false }
            deliveryMessage = "Phone requests could not be refreshed. " + error.localizedDescription
            recordRetryAfter(error, resource: .offers); return false
        }
    }

    private func refreshTasks() async -> Bool {
        guard let workAPI else { return true }
        let fence = visibilityGeneration
        do {
            let rows = try await workAPI.getWorkTasks(offset: 0)
            guard readable(fence), workVisible else { return false }
            guard rows.count <= 20, Set(rows.map(\.id)).count == rows.count else { throw identityError() }
            tasks = rows; tasksError = nil; tasksRefreshedAt = now()
            return true
        } catch {
            guard readable(fence), workVisible else { return false }
            if (error as? JarvisAPIError)?.code == "tasks_disabled" {
                tasks = []; tasksError = nil; tasksRefreshedAt = now()
                return true
            }
            tasksError = error.localizedDescription
            recordRetryAfter(error, resource: .tasks); return false
        }
    }
    private func refreshInbox() async -> Bool {
        guard let workAPI else { return true }
        let fence = visibilityGeneration
        do {
            let rows = try await workAPI.getWorkInbox(offset: 0, includeDismissed: false, taskID: nil)
            guard readable(fence), workVisible else { return false }
            guard rows.count <= 20, Set(rows.map(\.id)).count == rows.count else { throw identityError() }
            inbox = rows; inboxError = nil; inboxRefreshedAt = now()
            return true
        } catch {
            guard readable(fence), workVisible else { return false }
            inbox = [] // Do not leave a withdrawn answer offered as available.
            if (error as? JarvisAPIError)?.code == "attention_disabled" {
                inboxError = nil; inboxRefreshedAt = now()
                return true
            }
            inboxError = error.localizedDescription
            recordRetryAfter(error, resource: .inbox); return false
        }
    }

    /// Only invoked by a deliberate tap or a Core-approved fresh foreground intent.
    func perform(_ request: JarvisWorkspaceRequest) async {
        await perform(request, backgroundDownload: false)
    }

    private func perform(_ request: JarvisWorkspaceRequest, backgroundDownload: Bool) async {
        guard (canRead || (backgroundDownload && request.action == .download && !authorizationBlocked)),
              canHandleDeliveries, let store, workingRequestID == nil, !request.state.isTerminal, !Task.isCancelled else { return }
        workingRequestID = request.id; deliveryMessage = nil
        let visibility = visibilityGeneration
        let navigation = selectionGeneration
        defer { workingRequestID = nil }
        do {
            guard canPerform(request) else {
                throw JarvisWorkspaceError(code: "delivery_unavailable", message: "This delivery type is not available yet.")
            }
            guard request.contentReady else {
                throw JarvisWorkspaceError(code: "content_not_ready", message: "Jarvis is still preparing this saved version.")
            }
            let entry = try await store.prepareClaim(request, sessionID: sessionID, access: localAccess)
            guard active, canHandleDeliveries, !Task.isCancelled else { return }
            if let ack = entry.acknowledgement {
                try await sendCompletion(requestID: request.id, acknowledgement: ack)
                await refreshLocalEntries(); return
            }
            guard entry.phase != .effectStarted else {
                throw JarvisWorkspaceError(code: "delivery_requires_reconciliation", message: "This request may already have run. Use Check saved result; it will not download another copy.")
            }
            let response = try await api.acknowledgeWorkspaceRequest(requestID: request.id,
                ack: JarvisWorkspaceAcknowledgement(sessionID: sessionID, state: .claimed))
            guard active else { return }
            guard response.request.hasSameTarget(as: request) else { throw identityError() }
            guard response.request.state == .claimed else {
                deliveryMessage = "This request is already \(response.request.state.rawValue). No local action was repeated."
                await refreshLocalEntries(); return
            }
            try await store.confirmClaim(id: request.id, access: localAccess)
            guard backgroundDownload ? (active && !authorizationBlocked && !Task.isCancelled) : readable(visibility) else {
                await refreshLocalEntries(); return
            }
            guard request.action != .open || selectionGeneration == navigation else {
                deliveryMessage = "Your current view stayed open. The earlier request is still available to check."
                await refreshLocalEntries(); return
            }
            try await store.beginEffect(id: request.id, access: localAccess)
            guard active else { return }
            if request.action == .download {
                guard let fileID = request.fileID else { throw identityError() }
                let ack: JarvisWorkspaceAcknowledgement
                if request.transferPolicy != nil {
                    guard let mediaAPI = api as? any JarvisWorkspaceMediaAccess else { throw identityError() }
                    let media = try await mediaAPI.getWorkspaceMediaContent(request: request, sessionID: sessionID, access: localAccess)
                    defer { media.temporaryFile.discard() }
                    guard active, !Task.isCancelled else { return }
                    ack = try await store.saveDownloadFile(id: request.id, source: media.temporaryFile.urlForReading(),
                        filename: media.filename, contentType: media.contentType, access: localAccess)
                } else {
                    let file = try await api.getWorkspaceFileContent(fileID: fileID, sessionID: sessionID)
                    guard active, !Task.isCancelled else { return }
                    ack = try await store.saveDownload(id: request.id, file: file, access: localAccess)
                }
                guard active else { return }
                try await sendCompletion(requestID: request.id, acknowledgement: ack)
                if active { deliveryMessage = "Saved for offline access in Jarvis." }
            } else if request.artifactKind == "sandbox" {
                closeDocument()
                pendingDisplay = (request.id, selectionGeneration)
                displayRequestID = request.id
                guard await sandbox.open(allocationID: request.artifactID, version: request.versionID) else {
                    throw JarvisWorkspaceError(code: "display_unconfirmed", message: "The sandbox screen has not been shown.")
                }
                if let lease = sandbox.lease {
                    pendingSandbox = (request.id, request.artifactID, request.versionID, lease.id)
                }
            } else if request.artifactKind == "file" {
                sandbox.close()
                guard try await openMedia(request) else {
                    throw JarvisWorkspaceError(code: "display_unconfirmed", message: "The video has not been shown. You can check or dismiss this request.")
                }
            } else { throw identityError() }
        } catch {
            if active { deliveryMessage = "Completion is not confirmed. " + error.localizedDescription }
            handleAuthorizationFailure(error)
        }
        await refreshLocalEntries()
    }

    private func openMedia(_ request: JarvisWorkspaceRequest) async throws -> Bool {
        guard canRead, canSaveOffline, let mediaAPI = api as? any JarvisWorkspaceMediaAccess else { return false }
        closeDocument()
        let generation = selectionGeneration, visibility = visibilityGeneration
        let presentation = MediaPresentation(id: UUID(), requestID: request.id,
            artifactID: request.artifactID, versionID: request.versionID)
        mediaPresentation = presentation
        do {
            let media = try await mediaAPI.getWorkspaceMediaContent(request: request, sessionID: sessionID, access: localAccess)
            var adopted = false
            defer {
                if !adopted {
                    media.temporaryFile.discard()
                    if mediaPresentation == presentation {
                        mediaPreview.close(); mediaPresentation = nil; mediaReadyPresentationID = nil
                    }
                }
            }
            guard readable(visibility), generation == selectionGeneration, mediaPresentation == presentation,
                  !Task.isCancelled else { return false }
            let opened = await mediaPreview.open(file: media.temporaryFile, filename: media.filename,
                maximumByteCount: media.maximumByteCount,
                maximumDurationMilliseconds: media.maximumDurationMilliseconds)
            guard opened, readable(visibility), generation == selectionGeneration, mediaPresentation == presentation,
                  !Task.isCancelled else { return false }
            adopted = true
            pendingDisplay = (request.id, generation)
            displayRequestID = request.id
            return true
        } catch {
            if readable(visibility), generation == selectionGeneration {
                mediaPreview.close(); mediaPresentation = nil; mediaReadyPresentationID = nil
            }
            throw error
        }
    }

    /// Actual player-controller readiness is separate from successful asset load.
    func mediaReadinessChanged(presentationID: UUID, ready: Bool) {
        guard mediaPresentation?.id == presentationID else { return }
        mediaReadyPresentationID = ready ? presentationID : nil
    }

    private func didDisplaySandbox(_ lease: JarvisSandboxLease) async {
        guard canRead, let store, let display = pendingDisplay,
              displayRequestID == display.requestID, display.selectionGeneration == selectionGeneration,
              sandbox.lease?.id == lease.id, sandbox.phase == .viewing,
              let expected = pendingSandbox, expected.requestID == display.requestID, expected.leaseID == lease.id,
              expected.allocationID == lease.target.allocationID,
              expected.version == lease.target.generation else { return }
        pendingDisplay = nil
        pendingSandbox = nil
        defer { if displayRequestID == display.requestID { displayRequestID = nil } }
        do {
            let ack = try await store.completeDisplay(id: display.requestID, access: localAccess)
            try await sendCompletion(requestID: display.requestID, acknowledgement: ack)
        } catch {
            if active { deliveryMessage = "Shown on this phone; confirmation is pending." }
            handleAuthorizationFailure(error)
        }
        await refreshLocalEntries()
    }

    /// Called only by the visible renderer after its matching first frame is ready.
    func didDisplayMedia(presentationID: UUID, requestID: UUID, artifactID: UUID, versionID: String) async {
        guard canRead, let store, let display = pendingDisplay,
              display.requestID == requestID, displayRequestID == requestID,
              display.selectionGeneration == selectionGeneration,
              let media = mediaPresentation, media.id == presentationID, media.requestID == requestID,
              media.artifactID == artifactID, media.versionID == versionID,
              mediaReadyPresentationID == presentationID, mediaPreview.player?.currentItem != nil else { return }
        pendingDisplay = nil
        // Keep active display/confirmation out of recovery until its result is
        // reflected in the journal. A failed ACK becomes recoverable afterward.
        defer { if displayRequestID == requestID { displayRequestID = nil } }
        do {
            let ack = try await store.completeDisplay(id: requestID, access: localAccess)
            try await sendCompletion(requestID: requestID, acknowledgement: ack)
        } catch {
            if active { deliveryMessage = "Shown on this phone; confirmation is pending. " + error.localizedDescription }
            handleAuthorizationFailure(error)
        }
        await refreshLocalEntries()
    }


    func reconcile(_ entry: JarvisWorkspaceLocalStore.Entry) async {
        guard canRead, let store, entry.sessionID == sessionID, workingRequestID == nil else { return }
        workingRequestID = entry.id
        defer { workingRequestID = nil }
        do {
            let ack: JarvisWorkspaceAcknowledgement
            if let saved = entry.acknowledgement {
                if saved.state == .savedLocal {
                    if entry.request.transferPolicy != nil { _ = try await store.offlineFile(id: entry.id, access: localAccess) }
                    else { _ = try await store.offlineContent(id: entry.id, access: localAccess) }
                }
                ack = saved
            }
            else { ack = try await store.reconcileDownload(id: entry.id, access: localAccess) }
            try await sendCompletion(requestID: entry.id, acknowledgement: ack)
            guard active else { return }
            deliveryMessage = ack.state == .savedLocal ? "Verified the saved offline copy and reported it." : "Reported the existing result."
        } catch {
            if active { deliveryMessage = "The saved result could not be confirmed. No action was repeated. " + error.localizedDescription }
            handleAuthorizationFailure(error)
        }
        await refreshLocalEntries()
    }

    func dismiss(_ request: JarvisWorkspaceRequest) async {
        guard canRead, let store, workingRequestID == nil else { return }
        workingRequestID = request.id
        defer { workingRequestID = nil }
        do {
            _ = try await store.prepareClaim(request, sessionID: sessionID, access: localAccess)
            guard active else { return }
            let ack = try await store.completeWithoutEffect(id: request.id, state: .dismissed, access: localAccess)
            try await sendCompletion(requestID: request.id, acknowledgement: ack)
            guard active else { return }
            offers.removeAll { $0.id == request.id }
        } catch {
            if active { deliveryMessage = "Dismissal is not confirmed. " + error.localizedDescription }
            handleAuthorizationFailure(error)
        }
        await refreshLocalEntries()
    }

    func openOffline(_ entry: JarvisWorkspaceLocalStore.Entry) async {
        guard canSaveOffline, let store else { return }
        closeOfflinePreview()
        let generation = offlineGeneration
        deliveryMessage = nil
        do {
            if let policy = entry.request.transferPolicy {
                isShowingOfflineMedia = true
                let limit = try entry.request.validatedTransferByteLimit()
                let stored = try await store.offlineFile(id: entry.id, access: localAccess)
                guard active, generation == offlineGeneration, !Task.isCancelled else { return }
                let temporary = try await JarvisTemporaryFileVerifier().copyVerifiedFile(from: stored.url,
                    expectedByteCount: stored.byteCount, expectedSHA256: stored.contentSHA256,
                    maximumByteCount: limit, access: localAccess, format: .mp4)
                guard active, generation == offlineGeneration, !Task.isCancelled else { temporary.discard(); return }
                _ = await offlineMediaPreview.open(file: temporary, filename: stored.filename,
                    maximumByteCount: limit, maximumDurationMilliseconds: policy.maximumDurationMilliseconds)
                if active, generation == offlineGeneration, offlineMediaPreview.player != nil {
                    offlineMediaEntryID = entry.id
                }
                return
            }
            let file = try await store.offlineContent(id: entry.id, access: localAccess)
            guard active, generation == offlineGeneration else { return }
            offlinePreview = file
        } catch {
            if active, generation == offlineGeneration {
                isShowingOfflineMedia = false
                deliveryMessage = "The offline copy could not be opened. " + error.localizedDescription
            }
        }
    }

    /// Export a separately verified copy from the existing account journal.
    /// Offline playback/export never needs to refetch the complete cloud object.
    func prepareOfflineExport() async -> Bool {
        guard canSaveOffline, let store, let entryID = offlineMediaEntryID,
              !isPreparingOfflineExport, !Task.isCancelled else { return false }
        cancelOfflineExport()
        let generation = offlineGeneration, attempt = UUID(), access = JarvisWorkspaceLocalAccess()
        offlineExportAttempt = attempt; offlineExportAccess = access
        isPreparingOfflineExport = true; offlineExportMessage = nil
        defer { if offlineExportAttempt == attempt { isPreparingOfflineExport = false } }
        do {
            guard let entry = try await store.entry(id: entryID, access: localAccess),
                  entry.acknowledgement?.state == .savedLocal, entry.request.transferPolicy != nil else {
                throw identityError()
            }
            let limit = try entry.request.validatedTransferByteLimit()
            let stored = try await store.offlineFile(id: entryID, access: localAccess)
            let file = try await JarvisTemporaryFileVerifier().copyVerifiedFile(from: stored.url,
                expectedByteCount: stored.byteCount, expectedSHA256: stored.contentSHA256,
                maximumByteCount: limit, access: access, format: .mp4)
            guard active, canSaveOffline, generation == offlineGeneration,
                  offlineExportAttempt == attempt, !Task.isCancelled else { file.discard(); return false }
            try localAccess.check(); try access.check()
            offlineMediaExport = JarvisMediaExport(filename: stored.filename, file: file)
            return true
        } catch {
            guard offlineExportAttempt == attempt else { return false }
            cancelOfflineExport()
            if active, !(error is CancellationError), !Task.isCancelled {
                offlineExportMessage = "Export could not be prepared: " + error.localizedDescription
            }
            return false
        }
    }

    func finishOfflineExport(id: UUID, completed: Bool, failureDescription: String? = nil) {
        guard active, canSaveOffline, offlineMediaExport?.id == id,
              let access = offlineExportAccess else { return }
        guard (try? localAccess.check()) != nil, (try? access.check()) != nil else {
            cancelOfflineExport(); return
        }
        cancelOfflineExport()
        offlineExportMessage = completed ? "Exported a separate copy."
            : failureDescription.map { "Export was not completed: " + $0 }
    }

    func cancelOfflineExport() {
        offlineExportAttempt = nil
        offlineExportAccess?.invalidate(); offlineExportAccess = nil
        offlineMediaExport?.file.discard(); offlineMediaExport = nil
        isPreparingOfflineExport = false
    }
    func refreshLocalEntries() async {
        guard active, let store else { return }
        localEntriesGeneration += 1
        let generation = localEntriesGeneration
        do {
            let entries = try await store.entries(access: localAccess)
            if active, generation == localEntriesGeneration { deliveries = entries }
        } catch { if active, generation == localEntriesGeneration { deliveryMessage = "Saved phone results could not be read. " + error.localizedDescription } }
    }
    private func sendCompletion(requestID: UUID, acknowledgement: JarvisWorkspaceAcknowledgement) async throws {
        guard canHandleDeliveries else { throw CancellationError() }
        if let current = completionTasks[requestID] {
            guard current.acknowledgement == acknowledgement else { throw identityError() }
            // Join the same result, including failure; an in-flight ACK is not
            // success and must not let a second caller announce confirmation.
            try await current.task.value
            return
        }
        let id = UUID()
        let task = Task { [weak self] in
            guard let self else { throw CancellationError() }
            try await self.completeRequest(requestID: requestID, acknowledgement: acknowledgement)
        }
        completionTasks[requestID] = (id, acknowledgement, task)
        defer { if completionTasks[requestID]?.id == id { completionTasks[requestID] = nil } }
        try await task.value
    }
    private func completeRequest(requestID: UUID, acknowledgement: JarvisWorkspaceAcknowledgement) async throws {
        guard canHandleDeliveries, let store else { throw CancellationError() }
        try localAccess.check()
        let response = try await api.acknowledgeWorkspaceRequest(requestID: requestID, ack: acknowledgement)
        guard canHandleDeliveries, !Task.isCancelled else { throw CancellationError() }
        let entry = try await store.entry(id: requestID, access: localAccess)
        guard active, let entry, response.request.hasSameTarget(as: entry.request) else { throw identityError() }
        guard response.request.id == requestID, response.request.state == acknowledgement.state,
              response.request.localReceipt == acknowledgement.localReceipt,
              response.request.errorCode == acknowledgement.errorCode else { throw identityError() }
        try await store.markAcknowledged(id: requestID, acknowledgement: acknowledgement, access: localAccess)
        if active { offers.removeAll { $0.id == requestID } }
    }
    private var normalPollDelay: TimeInterval { voiceActive || selectedWorkRunning ? 1 : 5 }
    private var enabledResources: [ReadResource] {
        (store == nil ? [] : [.offers]) + (workAPI == nil || !workVisible ? [] : [.tasks, .inbox])
    }
    private func key(for resource: ReadResource) -> String {
        switch resource {
        case .offers, .tasks, .inbox: return sessionID.uuidString
        }
    }
    private func generation(for resource: ReadResource) -> Int {
        switch resource {
        case .offers, .tasks, .inbox: return 0
        }
    }
    private func prepareSchedule(_ resource: ReadResource) {
        let key = key(for: resource)
        if pollStates[resource]?.key != key { pollStates[resource] = PollState(key: key) }
    }
    private func delay(for resource: ReadResource) -> TimeInterval {
        guard let state = pollStates[resource], state.key == key(for: resource) else { return 0 }
        let cadence = state.failures == 0 ? normalPollDelay : min(30, pow(2, Double(min(state.failures, 5))))
        let due = state.completedAt?.addingTimeInterval(cadence) ?? now()
        return max(0, due.timeIntervalSince(now()), state.retryUntil?.timeIntervalSince(now()) ?? 0)
    }
    private func completeRead(_ resource: ReadResource, success: Bool) {
        prepareSchedule(resource)
        pollStates[resource]?.completedAt = now()
        if success {
            pollStates[resource]?.failures = 0
            pollStates[resource]?.retryUntil = nil
        } else {
            pollStates[resource]?.failures += 1
        }
    }
    private func refreshResource(_ resource: ReadResource, force: Bool) async {
        guard canRead, enabledResources.contains(resource) else { return }
        prepareSchedule(resource)
        // A deliberate refresh may bypass normal cadence/backoff, never a larger
        // server Retry-After. Reads never implicitly retry mutations or claims.
        if let until = pollStates[resource]?.retryUntil, until > now() { return }
        if !force && delay(for: resource) > 0 { return }
        let fence = visibilityGeneration, generation = generation(for: resource), key = key(for: resource)
        let gate = refreshGates[resource] ?? JarvisWorkspaceReadGate<Bool>()
        refreshGates[resource] = gate
        _ = try? await gate.read(key: "\(fence)/\(generation)/" + key) { [weak self] in
            guard let self, self.readable(fence) else { return false }
            let result: Bool
            switch resource {
            case .offers: result = await self.refreshOffers()
            case .tasks: result = await self.refreshTasks()
            case .inbox: result = await self.refreshInbox()
            }
            guard self.readable(fence), self.generation(for: resource) == generation,
                  self.key(for: resource) == key, self.enabledResources.contains(resource) else { return false }
            self.completeRead(resource, success: result)
            return result
        }
    }
    private func stopPolling() {
        pollTasks.values.forEach { $0.cancel() }
        pollTasks = [:]
    }
    private func startPolling() {
        guard canRead else { return }
        for resource in enabledResources where pollTasks[resource] == nil {
            pollTasks[resource] = Task { [weak self] in
                while !Task.isCancelled {
                    guard let seconds = await self?.pollIteration(resource) else { return }
                    do { try await Task.sleep(nanoseconds: UInt64(min(seconds, 86_400) * 1_000_000_000)) }
                    catch { return }
                }
            }
        }
    }
    private func pollIteration(_ resource: ReadResource) async -> TimeInterval? {
        guard canRead, !Task.isCancelled else { return nil }
        await refreshResource(resource, force: false)
        guard canRead, !Task.isCancelled else { return nil }
        return max(0.05, delay(for: resource))
    }
    private func readable(_ fence: Int) -> Bool { canRead && fence == visibilityGeneration && !Task.isCancelled }
    private func recordRetryAfter(_ error: Error, resource: ReadResource) {
        handleAuthorizationFailure(error)
        let seconds = (error as? JarvisAPIError)?.retryAfter ?? (error as? JarvisWorkspaceError)?.retryAfter
        guard let seconds, seconds.isFinite, seconds >= 0 else { return }
        prepareSchedule(resource)
        let until = now().addingTimeInterval(seconds)
        let previous = pollStates[resource]?.retryUntil ?? until
        pollStates[resource]?.retryUntil = max(previous, until)
    }
    private func handleAuthorizationFailure(_ error: Error) {
        guard active, !authorizationBlocked, let status = (error as? JarvisAPIError)?.statusCode, status == 401 || status == 403 else { return }
        lockJournal(clearRemoteContent: true)
        onAuthorizationFailure?()
    }
    private func identityError() -> JarvisWorkspaceError {
        JarvisWorkspaceError(code: "workspace_identity_mismatch", message: "The response did not match the requested document or delivery.")
    }
}
