import Combine
import Foundation

enum JarvisWorkDetailSelection: Hashable {
    case task(UUID), inbox(UUID)
}

/// A reply stays bound to the task version the user began writing against.
/// Reviewing a newer version is deliberate; polling must never silently rebase it.
struct JarvisWorkReplyDraft {
    private(set) var text = ""
    private(set) var baseTask: JarvisWorkTask?
    mutating func update(_ text: String, for task: JarvisWorkTask) {
        if self.text.isEmpty, !text.isEmpty { baseTask = task }
        self.text = text
        if text.isEmpty { baseTask = nil }
    }
    func needsReview(for task: JarvisWorkTask) -> Bool {
        guard let baseTask else { return false }
        return baseTask.id != task.id || baseTask.revision != task.revision
    }
    mutating func review(_ task: JarvisWorkTask) {
        guard !text.isEmpty, baseTask?.id == task.id else { return }
        baseTask = task
    }
    mutating func accepted(_ action: JarvisPendingWorkAction) {
        guard case let .task(id, operation, body) = action, operation == .update,
              baseTask?.id == id, baseTask?.revision == body.expectedRevision,
              text == body.message else { return }
        self = Self()
    }
}

@MainActor
final class JarvisWorkInboxViewModel: ObservableObject {
    @Published private(set) var tasks: [JarvisWorkTask] = []
    @Published private(set) var inbox: [JarvisInboxItem] = []
    @Published private(set) var tasksError: String?
    @Published private(set) var inboxError: String?
    @Published private(set) var isLoading = false
    @Published private(set) var isLoadingMore = false
    @Published private(set) var moreTasks = false
    @Published private(set) var moreInbox = false
    @Published var includeDismissed = false
    @Published private(set) var taskInbox: [JarvisInboxItem] = []
    @Published private(set) var pendingActionLabel: String?
    @Published private(set) var selectedTask: JarvisWorkTask?
    @Published private(set) var selectedInbox: JarvisInboxDetail?
    @Published private(set) var detailError: String?
    @Published private(set) var isLoadingDetail = false
    @Published private(set) var mutationMessage: String?
    @Published private(set) var pendingAction: JarvisPendingWorkAction?
    @Published private(set) var isMutating = false
    @Published private(set) var detailSelection: JarvisWorkDetailSelection?
    @Published private(set) var detailRefreshedAt: Date?
    @Published private(set) var lastAcceptedAction: JarvisPendingWorkAction?
    @Published private(set) var taskAnswersError: String?

    private let api: any JarvisWorkAccess
    private var listGeneration = 0
    private var detailGeneration = 0
    private var taskOffset = 0
    private var inboxOffset = 0
    private var visible = false
    private var active = true
    private var voiceActive = false
    private var detailRead: (id: UUID, task: Task<Void, Never>)?
    private var detailPoll: Task<Void, Never>?
    private var detailFailures = 0
    private var detailCompletedAt: Date?
    private var retryNotBefore: [JarvisWorkDetailSelection: Date] = [:]
    private var answersDue: Date?
    private var answersFailures = 0
    private var answersRetryNotBefore: [UUID: Date] = [:]
    private let automaticPolling: Bool
    private let now: () -> Date

    init(api: any JarvisWorkAccess, automaticPolling: Bool = true, now: @escaping () -> Date = Date.init) {
        self.api = api; self.automaticPolling = automaticPolling; self.now = now
    }
    deinit { detailRead?.task.cancel(); detailPoll?.cancel() }

    func beginPresentation() {
        guard active else { return }
        visible = true
        startDetailPolling()
    }

    func present() async {
        guard active else { return }
        beginPresentation()
        await reload()
    }

    func reload(refreshDetail: Bool = true) async {
        guard active, visible else { return }
        listGeneration += 1
        let generation = listGeneration
        let dismissed = includeDismissed
        let detailRequest = detailGeneration
        tasks = []; inbox = []
        taskOffset = 0; inboxOffset = 0
        moreTasks = false; moreInbox = false
        tasksError = nil; inboxError = nil
        isLoading = true
        isLoadingMore = false
        defer { if generation == listGeneration { isLoading = false } }
        do {
            let rows = try await api.getWorkTasks(offset: 0)
            guard canRead(generation) else { return }
            tasks = rows
            taskOffset = rows.count
            moreTasks = rows.count == 20
        } catch {
            guard canRead(generation) else { return }
            tasksError = error.localizedDescription
        }
        do {
            let rows = try await api.getWorkInbox(offset: 0, includeDismissed: dismissed, taskID: nil)
            guard canRead(generation), dismissed == includeDismissed else { return }
            inbox = rows
            inboxOffset = rows.count
            moreInbox = rows.count == 20
        } catch {
            guard canRead(generation) else { return }
            inboxError = error.localizedDescription
        }
        if refreshDetail, canRead(generation), detailRequest == detailGeneration {
            await refreshSelectedDetail(force: true)
        }
    }

    func loadMore(tasks loadingTasks: Bool) async {
        guard active, visible, !isLoading, !isLoadingMore,
              loadingTasks ? moreTasks : moreInbox else { return }
        let generation = listGeneration
        isLoadingMore = true
        defer { if generation == listGeneration { isLoadingMore = false } }
        do {
            if loadingTasks {
                let rows = try await api.getWorkTasks(offset: taskOffset)
                guard canRead(generation) else { return }
                let old = Set(tasks.map(\.id))
                tasks += rows.filter { !old.contains($0.id) }
                taskOffset += rows.count
                moreTasks = rows.count == 20 && taskOffset <= 100000
            } else {
                let dismissed = includeDismissed
                let rows = try await api.getWorkInbox(offset: inboxOffset, includeDismissed: dismissed, taskID: nil)
                guard canRead(generation), dismissed == includeDismissed else { return }
                let old = Set(inbox.map(\.id))
                inbox += rows.filter { !old.contains($0.id) }
                inboxOffset += rows.count
                moreInbox = rows.count == 20 && inboxOffset <= 100000
            }
        } catch {
            guard canRead(generation) else { return }
            if loadingTasks { tasksError = error.localizedDescription }
            else { inboxError = error.localizedDescription }
        }
    }

    func openTask(_ id: UUID) async {
        guard active, visible else { return }
        select(.task(id))
        await refreshSelectedDetail(force: true)
        startDetailPolling()
    }

    func openInbox(_ id: UUID) async {
        guard active, visible else { return }
        select(.inbox(id))
        await refreshSelectedDetail(force: true)
        startDetailPolling()
    }

    func setVoiceActive(_ active: Bool) {
        guard voiceActive != active else { return }
        voiceActive = active
        detailPoll?.cancel(); detailPoll = nil
        startDetailPolling()
    }
    private var normalDetailCadence: TimeInterval {
        let working = selectedTask.map { ["queued", "running", "waiting_for_input", "paused", "cancel_requested", "outcome_unknown"].contains($0.status) } ?? false
        return voiceActive || working ? 1 : 5
    }
    var nextDetailPollDelay: TimeInterval {
        let interval = detailFailures == 0 ? normalDetailCadence : min(30, pow(2, Double(min(detailFailures, 5))))
        let due = detailCompletedAt?.addingTimeInterval(interval) ?? now()
        let server = detailSelection.flatMap { retryNotBefore[$0] } ?? now()
        return max(0, due.timeIntervalSince(now()), server.timeIntervalSince(now()))
    }
    private func select(_ selection: JarvisWorkDetailSelection) {
        guard detailSelection != selection else { return }
        clearDetail()
        detailSelection = selection
        detailFailures = 0; detailCompletedAt = nil; answersDue = nil; answersFailures = 0
    }

    func refreshSelectedDetail(force: Bool = false) async {
        guard active, visible, !isMutating || pendingAction == nil, let selection = detailSelection else { return }
        if let until = retryNotBefore[selection], until > now() { return }
        if let current = detailRead { await current.task.value; return }
        if !force, nextDetailPollDelay > 0 { return }
        let id = UUID(), generation = detailGeneration
        let task = Task<Void, Never> { [weak self] in
            guard let self else { return }
            await self.loadSelectedDetail(selection, generation: generation)
        }
        detailRead = (id, task)
        await task.value
        if detailRead?.id == id { detailRead = nil }
    }
    private func loadSelectedDetail(_ selection: JarvisWorkDetailSelection, generation: Int) async {
        guard canReadDetail(generation), detailSelection == selection else { return }
        isLoadingDetail = selectedTask == nil && selectedInbox == nil
        defer { if generation == detailGeneration { isLoadingDetail = false } }
        do {
            switch selection {
            case .task(let id):
                let row = try await api.getWorkTask(id: id)
                guard canReadDetail(generation), detailSelection == selection else { return }
                guard row.id == id, row.revision >= (selectedTask?.revision ?? 0) else { throw identityError() }
                if row.revision != selectedTask?.revision { taskInbox = []; answersDue = nil }
                selectedTask = row
                detailError = nil
                let answersAllowed = (answersRetryNotBefore[id] ?? now()) <= now()
                if answersAllowed && (answersDue == nil || answersDue! <= now()) {
                    do {
                        let answers = try await api.getWorkInbox(offset: 0, includeDismissed: false, taskID: id)
                        guard canReadDetail(generation), detailSelection == selection else { return }
                        taskInbox = answers.filter { $0.taskID == id && $0.taskRevision == row.revision && $0.sourceAvailable }
                        answersFailures = 0; answersDue = now().addingTimeInterval(normalDetailCadence)
                        answersRetryNotBefore[id] = nil; taskAnswersError = nil
                    } catch {
                        guard canReadDetail(generation), detailSelection == selection else { return }
                        taskInbox = []
                        answersFailures += 1
                        let delay = max(min(30, pow(2, Double(min(answersFailures, 5)))), retryDelay(error) ?? 0)
                        answersDue = now().addingTimeInterval(delay)
                        if let advice = retryDelay(error) { answersRetryNotBefore[id] = now().addingTimeInterval(advice) }
                        taskAnswersError = "Task loaded; answers could not be refreshed. " + error.localizedDescription
                    }
                }
            case .inbox(let id):
                let row = try await api.getWorkInboxItem(id: id)
                guard canReadDetail(generation), detailSelection == selection else { return }
                guard row.item.id == id, row.item.generation >= (selectedInbox?.item.generation ?? 0) else { throw identityError() }
                selectedInbox = row; detailError = nil
            }
            guard canReadDetail(generation), detailSelection == selection else { return }
            detailRefreshedAt = now(); detailCompletedAt = now(); detailFailures = 0
            retryNotBefore[selection] = nil
        } catch {
            guard canReadDetail(generation), detailSelection == selection else { return }
            selectedTask = nil; selectedInbox = nil; taskInbox = []
            detailError = error.localizedDescription; detailCompletedAt = now(); detailFailures += 1
            if let delay = retryDelay(error) { retryNotBefore[selection] = now().addingTimeInterval(delay) }
        }
    }
    private func retryDelay(_ error: Error) -> TimeInterval? {
        guard let delay = (error as? JarvisAPIError)?.retryAfter, delay.isFinite, delay >= 0 else { return nil }
        return delay
    }
    private func startDetailPolling() {
        guard automaticPolling, active, visible, detailSelection != nil, detailPoll == nil else { return }
        detailPoll = Task { [weak self] in
            while !Task.isCancelled {
                guard let delay = await self?.pollDetail() else { return }
                do { try await Task.sleep(nanoseconds: UInt64(min(max(delay, 0.05), 86_400) * 1_000_000_000)) }
                catch { return }
            }
        }
    }
    private func pollDetail() async -> TimeInterval? {
        guard active, visible, detailSelection != nil, !Task.isCancelled else { return nil }
        if isMutating, pendingAction != nil { return normalDetailCadence }
        await refreshSelectedDetail()
        guard active, visible, !Task.isCancelled else { return nil }
        return nextDetailPollDelay
    }

    func suspend() {
        visible = false
        listGeneration += 1
        tasks = []; inbox = []
        tasksError = nil; inboxError = nil
        isLoading = false; isLoadingMore = false
        invalidateDetail(preserveSelection: true)
    }
    func resume(selection: JarvisWorkDetailSelection? = nil) async {
        guard active else { return }
        if let selection { select(selection) }
        beginPresentation()
        async let lists: Void = reload(refreshDetail: false)
        async let detail: Void = refreshSelectedDetail(force: true)
        _ = await (lists, detail)
    }

    func changeTask(_ task: JarvisWorkTask, operation: JarvisTaskOperation, message: String = "") async {
        guard active, visible, pendingAction == nil, !isMutating else { return }
        guard operation != .update || !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            mutationMessage = "Write your reply or correction before sending."
            return
        }
        guard message.count <= 32000 else {
            mutationMessage = "Please shorten the update to 32,000 characters."
            return
        }
        pendingActionLabel = "Task request: " + task.displayTitle
        pendingAction = .task(task.id, operation, JarvisTaskChange(
            requestID: UUID(), expectedRevision: task.revision, message: message))
        await retry()
    }

    func changeInbox(_ detail: JarvisInboxDetail, operation: JarvisInboxOperation) async {
        guard active, visible, pendingAction == nil, !isMutating else { return }
        guard operation == .dismiss ? detail.item.canDismiss : detail.canReplay else { return }
        pendingActionLabel = operation == .replay ? "Read this answer aloud again" : "Dismiss this inbox item"
        pendingAction = .inbox(detail.item.id, operation, JarvisInboxChange(
            requestID: UUID(), expectedGeneration: detail.item.generation))
        await retry()
    }

    func retry() async {
        guard active, visible, !isMutating, let action = pendingAction else { return }
        isMutating = true
        mutationMessage = nil
        // Hide potentially superseded content before any state-changing request.
        invalidateDetail(preserveSelection: true)
        let detailRequest = detailGeneration
        defer { isMutating = false }
        do {
            switch action {
            case let .task(id, operation, body):
                let result = try await api.changeWorkTask(id: id, operation: operation, body: body)
                guard result.id == id, result.revision > body.expectedRevision else { throw identityError() }
            case let .inbox(id, operation, body):
                let result = try await api.changeWorkInbox(id: id, operation: operation, body: body)
                guard result.id == id, result.generation > body.expectedGeneration else { throw identityError() }
            }
            guard active else { return }
            lastAcceptedAction = action
            pendingAction = nil
            pendingActionLabel = nil
            mutationMessage = "Request accepted. Refresh status below; acceptance is not completion or proof of hearing."
            if visible {
                await reload(refreshDetail: false)
                if detailRequest == detailGeneration {
                    switch action {
                    case let .task(id, _, _): await openTask(id)
                    case let .inbox(id, _, _): await openInbox(id)
                    }
                }
            }
        } catch {
            guard active else { return }
            if let error = error as? JarvisAPIError,
               let status = error.statusCode, [400, 401, 403, 404, 409, 422].contains(status) {
                pendingAction = nil
                pendingActionLabel = nil
                mutationMessage = "Request rejected. Refresh before another change. " + error.localizedDescription
            } else {
                // Keep the exact body/identity across refresh, navigation and
                // backgrounding. A lost response must not become a new action.
                mutationMessage = "Acceptance is uncertain. Retry the same request to check it safely. " + error.localizedDescription
            }
        }
    }

    func clearDetail() {
        invalidateDetail(preserveSelection: false)
    }
    private func invalidateDetail(preserveSelection: Bool) {
        detailPoll?.cancel(); detailPoll = nil
        detailRead?.task.cancel(); detailRead = nil
        detailGeneration += 1
        if !preserveSelection { detailSelection = nil }
        selectedTask = nil
        taskInbox = []
        selectedInbox = nil
        detailError = nil
        taskAnswersError = nil
        isLoadingDetail = false
        detailRefreshedAt = nil
        detailCompletedAt = nil
    }

    func hide() {
        visible = false
        listGeneration += 1
        tasks = []; inbox = []
        tasksError = nil; inboxError = nil
        moreTasks = false; moreInbox = false
        isLoading = false; isLoadingMore = false
        clearDetail()
    }

    func deactivate() {
        active = false
        hide()
        pendingAction = nil
        pendingActionLabel = nil
        mutationMessage = nil
        lastAcceptedAction = nil
        retryNotBefore = [:]
        answersRetryNotBefore = [:]
    }

    private func canRead(_ generation: Int) -> Bool {
        active && visible && generation == listGeneration && !Task.isCancelled
    }
    private func canReadDetail(_ generation: Int) -> Bool {
        active && visible && generation == detailGeneration && !Task.isCancelled
    }
    private func identityError() -> JarvisAPIError {
        JarvisAPIError(code: "work_identity_mismatch", message: "Jarvis returned an unexpected work identity. Refresh to check its state.")
    }
}
