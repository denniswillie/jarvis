import Foundation

@MainActor
private final class PhoneWorkFixture: JarvisWorkAccess {
    var tasks: [UUID: JarvisWorkTask] = [:]
    var details: [UUID: JarvisInboxDetail] = [:]
    var answers: [JarvisInboxItem] = []
    var taskError: Error?
    var answerError: Error?
    var detailError: Error?
    var heldTaskID: UUID?
    var waitingTask: CheckedContinuation<JarvisWorkTask, Error>?
    var taskReads = 0, answerReads = 0, detailReads = 0
    var loseNextMutationResponse = false
    var changes: [JarvisTaskChange] = []
    var receipts: [UUID: JarvisWorkTask] = [:]

    func getWorkTasks(offset: Int) async throws -> [JarvisWorkTask] { Array(tasks.values) }
    func getWorkInbox(offset: Int, includeDismissed: Bool, taskID: UUID?) async throws -> [JarvisInboxItem] {
        if taskID != nil { answerReads += 1; if let answerError { throw answerError } }
        return answers
    }
    func getWorkTask(id: UUID) async throws -> JarvisWorkTask {
        taskReads += 1
        if let taskError { throw taskError }
        if heldTaskID == id { return try await withCheckedThrowingContinuation { waitingTask = $0 } }
        guard let task = tasks[id] else { throw JarvisAPIError(code: "missing", message: "Missing", statusCode: 404) }
        return task
    }
    func getWorkInboxItem(id: UUID) async throws -> JarvisInboxDetail {
        detailReads += 1
        if let detailError { throw detailError }
        guard let detail = details[id] else { throw JarvisAPIError(code: "missing", message: "Missing", statusCode: 404) }
        return detail
    }
    func changeWorkTask(id: UUID, operation: JarvisTaskOperation, body: JarvisTaskChange) async throws -> JarvisWorkTask {
        changes.append(body)
        if let result = receipts[body.requestID] { return result }
        guard let current = tasks[id], current.revision == body.expectedRevision else {
            throw JarvisAPIError(code: "task_conflict", message: "Changed", statusCode: 409)
        }
        let updated = JarvisWorkTask(id: id, title: current.title, requestedOutcome: body.message,
            status: "queued", revision: current.revision + 1, reason: nil, updatedAt: current.updatedAt)
        tasks[id] = updated; receipts[body.requestID] = updated
        if loseNextMutationResponse { loseNextMutationResponse = false; throw URLError(.networkConnectionLost) }
        return updated
    }
    func changeWorkInbox(id: UUID, operation: JarvisInboxOperation, body: JarvisInboxChange) async throws -> JarvisInboxChangeReceipt {
        preconditionFailure("Read fixture must not implicitly mutate an inbox delivery")
    }
}

@main
enum PhoneWorkInboxRegression {
    @MainActor
    static func main() async throws {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let id = UUID(), other = UUID(), inboxID = UUID()
        func task(_ id: UUID, _ revision: Int, status: String = "running") -> JarvisWorkTask {
            JarvisWorkTask(id: id, title: "Edit this clip", requestedOutcome: "Prepare a result", status: status,
                revision: revision, reason: nil, updatedAt: "2026-11-01T22:00:00Z")
        }
        func answer(_ revision: Int) -> JarvisInboxItem {
            JarvisInboxItem(id: UUID(), sourceNamespace: "task_result", taskID: id, taskRevision: revision,
                state: "pending", generation: 1, attemptCount: 0, sourceAvailable: true,
                sourceUnavailableReason: nil, phoneNoticeState: nil, updatedAt: "2026-11-01T22:00:00Z")
        }
        func detail(_ generation: Int, available: Bool) throws -> JarvisInboxDetail {
            let object: [String: Any] = [
                "id": inboxID.uuidString, "source_namespace": "task_result", "task_id": id.uuidString,
                "task_revision": 2, "state": available ? "pending" : "revoked", "generation": generation,
                "attempt_count": 0, "source_available": available, "updated_at": "2026-11-01T22:00:00Z",
                "expert_result": ["answer": "Choose the next section", "status": "waiting_for_input", "uncertainties": []],
                "artifacts": [["file_id": UUID().uuidString, "filename": "result.txt"]], "speech_prepared": true
            ]
            return try JSONDecoder().decode(JarvisInboxDetail.self, from: JSONSerialization.data(withJSONObject: object))
        }
        @MainActor func waitForRead(_ fixture: PhoneWorkFixture) async throws {
            for _ in 0..<100 {
                if fixture.waitingTask != nil { return }
                try await Task.sleep(nanoseconds: 1_000_000)
            }
            preconditionFailure("Fixture task read never began")
        }
        var clock = Date(timeIntervalSince1970: 1_793_570_000)
        let fixture = PhoneWorkFixture()
        let first = task(id, 1), second = task(id, 2), otherTask = task(other, 1)
        fixture.tasks = [id: first, other: otherTask]
        fixture.answers = [answer(1), answer(0)]
        let model = JarvisWorkInboxViewModel(api: fixture, automaticPolling: false, now: { clock })
        model.beginPresentation()
        await model.openTask(id)
        check(model.selectedTask == first && model.detailSelection == .task(id), "Task detail pins the exact target")
        check(model.taskInbox.count == 1 && model.taskInbox.first?.taskRevision == 1,
              "Current task answers exclude older task revisions")
        check(model.nextDetailPollDelay == 1, "Selected nonterminal task uses the agreed short cadence")

        fixture.heldTaskID = id
        let refreshing = Task { await model.refreshSelectedDetail(force: true) }
        try await waitForRead(fixture)
        check(model.selectedTask == first && !model.isLoadingDetail, "Polling preserves rendered content instead of clearing the editor")
        let reads = fixture.taskReads
        let joining = Task { await model.refreshSelectedDetail(force: true) }
        for _ in 0..<5 { await Task.yield() }
        check(fixture.taskReads == reads, "Overlapping detail readers share one transport request")
        fixture.tasks[id] = second; fixture.answers = [answer(1), answer(2)]
        fixture.waitingTask?.resume(returning: second); fixture.waitingTask = nil; fixture.heldTaskID = nil
        await refreshing.value; await joining.value
        check(model.selectedTask == second && model.taskInbox.allSatisfy { $0.taskRevision == 2 }, "New task revision replaces only its own current projection")

        var draft = JarvisWorkReplyDraft()
        draft.update("Keep the chorus", for: first)
        check(draft.needsReview(for: second) && draft.baseTask?.revision == 1, "Polling cannot silently rebase a typed reply")
        draft.review(second)
        check(!draft.needsReview(for: second) && draft.text == "Keep the chorus", "Deliberate review updates the base without losing text")
        draft.review(otherTask)
        check(draft.baseTask?.id == id && draft.needsReview(for: otherTask), "A reply cannot silently move to another task")
        let body = JarvisTaskChange(requestID: UUID(), expectedRevision: 2, message: "Keep the chorus")
        draft.accepted(.task(id, .cancel, body))
        check(!draft.text.isEmpty, "Unrelated accepted action does not erase a reply")
        draft.update("Keep the chorus longer", for: second)
        draft.accepted(.task(id, .update, body))
        check(draft.text == "Keep the chorus longer", "Edits made after submit survive an older acceptance")
        draft.update("Keep the chorus", for: second)
        draft.accepted(.task(id, .update, body))
        check(draft.text.isEmpty && draft.baseTask == nil, "Exact accepted update clears its own unchanged draft")

        model.suspend()
        check(model.detailSelection == .task(id) && model.selectedTask == nil, "Background hides content while retaining the target identity")
        let suspendedReads = fixture.taskReads
        await model.refreshSelectedDetail(force: true)
        check(fixture.taskReads == suspendedReads, "Background cannot poll or refresh detail")
        fixture.tasks[id] = task(id, 3)
        await model.resume()
        check(model.selectedTask?.revision == 3 && model.detailSelection == .task(id), "Foreground return reauthorizes and reopens the same target")
        model.hide()
        await model.resume(selection: .task(other))
        check(model.selectedTask?.id == other, "Visible navigation identity can restore a target even before an initial read started")
        await model.openTask(id)

        fixture.taskError = JarvisAPIError(code: "busy", message: "Busy", statusCode: 503, retryAfter: 10)
        await model.refreshSelectedDetail(force: true)
        check(model.selectedTask == nil && model.nextDetailPollDelay == 10, "Failed revalidation hides content and honors server delay")
        let embargoReads = fixture.taskReads
        fixture.taskError = nil
        await model.refreshSelectedDetail(force: true)
        check(fixture.taskReads == embargoReads, "Manual detail refresh cannot bypass Retry-After")
        model.suspend(); await model.resume()
        check(fixture.taskReads == embargoReads, "Foreground return cannot bypass Retry-After")
        clock = clock.addingTimeInterval(10)
        await model.refreshSelectedDetail()
        check(model.selectedTask?.revision == 3, "Exact task recovers after the delay")

        fixture.answerError = JarvisAPIError(code: "answers_busy", message: "Answers busy", statusCode: 503, retryAfter: 12)
        fixture.tasks[id] = task(id, 4)
        await model.refreshSelectedDetail(force: true)
        check(model.selectedTask?.revision == 4 && model.taskAnswersError != nil, "Unavailable answers do not suppress healthy task status")
        let answerReads = fixture.answerReads
        clock = clock.addingTimeInterval(1); fixture.tasks[id] = task(id, 5)
        await model.refreshSelectedDetail()
        check(model.selectedTask?.revision == 5 && fixture.answerReads == answerReads,
              "New status revisions do not bypass the answers resource delay")
        check(model.taskAnswersError != nil, "Answers failure remains visible during its embargo")
        fixture.answerError = nil; fixture.answers = [answer(5)]
        clock = clock.addingTimeInterval(11)
        await model.refreshSelectedDetail()
        check(model.taskAnswersError == nil && model.taskInbox.first?.taskRevision == 5, "Answers recover independently")

        fixture.loseNextMutationResponse = true
        await model.changeTask(fixture.tasks[id]!, operation: .update, message: "Use the second section")
        guard let pending = model.pendingAction else { preconditionFailure("Lost mutation discarded its identity") }
        check(model.lastAcceptedAction == nil && fixture.receipts.count == 1, "Lost response does not claim acceptance even after remote commit")
        model.suspend(); await model.resume()
        check(model.pendingAction == pending, "Background/foreground preserves the exact uncertain mutation")
        await model.retry()
        check(fixture.changes.count == 2 && fixture.changes[0] == fixture.changes[1] && fixture.receipts.count == 1,
              "Retry reuses request ID, base and body rather than producing another update")
        check(model.pendingAction == nil && model.lastAcceptedAction == pending, "Confirmed receipt exposes exact accepted action")

        fixture.details[inboxID] = try detail(1, available: true)
        await model.openInbox(inboxID)
        check(model.selectedInbox?.expertResult != nil, "Exact available answer opens")
        fixture.details[inboxID] = try detail(2, available: false)
        await model.refreshSelectedDetail(force: true)
        check(model.selectedInbox?.item.sourceAvailable == false && model.selectedInbox?.expertResult == nil && model.selectedInbox?.artifacts.isEmpty == true,
              "Source withdrawal removes old answer and artifact links")

        fixture.heldTaskID = id
        let oldSelection = Task { await model.openTask(id) }
        try await waitForRead(fixture)
        await model.openTask(other)
        fixture.waitingTask?.resume(returning: task(id, 6)); fixture.waitingTask = nil
        await oldSelection.value
        check(model.selectedTask?.id == other && model.detailSelection == .task(other), "Late old detail cannot steal a newer target")
        fixture.heldTaskID = id
        let retired = Task { await model.openTask(id) }
        try await waitForRead(fixture)
        model.deactivate()
        fixture.waitingTask?.resume(returning: task(id, 6)); fixture.waitingTask = nil
        await retired.value
        check(model.selectedTask == nil && model.detailSelection == nil && model.pendingAction == nil,
              "Account/model retirement rejects late detail and clears mutation UI")
        print("PhoneWorkInboxRegression: \(checks) checks passed (host-only detail/draft fixtures)")
    }
}
