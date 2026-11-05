"""Opt-in actual Swift API/models/view-model contracts for durable work."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from uuid import uuid4

SWIFT = r'''
import Foundation

extension ISO8601DateFormatter { static let jarvisFormatter = ISO8601DateFormatter() }

func decode<T: Decodable>(_ value: Any, _ type: T.Type) throws -> T {
    try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: value))
}
func requestBody(_ request: URLRequest) -> [String: Any] {
    if let data = request.httpBody { return try! JSONSerialization.jsonObject(with: data) as! [String: Any] }
    let stream = request.httpBodyStream!
    stream.open()
    defer { stream.close() }
    var data = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while true {
        let count = stream.read(&buffer, maxLength: buffer.count)
        if count <= 0 { break }
        data.append(contentsOf: buffer.prefix(count))
    }
    return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
}

final class WorkProtocol: URLProtocol {
    static var handler: ((URLRequest) -> (Int, Any))!
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (status, payload) = Self.handler(request)
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                                       headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: try! JSONSerialization.data(withJSONObject: payload))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@MainActor final class WorkAPI: JarvisWorkAccess {
    var taskData: [String: Any]
    var detailData: [String: Any]
    var task: JarvisWorkTask { try! decode(taskData, JarvisWorkTask.self) }
    var detail: JarvisInboxDetail { try! decode(detailData, JarvisInboxDetail.self) }
    var inboxChanges: [JarvisInboxChange] = []
    var taskChanges: [JarvisTaskChange] = []
    var failMutation = false
    var rejected = false
    var pauseDetail = false
    var pauseList = false
    var pauseMutation = false
    var detailRead: CheckedContinuation<JarvisInboxDetail, Error>?
    var listRead: CheckedContinuation<[JarvisWorkTask], Error>?
    var mutation: CheckedContinuation<JarvisInboxChangeReceipt, Error>?
    var page: [JarvisWorkTask]?
    var requestedOffsets: [Int] = []
    var requestedTaskFilters: [UUID?] = []

    init(_ task: [String: Any], _ detail: [String: Any]) { taskData = task; detailData = detail }
    func getWorkTasks(offset: Int) async throws -> [JarvisWorkTask] {
        requestedOffsets.append(offset)
        if pauseList { return try await withCheckedThrowingContinuation { listRead = $0 } }
        return offset == 0 ? (page ?? [task]) : []
    }
    func getWorkTask(id: UUID) async throws -> JarvisWorkTask { task }
    func getWorkInbox(offset: Int, includeDismissed: Bool, taskID: UUID?) async throws -> [JarvisInboxItem] {
        requestedTaskFilters.append(taskID)
        return [detail.item]
    }
    func getWorkInboxItem(id: UUID) async throws -> JarvisInboxDetail {
        if pauseDetail { return try await withCheckedThrowingContinuation { detailRead = $0 } }
        return detail
    }
    func changeWorkTask(id: UUID, operation: JarvisTaskOperation, body: JarvisTaskChange) async throws -> JarvisWorkTask {
        taskChanges.append(body)
        taskData["revision"] = body.expectedRevision + 1
        taskData["status"] = operation == .cancel ? "cancelled" : "queued"
        if operation == .update { taskData["requested_outcome"] = body.message }
        return task
    }
    func changeWorkInbox(id: UUID, operation: JarvisInboxOperation, body: JarvisInboxChange) async throws -> JarvisInboxChangeReceipt {
        inboxChanges.append(body)
        if rejected { throw JarvisAPIError(code: "attention_generation_conflict", message: "Changed", statusCode: 409) }
        if failMutation { throw URLError(.timedOut) }
        detailData["generation"] = body.expectedGeneration + 1
        detailData["state"] = operation == .replay ? "pending" : "dismissed"
        if pauseMutation { return try await withCheckedThrowingContinuation { mutation = $0 } }
        return JarvisInboxChangeReceipt(id: id, generation: body.expectedGeneration + 1)
    }
}

@main struct WorkPhoneContract {
    @MainActor static func main() async throws {
        let payload = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [String: Any]
        let taskData = payload["task"] as! [String: Any]
        let detailData = payload["detail"] as! [String: Any]
        let task = try decode(taskData, JarvisWorkTask.self)
        let detail = try decode(detailData, JarvisInboxDetail.self)
        precondition(task.statusLabel == "Needs your input")
        precondition(detail.item.stateLabel == "Playback uncertain" && detail.canReplay)
        precondition(detail.artifacts.count == 1 && detail.artifacts[0].filename == "synthetic-report.txt")
        var phoneData = detailData
        phoneData["state"] = "pending"
        phoneData["phone_notice_state"] = "accepted"
        let onPhone = try decode(phoneData, JarvisInboxDetail.self)
        precondition(onPhone.item.stateLabel == "Notification submitted" && onPhone.canReplay)
        phoneData["phone_notice_state"] = "submitted"
        let submittingPhone = try decode(phoneData, JarvisInboxDetail.self)
        precondition(!submittingPhone.canReplay)
        let revoked = try decode(payload["revoked"]!, JarvisInboxDetail.self)
        precondition(revoked.expertResult == nil && revoked.notification == nil && !revoked.canReplay)
        precondition(revoked.artifacts.isEmpty)
        let reminder = try decode(payload["reminder"]!, JarvisInboxDetail.self)
        precondition(reminder.notification?.message == "Bring the synthetic notebook." && reminder.expertResult == nil)

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [WorkProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let api = JarvisAPIClient(configuration: JarvisConfiguration(
            backendBaseURL: URL(string: "https://work.invalid/api")!, staticBearerToken: "synthetic"), session: session)
        var methodCount = 0
        var reject = false
        var wrongIdentity = false
        var posted: [String: Any]?
        WorkProtocol.handler = { request in
            methodCount += 1
            precondition(request.value(forHTTPHeaderField: "Authorization") == "Bearer synthetic")
            precondition(request.cachePolicy == .reloadIgnoringLocalCacheData)
            if reject { return (409, ["error": ["code": "attention_generation_conflict", "message": "Refresh"]]) }
            if request.httpMethod == "POST" {
                posted = requestBody(request)
                precondition(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
                precondition(request.url!.path == "/api/attention/" + detail.item.id.uuidString.lowercased() + "/replay")
                return (200, ["id": detail.item.id.uuidString, "generation": 2])
            }
            if request.url!.path == "/api/tasks" {
                let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!
                precondition(query.first(where: {$0.name == "limit"})?.value == "20")
                precondition(query.first(where: {$0.name == "offset"})?.value == "20")
                return (200, [taskData])
            }
            if request.url!.path == "/api/attention" {
                let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!
                precondition(query.first(where: {$0.name == "include_dismissed"})?.value == "true")
                precondition(query.first(where: {$0.name == "task_id"})?.value == task.id.uuidString.lowercased())
                return (200, [detailData])
            }
            if request.url!.path.contains("/tasks/") { return (200, taskData) }
            var output = detailData
            if wrongIdentity { output["id"] = UUID().uuidString }
            return (200, output)
        }
        let page = try await api.getWorkTasks(offset: 20)
        precondition(page == [task])
        let related = try await api.getWorkInbox(offset: 0, includeDismissed: true, taskID: task.id)
        precondition(related.first?.id == detail.item.id)
        let fetched = try await api.getWorkInboxItem(id: detail.item.id)
        precondition(fetched == detail)
        precondition(methodCount == 3)
        let requestID = UUID()
        let receipt = try await api.changeWorkInbox(id: detail.item.id, operation: .replay,
            body: JarvisInboxChange(requestID: requestID, expectedGeneration: 1))
        precondition(receipt.generation == 2 && posted?["expected_generation"] as? Int == 1)
        precondition(UUID(uuidString: posted?["request_id"] as! String) == requestID)
        wrongIdentity = true
        do { _ = try await api.getWorkInboxItem(id: detail.item.id); preconditionFailure("Wrong identity accepted") }
        catch { precondition((error as? JarvisAPIError)?.code == "work_identity_mismatch") }
        reject = true
        do { _ = try await api.getWorkTask(id: task.id); preconditionFailure("Rejection accepted") }
        catch { precondition((error as? JarvisAPIError)?.statusCode == 409) }

        let fake = WorkAPI(taskData, detailData)
        let model = JarvisWorkInboxViewModel(api: fake)
        await model.present()
        await model.openTask(task.id)
        precondition(model.selectedTask == task && model.taskInbox.count == 1)
        precondition(fake.requestedTaskFilters.last! == task.id)
        await model.openInbox(detail.item.id)
        fake.detailData["source_available"] = false
        await model.reload()
        precondition(model.selectedInbox?.item.sourceAvailable == false && model.selectedInbox?.expertResult == nil)
        fake.detailData = detailData
        fake.failMutation = true
        await model.changeInbox(detail, operation: .replay)
        precondition(model.pendingAction != nil && model.selectedInbox == nil)
        precondition(fake.inboxChanges.count == 1)
        let pending = model.pendingAction
        model.hide()
        await model.present()
        precondition(model.pendingAction == pending && fake.inboxChanges.count == 1)
        fake.failMutation = false
        await model.retry()
        precondition(fake.inboxChanges.count == 2 && fake.inboxChanges[0] == fake.inboxChanges[1])
        precondition(model.pendingAction == nil && model.selectedInbox?.item.state == "pending")
        precondition(model.selectedInbox?.item.generation == 2)
        precondition(fake.taskChanges.isEmpty)

        await model.openTask(task.id)
        await model.changeTask(task, operation: .update, message: "Use a simulated rover; no real wiring.")
        precondition(fake.taskChanges.count == 1 && model.selectedTask?.revision == 2)
        precondition(model.selectedTask?.requestedOutcome == "Use a simulated rover; no real wiring.")

        fake.detailData = detailData
        fake.rejected = true
        await model.changeInbox(detail, operation: .replay)
        precondition(model.pendingAction == nil && model.mutationMessage?.contains("rejected") == true)
        fake.rejected = false
        fake.pauseDetail = true
        let late = Task { await model.openInbox(detail.item.id) }
        while fake.detailRead == nil { await Task.yield() }
        model.hide()
        fake.detailRead!.resume(returning: detail)
        await late.value
        precondition(model.selectedInbox == nil && !model.isLoadingDetail)
        fake.pauseDetail = false
        fake.pauseList = true
        let listing = Task { await model.present() }
        while fake.listRead == nil { await Task.yield() }
        model.hide()
        fake.listRead!.resume(returning: [task])
        await listing.value
        precondition(model.tasks.isEmpty && !model.isLoading)
        fake.pauseList = false

        fake.page = try (0..<20).map { index in
            var value = taskData
            value["id"] = UUID().uuidString
            return try decode(value, JarvisWorkTask.self)
        }
        await model.present()
        precondition(model.moreTasks && model.tasks.count == 20)
        await model.loadMore(tasks: true)
        precondition(!model.moreTasks && fake.requestedOffsets.last == 20)

        fake.pauseMutation = true
        let changing = Task { await model.changeInbox(detail, operation: .replay) }
        while fake.mutation == nil { await Task.yield() }
        await model.openTask(task.id)
        fake.mutation!.resume(returning: JarvisInboxChangeReceipt(id: detail.item.id, generation: 2))
        await changing.value
        precondition(model.selectedTask?.id == task.id && model.selectedInbox == nil)
        model.deactivate()
        await model.present()
        precondition(model.tasks.isEmpty && model.inbox.isEmpty && model.pendingAction == nil)
        print("Work phone contract passed")
    }
}
'''


@unittest.skipUnless(os.getenv("JARVIS_IOS_CONTRACT_TESTS") == "1" and shutil.which("swiftc"),
                     "Opt-in native Swift compiler contract")
class WorkPhoneContractTests(unittest.TestCase):
    def test_actual_models_client_and_lifecycle(self):
        task_id, intent_id = str(uuid4()), str(uuid4())
        task = {"id": task_id, "title": "Synthetic robot", "requested_outcome": "Ask before wiring.",
                "status": "waiting_for_input", "revision": 1, "reason": None, "updated_at": "2026-09-16T10:00:00Z"}
        item = {"id": intent_id, "source_namespace": "task_result", "task_id": task_id, "task_revision": 1,
                "state": "unknown", "generation": 1, "attempt_count": 1, "source_available": True,
                "source_unavailable_reason": None, "updated_at": "2026-09-16T10:00:00Z", "speech_prepared": True,
                "expert_result": {"answer": "Which controller?", "status": "waiting_for_input",
                                  "uncertainties": ["No controller identified."]}, "notification": None,
                "artifacts": [{"file_id": str(uuid4()), "filename": "synthetic-report.txt"}]}
        revoked = item | {"source_available": False}
        reminder = item | {"id": str(uuid4()), "source_namespace": "reminder_event", "task_id": None,
                           "expert_result": None, "notification": {"title": "Notebook",
                           "message": "Bring the synthetic notebook.", "authored_by": "stored_reminder"}}
        ios = Path(__file__).resolve().parents[2] / "ios/Jarvis/Jarvis"
        sources = [ios / "Models" / name for name in (
            "JarvisModels.swift", "JarvisMemoryModels.swift", "JarvisFileModels.swift",
            "JarvisWorkModels.swift", "JarvisReminderModels.swift", "JarvisCalendarModels.swift")]
        sources += [ios / "Services/JarvisAPIClient.swift", ios / "ViewModels/JarvisWorkInboxViewModel.swift"]
        with tempfile.TemporaryDirectory(prefix="jarvis-work-phone-swift-") as directory:
            root = Path(directory)
            (root / "Contract.swift").write_text(SWIFT)
            (root / "payloads.json").write_text(json.dumps({"task": task, "detail": item, "revoked": revoked, "reminder": reminder}))
            compiled = subprocess.run(["swiftc", *map(str, sources), str(root / "Contract.swift"),
                                       "-o", str(root / "check")], capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            ran = subprocess.run([str(root / "check"), str(root / "payloads.json")],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("Work phone contract passed", ran.stdout)
