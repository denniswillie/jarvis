import Foundation

/// Host-only product-state checks. No app, simulator, provider or database.
@MainActor
private final class PhoneWorkspaceFixture: JarvisWorkspaceAccess, JarvisWorkAccess {
    var requestReads = 0
    var requestError: Error?
    var holdRequests = false
    var waitingRequests: CheckedContinuation<JarvisWorkspaceRequestPage, Error>?
    var requests: [UUID: JarvisWorkspaceRequest] = [:]
    var acknowledgements: [JarvisWorkspaceAcknowledgement] = []
    var heldClaimID: UUID?
    var waitingClaim: CheckedContinuation<JarvisWorkspaceRequestResponse, Error>?
    var workTasks: [JarvisWorkTask] = []
    var inboxRows: [JarvisInboxItem] = []
    var holdTasks = false
    var waitingTasks: CheckedContinuation<[JarvisWorkTask], Error>?
    var taskReads = 0
    var taskError: Error?
    var inboxError: Error?
    var holdCompletion = false
    var waitingCompletion: CheckedContinuation<JarvisWorkspaceRequestResponse, Error>?
    var fileContents: [UUID: JarvisWorkspaceFileContent] = [:]
    var holdFiles = false
    var waitingFile: CheckedContinuation<JarvisWorkspaceFileContent, Error>?
    var fileReads: [(fileID: UUID, sessionID: UUID)] = []

    func listWorkspaceRequests(sessionID: UUID) async throws -> JarvisWorkspaceRequestPage {
        requestReads += 1
        if holdRequests { return try await withCheckedThrowingContinuation { waitingRequests = $0 } }
        if let requestError { throw requestError }
        return JarvisWorkspaceRequestPage(items: Array(requests.values))
    }
    func acknowledgeWorkspaceRequest(requestID: UUID, ack: JarvisWorkspaceAcknowledgement) async throws -> JarvisWorkspaceRequestResponse {
        guard let request = requests[requestID] else {
            throw JarvisWorkspaceError(code: "unexpected_ack", message: "No delivery was authorized in this fixture")
        }
        acknowledgements.append(ack)
        let result = Self.applying(ack, to: request)
        requests[requestID] = result
        if ack.state == .claimed, requestID == heldClaimID {
            return try await withCheckedThrowingContinuation { waitingClaim = $0 }
        }
        if ack.state.isTerminal, holdCompletion {
            return try await withCheckedThrowingContinuation { waitingCompletion = $0 }
        }
        return JarvisWorkspaceRequestResponse(request: result)
    }
    func getWorkspaceFileContent(fileID: UUID, sessionID: UUID) async throws -> JarvisWorkspaceFileContent {
        fileReads.append((fileID, sessionID))
        guard let file = fileContents[fileID] else {
            throw JarvisWorkspaceError(code: "unexpected_file", message: "No file was authorized in this fixture")
        }
        if holdFiles { return try await withCheckedThrowingContinuation { waitingFile = $0 } }
        return file
    }
    func getWorkTasks(offset: Int) async throws -> [JarvisWorkTask] {
        taskReads += 1
        if let taskError { throw taskError }
        if holdTasks { return try await withCheckedThrowingContinuation { waitingTasks = $0 } }
        return workTasks
    }
    func getWorkInbox(offset: Int, includeDismissed: Bool, taskID: UUID?) async throws -> [JarvisInboxItem] {
        if let inboxError { throw inboxError }
        return inboxRows
    }
    func getWorkTask(id: UUID) async throws -> JarvisWorkTask {
        guard let task = workTasks.first(where: { $0.id == id }) else { throw unexpectedAction() }
        return task
    }
    func getWorkInboxItem(id: UUID) async throws -> JarvisInboxDetail { throw unexpectedAction() }
    func changeWorkTask(id: UUID, operation: JarvisTaskOperation, body: JarvisTaskChange) async throws -> JarvisWorkTask { throw unexpectedAction() }
    func changeWorkInbox(id: UUID, operation: JarvisInboxOperation, body: JarvisInboxChange) async throws -> JarvisInboxChangeReceipt { throw unexpectedAction() }
    private func unexpectedAction() -> JarvisWorkspaceError { .init(code: "unexpected_work_action", message: "Read-only task projection fixture") }
    static func applying(_ ack: JarvisWorkspaceAcknowledgement, to request: JarvisWorkspaceRequest) -> JarvisWorkspaceRequest {
        JarvisWorkspaceRequest(requestID: request.id, action: request.action, artifactKind: request.artifactKind,
            artifactID: request.artifactID, versionID: request.versionID, format: request.format, contentReady: request.contentReady,
            fileID: request.fileID, byteCount: request.byteCount, contentSHA256: request.contentSHA256, state: ack.state,
            createdAt: request.createdAt, expiresAt: request.expiresAt, errorCode: ack.errorCode, localReceipt: ack.localReceipt)
    }
}

@main
enum PhoneWorkspaceRegression {
    @MainActor
    static func main() async throws {
        var checks = 0
        func check(_ condition: Bool, _ name: String) {
            precondition(condition, name); checks += 1
        }
        let id = UUID(), session = UUID(), version = "file-exact-v1"
        var clock = Date(timeIntervalSince1970: 1_793_570_000)
        let api = PhoneWorkspaceFixture()
        let model = JarvisWorkspaceViewModel(api: api, sessionID: session, automaticPolling: false, now: { clock })
        model.setVisibility(true, foreground: true)
        check(!model.canHandleDeliveries, "Unverified owner cannot claim deliveries")
        let workTask = JarvisWorkTask(id: UUID(), title: "Choose a section", requestedOutcome: "Edit a clip", status: "waiting_for_input",
            revision: 3, reason: nil, updatedAt: "2026-11-01T22:00:00Z")
        let inboxItem = JarvisInboxItem(id: UUID(), sourceNamespace: "task_result", taskID: workTask.id, taskRevision: 3,
            state: "pending", generation: 1, attemptCount: 0, sourceAvailable: true, sourceUnavailableReason: nil,
            phoneNoticeState: nil, updatedAt: "2026-11-01T22:00:00Z")
        api.workTasks = [workTask]; api.inboxRows = [inboxItem]
        await model.refresh()
        check(model.tasks.first?.id == workTask.id && model.inbox.first?.id == inboxItem.id,
              "Tasks and results do not depend on note readiness")
        check(api.requestReads == 0, "Unverified owner never polls delivery requests")
        let disabledAPI = PhoneWorkspaceFixture()
        disabledAPI.taskError = JarvisAPIError(code: "tasks_disabled", message: "Retired tasks", statusCode: 503)
        disabledAPI.inboxError = JarvisAPIError(code: "attention_disabled", message: "Retired inbox", statusCode: 503)
        let disabled = JarvisWorkspaceViewModel(api: disabledAPI, sessionID: UUID(), automaticPolling: false)
        disabled.setVisibility(true, foreground: true)
        await disabled.refresh()
        check(disabled.tasks.isEmpty && disabled.inbox.isEmpty && disabled.tasksError == nil && disabled.inboxError == nil,
              "Explicitly disabled legacy panels do not appear as failures")
        disabledAPI.taskError = JarvisAPIError(code: "database_unavailable", message: "Actual task outage", statusCode: 503)
        disabledAPI.inboxError = JarvisAPIError(code: "service_unavailable", message: "Actual inbox outage", statusCode: 503)
        await disabled.refresh()
        check(disabled.canRead && disabled.tasksError != nil && disabled.inboxError != nil,
              "Transient outages remain visible without retiring authorized access")
        disabledAPI.taskError = nil; disabledAPI.inboxError = nil
        disabledAPI.workTasks = [workTask]; disabledAPI.inboxRows = [inboxItem]
        await disabled.refresh()
        check(disabled.tasks.first?.id == workTask.id && disabled.inbox.first?.id == inboxItem.id,
              "Configured legacy resources remain available; no data or reminder feature is removed")
        disabledAPI.inboxError = JarvisAPIError(code: "unauthorized", message: "Actual authorization failure", statusCode: 401)
        await disabled.refresh()
        check(!disabled.canRead && disabled.tasks.isEmpty && disabled.inbox.isEmpty,
              "A 401 retires access and clears previously loaded protected content")
        let retiredTaskReads = disabledAPI.taskReads
        await disabled.refresh()
        check(disabledAPI.taskReads == retiredTaskReads && !disabled.canRead,
              "Retired authorization prevents subsequent protected resource reads")
        let taskReads = api.taskReads
        model.setVisibility(false, foreground: false)
        clock = clock.addingTimeInterval(10)
        await model.refresh(force: false)
        check(api.taskReads == taskReads, "Hidden app does not poll product resources")
        model.setVisibility(true, foreground: true)
        await model.refresh(force: false)
        check(api.taskReads > taskReads, "Foreground return revalidates tasks")
        model.lockJournal(clearRemoteContent: true)
        check(model.tasks.isEmpty && model.inbox.isEmpty && !model.canRead,
              "Authorization retirement clears all remote product content")

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-phone-regression-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("Exact saved version".utf8), requestID = UUID(), fileID = UUID()
        let request = JarvisWorkspaceRequest(requestID: requestID, action: .download, artifactKind: "note",
            artifactID: id, versionID: version, format: "text", contentReady: true,
            fileID: fileID, byteCount: bytes.count, contentSHA256: JarvisWorkspaceLocalStore.digest(bytes),
            state: .pending, createdAt: "2026-11-01T22:00:00Z", expiresAt: "2026-11-01T22:05:00Z",
            errorCode: nil, localReceipt: nil)
        let store = try JarvisWorkspaceLocalStore(ownerKey: "fixture-owner", root: root)
        _ = try await store.prepareClaim(request, sessionID: session)
        try await store.confirmClaim(id: requestID)
        try await store.beginEffect(id: requestID)
        let ack = try await store.saveDownload(id: requestID, file: .init(data: bytes, filename: "note.txt", contentType: "text/plain"))
        check(ack.state == .savedLocal && ack.localReceipt?.versionID == version,
              "Save receipt pins the exact selected version")
        let reopenedStore = try JarvisWorkspaceLocalStore(ownerKey: "fixture-owner", root: root)
        let offline = try await reopenedStore.offlineContent(id: requestID)
        check(offline.data == bytes, "Actual local bytes reopen through a fresh store without network")
        let journal = try await reopenedStore.entry(id: requestID)
        check(journal?.acknowledgement == ack && journal?.acknowledged == false,
              "Terminal local result persists before server acknowledgement")
        let otherOwner = try JarvisWorkspaceLocalStore(ownerKey: "other-fixture-owner", root: root)
        let isolated = try await otherOwner.entries()
        check(isolated.isEmpty, "Fixture owners have disjoint saved libraries")

        func downloadRequest(format: String = "text", kind: String = "file", data: Data? = nil) -> JarvisWorkspaceRequest {
            let payload = data ?? bytes
            return JarvisWorkspaceRequest(requestID: UUID(), action: .download, artifactKind: kind,
                artifactID: id, versionID: version, format: format, contentReady: true,
                fileID: fileID, byteCount: payload.count, contentSHA256: JarvisWorkspaceLocalStore.digest(payload),
                state: .pending, createdAt: request.createdAt, expiresAt: request.expiresAt,
                errorCode: nil, localReceipt: nil)
        }
        let ownerDirectory = root.appendingPathComponent(JarvisWorkspaceLocalStore.digest(Data("fixture-owner".utf8)))
        for format in ["text", "json"] {
            let payload = format == "text" ? bytes : Data("{\"value\":null}".utf8)
            let mime = format == "text" ? "text/plain" : "application/json"
            for legacy in [false, true] {
                let interrupted = downloadRequest(format: format, kind: "note", data: payload)
                _ = try await store.prepareClaim(interrupted, sessionID: session)
                try await store.confirmClaim(id: interrupted.id)
                try await store.beginEffect(id: interrupted.id)
                let output = ownerDirectory.appendingPathComponent(interrupted.id.uuidString.lowercased() + ".download")
                if !legacy {
                    // Force the byte write to fail after metadata persistence,
                    // then model process loss immediately after the byte write.
                    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
                    try Data("fixture".utf8).write(to: output.appendingPathComponent("sentinel"))
                    do {
                        _ = try await store.saveDownload(id: interrupted.id,
                            file: .init(data: payload, filename: "Recovered.\(format)", contentType: mime))
                        preconditionFailure("Fixture unexpectedly replaced a nonempty directory")
                    } catch {
                        let prepared = try await store.entry(id: interrupted.id)
                        check(prepared?.contentType == mime && prepared?.filename == "Recovered.\(format)" &&
                              prepared?.phase == .effectStarted && prepared?.acknowledgement == nil,
                              "\(format) representation is durable before bytes; failed write never completes")
                    }
                    try FileManager.default.removeItem(at: output)
                }
                try payload.write(to: output, options: .atomic)
                let restarted = try JarvisWorkspaceLocalStore(ownerKey: "fixture-owner", root: root)
                let recoveredAck = try await restarted.reconcileDownload(id: interrupted.id)
                let recovered = try await restarted.offlineContent(id: interrupted.id)
                check(recoveredAck.localReceipt?.versionID == version && recovered.data == payload &&
                      recovered.contentType == mime, "\(format) recovery keeps exact bytes, version and readable MIME (legacy=\(legacy))")
                if format == "text" {
                    check(String(data: recovered.data, encoding: .utf8) == "Exact saved version",
                          "Recovered text is decodable by the actual text preview branch")
                } else {
                    check((try JSONSerialization.jsonObject(with: recovered.data) as? [String: Any])?["value"] is NSNull,
                          "Historical JSON remains readable without the retired note model")
                }
            }
        }
        let unknownFile = downloadRequest(format: "json", kind: "file")
        _ = try await store.prepareClaim(unknownFile, sessionID: session)
        try await store.confirmClaim(id: unknownFile.id)
        try await store.beginEffect(id: unknownFile.id)
        try bytes.write(to: ownerDirectory.appendingPathComponent(unknownFile.id.uuidString.lowercased() + ".download"))
        _ = try await store.reconcileDownload(id: unknownFile.id)
        check(try await store.offlineContent(id: unknownFile.id).contentType == "application/octet-stream",
              "Legacy arbitrary FILE recovery never guesses MIME from a note export format")

        let downloadAPI = PhoneWorkspaceFixture()
        let exactDownload = downloadRequest()
        downloadAPI.requests[exactDownload.id] = exactDownload
        downloadAPI.fileContents[fileID] = .init(data: bytes, filename: "old-note.txt", contentType: "text/plain")
        downloadAPI.holdCompletion = true
        let downloadModel = JarvisWorkspaceViewModel(api: downloadAPI, sessionID: session, store: store, automaticPolling: false)
        downloadModel.setVisibility(true, foreground: true)
        let saving = Task { await downloadModel.perform(exactDownload) }
        for _ in 0..<100 {
            if downloadAPI.waitingCompletion != nil { break }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        precondition(downloadAPI.waitingCompletion != nil, "Download completion never started")
        let savedBeforeResponse = try await store.entry(id: exactDownload.id)!
        check(savedBeforeResponse.phase == .completed && savedBeforeResponse.acknowledgement?.state == .savedLocal &&
              savedBeforeResponse.acknowledgement?.localReceipt?.versionID == version && !savedBeforeResponse.acknowledged,
              "Full VM download durably saves the requested old note version before terminal response")
        check(downloadAPI.fileReads.count == 1 && downloadAPI.fileReads[0].fileID == fileID &&
              downloadAPI.fileReads[0].sessionID == session,
              "Full VM download passes its captured Workspace session and separate prepared file identity")
        downloadAPI.waitingCompletion?.resume(throwing: URLError(.networkConnectionLost))
        downloadAPI.waitingCompletion = nil
        await saving.value
        downloadAPI.holdCompletion = false
        await downloadModel.reconcile(savedBeforeResponse)
        await downloadModel.openOffline(savedBeforeResponse)
        check(downloadAPI.fileReads.count == 1 && downloadModel.offlinePreview?.data == bytes,
              "Lost download ACK reconciles and exact saved file reopens locally")

        let wrongRequest = downloadRequest()
        let wrongAPI = PhoneWorkspaceFixture()
        wrongAPI.requests[wrongRequest.id] = wrongRequest
        wrongAPI.fileContents[fileID] = .init(data: Data(repeating: 120, count: bytes.count), filename: "wrong.txt", contentType: "text/plain")
        let wrongModel = JarvisWorkspaceViewModel(api: wrongAPI, sessionID: session, store: store, automaticPolling: false)
        wrongModel.setVisibility(true, foreground: true)
        await wrongModel.perform(wrongRequest)
        let wrongEntry = try await store.entry(id: wrongRequest.id)
        check(wrongEntry?.phase == .effectStarted && wrongEntry?.acknowledgement == nil &&
              wrongAPI.acknowledgements.map(\.state) == [.claimed] &&
              !FileManager.default.fileExists(atPath: ownerDirectory.appendingPathComponent(wrongRequest.id.uuidString.lowercased() + ".download").path),
              "Same-length wrong bytes never produce a local file, completed receipt or saved_local ACK")

        let context = JarvisAuthenticatedContext(ownerID: "fixture-principal",
            backendNamespace: "jarvis-storage-v1:" + String(repeating: "a", count: 64), sessionID: session)
        let ownerKey = try JarvisWorkspaceLocalStore.ownerKey(for: context)
        let anotherSessionContext = JarvisAuthenticatedContext(ownerID: context.ownerID,
            backendNamespace: context.backendNamespace, sessionID: UUID())
        check(try JarvisWorkspaceLocalStore.ownerKey(for: anotherSessionContext) == ownerKey,
              "Verified owner/backend namespace persists across sessions without using session as account identity")
        let differentOwner = JarvisAuthenticatedContext(ownerID: "other-principal", backendNamespace: context.backendNamespace, sessionID: session)
        let differentBackend = JarvisAuthenticatedContext(ownerID: context.ownerID,
            backendNamespace: "jarvis-storage-v1:" + String(repeating: "b", count: 64), sessionID: session)
        check(try JarvisWorkspaceLocalStore.ownerKey(for: differentOwner) != ownerKey && JarvisWorkspaceLocalStore.ownerKey(for: differentBackend) != ownerKey,
              "Owner and backend changes isolate retained journals")
        let metadataJournal = try JarvisWorkspaceLocalStore(ownerKey: ownerKey, root: root, allowsFileBytes: false)
        let metadataModel = JarvisWorkspaceViewModel(api: PhoneWorkspaceFixture(), sessionID: session, automaticPolling: false)
        check(metadataModel.attachJournal(metadataJournal) && !metadataModel.canSaveOffline,
              "Metadata journal does not grant file byte access")
        check(!metadataModel.canPerform(request), "Retired note requests are never claimed")
        do {
            _ = try await metadataJournal.offlineContent(id: request.id)
            preconditionFailure("Metadata-only actor allowed byte access")
        } catch {
            check((error as? JarvisWorkspaceError)?.code == "offline_files_unavailable",
                  "Store rejects bytes independently from UI controls")
        }

        let cancelledDownload = JarvisWorkspaceRequest(requestID: UUID(), action: .download, artifactKind: "file",
            artifactID: id, versionID: version, format: "text", contentReady: true,
            fileID: fileID, byteCount: bytes.count, contentSHA256: JarvisWorkspaceLocalStore.digest(bytes),
            state: .pending, createdAt: "2026-11-01T22:00:00Z", expiresAt: "2026-11-01T22:05:00Z",
            errorCode: nil, localReceipt: nil)
        let retiredAPI = PhoneWorkspaceFixture()
        let fixtureFile = JarvisWorkspaceFileContent(data: bytes, filename: "exact.txt", contentType: "text/plain")
        retiredAPI.requests[cancelledDownload.id] = cancelledDownload
        retiredAPI.fileContents[fileID] = fixtureFile; retiredAPI.holdFiles = true
        let retired = JarvisWorkspaceViewModel(api: retiredAPI, sessionID: session, store: reopenedStore, automaticPolling: false)
        retired.setVisibility(true, foreground: true)
        let inFlightDownload = Task { await retired.perform(cancelledDownload) }
        for _ in 0..<100 {
            if retiredAPI.waitingFile != nil { break }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        precondition(retiredAPI.waitingFile != nil, "Fixture file read never started")
        retired.deactivate()
        let replacementSession = UUID()
        let replacement = JarvisWorkspaceViewModel(api: PhoneWorkspaceFixture(), sessionID: replacementSession,
                                                  store: otherOwner, automaticPolling: false)
        replacement.setVisibility(true, foreground: true)
        retiredAPI.waitingFile?.resume(returning: fixtureFile); retiredAPI.waitingFile = nil
        await inFlightDownload.value
        let cancelledEntry = try await reopenedStore.entry(id: cancelledDownload.id)
        check(cancelledEntry?.phase == .effectStarted && cancelledEntry?.acknowledgement == nil,
              "Late download after context replacement remains uncertain without fabricated save")
        check(retiredAPI.acknowledgements.map(\.state) == [.claimed] && retired.offlinePreview == nil,
              "Retired model neither reports saved_local nor exposes late bytes")
        let replacementEntries = try await otherOwner.entries()
        check(retiredAPI.fileReads.first?.sessionID == session && replacement.canRead &&
              replacement.offlinePreview == nil && replacementEntries.isEmpty,
              "Held old-session bytes cannot enter the replacement account's library or presentation")
        let access = JarvisWorkspaceLocalAccess()
        access.invalidate()
        do {
            _ = try await reopenedStore.saveDownload(id: cancelledDownload.id, file: fixtureFile, access: access)
            preconditionFailure("Revoked lifetime wrote a local download")
        } catch is CancellationError { check(true, "Queued actor write checks the VM lifetime fence") }
        do {
            _ = try await reopenedStore.entries(access: access)
            preconditionFailure("Revoked lifetime read an offline library")
        } catch is CancellationError { check(true, "Queued actor read checks the VM lifetime fence") }
        let cancelledPath = root.appendingPathComponent(JarvisWorkspaceLocalStore.digest(Data("fixture-owner".utf8)))
            .appendingPathComponent(cancelledDownload.id.uuidString.lowercased() + ".download")
        check(!FileManager.default.fileExists(atPath: cancelledPath.path), "Retired download leaves no new saved bytes")

        let filePath = root.appendingPathComponent(JarvisWorkspaceLocalStore.digest(Data("fixture-owner".utf8)))
            .appendingPathComponent(requestID.uuidString.lowercased() + ".download")
        try Data("Corrupt".utf8).write(to: filePath)
        do {
            _ = try await reopenedStore.offlineContent(id: requestID)
            preconditionFailure("Corrupted offline bytes accepted")
        } catch {
            check((error as? JarvisWorkspaceError)?.code == "download_mismatch", "Offline reopen verifies actual byte integrity")
        }
        print("PhoneWorkspaceRegression: \(checks) checks passed (host-only product fixtures)")
    }
}
