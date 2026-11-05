import Foundation

private final class PluginFixture: JarvisPluginsAccess {
    var snapshot: JarvisPluginCatalogSnapshot
    var error: Error?
    var holds = false
    var waits: [CheckedContinuation<JarvisPluginCatalogSnapshot, Error>] = []
    var writes: [JarvisPluginPolicyChange] = []
    var holdsWrites = false
    var writeWait: CheckedContinuation<JarvisPluginPolicyReceipt, Error>?
    var writeError: Error?
    init(_ snapshot: JarvisPluginCatalogSnapshot) { self.snapshot = snapshot }
    func fetchPluginCatalog() async throws -> JarvisPluginCatalogSnapshot {
        if holds { return try await withCheckedThrowingContinuation { waits.append($0) } }
        if let error { throw error }
        return snapshot
    }
    func changePluginPolicy(_ change: JarvisPluginPolicyChange) async throws -> JarvisPluginPolicyReceipt {
        writes.append(change)
        if holdsWrites { return try await withCheckedThrowingContinuation { writeWait = $0 } }
        if let writeError { throw writeError }
        throw JarvisAPIError(code: "plugin_policy_changes_unavailable", message: "Not connected", statusCode: 503)
    }
}

private final class PluginProtocol: URLProtocol {
    static var response: ((URLRequest) throws -> (Int, Data))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, data) = try Self.response!(request)
            let reply = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: reply, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@main
enum PhonePluginsRegression {
    @MainActor
    static func main() async throws {
        var count = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); count += 1 }
        let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let base = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        func snapshot(revision: Int = 7, enabled: Bool = true, preferred: Bool = true,
                      canChange: Bool = false) throws -> JarvisPluginCatalogSnapshot {
            var object = base
            object["policy_revision"] = revision
            var items = object["items"] as! [[String: Any]]
            items[0]["desired_enabled"] = enabled
            items[0]["selected_default"] = preferred
            items[0]["can_change"] = canChange
            object["items"] = items
            return try JSONDecoder().decode(JarvisPluginCatalogSnapshot.self, from: JSONSerialization.data(withJSONObject: object))
        }
        @MainActor func waitFor(_ predicate: () -> Bool) async throws {
            for _ in 0..<200 {
                if predicate() { return }
                try await Task.sleep(nanoseconds: 1_000_000)
            }
            preconditionFailure("Fixture request did not start")
        }

        let fixture = PluginFixture(try snapshot()), model = JarvisPluginsViewModel(api: PluginFixture(try snapshot()))
        check(model.catalog == nil && !model.preferencesAvailable, "No unverified initial preference")
        let loaded = JarvisPluginsViewModel(api: fixture)
        await loaded.refresh()
        check(loaded.catalog?.policyRevision == 7 && !loaded.preferencesAvailable, "Confirmed read cannot enable missing mutation adapter")
        await loaded.setEnabled(false, pluginID: "video_editor")
        await loaded.setPreferred(false, pluginID: "video_editor")
        check(fixture.writes.isEmpty, "Disabled controls cannot write even through direct model calls")
        fixture.snapshot = try snapshot(revision: 8, enabled: false)
        await loaded.refresh()
        let card = loaded.catalog!.presentations(at: loaded.catalog!.evaluatedAt)[0]
        check(card.confirmedEnabled == false && card.isSelectedDefault && card.defaultLabel == "Preferred when enabled",
              "Disable retains the remembered preferred tool")
        fixture.snapshot = try snapshot(revision: 7)
        await loaded.refresh()
        check(loaded.catalog?.policyRevision == 8, "Older policy cannot replace newer choices")
        fixture.snapshot = try snapshot(revision: 8, enabled: true)
        await loaded.refresh()
        check(loaded.catalog?.items[0].desiredEnabled == false && loaded.errorMessage != nil,
              "Contradictory same-revision policy is rejected")

        let held = PluginFixture(try snapshot()); held.holds = true
        let ordered = JarvisPluginsViewModel(api: held)
        let first = Task { await ordered.refresh() }
        try await waitFor { held.waits.count == 1 }
        let second = Task { await ordered.refresh() }
        try await waitFor { held.waits.count == 2 }
        held.waits[1].resume(returning: try snapshot(revision: 9, enabled: false))
        await second.value
        held.waits[0].resume(returning: try snapshot())
        await first.value
        check(ordered.catalog?.policyRevision == 9 && !ordered.isLoading, "Out-of-order reads preserve latest response")

        let retiring = PluginFixture(try snapshot()); retiring.holds = true
        let oldAccount = JarvisPluginsViewModel(api: retiring)
        let late = Task { await oldAccount.refresh() }
        try await waitFor { retiring.waits.count == 1 }
        oldAccount.deactivate()
        retiring.waits[0].resume(returning: try snapshot())
        await late.value
        check(oldAccount.catalog == nil && !oldAccount.isLoading, "Late retired-account reply cannot reveal catalog")
        let newAccount = JarvisPluginsViewModel(api: PluginFixture(try snapshot(revision: 0, enabled: false)))
        await newAccount.refresh()
        check(newAccount.catalog?.policyRevision == 0 && oldAccount.catalog == nil, "Replacement account has independent revision and cache")

        let baselineAPI = PluginFixture(try snapshot(revision: 0, canChange: true))
        baselineAPI.holdsWrites = true
        let baselineAccount = JarvisPluginsViewModel(api: baselineAPI)
        await baselineAccount.refresh()
        check(baselineAccount.preferencesAvailable, "Explicit server permission supports the known deployment baseline")
        let firstEdit = Task { await baselineAccount.setEnabled(false, pluginID: "video_editor") }
        try await waitFor { baselineAPI.writeWait != nil }
        let firstChange = baselineAPI.writes[0]
        check(firstChange.expectedRevision == 0 && baselineAccount.catalog?.items[0].desiredEnabled == true,
              "First account edit uses revision zero while retaining the confirmed default")
        baselineAPI.writeWait?.resume(returning: .init(mutationID: firstChange.mutationID, appliedRevision: 1,
            requestReplayed: false, catalog: try snapshot(revision: 1, enabled: false, canChange: true)))
        baselineAPI.writeWait = nil
        await firstEdit.value
        check(baselineAccount.catalog?.policyRevision == 1 && baselineAccount.pendingChange == nil
              && baselineAccount.catalog?.items[0].desiredEnabled == false
              && baselineAccount.catalog?.items[0].selectedDefault == true,
              "Exact first receipt moves to account policy and retains the disabled preferred plugin")

        fixture.error = JarvisAPIError(code: "denied", message: "Sign in", statusCode: 403)
        var locked = 0
        loaded.onAuthorizationFailure = { locked += 1 }
        await loaded.refresh()
        check(loaded.isLocked && loaded.catalog == nil && locked == 1, "Authorization failure clears catalog and notifies owner")
        await loaded.refresh()
        check(locked == 1 && loaded.catalog == nil, "Locked model cannot silently unlock by refresh")

        // Synthetic server permission exercises the client only. Production
        // still emits can_change=false until Core enforcement is qualified.
        let editingAPI = PluginFixture(try snapshot(canChange: true))
        editingAPI.holdsWrites = true
        let editing = JarvisPluginsViewModel(api: editingAPI)
        await editing.refresh()
        check(editing.preferencesAvailable, "Explicit server permission enables known choices")
        let savingDisable = Task { await editing.setEnabled(false, pluginID: "video_editor") }
        try await waitFor { editingAPI.writeWait != nil }
        let disabling = editingAPI.writes[0]
        check(editing.isSaving && editing.pendingChange == disabling && editing.catalog?.items[0].desiredEnabled == true,
              "In-flight save retains confirmed choice and exact pending request")
        await editing.setPreferred(false, pluginID: "video_editor")
        check(editingAPI.writes.count == 1, "Another choice cannot overwrite an unsettled mutation")
        editingAPI.writeWait?.resume(returning: .init(mutationID: disabling.mutationID, appliedRevision: 8,
            requestReplayed: false, catalog: try snapshot(revision: 8, enabled: false, canChange: true)))
        editingAPI.writeWait = nil
        await savingDisable.value
        check(!editing.isSaving && editing.pendingChange == nil && editing.catalog?.items[0].desiredEnabled == false
                && editing.catalog?.items[0].selectedDefault == true,
              "Only matching receipt confirms disable while remembering the default")

        let savingEnable = Task { await editing.setEnabled(true, pluginID: "video_editor") }
        try await waitFor { editingAPI.writeWait != nil }
        let enabling = editingAPI.writes[1]
        editingAPI.writeWait?.resume(throwing: URLError(.networkConnectionLost)); editingAPI.writeWait = nil
        await savingEnable.value
        check(editing.pendingChange == enabling && !editing.isSaving && editing.catalog?.items[0].desiredEnabled == false,
              "Lost response retains exact intent without claiming success")
        editingAPI.snapshot = try snapshot(revision: 9, canChange: true)
        await editing.refresh()
        check(editing.pendingChange == enabling && editing.catalog?.policyRevision == 9,
              "A later matching catalog alone cannot identify an uncertain mutation")
        await editing.setPreferred(false, pluginID: "video_editor")
        check(editingAPI.writes.count == 2, "Unknown mutation cannot silently become a fresh write")
        let rechecking = Task { await editing.recheckPendingChange() }
        try await waitFor { editingAPI.writeWait != nil }
        check(editingAPI.writes[2] == enabling, "Recheck uses identical UUID, expected revision and intent")
        editingAPI.writeWait?.resume(returning: .init(mutationID: enabling.mutationID, appliedRevision: 9,
            requestReplayed: true, catalog: try snapshot(revision: 9, canChange: true)))
        editingAPI.writeWait = nil
        await rechecking.value
        check(editing.pendingChange == nil && editing.statusMessage != nil, "Original replay receipt settles the uncertain change")

        let wrongResult = Task { await editing.setPreferred(false, pluginID: "video_editor") }
        try await waitFor { editingAPI.writeWait != nil }
        let defaultChange = editingAPI.writes[3]
        editingAPI.writeWait?.resume(returning: .init(mutationID: defaultChange.mutationID, appliedRevision: 10,
            requestReplayed: false, catalog: try snapshot(revision: 10, preferred: true, canChange: true)))
        editingAPI.writeWait = nil
        await wrongResult.value
        check(editing.pendingChange == defaultChange && editing.catalog?.policyRevision == 9 && editing.statusMessage == nil,
              "Same-revision receipt with the wrong requested default cannot claim a save")
        let exactDefault = Task { await editing.recheckPendingChange() }
        try await waitFor { editingAPI.writeWait != nil }
        editingAPI.writeWait?.resume(returning: .init(mutationID: defaultChange.mutationID, appliedRevision: 10,
            requestReplayed: true, catalog: try snapshot(revision: 10, preferred: false, canChange: true)))
        editingAPI.writeWait = nil
        await exactDefault.value
        check(editing.pendingChange == nil && editing.catalog?.items[0].selectedDefault == false,
              "Exact default receipt clears the remembered selection")

        editingAPI.holdsWrites = false
        editingAPI.writeError = JarvisAPIError(code: "plugin_policy_revision_conflict", message: "Changed elsewhere", statusCode: 409)
        editingAPI.snapshot = try snapshot(revision: 11, enabled: false, preferred: false, canChange: true)
        await editing.setEnabled(false, pluginID: "video_editor")
        check(editingAPI.writes.count == 6 && editing.pendingChange == nil && editing.catalog?.policyRevision == 11,
              "CAS conflict refreshes once and does not automatically retry the intent")
        editingAPI.writeError = JarvisAPIError(code: "denied", message: "Sign in", statusCode: 401)
        var writeAuthFailures = 0
        editing.onAuthorizationFailure = { writeAuthFailures += 1 }
        await editing.setEnabled(true, pluginID: "video_editor")
        check(editing.isLocked && editing.catalog == nil && editing.pendingChange == nil && writeAuthFailures == 1,
              "Write authorization failure locks all local plugin state")

        let retiringWriteAPI = PluginFixture(try snapshot(canChange: true)); retiringWriteAPI.holdsWrites = true
        let retiringWrite = JarvisPluginsViewModel(api: retiringWriteAPI)
        await retiringWrite.refresh()
        let oldSave = Task { await retiringWrite.setEnabled(false, pluginID: "video_editor") }
        try await waitFor { retiringWriteAPI.writeWait != nil }
        let oldChange = retiringWriteAPI.writes[0]
        retiringWrite.deactivate()
        retiringWriteAPI.writeWait?.resume(returning: .init(mutationID: oldChange.mutationID, appliedRevision: 8,
            requestReplayed: false, catalog: try snapshot(revision: 8, enabled: false, canChange: true)))
        retiringWriteAPI.writeWait = nil
        await oldSave.value
        check(retiringWrite.catalog == nil && retiringWrite.pendingChange == nil && retiringWrite.statusMessage == nil,
              "Late save from a retired account cannot display success or a previous account's catalog")

        let pausedAPI = PluginFixture(try snapshot(canChange: true))
        pausedAPI.writeError = URLError(.networkConnectionLost)
        let paused = JarvisPluginsViewModel(api: pausedAPI)
        await paused.refresh()
        await paused.setEnabled(false, pluginID: "video_editor")
        let unsettled = pausedAPI.writes[0]
        pausedAPI.writeError = JarvisAPIError(code: "plugin_policy_changes_unavailable", message: "Changes paused", statusCode: 503)
        await paused.recheckPendingChange()
        check(paused.pendingChange == unsettled && pausedAPI.writes[1] == unsettled && paused.statusMessage == nil,
              "A later disabled server gate cannot settle the original unknown write")
        let freshRejection = JarvisPluginsViewModel(api: pausedAPI)
        await freshRejection.refresh()
        await freshRejection.setEnabled(false, pluginID: "video_editor")
        check(freshRejection.pendingChange == nil && freshRejection.catalog?.items[0].desiredEnabled == true,
              "A first attempt explicitly rejected before admission leaves the confirmed choice unchanged")

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [PluginProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel(); PluginProtocol.response = nil }
        let api = JarvisAPIClient(configuration: JarvisConfiguration(backendBaseURL: URL(string: "https://plugin-fixture.invalid")!,
            staticBearerToken: "fixture-only"), session: session)
        PluginProtocol.response = { request in
            check(request.url?.host == "plugin-fixture.invalid" && request.url?.path == "/plugins" && request.httpMethod == "GET",
                  "Catalog uses the real authenticated account route")
            check(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-only" && request.url?.query == nil,
                  "No caller owner or voice session on read")
            return (200, data)
        }
        check(try await api.fetchPluginCatalog().policyRevision == 7, "Actual client decodes catalog")
        let change = JarvisPluginPolicyChange(mutationID: UUID(), expectedRevision: 7,
            enabledChanges: [], defaultChanges: [.init(category: "video_editing", pluginID: nil)])
        var attempt = 0
        PluginProtocol.response = { request in
            attempt += 1
            check(request.url?.path == "/plugins/policy" && request.httpMethod == "PUT" && request.url?.query == nil,
                  "Mutation is account-level PUT")
            var body = request.httpBody ?? Data()
            if body.isEmpty, let stream = request.httpBodyStream {
                stream.open(); defer { stream.close() }
                var bytes = [UInt8](repeating: 0, count: 2048)
                while stream.hasBytesAvailable {
                    let n = stream.read(&bytes, maxLength: bytes.count)
                    if n <= 0 { break }; body.append(contentsOf: bytes.prefix(n))
                }
            }
            let payload = try JSONSerialization.jsonObject(with: body) as! [String: Any]
            check(Set(payload.keys) == ["mutation_id", "expected_revision", "enabled_changes", "default_changes"],
                  "No client-supplied owner/grant/credentials in intent")
            check(UUID(uuidString: payload["mutation_id"] as! String) == change.mutationID && payload["expected_revision"] as? Int == 7,
                  "Exact mutation identity and CAS revision retained on replay")
            let defaults = payload["default_changes"] as! [[String: Any]]
            check(defaults[0]["plugin_id"] is NSNull, "Clearing a default sends explicit null")
            var current = base; current["policy_revision"] = 10
            return (200, try JSONSerialization.data(withJSONObject: ["mutation_id": change.mutationID.uuidString,
                "applied_revision": 8, "request_replayed": attempt > 1, "catalog": current]))
        }
        let applied = try await api.changePluginPolicy(change)
        let replayed = try await api.changePluginPolicy(change)
        check(applied.appliedRevision == 8 && !applied.requestReplayed && replayed.requestReplayed && replayed.catalog.policyRevision == 10,
              "Replay retains original applied revision and later current catalog")
        PluginProtocol.response = { _ in
            (200, try JSONSerialization.data(withJSONObject: ["mutation_id": UUID().uuidString,
                "applied_revision": 8, "request_replayed": false, "catalog": base]))
        }
        do { _ = try await api.changePluginPolicy(change); preconditionFailure("Mismatched receipt accepted") }
        catch let error as JarvisAPIError { check(error.code == "plugin_policy_receipt_mismatch", "Wrong mutation receipt rejected") }
        PluginProtocol.response = { _ in
            var current = base; current["policy_revision"] = 8
            return (200, try JSONSerialization.data(withJSONObject: ["mutation_id": change.mutationID.uuidString,
                "applied_revision": 8, "request_replayed": false, "catalog": current]))
        }
        do { _ = try await api.changePluginPolicy(change); preconditionFailure("Wrong result accepted") }
        catch let error as JarvisAPIError {
            check(error.code == "plugin_policy_receipt_mismatch", "Actual transport rejects exact-head receipt with the wrong choice")
        }
        print("Phone Plugins: \(count) checks passed")
    }
}
