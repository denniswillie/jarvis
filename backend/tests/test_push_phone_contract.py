"""Opt-in native Swift contracts for push registration and notification pointers."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SWIFT = r'''
import Foundation
extension ISO8601DateFormatter { static let jarvisFormatter = ISO8601DateFormatter() }

func decode<T: Decodable>(_ payload: Any, _ type: T.Type) throws -> T {
    try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: payload))
}
func body(_ request: URLRequest) -> [String: Any] {
    if let data = request.httpBody { return try! JSONSerialization.jsonObject(with: data) as! [String: Any] }
    let stream = request.httpBodyStream!
    stream.open()
    defer { stream.close() }
    var data = Data(), buffer = [UInt8](repeating: 0, count: 4096)
    while true {
        let count = stream.read(&buffer, maxLength: buffer.count)
        if count <= 0 { break }
        data.append(contentsOf: buffer.prefix(count))
    }
    return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
}
func device(_ id: UUID, revision: Int, enabled: Bool, quiet: String? = nil) -> [String: Any] {
    ["id": id.uuidString, "revision": revision, "environment": "sandbox",
     "topic": "com.symbiote.jarvis", "permission": "authorized", "enabled": enabled,
     "quiet_until": quiet as Any? ?? NSNull(), "invalid_reason": NSNull()]
}
struct JarvisKeychain {
    static var values: [String: String] = [:]
    func read(account: String) throws -> String? { Self.values[account] }
    func save(_ value: String, account: String) throws { Self.values[account] = value }
}
let ready: [String: Any] = ["state": "running", "provider_access": "unverified",
    "environment": "sandbox", "topic": "com.symbiote.jarvis"]
@MainActor final class PushStore: JarvisPushPersistence {
    var value = JarvisPushPreferences()
    var writes: [Data] = []
    func load() throws -> JarvisPushPreferences { value }
    func save(_ value: JarvisPushPreferences) throws {
        writes.append(try JSONEncoder().encode(value))
        self.value = value
    }
}
@MainActor final class Platform: JarvisPushPlatform {
    var environment: String? = "sandbox"
    var topic = "com.symbiote.jarvis"
    var presentationEnabled = false
    var grants = JarvisPushPermission.authorized
    var prompts = 0, registrations = 0, unregisters = 0
    func permission() async -> JarvisPushPermission { grants }
    func requestPermission() async throws -> JarvisPushPermission { prompts += 1; return grants }
    func register() { registrations += 1 }
    func unregister() { unregisters += 1 }
}
@MainActor final class PushAPI: JarvisPushAccess {
    var row: JarvisPushDevice?
    var calls: [JarvisPendingPushChange] = []
    var receipts: [UUID: Int] = [:]
    var loseResponse = false
    var pauseRead = false
    var read: CheckedContinuation<[JarvisPushDevice], Error>?
    var status = ready
    func getPushStatus() async throws -> JarvisPushStatus { try decode(status, JarvisPushStatus.self) }
    func getPushDevices() async throws -> [JarvisPushDevice] {
        if pauseRead { return try await withCheckedThrowingContinuation { read = $0 } }
        return row.map { [$0] } ?? []
    }
    func changePushDevice(id: UUID, change: JarvisPendingPushChange) async throws -> JarvisPushReceipt {
        calls.append(change)
        let command = change.mutation
        if receipts[command.requestID] == nil {
            guard command.expectedRevision == (row?.revision ?? 0) else {
                throw JarvisAPIError(code: "conflict", message: "Changed", statusCode: 409)
            }
            let enabled: Bool, quiet: String?
            switch change {
            case let .register(value):
                enabled = value.enabled
                quiet = value.changesQuietTime ? value.quietUntil : row?.quietUntil
            case .revoke:
                enabled = false; quiet = row?.quietUntil
            }
            row = try decode(device(id, revision: command.expectedRevision + 1, enabled: enabled, quiet: quiet),
                             JarvisPushDevice.self)
            receipts[command.requestID] = row!.revision
        }
        if loseResponse { loseResponse = false; throw URLError(.timedOut) }
        return JarvisPushReceipt(device: row!, acceptedRevision: receipts[command.requestID]!)
    }
}
final class PushProtocol: URLProtocol {
    static var handler: ((URLRequest) -> (Int, Any))!
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (status, payload) = Self.handler(request)
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status,
            httpVersion: nil, headerFields: ["Content-Type": "application/json"])!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: try! JSONSerialization.data(withJSONObject: payload))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
@MainActor func until(_ condition: () -> Bool) async {
    let end = Date().addingTimeInterval(5)
    while !condition() && Date() < end { await Task.yield() }
    precondition(condition(), "Async condition timed out")
}

@main struct PushContract {
    @MainActor static func main() async throws {
        let identity = UUID()
        let link = JarvisNotificationLink.parse(["jarvis": ["version": 1, "intent_id": identity.uuidString]])
        precondition(link?.intentID == identity)
        precondition(JarvisNotificationLink.parse(["jarvis": ["version": true, "intent_id": identity.uuidString]]) == nil)
        precondition(JarvisNotificationLink.parse(["jarvis": ["version": 2, "intent_id": identity.uuidString]]) == nil)
        precondition(JarvisNotificationLink.parse(["jarvis": ["version": 1, "intent_id": "../external"]]) == nil)
        precondition(JarvisNotificationLink.parse(["url": "https://untrusted.invalid"]) == nil)

        let oldConfiguration = JarvisConfiguration(backendBaseURL: URL(string: "https://identity.invalid")!,
                                                   staticBearerToken: "old-credential")
        let durable = JarvisPushStore(configuration: oldConfiguration)
        var savedPreferences = try durable.load()
        savedPreferences.wantsEnabled = true
        savedPreferences.serverMayBeEnabled = true
        try durable.save(savedPreferences)
        let repaired = JarvisPushStore(configuration: JarvisConfiguration(
            backendBaseURL: oldConfiguration.backendBaseURL, staticBearerToken: "new-credential"))
        let repairedPreferences = try repaired.load()
        precondition(repairedPreferences.deviceID == savedPreferences.deviceID && !repairedPreferences.wantsEnabled)
        let other = JarvisPushStore(configuration: JarvisConfiguration(
            backendBaseURL: URL(string: "https://another.invalid")!, staticBearerToken: "new-credential"))
        let otherPreferences = try other.load()
        precondition(otherPreferences.deviceID != savedPreferences.deviceID)
        precondition(!JarvisKeychain.values.keys.contains(where: { $0.contains("credential") }))

        let store = PushStore(), api = PushAPI(), platform = Platform()
        let model = JarvisPushViewModel(api: api, storage: store, platform: platform)
        await model.refresh()
        precondition(platform.prompts == 0 && platform.registrations == 0 && api.calls.isEmpty)
        await model.enable()
        precondition(platform.prompts == 1 && platform.registrations == 1 && api.calls.isEmpty)
        api.loseResponse = true
        model.receivedToken("ab" + String(repeating: "12", count: 64))
        await until { !model.isBusy && model.pending != nil }
        let original = model.pending!
        precondition(api.row?.enabled == true && model.preferences.reconciliationNeeded)
        await model.refresh()
        precondition(api.calls.count == 1)
        await model.retry()
        precondition(api.calls.count == 2 && api.calls[0] == api.calls[1])
        precondition(model.pending == nil && model.device?.enabled == true)
        let saved = String(decoding: try JSONEncoder().encode(store.value), as: UTF8.self)
        precondition(!saved.contains("token") && !saved.contains("ab1212"))
        if case let .register(value) = original {
            let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as! [String: Any]
            precondition(encoded["quiet_until"] == nil)
        } else { preconditionFailure("Expected registration") }

        await model.setQuiet(until: Date(timeIntervalSince1970: 1893456000))
        precondition(model.device?.quietUntil != nil)
        model.receivedToken("cd" + String(repeating: "34", count: 64))
        await until { !model.isBusy && api.calls.count == 4 }
        precondition(model.device?.quietUntil != nil)
        await model.setQuiet(until: nil)
        precondition(model.device?.quietUntil == nil)
        if case let .register(value) = api.calls.last! {
            let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as! [String: Any]
            precondition(encoded["quiet_until"] is NSNull)
        }
        await model.disable()
        precondition(!model.preferences.wantsEnabled && model.device?.enabled == false)
        precondition(!model.requiresDisconnectCleanup && !platform.presentationEnabled)

        platform.grants = .denied
        let before = api.calls.count
        await model.enable()
        precondition(api.calls.count == before && !platform.presentationEnabled)
        await model.disable()
        platform.grants = .authorized
        await model.enable()
        api.loseResponse = true
        model.receivedToken("ef" + String(repeating: "56", count: 64))
        await until { model.pending != nil && !model.isBusy }
        precondition(store.value.reconciliationNeeded)
        model.deactivate()
        let restartedPlatform = Platform()
        let restarted = JarvisPushViewModel(api: api, storage: store, platform: restartedPlatform)
        await restarted.refresh()
        precondition(restarted.pending == nil && restartedPlatform.registrations == 1)
        precondition(!restarted.preferences.reconciliationNeeded)
        let acceptedRevision = api.row!.revision
        restarted.receivedToken("aa" + String(repeating: "78", count: 64))
        await until { !restarted.isBusy && api.row!.revision == acceptedRevision + 1 }
        precondition(api.calls.last!.mutation.expectedRevision == acceptedRevision)

        api.pauseRead = true
        let reading = Task { await restarted.refresh() }
        await until { api.read != nil }
        restarted.deactivate()
        api.read!.resume(returning: [api.row!])
        await reading.value
        precondition(restarted.device == nil && restarted.provider == nil)

        let disablingStore = PushStore(), disablingAPI = PushAPI(), disablingPlatform = Platform()
        let disabling = JarvisPushViewModel(api: disablingAPI, storage: disablingStore, platform: disablingPlatform)
        await disabling.enable()
        disablingAPI.loseResponse = true
        disabling.receivedToken("bb" + String(repeating: "90", count: 64))
        await until { disabling.pending != nil && !disabling.isBusy }
        let unresolved = disabling.pending
        await disabling.disable()
        precondition(disabling.pending == unresolved && !disabling.preferences.wantsEnabled)
        await disabling.retry()
        precondition(disablingAPI.row?.enabled == false && disabling.pending == nil)
        precondition(disablingAPI.calls.count == 3)
        precondition(!disabling.requiresDisconnectCleanup)

        await disabling.enable()
        disabling.receivedToken("cc" + String(repeating: "91", count: 64))
        await until { disabling.device?.enabled == true && !disabling.isBusy }
        disabling.prepareForLocalReset()
        precondition(!disabling.preferences.wantsEnabled && disabling.preferences.reconciliationNeeded)
        precondition(disabling.preferences.serverMayBeEnabled && !disablingPlatform.presentationEnabled)
        disabling.deactivate()

        let wrongPlatform = Platform()
        wrongPlatform.environment = "production"
        let wrongStore = PushStore()
        wrongStore.value.wantsEnabled = true
        let mismatch = JarvisPushViewModel(api: api, storage: wrongStore, platform: wrongPlatform)
        api.pauseRead = false
        await mismatch.refresh()
        precondition(wrongPlatform.registrations == 0 && mismatch.message?.contains("environment") == true)

        let urlConfig = URLSessionConfiguration.ephemeral
        urlConfig.protocolClasses = [PushProtocol.self]
        let session = URLSession(configuration: urlConfig)
        defer { session.invalidateAndCancel() }
        let client = JarvisAPIClient(configuration: JarvisConfiguration(
            backendBaseURL: URL(string: "https://push.invalid/api")!, staticBearerToken: "fixture"), session: session)
        var wrongID = false
        PushProtocol.handler = { request in
            precondition(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture")
            precondition(request.cachePolicy == .reloadIgnoringLocalCacheData)
            if request.httpMethod == "GET" {
                return request.url!.path.hasSuffix("push-status") ? (200, ready) :
                    (200, [device(identity, revision: 1, enabled: true)])
            }
            precondition(request.httpMethod == "POST" && request.url!.path.hasSuffix("/revoke"))
            let values = body(request)
            precondition(values["expected_revision"] as? Int == 1)
            precondition(values["token"] == nil)
            return (200, ["device": device(wrongID ? UUID() : identity, revision: 2, enabled: false),
                          "accepted_revision": 2])
        }
        _ = try await client.getPushStatus()
        let fetchedDevices = try await client.getPushDevices()
        precondition(fetchedDevices.count == 1)
        let revocation = JarvisPendingPushChange.revoke(JarvisDeviceMutation(requestID: UUID(), expectedRevision: 1))
        _ = try await client.changePushDevice(id: identity, change: revocation)
        wrongID = true
        do { _ = try await client.changePushDevice(id: identity, change: revocation); preconditionFailure("Wrong receipt") }
        catch { precondition((error as? JarvisAPIError)?.code == "push_identity_mismatch") }
        print("Push phone contract passed")
    }
}
'''


@unittest.skipUnless(os.getenv("JARVIS_IOS_CONTRACT_TESTS") == "1" and shutil.which("swiftc"),
                     "Opt-in native Swift compiler contract")
class PushPhoneContractTests(unittest.TestCase):
    def test_actual_push_models_client_and_lifecycle(self):
        ios = Path(__file__).resolve().parents[2] / "ios/Jarvis/Jarvis"
        sources = [ios / "Models" / name for name in (
            "JarvisModels.swift", "JarvisMemoryModels.swift", "JarvisFileModels.swift",
            "JarvisWorkModels.swift", "JarvisReminderModels.swift", "JarvisCalendarModels.swift",
            "JarvisPushModels.swift")]
        sources += [ios / "Services/JarvisAPIClient.swift", ios / "Services/JarvisPushAPIClient.swift",
                    ios / "ViewModels/JarvisPushViewModel.swift", ios / "Services/JarvisPushStore.swift"]
        with tempfile.TemporaryDirectory(prefix="jarvis-push-phone-swift-") as directory:
            root = Path(directory)
            (root / "Contract.swift").write_text(SWIFT)
            compiled = subprocess.run(["swiftc", *map(str, sources), str(root / "Contract.swift"),
                                       "-o", str(root / "check")], capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            ran = subprocess.run([str(root / "check")], capture_output=True, text=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("Push phone contract passed", ran.stdout)
