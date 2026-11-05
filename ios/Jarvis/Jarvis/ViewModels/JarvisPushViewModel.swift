import Combine
import Foundation

@MainActor
final class JarvisPushViewModel: ObservableObject {
    @Published private(set) var preferences = JarvisPushPreferences()
    @Published private(set) var permission = JarvisPushPermission.notDetermined
    @Published private(set) var device: JarvisPushDevice?
    @Published private(set) var provider: JarvisPushStatus?
    @Published private(set) var pending: JarvisPendingPushChange?
    @Published private(set) var isBusy = false
    @Published private(set) var message: String?

    private let api: any JarvisPushAccess
    private let storage: any JarvisPushPersistence
    private let platform: any JarvisPushPlatform
    private var active = true
    private var storageAvailable = false
    private var token: String?
    private var acknowledgedToken: String?
    private var refreshRequested = false

    init(api: any JarvisPushAccess, storage: any JarvisPushPersistence, platform: any JarvisPushPlatform) {
        self.api = api; self.storage = storage; self.platform = platform
        do {
            preferences = try storage.load()
            storageAvailable = true
        } catch { message = "Secure notification settings could not be loaded. Notifications remain off." }
        platform.presentationEnabled = storageAvailable && preferences.wantsEnabled
    }

    var requiresDisconnectCleanup: Bool {
        preferences.wantsEnabled || preferences.serverMayBeEnabled || pending != nil || isBusy
    }
    var canSetQuietTime: Bool {
        active && !isBusy && pending == nil && preferences.wantsEnabled &&
            permission.allowsNotifications && device?.enabled == true && token != nil
    }

    func refresh() async {
        guard active, storageAvailable else { return }
        guard !isBusy else { refreshRequested = true; return }
        isBusy = true
        defer { finishOperation() }
        do {
            permission = await platform.permission()
            guard active else { return }
            provider = try await api.getPushStatus()
            guard active else { return }
            let devices = try await api.getPushDevices()
            guard active else { return }
            device = devices.first { $0.id == preferences.deviceID }
            if pending != nil {
                message = "A registration change is unconfirmed. Retry that exact request before making another change."
                return
            }
            if preferences.reconciliationNeeded {
                var next = preferences
                next.reconciliationNeeded = false
                next.serverMayBeEnabled = device?.enabled == true
                try persist(next)
                message = "Current server registration refreshed after an unconfirmed change. Check any quiet-time preference below."
            }
            platform.presentationEnabled = preferences.wantsEnabled && permission.allowsNotifications
            if preferences.wantsEnabled && permission.allowsNotifications {
                guard let environment = platform.environment,
                      provider?.environment == environment, provider?.topic == platform.topic,
                      provider?.state != "unavailable" else {
                    message = "Phone push is not ready for this signed app's environment and topic. The inbox remains available."
                    return
                }
                guard let token else {
                    platform.register()
                    message = "Waiting for Apple's device registration."
                    return
                }
                if acknowledgedToken != token || device?.enabled != true || device?.permission != permission {
                    await submit(registration(token: token, environment: environment))
                }
            } else {
                platform.unregister()
                token = nil; acknowledgedToken = nil
                if let device, device.enabled || preferences.serverMayBeEnabled {
                    await submit(.revoke(JarvisDeviceMutation(requestID: UUID(), expectedRevision: device.revision)))
                } else if preferences.serverMayBeEnabled {
                    var next = preferences
                    next.serverMayBeEnabled = false
                    try persist(next)
                }
            }
        } catch {
            if active { message = "Could not refresh notification registration. The inbox is still separate from push availability." }
        }
    }

    func enable() async {
        guard active, storageAvailable, !isBusy, pending == nil else { return }
        isBusy = true
        do {
            var next = preferences; next.wantsEnabled = true
            try persist(next)
            permission = try await platform.requestPermission()
            if active, !permission.allowsNotifications {
                message = "Notifications are not allowed by iOS. You can change this in iOS Settings."
            }
        } catch { if active { message = "Notification permission or secure settings could not be updated." } }
        finishOperation()
        await refresh()
    }

    func disable() async {
        guard active, storageAvailable, !isBusy else { return }
        platform.presentationEnabled = false
        platform.unregister()
        token = nil; acknowledgedToken = nil
        do {
            var next = preferences; next.wantsEnabled = false
            try persist(next)
            await refresh()
        } catch { message = "Local delivery stopped, but secure settings could not be saved. Server revocation is not confirmed." }
    }

    func receivedToken(_ value: String) {
        guard active, preferences.wantsEnabled else { return }
        token = value
        Task { await refresh() }
    }

    func registrationFailed() {
        guard active else { return }
        token = nil
        message = "Apple device registration failed. Check network access and push signing, then refresh."
    }

    func setQuiet(until: Date?) async {
        guard canSetQuietTime, let token, let environment = platform.environment else { return }
        isBusy = true
        defer { finishOperation() }
        await submit(registration(token: token, environment: environment, changesQuietTime: true,
                                  quietUntil: until.map { ISO8601DateFormatter().string(from: $0) }))
    }

    func retry() async {
        guard active, storageAvailable, !isBusy, let pending else { return }
        isBusy = true
        await submit(pending)
        let resolved = self.pending == nil
        finishOperation()
        if resolved { await refresh() }
    }

    func prepareForLocalReset() {
        platform.presentationEnabled = false
        platform.unregister()
        token = nil; acknowledgedToken = nil
        var next = preferences
        next.wantsEnabled = false
        next.reconciliationNeeded = true
        // Keep the uncertainty marker for a later authenticated reconciliation.
        // Reset itself must remain possible if Keychain is unavailable.
        try? persist(next)
    }

    func deactivate() {
        active = false
        platform.presentationEnabled = false
        token = nil; acknowledgedToken = nil
        device = nil; provider = nil; pending = nil
        refreshRequested = false
    }

    private func registration(token: String, environment: String, changesQuietTime: Bool = false,
                              quietUntil: String? = nil) -> JarvisPendingPushChange {
        .register(JarvisDeviceRegistration(
            mutation: JarvisDeviceMutation(requestID: UUID(), expectedRevision: device?.revision ?? 0),
            environment: environment, topic: platform.topic, token: token, permission: permission,
            enabled: true, changesQuietTime: changesQuietTime, quietUntil: quietUntil))
    }

    private func submit(_ change: JarvisPendingPushChange) async {
        do {
            var before = preferences
            before.reconciliationNeeded = true
            if case .register = change { before.serverMayBeEnabled = true }
            try persist(before)
            pending = change
            let receipt = try await api.changePushDevice(id: preferences.deviceID, change: change)
            guard active else { return }
            guard receipt.device.id == preferences.deviceID,
                  receipt.acceptedRevision == change.mutation.expectedRevision + 1,
                  receipt.device.revision >= receipt.acceptedRevision else {
                throw JarvisAPIError(code: "push_identity_mismatch", message: "Unexpected registration receipt.")
            }
            var next = preferences
            next.serverMayBeEnabled = receipt.device.enabled
            next.reconciliationNeeded = false
            try persist(next)
            device = receipt.device
            if case let .register(body) = change { acknowledgedToken = body.token }
            else { acknowledgedToken = nil }
            pending = nil
            message = receipt.device.enabled
                ? "Server registration accepted. Notification display is not proven."
                : "Server registration is disabled."
        } catch {
            guard active else { return }
            if let apiError = error as? JarvisAPIError, let status = apiError.statusCode,
               [400, 401, 403, 404, 409, 422].contains(status) {
                pending = nil
                message = "Registration change was rejected. Refresh before trying again."
            } else {
                message = "Registration acceptance is uncertain. Retry the same request; no automatic retry will be made."
            }
        }
    }

    private func persist(_ next: JarvisPushPreferences) throws {
        try storage.save(next)
        preferences = next
    }

    private func finishOperation() {
        isBusy = false
        if active, refreshRequested {
            refreshRequested = false
            Task { await refresh() }
        }
    }
}
