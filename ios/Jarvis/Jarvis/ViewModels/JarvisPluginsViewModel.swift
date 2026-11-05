import Combine
import Foundation

/// One immutable authenticated client per account lifetime; no voice session or
/// global cache. An uncertain PUT keeps its exact intent/UUID for reconciliation.
@MainActor
final class JarvisPluginsViewModel: ObservableObject {
    @Published private(set) var marketplace: JarvisExtensionMarketplace?
    @Published private(set) var catalog: JarvisPluginCatalogSnapshot?
    @Published private(set) var isLoading = false
    @Published private(set) var isSaving = false
    @Published private(set) var isLocked = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var statusMessage: String?
    @Published private(set) var pendingChange: JarvisPluginPolicyChange?
    var onAuthorizationFailure: (() -> Void)?

    private let api: any JarvisPluginsAccess
    private var active = true
    private var lifetime = UUID()
    private var readGeneration = 0
    private var maintenancePollTask: Task<Void, Never>?

    init(api: any JarvisPluginsAccess) { self.api = api }

    var preferencesAvailable: Bool {
        active && !isLocked && catalog?.policyAvailable == true && catalog?.items.contains(where: \.canChange) == true
    }

    func refresh(clearError: Bool = true) async {
        guard active, !isLocked, !Task.isCancelled else { return }
        let token = lifetime
        readGeneration += 1
        let generation = readGeneration
        isLoading = true
        if clearError { errorMessage = nil }
        defer { if lifetime == token && readGeneration == generation { isLoading = false } }
        do {
            if let ext = try await api.fetchExtensionMarketplace() {
                guard active, !isLocked, !Task.isCancelled, lifetime == token, readGeneration == generation else { return }
                marketplace = ext
                syncMaintenancePolling()
                return
            }
            let result = try await api.fetchPluginCatalog()
            guard active, !isLocked, !Task.isCancelled, lifetime == token, readGeneration == generation else { return }
            try accept(result)
        } catch {
            guard active, !isLocked, !Task.isCancelled, lifetime == token, readGeneration == generation else { return }
            handle(error)
        }
    }

    func installExtension(id: String) async {
        guard active, !isLocked, !isSaving, !Task.isCancelled else { return }
        let token = lifetime
        isSaving = true
        errorMessage = nil
        statusMessage = nil
        defer { if lifetime == token { isSaving = false } }
        do {
            let updated = try await api.installPluginExtension(id: id)
            guard active, !isLocked, lifetime == token else { return }
            marketplace = updated
            syncMaintenancePolling()
        } catch {
            guard active, !isLocked, lifetime == token else { return }
            handle(error)
        }
    }

    func removeExtension(id: String) async {
        guard active, !isLocked, !isSaving, !Task.isCancelled else { return }
        let token = lifetime
        isSaving = true
        errorMessage = nil
        statusMessage = nil
        defer { if lifetime == token { isSaving = false } }
        do {
            let updated = try await api.removePluginExtension(id: id)
            guard active, !isLocked, lifetime == token else { return }
            marketplace = updated
            syncMaintenancePolling()
        } catch {
            guard active, !isLocked, lifetime == token else { return }
            handle(error)
        }
    }

    private func syncMaintenancePolling() {
        guard active, !isLocked, marketplace?.sandboxMaintenance.active == true else {
            maintenancePollTask?.cancel()
            maintenancePollTask = nil
            return
        }
        guard maintenancePollTask == nil else { return }
        let token = lifetime
        maintenancePollTask = Task { [weak self] in
            defer {
                if let self, self.lifetime == token {
                    self.maintenancePollTask = nil
                }
            }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_200_000_000)
                guard let self, self.active, !self.isLocked, self.lifetime == token, !Task.isCancelled else { return }
                do {
                    guard let latest = try await self.api.fetchExtensionMarketplace() else { return }
                    guard self.active, !self.isLocked, self.lifetime == token else { return }
                    let wasActive = self.marketplace?.sandboxMaintenance.active == true
                    self.marketplace = latest
                    if !latest.sandboxMaintenance.active {
                        if wasActive {
                            let failed = (latest.available + latest.installed).contains { $0.operation?.phase == "failed" }
                            if !failed {
                                self.statusMessage = "Sandbox updated and resumed."
                            }
                        }
                        return
                    }
                } catch {
                    return
                }
            }
        }
    }

    func setEnabled(_ enabled: Bool, pluginID: String) async {
        guard let item = changeableItem(pluginID), let revision = catalog?.policyRevision,
              item.desiredEnabled != enabled else { return }
        await submit(.init(mutationID: UUID(), expectedRevision: revision,
                           enabledChanges: [.init(pluginID: item.id, enabled: enabled)], defaultChanges: []))
    }

    func setPreferred(_ selected: Bool, pluginID: String) async {
        guard let item = changeableItem(pluginID), let revision = catalog?.policyRevision,
              item.selectedDefault != selected else { return }
        await submit(.init(mutationID: UUID(), expectedRevision: revision, enabledChanges: [],
                           defaultChanges: [.init(category: item.category, pluginID: selected ? item.id : nil)]))
    }

    /// An explicit user action replays the original request; it never creates a
    /// fresh operation or silently rebases its expected revision.
    func recheckPendingChange() async {
        guard let pendingChange, !isSaving else { return }
        await submit(pendingChange)
    }

    private func changeableItem(_ rawID: String) -> JarvisPluginCatalogSnapshot.Item? {
        guard preferencesAvailable, !isSaving, pendingChange == nil,
              let revision = catalog?.policyRevision, revision < Int.max else { return nil }
        return catalog?.items.first { $0.id.rawValue == rawID && $0.canChange && $0.desiredEnabled != nil && $0.operation == nil }
    }

    private func submit(_ change: JarvisPluginPolicyChange) async {
        guard active, !isLocked, !isSaving, !Task.isCancelled else { return }
        let isRecheck = pendingChange == change
        let token = lifetime
        readGeneration += 1; isLoading = false
        pendingChange = change; isSaving = true; errorMessage = nil; statusMessage = nil
        defer { if lifetime == token { isSaving = false } }
        do {
            let receipt = try await api.changePluginPolicy(change)
            guard active, !isLocked, lifetime == token else { return }
            guard receipt.confirms(change) else {
                throw JarvisAPIError(code: "plugin_policy_receipt_mismatch", message: "The plugin change could not be confirmed.")
            }
            try accept(receipt.catalog)
            pendingChange = nil
            statusMessage = "Preference saved."
        } catch {
            guard active, !isLocked, lifetime == token else { return }
            if let failure = error as? JarvisAPIError,
               ["plugin_policy_revision_conflict", "plugin_policy_mutation_conflict", "plugin_policy_changes_unavailable",
                "plugin_choice_unavailable", "plugin_policy_unavailable", "plugin_policy_source_unavailable"].contains(failure.code),
               !(isRecheck && failure.code == "plugin_policy_changes_unavailable") {
                pendingChange = nil
                errorMessage = failure.message
                await refresh(clearError: false)
            } else if let failure = error as? JarvisAPIError, failure.statusCode == 400 || failure.statusCode == 422 {
                pendingChange = nil; errorMessage = failure.message
            } else {
                handle(error)
                if !isLocked { errorMessage = "This change is not confirmed. Check the same change before making another." }
            }
        }
    }

    private func accept(_ incoming: JarvisPluginCatalogSnapshot) throws {
        if let current = catalog, let oldRevision = current.policyRevision, let newRevision = incoming.policyRevision {
            if newRevision < oldRevision { return }
            if newRevision == oldRevision {
                let sameChoices = incoming.items.count == current.items.count && incoming.items.allSatisfy { item in
                    current.items.contains { $0.id == item.id && $0.desiredEnabled == item.desiredEnabled &&
                        $0.configurationID == item.configurationID && $0.selectedDefault == item.selectedDefault }
                }
                guard sameChoices else {
                    throw JarvisAPIError(code: "plugin_policy_revision_mismatch", message: "Plugin choices could not be verified. Refresh to try again.")
                }
                if incoming.evaluatedAt < current.evaluatedAt { return }
            }
        }
        catalog = incoming
    }

    private func handle(_ error: Error) {
        if let failure = error as? JarvisAPIError, failure.statusCode == 401 || failure.statusCode == 403 {
            lockForSignIn(); onAuthorizationFailure?()
        } else if let failure = error as? JarvisAPIError, failure.statusCode == 404 {
            errorMessage = "Plugin choices aren’t available right now. Try again later."
        } else { errorMessage = error.localizedDescription }
    }

    func lockForSignIn() {
        clear(); isLocked = true
        errorMessage = "Sign in again to view your plugin choices."
    }

    func deactivate() { active = false; clear() }

    private func clear() {
        maintenancePollTask?.cancel()
        maintenancePollTask = nil
        lifetime = UUID(); readGeneration += 1
        marketplace = nil; catalog = nil; pendingChange = nil; isLoading = false; isSaving = false
        errorMessage = nil; statusMessage = nil
    }
}
