import Foundation

/// Account catalog. Scope and response ordering belong to the authenticated
/// adapter; never persist this as an unscoped global cache.
struct JarvisPluginCatalogSnapshot: Decodable {
    let schemaVersion: Int
    let evaluatedAt: Date
    let policyRevision: Int?
    let policyAvailable: Bool
    let items: [Item]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", evaluatedAt = "evaluated_at"
        case policyRevision = "policy_revision", policyAvailable = "policy_available", items
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
        evaluatedAt = try values.decode(Timestamp.self, forKey: .evaluatedAt).date
        policyRevision = try values.decodeIfPresent(Int.self, forKey: .policyRevision)
        policyAvailable = try values.decode(Bool.self, forKey: .policyAvailable)
        items = try values.decode([Item].self, forKey: .items)
        guard schemaVersion == 1, items.count <= 3, Set(items.map(\.id)).count == items.count,
              policyAvailable ? policyRevision.map({ $0 >= 0 }) == true : policyRevision == nil,
              policyAvailable || items.allSatisfy({ $0.desiredEnabled == nil && !$0.selectedDefault && $0.configurationID == nil }) else {
            throw Self.invalid(decoder)
        }
    }

    /// Recompute at display time and observation deadlines. Receiving an older
    /// response must not extend its server-issued deadline. A future evaluation
    /// (including a clock mismatch) cannot establish present readiness.
    func presentations(at now: Date, pendingChange: JarvisPluginPolicyChange? = nil,
                       isSaving: Bool = false) -> [JarvisPluginCardPresentation] {
        items.map { item in
            let card = item.presentation(evaluatedAt: evaluatedAt, now: now)
            guard let pendingChange else { return card }
            let requested: JarvisPluginCardPresentation.Change?
            if let enabled = pendingChange.enabledChanges.first(where: { $0.pluginID == item.id }) {
                requested = enabled.enabled ? .enable : .disable
            } else if pendingChange.defaultChanges.contains(where: { $0.category == item.category }) {
                requested = .preference
            } else { requested = nil }
            guard let requested else { return card }
            // Keep every confirmed value. The local request state is distinct
            // from an admitted installer operation and is scoped to this card.
            return JarvisPluginCardPresentation(id: card.id, title: card.title, symbol: card.symbol,
                detail: card.detail, readinessLabel: card.readinessLabel, accountLabel: card.accountLabel,
                isSelectedDefault: card.isSelectedDefault, confirmedEnabled: card.confirmedEnabled,
                allowsPreferenceChange: false,
                operation: isSaving ? .requesting(requested)
                    : .unknown(message: "Use Check change to verify this preference.", recheckReference: nil),
                installationLabel: card.installationLabel)
        }
    }

    var observationDeadlines: [Date] {
        Array(Set(items.compactMap(\.observationsValidUntil))).sorted()
    }

    struct Item: Decodable {
        enum ID: String, Codable {
            case videoEditor = "video_editor", browser, filesData = "files_data"
            var category: String {
                switch self { case .videoEditor: return "video_editing"; case .browser: return "web_browsing"; case .filesData: return "files_data" }
            }
            var symbol: String {
                switch self { case .videoEditor: return "film"; case .browser: return "globe"; case .filesData: return "doc.on.doc" }
            }
        }
        enum Installation: String, Decodable { case installed, absent, unknown }
        enum Readiness: String, Decodable { case ready, unavailable, unknown }

        let id: ID
        let title: String
        let description: String
        let category: String
        let descriptorVersion: Int
        let selectedDefault: Bool
        let desiredEnabled: Bool?
        let configurationID: UUID?
        let installed: Installation
        let installedVersion: String?
        let readiness: Readiness
        let reasonCode: String?
        let observationsValidUntil: Date?
        let operation: Operation?
        let canChange: Bool

        enum CodingKeys: String, CodingKey {
            case id = "plugin_id", title, description, category, descriptorVersion = "descriptor_version"
            case selectedDefault = "selected_default", desiredEnabled = "desired_enabled", configurationID = "configuration_id"
            case installed, installedVersion = "installed_version", readiness, reasonCode = "reason_code"
            case observationsValidUntil = "observations_valid_until", operation, canChange = "can_change"
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            id = try values.decode(ID.self, forKey: .id)
            title = try values.decode(String.self, forKey: .title)
            description = try values.decode(String.self, forKey: .description)
            category = try values.decode(String.self, forKey: .category)
            descriptorVersion = try values.decode(Int.self, forKey: .descriptorVersion)
            selectedDefault = try values.decode(Bool.self, forKey: .selectedDefault)
            desiredEnabled = try values.decodeIfPresent(Bool.self, forKey: .desiredEnabled)
            configurationID = try values.decodeIfPresent(UUID.self, forKey: .configurationID)
            installed = try values.decode(Installation.self, forKey: .installed)
            installedVersion = try values.decodeIfPresent(String.self, forKey: .installedVersion)
            readiness = try values.decode(Readiness.self, forKey: .readiness)
            reasonCode = try values.decodeIfPresent(String.self, forKey: .reasonCode)
            observationsValidUntil = try values.decodeIfPresent(Timestamp.self, forKey: .observationsValidUntil)?.date
            operation = try values.decodeIfPresent(Operation.self, forKey: .operation)
            canChange = try values.decode(Bool.self, forKey: .canChange)
            guard descriptorVersion == 1, category == id.category,
                  (desiredEnabled == nil) == (configurationID == nil),
                  !selectedDefault || desiredEnabled != nil,
                  operation == nil || operation?.configurationID == configurationID,
                  !canChange || desiredEnabled != nil else { throw JarvisPluginCatalogSnapshot.invalid(decoder) }
        }

        fileprivate func presentation(evaluatedAt: Date, now: Date) -> JarvisPluginCardPresentation {
            let fresh = configurationID != nil && now >= evaluatedAt && observationsValidUntil.map { now < $0 } == true
            let activeOperation: JarvisPluginCardPresentation.Operation?
            if let operation {
                activeOperation = fresh ? operation.presentation : .unknown(message: "The change status needs to be refreshed.", recheckReference: nil)
            } else { activeOperation = nil }

            let label: String
            if desiredEnabled == false { label = "Disabled" }
            else if desiredEnabled == nil || !fresh { label = "Availability unknown" }
            else if let activeOperation {
                switch activeOperation {
                case .running, .requesting: label = activeOperation.label
                case .failed: label = "Unavailable"
                case .unknown: label = "Availability unknown"
                }
            } else {
                switch readiness {
                case .ready: label = installed == .installed && installedVersion?.isEmpty == false ? "Ready" : "Availability unknown"
                case .unavailable: label = reasonCode == "plugin_not_installed" ? "Not installed" : "Unavailable"
                case .unknown: label = "Availability unknown"
                }
            }
            let installation: String
            if !fresh { installation = "Installation status unknown" }
            else {
                switch installed {
                case .installed: installation = installedVersion.map { "Installed version: " + $0 } ?? "Installation status unknown"
                case .absent: installation = "Not installed"
                case .unknown: installation = "Installation status unknown"
                }
            }
            return JarvisPluginCardPresentation(id: id.rawValue, title: title, symbol: id.symbol, detail: description,
                readinessLabel: label, accountLabel: nil, isSelectedDefault: selectedDefault,
                confirmedEnabled: desiredEnabled, allowsPreferenceChange: canChange, operation: activeOperation,
                installationLabel: installation)
        }
    }

    struct Operation: Decodable {
        enum Phase: String, Decodable { case preparing, installing, removing, failed, unknown }
        struct Recovery: Decodable {
            enum Kind: String, Decodable { case retry, recheck }
            let kind: Kind
            let reference: String
        }
        let operationReference: String
        let configurationID: UUID
        let phase: Phase
        let recovery: Recovery?
        enum CodingKeys: String, CodingKey {
            case operationReference = "operation_ref", configurationID = "configuration_id", phase, recovery
        }
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            operationReference = try values.decode(String.self, forKey: .operationReference)
            configurationID = try values.decode(UUID.self, forKey: .configurationID)
            phase = try values.decode(Phase.self, forKey: .phase)
            recovery = try values.decodeIfPresent(Recovery.self, forKey: .recovery)
            let recoveryAllowed = phase == .failed || (phase == .unknown && recovery?.kind == .recheck)
            guard Self.validReference(operationReference),
                  recovery == nil || (recoveryAllowed && Self.validReference(recovery!.reference)) else {
                throw JarvisPluginCatalogSnapshot.invalid(decoder)
            }
        }
        private static func validReference(_ value: String) -> Bool {
            value.count <= 200 && value.range(of: "^[A-Za-z0-9][A-Za-z0-9_.:-]*$", options: .regularExpression) != nil
        }
        fileprivate var presentation: JarvisPluginCardPresentation.Operation {
            switch phase {
            case .preparing: return .running(.preparing, reference: operationReference)
            case .installing: return .running(.installing, reference: operationReference)
            case .removing: return .running(.removing, reference: operationReference)
            case .unknown: return .unknown(message: "The change could not be confirmed.", recheckReference: recovery?.reference)
            case .failed:
                let action: JarvisPluginCardPresentation.Recovery? = recovery.map {
                    $0.kind == .retry ? .retry($0.reference) : .recheck($0.reference)
                }
                return .failed(message: "The change could not be completed.", recovery: action)
            }
        }
    }

    private struct Timestamp: Decodable {
        let date: Date
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let parsed = formatter.date(from: raw) { date = parsed; return }
            formatter.formatOptions = [.withInternetDateTime]
            guard let parsed = formatter.date(from: raw) else { throw JarvisPluginCatalogSnapshot.invalid(decoder) }
            date = parsed
        }
    }
    private static func invalid(_ decoder: Decoder) -> DecodingError {
        .dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "The plugin catalog is incompatible or inconsistent."))
    }
}

struct JarvisPluginPolicyChange: Encodable, Equatable {
    struct EnabledChange: Encodable, Equatable {
        let pluginID: JarvisPluginCatalogSnapshot.Item.ID
        let enabled: Bool
        enum CodingKeys: String, CodingKey { case pluginID = "plugin_id", enabled }
    }
    struct DefaultChange: Encodable, Equatable {
        let category: String
        let pluginID: JarvisPluginCatalogSnapshot.Item.ID?
        enum CodingKeys: String, CodingKey { case category, pluginID = "plugin_id" }
        func encode(to encoder: Encoder) throws {
            var values = encoder.container(keyedBy: CodingKeys.self)
            try values.encode(category, forKey: .category)
            if let pluginID { try values.encode(pluginID, forKey: .pluginID) }
            else { try values.encodeNil(forKey: .pluginID) }
        }
    }
    let mutationID: UUID
    let expectedRevision: Int
    let enabledChanges: [EnabledChange]
    let defaultChanges: [DefaultChange]
    enum CodingKeys: String, CodingKey {
        case mutationID = "mutation_id", expectedRevision = "expected_revision"
        case enabledChanges = "enabled_changes", defaultChanges = "default_changes"
    }
}

struct JarvisPluginPolicyReceipt: Decodable {
    let mutationID: UUID
    let appliedRevision: Int
    let requestReplayed: Bool
    let catalog: JarvisPluginCatalogSnapshot
    enum CodingKeys: String, CodingKey {
        case mutationID = "mutation_id", appliedRevision = "applied_revision"
        case requestReplayed = "request_replayed", catalog
    }

    func confirms(_ change: JarvisPluginPolicyChange) -> Bool {
        guard change.expectedRevision >= 0, change.expectedRevision < Int.max,
              mutationID == change.mutationID, appliedRevision == change.expectedRevision + 1 else { return false }
        guard catalog.policyAvailable else { return true }
        guard let revision = catalog.policyRevision, revision >= appliedRevision else { return false }
        // A replay may return a newer policy changed by another request. When
        // the returned head is this exact revision, its choices must match.
        guard revision == appliedRevision else { return true }
        guard change.enabledChanges.allSatisfy({ choice in
            catalog.items.contains { $0.id == choice.pluginID && $0.desiredEnabled == choice.enabled }
        }) else { return false }
        return change.defaultChanges.allSatisfy { choice in
            let category = catalog.items.filter { $0.category == choice.category }
            guard !category.isEmpty else { return false }
            if let selected = choice.pluginID {
                return category.contains { $0.id == selected && $0.selectedDefault }
                    && category.allSatisfy { !$0.selectedDefault || $0.id == selected }
            }
            return category.allSatisfy { !$0.selectedDefault }
        }
    }
}

protocol JarvisPluginsAccess {
    func fetchPluginCatalog() async throws -> JarvisPluginCatalogSnapshot
    func changePluginPolicy(_ change: JarvisPluginPolicyChange) async throws -> JarvisPluginPolicyReceipt
    func fetchExtensionMarketplace() async throws -> JarvisExtensionMarketplace?
    func installPluginExtension(id: String) async throws -> JarvisExtensionMarketplace
    func removePluginExtension(id: String) async throws -> JarvisExtensionMarketplace
}

extension JarvisPluginsAccess {
    func fetchExtensionMarketplace() async throws -> JarvisExtensionMarketplace? { nil }
    func installPluginExtension(id: String) async throws -> JarvisExtensionMarketplace {
        throw JarvisAPIError(code: "plugin_extensions_unavailable", message: "Plugin installation is not available.", statusCode: 503)
    }
    func removePluginExtension(id: String) async throws -> JarvisExtensionMarketplace {
        throw JarvisAPIError(code: "plugin_extensions_unavailable", message: "Plugin removal is not available.", statusCode: 503)
    }
}

struct JarvisExtensionMarketplace: Decodable, Equatable {
    struct SandboxMaintenance: Decodable, Equatable {
        let active: Bool
        let pluginID: String?
        let action: String?
        let phase: String?
        let message: String?
        let pausedSandboxes: Int
        let closedStreams: Int

        enum CodingKeys: String, CodingKey {
            case active, pluginID = "plugin_id", action, phase, message
            case pausedSandboxes = "paused_sandboxes", closedStreams = "closed_streams"
        }
    }

    struct OperationStatus: Decodable, Equatable {
        let action: String?
        let phase: String
        let message: String?

        var isRunning: Bool { phase != "failed" }
    }

    struct ExtensionItem: Decodable, Equatable, Identifiable {
        let id: String
        let title: String
        let subtitle: String
        let description: String
        let symbol: String
        let categoryLabel: String
        let coreBuiltin: Bool
        let packages: [String]
        let installSizeHint: String
        let installed: Bool
        let version: String?
        let installedAt: String?
        let canInstall: Bool?
        let canUninstall: Bool?
        let operation: OperationStatus?

        enum CodingKeys: String, CodingKey {
            case id, title, subtitle, description, symbol
            case categoryLabel = "category_label", coreBuiltin = "core_builtin"
            case packages, installSizeHint = "install_size_hint"
            case installed, version, installedAt = "installed_at"
            case canInstall = "can_install", canUninstall = "can_uninstall", operation
        }
    }

    let schemaVersion: Int
    let evaluatedAt: String
    let sandboxMaintenance: SandboxMaintenance
    let installed: [ExtensionItem]
    let available: [ExtensionItem]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", evaluatedAt = "evaluated_at"
        case sandboxMaintenance = "sandbox_maintenance", installed, available
    }
}

struct JarvisLiveLocationUpdate: Encodable, Equatable {
    let latitude: Double
    let longitude: Double
    let horizontalAccuracyMeters: Double?
    let placeName: String?
    let locality: String?
    let country: String?
    let timezone: String?
    let capturedAt: String

    enum CodingKeys: String, CodingKey {
        case latitude, longitude
        case horizontalAccuracyMeters = "horizontal_accuracy_m"
        case placeName = "place_name", locality, country, timezone
        case capturedAt = "captured_at"
    }
}

struct JarvisLiveLocationReceipt: Decodable {
    let status: String
}
