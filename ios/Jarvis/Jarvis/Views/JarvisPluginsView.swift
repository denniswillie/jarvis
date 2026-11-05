import SwiftUI

struct JarvisPluginsAccountView: View {
    @ObservedObject var model: JarvisPluginsViewModel
    var isActive: Bool

    var body: some View {
        Group {
            if let marketplace = model.marketplace {
                JarvisExtensionMarketplaceView(
                    marketplace: marketplace,
                    isLoading: model.isLoading,
                    isSaving: model.isSaving,
                    errorMessage: model.errorMessage,
                    statusMessage: model.statusMessage,
                    onInstall: { id in Task { await model.installExtension(id: id) } },
                    onRemove: { id in Task { await model.removeExtension(id: id) } },
                    onRefresh: { await model.refresh() }
                )
            } else {
                JarvisPluginsView(
                    cards: [], catalog: model.catalog, isLoading: model.isLoading,
                    catalogError: model.errorMessage, preferencesAvailable: model.preferencesAvailable,
                    statusMessage: model.statusMessage, isSaving: model.isSaving,
                    pendingChange: model.pendingChange,
                    onEnabledChange: model.pendingChange == nil ? { id, enabled in
                        Task { await model.setEnabled(enabled, pluginID: id) }
                    } : nil,
                    onPreferredChange: model.pendingChange == nil ? { id, preferred in
                        Task { await model.setPreferred(preferred, pluginID: id) }
                    } : nil,
                    onRecheckChange: { Task { await model.recheckPendingChange() } },
                    onRefresh: { await model.refresh() }
                )
            }
        }
        .task(id: isActive) { if isActive { await model.refresh() } }
    }
}

struct JarvisExtensionMarketplaceView: View {
    let marketplace: JarvisExtensionMarketplace
    var isLoading = false
    var isSaving = false
    var errorMessage: String?
    var statusMessage: String?
    var onInstall: ((String) -> Void)?
    var onRemove: ((String) -> Void)?
    var onRefresh: (() async -> Void)?
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        let outerPadding: CGFloat = dynamicTypeSize.isAccessibilitySize ? 16 : 20
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Plugins")
                            .font(JarvisTypography.pageTitle)
                            .foregroundStyle(JarvisColors.primaryText)
                        Text("Install desktop apps and CLI software into Jarvis’s Linux sandbox. Active sandboxes pause briefly during installation and resume automatically.")
                            .font(.subheadline)
                            .foregroundStyle(JarvisColors.secondaryText)
                            .lineSpacing(3)
                    }

                    if marketplace.sandboxMaintenance.active {
                        maintenanceBanner
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .font(.callout)
                            .foregroundStyle(JarvisColors.clay)
                    }

                    if let statusMessage, !marketplace.sandboxMaintenance.active {
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(JarvisColors.accent)
                            Text(statusMessage)
                                .font(.footnote.weight(.medium))
                                .foregroundStyle(JarvisColors.secondaryText)
                        }
                    }

                    sectionHeader(title: "Installed", count: marketplace.installed.count)
                    VStack(spacing: 12) {
                        ForEach(marketplace.installed) { item in
                            JarvisExtensionCardView(
                                item: item,
                                maintenanceActive: marketplace.sandboxMaintenance.active || isSaving,
                                onInstall: onInstall,
                                onRemove: onRemove
                            )
                        }
                    }

                    sectionHeader(title: "Available Extensions", count: marketplace.available.count)
                    if marketplace.available.isEmpty {
                        Text("All curated extensions are installed.")
                            .font(.subheadline)
                            .foregroundStyle(JarvisColors.secondaryText)
                    } else {
                        VStack(spacing: 12) {
                            ForEach(marketplace.available) { item in
                                JarvisExtensionCardView(
                                    item: item,
                                    maintenanceActive: marketplace.sandboxMaintenance.active || isSaving,
                                    onInstall: onInstall,
                                    onRemove: onRemove
                                )
                            }
                        }
                    }
                }
                .padding(.horizontal, outerPadding)
                .padding(.top, 12)
                .padding(.bottom, 32)
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .navigationBarTitleDisplayMode(.inline)
            .refreshable { await onRefresh?() }
            .background(JarvisColors.background)
            .tint(JarvisColors.accent)
        }
    }

    private func sectionHeader(title: String, count: Int) -> some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundStyle(JarvisColors.primaryText)
            Text("\(count)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(JarvisColors.secondaryText)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Capsule().fill(JarvisColors.panelStrong))
            Spacer()
        }
        .padding(.top, 4)
    }

    private var maintenanceBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ProgressView()
                    .tint(JarvisColors.accent)
                Text(marketplace.sandboxMaintenance.message ?? "Updating sandbox software…")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(JarvisColors.primaryText)
            }
            Text("Desktop streaming is paused while root package changes apply.")
                .font(.caption)
                .foregroundStyle(JarvisColors.secondaryText)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(JarvisColors.softAccent, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(JarvisColors.accent.opacity(0.35), lineWidth: 1)
        )
    }
}

struct JarvisExtensionCardView: View {
    let item: JarvisExtensionMarketplace.ExtensionItem
    let maintenanceActive: Bool
    var onInstall: ((String) -> Void)?
    var onRemove: ((String) -> Void)?
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 12) {
                if !dynamicTypeSize.isAccessibilitySize {
                    Image(systemName: item.symbol)
                        .font(.body.weight(.semibold))
                        .foregroundStyle(item.installed ? JarvisColors.accent : JarvisColors.primaryText)
                        .frame(width: 42, height: 42)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(item.installed ? JarvisColors.softAccent : JarvisColors.panelStrong)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                                        .strokeBorder(item.installed ? Color.clear : JarvisColors.border, lineWidth: 1)
                                )
                        )
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(item.title)
                        .font(.headline)
                        .foregroundStyle(JarvisColors.primaryText)
                    Text("\(item.categoryLabel) · \(item.subtitle)")
                        .font(.caption)
                        .foregroundStyle(JarvisColors.secondaryText)
                }

                Spacer(minLength: 8)

                actionControl
            }

            Text(item.description)
                .font(.subheadline)
                .foregroundStyle(JarvisColors.secondaryText)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                if item.installed {
                    Label(item.coreBuiltin ? "Built-in" : "Installed", systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(JarvisColors.accent)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(JarvisColors.softAccent))

                    if let version = item.version, !version.isEmpty {
                        Text(version)
                            .font(.caption2.monospaced())
                            .foregroundStyle(JarvisColors.secondaryText)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Capsule().fill(JarvisColors.panelStrong))
                    }
                } else {
                    Text(item.installSizeHint)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(JarvisColors.secondaryText)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(JarvisColors.panelStrong))

                    if !item.packages.isEmpty {
                        Text("apt: " + item.packages.joined(separator: ", "))
                            .font(.caption2.monospaced())
                            .foregroundStyle(JarvisColors.secondaryText)
                            .lineLimit(1)
                    }
                }
            }

            if let op = item.operation {
                HStack(spacing: 8) {
                    if op.isRunning {
                        ProgressView()
                            .controlSize(.small)
                            .tint(JarvisColors.accent)
                    } else {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(JarvisColors.clay)
                    }
                    Text(op.message ?? op.phase)
                        .font(.caption)
                        .foregroundStyle(op.isRunning ? JarvisColors.primaryText : JarvisColors.clay)
                }
            }
        }
        .padding(dynamicTypeSize.isAccessibilitySize ? 16 : 18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(JarvisColors.border, lineWidth: 1)
        )
    }

    @ViewBuilder
    private var actionControl: some View {
        if let op = item.operation, op.isRunning {
            ProgressView()
                .tint(JarvisColors.accent)
        } else if item.installed {
            if !item.coreBuiltin {
                Button {
                    onRemove?(item.id)
                } label: {
                    Text("Remove")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(JarvisColors.secondaryText)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(Capsule().fill(JarvisColors.panelStrong))
                        .overlay(Capsule().strokeBorder(JarvisColors.border, lineWidth: 1))
                }
                .buttonStyle(.plain)
                .disabled(maintenanceActive || onRemove == nil)
            }
        } else {
            Button {
                onInstall?(item.id)
            } label: {
                Text("Install")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Capsule().fill(maintenanceActive ? JarvisColors.secondaryText.opacity(0.45) : JarvisColors.accent))
            }
            .buttonStyle(.plain)
            .disabled(maintenanceActive || onInstall == nil)
        }
    }
}

/// An agreed backend adapter can provide these display values and actions.
/// Default rows describe selected software only; no API or install is simulated.
struct JarvisPluginsView: View {
    var cards: [JarvisPluginCardPresentation] = JarvisPluginCardPresentation.selectedDefaults
    var catalog: JarvisPluginCatalogSnapshot?
    var isLoading = false
    var catalogError: String?
    var preferencesAvailable = false
    var statusMessage: String?
    var isSaving = false
    var pendingChange: JarvisPluginPolicyChange?
    var onEnabledChange: ((String, Bool) -> Void)?
    var onPreferredChange: ((String, Bool) -> Void)?
    var onRecovery: ((String, JarvisPluginCardPresentation.Recovery) -> Void)?
    var onRecheckChange: (() -> Void)?
    var onRefresh: (() async -> Void)?
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        if let catalog {
            // Local display expiry only. This never polls or retries an operation.
            TimelineView(.explicit(([catalog.evaluatedAt] + catalog.observationDeadlines).sorted())) { _ in
                content(cards: catalog.presentations(at: Date(), pendingChange: pendingChange, isSaving: isSaving))
            }
        } else { content(cards: cards) }
    }

    private func content(cards: [JarvisPluginCardPresentation]) -> some View {
        let outerPadding: CGFloat = dynamicTypeSize.isAccessibilitySize ? 16 : 20
        return NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Plugins")
                        .font(JarvisTypography.pageTitle)
                        .foregroundStyle(JarvisColors.primaryText)

                    Text(preferencesAvailable
                        ? "Choose plugins for new jobs."
                        : catalog != nil
                            ? "A tool may still need an account connection or your permission for a request."
                            : dynamicTypeSize.isAccessibilitySize
                                ? "Connections haven't been checked."
                                : "Suggested tools. Connections haven't been checked yet.")
                        .font(.subheadline)
                        .foregroundStyle(JarvisColors.secondaryText)
                        .lineSpacing(3)

                    if isLoading && cards.isEmpty {
                        ProgressView("Loading plugins").frame(maxWidth: .infinity, alignment: .leading)
                        ForEach(0..<3, id: \.self) { _ in
                            RoundedRectangle(cornerRadius: 20, style: .continuous)
                                .fill(JarvisColors.panelStrong)
                                .frame(height: 124)
                                .redacted(reason: .placeholder)
                                .accessibilityHidden(true)
                        }
                    } else {
                        if isLoading {
                            ProgressView("Refreshing plugins").frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if let catalogError, pendingChange == nil {
                            Text(catalogError).font(.callout).foregroundStyle(JarvisColors.clay)
                        }
                        if !isSaving && pendingChange != nil {
                            Button("Check change") { onRecheckChange?() }
                                .disabled(onRecheckChange == nil)
                        }
                        if let statusMessage {
                            Text(statusMessage).font(.footnote).foregroundStyle(JarvisColors.secondaryText)
                        }
                        ForEach(cards) { card in
                            JarvisPluginCardView(
                                card: card,
                                preferencesAvailable: preferencesAvailable,
                                onEnabledChange: onEnabledChange,
                                onPreferredChange: onPreferredChange,
                                onRecovery: onRecovery
                            )
                        }
                        if cards.isEmpty, catalogError == nil {
                            Text("No plugins were returned. Refresh to check availability.")
                                .foregroundStyle(JarvisColors.secondaryText)
                        }
                    }
                    if !preferencesAvailable {
                        Text("Plugin preferences are not available yet.")
                            .font(.footnote)
                            .foregroundStyle(JarvisColors.secondaryText)
                            .padding(.top, 2)
                    }
                }
                .padding(.horizontal, outerPadding)
                .padding(.top, 12)
                .padding(.bottom, 28)
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .navigationBarTitleDisplayMode(.inline)
            .refreshable { await onRefresh?() }
            .background(JarvisColors.background)
            .tint(JarvisColors.accent)
        }
    }
}

struct JarvisPluginCardView: View {
    let card: JarvisPluginCardPresentation
    var preferencesAvailable: Bool
    var onEnabledChange: ((String, Bool) -> Void)?
    var onPreferredChange: ((String, Bool) -> Void)?
    var onRecovery: ((String, JarvisPluginCardPresentation.Recovery) -> Void)?
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    private var isConfirmedEnabled: Bool { card.confirmedEnabled == true }
    private var cardPadding: CGFloat { dynamicTypeSize.isAccessibilitySize ? 16 : 18 }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 12) {
                if !dynamicTypeSize.isAccessibilitySize {
                    Image(systemName: card.symbol)
                        .font(.body.weight(.medium))
                        .foregroundStyle(isConfirmedEnabled ? JarvisColors.accent : JarvisColors.primaryText)
                        .frame(width: 40, height: 40)
                        .background(
                            RoundedRectangle(cornerRadius: 11, style: .continuous)
                                .fill(isConfirmedEnabled ? JarvisColors.softAccent : JarvisColors.panelStrong)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                                        .strokeBorder(isConfirmedEnabled ? Color.clear : JarvisColors.border, lineWidth: 1)
                                )
                        )
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text(card.title)
                        .font(.headline)
                        .foregroundStyle(JarvisColors.primaryText)
                    if card.operation == nil {
                        Text(card.readinessLabel)
                            .font(.caption)
                            .foregroundStyle(JarvisColors.secondaryText)
                    }
                }
                Spacer(minLength: 0)
            }

            Text(card.detail)
                .font(.subheadline)
                .foregroundStyle(JarvisColors.secondaryText)
                .fixedSize(horizontal: false, vertical: true)

            if let operation = card.operation {
                HStack(spacing: 10) {
                    if operation.isInProgress { ProgressView().tint(JarvisColors.accent) }
                    Text(operation.label)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(JarvisColors.primaryText)
                }
                .accessibilityElement(children: .combine)

                if let message = operation.message {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(JarvisColors.secondaryText)
                }
                if let recovery = operation.recovery {
                    Button(recovery.label) { onRecovery?(card.id, recovery) }
                        .font(.subheadline.weight(.semibold))
                        .disabled(onRecovery == nil)
                }
                if case .running(.removing, _) = operation {
                    Text("Jobs already running can finish.")
                        .font(.footnote)
                        .foregroundStyle(JarvisColors.secondaryText)
                }
            }

            if card.isSelectedDefault {
                defaultBadge
            }

            if let account = card.accountLabel {
                Text(account)
                    .font(.footnote)
                    .foregroundStyle(JarvisColors.secondaryText)
            }

            DisclosureGroup("Details") {
                VStack(alignment: .leading, spacing: 8) {
                    if let installation = card.installationLabel {
                        Text(installation)
                    }
                    if card.confirmedEnabled == false {
                        Text("Off for new Jarvis plugin jobs. Jobs already running can finish.")
                    }
                    Text("General terminal work uses separate permissions.")
                }
                .font(.footnote)
                .foregroundStyle(JarvisColors.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
            }
            .font(.subheadline)
            .tint(JarvisColors.secondaryText)

            if preferencesAvailable, let enabled = card.confirmedEnabled {
                Toggle("Enabled for new requests", isOn: Binding(get: { enabled }, set: { value in
                    onEnabledChange?(card.id, value)
                }))
                .font(.subheadline)
                .foregroundStyle(JarvisColors.primaryText)
                .tint(JarvisColors.accent)
                .disabled(!card.canChangePreference || onEnabledChange == nil)
                .accessibilityHint("The confirmed preference appears after saving. Existing jobs keep running.")

                Button(card.isSelectedDefault ? "Clear preferred tool" : "Use as preferred tool") {
                    onPreferredChange?(card.id, !card.isSelectedDefault)
                }
                .font(.subheadline)
                .disabled(!card.canChangePreference || onPreferredChange == nil)
                .accessibilityHint("Your preference is remembered while this tool is disabled.")
            }
        }
        .padding(cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(JarvisColors.border, lineWidth: 1)
        )
    }

    @ViewBuilder
    private var defaultBadge: some View {
        let badgeShape = RoundedRectangle(
            cornerRadius: dynamicTypeSize.isAccessibilitySize ? 12 : 100,
            style: .continuous
        )
        if isConfirmedEnabled {
            Label(card.defaultLabel, systemImage: "checkmark.circle.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(JarvisColors.accent)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(badgeShape.fill(JarvisColors.softAccent))
        } else {
            Text(card.defaultLabel)
                .font(.caption.weight(.medium))
                .foregroundStyle(JarvisColors.secondaryText)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(badgeShape.fill(JarvisColors.panelStrong))
                .overlay(badgeShape.strokeBorder(JarvisColors.border, lineWidth: 1))
        }
    }
}
