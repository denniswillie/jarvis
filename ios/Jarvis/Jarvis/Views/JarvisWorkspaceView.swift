import SwiftUI
import AVKit

/// Dynamic content surface. The presented document leads; background activity
/// stays reachable without displacing the content Jarvis was asked to show.
struct JarvisWorkspaceView: View {
    @ObservedObject var model: JarvisWorkspaceViewModel
    var openTask: ((UUID) -> Void)? = nil
    var openInbox: ((UUID) -> Void)? = nil
    var isActive = true
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var activityExpanded = false

    private var hasActivity: Bool {
        !model.tasks.isEmpty || !model.inbox.isEmpty || model.tasksError != nil || model.inboxError != nil
    }
    private var outerPadding: CGFloat { dynamicTypeSize.isAccessibilitySize ? 16 : 18 }
    private var sheetPadding: CGFloat { dynamicTypeSize.isAccessibilitySize ? 16 : 20 }

    var body: some View {
        JarvisWorkspaceScrollView(model: model, presentationActive: isActive) {
            VStack(alignment: .leading, spacing: 22) {
                Text("Workspace")
                    .font(JarvisTypography.pageTitle)
                    .foregroundStyle(JarvisColors.primaryText)
                    .accessibilityAddTraits(.isHeader)

                if model.sandbox.isShowingScreen || model.sandbox.phase == .choosing {
                    JarvisSandboxView(model: model.sandbox)
                        .id("jarvis-presented-content")
                } else if model.mediaPresentation != nil {
                    JarvisWorkspaceMediaView(model: model, preview: model.mediaPreview)
                        .padding(sheetPadding)
                        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(JarvisColors.border, lineWidth: 1))
                        .id("jarvis-presented-content")
                } else if let file = model.filePreview {
                    JarvisWorkspaceFileView(file: file)
                        .padding(sheetPadding)
                        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 20, style: .continuous)
                                .strokeBorder(JarvisColors.border, lineWidth: 1)
                        )
                } else {
                    JarvisSandboxView(model: model.sandbox)
                }

                JarvisWorkspaceOffersView(model: model)

                if hasActivity {
                    if model.mediaPresentation != nil || model.sandbox.isShowingScreen {
                        DisclosureGroup(isExpanded: $activityExpanded) {
                            activity.padding(.top, 16)
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: "tray").foregroundStyle(JarvisColors.accent)
                                Text("Activity")
                                    .font(.headline)
                                    .foregroundStyle(JarvisColors.primaryText)
                                Spacer()
                                Text("\(model.tasks.count + model.inbox.count)")
                                    .font(.caption.monospacedDigit().weight(.semibold))
                                    .foregroundStyle(JarvisColors.secondaryText)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(JarvisColors.panelStrong, in: Capsule())
                            }
                        }
                        .tint(JarvisColors.secondaryText)
                        .padding(18)
                        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .strokeBorder(JarvisColors.border, lineWidth: 1)
                        )
                    } else {
                        activity
                    }
                }

                JarvisWorkspaceDeliveryHistoryView(model: model)
            }
            .padding(.horizontal, outerPadding)
            .padding(.top, 16)
            .padding(.bottom, 24)
            .frame(maxWidth: 820, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .tint(JarvisColors.accent)
        .refreshable { await model.refresh() }
    }

    private var activity: some View {
        VStack(alignment: .leading, spacing: 20) {
            if !model.tasks.isEmpty || model.tasksError != nil {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Tasks")
                        .font(.headline)
                        .foregroundStyle(JarvisColors.primaryText)
                    if let error = model.tasksError {
                        Text(error).font(.footnote).foregroundStyle(JarvisColors.clay)
                    }
                    ForEach(model.tasks.prefix(4)) { task in
                        Button { openTask?(task.id) } label: {
                            HStack(alignment: .top, spacing: 12) {
                                if !dynamicTypeSize.isAccessibilitySize {
                                    Image(systemName: task.status == "waiting_for_input" ? "questionmark.bubble" : "arrow.triangle.branch")
                                        .foregroundStyle(JarvisColors.accent)
                                        .padding(.top, 3)
                                }
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(task.displayTitle)
                                        .font(.subheadline.weight(.medium))
                                        .foregroundStyle(JarvisColors.primaryText)
                                        .multilineTextAlignment(.leading)
                                        .lineLimit(dynamicTypeSize.isAccessibilitySize ? nil : 3)
                                    Text(task.statusLabel)
                                        .font(.caption)
                                        .foregroundStyle(JarvisColors.secondaryText)
                                }
                                Spacer(minLength: 4)
                                if !dynamicTypeSize.isAccessibilitySize {
                                    Image(systemName: "chevron.right")
                                        .font(.caption)
                                        .foregroundStyle(JarvisColors.secondaryText.opacity(0.6))
                                        .padding(.top, 4)
                                }
                            }
                            .padding(16)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .strokeBorder(JarvisColors.border, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .disabled(openTask == nil)
                    }
                }
            }
            if !model.inbox.isEmpty || model.inboxError != nil {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Answers & reminders")
                        .font(.headline)
                        .foregroundStyle(JarvisColors.primaryText)
                    if let error = model.inboxError {
                        Text(error).font(.footnote).foregroundStyle(JarvisColors.clay)
                    }
                    ForEach(model.inbox.prefix(4)) { item in
                        Button { openInbox?(item.id) } label: {
                            HStack(spacing: 12) {
                                if !dynamicTypeSize.isAccessibilitySize {
                                    Image(systemName: "tray").foregroundStyle(JarvisColors.accent)
                                }
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(item.sourceLabel)
                                        .font(.subheadline.weight(.medium))
                                        .foregroundStyle(JarvisColors.primaryText)
                                    Text(item.stateLabel)
                                        .font(.caption)
                                        .foregroundStyle(JarvisColors.secondaryText)
                                }
                                Spacer()
                                if !dynamicTypeSize.isAccessibilitySize {
                                    Image(systemName: "chevron.right")
                                        .font(.caption)
                                        .foregroundStyle(JarvisColors.secondaryText.opacity(0.6))
                                }
                            }
                            .padding(14)
                            .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .strokeBorder(JarvisColors.border, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .disabled(openInbox == nil)
                    }
                }
            }
        }
    }
}

/// Supplies the real visible scroll viewport to the document renderer. A fresh
/// presentation scrolls directly to its content, without an animation/ACK race.
struct JarvisWorkspaceScrollView<Content: View>: View {
    @ObservedObject var model: JarvisWorkspaceViewModel
    private let presentationActive: Bool
    private let content: Content
    init(model: JarvisWorkspaceViewModel, presentationActive: Bool = true, @ViewBuilder content: () -> Content) {
        self.model = model; self.presentationActive = presentationActive; self.content = content()
    }
    var body: some View {
        GeometryReader { viewport in
            ScrollViewReader { scroll in
                ScrollView { content }
                    .environment(\.jarvisDocumentViewport, viewport.frame(in: .global))
                    .environment(\.jarvisDocumentPresentationActive, presentationActive)
                    .onChange(of: model.displayRequestID, initial: true) { _, request in
                        guard request != nil, presentationActive, model.canRead else { return }
                        scroll.scrollTo("jarvis-presented-content", anchor: .top)
                    }
            }
        }
        .background(JarvisColors.background)
    }
}

private struct JarvisDocumentViewportKey: EnvironmentKey {
    static let defaultValue: CGRect = .null
}
private struct JarvisDocumentPresentationActiveKey: EnvironmentKey {
    static let defaultValue = false
}
extension EnvironmentValues {
    var jarvisDocumentPresentationActive: Bool {
        get { self[JarvisDocumentPresentationActiveKey.self] }
        set { self[JarvisDocumentPresentationActiveKey.self] = newValue }
    }
    var jarvisDocumentViewport: CGRect {
        get { self[JarvisDocumentViewportKey.self] }
        set { self[JarvisDocumentViewportKey.self] = newValue }
    }
}
private struct JarvisMediaBodySample: Equatable {
    let presentationID: UUID
    let frame: CGRect
}
private struct JarvisMediaBodyFrameKey: PreferenceKey {
    static let defaultValue: JarvisMediaBodySample? = nil
    static func reduce(value: inout JarvisMediaBodySample?, nextValue: () -> JarvisMediaBodySample?) {
        if let next = nextValue() { value = next }
    }
}

private struct JarvisWorkspaceMediaView: View {
    @ObservedObject var model: JarvisWorkspaceViewModel
    @ObservedObject var preview: JarvisMediaPreviewViewModel
    @Environment(\.jarvisDocumentViewport) private var viewport
    @Environment(\.jarvisDocumentPresentationActive) private var presentationActive
    @State private var sample: JarvisMediaBodySample?
    @State private var readyPresentationID: UUID?

    private var visiblePresentation: JarvisWorkspaceViewModel.MediaPresentation? {
        guard presentationActive, model.canRead, let media = model.mediaPresentation,
              model.displayRequestID == media.requestID, readyPresentationID == media.id,
              let sample, sample.presentationID == media.id,
              !viewport.isNull, !viewport.isEmpty, !sample.frame.isNull, !sample.frame.isEmpty else { return nil }
        let visible = sample.frame.intersection(viewport)
        guard !visible.isNull, visible.width >= min(80, sample.frame.width),
              visible.height >= min(32, sample.frame.height) else { return nil }
        return media
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(preview.filename ?? "Video")
                .font(JarvisTypography.headline(22))
                .foregroundStyle(JarvisColors.primaryText)
                .accessibilityAddTraits(.isHeader)
            if let media = model.mediaPresentation, let player = preview.player {
                JarvisReadyVideoPlayer(player: player) { ready in
                    guard model.mediaPresentation?.id == media.id else { return }
                    readyPresentationID = ready ? media.id : nil
                    model.mediaReadinessChanged(presentationID: media.id, ready: ready)
                }
                .id(media.id)
                .frame(minHeight: 240, idealHeight: 320, maxHeight: 420)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .background(GeometryReader { geometry in
                    Color.clear.preference(key: JarvisMediaBodyFrameKey.self,
                        value: .init(presentationID: media.id, frame: geometry.frame(in: .global)))
                })
                .accessibilityLabel("Selected saved video player")
                .accessibilityIdentifier("workspaceVideoPlayer")
            } else if let error = preview.errorMessage {
                Text(error).font(.callout).foregroundStyle(JarvisColors.clay)
            } else {
                ProgressView("Opening video…").frame(maxWidth: .infinity, minHeight: 120)
            }
        }
        .onPreferenceChange(JarvisMediaBodyFrameKey.self) { sample = $0 }
        .task(id: visiblePresentation) {
            guard let presentation = visiblePresentation else { return }
            await Task.yield()
            guard !Task.isCancelled else { return }
            await model.didDisplayMedia(presentationID: presentation.id, requestID: presentation.requestID,
                artifactID: presentation.artifactID, versionID: presentation.versionID)
        }
        .onDisappear {
            let retiredID = sample?.presentationID ?? readyPresentationID
            sample = nil; readyPresentationID = nil
            if let retiredID { model.mediaReadinessChanged(presentationID: retiredID, ready: false) }
        }
    }
}

/// Native playback controls with a real first-frame readiness signal. The parent
/// separately proves its matching video area is in the active visible viewport.
private struct JarvisReadyVideoPlayer: UIViewControllerRepresentable {
    let player: AVPlayer
    let onReadinessChanged: (Bool) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onReadinessChanged: onReadinessChanged) }
    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        controller.player = player
        controller.allowsPictureInPicturePlayback = false
        controller.updatesNowPlayingInfoCenter = false
        context.coordinator.observe(controller, player: player)
        return controller
    }
    func updateUIViewController(_ controller: AVPlayerViewController, context: Context) {
        context.coordinator.onReadinessChanged = onReadinessChanged
    }
    static func dismantleUIViewController(_ controller: AVPlayerViewController, coordinator: Coordinator) {
        coordinator.stop()
        controller.player = nil
    }

    @MainActor
    final class Coordinator {
        var onReadinessChanged: (Bool) -> Void
        private var observation: NSKeyValueObservation?
        private weak var controller: AVPlayerViewController?
        private weak var expectedPlayer: AVPlayer?
        private var active = false

        init(onReadinessChanged: @escaping (Bool) -> Void) { self.onReadinessChanged = onReadinessChanged }
        func observe(_ controller: AVPlayerViewController, player: AVPlayer) {
            self.controller = controller; expectedPlayer = player; active = true
            observation = controller.observe(\.isReadyForDisplay, options: [.initial, .new]) { [weak self] _, change in
                let ready = change.newValue == true
                Task { @MainActor [weak self] in
                    guard let self, self.active, let controller = self.controller,
                          controller.player === self.expectedPlayer else { return }
                    self.onReadinessChanged(ready && controller.isReadyForDisplay && controller.player?.currentItem != nil)
                }
            }
        }
        func stop() { active = false; observation?.invalidate(); observation = nil; controller = nil; expectedPlayer = nil }
    }
}

struct JarvisWorkspaceOffersView: View {
    @ObservedObject var model: JarvisWorkspaceViewModel
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    private var pendingOffers: [JarvisWorkspaceRequest] {
        let completed = Set(model.deliveries.filter { $0.acknowledgement?.state.isTerminal == true }.map(\.id))
        return model.offers.filter { !completed.contains($0.id) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(pendingOffers) { request in
                VStack(alignment: .leading, spacing: 10) {
                    Label(
                        request.action == .download ? "Save a copy on this phone" : "Ready to show",
                        systemImage: request.action == .download ? "arrow.down.circle" : (request.artifactKind == "sandbox" ? "display" : "doc.text")
                    )
                    .font(.headline)
                    .foregroundStyle(JarvisColors.primaryText)

                    Text(request.artifactKind == "sandbox" ? "Live sandbox · View only" :
                         (request.contentReady ? "Selected saved version" : "Preparing the selected version…"))
                        .font(.subheadline)
                        .foregroundStyle(JarvisColors.secondaryText)

                    (dynamicTypeSize.isAccessibilitySize
                        ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10))
                        : AnyLayout(HStackLayout(alignment: .center, spacing: 14))) {
                        Button(request.action == .download ? "Save offline" : "Open") {
                            Task { await model.perform(request) }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(JarvisColors.accent)
                        .disabled(!request.contentReady || !model.canPerform(request) || model.workingRequestID != nil)

                        Button("Dismiss") { Task { await model.dismiss(request) } }
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(JarvisColors.secondaryText)
                            .disabled(model.workingRequestID != nil)

                        if model.workingRequestID == request.id {
                            ProgressView()
                                .tint(JarvisColors.accent)
                        }
                    }

                    if request.action == .download, !model.canSaveOffline {
                        Text("Offline saving is not available yet.")
                            .font(.footnote)
                            .foregroundStyle(JarvisColors.secondaryText)
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(JarvisColors.softAccent.opacity(0.65), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(JarvisColors.accent.opacity(0.18), lineWidth: 1)
                )
            }
            if let message = model.deliveryMessage {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(JarvisColors.secondaryText)
                    .accessibilityAddTraits(.updatesFrequently)
            }
        }
    }
}

struct JarvisWorkspaceFileView: View {
    let file: JarvisWorkspaceFileContent
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(file.filename)
                .font(.title2.bold())
                .foregroundStyle(JarvisColors.primaryText)
            JarvisRetainedContentView(contentType: file.contentType, data: file.data)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The same verified small-file bytes render consistently in Workspace and Saved files.
struct JarvisRetainedContentView: View {
    let contentType: String
    let data: Data
    var body: some View {
        let mediaType = contentType.components(separatedBy: ";")[0].trimmingCharacters(in: .whitespaces).lowercased()
        Group {
            if mediaType == "application/json",
               let json = try? JSONSerialization.jsonObject(with: data),
               let pretty = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
               let text = String(data: pretty, encoding: .utf8) {
                Text(text)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(JarvisColors.primaryText)
                    .textSelection(.enabled)
            } else if mediaType.hasPrefix("text/"), let text = String(data: data, encoding: .utf8) {
                Text(text)
                    .foregroundStyle(JarvisColors.primaryText)
                    .textSelection(.enabled)
            } else {
                #if os(iOS)
                if mediaType.hasPrefix("image/"), let image = UIImage(data: data) {
                    Image(uiImage: image).resizable().scaledToFit()
                        .accessibilityLabel("Saved image")
                } else { unsupportedPreview }
                #else
                unsupportedPreview
                #endif
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private var unsupportedPreview: some View {
        Text("This file is available, but this format does not have a preview here.")
            .foregroundStyle(JarvisColors.secondaryText)
    }
}

struct JarvisWorkspaceDeliveryHistoryView: View {
    @ObservedObject var model: JarvisWorkspaceViewModel
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        if !model.unresolvedDeliveries.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Text("Phone requests to check")
                    .font(.headline)
                    .foregroundStyle(JarvisColors.primaryText)

                ForEach(model.unresolvedDeliveries) { entry in
                    VStack(alignment: .leading, spacing: 10) {
                        Text(recoveryMessage(entry))
                            .font(.subheadline)
                            .foregroundStyle(JarvisColors.secondaryText)

                        (dynamicTypeSize.isAccessibilitySize
                            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10))
                            : AnyLayout(HStackLayout(spacing: 14))) {
                            if entry.acknowledgement != nil || (entry.phase == .effectStarted && entry.request.action == .download) {
                                Button("Check saved result") { Task { await model.reconcile(entry) } }
                                    .font(.subheadline.weight(.semibold))
                            } else if entry.phase != .effectStarted {
                                Button("Continue request") { Task { await model.perform(entry.request) } }
                                    .font(.subheadline.weight(.semibold))
                            }
                            Button("Dismiss request") { Task { await model.dismiss(entry.request) } }
                                .font(.subheadline)
                                .foregroundStyle(JarvisColors.secondaryText)
                        }
                        .disabled(model.workingRequestID != nil)
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(JarvisColors.border, lineWidth: 1)
                    )
                }
            }
        }
    }

    private func recoveryMessage(_ entry: JarvisWorkspaceLocalStore.Entry) -> String {
        switch entry.acknowledgement?.state {
        case .savedLocal: return "Saved on this phone; confirmation pending"
        case .displayed: return "Shown on this phone; confirmation pending"
        default: return "This request needs a check"
        }
    }
}
