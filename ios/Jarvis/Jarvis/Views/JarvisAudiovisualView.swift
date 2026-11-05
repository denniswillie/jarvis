import SwiftUI
import AVKit

/// Bottom-tab content. The app owns data-only preparation and account lifetime;
/// tab switches suspend this model instead of permanently retiring it.
struct JarvisAudiovisualView: View {
    @ObservedObject var model: JarvisFilesViewModel
    let isActive: Bool
    var openSavedCopies: (() -> Void)? = nil
    var transferMessage: String? = nil
    @State private var showingPreview = false
    @State private var videoFullscreen = false
    @State private var opening: Task<Void, Never>?
    @Environment(\.scenePhase) private var scenePhase

    private var visible: Bool { isActive && scenePhase == .active }
    private let columns = [GridItem(.adaptive(minimum: 145), spacing: 16)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let transferMessage {
                        Text(transferMessage).font(.subheadline)
                            .foregroundStyle(JarvisColors.secondaryText)
                            .accessibilityIdentifier("audiovisualTransferStatus")
                    }
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(JarvisColors.mutedWhite)
                        TextField("Search names or descriptions", text: $model.query)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                            .submitLabel(.search)
                            .onSubmit { Task { await model.reload() } }
                            .accessibilityLabel("Search audiovisual library")
                    }.font(JarvisTypography.body(15))
                        .padding(12)
                        .background(JarvisColors.cream.opacity(0.04), in: RoundedRectangle(cornerRadius: 12))
                    if model.isLoading {
                        LazyVGrid(columns: columns, spacing: 22) {
                            ForEach(0..<6) { _ in
                                VStack(alignment: .leading, spacing: 10) {
                                    JarvisGallerySkeleton().frame(height: 132).clipShape(RoundedRectangle(cornerRadius: 14))
                                    RoundedRectangle(cornerRadius: 4).fill(JarvisColors.border).frame(width: 110, height: 12)
                                }
                            }
                        }.accessibilityElement(children: .ignore).accessibilityLabel("Loading library")
                    } else if let error = model.errorMessage, model.audiovisualFiles.isEmpty {
                        Text(error).foregroundStyle(JarvisColors.amber)
                        Button("Try again") { Task { await model.reload() } }
                    } else if model.audiovisualFiles.isEmpty {
                        Text(model.query.isEmpty ? "Saved images and completed recordings appear here." : "No images or videos match this search.")
                            .foregroundStyle(JarvisColors.softWhite)
                    } else {
                        LazyVGrid(columns: columns, alignment: .leading, spacing: 22) {
                            ForEach(model.audiovisualFiles) { file in
                                Button {
                                    opening?.cancel()
                                    showingPreview = true
                                    opening = Task { _ = await model.open(file) }
                                } label: {
                                    JarvisAudiovisualTile(file: file, reference: model.resourceReference(for: file),
                                        thumbnails: model.thumbnails,
                                        selected: model.resourceContext?.selectedResource?.fileID == UUID(uuidString: file.id))
                                }
                                .buttonStyle(.plain)
                                .accessibilityIdentifier("audiovisual-" + file.id)
                                .onAppear { model.setResourceVisible(file.id, visible: true) }
                                .onDisappear { model.setResourceVisible(file.id, visible: false) }
                            }
                        }
                    }
                    if let error = model.errorMessage, !model.audiovisualFiles.isEmpty {
                        Text(error).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.mutedWhite)
                    }
                    if model.nextResourceCursor != nil {
                        Button {
                            Task { await model.loadMoreResources() }
                        } label: {
                            if model.isLoadingMore { ProgressView() }
                            else { Text("Load more") }
                        }
                        .disabled(model.isLoadingMore)
                        .font(JarvisTypography.label(14))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                    }
                    if let message = model.deletionMessage {
                        Text(message).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.softWhite)
                    }
                }
                .foregroundStyle(JarvisColors.cream).padding(22)
            }
            .background(JarvisColors.graphite)
            .navigationTitle("Audiovisual")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                if let openSavedCopies {
                    ToolbarItem(placement: .primaryAction) {
                        Button(action: openSavedCopies) {
                            Label("On this phone", systemImage: "arrow.down.circle")
                        }
                        .accessibilityIdentifier("audiovisualSavedCopiesButton")
                    }
                }
            }
            .refreshable { await model.reload() }
            .sheet(isPresented: $showingPreview, onDismiss: {
                videoFullscreen = false
                opening?.cancel()
                model.closePreview()
            }) {
                JarvisFilePreviewView(model: model, allowsVideoFullscreen: true,
                    onVideoFullscreenChanged: { videoFullscreen = $0 })
            }
            .task(id: visible) {
                model.setResourceContextVisible(visible)
                if visible { await model.resume() }
                else { showingPreview = false; opening?.cancel(); model.suspend() }
            }
            .onDisappear {
                guard !videoFullscreen else { return }
                opening?.cancel()
                model.setResourceContextVisible(false)
                model.suspend()
            }
        }
        .tint(JarvisColors.amber)
    }
}

private struct JarvisAudiovisualTile: View {
    let file: JarvisRetainedFile
    let reference: JarvisResourceReference?
    @ObservedObject var thumbnails: JarvisAudiovisualThumbnailLoader
    let selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            GeometryReader { geometry in
              ZStack {
                RoundedRectangle(cornerRadius: 14).fill(JarvisColors.cream.opacity(0.06))
                if let thumbnail = thumbnails.images[file.id] {
                    Image(decorative: thumbnail, scale: 1).resizable().scaledToFill()
                        .frame(width: geometry.size.width, height: geometry.size.height).clipped()
                } else {
                    JarvisGallerySkeleton()
                    if file.mediaType != "video/mp4" {
                        Image(systemName: "photo").font(.system(size: 27, weight: .light))
                            .foregroundStyle(JarvisColors.secondaryText.opacity(0.55))
                    }
                }
                if file.mediaType == "video/mp4" {
                    Image(systemName: "play.circle.fill").font(.system(size: 32))
                        .foregroundStyle(.white).shadow(radius: 3)
                }
              }
              .frame(width: geometry.size.width, height: geometry.size.height)
            }
            .frame(height: 132)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(selected ? JarvisColors.amber : .clear, lineWidth: 2))
            Text(file.filename.isEmpty ? "Untitled" : file.filename)
                .font(JarvisTypography.label(14)).lineLimit(2)
            Text("ID " + (reference?.displayID ?? String(file.id.lowercased().prefix(8))))
                .font(JarvisTypography.mono(11)).foregroundStyle(JarvisColors.mutedWhite)
            if selected {
                Text("Selected").font(JarvisTypography.body(11)).foregroundStyle(JarvisColors.amber)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .multilineTextAlignment(.leading)
        .contentShape(Rectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(file.filename + (file.mediaType == "video/mp4" ? ", video" : ", image"))
        .accessibilityHint("Opens the original")
        .task(id: (reference?.versionID ?? file.memorySource.contentDigest) + (file.thumbnail?.contentSHA256 ?? "")) {
            await thumbnails.load(file, reference: reference)
        }
    }
}


private struct JarvisGallerySkeleton: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var breathing = false

    var body: some View {
        LinearGradient(colors: [JarvisColors.border.opacity(0.5), JarvisColors.surface,
                                JarvisColors.border.opacity(0.65)],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
            .opacity(reduceMotion ? 0.85 : (breathing ? 0.6 : 1))
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) { breathing = true }
            }
            .accessibilityHidden(true)
    }
}

/// Audiovisual recordings use AVKit’s native fullscreen and playback controls.
/// The preview model retains ownership of the same verified file and player.
struct JarvisAudiovisualVideoPlayer: UIViewControllerRepresentable {
    let player: AVPlayer
    var onFullscreenChanged: (Bool) -> Void = { _ in }

    func makeCoordinator() -> Coordinator { Coordinator(onFullscreenChanged: onFullscreenChanged) }
    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        controller.player = player
        controller.delegate = context.coordinator
        controller.showsPlaybackControls = true
        controller.allowsPictureInPicturePlayback = false
        controller.updatesNowPlayingInfoCenter = false
        context.coordinator.observe(controller, player: player)
        return controller
    }
    func updateUIViewController(_ controller: AVPlayerViewController, context: Context) {
        context.coordinator.onFullscreenChanged = onFullscreenChanged
        if controller.player !== player {
            context.coordinator.stop()
            controller.player = player
            controller.delegate = context.coordinator
            context.coordinator.observe(controller, player: player)
        }
    }
    static func dismantleUIViewController(_ controller: AVPlayerViewController, coordinator: Coordinator) {
        coordinator.stop()
        controller.player = nil
    }

    @MainActor
    final class Coordinator: NSObject, AVPlayerViewControllerDelegate {
        var onFullscreenChanged: (Bool) -> Void
        private var itemObservation: NSKeyValueObservation?
        private var originalOrientation: UIInterfaceOrientation = .unknown
        private weak var fullscreenScene: UIWindowScene?
        private(set) var isFullscreen = false
        private weak var controller: AVPlayerViewController?
        private weak var expectedPlayer: AVPlayer?
        private var active = false

        init(onFullscreenChanged: @escaping (Bool) -> Void) {
            self.onFullscreenChanged = onFullscreenChanged
        }

        func observe(_ controller: AVPlayerViewController, player: AVPlayer) {
            self.controller = controller; expectedPlayer = player; active = true
            itemObservation = player.observe(\.currentItem, options: [.new]) { [weak self] _, _ in
                Task { @MainActor [weak self] in
                    guard let self, self.expectedPlayer?.currentItem == nil else { return }
                    self.endFullscreen()
                }
            }
        }

        func playerViewController(_ playerViewController: AVPlayerViewController,
            willBeginFullScreenPresentationWithAnimationCoordinator coordinator: UIViewControllerTransitionCoordinator) {
            // AVKit moves its playback surface into a separate presentation;
            // the inline controller's view has no window after that move.
            fullscreenScene = playerViewController.view.window?.windowScene
            originalOrientation = fullscreenScene?.interfaceOrientation ?? .unknown
            isFullscreen = true
            // AVKit temporarily hides the containing SwiftUI sheet. That is
            // a presentation change, not permission to close its media owner.
            onFullscreenChanged(true)
            coordinator.animate(alongsideTransition: nil) { [weak self, weak playerViewController] context in
                guard let self, let playerViewController else { return }
                if context.isCancelled {
                    self.isFullscreen = false; self.onFullscreenChanged(false); return
                }
                guard self.active, self.expectedPlayer?.currentItem != nil else { self.endFullscreen(); return }
                // Widescreen recordings benefit from landscape; portrait videos
                // retain the current orientation and may rotate with the phone.
                let size = self.expectedPlayer?.currentItem?.presentationSize ?? .zero
                if playerViewController.traitCollection.userInterfaceIdiom == .phone, size.width > size.height {
                    playerViewController.setNeedsUpdateOfSupportedInterfaceOrientations()
                    self.fullscreenScene?.requestGeometryUpdate(
                        .iOS(interfaceOrientations: .landscape)) { error in
                            #if DEBUG
                            print("Audiovisual fullscreen orientation: \(error.localizedDescription)")
                            #endif
                        }
                }
            }
        }

        func playerViewController(_ playerViewController: AVPlayerViewController,
            willEndFullScreenPresentationWithAnimationCoordinator coordinator: UIViewControllerTransitionCoordinator) {
            let rate = playerViewController.player?.rate ?? 0
            let scene = fullscreenScene
            coordinator.animate(alongsideTransition: nil) { [weak self] context in
                guard let self, !context.isCancelled else { return }
                self.isFullscreen = false
                self.onFullscreenChanged(false)
                if self.active, self.expectedPlayer?.currentItem != nil, rate > 0 {
                    self.expectedPlayer?.rate = rate
                }
                self.restoreOrientation(in: scene)
            }
        }

        private func restoreOrientation(in scene: UIWindowScene?) {
            let mask: UIInterfaceOrientationMask
            switch originalOrientation {
            case .portrait: mask = .portrait
            case .portraitUpsideDown: mask = .portraitUpsideDown
            case .landscapeLeft: mask = .landscapeLeft
            case .landscapeRight: mask = .landscapeRight
            default: return
            }
            scene?.requestGeometryUpdate(.iOS(interfaceOrientations: mask)) { _ in }
        }

        private func endFullscreen() {
            guard isFullscreen, let controller else { return }
            // The account/preview owner has already emptied the player. Clear
            // AVKit's full-screen presentation as well as the inline renderer.
            let scene = fullscreenScene
            controller.player = nil
            controller.dismiss(animated: false) { [self] in
                isFullscreen = false
                onFullscreenChanged(false)
                restoreOrientation(in: scene)
            }
        }

        func stop() {
            active = false
            endFullscreen()
            itemObservation?.invalidate(); itemObservation = nil
            controller?.delegate = nil
            controller = nil; expectedPlayer = nil
        }
    }
}
