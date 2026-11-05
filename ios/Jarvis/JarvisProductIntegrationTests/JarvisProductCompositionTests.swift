import Foundation
import AVKit
import CryptoKit
import SwiftUI
import UIKit
import XCTest
@testable import Jarvis

/// Native composition fixtures, not provider/voice or physical-device evidence.
/// Requires the existing safe DEBUG launch branch; never starts audio or replay.
@MainActor
final class JarvisProductCompositionTests: XCTestCase {
    private var wire: ProductFixture!
    private var app: JarvisAppViewModel?
    private var fixtureStore: JarvisSessionStore?
    private var defaultsName: String?
    private var window: UIWindow?
    private var previousKeyWindow: UIWindow?
    private var displayedFrame: DisplayedFrame?

    private struct DisplayedFrame {
        let image: UIImage
        let requestID: UUID
        let artifactID: UUID?
        let versionID: String?
        let productTab: JarvisProductTab?
        let nativeTabTitle: String?
        let observedAt: TimeInterval
    }

    override func setUp() async throws {
        try await super.setUp()
        let args = ProcessInfo.processInfo.arguments
        guard let index = args.firstIndex(of: "--jarvis-replay-manifest"),
              index + 1 < args.count, args[index + 1] == "/dev/null",
              !args.contains("--jarvis-replay-run") else {
            throw FixtureFailure("Unsafe host launch: require --jarvis-replay-manifest /dev/null without --jarvis-replay-run")
        }
        wire = ProductFixture()
        displayedFrame = nil
        ProductFixtureProtocol.install(wire)
        guard URLProtocol.registerClass(ProductFixtureProtocol.self) else {
            throw FixtureFailure("URLProtocol registration failed; no product requests were started")
        }
        // Prove interception of the exact default session used by AppViewModel's
        // APIClient before allowing any product configuration or side reads.
        var probe = URLRequest(url: wire.baseURL.appendingPathComponent("shared-probe"))
        probe.timeoutInterval = 3
        do {
            let (data, response) = try await URLSession.shared.data(for: probe)
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  data == Data("shared-session-intercepted".utf8), wire.probeCount == 1 else {
                throw FixtureFailure("URLSession.shared interception was not proved")
            }
        } catch {
            throw FixtureFailure("URLSession.shared interception failed; stop here instead of adding a server or exercising real configuration")
        }
    }

    func testColdSandboxLookupUsesOnlyExistingSession() async throws {
        let app = try makeApp()
        try configure(app, token: "fixture-sandbox")
        app.updateProductScene(foreground: true)
        app.productTab = .workspace
        let missing = await app.openExistingSandbox()
        XCTAssertEqual(missing?.title, "No sandbox to view")
        XCTAssertNil(app.workspace)
        wire.setExistingSession(true)
        let notice = await app.openExistingSandbox()
        XCTAssertNil(notice)
        XCTAssertEqual(app.workspace?.sessionID, wire.rootSessionID)
        XCTAssertEqual(app.workspace?.sandbox.notice?.title, "No sandbox to view")
        XCTAssertTrue(wire.observedPaths.contains("/sessions/active"))
        XCTAssertTrue(wire.observedPaths.contains("/sandbox-views"))
        XCTAssertFalse(wire.observedPaths.contains("/sessions/start"))
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testDefaultCompositionEnablesMediaOnlyAfterOwnerVerification() async throws {
        let name = "jarvis.default-media.fixture." + UUID().uuidString
        defaultsName = name
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        let store = JarvisSessionStore(defaults: defaults, keychain: JarvisKeychain(service: name))
        fixtureStore = store
        // Deliberately omit workspaceJournalFactory: exercise the shipping default.
        let app = JarvisAppViewModel(store: store, audioSessionViewModel: JarvisAudioSessionViewModel())
        self.app = app
        wire.holdContext(wire.rootSessionID)
        try configure(app, token: "fixture-default-media")
        app.synchronizeProductSession(wire.rootSessionID.uuidString)
        let workspace = try XCTUnwrap(app.workspace)
        try await eventually("Owner context must be held") { self.wire.hasHeldContext(self.wire.rootSessionID) }
        XCTAssertFalse(workspace.canHandleDeliveries)
        XCTAssertFalse(workspace.canSaveOffline)
        wire.releaseContext(wire.rootSessionID)
        try await eventually("Default verified composition must enable file access") { workspace.canSaveOffline }
        app.prepareSavedFiles()
        try await eventually("Explicit library access must compose its media model") { app.savedFiles?.media != nil }
        XCTAssertNil(workspace.mediaPresentation, "Verification must not open an unrequested file")
        XCTAssertNil(app.savedFiles?.selectedFile)
        app.resetConfiguration()
        XCTAssertFalse(workspace.canSaveOffline, "Account reset retires file access")
        XCTAssertNil(app.savedFiles)
    }

    override func tearDown() async throws {
        window?.isHidden = true
        window?.rootViewController = nil
        window = nil
        previousKeyWindow?.makeKey()
        previousKeyWindow = nil
        if let app {
            let deadline = Date().addingTimeInterval(1)
            while app.pushNotifications?.isBusy == true && Date() < deadline {
                try await Task.sleep(nanoseconds: 10_000_000)
            }
            app.resetConfiguration()
            XCTAssertNil(app.configuration, "Fixture reset must use and complete the real product lifecycle")
        }
        wire?.releaseAllContexts()
        if let host = wire?.baseURL.host {
            let deadline = Date().addingTimeInterval(1)
            var pending = await fixtureTasks(host: host)
            pending.forEach { $0.cancel() }
            while !pending.isEmpty && Date() < deadline {
                try await Task.sleep(nanoseconds: 10_000_000)
                pending = await fixtureTasks(host: host)
                pending.forEach { $0.cancel() }
            }
            XCTAssertTrue(pending.isEmpty, "Only this fixture host's HTTP tasks must drain before the next case")
        }
        app = nil
        // Only the unique fixture session credential/defaults are removed.
        // Retained fixture journals remain in the simulator for evidence.
        fixtureStore?.clearConfiguration()
        fixtureStore = nil
        if let defaultsName { UserDefaults.standard.removePersistentDomain(forName: defaultsName) }
        defaultsName = nil
        ProductFixtureProtocol.install(nil)
        // Keep the fail-closed protocol registered for this dedicated test
        // process: an unexpectedly late task must not escape onto the network.
        wire = nil
        displayedFrame = nil
        try await super.tearDown()
    }

    func testMediaDisplayRequiresReadyFrameInActiveWorkspace() async throws {
        let args = ProcessInfo.processInfo.arguments
        guard let index = args.firstIndex(of: "--jarvis-media-fixture"), index + 1 < args.count else {
            throw XCTSkip("No pinned silent synthetic MP4 was supplied via --jarvis-media-fixture")
        }
        let source = URL(fileURLWithPath: args[index + 1])
        let reader = try FileHandle(forReadingFrom: source)
        defer { try? reader.close() }
        var hash = SHA256(), byteCount = 0
        while let chunk = try reader.read(upToCount: 65_536), !chunk.isEmpty {
            hash.update(data: chunk); byteCount += chunk.count
        }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard byteCount == 10_877_150,
              digest == "9baaa37c31f8ff2ffff8de10ec63cc0cef615dd60f5e08bd5e9f99de2bca11de" else {
            throw FixtureFailure("Only the pinned three-second synthetic silent H264 fixture is allowed")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-media-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let media = NativeMediaFixture(source: source, root: root, byteCount: byteCount, digest: digest)
        defer { media.beforeDisplayed = nil }
        let store = try JarvisWorkspaceLocalStore(ownerKey: "native-media-fixture", root: root, allowsFileBytes: true)
        let model = JarvisWorkspaceViewModel(api: media, sessionID: UUID(), store: store, automaticPolling: false)
        defer { model.deactivate() }
        model.setVisibility(true, foreground: true)
        try mount(JarvisWorkspaceView(model: model, isActive: false), height: 844)
        await model.perform(media.offer)
        try await eventually("The actual native player must prepare its first frame") {
            self.nativeMediaController()?.isReadyForDisplay == true
        }
        XCTAssertEqual(media.acknowledgements.map(\.state), [.claimed], "An inactive renderer must not acknowledge display")
        XCTAssertEqual(model.displayRequestID, media.offer.id)
        XCTAssertTrue(model.unresolvedDeliveries.isEmpty, "A prepared active display is not a recovery request")
        XCTAssertEqual(model.mediaPreview.player?.rate, 0, "Preparing a display must not autoplay")
        let before = try await store.entry(id: media.offer.id)
        XCTAssertNil(before?.acknowledgement?.localReceipt)
        XCTAssertTrue(model.offlineFiles.isEmpty)

        var boundaryImage: UIImage?
        media.beforeDisplayed = { [weak self] in
            guard let self, let window = self.window, let controller = self.nativeMediaController() else {
                XCTFail("Displayed receipt lacked a native player controller"); return
            }
            XCTAssertTrue(controller.isReadyForDisplay)
            XCTAssertTrue(controller.player === model.mediaPreview.player)
            XCTAssertEqual(controller.player?.rate, 0)
            XCTAssertTrue(model.unresolvedDeliveries.isEmpty, "The displayed confirmation must not expose a recovery card")
            XCTAssertTrue(controller.view.window === window)
            let intersection = controller.view.convert(controller.view.bounds, to: window).intersection(window.bounds)
            XCTAssertGreaterThanOrEqual(intersection.width, 80)
            XCTAssertGreaterThanOrEqual(intersection.height, 32)
            boundaryImage = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: false)
            }
        }
        window?.rootViewController = UIHostingController(rootView: JarvisWorkspaceView(model: model).preferredColorScheme(.light))
        window?.layoutIfNeeded()
        try await eventually("The active renderer must acknowledge the exact ready video once") {
            media.acknowledgements.map(\.state) == [.claimed, .displayed]
        }
        let image = XCTAttachment(image: try XCTUnwrap(boundaryImage))
        image.name = "media-displayed-boundary-requires-visual-review"
        image.lifetime = .keepAlways
        add(image)
        let after = try await store.entry(id: media.offer.id)
        XCTAssertEqual(after?.request.artifactID, media.offer.artifactID)
        XCTAssertEqual(after?.request.versionID, media.offer.versionID)
        XCTAssertEqual(after?.acknowledgement?.state, .displayed)
        XCTAssertNil(after?.acknowledgement?.localReceipt, "Viewing a temporary copy must not claim an offline save")
        XCTAssertTrue(model.offlineFiles.isEmpty)
        let player = try XCTUnwrap(model.mediaPreview.player)
        let temporaryURL = try XCTUnwrap(media.temporaryURL)
        model.deactivate()
        XCTAssertNil(player.currentItem)
        XCTAssertFalse(FileManager.default.fileExists(atPath: temporaryURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path), "Fixture source must remain intact")
        XCTAssertEqual(media.acknowledgements.map(\.state), [.claimed, .displayed])
        XCTAssertTrue(wire.unexpectedRequests.isEmpty)
    }

    private func nativeMediaController() -> AVPlayerViewController? {
        func find(_ controller: UIViewController) -> AVPlayerViewController? {
            if let player = controller as? AVPlayerViewController { return player }
            if let presented = controller.presentedViewController, let player = find(presented) { return player }
            return controller.children.lazy.compactMap(find).first
        }
        return window?.rootViewController.flatMap(find)
    }

    func testHeldContextCannotAttachAfterResetAndConfigurationReplacement() async throws {
        let first = UUID(), second = UUID()
        wire.holdContext(first)
        let app = try makeApp()
        try configure(app, token: "fixture-A")
        app.synchronizeProductSession(first.uuidString)
        _ = try XCTUnwrap(app.workspace, "Valid fixture configuration must create a Workspace")
        try await eventually("A context must reach the intercepted transport") { self.wire.hasHeldContext(first) }
        let retired = try XCTUnwrap(app.workspace)
        XCTAssertFalse(retired.journalReady)
        try await waitForSetupIdle(app)
        app.resetConfiguration()
        XCTAssertNil(app.configuration)
        XCTAssertNil(app.workspace)
        try configure(app, token: "fixture-B")
        app.synchronizeProductSession(second.uuidString)
        try await eventually("B context must independently unlock B's note journal") { app.workspace?.journalReady == true }
        let current = try XCTUnwrap(app.workspace)
        XCTAssertEqual(current.sessionID, second)
        XCTAssertFalse(current.canSaveOffline)
        wire.releaseContext(first)
        // A bounded settling period checks the late response after its release.
        try await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertTrue(app.workspace === current)
        XCTAssertFalse(retired.canRead)
        XCTAssertFalse(retired.journalReady)
        XCTAssertEqual(wire.releasedContexts, [first])
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testAuthorizationFailureLocksButRetainsTheRenderedJournal() async throws {
        let session = wire.rootSessionID
        let app = try await readyApp(session: session)
        let model = try XCTUnwrap(app.workspace)
        try mount(JarvisWorkspaceView(model: model, isActive: false), height: 844)
        app.audioSessionViewModel.onWorkspacePresentationIntent?(session, wire.requestID)
        try await eventually("The exact request must be claimed before the visibility check") {
            self.wire.ackStates == ["claimed"] && model.displayRequestID == self.wire.requestID
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertFalse(wire.didDisplay, "A mounted but inactive product tab must not acknowledge display")
        window?.rootViewController = UIHostingController(rootView: JarvisWorkspaceView(model: model).preferredColorScheme(.light))
        window?.layoutIfNeeded()
        try await eventually("The real document task must send displayed") { self.wire.didDisplay }
        try attachBoundaryFrame()
        let journal = try journalFor(session)
        try await eventually("The actual persisted outcome must be acknowledged") {
            try await journal.entry(id: self.wire.requestID)?.acknowledged == true
        }
        let before = try await journal.entry(id: wire.requestID)
        XCTAssertEqual(before?.acknowledgement?.state, .displayed)
        XCTAssertEqual(before?.request.versionID, "fixture-v1")
        wire.denyWorkspace()
        await model.refresh()
        XCTAssertFalse(model.canRead)
        XCTAssertFalse(model.canHandleDeliveries)
        XCTAssertNil(model.mediaPresentation)
        XCTAssertTrue(model.deliveries.isEmpty)
        let retained = try await journal.entry(id: wire.requestID)
        XCTAssertEqual(retained, before, "An auth lock must not delete or rewrite the saved outcome")
        XCTAssertFalse(model.attachJournal(journal), "A revoked lifetime cannot unlock in place")
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }


    func testRootFourDestinationsAtOrdinaryTextAndFreshPresentation() async throws {
        try await captureRootDestinations(accessibility: false)
    }

    func testRootFourDestinationsAtLargestAccessibilityText() async throws {
        try await captureRootDestinations(accessibility: true)
    }

    func testAudiovisualTabLoadsRealThumbnailsWithoutVoice() async throws {
        try await captureAudiovisual(accessibility: false)
    }

    func testAudiovisualTabAtLargestAccessibilityText() async throws {
        try await captureAudiovisual(accessibility: true)
    }

    func testAudiovisualMissingPreviewStillOpensOriginal() async throws {
        try await openWithoutPreview(failing: false)
    }

    func testAudiovisualFailedPreviewStillOpensOriginal() async throws {
        try await openWithoutPreview(failing: true)
    }

    private func openWithoutPreview(failing: Bool) async throws {
        let bytes = UIGraphicsImageRenderer(size: CGSize(width: 480, height: 320)).pngData { context in
            UIColor.systemTeal.setFill(); context.fill(CGRect(x: 0, y: 0, width: 480, height: 320))
        }
        wire.enableAudiovisualImages([bytes], previews: failing, failPreviews: failing)
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-preview-missing")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 844)
        app.productTab = .audiovisual
        try await eventually("Missing/failed preview leaves a listed item") {
            app.audiovisual?.audiovisualFiles.count == 1 && app.audiovisual?.thumbnails.messages.count == 1
        }
        let model = try XCTUnwrap(app.audiovisual)
        XCTAssertTrue(model.thumbnails.images.isEmpty)
        XCTAssertFalse(wire.observedPaths.contains { $0.hasSuffix("/content") })
        try await attachGalleryFrame(name: failing ? "gallery-failed-preview" : "gallery-missing-preview")
        // Invoke the same selection action as the tile; thumbnail state cannot gate it.
        let file = try XCTUnwrap(model.audiovisualFiles.first)
        let opened = await model.open(file)
        XCTAssertEqual(opened?.data, bytes)
        XCTAssertEqual(wire.observedPaths.filter { $0.hasSuffix("/content") }.count, 1)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty)
    }

    func testHomeRecordingStatesAtBothTextSizes() async throws {
        let app = try makeApp()
        try configure(app, token: "fixture-recording-home")
        try await waitForSetupIdle(app)
        for large in [false, true] {
            try mount(JarvisRootView(viewModel: app).dynamicTypeSize(large ? .accessibility5 : .large),
                height: 844, contentSizeCategory: large ? .accessibilityExtraExtraExtraLarge : .large)
            for phase: JarvisRecordingSession.State? in [nil, .recording, .stopping, .retaining, .retained] {
                let label = phase?.rawValue ?? "idle"
                let message = phase == .retaining ? "Finishing upload…" : (phase == .retained ? "Saved to Audiovisual." : "Stopping recording…")
                // Exercise the actual controller-to-view-model seam, with no camera or voice started.
                app.audioSessionViewModel.handleRecordingStatus(.init(recordingID: nil,
                    capturing: phase == .recording, message: message, phase: phase))
                XCTAssertEqual(app.audioSessionViewModel.recordingPhase, phase)
                try await Task.sleep(nanoseconds: 120_000_000)
                try attachGalleryFrame(name: "recording-" + label + (large ? "-large" : "-normal"))
            }
        }
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
    }

    func testOriginalOpeningLoadingFailureAndRetry() async throws {
        let bytes = UIGraphicsImageRenderer(size: CGSize(width: 480, height: 320)).pngData { context in
            UIColor.systemTeal.setFill(); context.fill(CGRect(x: 0, y: 0, width: 480, height: 320))
        }
        wire.enableAudiovisualImages([bytes], previews: false)
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-original-loading")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 844)
        app.productTab = .audiovisual
        try await eventually("Metadata is available") { app.audiovisual?.audiovisualFiles.count == 1 }
        let model = try XCTUnwrap(app.audiovisual), file = try XCTUnwrap(app.audiovisual?.audiovisualFiles.first)
        wire.holdNextOriginal()
        let opening = Task { await model.open(file) }
        try await eventually("Explicit open is waiting for original bytes") { self.wire.hasHeldOriginal }
        XCTAssertTrue(model.isLoadingContent)
        try mount(JarvisFilePreviewView(model: model), height: 844)
        try await Task.sleep(nanoseconds: 150_000_000)
        try attachGalleryFrame(name: "original-loading")
        wire.releaseOriginal(status: 500)
        _ = await opening.value
        XCTAssertFalse(model.isLoadingContent)
        XCTAssertNotNil(model.contentError)
        try await Task.sleep(nanoseconds: 150_000_000)
        try attachGalleryFrame(name: "original-retry")
        let retried = await model.open(file)
        XCTAssertEqual(retried?.data, bytes)
        XCTAssertNil(model.contentError)
        for large in [false, true] {
            try mount(JarvisOriginalOpeningView(isVideo: true, progress: 0.42)
                .padding(22).background(JarvisColors.background)
                .dynamicTypeSize(large ? .accessibility5 : .large), height: 844,
                contentSizeCategory: large ? .accessibilityExtraExtraExtraLarge : .large)
            try await Task.sleep(nanoseconds: 100_000_000)
            try attachGalleryFrame(name: large ? "video-progress-large" : "video-progress-normal")
        }
    }

    private func captureAudiovisual(accessibility: Bool) async throws {
        let images = [UIColor.systemTeal, UIColor.systemOrange].map { color in
            UIGraphicsImageRenderer(size: CGSize(width: 480, height: 320)).pngData { context in
                color.withAlphaComponent(0.2).setFill()
                context.fill(CGRect(x: 0, y: 0, width: 480, height: 320))
                color.setFill()
                context.fill(CGRect(x: 0, y: 200, width: 480, height: 120))
                UIColor.white.setFill()
                context.cgContext.fillEllipse(in: CGRect(x: 320, y: 40, width: 90, height: 90))
            }
        }
        wire.enableAudiovisualImages(images)
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-audiovisual")
        try await waitForSetupIdle(app)
        let size: DynamicTypeSize = accessibility ? .accessibility5 : .large
        let category: UIContentSizeCategory = accessibility ? .accessibilityExtraExtraExtraLarge : .large
        try mount(JarvisRootView(viewModel: app).dynamicTypeSize(size), height: 844, contentSizeCategory: category)
        app.productTab = .audiovisual
        try await eventually("Actual gallery must prepare its verified data-only session and decode both thumbnails") {
            self.nativeTabBar()?.items?.count == 4 && app.audiovisual?.thumbnails.images.count == 2
                && app.audiovisual?.resourceContext?.visibleResources.count == 2
        }
        let model = try XCTUnwrap(app.audiovisual)
        XCTAssertEqual(model.audiovisualFiles.count, 2)
        XCTAssertFalse(wire.observedPaths.contains { $0.hasSuffix("/content") }, "Browsing only requests thumbnails")
        XCTAssertEqual(wire.observedPaths.filter { $0.hasSuffix("/thumbnail") }.count, 2)
        XCTAssertNil(model.errorMessage)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        try await captureRootTab(.audiovisual, title: "Audiovisual",
            label: accessibility ? "audiovisual-accessibility" : "audiovisual-ordinary", continuation: accessibility)
        app.productTab = .home
        try await eventually("Leaving the actual tab clears generic resource context") { model.resourceContext == nil }
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testMinimalHomeAtOrdinaryText() async throws {
        try await captureMinimalHome(accessibility: false)
    }

    func testMinimalHomeAtLargestAccessibilityText() async throws {
        try await captureMinimalHome(accessibility: true)
    }

    func testPluginsAccountReadWithoutVoiceAtOrdinaryText() async throws {
        try await captureAccountPlugins(accessibility: false)
    }

    func testPluginsAccountReadWithoutVoiceAtLargestAccessibilityText() async throws {
        try await captureAccountPlugins(accessibility: true)
    }

    func testPluginSaveStatesAtOrdinaryText() async throws {
        try await capturePluginSaveStates(accessibility: false)
    }

    func testPluginSaveStatesAtLargestAccessibilityText() async throws {
        try await capturePluginSaveStates(accessibility: true)
    }

    private func capturePluginSaveStates(accessibility: Bool) async throws {
        wire.enablePluginMutationFixture()
        let app = try makeApp()
        try configure(app, token: "fixture-plugin-save")
        try await waitForSetupIdle(app)
        let model = try XCTUnwrap(app.plugins)
        let size: DynamicTypeSize = accessibility ? .accessibility5 : .large
        let category: UIContentSizeCategory = accessibility ? .accessibilityExtraExtraExtraLarge : .large
        let label = accessibility ? "save-accessibility-largest" : "save-ordinary"
        try mount(JarvisRootView(viewModel: app).dynamicTypeSize(size), height: 844, contentSizeCategory: category)
        app.productTab = .plugins
        try await eventually("Native plugin model must accept explicit fixture permission") { model.preferencesAvailable }
        XCTAssertEqual(model.catalog?.items.count, 3)

        // Drive the actual model/HTTP/render path; this is not evidence of taps
        // or live server authority. The fixture holds the reply before commit.
        let saving = Task { await model.setEnabled(false, pluginID: "video_editor") }
        try await eventually("Fixture must hold the actual policy request") { self.wire.hasHeldPluginWrite }
        XCTAssertTrue(model.isSaving)
        XCTAssertEqual(model.catalog?.items.first?.desiredEnabled, true)
        XCTAssertEqual(model.pendingChange?.expectedRevision, 3)
        try await captureRootTab(.plugins, title: "Plugins", label: label + "-saving",
                                 continuation: accessibility, requiresOverflow: accessibility)

        wire.releasePluginWrite(loseReply: true)
        await saving.value
        XCTAssertFalse(model.isSaving)
        XCTAssertNotNil(model.pendingChange)
        XCTAssertNil(model.statusMessage)
        XCTAssertEqual(model.catalog?.items.first?.desiredEnabled, true,
                       "A lost reply must not replace the last confirmed choice")
        try await captureRootTab(.plugins, title: "Plugins", label: label + "-uncertain",
                                 continuation: accessibility, requiresOverflow: accessibility)

        await model.recheckPendingChange()
        XCTAssertEqual(wire.pluginWriteCount, 2, "Recheck must reuse the original request")
        XCTAssertNil(model.pendingChange)
        XCTAssertEqual(model.catalog?.policyRevision, 4)
        let card = try XCTUnwrap(model.catalog?.items.first)
        XCTAssertEqual(card.desiredEnabled, false)
        XCTAssertTrue(card.selectedDefault)
        XCTAssertNotNil(model.statusMessage)
        try await captureRootTab(.plugins, title: "Plugins", label: label + "-confirmed",
                                 continuation: accessibility, requiresOverflow: accessibility)
        XCTAssertNil(app.productFallbackSessionID)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertFalse(wire.observedPaths.contains("/sessions/start"))
        XCTAssertFalse(wire.observedPaths.contains("/client-context"))
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    private func captureAccountPlugins(accessibility: Bool) async throws {
        let app = try makeApp()
        try configure(app, token: "fixture-plugins-A")
        try await waitForSetupIdle(app)
        let model = try XCTUnwrap(app.plugins)
        let size: DynamicTypeSize = accessibility ? .accessibility5 : .large
        let category: UIContentSizeCategory = accessibility ? .accessibilityExtraExtraExtraLarge : .large
        try mount(JarvisRootView(viewModel: app).dynamicTypeSize(size), height: 844, contentSizeCategory: category)
        app.productTab = .plugins
        try await eventually("Plugins must load through the mounted account route") { model.catalog?.policyRevision == 3 }
        XCTAssertFalse(model.preferencesAvailable, "No mutation control before enforcement integration")
        let card = try XCTUnwrap(model.catalog?.items.first)
        XCTAssertEqual(card.desiredEnabled, false)
        XCTAssertTrue(card.selectedDefault, "Disabled preference remains remembered")
        XCTAssertNil(app.productFallbackSessionID)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertNil(app.workspace)
        XCTAssertFalse(wire.observedPaths.contains("/sessions/start"))
        XCTAssertFalse(wire.observedPaths.contains("/client-context"))
        XCTAssertTrue(wire.ackStates.isEmpty)
        try await captureRootTab(.plugins, title: "Plugins",
            label: accessibility ? "account-accessibility-largest" : "account-ordinary", continuation: accessibility)
        try configure(app, token: "fixture-plugins-B")
        XCTAssertNil(model.catalog, "Account replacement clears the previous catalog")
        XCTAssertFalse(app.plugins === model)
        try await waitForSetupIdle(app)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    /// Home-only visual follow-up to Dennis's simplified design. No session is
    /// started, and the superseded note-list flow is not visited or qualified.

    func testPreferencesAtOrdinaryText() async throws {
        try await captureAccountProducts(accessibility: false)
    }

    func testRemindersRejectOldAccountAndOlderRefreshResponses() async throws {
        let app = try makeApp()
        try configure(app, token: "fixture-reminders-A")
        try await waitForSetupIdle(app)
        wire.setReminderSnapshot(1)
        await app.refreshReminderState().value
        XCTAssertEqual(app.activeReminders.map(\.title), ["Reminder 1"])
        XCTAssertEqual(app.reminderEvents.map(\.message), ["Event 1"])

        wire.holdNextReminderRead("/reminders/active")
        let olderRead = app.refreshReminderState()
        try await eventually("Older reminder read must be held") { self.wire.hasHeldReminderRead("/reminders/active") }
        wire.setReminderSnapshot(2)
        await app.refreshReminderState().value
        wire.releaseReminderRead("/reminders/active")
        await olderRead.value
        XCTAssertEqual(app.activeReminders.map(\.title), ["Reminder 2"], "Older refresh cannot undo newer content")
        XCTAssertEqual(app.reminderEvents.map(\.message), ["Event 2"])

        wire.holdNextReminderRead("/reminders/active")
        let oldAccountRead = app.refreshReminderState()
        try await eventually("Old-account reminder read must be held") { self.wire.hasHeldReminderRead("/reminders/active") }
        try configure(app, token: "fixture-reminders-B")
        XCTAssertTrue(app.activeReminders.isEmpty)
        XCTAssertTrue(app.reminderEvents.isEmpty)
        XCTAssertNil(app.reminderLastRefreshedAt)
        XCTAssertNil(app.reminderRefreshError)
        try await waitForSetupIdle(app)
        wire.setReminderSnapshot(3)
        await app.refreshReminderState().value
        let currentStamp = app.reminderLastRefreshedAt
        wire.releaseReminderRead("/reminders/active")
        await oldAccountRead.value
        XCTAssertEqual(app.activeReminders.map(\.title), ["Reminder 3"], "Previous-account success must remain hidden")
        XCTAssertEqual(app.reminderEvents.map(\.message), ["Event 3"])
        XCTAssertEqual(app.reminderLastRefreshedAt, currentStamp)

        wire.holdNextReminderRead("/reminders/active")
        let failedOlderRead = app.refreshReminderState()
        try await eventually("Older failure must be held") { self.wire.hasHeldReminderRead("/reminders/active") }
        await app.refreshReminderState().value
        wire.releaseReminderRead("/reminders/active", status: 403)
        await failedOlderRead.value
        XCTAssertNil(app.reminderRefreshError, "A retired read cannot overwrite the latest success with an error")

        wire.setReminderSnapshot(4)
        wire.holdNextReminderRead("/reminders/events")
        let resetRead = app.refreshReminderState()
        try await eventually("The second reminder read must be held") { self.wire.hasHeldReminderRead("/reminders/events") }
        XCTAssertEqual(app.activeReminders.map(\.title), ["Reminder 3"], "Publish reminders/events together after both reads")
        app.resetConfiguration()
        wire.releaseReminderRead("/reminders/events")
        await resetRead.value
        XCTAssertNil(app.configuration)
        XCTAssertTrue(app.activeReminders.isEmpty)
        XCTAssertTrue(app.reminderEvents.isEmpty)
        XCTAssertNil(app.reminderLastRefreshedAt)
        XCTAssertNil(app.reminderRefreshError)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testAccountProductsCatchUpAfterDelayedOwnerVerification() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-delayed-account-verification")
        try await waitForSetupIdle(app)
        let preferences = try XCTUnwrap(app.preferences)
        await preferences.refresh()
        XCTAssertEqual(preferences.savedProfile?.revision, 1)
        preferences.draft = "Keep this unsaved draft"
        wire.holdContext(wire.rootSessionID)
        app.showSavedCopies()
        try await eventually("Owner verification must still be pending") { self.wire.hasHeldContext(self.wire.rootSessionID) }

        wire.correct(to: 2)
        let profileEvent = wire.commitProfile(text: "Saved while verification was pending")
        app.audioSessionViewModel.onBehaviorProfileCommitted?(wire.rootSessionID, profileEvent)
        XCTAssertEqual(preferences.savedProfile?.revision, 1)

        wire.releaseContext(wire.rootSessionID)
        try await eventually("Verification must recover commits missed during startup without a new hint") {
            preferences.savedProfile?.revision == 2
        }
        XCTAssertEqual(preferences.draft, "Keep this unsaved draft")
        XCTAssertEqual(preferences.baseRevision, 1)
        XCTAssertTrue(preferences.needsReview)
        XCTAssertFalse(preferences.canSave)
        XCTAssertEqual(app.productTab, .home)
        XCTAssertEqual(app.state, .ready)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testSavedCopiesPrepareWithoutVoiceAndReuseVerifiedJournal() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-cold-copies")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 844)
        let state = app.state, transcript = app.transcriptPreview, visual = app.visualStatus
        XCTAssertNil(app.workspace)
        wire.holdContext(wire.rootSessionID)
        app.showSavedCopies()
        app.showSavedCopies()
        try await eventually("Copy entry must await independent account verification") { self.wire.hasHeldContext(self.wire.rootSessionID) }
        XCTAssertTrue(app.isPreparingSavedCopies)
        XCTAssertEqual(app.presentedSheet, .offlineCopies)
        XCTAssertEqual(wire.observedPaths.filter { $0 == "/sessions/start" }.count, 1, "Repeated navigation coalesces preparation")
        XCTAssertFalse(app.workspace?.canSaveOffline ?? true)
        wire.releaseContext(wire.rootSessionID)
        try await eventually("Verified context unlocks the copy library") { app.workspace?.canSaveOffline == true && !app.isPreparingSavedCopies }
        XCTAssertNil(app.savedCopiesPreparationError)
        XCTAssertEqual(app.state, state)
        XCTAssertEqual(app.transcriptPreview, transcript)
        XCTAssertEqual(app.visualStatus, visual)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        let sessions = wire.observedPaths.filter { $0 == "/sessions/start" }.count
        let contexts = wire.observedPaths.filter { $0 == "/client-context" }.count
        app.showSavedCopies()
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(wire.observedPaths.filter { $0 == "/sessions/start" }.count, sessions)
        XCTAssertEqual(wire.observedPaths.filter { $0 == "/client-context" }.count, contexts,
                       "Opening an already verified local library must not require another network verification")
        try await Task.sleep(nanoseconds: 300_000_000)
        try attachGalleryFrame(name: "saved-copies-verified-empty")
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testSavedFilesPrepareWithoutVoiceAndRetireOnAccountChange() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-cold-files")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 844)
        let state = app.state, transcript = app.transcriptPreview, visual = app.visualStatus
        wire.holdContext(wire.rootSessionID)
        app.showSavedFiles()
        try await eventually("Files must await independently verified account context") { self.wire.hasHeldContext(self.wire.rootSessionID) }
        XCTAssertNil(app.savedFiles)
        wire.releaseContext(wire.rootSessionID)
        try await eventually("Scoped library is mounted and loaded") { app.savedFiles != nil && self.wire.observedPaths.contains("/files") && !app.isPreparingSavedCopies }
        let files = try XCTUnwrap(app.savedFiles)
        XCTAssertNil(files.errorMessage)
        XCTAssertEqual(app.state, state)
        XCTAssertEqual(app.transcriptPreview, transcript)
        XCTAssertEqual(app.visualStatus, visual)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        try await Task.sleep(nanoseconds: 300_000_000)
        try attachGalleryFrame(name: "saved-files-data-only-empty")
        app.resetConfiguration()
        XCTAssertNil(app.savedFiles)
        let reads = wire.observedPaths.filter { $0 == "/files" }.count
        await files.reload()
        XCTAssertEqual(wire.observedPaths.filter { $0 == "/files" }.count, reads, "Retired library cannot read the previous account")
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testTaskFilePreviewVerifiesAccountAndLoadsOnlyRequestedFileWithoutVoice() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-task-file")
        try await waitForSetupIdle(app)
        wire.holdContext(wire.rootSessionID)
        try mount(JarvisFilesAccessView(viewModel: app, fileID: wire.libraryFileID), height: 844)
        try await eventually("Task file must verify account before loading") { self.wire.hasHeldContext(self.wire.rootSessionID) }
        XCTAssertNil(app.savedFiles)
        XCTAssertFalse(wire.observedPaths.contains { $0.hasPrefix("/files") })
        wire.releaseContext(wire.rootSessionID)
        try await eventually("Requested retained file loads in its preview") {
            app.savedFiles?.selectedFile?.id == self.wire.libraryFileID.uuidString.lowercased()
                && app.savedFiles?.content == Data(self.wire.libraryContent.utf8)
        }
        XCTAssertFalse(wire.observedPaths.contains("/files"), "A task artifact opens directly without browsing the library")
        XCTAssertNil(app.savedFiles?.contentError)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        try await Task.sleep(nanoseconds: 300_000_000)
        try attachGalleryFrame(name: "task-file-authorized-direct-preview")
        app.closeSavedFiles()
        XCTAssertNil(app.savedFiles)
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testSavedFilesRejectLateVerificationAfterConfigurationChange() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-files-A")
        try await waitForSetupIdle(app)
        wire.holdContext(wire.rootSessionID)
        app.prepareSavedFiles()
        try await eventually("Library verification reaches transport") { self.wire.hasHeldContext(self.wire.rootSessionID) }
        app.resetConfiguration()
        try configure(app, token: "fixture-files-B")
        wire.releaseContext(wire.rootSessionID)
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertNil(app.savedFiles)
        XCTAssertNil(app.workspace)
        XCTAssertFalse(app.isPreparingSavedCopies)
        XCTAssertFalse(wire.observedPaths.contains("/files"))
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testSavedCopiesRejectLateVerificationAfterConfigurationChange() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-copies-A")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 844)
        wire.holdContext(wire.rootSessionID)
        app.showSavedCopies()
        try await eventually("Old copy verification must reach transport") { self.wire.hasHeldContext(self.wire.rootSessionID) }
        let old = try XCTUnwrap(app.workspace)
        app.resetConfiguration()
        try configure(app, token: "fixture-copies-B")
        wire.releaseContext(wire.rootSessionID)
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertNil(app.workspace)
        XCTAssertNil(app.productFallbackSessionID)
        XCTAssertNil(app.presentedSheet)
        XCTAssertFalse(app.isPreparingSavedCopies)
        XCTAssertNil(app.savedCopiesPreparationError)
        XCTAssertFalse(old.canSaveOffline)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testSavedCopiesUnavailableAccountShowsVerificationNeeded() async throws {
        let app = try makeApp(allowsFileBytes: true)
        try configure(app, token: "fixture-copies-unavailable")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 844)
        wire.setSessionStartStatus(503)
        app.showSavedCopies()
        try await eventually("Unverified copies must show a recoverable error") { !app.isPreparingSavedCopies && app.savedCopiesPreparationError != nil }
        XCTAssertNil(app.workspace)
        XCTAssertEqual(app.state, .ready)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        try await Task.sleep(nanoseconds: 300_000_000)
        try attachGalleryFrame(name: "saved-copies-verification-needed")
        wire.setSessionStartStatus(200)
        app.prepareSavedCopies()
        try await eventually("Explicit retry can verify the same configured account") { app.workspace?.canSaveOffline == true && !app.isPreparingSavedCopies }
        XCTAssertNil(app.savedCopiesPreparationError)
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    func testPreferencesAtLargestAccessibilityText() async throws {
        try await captureAccountProducts(accessibility: true)
    }

    /// Synthetic profile/transport component: real Root, keyboard and native UI
    /// controls, with fail-closed in-process URLProtocol instead of any backend.
    func testPreferencesEditSaveWithSeparateNativeDriver() async throws {
        guard let path = ProcessInfo.processInfo.environment["JARVIS_PREFERENCES_UI_COMPONENT_MANIFEST"] else {
            throw XCTSkip("Requires the separately allocated native component operator")
        }
        continueAfterFailure = false
        executionTimeAllowance = 120
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any])
        let run = try XCTUnwrap(manifest["run_id"] as? String)
        let stage = URL(fileURLWithPath: try XCTUnwrap(manifest["stage_path"] as? String))
        let directory = stage.deletingLastPathComponent()
        let draft = try XCTUnwrap(manifest["draft"] as? String)
        let dirty = try XCTUnwrap(manifest["pushed_while_dirty"] as? String)
        let clean = try XCTUnwrap(manifest["pushed_while_clean"] as? String)
        guard UUID(uuidString: run)?.uuidString == run,
              manifest["exercise_save_failure"] as? Bool == true,
              directory.lastPathComponent == "phone-preferences-" + run,
              !FileManager.default.fileExists(atPath: stage.path) else {
            throw FixtureFailure("Fresh run-bound component stages required")
        }
        wire.prepareEmptyProfileForUIComponent()
        let app = try makeApp()
        try configure(app, token: "synthetic-preferences-ui-component")
        try await waitForSetupIdle(app)
        try mount(JarvisRootView(viewModel: app), height: 874, contentSizeCategory: .large)
        if let window, let scene = window.windowScene {
            window.frame = scene.coordinateSpace.bounds
            window.layoutIfNeeded()
        }
        // This request is intercepted by the already-proved fail-closed fixture;
        // it creates no backend session and starts no voice or provider.
        app.startSession()
        try await eventually("Synthetic owner/profile must be ready") {
            app.workspace?.journalReady == true && app.preferences?.savedProfile?.revision == 0
        }
        let preferences = try XCTUnwrap(app.preferences)
        func post(_ phase: String) async throws {
            let at = Date().timeIntervalSince1970
            try JSONSerialization.data(withJSONObject: ["run_id": run, "phase": phase, "at": at])
                .write(to: stage, options: .atomic)
            if phase == "finished-awaiting-review" { return }
            try await eventually("Component UI progress for " + phase, timeout: 20) {
                guard let data = try? Data(contentsOf: directory.appendingPathComponent("ui-progress.json")),
                      let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
                return value["run_id"] as? String == run && value["phase"] as? String == phase
                    && (value["at"] as? Double ?? 0) >= at
            }
        }
        try await post("open-preferences")
        try await post("edit-draft")
        XCTAssertEqual(preferences.draft, draft)
        XCTAssertTrue(try XCTUnwrap(preferencesTextEditor()).isFirstResponder,
                      "The actual editor must still have keyboard focus before conflict review")
        // Explicit synthetic state supply for the UI component, not transport proof.
        await preferences.committed(wire.commitProfile(text: dirty))
        try await post("review-update")
        wire.failNextProfileWrite()
        try await post("save-failure")
        XCTAssertEqual(wire.profileWriteAttemptCount, 1)
        XCTAssertEqual(wire.profileWriteCount, 0)
        XCTAssertEqual(preferences.draft, draft)
        XCTAssertEqual(preferences.baseRevision, 1)
        XCTAssertNotNil(preferences.errorMessage)
        XCTAssertTrue(try XCTUnwrap(preferencesTextEditor()).isFirstResponder)
        wire.holdNextProfileWrite()
        try await post("save-draft")
        XCTAssertTrue(preferences.isSaving, "Editing must remain open while the HTTP confirmation is withheld")
        XCTAssertTrue(wire.hasHeldProfileWrite)
        XCTAssertTrue(try XCTUnwrap(preferencesTextEditor()).isFirstResponder)
        let duplicateSave = await preferences.save()
        XCTAssertFalse(duplicateSave, "The model must reject a duplicate in-flight save")
        XCTAssertEqual(wire.profileWriteAttemptCount, 2)
        wire.releaseProfileWrite()
        try await eventually("Real UI Save must reach the intercepted profile endpoint") {
            preferences.savedProfile?.revision == 2 && self.wire.profileWriteCount == 1 && !preferences.isDirty
        }
        XCTAssertEqual(preferences.savedProfile?.behaviorInstructions, draft)
        try await post("verify-saved")
        XCTAssertNil(preferencesTextEditor(), "Confirmed save must leave read-only text instead of an editor")
        await preferences.committed(wire.commitProfile(text: clean))
        try await post("verify-clean-push")
        try await post("finished-awaiting-review")
        try await eventually("Native UI operator must finish its ordered stages", timeout: 10) {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("native-ui-driver.json")),
                  let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
            return value["run_id"] as? String == run && value["completed_ui_stages"] as? Bool == true
        }
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
        try JSONSerialization.data(withJSONObject: ["run_id": run, "ui_component_passed": true,
            "synthetic_profile_and_transport": true, "provider_session_started": false,
            "failed_save_retained_draft": true, "confirmation_withheld_until_ui_checked": true,
            "profile_write_attempt_count": wire.profileWriteAttemptCount,
            "profile_write_count": wire.profileWriteCount, "profile_revision": preferences.savedProfile!.revision])
            .write(to: directory.appendingPathComponent("synthetic-component-result.json"), options: .atomic)
    }

    private func captureAccountProducts(accessibility: Bool) async throws {
        wire.includeTableInNote()
        let app = try makeApp()
        try configure(app, token: "fixture-account-products")
        try await waitForSetupIdle(app)
        let size: DynamicTypeSize = accessibility ? .accessibility5 : .large
        let category: UIContentSizeCategory = accessibility ? .accessibilityExtraExtraExtraLarge : .large
        let label = accessibility ? "account-accessibility-largest" : "account-ordinary"
        try mount(JarvisRootView(viewModel: app).dynamicTypeSize(size), height: 844, contentSizeCategory: category)
        app.startSession()
        try await eventually("Existing context verifies the profile owner") { app.workspace?.journalReady == true }
        app.presentedSheet = .preferences
        let preferences = try XCTUnwrap(app.preferences)
        try await eventually("Dedicated Preferences must read persistent backend profile") { preferences.savedProfile?.revision == 1 }
        try await Task.sleep(nanoseconds: 300_000_000)
        let presented = try XCTUnwrap(window?.rootViewController?.presentedViewController)
        XCTAssertNil(preferencesTextEditor(), "Preferences opens in read-only mode until Edit is tapped")
        XCTAssertEqual(presented.traitCollection.preferredContentSizeCategory, category,
                       "The presented sheet must inherit the requested text size")
        try attachGalleryFrame(name: label + "-preferences")
        if accessibility { try await capturePreferencesContinuation(in: presented.view, name: label + "-preferences-scrolled") }
        preferences.draft = "Use short answers and explain unfamiliar terms."
        let profileEvent = wire.commitProfile(text: "Give me a little more detail.")
        app.audioSessionViewModel.onBehaviorProfileCommitted?(wire.rootSessionID, profileEvent)
        try await eventually("External preference update must preserve the open draft") {
            preferences.savedProfile?.revision == 2 && preferences.needsReview
        }
        XCTAssertEqual(preferences.draft, "Use short answers and explain unfamiliar terms.")
        XCTAssertEqual(preferences.baseRevision, 1)
        XCTAssertFalse(preferences.canSave)
        try await Task.sleep(nanoseconds: 100_000_000)
        try attachGalleryFrame(name: label + "-preferences-draft-preserved")
        if accessibility { try await capturePreferencesContinuation(in: presented.view, name: label + "-preferences-conflict-scrolled") }
        // Programmatic review/save checks actual AppVM/client composition, not a touch interaction.
        preferences.keepDraftAfterReview()
        await preferences.save()
        XCTAssertEqual(preferences.savedProfile?.revision, 3)
        XCTAssertFalse(preferences.isDirty)
        XCTAssertEqual(wire.profileWriteCount, 1)
        app.presentedSheet = nil
        try await Task.sleep(nanoseconds: 350_000_000)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertTrue(wire.ackStates.isEmpty)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    private func captureMinimalHome(accessibility: Bool) async throws {
        let app = try makeApp()
        try configure(app, token: "fixture-home-only")
        try await waitForSetupIdle(app)
        let size: DynamicTypeSize = accessibility ? .accessibility5 : .large
        let category: UIContentSizeCategory = accessibility ? .accessibilityExtraExtraExtraLarge : .large
        let label = accessibility ? "minimal-home-accessibility-largest" : "minimal-home-ordinary"
        try mount(JarvisRootView(viewModel: app).dynamicTypeSize(size), height: 844,
                  contentSizeCategory: category)
        try await eventually("Minimal Home must be selected in the real Root") {
            self.nativeTabBar()?.selectedItem?.title == "Home"
        }
        try await Task.sleep(nanoseconds: 250_000_000)
        try attachGalleryFrame(name: label)
        // Programmatic sheet presentation supplies screenshot coverage only.
        // It does not claim that the new header button has been tapped.
        app.presentedSheet = .reminders
        try await Task.sleep(nanoseconds: 350_000_000)
        try attachGalleryFrame(name: label + "-reminders")
        app.presentedSheet = nil
        try await Task.sleep(nanoseconds: 350_000_000)
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertNil(app.productFallbackSessionID)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    private func captureRootDestinations(accessibility: Bool) async throws {
        wire.includeTableInNote()
        let app = try makeApp()
        try configure(app, token: "fixture-gallery-only")
        try await waitForSetupIdle(app)
        let size: DynamicTypeSize = accessibility ? .accessibility5 : .large
        let category: UIContentSizeCategory = accessibility ? .accessibilityExtraExtraExtraLarge : .large
        let label = accessibility ? "accessibility-largest" : "ordinary"
        try mount(JarvisRootView(viewModel: app).dynamicTypeSize(size), height: 844, contentSizeCategory: category)
        try await eventually("The actual Root must expose all four native tabs") {
            self.nativeTabBar()?.items?.count == 5
        }
        XCTAssertEqual(window?.rootViewController?.traitCollection.preferredContentSizeCategory, category)
        try await captureRootTab(.home, title: "Home", label: label, continuation: accessibility)

        // The existing data-only session method supplies Root's real fallback
        // session identity. No private state assignment, microphone, audio
        // startup, websocket, or replacement product shell is used.
        app.startSession()
        try await eventually("Root must attach the intercepted session and verified context itself") {
            app.productFallbackSessionID == self.wire.rootSessionID.uuidString && app.workspace?.journalReady == true
        }
        let model = try XCTUnwrap(app.workspace)
        app.productTab = .workspace
        try await eventually("Root must make the selected Workspace readable") { model.canRead }
        await model.refresh()
        try await captureRootTab(.workspace, title: "Workspace", label: label,
                                 continuation: accessibility, requiresOverflow: accessibility)
        try await captureRootTab(.audiovisual, title: "Audiovisual", label: label, continuation: accessibility)
        try await captureRootTab(.plugins, title: "Plugins", label: label, continuation: accessibility)
        try await eventually("Root must retire Workspace visibility on the Plugins tab") { !model.canRead }
        XCTAssertTrue(wire.ackStates.isEmpty, "Browsing and tab changes must not acknowledge a pending offer")
        XCTAssertFalse(app.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(app.audioSessionViewModel.activeSession)
        XCTAssertTrue(wire.unexpectedRequests.isEmpty, wire.unexpectedRequests.joined(separator: ", "))
    }

    private func captureRootTab(_ tab: JarvisProductTab, title: String, label: String, continuation: Bool,
                                requiresOverflow: Bool = false) async throws {
        let app = try XCTUnwrap(app)
        app.productTab = tab
        try await eventually("Native tab selection must show " + title) { self.nativeTabBar()?.selectedItem?.title == title }
        // A bounded layout settle is only for screenshot collection. The saved
        // pixels still require visual inspection; it is not a style assertion.
        try await Task.sleep(nanoseconds: 250_000_000)
        try attachGalleryFrame(name: label + "-" + title.lowercased() + "-top")
        if continuation {
            let candidate = primaryScrollView()
            if requiresOverflow { XCTAssertNotNil(candidate, "The long Workspace note must have a reachable scroll view") }
            guard let scroll = candidate else { return }
            let limit = max(-scroll.adjustedContentInset.top,
                            scroll.contentSize.height - scroll.bounds.height + scroll.adjustedContentInset.bottom)
            let next = min(limit, scroll.contentOffset.y + scroll.bounds.height * 0.7)
            if requiresOverflow { XCTAssertGreaterThan(next, scroll.contentOffset.y + 20, "The long note must support vertical scrolling") }
            if next > scroll.contentOffset.y + 20 {
                let previous = scroll.contentOffset
                scroll.setContentOffset(CGPoint(x: previous.x, y: next), animated: false)
                try await Task.sleep(nanoseconds: 50_000_000)
                try attachGalleryFrame(name: label + "-" + title.lowercased() + "-scrolled")
                if tab == .workspace && requiresOverflow {
                    try await captureRightmostTable(parent: scroll, label: label)
                }
                scroll.setContentOffset(previous, animated: false)
            }
        }
    }

    private func captureRightmostTable(parent: UIScrollView, label: String) async throws {
        let window = try XCTUnwrap(window)
        var candidates: [UIScrollView] = []
        func collect(_ view: UIView) {
            guard !view.isHidden, view.alpha > 0 else { return }
            if let scroll = view as? UIScrollView, scroll !== parent,
               scroll.bounds.width > 100, scroll.bounds.height > 40,
               scroll.contentSize.width > scroll.bounds.width + 20,
               scroll.window === window,
               scroll.convert(scroll.bounds, to: window).intersects(window.bounds) {
                candidates.append(scroll)
            }
            view.subviews.forEach(collect)
        }
        collect(parent)
        XCTAssertEqual(candidates.count, 1, "This fixture has one visible horizontally overflowing table")
        let table = try XCTUnwrap(candidates.first)
        XCTAssertTrue(table.isScrollEnabled)
        XCTAssertTrue(table.panGestureRecognizer.isEnabled)
        let previous = table.contentOffset, parentOffset = parent.contentOffset
        defer { table.setContentOffset(previous, animated: false) }
        let right = table.contentSize.width - table.bounds.width + table.adjustedContentInset.right
        XCTAssertGreaterThan(right, previous.x + 20)
        table.setContentOffset(CGPoint(x: right, y: previous.y), animated: false)
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(table.contentOffset.x, right, accuracy: 1)
        XCTAssertEqual(parent.contentOffset.y, parentOffset.y, accuracy: 1,
                       "Horizontal table movement must preserve the document's vertical position")
        try attachGalleryFrame(name: label + "-workspace-table-rightmost")
        let evidence = XCTAttachment(string: "tableWidth=\(table.bounds.width); contentWidth=\(table.contentSize.width); beforeX=\(previous.x); afterX=\(table.contentOffset.x); parentY=\(parentOffset.y); expected readable right column: Reading, 42.5, Unknown. Native offset movement only; actual pixels require review and this is not touch-gesture qualification.")
        evidence.name = "root-" + label + "-table-scroll-geometry"
        evidence.lifetime = .keepAlways
        add(evidence)
    }

    private func attachGalleryFrame(name: String) throws {
        let window = try XCTUnwrap(window)
        window.layoutIfNeeded()
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        let attachment = XCTAttachment(image: image)
        attachment.name = "root-" + name + "-requires-visual-review"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func nativeTabBar() -> UITabBar? {
        guard let window else { return nil }
        func find(_ view: UIView) -> UITabBar? {
            guard !view.isHidden, view.alpha > 0 else { return nil }
            if let bar = view as? UITabBar { return bar }
            return view.subviews.lazy.compactMap(find).first
        }
        return find(window)
    }

    private func makeApp(allowsFileBytes: Bool = false) throws -> JarvisAppViewModel {
        let name = "jarvis.product.fixture." + UUID().uuidString
        defaultsName = name
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        let store = JarvisSessionStore(defaults: defaults, keychain: JarvisKeychain(service: name))
        fixtureStore = store
        let app = JarvisAppViewModel(store: store, audioSessionViewModel: JarvisAudioSessionViewModel(),
            workspaceJournalFactory: { context in
                try JarvisWorkspaceLocalStore(ownerKey: JarvisWorkspaceLocalStore.ownerKey(for: context),
                    root: allowsFileBytes ? FileManager.default.temporaryDirectory.appendingPathComponent(name) : nil,
                    allowsFileBytes: allowsFileBytes)
            })
        self.app = app
        return app
    }

    private func readyApp(session: UUID) async throws -> JarvisAppViewModel {
        let app = try makeApp()
        try configure(app, token: "fixture-only")
        app.synchronizeProductSession(session.uuidString)
        _ = try XCTUnwrap(app.workspace, "Valid fixture configuration must create a Workspace")
        try await eventually("Verified context must attach before fresh presentation") { app.workspace?.journalReady == true }
        try await waitForSetupIdle(app)
        app.updateProductScene(foreground: true)
        app.workspace?.setVisibility(true, foreground: true)
        return app
    }

    private func configure(_ app: JarvisAppViewModel, token: String) throws {
        app.saveSetup(backendBaseURL: wire.baseURL.absoluteString, staticBearerToken: token)
        guard let configuration = app.configuration else {
            throw FixtureFailure("Native fixture setup failed before HTTP: " + (app.setupErrorMessage ?? "no setup error was published"))
        }
        XCTAssertEqual(configuration.backendBaseURL, wire.baseURL)
    }

    private func waitForSetupIdle(_ app: JarvisAppViewModel) async throws {
        XCTAssertNil(app.pushNotifications, "Remote push is unsupported; setup must not activate it")
    }

    private func journalFor(_ session: UUID) throws -> JarvisWorkspaceLocalStore {
        let context = wire.context(for: session)
        return try JarvisWorkspaceLocalStore(ownerKey: JarvisWorkspaceLocalStore.ownerKey(for: context), allowsFileBytes: false)
    }

    private func eventually(_ message: String, timeout: TimeInterval = 4,
                            _ predicate: @MainActor () async throws -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if try await predicate() { return }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let contextMessage = app?.workspace?.deliveryMessage ?? "no delivery message"
        throw FixtureFailure(message + "; " + contextMessage + "; observed paths: " + wire.observedPaths.joined(separator: ", "))
    }

    private func mount<Content: View>(_ content: Content, height: CGFloat,
                                     contentSizeCategory: UIContentSizeCategory? = nil) throws {
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .first(where: { $0.activationState == .foregroundActive }))
        previousKeyWindow = scene.windows.first(where: \.isKeyWindow)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 390, height: height)
        // A presented sheet can sit outside the presenting host's trait subtree.
        // Scope the override to this disposable window so every presented view
        // receives it without changing the simulator's global text settings.
        if let contentSizeCategory { window.traitOverrides.preferredContentSizeCategory = contentSizeCategory }
        // A raw UIHostingController has no SwiftUI App/WindowGroup to inject
        // scenePhase. Bridge the actual scene selected above into this hosted
        // snapshot. This tests Root with an active scene input, not OS-driven
        // background/foreground delivery or an application lifecycle transition.
        let phase: ScenePhase = scene.activationState == .foregroundActive ? .active : .inactive
        let host = UIHostingController(rootView: content.preferredColorScheme(.light).environment(\.scenePhase, phase))
        if let contentSizeCategory { host.traitOverrides.preferredContentSizeCategory = contentSizeCategory }
        window.rootViewController = host
        self.window = window
        window.makeKeyAndVisible()
        window.layoutIfNeeded()
        let capture = { @MainActor [weak self] in
            guard let self else { return }
            guard let window = self.window else { return }
            let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: false)
            }
            self.displayedFrame = DisplayedFrame(image: image, requestID: self.wire.requestID,
                artifactID: self.app?.workspace?.mediaPresentation?.artifactID,
                versionID: self.app?.workspace?.mediaPresentation?.versionID,
                productTab: self.app?.productTab, nativeTabTitle: self.nativeTabBar()?.selectedItem?.title,
                observedAt: Date().timeIntervalSince1970)
        }
        wire.observeDisplayed {
            // URLProtocol can call on a loading thread. Release all fixture locks
            // before entering the main actor, and never synchronously wait on
            // main from main. Save this observation before recording/responding
            // to displayed. Later scrolling cannot change the saved image.
            if Thread.isMainThread {
                MainActor.assumeIsolated { capture() }
            } else {
                DispatchQueue.main.sync { MainActor.assumeIsolated { capture() } }
            }
        }
    }

    private func primaryScrollView() -> UIScrollView? {
        guard let window else { return nil }
        func find(_ view: UIView) -> UIScrollView? {
            guard !view.isHidden, view.alpha > 0 else { return nil }
            // Hosting wrappers may have zero-sized bounds while their children
            // draw normally. Check viewport geometry only on scroll candidates.
            if let scroll = view as? UIScrollView, scroll.bounds.height > 100,
               scroll.window != nil, scroll.convert(scroll.bounds, to: window).intersects(window.bounds) { return scroll }
            return view.subviews.lazy.compactMap(find).first
        }
        return find(window)
    }

    private func preferencesTextEditor() -> UITextView? {
        guard let window else { return nil }
        func find(_ view: UIView) -> UITextView? {
            guard !view.isHidden, view.alpha > 0 else { return nil }
            if let editor = view as? UITextView, editor.isEditable, editor.window === window {
                return editor
            }
            return view.subviews.lazy.compactMap(find).first
        }
        return find(window)
    }

    private func capturePreferencesContinuation(in presented: UIView, name: String) async throws {
        func find(_ view: UIView) -> UIScrollView? {
            if let scroll = view as? UIScrollView, scroll.bounds.height > 100 { return scroll }
            return view.subviews.lazy.compactMap(find).first
        }
        let scroll = try XCTUnwrap(find(presented), "Preferences must remain scrollable at large text sizes")
        let top = scroll.contentOffset
        let bottom = max(-scroll.adjustedContentInset.top,
                         scroll.contentSize.height - scroll.bounds.height + scroll.adjustedContentInset.bottom)
        XCTAssertGreaterThan(bottom, top.y + 20, "The largest-text form should exercise real overflow")
        scroll.setContentOffset(CGPoint(x: top.x, y: bottom), animated: false)
        try await Task.sleep(nanoseconds: 100_000_000)
        try attachGalleryFrame(name: name)
        scroll.setContentOffset(top, animated: false)
        try await Task.sleep(nanoseconds: 100_000_000)
    }

    private func fixtureTasks(host: String) async -> [URLSessionTask] {
        let tasks: [URLSessionTask] = await withCheckedContinuation { continuation in
            URLSession.shared.getAllTasks { continuation.resume(returning: $0) }
        }
        return tasks.filter { $0.originalRequest?.url?.host == host || $0.currentRequest?.url?.host == host }
    }

    private func attachBoundaryFrame() throws {
        let frame = try XCTUnwrap(displayedFrame, "The intercepted displayed request must capture a frame before responding")
        XCTAssertEqual(frame.requestID, wire.requestID)
        XCTAssertEqual(frame.artifactID, wire.noteID)
        XCTAssertEqual(frame.versionID, "fixture-v1")
        let image = XCTAttachment(image: frame.image)
        image.name = "displayed-boundary-requires-visual-review"
        image.lifetime = .keepAlways
        add(image)
        let identity = XCTAttachment(string: "request=\(frame.requestID); artifact=\(wire.noteID); version=fixture-v1; capturedAt=\(frame.observedAt); productTab=\(String(describing: frame.productTab)); nativeTab=\(frame.nativeTabTitle ?? "none"); expected visible body: Saved fixture value 1. Visual review is required; XCTest success is not a visibility assertion. Native selection at this exact boundary is recorded separately from a settled image.")
        identity.name = "displayed-boundary-identity"
        identity.lifetime = .keepAlways
        add(identity)
    }

}

/// Local native-renderer fixture. It never uses HTTP, sensors, audio or a provider.
@MainActor
private final class NativeMediaFixture: JarvisWorkspaceAccess, JarvisWorkspaceMediaAccess {
    let offer = JarvisWorkspaceRequest(requestID: UUID(), action: .open, artifactKind: "file", artifactID: UUID(),
        versionID: "native-synthetic-video-v1", format: nil, contentReady: true, fileID: nil, byteCount: nil,
        contentSHA256: nil, state: .pending, createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil)
    let source: URL
    let verifier: JarvisTemporaryFileVerifier
    let byteCount: Int
    let digest: String
    var acknowledgements: [JarvisWorkspaceAcknowledgement] = []
    var temporaryURL: URL?
    var beforeDisplayed: (() -> Void)?

    init(source: URL, root: URL, byteCount: Int, digest: String) {
        self.source = source; verifier = JarvisTemporaryFileVerifier(root: root)
        self.byteCount = byteCount; self.digest = digest
    }



    func listWorkspaceRequests(sessionID: UUID) async throws -> JarvisWorkspaceRequestPage { .init(items: [offer]) }
    func acknowledgeWorkspaceRequest(requestID: UUID, ack: JarvisWorkspaceAcknowledgement) async throws -> JarvisWorkspaceRequestResponse {
        guard requestID == offer.id else { throw FixtureFailure("Native media request identity changed") }
        if ack.state == .displayed { beforeDisplayed?() }
        acknowledgements.append(ack)
        return .init(request: .init(requestID: offer.id, action: offer.action, artifactKind: offer.artifactKind,
            artifactID: offer.artifactID, versionID: offer.versionID, format: nil, contentReady: true,
            fileID: nil, byteCount: nil, contentSHA256: nil, state: ack.state, createdAt: offer.createdAt,
            expiresAt: offer.expiresAt, errorCode: ack.errorCode, localReceipt: ack.localReceipt))
    }
    func getWorkspaceFileContent(fileID: UUID, sessionID: UUID) async throws -> JarvisWorkspaceFileContent {
        throw FixtureFailure("Native media fixture must use the file-backed path")
    }
    func getWorkspaceMediaContent(request: JarvisWorkspaceRequest, sessionID: UUID,
                                  access: JarvisWorkspaceLocalAccess) async throws -> JarvisWorkspaceMediaFileContent {
        guard request.hasSameTarget(as: offer) else { throw FixtureFailure("Native media target changed") }
        let file = try await verifier.copyVerifiedFile(from: source, expectedByteCount: byteCount,
            expectedSHA256: digest, maximumByteCount: 64 * 1024 * 1024, access: access, format: .mp4)
        temporaryURL = try file.urlForReading()
        return .init(temporaryFile: file, filename: "Synthetic video.mp4", contentType: "video/mp4")
    }
}

private struct FixtureFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// Thread-safe, finite HTTP fixture. It owns no socket and rejects every host
/// except this invocation's .invalid host. Background setup reads get explicit
/// unavailable responses; they cannot contact an account or provider.
private final class ProductFixture: @unchecked Sendable {
    let baseURL = URL(string: "https://phone-" + UUID().uuidString.lowercased() + ".invalid")!
    let noteID = UUID(), requestID = UUID(), rootSessionID = UUID(), libraryFileID = UUID()
    let libraryContent = "{\"message\":\"Saved task file.\"}"
    private var audiovisualFiles: [[String: Any]] = []
    private var audiovisualImages: [String: ProductFixtureImage] = [:]
    private var audiovisualPreviews: [String: ProductFixtureImage] = [:]
    private var failAudiovisualPreviews = false
    private var holdOriginal = false
    private var heldOriginal: (ProductFixtureProtocol, ProductFixtureImage)?
    func holdNextOriginal() { locked { holdOriginal = true } }
    var hasHeldOriginal: Bool { locked { heldOriginal != nil } }
    func releaseOriginal(status: Int = 200) {
        let held = locked { let held = heldOriginal; heldOriginal = nil; return held }
        if let held { held.0.send(status: status, object: status == 200 ? held.1 as Any : error("original_unavailable")) }
    }
    let title = "Exact phone note " + String(UUID().uuidString.prefix(8))
    private let lock = NSLock()
    func enableAudiovisualImages(_ images: [Data], previews: Bool = true, failPreviews: Bool = false) {
        failAudiovisualPreviews = failPreviews
        lock.lock(); defer { lock.unlock() }
        for (index, data) in images.enumerated() {
            let id = UUID().uuidString.lowercased(), version = "native-gallery-image-v1"
            let stamp = "2026-11-02T19:00:00Z"
            audiovisualFiles.append(["id": id, "filename": index == 0 ? "Afternoon light.png" : "First take.png",
                "content_type": "image/png", "size_bytes": data.count, "description": "Synthetic native gallery image",
                "source": "generated", "created_at": stamp,
                "memory_source": ["source_id": id, "revision": 1, "generation": 1, "document_id": version,
                    "content_digest": "fixture-image", "source_type": "file", "trust": "user_provided",
                    "time_basis": "recorded", "recorded_at": stamp, "is_current": true, "ingestion_status": "ready"]])
            audiovisualImages[id] = ProductFixtureImage(data: data, version: version)
            if previews, let image = UIImage(data: data) {
                let format = UIGraphicsImageRendererFormat(); format.scale = 1
                let small = UIGraphicsImageRenderer(size: CGSize(width: 480, height: 320), format: format).image { _ in
                    image.draw(in: CGRect(x: 0, y: 0, width: 480, height: 320))
                }
                let jpeg = small.jpegData(compressionQuality: 0.7)!
                let checksum = SHA256.hash(data: jpeg).map { String(format: "%02x", $0) }.joined()
                audiovisualFiles[audiovisualFiles.count - 1]["thumbnail"] = ["path": "/files/" + id + "/thumbnail",
                    "version_id": version, "content_type": "image/jpeg", "content_sha256": checksum,
                    "size_bytes": jpeg.count, "width": 480, "height": 320]
                audiovisualPreviews[id] = ProductFixtureImage(data: jpeg, version: version, contentType: "image/jpeg")
            }
        }
    }
    private var heldSessions = Set<UUID>()
    private var held: [UUID: ProductFixtureProtocol] = [:]
    private var holdAccountRead = false
    private var heldAccountNote: (transport: ProductFixtureProtocol, body: [String: Any])?
    private var reminderSnapshot: Int?
    private var holdReminderPaths = Set<String>()
    private var heldReminderReads: [String: (transport: ProductFixtureProtocol, body: [String: Any])] = [:]
    private var pluginChangesEnabled = false, pluginEnabled = false
    private var pluginRevision = 3, pluginWrites = 0
    private var heldPluginWrite: (transport: ProductFixtureProtocol, body: [String: Any])?
    private var appliedPluginIntent: [String: Any]?
    private var released: [UUID] = []
    private var probes = 0, revision = 1
    private var profileRevision = 1, profileWrites = 0
    private var profileWriteAttempts = 0, failProfileWrite = false, holdProfileWrite = false
    private var heldProfileWrite: (transport: ProductFixtureProtocol, body: [String: Any])?
    private var sessionStartStatus = 200
    private var existingSession = false
    private var profileText = "Keep answers clear and practical."
    private var populated = false, denied = false
    private var states: [String] = []
    private var unexpected: [String] = []
    private var observed: [String] = []
    private var includesTable = false
    private var displayObserver: (@Sendable () -> Void)?
    private let taskIDs = (0..<4).map { _ in UUID() }, inboxIDs = (0..<4).map { _ in UUID() }
    private let namespace = "jarvis-storage-v1:" + String(repeating: "c", count: 64)
    private let stamp = "2026-11-02T00:00:00Z"
    private func locked<T>(_ body: () throws -> T) rethrows -> T { lock.lock(); defer { lock.unlock() }; return try body() }
    var probeCount: Int { locked { probes } }
    var didDisplay: Bool { locked { states.contains("displayed") } }
    var ackStates: [String] { locked { states } }
    var releasedContexts: [UUID] { locked { released } }
    var unexpectedRequests: [String] { locked { unexpected } }
    var observedPaths: [String] { locked { observed } }
    var pluginWriteCount: Int { locked { pluginWrites } }
    var hasHeldPluginWrite: Bool { locked { heldPluginWrite != nil } }
    func enablePluginMutationFixture() { locked { pluginChangesEnabled = true; pluginEnabled = true } }
    func releasePluginWrite(loseReply: Bool = false) {
        let pending = locked { () -> (ProductFixtureProtocol, [String: Any])? in
            guard let held = heldPluginWrite else { return nil }
            heldPluginWrite = nil
            pluginRevision += 1; pluginEnabled = false; appliedPluginIntent = held.body
            return (held.transport, pluginReceipt(held.body, replayed: false))
        }
        guard let pending else { return }
        if loseReply { pending.0.fail(URLError(.networkConnectionLost)) }
        else { pending.0.send(status: 200, object: pending.1) }
    }
    private func pluginReceipt(_ intent: [String: Any], replayed: Bool) -> [String: Any] {
        ["mutation_id": intent["mutation_id"]!, "applied_revision": 4,
         "request_replayed": replayed, "catalog": pluginCatalog()]
    }
    private func pluginCatalog() -> [String: Any] {
        var items: [[String: Any]] = [[
            "plugin_id": "video_editor", "title": "Video Editor", "description": "Create and edit videos",
            "category": "video_editing", "descriptor_version": 1, "selected_default": true,
            "desired_enabled": pluginEnabled,
            "configuration_id": pluginRevision == 3 ? "00000000-0000-4000-8000-000000000001" : "00000000-0000-4000-8000-000000000002",
            "installed": "unknown", "installed_version": NSNull(), "readiness": pluginEnabled ? "unknown" : "unavailable",
            "reason_code": pluginEnabled ? "plugin_inventory_unknown" : "plugin_disabled", "observations_valid_until": NSNull(),
            "operation": NSNull(), "can_change": pluginChangesEnabled]]
        if pluginChangesEnabled {
            for (id, title, detail, category, config) in [
                ("browser", "Browser", "Browse websites and find information", "web_browsing", "00000000-0000-4000-8000-000000000003"),
                ("files_data", "Files & data", "Work with files, tables, and calculations", "files_data", "00000000-0000-4000-8000-000000000004")] {
                items.append(["plugin_id": id, "title": title, "description": detail, "category": category,
                    "descriptor_version": 1, "selected_default": false, "desired_enabled": true, "configuration_id": config,
                    "installed": "unknown", "installed_version": NSNull(), "readiness": "unknown",
                    "reason_code": "plugin_inventory_unknown", "observations_valid_until": NSNull(),
                    "operation": NSNull(), "can_change": true])
            }
        }
        return ["schema_version": 1, "evaluated_at": stamp, "policy_revision": pluginRevision,
                "policy_available": true, "items": items]
    }
    func setSessionStartStatus(_ status: Int) { locked { sessionStartStatus = status } }
    func setExistingSession(_ value: Bool) { locked { existingSession = value } }
    func setReminderSnapshot(_ revision: Int) { locked { reminderSnapshot = revision } }
    func holdNextReminderRead(_ path: String) { locked { _ = holdReminderPaths.insert(path) } }
    func hasHeldReminderRead(_ path: String) -> Bool { locked { heldReminderReads[path] != nil } }
    func releaseReminderRead(_ path: String, status: Int = 200) {
        let pending = locked { heldReminderReads.removeValue(forKey: path) }
        if let pending {
            pending.transport.send(status: status, object: status == 200 ? pending.body : error("fixture_reminders_unavailable"))
        }
    }

    var hasHeldAccountNote: Bool { locked { heldAccountNote != nil } }

    func holdContext(_ session: UUID) { locked { _ = heldSessions.insert(session) } }
    func hasHeldContext(_ session: UUID) -> Bool { locked { held[session] != nil } }
    func populateCards() { locked { populated = true } }
    func includeTableInNote() { locked { includesTable = true } }
    func correct(to revision: Int) { locked { self.revision = revision } }
    func denyWorkspace() { locked { denied = true } }
    func observeDisplayed(_ observer: @escaping @Sendable () -> Void) { locked { displayObserver = observer } }
    var profileWriteCount: Int { locked { profileWrites } }
    var profileWriteAttemptCount: Int { locked { profileWriteAttempts } }
    var hasHeldProfileWrite: Bool { locked { heldProfileWrite != nil } }
    func failNextProfileWrite() { locked { failProfileWrite = true } }
    func holdNextProfileWrite() { locked { holdProfileWrite = true } }
    func releaseProfileWrite() {
        let pending = locked { () -> (transport: ProductFixtureProtocol, body: [String: Any])? in
            defer { heldProfileWrite = nil }
            return heldProfileWrite
        }
        if let pending { pending.transport.send(status: 200, object: pending.body) }
    }
    func prepareEmptyProfileForUIComponent() { locked { profileRevision = 0; profileText = ""; profileWrites = 0 } }
    func commitProfile(text: String) -> JarvisBehaviorProfileCommitted {
        locked {
            profileRevision += 1; profileText = text
            return JarvisBehaviorProfileCommitted(type: "behavior_profile.committed", eventID: "native-profile-\(profileRevision)",
                ownerID: context(for: rootSessionID).ownerID, revision: Int64(profileRevision),
                behaviorInstructions: text, committedAt: stamp)
        }
    }
    private func profileJSON() -> [String: Any] {
        ["owner_id": context(for: rootSessionID).ownerID, "behavior_instructions": profileText,
         "revision": profileRevision, "created_at": stamp, "updated_at": stamp]
    }
    func context(for session: UUID) -> JarvisAuthenticatedContext {
        JarvisAuthenticatedContext(ownerID: "fixture-owner-" + session.uuidString, backendNamespace: namespace, sessionID: session)
    }
    private func contextJSON(_ session: UUID) -> [String: Any] {
        ["owner_id": context(for: session).ownerID, "backend_namespace": namespace, "session_id": session.uuidString]
    }
    func releaseContext(_ session: UUID) {
        let pending = locked { () -> ProductFixtureProtocol? in
            heldSessions.remove(session); released.append(session); return held.removeValue(forKey: session)
        }
        pending?.send(status: 200, object: contextJSON(session))
    }
    func releaseAllContexts() {
        releaseProfileWrite()
        let sessions = locked { Array(held.keys) }
        sessions.forEach(releaseContext)
        releasePluginWrite()
        locked { Array(heldReminderReads.keys) }.forEach { releaseReminderRead($0) }
    }
    func receive(_ request: URLRequest, via transport: ProductFixtureProtocol) {
        var displayed = false
        let result: (Int, Any)? = locked {
            guard request.url?.host == baseURL.host else {
                unexpected.append(ProductFixtureProtocol.rejectionDescription(for: request.url?.host))
                return (400, error("nonfixture_host"))
            }
            let path = request.url!.path
            if observed.count < 40 { observed.append(path) }
            let query = Dictionary(uniqueKeysWithValues: (URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems ?? []).map { ($0.name, $0.value ?? "") })
            if path == "/shared-probe" { probes += 1; return (200, "shared-session-intercepted") }
            if path == "/sessions/start", request.httpMethod == "POST" {
                if sessionStartStatus != 200 { return (sessionStartStatus, error("fixture_session_unavailable")) }
                return (200, ["session": ["id": rootSessionID.uuidString, "title": "Native UI fixture", "status": "active",
                    "created_at": stamp, "updated_at": stamp, "last_activity_at": stamp, "behavior_instructions": ""],
                    "realtime": ["audio_websocket_url": "wss://unused.invalid/audio", "visual_websocket_url": "wss://unused.invalid/visual"]])
            }
            if path == "/sessions/active", request.httpMethod == "GET" {
                if !existingSession { return (200, ["session": NSNull()]) }
                return (200, ["session": ["id": rootSessionID.uuidString, "title": "Native UI fixture", "status": "active",
                    "created_at": stamp, "updated_at": stamp, "last_activity_at": stamp, "behavior_instructions": ""]])
            }
            if path == "/sandbox-views", request.httpMethod == "GET", query["session_id"]?.lowercased() == rootSessionID.uuidString.lowercased() {
                return (200, ["items": [], "has_more": false])
            }
            if let revision = reminderSnapshot, ["/reminders/active", "/reminders/events"].contains(path) {
                let body: [String: Any] = path == "/reminders/active"
                    ? ["reminders": [["id": "reminder-\(revision)", "title": "Reminder \(revision)", "details": "Fixture account data",
                        "remind_at": NSNull(), "condition": NSNull(), "status": "active", "updated_at": stamp]]]
                    : ["events": [["id": "event-\(revision)", "reminder_id": "reminder-\(revision)", "session_id": NSNull(),
                        "event_type": "prepared", "message": "Event \(revision)", "provider_response_id": NSNull(), "created_at": stamp]]]
                if holdReminderPaths.remove(path) != nil {
                    heldReminderReads[path] = (transport, body)
                    return nil
                }
                return (200, body)
            }
            if path == "/assistant/profile" {
                if request.httpMethod == "GET" { return (200, profileJSON()) }
                if request.httpMethod == "PUT", let body = try? Self.readBody(request),
                   let expected = body["expected_revision"] as? Int,
                   let text = body["behavior_instructions"] as? String {
                    guard expected == profileRevision else { return (409, error("behavior_profile_revision_conflict")) }
                    profileWriteAttempts += 1
                    if failProfileWrite {
                        failProfileWrite = false
                        return (503, error("fixture_profile_save_failed"))
                    }
                    profileRevision += 1; profileWrites += 1; profileText = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if holdProfileWrite {
                        holdProfileWrite = false
                        heldProfileWrite = (transport, profileJSON())
                        return nil
                    }
                    return (200, profileJSON())
                }
                unexpected.append("invalid profile method/body")
                return (400, error("invalid_fixture_profile"))
            }
            if path == "/plugins" {
                guard request.httpMethod == "GET", query.isEmpty else {
                    unexpected.append("Plugins must use an account GET without session parameters")
                    return (400, error("invalid_plugin_scope"))
                }
                return (200, pluginCatalog())
            }
            if path == "/plugins/policy" {
                guard pluginChangesEnabled, request.httpMethod == "PUT", query.isEmpty,
                      let body = try? Self.readBody(request),
                      Set(body.keys) == ["mutation_id", "expected_revision", "enabled_changes", "default_changes"],
                      let rawID = body["mutation_id"] as? String, UUID(uuidString: rawID) != nil,
                      body["expected_revision"] as? Int == 3,
                      let choices = body["enabled_changes"] as? [[String: Any]], choices.count == 1,
                      choices[0]["plugin_id"] as? String == "video_editor", choices[0]["enabled"] as? Bool == false,
                      let defaults = body["default_changes"] as? [Any], defaults.isEmpty else {
                    unexpected.append("Invalid fixture plugin mutation scope or intent")
                    return (400, error("invalid_plugin_mutation"))
                }
                pluginWrites += 1
                if let applied = appliedPluginIntent {
                    guard NSDictionary(dictionary: applied).isEqual(to: body) else {
                        unexpected.append("Plugin recheck replaced its original intent")
                        return (409, error("plugin_policy_mutation_conflict"))
                    }
                    return (200, pluginReceipt(applied, replayed: true))
                }
                guard heldPluginWrite == nil else {
                    unexpected.append("Overlapping plugin writes")
                    return (409, error("plugin_change_pending"))
                }
                heldPluginWrite = (transport, body)
                return nil
            }
            if path == "/client-context", let raw = query["session_id"], let session = UUID(uuidString: raw) {
                if heldSessions.contains(session) { held[session] = transport; return nil }
                return (200, contextJSON(session))
            }
            if path == "/files" {
                guard query["session_id"] == rootSessionID.uuidString.lowercased(), query["limit"] == "8" else {
                    unexpected.append("Files library missing owned session or bounded limit")
                    return (400, error("invalid_files_scope"))
                }
                if query["media_only"] == "true" {
                    return (200, ["files": audiovisualFiles, "page_full": false, "next_cursor": NSNull()])
                }
                return (200, ["files": [], "page_full": false])
            }
            if path.hasPrefix("/files/") {
                guard query["session_id"] == rootSessionID.uuidString.lowercased() else {
                    unexpected.append("Task file missing the verified owned session")
                    return (400, error("invalid_task_file_scope"))
                }
                let parts = path.split(separator: "/")
                if parts.count >= 2, let image = audiovisualImages[String(parts[1])] {
                    if parts.count == 3, parts[2] == "thumbnail" {
                        guard query["version_id"] == image.version else { return (409, error("version_mismatch")) }
                        if failAudiovisualPreviews { return (500, error("fixture_preview_failed")) }
                        guard let preview = audiovisualPreviews[String(parts[1])] else { return (404, error("thumbnail_unavailable")) }
                        return (200, preview)
                    }
                    if parts.count == 3, parts[2] == "content" {
                        guard query["version_id"] == image.version else { return (409, error("version_mismatch")) }
                        if holdOriginal { holdOriginal = false; heldOriginal = (transport, image); return nil }
                        return (200, image)
                    }
                    guard query["version_id"] == nil || query["version_id"] == image.version else {
                        return (409, error("version_mismatch"))
                    }
                    if parts.count == 2, let file = audiovisualFiles.first(where: { $0["id"] as? String == String(parts[1]) }) {
                        return (200, ["file": file])
                    }
                }
                let filePath = "/files/" + libraryFileID.uuidString.lowercased()
                if path == filePath + "/content" { return (200, libraryContent) }
                if path == filePath {
                    return (200, ["file": ["id": libraryFileID.uuidString.lowercased(),
                        "filename": "Task result.json", "content_type": "application/json",
                        "size_bytes": libraryContent.utf8.count, "description": "A saved task result",
                        "source": "generated", "created_at": stamp,
                        "memory_source": ["source_id": libraryFileID.uuidString, "revision": 1, "generation": 1,
                            "content_digest": "fixture", "source_type": "file", "trust": "user_provided",
                            "time_basis": "recorded", "recorded_at": stamp, "is_current": true, "ingestion_status": "ready"]]])
                }
                unexpected.append("Unexpected task file target: " + path)
                return (404, error("unknown_task_file"))
            }
            if path.hasPrefix("/workspace/"), denied { return (403, error("jarvis_auth_invalid")) }
            if path == "/workspace/note" {
                guard query.isEmpty else {
                    unexpected.append("account note unexpectedly required session/query")
                    return (400, error("unexpected_account_note_query"))
                }
                let body: [String: Any] = ["note": documentJSON(version: revision, summaryOnly: false)]
                if holdAccountRead {
                    holdAccountRead = false; heldAccountNote = (transport, body)
                    return nil
                }
                return (200, body)
            }
            if path == "/workspace/artifacts" { return (200, ["items": [artifact(revision)], "next_cursor": NSNull()]) }
            if path == "/workspace/artifacts/" + noteID.uuidString.lowercased() + "/versions" {
                return (200, ["items": (1...revision).reversed().map { artifact($0) }, "unavailable_count": 0])
            }
            if path == "/workspace/artifacts/" + noteID.uuidString.lowercased() {
                let version = query["version_id"].flatMap { Int($0.replacingOccurrences(of: "fixture-v", with: "")) } ?? revision
                guard (1...revision).contains(version) else { return (404, error("unknown_fixture_version")) }
                return (200, documentJSON(version: version, summaryOnly: query["view"] == "summary"))
            }
            if path == "/workspace/requests" { return (200, ["items": [presentation(states.last ?? "pending")]]) }
            if path == "/workspace/requests/" + requestID.uuidString.lowercased() + "/ack", request.httpMethod == "POST" {
                guard let body = try? Self.readBody(request), let state = body["state"] as? String,
                      ["claimed", "displayed"].contains(state), body["session_id"] is String else {
                    unexpected.append("invalid ACK body"); return (400, error("invalid_fixture_ack"))
                }
                if state == "displayed" { displayed = true }
                else { states.append(state) }
                return (200, ["request": presentation(state)])
            }
            if path == "/tasks" {
                return (200, populated ? taskIDs.enumerated().map { index, id -> [String: Any] in
                    ["id": id.uuidString, "title": "Task \(index + 1): " + String(repeating: "Existing work needs attention. ", count: 4),
                     "status": "waiting_for_input", "revision": 1, "updated_at": stamp]
                } : [])
            }
            if path == "/attention" {
                return (200, populated ? inboxIDs.map { id -> [String: Any] in
                    ["id": id.uuidString, "source_namespace": "task_result", "state": "pending", "generation": 1,
                     "attempt_count": 0, "source_available": true, "updated_at": stamp]
                } : [])
            }
            if ["/attention/push-status", "/attention/devices", "/oauth/google/status", "/memory/recent", "/memory/working-state", "/reminders/active", "/reminders/events"].contains(path) {
                return (503, error("fixture_side_read_unavailable"))
            }
            unexpected.append("\(request.httpMethod ?? "?") \(path)")
            return (404, error("unexpected_fixture_route"))
        }
        if let (status, object) = result {
            if displayed {
                let observer = locked { displayObserver }
                observer?()
                locked { states.append("displayed") }
            }
            transport.send(status: status, object: object)
        }
    }
    private func error(_ code: String) -> [String: Any] { ["error": ["code": code, "message": "Controlled native fixture response"]] }
    private func documentJSON(version: Int, summaryOnly: Bool) -> [String: Any] {
        var value: [String: Any] = ["artifact": artifact(version)]
                if !summaryOnly {
                    let retainedText = String(repeating: "Retained note content for native scrolling.\n", count: 30)
                    var blocks: [[String: Any]] = [["id": "body", "type": "text",
                        "text": "Saved fixture value \(version).\n" + (includesTable ? "" : retainedText)]]
                    if includesTable {
                        // Actual structured rendering in both Root galleries;
                        // unknown stays null and the long scroll case remains.
                        blocks.append(["id": "readings", "type": "table",
                            "columns": [["id": "sample", "label": "Sample"], ["id": "value", "label": "Reading"]],
                            "rows": [["id": "a", "cells": ["sample": "A", "value": 42.5]],
                                     ["id": "b", "cells": ["sample": "B", "value": NSNull()]]]])
                        blocks.append(["id": "details", "type": "text", "text": retainedText])
                    }
                    value["content"] = ["schema_version": 1, "blocks": blocks]
                }
        return value
    }
    private func artifact(_ version: Int) -> [String: Any] {
        ["artifact_id": noteID.uuidString, "version_id": "fixture-v\(version)", "revision": version,
         "current_version_id": "fixture-v\(revision)", "current_revision": revision, "is_current": version == revision,
         "kind": "note", "title": title, "updated_at": stamp]
    }
    private func presentation(_ state: String) -> [String: Any] {
        ["request_id": requestID.uuidString, "action": "open", "artifact_kind": "note", "artifact_id": noteID.uuidString,
         "version_id": "fixture-v1", "content_ready": true, "state": state, "created_at": stamp,
         "expires_at": ISO8601DateFormatter().string(from: Date().addingTimeInterval(300))]
    }
    private static func readBody(_ request: URLRequest) throws -> [String: Any] {
        var data = request.httpBody ?? Data()
        if data.isEmpty, let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var bytes = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&bytes, maxLength: bytes.count)
                guard count > 0 else { break }
                data.append(contentsOf: bytes.prefix(count))
            }
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }
}

private final class ProductFixtureProtocol: URLProtocol, @unchecked Sendable {
    private static let registryLock = NSLock()
    private static var fixture: ProductFixture?
    private static var retiredHosts = Set<String>()
    private let stateLock = NSRecursiveLock()
    private var stopped = false
    static func install(_ value: ProductFixture?) {
        registryLock.lock(); defer { registryLock.unlock() }
        if let old = fixture?.baseURL.host, old != value?.baseURL.host { retiredHosts.insert(old) }
        fixture = value
    }
    static func rejectionDescription(for host: String?) -> String {
        registryLock.lock(); defer { registryLock.unlock() }
        return host.map { retiredHosts.contains($0) } == true
            ? "retired fixture host rejected" : "unrecognized host rejected"
    }
    private static func current() -> ProductFixture? { registryLock.lock(); defer { registryLock.unlock() }; return fixture }
    override class func canInit(with request: URLRequest) -> Bool {
        ["http", "https"].contains(request.url?.scheme ?? "")
    }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let fixture = Self.current() else { client?.urlProtocol(self, didFailWithError: URLError(.cancelled)); return }
        fixture.receive(request, via: self)
    }
    override func stopLoading() { stateLock.lock(); stopped = true; stateLock.unlock() }
    func fail(_ error: Error) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard !stopped else { return }
        client?.urlProtocol(self, didFailWithError: error)
    }
    func send(status: Int, object: Any) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard !stopped else { return }
        do {
            let data: Data
            var headers = ["Content-Type": "application/json", "Cache-Control": "no-store"]
            if let image = object as? ProductFixtureImage {
                data = image.data
                headers["Content-Type"] = image.contentType
                headers["Content-Length"] = String(data.count)
                headers["X-Jarvis-Content-Sha256"] = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                headers["X-Jarvis-Version-Id"] = image.version
            }
            else if let text = object as? String { data = Data(text.utf8) }
            else { data = try JSONSerialization.data(withJSONObject: object) }
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: headers)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
}

private struct ProductFixtureImage {
    let data: Data
    let version: String
    var contentType: String = "image/png"
}
