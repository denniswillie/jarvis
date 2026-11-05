import XCTest
import SwiftUI
import WebKit
@testable import Jarvis

@MainActor
final class JarvisSandboxViewTests: XCTestCase {
    private var window: UIWindow?
    private var previous: UIWindow?
    private var hostedModel: JarvisSandboxViewModel?
    private var style: SandboxProofStyle?
    override func tearDown() async throws {
        window?.isHidden = true; window?.rootViewController = nil; window = nil
        hostedModel = nil; style = nil
        previous?.makeKeyAndVisible(); previous = nil
        try await super.tearDown()
    }
    func testIdleUnavailableAndSelectionAtBothTextSizes() async throws {
        let api = SandboxFixture(), model = JarvisSandboxViewModel(api: SandboxFixture(), sessionID: UUID())
        for size in [DynamicTypeSize.large, .accessibility5] {
            try show(model, size: size)
            try await Task.sleep(for: .milliseconds(250))
            capture("sandbox-idle-\(size)")
        }
        let unavailable = JarvisSandboxViewModel(api: api, sessionID: UUID())
        try show(unavailable, size: .large)
        await unavailable.choose()
        XCTAssertEqual(unavailable.phase, .idle)
        XCTAssertNotNil(unavailable.notice)
        try await Task.sleep(for: .milliseconds(350))
        capture("sandbox-unavailable")
        unavailable.notice = nil
        api.items = [api.target(), api.target()]
        for size in [DynamicTypeSize.large, .accessibility5] {
            let choices = JarvisSandboxViewModel(api: api, sessionID: UUID())
            await choices.choose()
            XCTAssertEqual(choices.phase, .choosing)
            XCTAssertEqual(api.opens, 0)
            try show(choices, size: size)
            try await Task.sleep(for: .milliseconds(250))
            capture("sandbox-selection-\(size)")
            choices.deactivate()
        }
    }
    func testLateOpenCannotSurviveAccountRetirement() async throws {
        let api = SandboxFixture(), model = JarvisSandboxViewModel(api: SandboxFixture(), sessionID: UUID())
        model.deactivate(); await model.choose(); XCTAssertEqual(model.phase, .idle)
        let active = JarvisSandboxViewModel(api: api, sessionID: UUID()), selected = api.target()
        api.holdOpen = true
        let opening = Task { await active.open(allocationID: selected.id, version: selected.generation) }
        for _ in 0..<30 where api.pending == nil { await Task.yield() }
        XCTAssertNotNil(api.pending)
        active.deactivate()
        api.release(selected)
        let opened = await opening.value
        XCTAssertFalse(opened); XCTAssertNil(active.pageRequest); XCTAssertNil(active.lease)
        XCTAssertEqual(api.closed.count, 1)
    }
    func testFrameReceiptWaitsForVisibleCurrentView() async throws {
        let api = SandboxFixture(), model = JarvisSandboxViewModel(api: SandboxFixture(), sessionID: UUID())
        model.deactivate()
        let active = JarvisSandboxViewModel(api: api, sessionID: UUID()), selected = api.target()
        api.holdOpen = true
        var receipts = 0
        active.onFirstFrame = { _ in receipts += 1 }
        let opening = Task { await active.open(allocationID: selected.id, version: selected.generation) }
        for _ in 0..<30 where api.pending == nil { await Task.yield() }
        api.release(selected)
        let opened = await opening.value
        XCTAssertTrue(opened)
        let page = active.pageID
        active.pageEvent("connected", pageID: page)
        active.confirmVisibleFrame(true)
        XCTAssertEqual(receipts, 0)
        active.pageEvent("frame", pageID: UUID())
        XCTAssertEqual(active.phase, .connecting)
        active.pageEvent("frame", pageID: page)
        active.confirmVisibleFrame(false)
        XCTAssertEqual(receipts, 0)
        active.confirmVisibleFrame(true)
        active.confirmVisibleFrame(true)
        XCTAssertEqual(receipts, 1)
        active.deactivate()
        active.pageEvent("frame", pageID: page)
        active.confirmVisibleFrame(true)
        XCTAssertEqual(receipts, 1)
    }
    func testTransportFailureIsNotShownAsShutdown() async throws {
        let api = SandboxFixture(), model = JarvisSandboxViewModel(api: api, sessionID: UUID())
        api.openError = JarvisAPIError(code: "sandbox_view_connection_failed", message: "The screen couldn't connect. Try again.")
        let selected = api.target()
        await model.open(allocationID: selected.id, version: selected.generation)
        XCTAssertEqual(model.phase, .failed); XCTAssertNil(model.notice)
        for size in [DynamicTypeSize.large, .accessibility5] {
            try show(model, size: size)
            try await Task.sleep(for: .milliseconds(250))
            capture("sandbox-error-\(size)")
        }
        model.deactivate()
        let stopped = JarvisSandboxViewModel(api: api, sessionID: UUID())
        api.openError = nil; api.holdOpen = true
        api.statusError = JarvisAPIError(code: "sandbox_not_running", message: "The sandbox stopped.")
        let opening = Task { await stopped.open(allocationID: selected.id, version: selected.generation) }
        for _ in 0..<30 where api.pending == nil { await Task.yield() }
        api.release(selected)
        _ = await opening.value
        stopped.pageEvent("disconnected", pageID: stopped.pageID)
        for _ in 0..<30 where stopped.phase != .idle { await Task.yield() }
        XCTAssertEqual(stopped.phase, .idle)
        XCTAssertNil(stopped.pageRequest); XCTAssertNil(stopped.lease)
        XCTAssertEqual(stopped.notice?.title, "Sandbox not running")
        stopped.deactivate()
    }
    func testMaximizeKeepsSameViewerAndReturnsToWorkspace() async throws {
        let (api, model) = try await prepareFullscreenFixture()
        defer { model.deactivate() }
        let controller = try XCTUnwrap(findController(JarvisSandboxScreenController.self, in: window?.rootViewController))
        let web = try XCTUnwrap(findView(WKWebView.self, in: window))
        let lease = model.lease, page = model.pageID
        let originalOrientation = window?.windowScene?.interfaceOrientation
        controller.maximizedChanged?(true)
        try await waitUntil { controller.fullscreen?.view.window != nil }
        try await waitUntil { self.window?.windowScene?.interfaceOrientation.isLandscape == true }
        let expanded = try XCTUnwrap(controller.fullscreen)
        try await waitUntil { !expanded.isBeingPresented && expanded.transitionCoordinator == nil }
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(expanded.view.bounds, window?.bounds)
        XCTAssertTrue(findView(WKWebView.self, in: expanded.view) === web)
        XCTAssertGreaterThan(web.bounds.width, web.bounds.height)
        let expandedText = try await web.evaluateJavaScript("document.body.innerText") as? String
        XCTAssertTrue(expandedText?.contains("Room to focus.") == true)
        XCTAssertEqual(model.lease, lease); XCTAssertEqual(model.pageID, page)
        XCTAssertEqual(api.opens, 1); XCTAssertTrue(api.closed.isEmpty); XCTAssertTrue(api.commands.isEmpty)
        capture("sandbox-fullscreen-landscape")
        let pixels = try await web.takeSnapshot(configuration: nil)
        let attachment = XCTAttachment(image: pixels); attachment.name = "fullscreen-synthetic-desktop-webkit"
        attachment.lifetime = .keepAlways; add(attachment)
        let minimize = try XCTUnwrap(findView(UIButton.self, in: expanded.view))
        XCTAssertEqual(minimize.accessibilityIdentifier, "sandbox.minimize")
        minimize.sendActions(for: .touchUpInside)
        try await waitUntil { controller.fullscreen == nil }
        try await waitUntil { self.window?.windowScene?.interfaceOrientation == originalOrientation }
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertTrue(findView(WKWebView.self, in: controller.view) === web)
        XCTAssertEqual(model.lease, lease); XCTAssertEqual(model.pageID, page)
        XCTAssertEqual(api.opens, 1); XCTAssertTrue(api.closed.isEmpty); XCTAssertTrue(api.commands.isEmpty)
        let returnedText = try await web.evaluateJavaScript("document.body.innerText") as? String
        XCTAssertTrue(returnedText?.contains("Room to focus.") == true)
        capture("sandbox-returned-to-workspace")
    }

    func testFullscreenClearsWhenAccountRetires() async throws {
        let (_, model) = try await prepareFullscreenFixture()
        let controller = try XCTUnwrap(findController(JarvisSandboxScreenController.self, in: window?.rootViewController))
        controller.maximizedChanged?(true)
        try await waitUntil { controller.fullscreen?.view.window != nil }
        model.deactivate()
        try await waitUntil { controller.fullscreen == nil }
        XCTAssertNil(model.lease); XCTAssertNil(model.pageRequest)
        XCTAssertNil(findView(WKWebView.self, in: window))
    }

    func testFullscreenConnectionStateAtLargestTextSize() async throws {
        let (_, model) = try await prepareFullscreenFixture()
        defer { model.deactivate() }
        style?.size = .accessibility5
        let controller = try XCTUnwrap(findController(JarvisSandboxScreenController.self, in: window?.rootViewController))
        controller.maximizedChanged?(true)
        try await waitUntil { self.window?.windowScene?.interfaceOrientation.isLandscape == true }
        try await Task.sleep(for: .milliseconds(700))
        model.pageEvent("disconnected", pageID: model.pageID)
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertNotNil(controller.fullscreen)
        capture("sandbox-fullscreen-reconnecting-large")
        controller.minimize(animated: false)
        try await waitUntil { controller.fullscreen == nil }
    }

    private func prepareFullscreenFixture() async throws -> (SandboxFixture, JarvisSandboxViewModel) {
        let api = SandboxFixture()
        let active = JarvisSandboxViewModel(api: api, sessionID: UUID()), selected = api.target()
        api.holdOpen = true; api.pageURL = URL(string: "about:blank")!
        let opening = Task { await active.open(allocationID: selected.id, version: selected.generation) }
        try await waitUntil { api.pending != nil }
        api.release(selected)
        let opened = await opening.value
        XCTAssertTrue(opened)
        api.statusReply = active.lease
        try show(active, size: .large)
        if let scene = window?.windowScene { window?.frame = scene.coordinateSpace.bounds }
        try await waitUntil { self.findView(WKWebView.self, in: self.window) != nil }
        let web = try XCTUnwrap(findView(WKWebView.self, in: window))
        try await waitUntil { !web.isLoading && web.url != nil }
        // Populate the already loaded test page, avoiding a second navigation
        // racing the viewer's initial load. No real desktop or lease is involved.
        let html = """
        <head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
        <body style="margin:0;background:#162625;color:#f7f7f4;font:20px -apple-system;display:grid;place-items:center;height:100vh">
        <main style="width:80%;padding:24px;border:1px solid #66897e;border-radius:12px;background:linear-gradient(120deg,#254d40,#1b302d)">
        <small>JARVIS • SYNTHETIC TEST DESKTOP</small><h1>Room to focus.</h1><p>Your current sandbox, expanded.</p>
        </main></body>
        """
        _ = try await web.callAsyncJavaScript("document.documentElement.innerHTML = html; return document.body.innerText;",
            arguments: ["html": html], in: nil, contentWorld: .page)
        try await Task.sleep(for: .milliseconds(250))
        active.pageEvent("frame", pageID: active.pageID)
        XCTAssertEqual(active.phase, .viewing)
        return (api, active)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<100 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(50))
        }
        throw NativeSandboxFailure("Timed out waiting for native sandbox presentation")
    }
    private func findView<T: UIView>(_ type: T.Type, in view: UIView?) -> T? {
        guard let view else { return nil }
        if let match = view as? T { return match }
        return view.subviews.lazy.compactMap { self.findView(type, in: $0) }.first
    }
    private func findController<T: UIViewController>(_ type: T.Type, in controller: UIViewController?) -> T? {
        guard let controller else { return nil }
        if let match = controller as? T { return match }
        return controller.children.lazy.compactMap { self.findController(type, in: $0) }.first
    }

    func testRealProtectedDesktopThroughNativeViewer() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["JARVIS_VIEW_PROOF_URL"], let base = URL(string: address),
              let session = env["JARVIS_VIEW_PROOF_SESSION"].flatMap(UUID.init(uuidString:)) else {
            throw XCTSkip("Explicit finite provider fixture required")
        }
        let bearerFile = env["JARVIS_VIEW_PROOF_BEARER_FILE"]
        let allocationID = env["JARVIS_VIEW_PROOF_ALLOCATION"].flatMap(UUID.init(uuidString:))
        guard bearerFile == nil || allocationID != nil else {
            throw XCTSkip("Serving proof requires an exact existing allocation")
        }
        let bearer = try bearerFile.map {
            try String(contentsOfFile: $0, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        } ?? "local-sandbox-view-proof"
        let api = JarvisAPIClient(configuration: JarvisConfiguration(backendBaseURL: base, staticBearerToken: bearer))
        let model = JarvisSandboxViewModel(api: api, sessionID: session)
        defer { model.deactivate() }
        await model.choose()
        try show(model, size: .large)
        let selected = try XCTUnwrap(model.choices.first { allocationID == nil || $0.id == allocationID })
        XCTAssertTrue(selected.canOpen, "This renderer fixture needs an already running desktop")
        _ = await model.open(allocationID: selected.allocationID, version: selected.generation)
        for _ in 0..<200 {
            if model.phase == .viewing || model.phase == .failed { break }
            try await Task.sleep(for: .milliseconds(100))
        }
        let leaseID = try XCTUnwrap(model.lease?.id)
        do {
            XCTAssertEqual(model.phase, .viewing, model.errorMessage ?? "No rendered framebuffer")
            let receipt = XCTAttachment(string: "allocation_id=\(selected.id)\nview_id=\(leaseID)")
            receipt.name = "sandbox-exact-view-receipt"; receipt.lifetime = .keepAlways; add(receipt)
            try await captureDesktop("sandbox-real-desktop-normal", base: base, fixtureCompositor: bearerFile == nil)
            try show(model, size: .accessibility5)
            try await Task.sleep(for: .milliseconds(800))
            XCTAssertEqual(model.phase, .viewing)
            try await captureDesktop("sandbox-real-desktop-accessibility", base: base, fixtureCompositor: bearerFile == nil)
        } catch {
            model.close(sendRevoke: false)
            try? await api.closeSandboxView(sessionID: session, id: leaseID)
            throw error
        }
        model.close(sendRevoke: false)
        try await api.closeSandboxView(sessionID: session, id: leaseID)
        XCTAssertNil(model.pageRequest); XCTAssertEqual(model.phase, .idle)
    }
    private func show(_ model: JarvisSandboxViewModel, size: DynamicTypeSize) throws {
        if hostedModel === model, let style, let window {
            style.size = size
            window.layoutIfNeeded()
            return
        }
        window?.isHidden = true
        window?.rootViewController = nil
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first { $0.activationState == .foregroundActive })
        if previous == nil { previous = scene.windows.first { $0.isKeyWindow } }
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 390, height: 844)
        let style = SandboxProofStyle(size: size)
        let content = SandboxProofSurface(model: model, style: style)
        self.style = style; self.hostedModel = model
        window.rootViewController = UIHostingController(rootView: content)
        self.window = window; window.makeKeyAndVisible(); window.layoutIfNeeded()
    }

    func testSinglePausedCardRequiresExplicitResumeAndUsesSharedControl() async throws {
        let api = SandboxFixture(), model = JarvisSandboxViewModel(api: api, sessionID: UUID())
        let target = JarvisSandboxTarget(allocationID: UUID(), generation: String(repeating: "a", count: 64),
            title: "Retained edit", hermesSessionID: "retained-session", state: "paused", capability: "desktop",
            canView: false, canResumeAndView: true)
        api.items = [target]
        var requested = 0
        model.onPresentationIntent = { _ in requested += 1 }
        await model.choose()
        XCTAssertEqual(model.phase, .choosing)
        XCTAssertEqual(api.opens, 0)
        XCTAssertTrue(api.commands.isEmpty)
        await model.requestView(target)
        XCTAssertEqual(model.resumeCandidate?.id, target.id)
        XCTAssertTrue(api.commands.isEmpty)
        await model.resumeAndView()
        XCTAssertEqual(api.commands.count, 1)
        XCTAssertEqual(api.commands[0]["resume"] as? Bool, true)
        XCTAssertEqual(api.commands[0]["allocation_id"] as? String, target.id.uuidString.lowercased())
        XCTAssertEqual(requested, 1)
        XCTAssertEqual(api.opens, 0, "Only the existing presentation renderer opens the transport")
    }

    func testTerminalCardNeverRequestsResumeOrDisplay() async throws {
        let api = SandboxFixture(), model = JarvisSandboxViewModel(api: api, sessionID: UUID())
        let target = JarvisSandboxTarget(allocationID: UUID(), generation: String(repeating: "b", count: 64),
            title: "Terminal work", state: "running", capability: "terminal", canView: false)
        api.items = [target]
        model.onPresentationIntent = { _ in XCTFail("Terminal cannot be displayed") }
        await model.choose(); await model.requestView(target)
        XCTAssertTrue(api.commands.isEmpty)
        XCTAssertEqual(api.opens, 0)
        XCTAssertNotNil(model.notice)
    }

    func testThumbnailIsLazyCachedAndBoundToExactSelection() async throws {
        let api = SandboxFixture(), target = SandboxFixture().target()
        api.items = [target]
        api.thumbnailReply = JarvisSandboxThumbnail(allocationID: target.id, generation: target.generation,
            contentType: "image/jpeg", image: Data([1, 2, 3]), capturedAt: "2026-11-04T00:00:00Z", stale: false)
        let model = JarvisSandboxViewModel(api: api, sessionID: UUID())
        await model.choose()
        XCTAssertEqual(api.thumbnailCalls, 0)
        await model.loadThumbnail(target); await model.loadThumbnail(target)
        XCTAssertEqual(api.thumbnailCalls, 1)
        XCTAssertEqual(model.thumbnail(for: target)?.image, Data([1, 2, 3]))
        XCTAssertEqual(api.opens, 0); XCTAssertTrue(api.commands.isEmpty)
        model.deactivate()
        XCTAssertTrue(model.thumbnails.isEmpty)
        api.thumbnailReply = JarvisSandboxThumbnail(allocationID: UUID(), generation: target.generation,
            contentType: "image/jpeg", image: Data([1]), capturedAt: "2026-11-04T00:00:00Z", stale: false)
        let other = JarvisSandboxViewModel(api: api, sessionID: UUID())
        await other.choose(); await other.loadThumbnail(target)
        XCTAssertNil(other.thumbnail(for: target))
        other.deactivate()
    }

    func testRealGalleryThumbnailWithoutOpeningStream() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["JARVIS_VIEW_PROOF_URL"], let base = URL(string: address),
              let session = env["JARVIS_VIEW_PROOF_SESSION"].flatMap(UUID.init(uuidString:)),
              let allocation = env["JARVIS_VIEW_PROOF_ALLOCATION"].flatMap(UUID.init(uuidString:)),
              let tokenPath = env["JARVIS_VIEW_PROOF_BEARER_FILE"] else {
            throw XCTSkip("Explicit existing-resource native thumbnail lane required")
        }
        let bearer = try String(contentsOfFile: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        let api = JarvisAPIClient(configuration: JarvisConfiguration(backendBaseURL: base, staticBearerToken: bearer))
        let model = JarvisSandboxViewModel(api: api, sessionID: session)
        defer { model.deactivate() }
        await model.choose()
        let target = try XCTUnwrap(model.choices.first { $0.id == allocation })
        XCTAssertTrue(target.canOpen)
        await model.loadThumbnail(target)
        let preview = try XCTUnwrap(model.thumbnail(for: target), "Real thumbnail unavailable")
        let image = try XCTUnwrap(UIImage(data: preview.image))
        XCTAssertGreaterThan(image.size.width, 1)
        XCTAssertLessThanOrEqual(image.size.width, 320)
        try show(model, size: .large)
        try await Task.sleep(for: .milliseconds(400))
        capture("real-sandbox-gallery-thumbnail")
        XCTAssertNil(model.lease); XCTAssertNil(model.pageRequest)
    }
    private func captureDesktop(_ name: String, base: URL, fixtureCompositor: Bool = true) async throws {
        func findWebView(_ view: UIView) -> WKWebView? {
            if let web = view as? WKWebView { return web }
            return view.subviews.lazy.compactMap { findWebView($0) }.first
        }
        let web = try XCTUnwrap(window.flatMap(findWebView))
        try await Task.sleep(for: .milliseconds(250))
        let rendered = try await web.takeSnapshot(configuration: nil)
        let pixels = try XCTUnwrap(rendered.cgImage?.dataProvider?.data) as Data
        let colors = Set(stride(from: 0, to: max(0, pixels.count - 4), by: 4).map {
            Array(pixels[$0..<($0 + 4)])
        })
        XCTAssertGreaterThan(colors.count, 16, "Native WebKit snapshot must contain the desktop, not a blank canvas")
        let framebuffer = XCTAttachment(image: rendered)
        framebuffer.name = name + "-webkit"; framebuffer.lifetime = .keepAlways; add(framebuffer)
        if !fixtureCompositor {
            capture(name + "-native")
            return
        }
        // Hosted unit tests cannot request XCUIScreen captures. The explicit
        // loopback fixture captures the simulator compositor while we wait.
        var request = URLRequest(url: base.appending(path: "proof/screenshot/" + name))
        request.httpMethod = "POST"
        request.setValue("Bearer local-sandbox-view-proof", forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
    }
    private func capture(_ name: String) {
        guard let window else { return }
        window.layoutIfNeeded()
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        let attachment = XCTAttachment(image: image); attachment.name = name
        attachment.lifetime = .keepAlways; add(attachment)
    }
}

@MainActor
private final class SandboxProofStyle: ObservableObject {
    @Published var size: DynamicTypeSize
    init(size: DynamicTypeSize) { self.size = size }
}

private struct SandboxProofSurface: View {
    @ObservedObject var model: JarvisSandboxViewModel
    @ObservedObject var style: SandboxProofStyle
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("Workspace").font(JarvisTypography.pageTitle)
                JarvisSandboxView(model: model)
            }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
        }.background(JarvisColors.background).environment(\.dynamicTypeSize, style.size).preferredColorScheme(.light)
    }
}

@MainActor
private final class SandboxFixture: JarvisSandboxAccess {
    var items: [JarvisSandboxTarget] = []
    var opens = 0
    var closed: [UUID] = []
    var holdOpen = false
    var pending: CheckedContinuation<JarvisSandboxLease, Error>?
    var openError: Error?
    var statusError: Error = CancellationError()
    var statusReply: JarvisSandboxLease?
    var pageURL = URL(string: "http://127.0.0.1:1/unreachable")!
    var commands: [[String: Any]] = []
    var thumbnailCalls = 0
    var thumbnailReply: JarvisSandboxThumbnail?
    func sandboxThumbnail(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxThumbnail {
        thumbnailCalls += 1
        guard let thumbnailReply else { throw JarvisAPIError(code: "sandbox_thumbnail_unavailable", message: "No preview") }
        return thumbnailReply
    }
    func controlApp(sessionID: UUID, operationID: UUID, command: [String: Any]) async throws -> JarvisAppControlResult {
        commands.append(command)
        let data = try JSONSerialization.data(withJSONObject: ["outcome": "accepted", "summary": "Pending display",
            "presentation": ["request_id": operationID.uuidString, "state": "pending"]])
        return try JSONDecoder().decode(JarvisAppControlResult.self, from: data)
    }
    func target() -> JarvisSandboxTarget {
        JarvisSandboxTarget(taskID: UUID(), allocationID: UUID(), generation: String(repeating: "a", count: 64),
            title: items.isEmpty ? "Research notes" : "A new video", expiresAt: "2026-11-04T00:00:00Z")
    }
    func sandboxChoices(sessionID: UUID) async throws -> JarvisSandboxChoices { JarvisSandboxChoices(items: items, hasMore: false) }
    func openSandboxView(sessionID: UUID, allocationID: UUID, generation: String) async throws -> JarvisSandboxLease {
        opens += 1
        if let openError { throw openError }
        if holdOpen { return try await withCheckedThrowingContinuation { pending = $0 } }
        throw JarvisAPIError(code: "sandbox_not_running", message: "Not running")
    }
    func release(_ target: JarvisSandboxTarget) {
        pending?.resume(returning: JarvisSandboxLease(id: UUID(), target: target, expiresAt: target.expiresAt ?? "")); pending = nil
    }
    func sandboxViewStatus(sessionID: UUID, id: UUID) async throws -> JarvisSandboxLease {
        if let statusReply { return statusReply }; throw statusError
    }
    func closeSandboxView(sessionID: UUID, id: UUID) async throws { closed.append(id) }
    func sandboxPageRequest(sessionID: UUID, id: UUID) throws -> URLRequest { URLRequest(url: pageURL) }
}

#if DEBUG && targetEnvironment(simulator)
/// Real-account input qualification, not an isolated replay or hardware test.
/// The existing native ADM replaces physical input only. Product routing,
/// authentication, Live transport and presentation handlers remain unchanged.
@MainActor
final class JarvisNativeLiveSandboxTests: XCTestCase {
    private var application: JarvisAppViewModel?
    private var audioDevice: JarvisReplayAudioDevice?
    private var window: UIWindow?
    private var previousWindow: UIWindow?
    private var store: JarvisSessionStore?
    private var defaultsName: String?
    private var previousRoute: String?
    private var restoreRoute = false
    private var output: URL?
    private var timeline: [[String: Any]] = []
    private var ownsDataOnlySession = false

    func testServingAudioHookAttachesAndDetachesWithoutEvaluationScope() throws {
        let audio = JarvisAudioSessionViewModel()
        let first = JarvisReplayAudioDevice(maxDuration: 10)
        let second = JarvisReplayAudioDevice(maxDuration: 10)
        defer { _ = first.close(); _ = second.close() }
        try audio.useServingTestAudioDevice(first)
        XCTAssertNil(audio.replayRunID)
        XCTAssertNil(audio.activeSession)
        XCTAssertEqual(audio.connectionState, .disconnected)
        XCTAssertThrowsError(try audio.useServingTestAudioDevice(second))
        XCTAssertFalse(first.isRetired)
        XCTAssertFalse(second.isRetired)
        try audio.useServingTestAudioDevice(nil)
        XCTAssertTrue(first.isRetired)
        try audio.useServingTestAudioDevice(second)
        try audio.useServingTestAudioDevice(nil)
        XCTAssertTrue(second.isRetired)
    }

    func testServingAudioHookRejectsRetiredDeviceAndNonIdleState() async throws {
        let audio = JarvisAudioSessionViewModel()
        let retired = JarvisReplayAudioDevice(maxDuration: 10)
        _ = retired.close()
        XCTAssertThrowsError(try audio.useServingTestAudioDevice(retired))
        let device = JarvisReplayAudioDevice(maxDuration: 10)
        defer { _ = device.close() }
        try audio.useServingTestAudioDevice(device)
        // No configuration: this fails locally, without any provider request.
        audio.startSession()
        XCTAssertNotEqual(audio.connectionState, .disconnected)
        XCTAssertThrowsError(try audio.useServingTestAudioDevice(nil))
        XCTAssertThrowsError(try audio.useServingTestAudioDevice(device))
        audio.stopSession()
        try await wait("local stop", until: .now.addingTimeInterval(5)) { !audio.shouldShowStopJarvis }
        try audio.useServingTestAudioDevice(nil)
        XCTAssertTrue(device.isRetired)
        XCTAssertNil(audio.replayRunID)
    }

    func testRealPausedSandboxResumeAndSharedStop() async throws {
        try await runDataOnlySandboxScenario(switching: false)
    }

    func testRealExistingSandboxSwitchAndSharedStop() async throws {
        try await runDataOnlySandboxScenario(switching: true)
    }

    private func runDataOnlySandboxScenario(switching: Bool) async throws {
        guard let path = ProcessInfo.processInfo.environment["JARVIS_NATIVE_LIVE_SANDBOX_MANIFEST"] else {
            throw XCTSkip("Requires explicit existing paused sandbox and private serving manifest")
        }
        executionTimeAllowance = 120
        let manifest = try JSONDecoder().decode(NativeLiveSandboxManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        guard manifest.nonIsolatedRealAccount, manifest.backendURL.scheme == "http",
              manifest.backendURL.host == "127.0.0.1", !manifest.ownerID.isEmpty else {
            throw NativeSandboxFailure("Explicit ordinary-account loopback configuration required")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: manifest.bearerTokenFile)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600 else {
            throw NativeSandboxFailure("Credential must be a private regular file")
        }
        let token = try String(contentsOfFile: manifest.bearerTokenFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else { throw NativeSandboxFailure("Missing credential") }
        let directory = URL(fileURLWithPath: manifest.outputDirectory, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw NativeSandboxFailure("Evidence directory must be new")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        output = directory
        let configuration = JarvisConfiguration(backendBaseURL: manifest.backendURL, staticBearerToken: token)
        let api = JarvisAPIClient(configuration: configuration)
        guard try await api.getActiveSession() == nil else {
            throw NativeSandboxFailure("Refusing to adopt or stop an existing active session")
        }
        record("data_only_preflight", ["physical_audio": false, "voice_started": false,
            "allocation_id": manifest.allocationID.uuidString.lowercased()])
        let name = "jarvis.native-paused-sandbox." + UUID().uuidString
        defaultsName = name
        let store = JarvisSessionStore(defaults: try XCTUnwrap(UserDefaults(suiteName: name)),
            keychain: JarvisKeychain(service: name))
        self.store = store
        let audio = JarvisAudioSessionViewModel()
        let product = JarvisAppViewModel(store: store, audioSessionViewModel: audio,
            workspaceJournalFactory: { context in
                guard context.hasValidIdentity, context.ownerID == manifest.ownerID else {
                    throw NativeSandboxFailure("Authenticated owner mismatch")
                }
                return try JarvisWorkspaceLocalStore(ownerKey: JarvisWorkspaceLocalStore.ownerKey(for: context),
                    root: directory.appendingPathComponent("journal"), allowsFileBytes: false)
            })
        application = product
        product.saveSetup(backendBaseURL: configuration.backendBaseURL.absoluteString, staticBearerToken: token)
        guard product.configuration == configuration else { throw NativeSandboxFailure("Setup failed") }
        try mount(product)
        product.productTab = .workspace
        ownsDataOnlySession = true
        product.startSession()
        let deadline = Date().addingTimeInterval(90)
        try await wait("data-only authenticated workspace ready", until: min(deadline, .now.addingTimeInterval(25))) {
            product.workspace?.journalReady == true && product.activeSession != nil
                && product.workspace?.sessionID == (product.activeSession?.id).flatMap(UUID.init(uuidString:))
        }
        let sessionID = try XCTUnwrap((product.activeSession?.id).flatMap(UUID.init(uuidString:)))
        let viewer = try XCTUnwrap(product.workspace?.sandbox)
        if switching {
            let secondID = try XCTUnwrap(manifest.secondAllocationID, "Switch requires a distinct retained allocation")
            XCTAssertNotEqual(secondID, manifest.allocationID)
            let before = try await api.sandboxChoices(sessionID: sessionID)
            let first = try XCTUnwrap(before.items.first { $0.id == manifest.allocationID })
            let second = try XCTUnwrap(before.items.first { $0.id == secondID })
            guard first.canOpen, second.canOpen, first.state == "running", second.state == "running" else {
                throw NativeSandboxFailure("Both exact retained desktops must already be running")
            }
            func leaseStatus(_ id: UUID) async throws -> Int {
                var components = try XCTUnwrap(URLComponents(url: configuration.backendBaseURL
                    .appendingPathComponent("sandbox-views/" + id.uuidString.lowercased()), resolvingAgainstBaseURL: false))
                components.queryItems = [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())]
                var request = URLRequest(url: try XCTUnwrap(components.url))
                request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
                request.timeoutInterval = 10
                let (_, response) = try await URLSession.shared.data(for: request)
                return try XCTUnwrap(response as? HTTPURLResponse).statusCode
            }
            record("switch_ready", ["session_id": sessionID.uuidString.lowercased(),
                "first_allocation_id": first.id.uuidString.lowercased(), "second_allocation_id": second.id.uuidString.lowercased()])
            var leases: [JarvisSandboxLease] = []
            for (index, target) in [first, second].enumerated() {
                await viewer.requestView(target)
                try await wait("switch exact desktop \(index + 1)", until: min(deadline, .now.addingTimeInterval(35))) {
                    viewer.phase == .viewing && viewer.lease?.target.id == target.id
                }
                let lease = try XCTUnwrap(viewer.lease)
                XCTAssertEqual(lease.target.generation, target.generation)
                leases.append(lease)
                record("switch_desktop_presented", ["index": index + 1, "allocation_id": target.id.uuidString.lowercased(),
                    "view_id": lease.id.uuidString.lowercased()])
                capture("switch-desktop-\(index + 1)")
            }
            XCTAssertNotEqual(leases[0].id, leases[1].id)
            let oldStatus = try await leaseStatus(leases[0].id)
            guard [404, 410].contains(oldStatus) else { throw NativeSandboxFailure("Previous view survived switching") }
            record("switch_revoked_original", ["http_status": oldStatus])
            await viewer.stop()
            try await wait("switch stop removed view", until: min(deadline, .now.addingTimeInterval(10))) {
                viewer.lease == nil && viewer.pageRequest == nil && !viewer.isShowingScreen
                    && product.workspace?.sandbox === viewer
            }
            let stoppedStatus = try await leaseStatus(leases[1].id)
            guard [404, 410].contains(stoppedStatus) else { throw NativeSandboxFailure("Second view survived stop") }
            let after = try await api.sandboxChoices(sessionID: sessionID)
            guard Set(after.items.map(\.id)) == Set(before.items.map(\.id)),
                  [first.id, second.id].allSatisfy({ id in after.items.first(where: { $0.id == id })?.state == "running" }),
                  !audio.isConnected, audio.activeSession == nil else {
                throw NativeSandboxFailure("Switch changed inventory, stopped a computer or started voice")
            }
            XCTAssertNil(product.workspace?.tasksError)
            XCTAssertNil(product.workspace?.inboxError)
            record("switch_stop_confirmed", ["view_http_status": stoppedStatus, "both_sandboxes_running": true,
                "legacy_disabled_errors_absent": true, "voice_started": false])
            capture("switch-stopped-gallery")
            try persistEvidence()
            return
        }
        try await wait("paused gallery loaded", until: min(deadline, .now.addingTimeInterval(15))) {
            viewer.phase == .choosing && viewer.choices.contains { $0.id == manifest.allocationID }
        }
        let target = try XCTUnwrap(viewer.choices.first { $0.id == manifest.allocationID })
        guard target.state == "paused", target.capability == "desktop", target.canResumeAndView == true,
              viewer.lease == nil, !audio.isConnected, audio.activeSession == nil else {
            throw NativeSandboxFailure("Require exact paused desktop with no voice or current view")
        }
        record("paused_gallery", ["session_id": sessionID.uuidString.lowercased(), "generation": target.generation])
        capture("paused-sandbox-gallery")
        await viewer.requestView(target)
        XCTAssertEqual(viewer.resumeCandidate?.id, target.id)
        XCTAssertNil(viewer.lease)
        let before = try await api.sandboxChoices(sessionID: sessionID)
        guard before.items.first(where: { $0.id == target.id })?.state == "paused" else {
            throw NativeSandboxFailure("Gallery or confirmation resumed the sandbox before explicit consent")
        }
        capture("paused-sandbox-confirmation")
        record("explicit_resume_requested")
        await viewer.resumeAndView()
        try await wait("resumed exact desktop presented", until: min(deadline, .now.addingTimeInterval(40))) {
            if let lease = viewer.lease, lease.target.id != target.id {
                throw NativeSandboxFailure("Resume selected a different sandbox")
            }
            return viewer.phase == .viewing && viewer.lease?.target.id == target.id
        }
        let lease = try XCTUnwrap(viewer.lease)
        XCTAssertEqual(lease.target.generation, target.generation)
        record("resumed_exact_desktop", ["view_id": lease.id.uuidString.lowercased(),
            "allocation_id": lease.target.id.uuidString.lowercased()])
        capture("resumed-existing-sandbox")
        await viewer.stop()
        try await wait("shared stop removed view", until: min(deadline, .now.addingTimeInterval(10))) {
            viewer.lease == nil && viewer.pageRequest == nil && !viewer.isShowingScreen
                && product.workspace?.sandbox === viewer
        }
        var components = try XCTUnwrap(URLComponents(url: configuration.backendBaseURL
            .appendingPathComponent("sandbox-views/" + lease.id.uuidString.lowercased()), resolvingAgainstBaseURL: false))
        components.queryItems = [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())]
        var request = URLRequest(url: try XCTUnwrap(components.url))
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 10
        let (_, response) = try await URLSession.shared.data(for: request)
        let status = try XCTUnwrap(response as? HTTPURLResponse).statusCode
        guard [404, 410].contains(status) else { throw NativeSandboxFailure("View lease remains accessible") }
        let after = try await api.sandboxChoices(sessionID: sessionID)
        guard Set(after.items.map(\.id)) == Set(before.items.map(\.id)),
              after.items.first(where: { $0.id == target.id })?.state == "running",
              !audio.isConnected, audio.activeSession == nil else {
            throw NativeSandboxFailure("Resume changed inventory, stopped the computer or started voice")
        }
        record("shared_stop_confirmed", ["view_http_status": status, "sandbox_preserved_running": true,
            "voice_started": false])
        capture("resumed-sandbox-stopped-view")
        try persistEvidence()
    }

    func testRealAccountSpokenSandboxStartAndStop() async throws {
        guard let path = ProcessInfo.processInfo.environment["JARVIS_NATIVE_LIVE_SANDBOX_MANIFEST"] else {
            throw XCTSkip("Requires explicit real-account native voice lane and private manifest")
        }
        executionTimeAllowance = 150
        let manifest = try JSONDecoder().decode(NativeLiveSandboxManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        guard manifest.nonIsolatedRealAccount, manifest.backendURL.scheme == "http",
              manifest.backendURL.host == "127.0.0.1", !manifest.ownerID.isEmpty else {
            throw NativeSandboxFailure("Explicit ordinary-account loopback configuration required")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: manifest.bearerTokenFile)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600 else {
            throw NativeSandboxFailure("Credential must be a private regular file")
        }
        let token = try String(contentsOfFile: manifest.bearerTokenFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else { throw NativeSandboxFailure("Missing credential") }
        let startPCM = try manifest.startSpeech.loadPCM16()
        let stopPCM = try manifest.stopSpeech.loadPCM16()
        let directory = URL(fileURLWithPath: manifest.outputDirectory, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw NativeSandboxFailure("Evidence directory must be new")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        output = directory
        let configuration = JarvisConfiguration(backendBaseURL: manifest.backendURL, staticBearerToken: token)
        let api = JarvisAPIClient(configuration: configuration)
        guard try await api.getActiveSession() == nil else {
            throw NativeSandboxFailure("Refusing to adopt or stop an existing active session")
        }
        record("preflight", ["non_isolated_real_account": true, "physical_audio": false,
            "allocation_id": manifest.allocationID.uuidString.lowercased(),
            "start_asset_sha256": manifest.startSpeech.sha256, "stop_asset_sha256": manifest.stopSpeech.sha256])
        let name = "jarvis.native-live-sandbox." + UUID().uuidString
        defaultsName = name
        let store = JarvisSessionStore(defaults: try XCTUnwrap(UserDefaults(suiteName: name)),
            keychain: JarvisKeychain(service: name))
        self.store = store
        let audio = JarvisAudioSessionViewModel()
        let device = JarvisReplayAudioDevice(maxDuration: 120)
        audioDevice = device
        try audio.useServingTestAudioDevice(device)
        previousRoute = UserDefaults.standard.string(forKey: JarvisVoiceRouteChoice.preferenceKey)
        restoreRoute = true
        audio.setPreferredVoiceRoute(.live)
        audio.captureMode = .autoDetect
        let product = JarvisAppViewModel(store: store, audioSessionViewModel: audio,
            workspaceJournalFactory: { context in
                guard context.hasValidIdentity, context.ownerID == manifest.ownerID else {
                    throw NativeSandboxFailure("Authenticated owner mismatch")
                }
                return try JarvisWorkspaceLocalStore(ownerKey: JarvisWorkspaceLocalStore.ownerKey(for: context),
                    root: directory.appendingPathComponent("journal"), allowsFileBytes: false)
            })
        application = product
        product.saveSetup(backendBaseURL: configuration.backendBaseURL.absoluteString, staticBearerToken: token)
        guard product.configuration == configuration else { throw NativeSandboxFailure("Setup failed") }
        try mount(product)
        let deadline = Date().addingTimeInterval(120)
        record("ordinary_audio_session_start")
        audio.startSession() // Deliberately not unified startJarvis: no camera/glasses input.
        try await wait("Live and authenticated workspace ready", until: min(deadline, .now.addingTimeInterval(40))) {
            audio.isConnected && product.workspace?.journalReady == true
                && product.workspace?.sessionID == (audio.activeSession?.id).flatMap(UUID.init(uuidString:))
        }
        XCTAssertNil(audio.replayRunID)
        let sessionID = try XCTUnwrap((audio.activeSession?.id).flatMap(UUID.init(uuidString:)))
        let workspace = try XCTUnwrap(product.workspace)
        let viewer = workspace.sandbox
        let choices = try await api.sandboxChoices(sessionID: sessionID)
        guard !choices.hasMore, choices.items.filter({ $0.canOpen }).map(\.id) == [manifest.allocationID],
              let target = choices.items.first(where: { $0.id == manifest.allocationID }),
              target.state == "running", target.capability == "desktop", viewer.lease == nil else {
            throw NativeSandboxFailure("Require the exact sole viewable existing desktop and no active app view")
        }
        record("ready", ["session_id": sessionID.uuidString.lowercased(),
            "audio_session_id": audio.audioSessionId ?? "", "generation": target.generation])
        try device.enqueuePCM16(startPCM)
        record("spoken_start_enqueued")
        try await wait("spoken request presented desktop", until: min(deadline, .now.addingTimeInterval(45))) {
            if let lease = viewer.lease, lease.target.id != manifest.allocationID {
                throw NativeSandboxFailure("Voice selected a different sandbox")
            }
            return viewer.phase == .viewing && viewer.lease?.target.id == manifest.allocationID
                && product.productTab == .workspace
        }
        let lease = try XCTUnwrap(viewer.lease)
        XCTAssertEqual(lease.target.generation, target.generation)
        record("exact_desktop_presented", ["view_id": lease.id.uuidString.lowercased(),
            "allocation_id": lease.target.id.uuidString.lowercased()])
        capture("voice-requested-desktop")
        try await wait("first speech drained", until: deadline) { device.inputIdle }
        try device.enqueuePCM16(stopPCM)
        record("spoken_stop_enqueued")
        try await wait("spoken request closed same view", until: min(deadline, .now.addingTimeInterval(35))) {
            viewer.lease == nil && viewer.pageRequest == nil && !viewer.isShowingScreen
                && product.workspace?.sandbox === viewer
        }
        // Read-only verification; never invoke a close control to make this pass.
        var request = URLRequest(url: configuration.backendBaseURL
            .appendingPathComponent("sandbox-views/" + lease.id.uuidString.lowercased()))
        var components = try XCTUnwrap(URLComponents(url: request.url!, resolvingAgainstBaseURL: false))
        components.queryItems = [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())]
        request.url = components.url
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 10
        let (_, response) = try await URLSession.shared.data(for: request)
        let status = try XCTUnwrap(response as? HTTPURLResponse).statusCode
        guard [404, 410].contains(status) else { throw NativeSandboxFailure("View lease remains accessible") }
        let after = try await api.sandboxChoices(sessionID: sessionID)
        guard Set(after.items.map(\.id)) == Set(choices.items.map(\.id)),
              after.items.first(where: { $0.id == manifest.allocationID })?.state == "running" else {
            throw NativeSandboxFailure("Sandbox inventory changed or viewing stopped the computer")
        }
        record("spoken_stop_confirmed", ["view_http_status": status, "sandbox_preserved_running": true])
        capture("voice-stopped-desktop")
        try persistEvidence()
    }

    override func tearDown() async throws {
        if let application {
            if let lease = application.workspace?.sandbox.lease,
               let configuration = application.configuration, let sessionID = application.workspace?.sessionID {
                do {
                    try await JarvisAPIClient(configuration: configuration).closeSandboxView(sessionID: sessionID, id: lease.id)
                    record("cleanup_owned_view", ["view_id": lease.id.uuidString.lowercased()])
                } catch { record("cleanup_view_failed", ["error_type": String(describing: type(of: error))]) }
            }
            if ownsDataOnlySession {
                application.stopSession()
                do {
                    try await wait("owned data-only session cleanup", until: .now.addingTimeInterval(20)) {
                        application.activeSession == nil
                    }
                    record("data_session_closed")
                } catch {
                    record("cleanup_data_session_failed", ["error_type": String(describing: type(of: error))])
                    XCTFail("Data-only test session cleanup did not finish")
                }
            } else {
            application.audioSessionViewModel.stopSession()
            do {
                try await wait("owned session cleanup", until: .now.addingTimeInterval(20)) {
                    !application.audioSessionViewModel.shouldShowStopJarvis
                }
                try application.audioSessionViewModel.useServingTestAudioDevice(nil)
                record("session_closed")
            } catch {
                record("cleanup_failed", ["error_type": String(describing: type(of: error))])
                XCTFail("Native test session cleanup did not finish")
            }
            }
        }
        if let audioDevice { record("audio_fenced", ["success": audioDevice.close()]) }
        try persistEvidence()
        window?.isHidden = true; window?.rootViewController = nil; window = nil
        previousWindow?.makeKeyAndVisible(); previousWindow = nil
        application = nil; audioDevice = nil
        store?.clearConfiguration(); store = nil
        if let defaultsName { UserDefaults.standard.removePersistentDomain(forName: defaultsName) }
        if restoreRoute {
            if let previousRoute { UserDefaults.standard.set(previousRoute, forKey: JarvisVoiceRouteChoice.preferenceKey) }
            else { UserDefaults.standard.removeObject(forKey: JarvisVoiceRouteChoice.preferenceKey) }
        }
        try await super.tearDown()
    }

    private func wait(_ label: String, until deadline: Date, condition: () throws -> Bool) async throws {
        while Date() < deadline {
            if try condition() { return }
            try await Task.sleep(for: .milliseconds(50))
        }
        record("timeout", ["boundary": label])
        throw NativeSandboxFailure("Timed out: " + label)
    }
    private func mount(_ product: JarvisAppViewModel) throws {
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive })
        previousWindow = scene.windows.first { $0.isKeyWindow }
        let window = UIWindow(windowScene: scene)
        window.rootViewController = UIHostingController(rootView:
            JarvisRootView(viewModel: product).environment(\.scenePhase, .active))
        self.window = window; window.makeKeyAndVisible(); window.layoutIfNeeded()
    }
    private func record(_ event: String, _ details: [String: Any] = [:]) {
        timeline.append(details.merging(["event": event, "at": Date().timeIntervalSince1970,
            "monotonic": ProcessInfo.processInfo.systemUptime]) { _, new in new })
    }
    private func persistEvidence() throws {
        guard let output else { return }
        var receipt: [String: Any] = ["non_isolated_real_account": true,
            "physical_playback_measured": false, "timeline": timeline]
        if let audioDevice {
            let evidence = audioDevice.evidence
            receipt["audio"] = ["input_blocks": evidence.inputBlocks, "decoded_blocks": evidence.decodedBlocks,
                "input_first_non_silent": evidence.inputFirstNonSilent,
                "input_last_non_silent": evidence.inputLastNonSilent,
                "truncated": evidence.truncated, "callback_failure": evidence.callbackFailure as Any? ?? NSNull()]
            try evidence.decodedPCM16.write(to: output.appendingPathComponent("decoded-output.pcm16"), options: .atomic)
        }
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("native-voice-sandbox.json"), options: .atomic)
    }
    private func capture(_ name: String) {
        guard let window else { return }
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        let attachment = XCTAttachment(image: image); attachment.name = name
        attachment.lifetime = .keepAlways; add(attachment)
    }
}

private struct NativeLiveSandboxManifest: Decodable {
    let nonIsolatedRealAccount: Bool
    let backendURL: URL
    let bearerTokenFile: String
    let ownerID: String
    let allocationID: UUID
    let secondAllocationID: UUID?
    let outputDirectory: String
    let startSpeech: JarvisReplayAudioAsset
    let stopSpeech: JarvisReplayAudioAsset
}
private struct NativeSandboxFailure: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
#endif
