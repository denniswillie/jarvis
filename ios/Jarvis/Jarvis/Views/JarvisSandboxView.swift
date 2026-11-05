import SwiftUI
import WebKit

struct JarvisSandboxView: View {
    @ObservedObject var model: JarvisSandboxViewModel
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.jarvisDocumentViewport) private var viewport
    @Environment(\.jarvisDocumentPresentationActive) private var presentationActive
    @State private var screenFrame: CGRect = .null
    @State private var isMaximized = false
    @State private var fullscreenVisible = false

    private var visibleFrame: Bool {
        guard presentationActive else { return false }
        if fullscreenVisible { return true }
        guard !viewport.isNull, !screenFrame.isNull else { return false }
        let intersection = viewport.intersection(screenFrame)
        return !intersection.isNull && intersection.width >= 80 && intersection.height >= 32
    }

    var body: some View {
        Group {
            if model.isShowingScreen { screen }
            else if model.phase == .choosing { selection }
            else if model.phase == .loadingGallery {
                ProgressView("Checking your computers...").tint(JarvisColors.accent).padding(24)
            }
            else if model.phase == .failed {
                VStack(spacing: 12) {
                    Text(model.errorMessage ?? "Computer status is unavailable.")
                    Button("Refresh computers") { Task { await model.choose() } }
                }.padding(20)
            }
            else {
                JarvisSandboxConnectButton { Task { await model.choose() } }
            }
        }
        .alert(item: $model.notice) { notice in
            Alert(title: Text(notice.title), message: Text(notice.message), dismissButton: .default(Text("Got it")))
        }
        .confirmationDialog("Resume this computer?", isPresented: Binding(
            get: { model.resumeCandidate != nil }, set: { if !$0 { model.resumeCandidate = nil } }), titleVisibility: .visible) {
                Button("Resume and view") { Task { await model.resumeAndView() } }
                Button("Cancel", role: .cancel) { model.resumeCandidate = nil }
        } message: {
            Text("Resume the same retained computer and open its screen. No new computer will be created.")
        }
        .task { if model.phase == .idle { await model.choose() } }
        .onChange(of: model.phase) { _, _ in model.confirmVisibleFrame(visibleFrame) }
        .onChange(of: model.isShowingScreen) { _, showing in
            if !showing { isMaximized = false; fullscreenVisible = false }
        }
        .onChange(of: visibleFrame) { _, value in model.confirmVisibleFrame(value) }
    }

    private var screenHeaderLayout: AnyLayout {
        typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 12))
            : AnyLayout(HStackLayout(alignment: .top, spacing: 16))
    }

    private var screen: some View {
        VStack(alignment: .leading, spacing: 16) {
            screenHeaderLayout {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Sandbox").font(.system(.title2, design: .serif))
                    Text(model.phase == .viewing ? "Live · View only" : "View only")
                        .font(.caption).foregroundStyle(JarvisColors.secondaryText)
                }.frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 12) {
                    Button { isMaximized = true } label: {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 16, weight: .medium))
                            .frame(width: 44, height: 44)
                            .background(JarvisColors.background, in: Circle())
                    }
                    .disabled(model.pageRequest == nil)
                    .accessibilityLabel("Maximise sandbox")
                    .accessibilityHint("Show this computer full screen")
                    .accessibilityIdentifier("sandbox.maximize")
                    Button { Task { await model.stop() } } label: {
                        Image(systemName: "xmark").font(.system(size: 16, weight: .medium))
                            .frame(width: 44, height: 44)
                            .background(JarvisColors.background, in: Circle())
                    }
                    .accessibilityLabel("Close sandbox view")
                    .accessibilityIdentifier("sandbox.close")
                }.frame(maxWidth: typeSize.isAccessibilitySize ? .infinity : nil, alignment: .trailing)
            }
            JarvisSandboxScreenHost(model: model, typeSize: typeSize, isMaximized: $isMaximized,
                fullscreenVisible: $fullscreenVisible)
                .frame(height: typeSize.isAccessibilitySize ? 360 : 260)
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { screenFrame = $0 }
                .accessibilityIdentifier("sandbox.screen")
            Text("Keep talking to Jarvis to guide what happens next.")
                .font(.callout).foregroundStyle(JarvisColors.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .foregroundStyle(JarvisColors.primaryText)
        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 22))
        .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(JarvisColors.border, lineWidth: 1))
    }

    private var selection: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Agent computers").font(.system(.title2, design: .serif))
                    Text("Running and paused workspaces").font(.caption).foregroundStyle(JarvisColors.secondaryText)
                }
                Spacer()
                Button { Task { await model.choose() } } label: {
                    Image(systemName: "arrow.clockwise").frame(width: 44, height: 44)
                }.accessibilityLabel("Refresh computer status")
            }
            LazyVGrid(columns: typeSize.isAccessibilitySize ? [GridItem(.flexible())]
                : [GridItem(.adaptive(minimum: 145), spacing: 14)], spacing: 14) {
                ForEach(model.choices) { target in
                    Button { Task { await model.requestView(target) } } label: {
                        VStack(alignment: .leading, spacing: 12) {
                            ZStack {
                                LinearGradient(colors: [JarvisColors.softAccent, JarvisColors.background],
                                               startPoint: .topLeading, endPoint: .bottomTrailing)
                                if let preview = model.thumbnail(for: target), let image = UIImage(data: preview.image) {
                                    Image(uiImage: image).resizable().scaledToFill()
                                        .frame(height: 124).clipped()
                                        .accessibilityLabel("Last captured screen of \(target.title)")
                                    VStack { Spacer(); Text("Last preview").font(.caption2)
                                        .padding(.horizontal, 8).padding(.vertical, 3)
                                        .background(.regularMaterial, in: Capsule()).padding(6) }
                                } else {
                                Image(systemName: target.isDesktop ? "display" : "terminal")
                                    .font(.system(size: 34, weight: .ultraLight))
                                    .foregroundStyle(JarvisColors.accent)
                                VStack { Spacer(); Text(target.isPaused ? "Retained computer" : "Screen preview unavailable")
                                    .font(.caption2).foregroundStyle(JarvisColors.secondaryText).padding(10) }
                                }
                            }
                            .frame(height: 124).clipShape(RoundedRectangle(cornerRadius: 13))
                            Text(target.title).font(.system(.headline, design: .serif))
                                .foregroundStyle(JarvisColors.primaryText).multilineTextAlignment(.leading)
                                .fixedSize(horizontal: false, vertical: true)
                            HStack(spacing: 6) {
                                Circle().fill(target.canOpen ? JarvisColors.accent : JarvisColors.secondaryText)
                                    .frame(width: 5, height: 5)
                                Text(target.stateLabel).font(.caption)
                            }.foregroundStyle(JarvisColors.secondaryText)
                            Text(target.isPaused ? "Resume and view" : target.canOpen ? "View screen" : "No screen available")
                                .font(.caption.weight(.medium)).foregroundStyle(JarvisColors.accent)
                        }
                        .frame(maxWidth: .infinity, alignment: .topLeading).padding(12)
                        .background(JarvisColors.background, in: RoundedRectangle(cornerRadius: 18))
                        .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(JarvisColors.border, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .disabled(!target.isDesktop || (!target.canOpen && !target.isPaused))
                    .accessibilityLabel("\(target.title), \(target.stateLabel)")
                    .task(id: target.generation + target.stateLabel) { await model.loadThumbnail(target) }
                }
            }
            if model.hasMore {
                Button(model.loadingMore ? "Loading..." : "More computers") { Task { await model.loadMore() } }
                    .disabled(model.loadingMore)
            }
        }
        .tint(JarvisColors.accent)
        .padding(20)
        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 22))
    }
}

/// The hosting controller moves between containers; its WebKit view and live
/// connection survive expansion, rotation and return to the Workspace card.
private struct JarvisSandboxScreenHost: UIViewControllerRepresentable {
    @ObservedObject var model: JarvisSandboxViewModel
    let typeSize: DynamicTypeSize
    @Binding var isMaximized: Bool
    @Binding var fullscreenVisible: Bool

    func makeUIViewController(context: Context) -> JarvisSandboxScreenController {
        JarvisSandboxScreenController(model: model, typeSize: typeSize)
    }
    func updateUIViewController(_ controller: JarvisSandboxScreenController, context: Context) {
        controller.maximizedChanged = { isMaximized = $0 }
        controller.visibilityChanged = { fullscreenVisible = $0 }
        controller.setTypeSize(typeSize)
        controller.setMaximized(isMaximized)
    }
    static func dismantleUIViewController(_ controller: JarvisSandboxScreenController, coordinator: ()) {
        controller.shutdown()
    }
}

private struct JarvisSandboxScreenContent: View {
    @ObservedObject var model: JarvisSandboxViewModel
    var typeSize: DynamicTypeSize
    var body: some View {
        ZStack {
            JarvisColors.primaryText
            if let request = model.pageRequest {
                JarvisSandboxWebView(request: request, pageID: model.pageID) { event, pageID in
                    model.pageEvent(event, pageID: pageID)
                }
                .id(model.pageID)
            }
            if model.phase != .viewing {
                JarvisColors.primaryText
                GeometryReader { geometry in
                    ScrollView {
                        VStack(spacing: 14) {
                            if model.phase == .failed {
                                Image(systemName: "wifi.exclamationmark").font(.system(size: 24))
                                Text("Can’t connect").font(.headline)
                                    .fixedSize(horizontal: false, vertical: true)
                                Text(model.errorMessage ?? "Try connecting again.")
                                    .font(.callout).multilineTextAlignment(.center)
                                    .fixedSize(horizontal: false, vertical: true)
                                Button("Try again") { Task { await model.retry() } }
                                    .buttonStyle(.bordered).tint(.white)
                                    .accessibilityIdentifier("sandbox.retry")
                            } else {
                                ProgressView().tint(.white)
                                Text(model.phase == .reconnecting ? "Reconnecting…" : "Connecting to your screen…")
                                    .font(.callout).multilineTextAlignment(.center)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .foregroundStyle(.white.opacity(0.9)).padding(24)
                        .frame(minHeight: geometry.size.height)
                    }
                    .scrollBounceBehavior(.basedOnSize)
                }
            }
        }
        .environment(\.dynamicTypeSize, typeSize)
    }
}

@MainActor
final class JarvisSandboxScreenController: UIViewController {
    private var content: UIHostingController<JarvisSandboxScreenContent>?
    private(set) var fullscreen: JarvisSandboxFullscreenController?
    var maximizedChanged: ((Bool) -> Void)?
    var visibilityChanged: ((Bool) -> Void)?
    private var wantsMaximized = false
    private var retiring = false
    private var dismissing = false
    private var originalOrientation: UIInterfaceOrientation = .unknown

    init(model: JarvisSandboxViewModel, typeSize: DynamicTypeSize) {
        content = UIHostingController(rootView: JarvisSandboxScreenContent(model: model, typeSize: typeSize))
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        embedContent(in: self)
    }
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        setMaximized(wantsMaximized)
    }

    func setTypeSize(_ typeSize: DynamicTypeSize) {
        guard content?.rootView.typeSize != typeSize else { return }
        content?.rootView.typeSize = typeSize
    }

    func setMaximized(_ value: Bool) {
        wantsMaximized = value
        guard !retiring, !dismissing, isViewLoaded, view.window != nil else { return }
        if value, fullscreen == nil, presentedViewController == nil {
            let expanded = JarvisSandboxFullscreenController()
            originalOrientation = view.window?.windowScene?.interfaceOrientation ?? .unknown
            expanded.onMinimize = { [weak self] in self?.minimize() }
            expanded.onVisible = { [weak self] in self?.visibilityChanged?(true) }
            fullscreen = expanded
            expanded.loadViewIfNeeded()
            embedContent(in: expanded)
            present(expanded, animated: !UIAccessibility.isReduceMotionEnabled) { [self] in
                if !self.wantsMaximized { self.minimize() }
            }
        } else if !value, fullscreen != nil { minimize() }
    }

    func minimize(animated: Bool = true) {
        wantsMaximized = false
        guard let expanded = fullscreen, !dismissing, !expanded.isBeingPresented else { return }
        dismissing = true
        let scene = expanded.view.window?.windowScene
        expanded.dismiss(animated: animated && !UIAccessibility.isReduceMotionEnabled) { [self] in
            self.fullscreen = nil; self.dismissing = false
            self.embedContent(in: self)
            self.visibilityChanged?(false)
            self.maximizedChanged?(false)
            if let mask = self.originalOrientation.singleOrientationMask {
                self.setNeedsUpdateOfSupportedInterfaceOrientations()
                scene?.requestGeometryUpdate(.iOS(interfaceOrientations: mask)) { _ in }
            }
        }
    }

    func shutdown() {
        retiring = true
        // Clear protected pixels immediately, including during a modal animation.
        fullscreen?.view.isHidden = true
        content?.willMove(toParent: nil)
        content?.view.removeFromSuperview()
        content?.removeFromParent()
        content = nil
        minimize(animated: false)
    }

    private func embedContent(in parent: UIViewController) {
        guard !retiring, let content else { return }
        content.willMove(toParent: nil)
        content.view.removeFromSuperview()
        content.removeFromParent()
        parent.addChild(content)
        content.view.backgroundColor = .black
        content.view.translatesAutoresizingMaskIntoConstraints = false
        parent.view.insertSubview(content.view, at: 0)
        let bounds = parent === self ? parent.view.layoutMarginsGuide : parent.view.safeAreaLayoutGuide
        if parent === self { parent.view.directionalLayoutMargins = .zero }
        NSLayoutConstraint.activate([
            content.view.leadingAnchor.constraint(equalTo: bounds.leadingAnchor),
            content.view.trailingAnchor.constraint(equalTo: bounds.trailingAnchor),
            content.view.topAnchor.constraint(equalTo: bounds.topAnchor),
            content.view.bottomAnchor.constraint(equalTo: bounds.bottomAnchor)
        ])
        content.didMove(toParent: parent)
    }
}

@MainActor
final class JarvisSandboxFullscreenController: UIViewController {
    var onMinimize: (() -> Void)?
    var onVisible: (() -> Void)?
    override var prefersStatusBarHidden: Bool { true }
    override var prefersHomeIndicatorAutoHidden: Bool { true }
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .allButUpsideDown }

    init() {
        super.init(nibName: nil, bundle: nil)
        modalPresentationStyle = .overFullScreen
        modalTransitionStyle = .crossDissolve
        modalPresentationCapturesStatusBarAppearance = true
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        view.accessibilityIdentifier = "sandbox.fullscreen"
        var configuration = UIButton.Configuration.filled()
        configuration.image = UIImage(systemName: "arrow.down.right.and.arrow.up.left")
        configuration.baseForegroundColor = .white
        configuration.baseBackgroundColor = UIColor.black.withAlphaComponent(0.65)
        configuration.cornerStyle = .capsule
        let button = UIButton(configuration: configuration, primaryAction: UIAction { [weak self] _ in self?.onMinimize?() })
        button.accessibilityLabel = "Minimise sandbox"
        button.accessibilityHint = "Return to Workspace"
        button.accessibilityIdentifier = "sandbox.minimize"
        button.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(button)
        NSLayoutConstraint.activate([
            button.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            button.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            button.widthAnchor.constraint(equalToConstant: 48), button.heightAnchor.constraint(equalToConstant: 48)
        ])
    }
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        onVisible?()
        if traitCollection.userInterfaceIdiom == .phone,
           let scene = view.window?.windowScene, !scene.interfaceOrientation.isLandscape {
            setNeedsUpdateOfSupportedInterfaceOrientations()
            scene.requestGeometryUpdate(.iOS(interfaceOrientations: .landscape)) { _ in }
        }
    }
    override func accessibilityPerformEscape() -> Bool { onMinimize?(); return true }
}

private extension UIInterfaceOrientation {
    var singleOrientationMask: UIInterfaceOrientationMask? {
        switch self {
        case .portrait: .portrait
        case .portraitUpsideDown: .portraitUpsideDown
        case .landscapeLeft: .landscapeLeft
        case .landscapeRight: .landscapeRight
        default: nil
        }
    }
}

private struct JarvisSandboxWebView: UIViewRepresentable {
    let request: URLRequest
    let pageID: UUID
    let event: (String, UUID) -> Void
    func makeCoordinator() -> Coordinator { Coordinator(pageID: pageID, origin: request.url!, event: event) }
    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        configuration.userContentController.add(context.coordinator, name: "sandboxView")
        let view = WKWebView(frame: .zero, configuration: configuration)
        // This is an authenticated app viewer. Identify it consistently on
        // page, asset and socket requests, including through the demo tunnel.
        view.customUserAgent = "JarvisSandboxViewer/1.0"
        view.navigationDelegate = context.coordinator
        view.isOpaque = false
        view.scrollView.isScrollEnabled = false
        view.isUserInteractionEnabled = false
        view.isAccessibilityElement = true
        view.accessibilityLabel = "Live sandbox desktop. View only. Use your voice to guide Jarvis."
        view.load(request)
        return view
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
    static func dismantleUIView(_ uiView: WKWebView, coordinator: Coordinator) {
        coordinator.active = false
        uiView.stopLoading()
        uiView.configuration.userContentController.removeScriptMessageHandler(forName: "sandboxView")
        uiView.navigationDelegate = nil
        uiView.loadHTMLString("", baseURL: nil)
    }
    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var active = true
        let pageID: UUID
        let origin: URL
        let event: (String, UUID) -> Void
        init(pageID: UUID, origin: URL, event: @escaping (String, UUID) -> Void) {
            self.pageID = pageID; self.origin = origin; self.event = event
        }
        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard active, message.frameInfo.isMainFrame,
                  let body = message.body as? [String: String], let state = body["state"] else { return }
            event(state, pageID)
        }
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard active, action.targetFrame?.isMainFrame == true, let url = action.request.url,
                  url.scheme == origin.scheme, url.host == origin.host, url.port == origin.port,
                  url.path == origin.path, action.navigationType == .other else {
                decisionHandler(.cancel); return
            }
            decisionHandler(.allow)
        }
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            if active { event("failed", pageID) }
        }
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            if active { event("failed", pageID) }
        }
    }
}


struct JarvisSandboxConnectButton: View {
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: "display")
                    .font(.system(size: 22, weight: .light))
                    .frame(width: 48, height: 48)
                    .background(JarvisColors.softAccent, in: RoundedRectangle(cornerRadius: 15))
                Text("View sandbox")
                    .font(.title3.weight(.medium))
                Spacer(minLength: 8)
                Image(systemName: "arrow.up.right").font(.system(size: 16, weight: .medium))
            }
            .foregroundStyle(JarvisColors.accent)
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(JarvisColors.border, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("sandbox.view")
        .accessibilityHint("View the screen of your active sandbox")
    }
}

struct JarvisSandboxEmptyWorkspace: View {
    let open: () async -> JarvisSandboxViewModel.Notice?
    @State private var notice: JarvisSandboxViewModel.Notice?
    @State private var opening = false
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Text("Workspace").font(JarvisTypography.pageTitle)
                    .foregroundStyle(JarvisColors.primaryText).accessibilityAddTraits(.isHeader)
                JarvisSandboxConnectButton {
                    opening = true
                    Task { notice = await open(); opening = false }
                }.disabled(opening)
                if opening { ProgressView().tint(JarvisColors.accent) }
            }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(JarvisColors.background)
        .alert(item: $notice) { value in
            Alert(title: Text(value.title), message: Text(value.message), dismissButton: .default(Text("Got it")))
        }
    }
}
