import SwiftUI
import UIKit
import os

struct JarvisAppLinkRequest {
    let operationID: UUID
    let sessionID: UUID
    let audioSessionID: String
    let url: URL?
    let description: String?

    init?(event: JarvisRealtimeServerEvent) {
        guard event.type == "app.link",
              let operation = event.values["operation_id"] as? String,
              let operationID = UUID(uuidString: operation),
              let session = event.sessionId, let sessionID = UUID(uuidString: session),
              let audioSessionID = event.audioSessionId, !audioSessionID.isEmpty else { return nil }
        self.operationID = operationID
        self.sessionID = sessionID
        self.audioSessionID = audioSessionID
        self.url = Self.safeURL(event.values["url"] as? String)
        if let text = event.values["description"] as? String,
           text.unicodeScalars.count <= 1_000, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            self.description = text
        } else { self.description = nil }
    }

    private static func safeURL(_ raw: String?) -> URL? {
        guard let raw, raw.utf8.count <= 8_192,
              raw.rangeOfCharacter(from: .whitespacesAndNewlines.union(.controlCharacters)) == nil,
              let parts = URLComponents(string: raw),
              let scheme = parts.scheme?.lowercased(), ["https", "http"].contains(scheme),
              let host = parts.host, !host.isEmpty, parts.user == nil, parts.password == nil else { return nil }
        return parts.url
    }
}

extension JarvisAPIClient {
    func acknowledgeAppLink(_ link: JarvisAppLinkRequest, outcome: String) async throws {
        // The response body is deliberately not a new client-side wire dependency.
        struct Receipt: Decodable {}
        let body = try JSONSerialization.data(withJSONObject: [
            "session_id": link.sessionID.uuidString.lowercased(),
            "operation_id": link.operationID.uuidString.lowercased(), "outcome": outcome
        ])
        let _: Receipt = try await request(path: "/app-controls/link-ack", method: "POST",
                                          body: body, responseType: Receipt.self)
    }
}

/// A root-owned presenter, independent of tab navigation and media ownership.
/// An alert sits above the current sheet without dismissing or replacing it.
@MainActor
final class JarvisAppLinkPresenter {
    weak var window: UIWindow?
    private weak var alert: UIAlertController?
    private var sessionID: UUID?
    private var audioSessionID: String?
    private var activeOperation: UUID?
    private var outcomes: [UUID: String] = [:]
    private var acknowledgements: [UUID: Task<Void, Never>] = [:]
    private var cancelPresentation: (() -> Void)?
    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "app-link")

    func receive(_ link: JarvisAppLinkRequest, api: JarvisAPIClient,
                 isCurrent: @escaping @MainActor () -> Bool) {
        guard isCurrent() else { return }
        if sessionID != link.sessionID {
            reset()
            sessionID = link.sessionID
        } else if audioSessionID != link.audioSessionID {
            dismiss()
            for task in acknowledgements.values { task.cancel() }
            acknowledgements.removeAll()
        }
        audioSessionID = link.audioSessionID
        if let outcome = outcomes[link.operationID] {
            acknowledge(link, outcome: outcome, api: api, isCurrent: isCurrent)
            return
        }
        guard activeOperation != link.operationID else { return }
        // Do not evict remembered operations and accidentally display a replay.
        guard outcomes.count < 256 else {
            acknowledge(link, outcome: "unavailable", api: api, isCurrent: isCurrent)
            return
        }
        let complete: (String) -> Void = { [weak self] outcome in
            guard let self, self.sessionID == link.sessionID,
                  self.audioSessionID == link.audioSessionID,
                  self.outcomes[link.operationID] == nil else { return }
            self.outcomes[link.operationID] = outcome
            self.acknowledge(link, outcome: outcome, api: api, isCurrent: isCurrent)
        }
        guard activeOperation == nil, let url = link.url, let description = link.description,
              let window, !window.isHidden, window.windowScene?.activationState == .foregroundActive,
              let presenter = topController(window.rootViewController),
              presenter.viewIfLoaded?.window != nil, !(presenter is UIAlertController),
              !presenter.isBeingPresented, !presenter.isBeingDismissed,
              presenter.transitionCoordinator == nil else {
            complete("unavailable")
            return
        }
        let alert = UIAlertController(title: "Jarvis", message: description + "\n\n" + (url.host ?? ""),
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Open link", style: .default) { [weak self] _ in
            guard let self, self.activeOperation == link.operationID else { return }
            complete("displayed")
            self.clearActivePresentation()
            guard isCurrent() else { return }
            UIApplication.shared.open(url, options: [:], completionHandler: nil)
        })
        alert.addAction(UIAlertAction(title: "Dismiss", style: .cancel) { [weak self] _ in
            guard self?.activeOperation == link.operationID else { return }
            complete("displayed")
            self?.clearActivePresentation()
        })
        activeOperation = link.operationID
        self.alert = alert
        cancelPresentation = { complete("unavailable") }
        presenter.present(alert, animated: true) { [weak self, weak alert] in
            guard let self, let alert, self.activeOperation == link.operationID else { return }
            guard isCurrent(), alert.viewIfLoaded?.window != nil,
                  window.windowScene?.activationState == .foregroundActive else {
                self.dismiss()
                return
            }
            self.cancelPresentation = nil
            complete("displayed")
        }
        if alert.presentingViewController == nil { dismiss() }
    }

    func dismiss() {
        cancelPresentation?()
        let presented = alert
        clearActivePresentation()
        presented?.dismiss(animated: false)
    }

    func reset() {
        dismiss()
        for task in acknowledgements.values { task.cancel() }
        acknowledgements.removeAll()
        outcomes.removeAll()
        sessionID = nil; audioSessionID = nil
    }

    private func clearActivePresentation() {
        cancelPresentation = nil
        alert = nil
        activeOperation = nil
    }

    private func acknowledge(_ link: JarvisAppLinkRequest, outcome: String, api: JarvisAPIClient,
                             isCurrent: @escaping @MainActor () -> Bool) {
        guard acknowledgements[link.operationID] == nil else { return }
        acknowledgements[link.operationID] = Task { [weak self] in
            defer {
                if self?.sessionID == link.sessionID, self?.audioSessionID == link.audioSessionID {
                    self?.acknowledgements[link.operationID] = nil
                }
            }
            guard !Task.isCancelled, isCurrent() else { return }
            do { try await api.acknowledgeAppLink(link, outcome: outcome) }
            catch is CancellationError {}
            catch { self?.logger.error("Link presentation acknowledgement failed") }
        }
    }

    private func topController(_ root: UIViewController?) -> UIViewController? {
        guard let root else { return nil }
        if let presented = root.presentedViewController { return topController(presented) }
        if let navigation = root as? UINavigationController { return topController(navigation.visibleViewController) }
        if let tabs = root as? UITabBarController { return topController(tabs.selectedViewController) }
        return root
    }
}

struct JarvisAppLinkHost: UIViewRepresentable {
    let presenter: JarvisAppLinkPresenter

    func makeUIView(context: Context) -> AnchorView {
        let view = AnchorView()
        view.presenter = presenter
        view.isUserInteractionEnabled = false
        view.isAccessibilityElement = false
        return view
    }
    func updateUIView(_ view: AnchorView, context: Context) {
        if let window = view.window { presenter.window = window }
    }
    static func dismantleUIView(_ view: AnchorView, coordinator: ()) {
        view.presenter?.dismiss()
    }

    final class AnchorView: UIView {
        weak var presenter: JarvisAppLinkPresenter?
        override func didMoveToWindow() {
            super.didMoveToWindow()
            // A fullscreen presentation may temporarily detach the root view.
            if let window { presenter?.window = window }
        }
    }
}
