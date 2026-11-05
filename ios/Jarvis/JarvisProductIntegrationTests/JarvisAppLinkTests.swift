import SwiftUI
import UIKit
import XCTest
@testable import Jarvis

@MainActor
final class JarvisAppLinkTests: XCTestCase {
    private var window: UIWindow?
    private var previousWindow: UIWindow?
    private var transport: URLSession?
    private var presenter: JarvisAppLinkPresenter?

    override func tearDown() {
        presenter?.reset()
        window?.isHidden = true
        window?.rootViewController = nil
        previousWindow?.makeKey()
        transport?.invalidateAndCancel()
        window = nil; previousWindow = nil; presenter = nil; transport = nil
        super.tearDown()
    }

    func testRejectsUnsafeURLsAndMalformedDescriptions() throws {
        for url in ["javascript:alert(1)", "file:///tmp/private", "data:text/html,hello", "//example.com",
                    "https://user:secret@example.com", "https://", "https://example.com/a\nb"] {
            XCTAssertNil(try request(url: url).url, url)
        }
        XCTAssertEqual(try request(url: "https://example.com/a?b=1#c").url?.scheme, "https")
        XCTAssertEqual(try request(url: "http://example.com").url?.scheme, "http")
        XCTAssertNil(try request(description: String(repeating: "a", count: 1_001)).description)
        XCTAssertNil(try request(description: "  ").description)
        XCTAssertNil(JarvisAppLinkRequest(event: .init(type: "app.link", values: ["operation_id": "bad"])))
    }

    func testPresentsAboveExistingSheetAndAcknowledgesOnlyAfterPresentation() async throws {
        let (owner, api, root) = try mount()
        let sheet = UIHostingController(rootView: Text("Existing preferences"))
        root.present(sheet, animated: false)
        try await eventually { sheet.viewIfLoaded?.window != nil && sheet.transitionCoordinator == nil }
        let link = try request()
        owner.receive(link, api: api, isCurrent: { true })
        XCTAssertTrue(AppLinkProtocol.records.isEmpty, "Enqueueing is not a displayed receipt")
        try await eventually { AppLinkProtocol.records.count == 1 }
        let alert = try XCTUnwrap(sheet.presentedViewController as? UIAlertController)
        XCTAssertEqual(alert.actions.map(\.title), ["Open link", "Dismiss"])
        XCTAssertEqual(alert.message, "Open the finished report.\n\nexample.com")
        XCTAssertNotNil(alert.viewIfLoaded?.window)
        let ack = try XCTUnwrap(AppLinkProtocol.records.first)
        XCTAssertEqual(ack.url.path, "/app-controls/link-ack")
        XCTAssertEqual(ack.method, "POST")
        XCTAssertEqual(ack.authorization, "Bearer local-fixture-only")
        XCTAssertEqual(ack.body["outcome"] as? String, "displayed")
        XCTAssertEqual(ack.body["operation_id"] as? String, link.operationID.uuidString.lowercased())
        XCTAssertEqual(ack.body["session_id"] as? String, link.sessionID.uuidString.lowercased())
        owner.receive(link, api: api, isCurrent: { true })
        XCTAssertTrue(sheet.presentedViewController === alert, "Duplicate must reuse the same presentation")
        let attachment = XCTAttachment(image: UIGraphicsImageRenderer(bounds: try XCTUnwrap(window).bounds).image { _ in
            window?.drawHierarchy(in: window?.bounds ?? .zero, afterScreenUpdates: true)
        })
        attachment.name = "App link above existing sheet"; attachment.lifetime = .keepAlways; add(attachment)
        owner.dismiss()
        XCTAssertTrue(root.presentedViewController === sheet, "The existing sheet must be preserved")
        try await eventually { sheet.presentedViewController == nil }
        owner.receive(link, api: api, isCurrent: { true })
        XCTAssertNil(sheet.presentedViewController, "Dismissed operations must not reappear")
    }

    func testBusyAndInvalidLinksReportUnavailableAndStaleLinksAreIgnored() async throws {
        let (owner, api, root) = try mount()
        let existing = UIAlertController(title: "Existing alert", message: nil, preferredStyle: .alert)
        existing.addAction(UIAlertAction(title: "OK", style: .default))
        root.present(existing, animated: false)
        try await eventually { existing.viewIfLoaded?.window != nil }
        let link = try request()
        owner.receive(link, api: api, isCurrent: { true })
        try await eventually { AppLinkProtocol.records.count == 1 }
        XCTAssertEqual(AppLinkProtocol.records[0].body["outcome"] as? String, "unavailable")
        XCTAssertTrue(root.presentedViewController === existing)
        let invalid = try request(url: "javascript:alert(1)", sessionID: link.sessionID)
        owner.receive(invalid, api: api, isCurrent: { true })
        try await eventually { AppLinkProtocol.records.count == 2 }
        XCTAssertEqual(AppLinkProtocol.records[1].body["outcome"] as? String, "unavailable")
        owner.receive(try request(), api: api, isCurrent: { false })
        await Task.yield()
        XCTAssertEqual(AppLinkProtocol.records.count, 2)
    }

    func testResetRemovesPopupAndOffscreenCannotClaimDisplayed() async throws {
        let (owner, api, root) = try mount()
        owner.receive(try request(), api: api, isCurrent: { true })
        try await eventually { AppLinkProtocol.records.count == 1 }
        owner.reset()
        try await eventually { root.presentedViewController == nil }
        owner.window = nil
        owner.receive(try request(), api: api, isCurrent: { true })
        try await eventually { AppLinkProtocol.records.count == 2 }
        XCTAssertEqual(AppLinkProtocol.records[1].body["outcome"] as? String, "unavailable")
    }

    private func request(url: String = "https://example.com/report", description: String = "Open the finished report.",
                         sessionID: UUID = UUID()) throws -> JarvisAppLinkRequest {
        try XCTUnwrap(JarvisAppLinkRequest(event: .init(type: "app.link", values: [
            "operation_id": UUID().uuidString, "session_id": sessionID.uuidString,
            "audio_session_id": "link-fixture-audio", "url": url, "description": description
        ])))
    }

    private func mount() throws -> (JarvisAppLinkPresenter, JarvisAPIClient, UIViewController) {
        let args = ProcessInfo.processInfo.arguments
        guard let index = args.firstIndex(of: "--jarvis-replay-manifest"),
              args.indices.contains(index + 1), args[index + 1] == "/dev/null" else {
            throw XCTSkip("Requires isolated launch without voice/account startup")
        }
        AppLinkProtocol.clear()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [AppLinkProtocol.self]
        let transport = URLSession(configuration: configuration)
        self.transport = transport
        let api = JarvisAPIClient(configuration: .init(backendBaseURL: URL(string: "https://app-link.invalid")!,
                                                      staticBearerToken: "local-fixture-only"), session: transport)
        let owner = JarvisAppLinkPresenter()
        presenter = owner
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        previousWindow = scene.windows.first(where: \.isKeyWindow)
        let window = UIWindow(windowScene: scene)
        let root = UIHostingController(rootView: Text("Jarvis link fixture")
            .background(JarvisAppLinkHost(presenter: owner).frame(width: 0, height: 0)))
        window.rootViewController = root
        window.makeKeyAndVisible()
        self.window = window
        root.view.layoutIfNeeded()
        XCTAssertTrue(owner.window === window, "The root anchor must supply the current window")
        return (owner, api, root)
    }

    private func eventually(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(3)
        while !predicate(), Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertTrue(predicate(), "Expected native presentation/receipt did not complete")
    }
}

private final class AppLinkProtocol: URLProtocol, @unchecked Sendable {
    struct Record { let url: URL; let method: String?; let authorization: String?; let body: [String: Any] }
    private static let lock = NSLock()
    private static var stored: [Record] = []
    static var records: [Record] { lock.withLock { stored } }
    static func clear() { lock.withLock { stored.removeAll() } }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url, url.host == "app-link.invalid" else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL)); return
        }
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var bytes = [UInt8](repeating: 0, count: 1_024)
            while stream.hasBytesAvailable {
                let count = stream.read(&bytes, maxLength: bytes.count)
                guard count > 0 else { break }
                data.append(contentsOf: bytes.prefix(count))
            }
        }
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        Self.lock.withLock { Self.stored.append(Record(url: url, method: request.httpMethod,
            authorization: request.value(forHTTPHeaderField: "Authorization"), body: body)) }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "application/json"])!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("{}".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
