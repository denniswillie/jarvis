import CryptoKit
import CoreFoundation
import Foundation
import XCTest
@testable import Jarvis

/// Native URLSession/APIClient tests with an injected, fail-closed URLProtocol.
/// No real credentials, networking, media capture or backend save is exercised.
/// The protocol observes metadata, NOT the bytes emitted by a file upload.
@MainActor
final class JarvisRecordingTransportTests: XCTestCase {
    private var fixtures: [RecordingTransportHarness] = []

    override func setUp() async throws {
        try await super.setUp()
        // Match the product target's existing safe launch convention. The test
        // session is independently injected; no shared-session registration.
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "--jarvis-replay-manifest"),
              index + 1 < arguments.count, arguments[index + 1] == "/dev/null",
              !arguments.contains("--jarvis-replay-run") else {
            throw RecordingTransportFixtureError.unsafeHostLaunch
        }
    }

    override func tearDown() async throws {
        for fixture in fixtures {
            fixture.session.invalidateAndCancel()
            RecordingTransportRegistry.shared.remove(host: fixture.host)
            // Only this test's synthetic temporary bytes are removed.
            try FileManager.default.removeItem(at: fixture.directory)
        }
        fixtures.removeAll()
        try await super.tearDown()
    }

    func testPUTUsesExactEndpointHeadersAndAnUploadTask() async throws {
        let fixture = try makeFixture([.hold])
        let call = begin(fixture, file: fixture.file, descriptor: fixture.descriptor)
        defer { call.task?.cancel() }
        guard await wait(fixture.state.started, message: "Upload never reached the isolated protocol") else { return }
        let request = try XCTUnwrap(fixture.state.requests.first)
        assertRequest(request, fixture: fixture, method: "PUT", suffix: "content")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "video/mp4")
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-Jarvis-Recording-Manifest"), fixture.descriptor)
        let tasks = await fixture.session.recordingTransportTasks()
        XCTAssertEqual(tasks.count, 1)
        XCTAssertTrue(tasks.first is URLSessionUploadTask,
            "Exercise the real upload API, not a substituted data request")
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.file.path))
        // An UploadTask and a file argument do not prove transmitted body bytes.
        // URLProtocol may expose neither the file nor its body stream reliably.
        let body = Data("{\"outcome\":\"saved\",\"fixture_only\":true}\n".utf8)
        fixture.state.releaseHeld(status: 200, body: body, headers: ["X-Fixture-Outcome": "unchanged"])
        guard await wait(call.finished, message: "Upload did not complete") else { return }
        let (actual, response) = try XCTUnwrap(call.result).get()
        XCTAssertEqual(actual, body)
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.value(forHTTPHeaderField: "X-Fixture-Outcome"), "unchanged")
        XCTAssertEqual(fixture.state.requests.count, 1)
    }

    func test202UploadThen200ReconcilePreservesSameIdentityAndRawOutcomes() async throws {
        let unresolved = Data("{\"outcome\":\"publication_unresolved\",\"phase\":\"prepared\"}\n".utf8)
        // Deliberately not a receipt. HTTP success must not fabricate one or
        // silently replace its bytes; Core owns receipt validation and linkage.
        let rawOutcome = Data([0x00, 0xff, 0x31])
        let fixture = try makeFixture([
            .http(202, unresolved, ["Retry-After": "4"], nil),
            .http(200, rawOutcome, ["X-Fixture-Outcome": "raw"], nil)
        ])
        let upload = begin(fixture, file: fixture.file, descriptor: fixture.descriptor)
        defer { upload.task?.cancel() }
        guard await wait(upload.finished, message: "202 upload did not complete") else { return }
        let first = try XCTUnwrap(upload.result).get()
        XCTAssertEqual(first.0, unresolved)
        XCTAssertEqual(first.1.statusCode, 202)
        XCTAssertEqual(first.1.value(forHTTPHeaderField: "Retry-After"), "4")
        XCTAssertEqual(fixture.state.requests.count, 1, "202 must not trigger an automatic retry")

        let reconcile = begin(fixture)
        defer { reconcile.task?.cancel() }
        guard await wait(reconcile.finished, message: "Reconciliation did not complete") else { return }
        let second = try XCTUnwrap(reconcile.result).get()
        XCTAssertEqual(second.0, rawOutcome)
        XCTAssertEqual(second.1.statusCode, 200)
        let requests = fixture.state.requests
        XCTAssertEqual(requests.count, 2)
        guard requests.count == 2 else { return }
        assertRequest(requests[0], fixture: fixture, method: "PUT", suffix: "content")
        assertRequest(requests[1], fixture: fixture, method: "POST", suffix: "reconcile")
        XCTAssertNil(requests[1].value(forHTTPHeaderField: "X-Jarvis-Recording-Manifest"))
        XCTAssertNil(requests[1].value(forHTTPHeaderField: "Content-Type"))
        XCTAssertNil(requests[1].httpBody)
        XCTAssertNil(requests[1].httpBodyStream)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.file.path), "Transport must not delete an original")
    }

    func testPOSTPreserves202WithoutInventingASavedOutcome() async throws {
        let body = Data("{\"outcome\":\"publication_unresolved\"}".utf8)
        let fixture = try makeFixture([.http(202, body, [:], nil)])
        let call = begin(fixture)
        defer { call.task?.cancel() }
        guard await wait(call.finished, message: "202 reconciliation did not complete") else { return }
        let result = try XCTUnwrap(call.result).get()
        XCTAssertEqual(result.0, body)
        XCTAssertEqual(result.1.statusCode, 202)
        XCTAssertEqual(fixture.state.requests.count, 1)
    }

    func testNonSuccessPreservesServerErrorAndRetryMetadataWithoutRetrying() async throws {
        for status in [400, 401, 409, 429, 503] {
            let body = Data("{\"error\":{\"code\":\"recording_fixture_rejected\",\"message\":\"Original remains unconfirmed.\"}}".utf8)
            let fixture = try makeFixture([
                .http(status, body, ["Retry-After": "7"], nil),
                .http(200, Data("unexpected retry".utf8), [:], nil)
            ])
            let call = begin(fixture, file: fixture.file, descriptor: fixture.descriptor)
            defer { call.task?.cancel() }
            guard await wait(call.finished, message: "HTTP \(status) error did not complete") else { return }
            let error = try XCTUnwrap(failure(call) as? JarvisAPIError)
            XCTAssertEqual(error.code, "recording_fixture_rejected")
            XCTAssertEqual(error.message, "Original remains unconfirmed.")
            XCTAssertEqual(error.statusCode, status)
            XCTAssertEqual(error.retryAfter, 7)
            XCTAssertEqual(fixture.state.requests.count, 1, "Retry-After is metadata, not retry permission")
            XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.file.path))
        }
    }

    func testNonJSONErrorAndUnexpectedSuccessStatusAreTruthfulFailures() async throws {
        for status in [201, 204, 502] {
            let fixture = try makeFixture([.http(status, Data("not an error envelope".utf8), [:], nil)])
            let call = begin(fixture)
            defer { call.task?.cancel() }
            guard await wait(call.finished, message: "Unexpected status did not complete") else { return }
            let error = try XCTUnwrap(failure(call) as? JarvisAPIError)
            XCTAssertEqual(error.code, "http_\(status)")
            XCTAssertEqual(error.statusCode, status)
            XCTAssertEqual(error.message, "Jarvis request failed with status \(status).")
            XCTAssertEqual(fixture.state.requests.count, 1)
        }
    }

    func testInvalidDescriptorEnvelopeIsRejectedBeforeProtocolDispatch() async throws {
        let fixture = try makeFixture([.http(200, Data(), [:], nil)])
        // This helper checks header safety/bounds, not JSON or the report schema.
        // Semantic descriptor validation belongs to JarvisRecordingManifest.
        let invalid: [(URL?, String?)] = [
            (fixture.file, nil), (fixture.file, ""),
            (fixture.file, String(repeating: "x", count: 8193)),
            (fixture.file, "\u{00E9}"), (fixture.file, "bad\rheader"),
            (fixture.file, "bad\nheader"), (nil, fixture.descriptor),
            (URL(string: "https://file-argument.invalid/not-a-local-file")!, fixture.descriptor)
        ]
        for (file, descriptor) in invalid {
            let call = begin(fixture, file: file, descriptor: descriptor)
            defer { call.task?.cancel() }
            guard await wait(call.finished, message: "Invalid descriptor did not fail promptly") else { return }
            let error = try XCTUnwrap(failure(call) as? JarvisAPIError)
            XCTAssertEqual(error.code, "recording_manifest_invalid")
            XCTAssertNil(error.statusCode, "A local preflight rejection is not an HTTP response")
            XCTAssertTrue(fixture.state.requests.isEmpty, "Invalid descriptor reached the protocol")
        }
    }

    func testOversizedAndDifferentURLResponsesAreNotAccepted() async throws {
        let other = URL(string: "https://response-identity.invalid/other")!
        for reply in [RecordingTransportFixture.Step.http(200, Data(repeating: 0x61, count: 65_537), [:], nil),
                      .http(200, Data("{}".utf8), [:], other)] {
            let fixture = try makeFixture([reply])
            let call = begin(fixture)
            defer { call.task?.cancel() }
            guard await wait(call.finished, message: "Unverifiable response did not complete") else { return }
            let error = try XCTUnwrap(failure(call) as? JarvisAPIError)
            XCTAssertEqual(error.code, "recording_response_invalid")
            XCTAssertNil(error.statusCode)
            XCTAssertEqual(fixture.state.requests.count, 1)
        }
    }

    func testNetworkFailureIsNotRetriedOrConvertedToSaved() async throws {
        let fixture = try makeFixture([
            .failure(.networkConnectionLost), .http(200, Data("unexpected retry".utf8), [:], nil)
        ])
        let call = begin(fixture, file: fixture.file, descriptor: fixture.descriptor)
        defer { call.task?.cancel() }
        guard await wait(call.finished, message: "Network failure did not complete") else { return }
        let error = try XCTUnwrap(failure(call)) as NSError
        XCTAssertEqual(error.domain, NSURLErrorDomain)
        XCTAssertEqual(error.code, URLError.networkConnectionLost.rawValue)
        XCTAssertEqual(fixture.state.requests.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.file.path))
    }

    func testCancellationStopsTheHeldUploadWithoutRetryOrDeletion() async throws {
        let fixture = try makeFixture([.hold, .http(200, Data("unexpected retry".utf8), [:], nil)])
        let call = begin(fixture, file: fixture.file, descriptor: fixture.descriptor)
        defer { call.task?.cancel() }
        guard await wait(fixture.state.started, message: "Upload did not reach cancellation barrier") else { return }
        call.task?.cancel()
        guard await wait(call.finished, message: "Cancellation waited for the 120-second transport timeout") else { return }
        let error = try XCTUnwrap(failure(call))
        let nsError = error as NSError
        XCTAssertTrue(error is CancellationError ||
            (nsError.domain == NSURLErrorDomain && nsError.code == URLError.cancelled.rawValue))
        guard await wait(fixture.state.stopped, message: "Cancellation did not stop the protocol load") else { return }
        XCTAssertEqual(fixture.state.requests.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.file.path))
    }

    func test307StatusIsRejectedWithoutTreatingLocationAsSuccess() async throws {
        let fixture = try makeFixture([.http(307, Data(), ["Location": "/not-followed"], nil)])
        let call = begin(fixture)
        defer { call.task?.cancel() }
        guard await wait(call.finished, message: "307 status did not complete") else { return }
        let error = try XCTUnwrap(failure(call) as? JarvisAPIError)
        XCTAssertEqual(error.statusCode, 307)
        XCTAssertEqual(fixture.state.requests.count, 1)
        // A didReceive(307) stub alone does NOT exercise redirect delegation.
    }

    func testNativeRedirectPositiveControlFollowsTheTargetExactlyOnce() async throws {
        let targetBody = Data("native-follow-control-target-only".utf8)
        let fixture = try makeFixture([.redirect, .http(200, targetBody, [:], nil)])
        let originalURL = URL(string: "https://\(fixture.host)/api/recordings/\(fixture.recordingID.uuidString.lowercased())/content?session_id=\(fixture.sessionID.uuidString.lowercased())")!
        let targetURL = originalURL.deletingLastPathComponent().appendingPathComponent("redirect-target")
        var request = URLRequest(url: originalURL, cachePolicy: .reloadIgnoringLocalCacheData)
        request.httpMethod = "PUT"
        request.timeoutInterval = 120
        request.setValue("Bearer recording-transport-fixture-only", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("video/mp4", forHTTPHeaderField: "Content-Type")
        request.setValue(fixture.descriptor, forHTTPHeaderField: "X-Jarvis-Recording-Manifest")
        let outgoing = request
        let delegate = RecordingTransportFollowRedirect()
        let call = RecordingTransportCall()
        call.task = Task { @MainActor in
            do {
                let (data, response) = try await fixture.session.upload(for: outgoing, fromFile: fixture.file, delegate: delegate)
                guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
                call.result = .success((data, http))
            } catch { call.result = .failure(error) }
            call.finished.fulfill()
        }
        defer { call.task?.cancel() }
        // One shared three-second bound, not separate waits that add time.
        let result = await XCTWaiter.fulfillment(of: [delegate.called, call.finished], timeout: 3, enforceOrder: true)
        XCTAssertEqual(result, .completed, "Native follow control did not qualify the fixture; preserve failure and stop")
        guard result == .completed else { return }
        let observations = delegate.observations
        XCTAssertEqual(observations.count, 1, "The actual URLSession delegate must receive exactly one redirect")
        let observation = try XCTUnwrap(observations.first)
        XCTAssertEqual(observation.status, 307)
        XCTAssertEqual(observation.originalURL, originalURL)
        XCTAssertEqual(observation.proposedURL, targetURL)
        let (data, response) = try XCTUnwrap(call.result).get()
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.url, targetURL)
        XCTAssertEqual(data, targetBody)
        let requests = fixture.state.requests
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests.filter { $0.url == originalURL }.count, 1)
        XCTAssertEqual(requests.filter { $0.url == targetURL }.count, 1)
        XCTAssertEqual(requests.last?.httpMethod, "PUT")
        // This control establishes native redirect behavior, not wire-media bytes.
    }

    func testNativeRedirectCallbackDoesNotFollowTheTarget() async throws {
        let fixture = try makeFixture([.redirect, .http(200, Data("redirect was followed".utf8), [:], nil)])
        let originalHash = SHA256.hash(data: try Data(contentsOf: fixture.file))
        let call = begin(fixture, file: fixture.file, descriptor: fixture.descriptor)
        defer { call.task?.cancel() }
        // Passing this case is redirect acceptance ONLY alongside the native
        // follow control. The status-only 307 case is not a substitute.
        guard await wait(call.finished, message: "Native redirect callback did not complete; preserve failure and stop") else { return }
        let error = try XCTUnwrap(failure(call) as? JarvisAPIError)
        XCTAssertEqual(error.statusCode, 307)
        // These unique fields come from the original 307 JSON body, not from a
        // fabricated success or the redirect target. APIClient throws that body
        // as JarvisAPIError, rather than returning its raw bytes on non-success.
        XCTAssertEqual(error.code, RecordingTransportFixture.redirectErrorCode)
        XCTAssertEqual(error.message, RecordingTransportFixture.redirectErrorMessage)
        let requests = fixture.state.requests
        XCTAssertEqual(requests.count, 1, "The redirect target must never be requested")
        assertRequest(try XCTUnwrap(requests.first), fixture: fixture, method: "PUT", suffix: "content")
        XCTAssertEqual(requests.filter { $0.url?.lastPathComponent == "redirect-target" }.count, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.file.path))
        XCTAssertEqual(SHA256.hash(data: try Data(contentsOf: fixture.file)), originalHash)
    }

    private func makeFixture(_ steps: [RecordingTransportFixture.Step]) throws -> RecordingTransportHarness {
        let fixture = try RecordingTransportHarness(steps: steps)
        fixtures.append(fixture)
        return fixture
    }

    private func assertRequest(_ request: URLRequest, fixture: RecordingTransportHarness, method: String, suffix: String,
                               file: StaticString = #filePath, line: UInt = #line) {
        let expected = "https://\(fixture.host)/api/recordings/\(fixture.recordingID.uuidString.lowercased())/\(suffix)?session_id=\(fixture.sessionID.uuidString.lowercased())"
        XCTAssertEqual(request.url?.absoluteString, expected, file: file, line: line)
        XCTAssertEqual(request.httpMethod, method, file: file, line: line)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer recording-transport-fixture-only", file: file, line: line)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json", file: file, line: line)
        XCTAssertEqual(request.timeoutInterval, 120, accuracy: 0.001, file: file, line: line)
        XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalCacheData, file: file, line: line)
    }

    private func begin(_ fixture: RecordingTransportHarness, file: URL? = nil, descriptor: String? = nil) -> RecordingTransportCall {
        let call = RecordingTransportCall()
        call.task = Task { @MainActor in
            do {
                call.result = .success(try await fixture.api.recordingResponse(
                    recordingID: fixture.recordingID, sessionID: fixture.sessionID, file: file, descriptor: descriptor))
            } catch { call.result = .failure(error) }
            call.finished.fulfill()
        }
        return call
    }

    private func wait(_ expectation: XCTestExpectation, message: String, file: StaticString = #filePath, line: UInt = #line) async -> Bool {
        let result = await XCTWaiter.fulfillment(of: [expectation], timeout: 3)
        XCTAssertEqual(result, .completed, message, file: file, line: line)
        return result == .completed
    }

    private func failure(_ call: RecordingTransportCall, file: StaticString = #filePath, line: UInt = #line) -> Error? {
        switch call.result {
        case .failure(let error): return error
        case .success: XCTFail("Unexpected transport success", file: file, line: line); return nil
        case nil: XCTFail("Transport has not completed", file: file, line: line); return nil
        }
    }
}

@MainActor
private final class RecordingTransportCall {
    let finished = XCTestExpectation(description: "recordingResponse returned")
    var result: Result<(Data, HTTPURLResponse), Error>?
    var task: Task<Void, Never>?
}

@MainActor
private final class RecordingTransportHarness {
    let recordingID = UUID()
    let sessionID = UUID()
    let host = "recording-" + UUID().uuidString.lowercased() + ".invalid"
    let directory: URL
    let file: URL
    let descriptor: String
    let state: RecordingTransportFixture
    let session: URLSession
    let api: JarvisAPIClient

    init(steps: [RecordingTransportFixture.Step]) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("JarvisRecordingTransportTests-" + UUID().uuidString, isDirectory: true)
        file = directory.appendingPathComponent("synthetic-original.mp4")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        // A synthetic file argument, deliberately not presented as playable MP4.
        let bytes = Data("offline transport fixture; not captured or playable media".utf8)
        try bytes.write(to: file, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let body: [String: Any] = ["filename": "synthetic-original.mp4", "description": "Offline transport fixture only.",
            "size_bytes": bytes.count, "content_sha256": hash,
            "client_report": ["format": "jarvis.recording_client_report.v1", "generation_id": UUID().uuidString.lowercased(),
                "timeline_end_ms": 100, "video_frames": 1, "source_audio_frames": 0, "padded_audio_frames": 4800,
                "missing_intervals": [], "omitted_interval_details": 0]]
        descriptor = String(decoding: try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]), as: UTF8.self)
        state = RecordingTransportFixture(steps: steps)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RecordingTransportProtocol.self]
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        session = URLSession(configuration: configuration)
        api = JarvisAPIClient(configuration: JarvisConfiguration(
            backendBaseURL: URL(string: "https://\(host)/api")!, staticBearerToken: "recording-transport-fixture-only"), session: session)
        RecordingTransportRegistry.shared.install(state, host: host)
    }
}

private enum RecordingTransportFixtureError: Error { case unsafeHostLaunch }

private final class RecordingTransportFixture: @unchecked Sendable {
    static let redirectErrorCode = "fixture_original_redirect_refused"
    static let redirectErrorMessage = "Original 307 body retained; target must not run."
    static let redirectBody = Data("{\"error\":{\"code\":\"fixture_original_redirect_refused\",\"message\":\"Original 307 body retained; target must not run.\"}}".utf8)
    enum Step: Sendable {
        case hold
        case http(Int, Data, [String: String], URL?)
        case failure(URLError.Code)
        case redirect
    }
    let started = XCTestExpectation(description: "isolated protocol started")
    let stopped = XCTestExpectation(description: "isolated protocol stopped")
    private let lock = NSLock()
    private let steps: [Step]
    private var observed: [URLRequest] = []
    private var held: [RecordingTransportProtocol] = []
    private var stopObserved = false
    init(steps: [Step]) { self.steps = steps }
    var requests: [URLRequest] { lock.withLock { observed } }

    func accept(_ request: URLRequest, loader: RecordingTransportProtocol) -> Step {
        let value: (Step, Bool) = lock.withLock {
            let index = observed.count
            observed.append(request)
            let step = index < steps.count ? steps[index] : .failure(.unsupportedURL)
            if case .hold = step { held.append(loader) }
            return (step, index == 0)
        }
        if value.1 { started.fulfill() }
        return value.0
    }

    func didStop(_ loader: RecordingTransportProtocol) {
        let first = lock.withLock {
            held.removeAll { $0 === loader }
            if stopObserved { return false }
            stopObserved = true; return true
        }
        if first { stopped.fulfill() }
    }

    func releaseHeld(status: Int, body: Data, headers: [String: String]) {
        let loaders = lock.withLock { let result = held; held.removeAll(); return result }
        loaders.forEach { $0.send(status: status, body: body, headers: headers) }
    }
}

private final class RecordingTransportRegistry: @unchecked Sendable {
    static let shared = RecordingTransportRegistry()
    private let lock = NSLock()
    private var fixtures: [String: RecordingTransportFixture] = [:]
    func install(_ fixture: RecordingTransportFixture, host: String) { lock.withLock { fixtures[host] = fixture } }
    func remove(host: String) { _ = lock.withLock { fixtures.removeValue(forKey: host) } }
    func fixture(host: String?) -> RecordingTransportFixture? { lock.withLock { host.flatMap { fixtures[$0] } } }
}

/// Positive-control delegate only. The rejection test still uses the actual
/// private JarvisAPIClient delegate, unchanged by this fixture-only patch.
private final class RecordingTransportFollowRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    struct Observation: Sendable {
        let status: Int
        let originalURL: URL?
        let proposedURL: URL?
    }
    let called = XCTestExpectation(description: "native follow-redirect delegate invoked")
    private let lock = NSLock()
    private var seen: [Observation] = []
    var observations: [Observation] { lock.withLock { seen } }
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        let first = lock.withLock {
            seen.append(Observation(status: response.statusCode, originalURL: response.url, proposedURL: request.url))
            return seen.count == 1
        }
        if first { called.fulfill() }
        completionHandler(request)
    }
}

private final class RecordingTransportProtocol: URLProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    private var terminalStarted = false
    private var loadingRunLoop: CFRunLoop?
    private weak var fixture: RecordingTransportFixture?
    override class func canInit(with request: URLRequest) -> Bool {
        // Every HTTP(S) destination, including unknown/retired redirect hosts,
        // remains intercepted by this injected session. Never fall back to wire.
        ["http", "https"].contains(request.url?.scheme ?? "")
    }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let canStart = lock.withLock {
            loadingRunLoop = CFRunLoopGetCurrent()
            return !stopped
        }
        guard canStart else { return }
        guard let active = RecordingTransportRegistry.shared.fixture(host: request.url?.host) else {
            fail(URLError(.unsupportedURL)); return
        }
        lock.withLock { fixture = active }
        let step = active.accept(request, loader: self)
        guard !isStopped else { active.didStop(self); return }
        switch step {
        case .hold: break
        case let .http(status, body, headers, responseURL): send(status: status, body: body, headers: headers, responseURL: responseURL)
        case .failure(let code): fail(URLError(code))
        case .redirect:
            let destination = request.url!.deletingLastPathComponent().appendingPathComponent("redirect-target")
            var next = request; next.url = destination
            let redirected = next
            let headers = ["Location": destination.absoluteString, "Content-Type": "application/json",
                "Content-Length": String(RecordingTransportFixture.redirectBody.count)]
            let response = HTTPURLResponse(url: request.url!, statusCode: 307, httpVersion: "HTTP/1.1", headerFields: headers)!
            onLoadingRunLoop { [self] in
                guard !isStopped else { return }
                client?.urlProtocol(self, wasRedirectedTo: redirected, redirectResponse: response)
                // Refusing redirection means consuming the ORIGINAL response,
                // not leaving its load unfinished. Queue its normal callbacks
                // separately on the same originating run loop, honoring stop.
                // No delay/timer, delegate replacement, or synthetic decision.
                send(status: 307, body: RecordingTransportFixture.redirectBody, headers: headers)
            }
        }
    }
    override func stopLoading() {
        let active = lock.withLock { stopped = true; return fixture }
        active?.didStop(self)
    }
    func send(status: Int, body: Data, headers: [String: String], responseURL: URL? = nil) {
        let response = HTTPURLResponse(url: responseURL ?? request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        onLoadingRunLoop { [self] in
            guard claimTerminal() else { return }
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            guard !isStopped else { return }
            client?.urlProtocol(self, didLoad: body)
            guard !isStopped else { return }
            client?.urlProtocolDidFinishLoading(self)
        }
    }
    private func fail(_ error: URLError) {
        onLoadingRunLoop { [self] in
            guard claimTerminal() else { return }
            client?.urlProtocol(self, didFailWithError: error)
        }
    }
    private var isStopped: Bool { lock.withLock { stopped } }
    private func claimTerminal() -> Bool {
        lock.withLock {
            guard !stopped, !terminalStarted else { return false }
            terminalStarted = true
            return true
        }
    }
    private func onLoadingRunLoop(_ action: @escaping @Sendable () -> Void) {
        guard let loop = lock.withLock({ loadingRunLoop }) else { return }
        // Capture this loop at startLoading, never the main actor's loop when a
        // held fixture is released by the test. No client invocation holds lock.
        CFRunLoopPerformBlock(loop, CFRunLoopMode.defaultMode.rawValue, action)
        CFRunLoopWakeUp(loop)
    }
}

private extension URLSession {
    func recordingTransportTasks() async -> [URLSessionTask] {
        await withCheckedContinuation { continuation in
            getAllTasks { continuation.resume(returning: $0) }
        }
    }
}
