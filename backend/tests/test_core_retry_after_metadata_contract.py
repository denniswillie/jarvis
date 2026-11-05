"""Opt-in isolated Swift metadata contract; no Xcode, app, or network services.

Compile the real leased client/error files with inert unrelated model stubs.
URLProtocol intercepts every request, including mutations, within this process.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SWIFT = r'''
import Foundation

// Unrelated feature types are compile-only stubs, not alternate transports.
protocol JarvisFileAccess {}
protocol JarvisWorkAccess {}
struct ContractPlaceholder: Codable {}
struct JarvisWorkTask: Decodable { let id: UUID }
struct JarvisInboxItem: Decodable { let id: UUID }
struct JarvisInboxDetail: Decodable { let item: JarvisInboxItem }
enum ContractOperation: String { case cancel }
typealias JarvisTaskOperation = ContractOperation
typealias JarvisInboxOperation = ContractOperation
typealias JarvisTaskChange = ContractPlaceholder
typealias JarvisInboxChange = ContractPlaceholder
typealias JarvisInboxChangeReceipt = ContractPlaceholder
typealias JarvisReminderSummary = ContractPlaceholder
typealias JarvisReminderEvent = ContractPlaceholder
struct ActiveRemindersResponse: Decodable { let reminders: [JarvisReminderSummary] }
struct ReminderEventsResponse: Decodable { let events: [JarvisReminderEvent] }
typealias JarvisRecentMemoryPage = ContractPlaceholder
typealias JarvisMemoryDetail = ContractPlaceholder
typealias JarvisMemoryEvidencePage = ContractPlaceholder
typealias JarvisWorkingMemoryState = ContractPlaceholder
typealias JarvisRetainedFilePage = ContractPlaceholder
struct JarvisRetainedFile: Decodable {
    let id: String
    let mediaType: String
    func validateContent(_ data: Data, contentType: String?) throws {}
}
struct JarvisRetainedFileResponse: Decodable { let file: JarvisRetainedFile }
typealias JarvisFileDeletion = ContractPlaceholder
typealias GoogleCalendarStatusResponse = ContractPlaceholder
typealias GoogleOAuthStartResponse = ContractPlaceholder

func instant(_ value: String) -> Date {
    ISO8601DateFormatter().date(from: value)!
}

func httpDate(_ date: Date, format: String = "EEE, dd MMM yyyy HH:mm:ss 'GMT'") -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = format
    return formatter.string(from: date)
}

func checkParsing() {
    let old = JarvisAPIError(code: "old", message: "Safe message")
    precondition(old.statusCode == nil && old.retryAfter == nil)
    precondition(old.errorDescription == "Safe message")
    let oldInitializer: (String, String, Int?) -> JarvisAPIError = JarvisAPIError.init(code:message:statusCode:)
    precondition(oldInitializer("old", "Safe message", nil) == old)
    precondition(JarvisAPIError(code: "old", message: "Safe message", statusCode: 429).retryAfter == nil)

    let date = instant("1994-11-06T08:49:37Z")
    let now = date.addingTimeInterval(-60)
    precondition(JarvisAPIClient.retryAfterDelay(from: nil, relativeTo: now) == nil)
    for (header, expected) in [("0", 0.0), ("12", 12.0), (" \t0012\t ", 12.0)] {
        precondition(JarvisAPIClient.retryAfterDelay(from: header, relativeTo: now) == expected)
    }
    for header in [
        "Sun, 06 Nov 1994 08:49:37 GMT",
        "Sunday, 06-Nov-94 08:49:37 GMT",
        "Sun Nov  6 08:49:37 1994",
    ] {
        precondition(JarvisAPIClient.retryAfterDelay(from: header, relativeTo: now) == 60)
        precondition(JarvisAPIClient.retryAfterDelay(from: header, relativeTo: date) == 0)
        precondition(JarvisAPIClient.retryAfterDelay(from: header, relativeTo: date.addingTimeInterval(1)) == 0)
    }
    let doubleDigitDay = instant("1994-11-16T08:49:37Z")
    precondition(JarvisAPIClient.retryAfterDelay(
        from: httpDate(doubleDigitDay, format: "EEE MMM dd HH:mm:ss yyyy"),
        relativeTo: doubleDigitDay.addingTimeInterval(-60)) == 60)

    // RFC 850's two-digit year uses a rolling 50-year future boundary, not 2049.
    let modernNow = instant("2026-11-01T12:00:00Z")
    let future = instant("2050-01-01T00:00:00Z")
    precondition(JarvisAPIClient.retryAfterDelay(
        from: httpDate(future, format: "EEEE, dd-MMM-yy HH:mm:ss 'GMT'"),
        relativeTo: modernNow) == future.timeIntervalSince(modernNow))
    precondition(JarvisAPIClient.retryAfterDelay(
        from: "Sunday, 06-Nov-94 08:49:37 GMT", relativeTo: modernNow) == 0)

    for header in [
        "", " \t", "-1", "+2", "1.5", "1e3", "nan", "inf", "10, 20", "\u{0661}",
        "12\n", String(repeating: "9", count: 400),
        "Sun, 06 Nov 1994 08:49:37 GMT trailing",
        "Sun, 06 Nov 1994 08:49:37 UTC",
        "Sun, 31 Feb 1994 08:49:37 GMT",
        "1994-11-06T08:49:37Z",
    ] {
        precondition(JarvisAPIClient.retryAfterDelay(from: header, relativeTo: now) == nil,
                     "Unexpected Retry-After parse: \(header)")
    }
    print("Core Retry-After parsing passed")
}

final class CoreRetryAfterProtocol: URLProtocol {
    static var handler: ((URLRequest) -> (Int, [String: String], Data))!

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (status, headers, data) = Self.handler(request)
        let response = HTTPURLResponse(url: request.url!, statusCode: status,
                                       httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class RequestLog {
    private let lock = NSLock()
    private var values: [URLRequest] = []
    func record(_ request: URLRequest) {
        lock.lock()
        defer { lock.unlock() }
        values.append(request)
    }
    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

struct Probe: Decodable { let value: String }

func checkTransport() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [CoreRetryAfterProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let client = JarvisAPIClient(configuration: JarvisConfiguration(
        backendBaseURL: URL(string: "https://core-retry-after.invalid/api")!,
        staticBearerToken: "synthetic-contract-token"), session: session)
    let envelope = #"{"error":{"code":"busy","message":"Try later."}}"#
    let fixtures: [(String, Int, [String: String], String, String, Double?)] = [
        ("POST", 429, ["rEtRy-AfTeR": "12"], envelope, "busy", 12),
        ("PATCH", 503, ["Retry-After": "0"], "<html>PRIVATE RAW BODY</html>", "http_503", 0),
        ("DELETE", 503, ["Retry-After": "Sun, 06 Nov 1994 08:49:37 GMT"], envelope, "busy", 0),
        ("GET", 401, [:], envelope, "busy", nil),
        ("POST", 409, ["Retry-After": "-1"], #"{"error":{"code":7}}"#, "http_409", nil),
    ]
    for (method, status, headers, responseBody, code, delay) in fixtures {
        let log = RequestLog()
        CoreRetryAfterProtocol.handler = { request in
            log.record(request)
            return (status, headers, Data(responseBody.utf8))
        }
        do {
            let _: Probe = try await client.request(
                path: "/fixture", method: method, body: Data(#"{"value":1}"#.utf8),
                queryItems: [URLQueryItem(name: "cursor", value: "a+b")], responseType: Probe.self)
            preconditionFailure("Expected an HTTP error")
        } catch let error as JarvisAPIError {
            precondition(error.statusCode == status && error.code == code && error.retryAfter == delay)
            precondition(error.message == (code == "busy" ? "Try later." : "Jarvis request failed with status \(status)."))
        }
        precondition(log.requests.count == 1, "An HTTP failure must not repeat a mutation")
        let request = log.requests[0]
        precondition(request.httpMethod == method && request.url?.path == "/api/fixture")
        precondition(request.value(forHTTPHeaderField: "Authorization") == "Bearer synthetic-contract-token")
        precondition(request.value(forHTTPHeaderField: "Accept") == "application/json")
        precondition(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        precondition(URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems ==
                     [URLQueryItem(name: "cursor", value: "a+b")])
    }

    let future = Date().addingTimeInterval(120)
    CoreRetryAfterProtocol.handler = { _ in (503, ["Retry-After": httpDate(future)], Data(envelope.utf8)) }
    do {
        let _: Probe = try await client.request(path: "/date", method: "GET", responseType: Probe.self)
        preconditionFailure("Expected a dated HTTP error")
    } catch let error as JarvisAPIError {
        precondition((115...120).contains(error.retryAfter ?? -1))
    }

    // Raw file retrieval shares the same response/error seam.
    let file = JarvisRetainedFile(id: "f47ac10b-58cc-4372-a567-0e02b2c3d479", mediaType: "text/plain")
    let fileLog = RequestLog()
    CoreRetryAfterProtocol.handler = { request in
        fileLog.record(request)
        return (429, ["Retry-After": "3"], Data(envelope.utf8))
    }
    do {
        _ = try await client.getRetainedFileContent(file: file)
        preconditionFailure("Expected a file HTTP error")
    } catch let error as JarvisAPIError {
        precondition(error.retryAfter == 3 && error.code == "busy")
    }
    precondition(fileLog.requests.count == 1)
    precondition(fileLog.requests[0].value(forHTTPHeaderField: "Accept") == "text/plain")

    CoreRetryAfterProtocol.handler = { _ in (200, ["Retry-After": "12"], Data(#"{"value":"ok"}"#.utf8)) }
    let success: Probe = try await client.request(path: "/success", method: "GET", responseType: Probe.self)
    precondition(success.value == "ok")
    print("Core Retry-After transport passed")
}

@main struct CoreRetryAfterContract {
    static func main() async throws {
        if CommandLine.arguments[1] == "parsing" { checkParsing() }
        else { try await checkTransport() }
    }
}
'''


@unittest.skipUnless(
    os.getenv("JARVIS_IOS_CONTRACT_TESTS") == "1" and shutil.which("swiftc"),
    "Opt-in native Swift compiler contract",
)
class CoreRetryAfterMetadataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="jarvis-core-retry-after-")
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        harness = root / "Contract.swift"
        harness.write_text(SWIFT)
        ios = Path(__file__).resolve().parents[2] / "ios/Jarvis/Jarvis"
        cls.binary = root / "contract"
        compiled = subprocess.run(
            [
                "swiftc", "-swift-version", "5",
                str(ios / "Models/JarvisModels.swift"),
                str(ios / "Services/JarvisAPIClient.swift"),
                str(harness), "-o", str(cls.binary),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if compiled.returncode != 0:
            raise AssertionError(compiled.stdout + compiled.stderr)

    def run_contract(self, mode):
        result = subprocess.run(
            [str(self.binary), mode], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Core Retry-After {mode} passed", result.stdout)

    def test_initializer_compatibility_and_seconds_http_date_semantics(self):
        self.run_contract("parsing")

    def test_single_attempt_transport_and_safe_error_metadata(self):
        self.run_contract("transport")
