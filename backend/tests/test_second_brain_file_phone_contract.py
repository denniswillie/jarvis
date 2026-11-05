"""Opt-in actual Swift model, HTTP client and file-view-model contracts."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from uuid import uuid4

from app.files.models import JarvisFile
from app.files.tools import _file_payload

SWIFT = r'''
import Foundation

// Reminder date rendering is outside this Foundation-only file contract.
// The complete iOS app build uses the production formatter extension.
extension ISO8601DateFormatter {
    static let jarvisFormatter = ISO8601DateFormatter()
}

private final class FileProtocol: URLProtocol {
    static var handler: ((URLRequest) -> (Int, [String: String], Data))!
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (status, headers, data) = Self.handler(request)
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@MainActor
private final class FileAPI: JarvisFileAccess {
    let file: JarvisRetainedFile
    let bytes: Data
    var unavailable = false
    var removed = false
    var pauseContent = false
    var pending: CheckedContinuation<Data, Error>?
    var pauseList = false
    var pendingList: CheckedContinuation<JarvisRetainedFilePage, Error>?
    var reads = 0
    init(file: JarvisRetainedFile, bytes: Data) { self.file = file; self.bytes = bytes }
    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage {
        if pauseList { return try await withCheckedThrowingContinuation { pendingList = $0 } }
        if unavailable { throw JarvisAPIError(code: "unavailable", message: "Source unavailable") }
        return JarvisRetainedFilePage(files: removed ? [] : [file], pageFull: !removed)
    }
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        reads += 1
        if unavailable { throw JarvisAPIError(code: "unavailable", message: "Source unavailable") }
        return file
    }
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        if pauseContent { return try await withCheckedThrowingContinuation { pending = $0 } }
        return bytes
    }
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        removed = true
        return JarvisFileDeletion(deleted: true, storageCleanup: "deleted", fullErasureVerified: false)
    }
}

@main struct FilePhoneContract {
    @MainActor static func main() async throws {
        let json = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let file = try JSONDecoder().decode(JarvisRetainedFile.self, from: json)
        let bytes = Data("\n    return middle\n\n".utf8)
        precondition(file.filename == "../code: draft.txt")
        precondition(file.exportFilename == "..-code- draft.txt")
        precondition(file.memorySource.trust == "assistant_output")
        try file.validateContent(bytes, contentType: "text/plain; charset=utf-8")
        for (body, type) in [(Data("truncated".utf8), "text/plain"), (bytes, "text/html")] {
            var rejected = false
            do { try file.validateContent(body, contentType: type) } catch { rejected = true }
            precondition(rejected)
        }
        var payload = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        payload["size_bytes"] = 8 * 1024 * 1024 + 1
        let oversized = try JSONDecoder().decode(JarvisRetainedFile.self,
            from: JSONSerialization.data(withJSONObject: payload))
        var rejected = false
        do { try oversized.validateContent(bytes, contentType: "text/plain") } catch { rejected = true }
        precondition(rejected)

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FileProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let api = JarvisAPIClient(configuration: JarvisConfiguration(
            backendBaseURL: URL(string: "https://files.invalid/api")!, staticBearerToken: "synthetic-token"),
            session: session)
        var requestCount = 0
        var mode = "normal"
        FileProtocol.handler = { request in
            requestCount += 1
            precondition(request.value(forHTTPHeaderField: "Authorization") == "Bearer synthetic-token")
            precondition(request.cachePolicy == .reloadIgnoringLocalCacheData)
            let path = request.url!.path
            if path == "/api/files" {
                let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems
                precondition(query?.first(where: { $0.name == "query" })?.value == "draft & ?")
                return (200, ["Content-Type": "application/json"],
                    Data(("{\"files\":[" + String(data: json, encoding: .utf8)! + "],\"page_full\":true}").utf8))
            }
            if path == "/api/sessions/active" {
                return (200, ["Content-Type": "application/json"], Data("{\"session\":null,\"realtime\":null}".utf8))
            }
            precondition(path == "/api/files/" + file.id || path == "/api/files/" + file.id + "/content")
            if mode == "revoked" {
                return (409, ["Content-Type": "application/json"],
                    Data("{\"error\":{\"code\":\"knowledge_source_unavailable\",\"message\":\"Original no longer readable.\"}}".utf8))
            }
            if request.httpMethod == "DELETE" {
                return (200, ["Content-Type": "application/json"],
                    Data("{\"deleted\":true,\"storage_cleanup\":\"deleted\",\"full_erasure_verified\":false}".utf8))
            }
            if path.hasSuffix("/content") {
                precondition(request.value(forHTTPHeaderField: "Accept") == "text/plain")
                return (200, ["Content-Type": mode == "bad-mime" ? "text/html" : file.contentType], bytes)
            }
            return (200, ["Content-Type": "application/json"],
                Data(("{\"file\":" + String(data: json, encoding: .utf8)! + "}").utf8))
        }
        let page = try await api.getRetainedFiles(query: "draft & ?")
        precondition(page.files == [file] && page.pageFull)
        let metadata = try await api.getRetainedFile(id: file.id)
        precondition(metadata == file)
        let downloaded = try await api.getRetainedFileContent(file: metadata)
        precondition(downloaded == bytes)
        let oldCount = requestCount
        do { _ = try await api.getRetainedFile(id: "../bad"); preconditionFailure("Invalid ID reached transport") }
        catch { precondition(requestCount == oldCount) }
        mode = "bad-mime"
        do { _ = try await api.getRetainedFileContent(file: file); preconditionFailure("Bad MIME accepted") }
        catch { precondition((error as? JarvisAPIError)?.code == "file_content_mismatch") }
        mode = "revoked"
        do { _ = try await api.getRetainedFileContent(file: file); preconditionFailure("Revoked source accepted") }
        catch { precondition((error as? JarvisAPIError)?.code == "knowledge_source_unavailable") }
        mode = "normal"
        let removed = try await api.deleteRetainedFile(id: file.id)
        precondition(removed.deleted && !removed.fullErasureVerified)
        let legacySession = try await api.getActiveSession()
        precondition(legacySession == nil)

        let fake = FileAPI(file: file, bytes: bytes)
        let model = JarvisFilesViewModel(api: fake)
        await model.reload()
        precondition(model.files == [file] && model.pageFull)
        let first = await model.open(file)
        precondition(first?.data == bytes)
        let freshExport = await model.open(file)
        precondition(freshExport?.data == bytes && fake.reads == 2)
        model.closePreview()
        let linked = await model.open(id: file.id)
        precondition(linked?.data == bytes && model.selectedFile == file)
        let wrongLink = await model.open(id: UUID().uuidString)
        precondition(wrongLink == nil && model.content == nil && model.selectedFile == nil)
        precondition(model.contentError != nil)
        fake.unavailable = true
        let denied = await model.open(id: file.id)
        precondition(denied == nil && model.content == nil && model.contentError != nil)
        await model.reload()
        precondition(model.files.isEmpty && model.errorMessage != nil)
        fake.unavailable = false
        fake.pauseContent = true
        let late = Task { await model.open(file) }
        while fake.pending == nil { await Task.yield() }
        model.closePreview()
        fake.pending!.resume(returning: bytes)
        _ = await late.value
        precondition(model.content == nil && model.selectedFile == nil)
        fake.pauseContent = false
        fake.pauseList = true
        let lateList = Task { await model.reload() }
        while fake.pendingList == nil { await Task.yield() }
        model.clear()
        fake.pendingList!.resume(returning: JarvisRetainedFilePage(files: [file], pageFull: true))
        await lateList.value
        precondition(model.files.isEmpty && !model.isLoading)
        fake.pauseList = false
        await model.reload()
        let deleted = await model.remove(file)
        precondition(deleted && model.files.isEmpty && model.content == nil)
        precondition(model.deletionMessage?.contains("Exported copies are not removed") == true)
        print("File phone contract passed")
    }
}
'''


@unittest.skipUnless(os.getenv("JARVIS_IOS_CONTRACT_TESTS") == "1" and shutil.which("swiftc"),
                     "Opt-in native Swift compiler contract")
class FilePhoneContractTests(unittest.TestCase):
    def test_actual_file_payload_client_auth_and_view_model_revocation(self):
        content = "\n    return middle\n\n"
        file = JarvisFile(id=uuid4(), session_id=uuid4(), filename="../code: draft.txt",
            content_type="text/plain; charset=utf-8", storage_bucket="synthetic",
            storage_path="must-not-reach-phone", size_bytes=len(content.encode()), description="Draft",
            source="assistant", source_visual_frame_id=None, created_at=datetime.now(UTC),
            memory_source={"source_id": str(uuid4()), "revision": 1, "generation": 1,
                "content_digest": "a" * 64, "source_type": "text_file", "trust": "assistant_output",
                "source_time": None, "time_basis": "Original adoption time", "recorded_at": "2026-09-15T12:00:00Z",
                "is_current": True, "ingestion_status": "pending"})
        payload = _file_payload(file)
        self.assertNotIn("storage_path", payload)
        ios = Path(__file__).resolve().parents[2] / "ios/Jarvis/Jarvis"
        sources = [ios / "Models" / name for name in (
            "JarvisModels.swift", "JarvisMemoryModels.swift", "JarvisFileModels.swift", "JarvisWorkModels.swift",
            "JarvisReminderModels.swift", "JarvisCalendarModels.swift")]
        sources += [ios / "Services/JarvisAPIClient.swift", ios / "ViewModels/JarvisFilesViewModel.swift"]
        with tempfile.TemporaryDirectory(prefix="jarvis-file-phone-swift-") as folder:
            root = Path(folder)
            (root / "Contract.swift").write_text(SWIFT)
            (root / "file.json").write_text(json.dumps(payload))
            compiled = subprocess.run(["swiftc", *map(str, sources), str(root / "Contract.swift"),
                                       "-o", str(root / "check")], capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(root / "check"), str(root / "file.json")],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("File phone contract passed", result.stdout)
