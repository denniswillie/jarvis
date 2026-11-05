import CryptoKit
import Foundation

private final class MediaFileProtocol: URLProtocol {
    static var handler: ((URLProtocol) throws -> Void)?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do { try Self.handler!(self) }
        catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@main
enum PhoneMediaTransportRegression {
    static func main() async throws {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MediaFileProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel(); MediaFileProtocol.handler = nil }
        let api = JarvisAPIClient(configuration: JarvisConfiguration(backendBaseURL: URL(string: "https://media-fixture.invalid")!,
            staticBearerToken: "fixture-only"), session: session)
        let id = UUID(), sessionID = UUID(), version = "exact/source+version"
        let chunk = Data((0..<(64 * 1024)).map { UInt8($0 % 251) }), chunks = 145
        let size = chunk.count * chunks
        var hash = SHA256()
        for _ in 0..<chunks { hash.update(data: chunk) }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let payload: [String: Any] = ["file": ["id": id.uuidString.lowercased(), "filename": "clip.mp4",
            "content_type": "video/mp4", "size_bytes": size, "description": "Clip", "source": "retained",
            "created_at": "2026-11-02T00:00:00Z", "version_id": version, "content_sha256": digest,
            "transfer_limit_bytes": 32 * 1024 * 1024,
            "memory_source": ["source_id": UUID().uuidString, "revision": 1, "generation": 1,
                "content_digest": "source-digest", "source_type": "file", "trust": "user", "time_basis": "recorded",
                "recorded_at": "2026-11-02T00:00:00Z", "is_current": true, "ingestion_status": "ready"],
            "media_inspection": ["format": "jarvis.mp4_inspection.v1", "content_sha256": digest,
                "size_bytes": size, "duration_ms": 20_000, "width": 1920, "height": 1080,
                "video_codec": "h264", "audio_codec": "aac"]]]
        let encoded = try JSONSerialization.data(withJSONObject: payload)
        for (policy, duration, accepted) in [("mp4_capture_v1", 1_800_100, true),
            ("mp4_capture_v1", 1_800_101, false), ("mp4_output_v1", 30_000, true),
            ("mp4_output_v1", 30_001, false), ("", 1_800_100, false)] {
            var row = payload["file"] as! [String: Any]
            var inspection = row["media_inspection"] as! [String: Any]
            inspection["duration_ms"] = duration; row["media_inspection"] = inspection
            if !policy.isEmpty { row["transfer_policy"] = policy }
            let selected = try JSONDecoder().decode(JarvisMediaFile.self,
                from: JSONSerialization.data(withJSONObject: row))
            let valid = (try? selected.validate(fileID: id, versionID: version,
                maximumByteCount: 1024 * 1024 * 1024, maximumDurationMilliseconds: 1_800_100)) != nil
            check(valid == accepted, "Only authenticated capture policy grants codec overhead: \(policy) \(duration)")
        }
        var requests: [URLRequest] = []
        enum Outcome { case success, corrupt, wrongVersion, missingLength, oversized, denied, retired }
        var outcome = Outcome.success
        var activeAccess = JarvisWorkspaceLocalAccess()
        MediaFileProtocol.handler = { transport in
            let request = transport.request
            requests.append(request)
            precondition(request.url?.host == "media-fixture.invalid")
            var headers = ["Content-Type": "application/json"]
            let isContent = request.url!.path.hasSuffix("/content")
            if isContent {
                headers = ["Content-Type": "video/mp4", "Content-Length": String(size), "X-Jarvis-Version-Id": version,
                    "X-Content-SHA256": digest, "ETag": "\"\(digest)\"", "Cache-Control": "private, no-store", "Vary": "Authorization"]
                if outcome == .wrongVersion { headers["X-Jarvis-Version-Id"] = "other" }
                if outcome == .missingLength { headers.removeValue(forKey: "Content-Length") }
            }
            let status = isContent && outcome == .denied ? 403 : 200
            let deniedBody = Data("{\"error\":{\"code\":\"denied\",\"message\":\"Sign in\"}}".utf8)
            if status == 403 { headers = ["Content-Type": "application/json", "Content-Length": String(deniedBody.count)] }
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
            transport.client?.urlProtocol(transport, didReceive: response, cacheStoragePolicy: .notAllowed)
            if status == 403 {
                transport.client?.urlProtocol(transport, didLoad: deniedBody)
                transport.client?.urlProtocolDidFinishLoading(transport)
                return
            }
            if isContent {
                for index in 0..<chunks {
                    if index == 1 && outcome == .retired { activeAccess.invalidate() }
                    let data = index == 0 && outcome == .corrupt ? Data(repeating: 0, count: chunk.count) : chunk
                    transport.client?.urlProtocol(transport, didLoad: data)
                }
                if outcome == .oversized { transport.client?.urlProtocol(transport, didLoad: chunk) }
            } else { transport.client?.urlProtocol(transport, didLoad: encoded) }
            transport.client?.urlProtocolDidFinishLoading(transport)
        }

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-media-transport-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let verifier = JarvisTemporaryFileVerifier(root: root)
        let media = try await api.getRetainedMediaFile(id: id, versionID: version, sessionID: sessionID,
            maximumByteCount: 32 * 1024 * 1024, maximumDurationMilliseconds: 30_000, access: activeAccess)
        check(media.versionID == version && media.file.sizeBytes == size, "Exact real-client metadata envelope decodes")
        let saved = try await api.downloadRetainedMedia(media, sessionID: sessionID, maximumByteCount: 32 * 1024 * 1024,
            maximumDurationMilliseconds: 30_000, access: activeAccess, verifier: verifier)
        let location = try saved.urlForReading()
        check(saved.byteCount == size && saved.contentSHA256 == digest && FileManager.default.fileExists(atPath: location.path),
              "URLSession downloads a greater-than-8-MiB private file through the real client and verifier")
        check(location.pathExtension == "mp4", "Verified video preserves the container extension required by AVFoundation")
        check(requests.count == 2, "One metadata read and one content read; no receipt or hidden retry")
        for (index, request) in requests.enumerated() {
            let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!
            check(request.httpMethod == "GET" && request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-only",
                  "Both reads use the existing authenticated client")
            check(query.count == 2 && query.first(where: { $0.name == "session_id" })?.value == sessionID.uuidString.lowercased()
                  && query.first(where: { $0.name == "version_id" })?.value == version,
                  "Both requests preserve the captured session and opaque exact version")
            check(request.url!.absoluteString.contains("%2B"), "Literal plus survives the server's form query decoding")
            check(request.url?.path == "/files/" + id.uuidString.lowercased() + (index == 1 ? "/content" : ""),
                  "Existing Files routes are retained")
        }
        saved.discard()
        check(!FileManager.default.fileExists(atPath: location.path), "Discard removes the verified temporary file")
        var prepared = JarvisWorkspaceRequest(requestID: UUID(), action: .download, artifactKind: "file", artifactID: id,
            versionID: version, format: nil, contentReady: true, fileID: id, byteCount: size, contentSHA256: digest,
            state: .pending, createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil)
        prepared.transferPolicy = .output; prepared.transferLimitBytes = 12 * 1024 * 1024
        let beforePrepared = requests.count
        let preparedFile = try await api.getWorkspaceMediaContent(request: prepared, sessionID: sessionID, access: activeAccess)
        check(requests.count == beforePrepared + 2 && preparedFile.temporaryFile.contentSHA256 == digest
              && preparedFile.filename == "clip.mp4" && preparedFile.contentType == "video/mp4",
              "The actual prepared-offer adapter verifies metadata then downloads the exact selected media")
        check(preparedFile.maximumByteCount == 12 * 1024 * 1024 && preparedFile.maximumDurationMilliseconds == 20_000,
              "Playback retains the narrower prepared byte policy and exact inspected duration")
        preparedFile.temporaryFile.discard()
        let openRequest = JarvisWorkspaceRequest(requestID: UUID(), action: .open, artifactKind: "file", artifactID: id,
            versionID: version, format: nil, contentReady: true, fileID: nil, byteCount: nil, contentSHA256: nil,
            state: .pending, createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil)
        let beforeOpen = requests.count
        let openFile = try await api.getWorkspaceMediaContent(request: openRequest, sessionID: sessionID, access: activeAccess)
        check(requests.count == beforeOpen + 2 && openFile.temporaryFile.contentSHA256 == digest,
              "Open offers without save-policy fields resolve exact metadata and private bytes, without inventing a save receipt")
        check(openFile.maximumByteCount == 32 * 1024 * 1024 && openFile.maximumDurationMilliseconds == 20_000,
              "Generic open carries the selected verified media bounds instead of an unrelated60second limit")
        openFile.temporaryFile.discard()
        let wrongPrepared = JarvisWorkspaceRequest(requestID: UUID(), action: .download, artifactKind: "file", artifactID: id,
            versionID: version, format: nil, contentReady: true, fileID: id, byteCount: size, contentSHA256: String(repeating: "0", count: 64),
            state: .pending, createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil,
            transferPolicy: .output, transferLimitBytes: 12 * 1024 * 1024)
        let beforeMismatch = requests.count
        do { _ = try await api.getWorkspaceMediaContent(request: wrongPrepared, sessionID: sessionID, access: activeAccess)
            preconditionFailure("Changed prepared bytes downloaded")
        } catch is JarvisWorkspaceError { check(requests.count == beforeMismatch + 1, "Prepared hash mismatch stops after metadata, before content download") }
        for failure in [Outcome.corrupt, .wrongVersion, .missingLength, .oversized, .denied, .retired] {
            outcome = failure; activeAccess = JarvisWorkspaceLocalAccess()
            let previousRequests = requests.count
            do {
                _ = try await api.downloadRetainedMedia(media, sessionID: sessionID, maximumByteCount: 32 * 1024 * 1024,
                    maximumDurationMilliseconds: 30_000, access: activeAccess, verifier: verifier)
                preconditionFailure("An invalid transfer produced a usable file")
            } catch {
                if failure == .denied { check((error as? JarvisAPIError)?.statusCode == 403, "Content denial preserves status for immediate account retirement") }
                check(requests.count == previousRequests + 1, "Failed content transfer is not retried")
                check(try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).isEmpty,
                      "Failed or retired transfer leaves no private partial or usable result")
            }
        }
        let before = requests.count
        do {
            _ = try await api.downloadRetainedMedia(media, sessionID: sessionID, maximumByteCount: 32 * 1024 * 1024,
                maximumDurationMilliseconds: 30_000, access: activeAccess, verifier: verifier)
            preconditionFailure("Retired account started a download")
        } catch is CancellationError { check(requests.count == before, "Retired account cannot start network work") }
        print("Phone media transport regressions passed: \(checks) checks; fixture_bytes=\(size); URLProtocol only")
    }
}
