import AVFoundation
import CryptoKit
import Foundation

private final class FilesURLProtocol: URLProtocol {
    static var responder: ((URLRequest) throws -> (Int, Data))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, bytes) = try Self.responder!(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: bytes)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@MainActor
private final class FilesMediaFixture: JarvisFilesMediaAccess {
    let context = JarvisWorkspaceLocalAccess()
    let source: URL, verifier: JarvisTemporaryFileVerifier
    var media: JarvisMediaFile
    var versions: [String?] = []
    var downloads: [URL] = []
    var documentReads = 0
    var metadataError: Error?
    var holdDownload = false
    var pending: CheckedContinuation<Void, Never>?
    init(source: URL, root: URL, media: JarvisMediaFile) {
        self.source = source; self.media = media
        verifier = JarvisTemporaryFileVerifier(root: root)
    }
    nonisolated func checkContext() throws { try context.check() }
    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage {
        try checkContext(); return .init(files: [media.file], pageFull: false)
    }
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile { try checkContext(); return media.file }
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        documentReads += 1; preconditionFailure("MP4 must never enter whole-Data delivery")
    }
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        .init(deleted: true, storageCleanup: "deleted", fullErasureVerified: false)
    }
    func getMediaFile(id: UUID, versionID: String?, access: JarvisWorkspaceLocalAccess) async throws -> JarvisMediaFile {
        try checkContext(); try access.check(); versions.append(versionID)
        if let metadataError { throw metadataError }; return media
    }
    func getMediaContent(_ media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess) async throws -> JarvisVerifiedTemporaryFile {
        let file = try await verifier.copyVerifiedFile(from: source, expectedByteCount: media.file.sizeBytes,
            expectedSHA256: media.contentSHA256, maximumByteCount: 64 * 1024 * 1024, access: access, format: .mp4)
        downloads.append(try file.urlForReading())
        if holdDownload { await withCheckedContinuation { pending = $0 } }
        return file
    }
}

@main
enum PhoneFilesMediaRegression {
    @MainActor
    static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-files-media-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let reader = try FileHandle(forReadingFrom: source)
        var hash = SHA256(), byteCount = 0
        while let chunk = try reader.read(upToCount: 65_536), !chunk.isEmpty { hash.update(data: chunk); byteCount += chunk.count }
        try reader.close()
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let id = UUID(), sessionID = UUID(), version = "source_exact/+v1"
        func metadata(version: String) -> [String: Any] {
            ["id": id.uuidString, "filename": "Synthetic clip.mp4", "content_type": "video/mp4",
             "size_bytes": byteCount, "description": "Synthetic test clip", "source": "generated",
             "created_at": "2026-11-02T07:00:00Z", "version_id": version, "content_sha256": digest,
             "transfer_limit_bytes": 64 * 1024 * 1024,
             "memory_source": ["source_id": id.uuidString, "revision": 1, "generation": 1,
                "content_digest": digest, "source_type": "file", "trust": "user_provided",
                "time_basis": "recorded", "recorded_at": "2026-11-02T07:00:00Z", "is_current": true, "ingestion_status": "ready"],
             "media_inspection": ["format": "jarvis.mp4_inspection.v1", "content_sha256": digest,
                "size_bytes": byteCount, "duration_ms": 3000, "width": 1280, "height": 720, "video_codec": "h264"]]
        }
        func bytes(_ value: Any) throws -> Data { try JSONSerialization.data(withJSONObject: value) }
        func decode(_ version: String) throws -> JarvisMediaFile {
            try JSONDecoder().decode(JarvisMediaFile.self, from: bytes(metadata(version: version)))
        }
        var checks = 0
        func check(_ condition: Bool, _ label: String) { precondition(condition, label); checks += 1 }
        let media = try decode(version)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FilesURLProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel(); FilesURLProtocol.responder = nil }
        let api = JarvisAPIClient(configuration: .init(backendBaseURL: URL(string: "https://files-fixture.invalid")!,
            staticBearerToken: "fixture-only"), session: session)
        let context = JarvisWorkspaceLocalAccess()
        let client = JarvisSessionFilesClient(api: api, sessionID: sessionID, contextAccess: context,
            mediaLimits: .init(byteCount: 64 * 1024 * 1024, durationMilliseconds: 60_000))
        var requests: [URLRequest] = []
        FilesURLProtocol.responder = { request in
            requests.append(request)
            let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems ?? []
            check(query.first { $0.name == "session_id" }?.value == sessionID.uuidString.lowercased(), "Every library operation retains the verified session")
            check(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-only", "Existing authentication reused")
            if request.httpMethod == "DELETE" {
                return (200, try bytes(["deleted": true, "storage_cleanup": "deleted", "full_erasure_verified": false]))
            }
            if request.url!.path == "/files" {
                check(query.first { $0.name == "query" }?.value == "A&B café", "Library search is encoded safely")
                return (200, try bytes(["files": [metadata(version: version)], "page_full": false]))
            }
            return (200, try bytes(["file": metadata(version: version)]))
        }
        _ = try await client.getRetainedFiles(query: "A&B café")
        _ = try await client.getRetainedFile(id: id.uuidString)
        _ = try await client.getMediaFile(id: id, versionID: version, access: JarvisWorkspaceLocalAccess())
        let query = URLComponents(url: requests.last!.url!, resolvingAgainstBaseURL: false)!.queryItems!
        check(query.first { $0.name == "version_id" }?.value == version, "Opaque exact version survives export metadata lookup")
        _ = try await client.deleteRetainedFile(id: id.uuidString)
        // Match the real gallery's encoded playback policy, not a smaller
        // fixture-only limit which hid rejection of every production movie.
        let playbackDuration = JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumDurationMilliseconds
        let gallery = JarvisSessionFilesClient(api: api, sessionID: sessionID, contextAccess: context,
            mediaLimits: .init(byteCount: 1024 * 1024 * 1024, durationMilliseconds: playbackDuration))
        let openedMetadata = try await gallery.getMediaFile(id: id, versionID: version,
            access: JarvisWorkspaceLocalAccess())
        check(openedMetadata.versionID == version, "Production gallery policy allows an exact saved-video lookup")
        for invalid in [
            JarvisSessionFilesClient.MediaLimits(byteCount: 1024 * 1024 * 1024, durationMilliseconds: playbackDuration + 1),
            JarvisSessionFilesClient.MediaLimits(byteCount: 1024 * 1024 * 1024 + 1, durationMilliseconds: playbackDuration)
        ] {
            let count = requests.count
            let invalidClient = JarvisSessionFilesClient(api: api, sessionID: sessionID,
                contextAccess: context, mediaLimits: invalid)
            do {
                _ = try await invalidClient.getMediaFile(id: id, versionID: version, access: JarvisWorkspaceLocalAccess())
                preconditionFailure("Oversized media policy accepted")
            } catch {
                check(requests.count == count, "Over-limit playback configuration is rejected before network access")
            }
        }
        let requestCount = requests.count
        do { _ = try await client.getRetainedFileContent(file: media.file); preconditionFailure("Media Data fallback") }
        catch { check(requests.count == requestCount, "MP4 rejected before a Data request") }
        var smallMetadata = metadata(version: version)
        smallMetadata["size_bytes"] = 4
        let smallMedia = try JSONDecoder().decode(JarvisRetainedFile.self, from: bytes(smallMetadata))
        do { _ = try await client.getRetainedFileContent(file: smallMedia); preconditionFailure("Small media Data fallback") }
        catch { check(requests.count == requestCount, "MP4 smaller than the document cap still cannot use Data delivery") }
        let gated = JarvisSessionFilesClient(api: api, sessionID: sessionID, contextAccess: context)
        do { _ = try await gated.getMediaFile(id: id, versionID: nil, access: JarvisWorkspaceLocalAccess()); preconditionFailure("Disabled media accepted") }
        catch { check(requests.count == requestCount, "Unqualified media stays disabled") }
        context.invalidate()
        do { _ = try await client.getRetainedFiles(query: nil); preconditionFailure("Retired account requested") }
        catch { check(requests.count == requestCount, "Retired account cannot request files") }

        let lateContext = JarvisWorkspaceLocalAccess()
        let lateClient = JarvisSessionFilesClient(api: api, sessionID: sessionID, contextAccess: lateContext)
        FilesURLProtocol.responder = { _ in
            lateContext.invalidate()
            return (200, try bytes(["file": metadata(version: version)]))
        }
        do { _ = try await lateClient.getRetainedFile(id: id.uuidString); preconditionFailure("Late context metadata accepted") }
        catch { check(true, "A response arriving after account retirement is rejected") }

        let fixture = FilesMediaFixture(source: source, root: root, media: media)
        let library = JarvisFilesViewModel(api: fixture)
        await library.reload()
        _ = await library.open(media.file)
        let model = library.media!
        check(model.selected?.versionID == version && model.preview.player?.currentItem != nil,
              "Library preview opens a real local H264 asset")
        check(library.content == nil && fixture.documentReads == 0 && model.preview.player?.rate == 0,
              "Preview stays paused without whole-Data delivery")
        let playerURL = fixture.downloads.last!
        let prepared = await model.prepareExport()
        let exported = model.preparedExport!
        let exportURL = try exported.file.urlForReading()
        check(prepared && exportURL != playerURL && fixture.versions.last! == version,
              "Export reauthorizes exact selection and owns a separate verified file")
        model.finishExport(id: UUID(), completed: true)
        check(model.preparedExport?.id == exported.id && model.exportMessage == nil, "Unrelated picker completion cannot claim export")
        model.finishExport(id: exported.id, completed: false)
        check(model.preparedExport == nil && model.exportMessage == nil && !FileManager.default.fileExists(atPath: exportURL.path),
              "Picker cancellation releases export without success")
        check(FileManager.default.fileExists(atPath: playerURL.path) && model.preview.player?.currentItem != nil,
              "Export cancellation preserves active preview")
        fixture.media = try decode("source_newer")
        let reads = fixture.downloads.count
        let changed = await model.prepareExport()
        check(!changed && model.preparedExport == nil && fixture.downloads.count == reads,
              "Changed version is rejected before downloading export")
        fixture.media = media
        let second = await model.prepareExport()
        check(second, "Same selected version remains exportable")
        let secondID = model.preparedExport!.id
        model.finishExport(id: secondID, completed: true)
        check(model.exportMessage == "Exported a separate copy." && model.preparedExport == nil,
              "Only matching successful completion reports export")
        fixture.holdDownload = true
        let pending = Task { await model.prepareExport() }
        for _ in 0..<1000 where fixture.pending == nil { try await Task.sleep(nanoseconds: 1_000_000) }
        check(fixture.pending != nil, "Export reaches held byte completion")
        let lateURL = fixture.downloads.last!, retainedPlayer = model.preview.player
        fixture.context.invalidate(); library.deactivate()
        fixture.pending?.resume(); fixture.pending = nil
        let late = await pending.value
        check(!late && model.preparedExport == nil && model.preview.player == nil && retainedPlayer?.currentItem == nil,
              "Account retirement rejects late export and releases player")
        check(!FileManager.default.fileExists(atPath: lateURL.path) && !FileManager.default.fileExists(atPath: playerURL.path),
              "Retirement removes private player and late export files")
        check(FileManager.default.fileExists(atPath: source.path), "Original fixture survives private copy cleanup")

        let denied = FilesMediaFixture(source: source, root: root, media: media)
        denied.metadataError = JarvisAPIError(code: "denied", message: "Sign in", statusCode: 403)
        let deniedModel = JarvisFilesViewModel(api: denied)
        var locked = false
        deniedModel.onAuthorizationFailure = { locked = true; deniedModel.deactivate() }
        _ = await deniedModel.open(media.file)
        check(locked && deniedModel.selectedFile == nil && denied.downloads.isEmpty,
              "Media authorization failure locks the enclosing account before bytes")
        print("Phone Files media regressions passed: \(checks) checks; synthetic bytes=\(byteCount); no playback or external export")
    }
}
