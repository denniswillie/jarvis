import AVFoundation
import CryptoKit
import Foundation

@MainActor
private final class LocalPreviewFixture: JarvisFilesMediaAccess {
    let context = JarvisWorkspaceLocalAccess()
    let source: URL
    let verifier: JarvisTemporaryFileVerifier
    var media: JarvisMediaFile
    var remoteReads = 0
    var versions: [String?] = []
    init(source: URL, root: URL, media: JarvisMediaFile) {
        self.source = source; self.media = media
        verifier = JarvisTemporaryFileVerifier(root: root)
    }
    nonisolated func checkContext() throws { try context.check() }
    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage {
        try checkContext(); return .init(files: [media.file], pageFull: false)
    }
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        try checkContext(); return media.file
    }
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        preconditionFailure("Video preview must never use document Data")
    }
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        preconditionFailure("Local preview does not delete a source")
    }
    func getMediaFile(id: UUID, versionID: String?, access: JarvisWorkspaceLocalAccess) async throws -> JarvisMediaFile {
        try checkContext(); try access.check(); versions.append(versionID); return media
    }
    func getMediaContent(_ media: JarvisMediaFile, access: JarvisWorkspaceLocalAccess) async throws -> JarvisVerifiedTemporaryFile {
        try checkContext(); try access.check(); remoteReads += 1
        return try await verifier.copyVerifiedFile(from: source, expectedByteCount: media.file.sizeBytes,
            expectedSHA256: media.contentSHA256, maximumByteCount: media.transferLimitBytes,
            access: access, format: .mp4)
    }
}

@main
enum PhoneFilesLocalPreviewRegression {
    @MainActor static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-local-preview-check-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let input = try FileHandle(forReadingFrom: source)
        var hash = SHA256(), byteCount = 0
        while let chunk = try input.read(upToCount: 65_536), !chunk.isEmpty { hash.update(data: chunk); byteCount += chunk.count }
        try input.close()
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let id = UUID(), version = "source/local-playback+v1"
        func metadata(limit: Int? = nil, versionID: String? = nil) throws -> JarvisMediaFile {
            let value: [String: Any] = ["id": id.uuidString, "filename": "Saved clip.mp4", "content_type": "video/mp4",
                "size_bytes": byteCount, "description": "Saved test clip", "source": "generated",
                "created_at": "2026-11-02T19:00:00Z", "version_id": versionID ?? version,
                "content_sha256": digest, "transfer_limit_bytes": limit ?? byteCount,
                "memory_source": ["source_id": id.uuidString, "revision": 1, "generation": 1,
                    "content_digest": digest, "source_type": "file", "trust": "user_provided",
                    "time_basis": "recorded", "recorded_at": "2026-11-02T19:00:00Z", "is_current": true, "ingestion_status": "ready"],
                "media_inspection": ["format": "jarvis.mp4_inspection.v1", "content_sha256": digest,
                    "size_bytes": byteCount, "duration_ms": 3000, "width": 1280, "height": 720, "video_codec": "h264"]]
            return try JSONDecoder().decode(JarvisMediaFile.self, from: JSONSerialization.data(withJSONObject: value))
        }
        let media = try metadata()
        func saved(url: URL? = nil, hash: String? = nil, size: Int? = nil) -> JarvisWorkspaceLocalFileContent {
            .init(url: url ?? source, filename: "Saved clip.mp4", contentType: "video/mp4",
                byteCount: size ?? byteCount, contentSHA256: hash ?? digest)
        }
        var checks = 0
        func check(_ condition: Bool, _ message: String) { precondition(condition, message); checks += 1 }

        let fixture = LocalPreviewFixture(source: source, root: root, media: media)
        var lookups: [JarvisResourceReference] = []
        let library = JarvisFilesViewModel(api: fixture, localMediaLookup: { reference, access in
            try access.check(); lookups.append(reference); return saved()
        })
        _ = await library.open(media.file)
        let localModel = library.media!
        check(localModel.selected?.versionID == version && localModel.preview.player?.currentItem != nil,
              "FilesVM passes the local lookup into actual video preview")
        check(fixture.remoteReads == 0 && lookups.count == 1 && lookups[0].fileID == id && lookups[0].versionID == version,
              "Exact saved file/version opens without a second remote content download")
        let privateURL = (localModel.preview.player!.currentItem!.asset as! AVURLAsset).url
        let preparedLocalExport = await localModel.prepareExport()
        check(preparedLocalExport && fixture.remoteReads == 0 && lookups.count == 2,
              "Export revalidates metadata and reuses exact saved bytes without a second cloud download")
        let export = localModel.preparedExport!
        let exportURL = try export.file.urlForReading()
        check(exportURL != privateURL && exportURL != source && export.file.contentSHA256 == digest,
              "Export owns a separate verified copy while playback retains its own file")
        localModel.finishExport(id: UUID(), completed: true)
        check(localModel.preparedExport?.id == export.id && localModel.exportMessage == nil,
              "An unrelated picker completion cannot claim export success")
        localModel.finishExport(id: export.id, completed: false)
        check(!FileManager.default.fileExists(atPath: exportURL.path)
              && FileManager.default.fileExists(atPath: privateURL.path),
              "Cancelled export discards only export bytes and leaves playback usable")
        check(privateURL != source && FileManager.default.fileExists(atPath: privateURL.path),
              "Player owns a distinct verified private copy rather than the durable original")
        check(localModel.preview.player?.rate == 0, "Opening a saved copy does not auto-play")
        library.closePreview()
        check(!FileManager.default.fileExists(atPath: privateURL.path) && FileManager.default.fileExists(atPath: source.path),
              "Closing preview removes only the private viewing copy")

        let absent = LocalPreviewFixture(source: source, root: root, media: media)
        let remoteModel = JarvisFileMediaPreviewModel(api: absent, localMediaLookup: { _, _ in nil })
        let remoteOpened = await remoteModel.open(id: id, versionID: version)
        check(remoteOpened && absent.remoteReads == 1 && absent.versions == [version],
              "A missing saved copy uses the existing exact-version remote fallback once")
        remoteModel.close()

        let mismatched = LocalPreviewFixture(source: source, root: root, media: media)
        let wrongLocal = JarvisFileMediaPreviewModel(api: mismatched,
            localMediaLookup: { _, _ in saved(hash: String(repeating: "0", count: 64)) },
            localFileVerifier: JarvisTemporaryFileVerifier(root: root))
        let mismatchOpened = await wrongLocal.open(id: id, versionID: version)
        check(!mismatchOpened && wrongLocal.selected == nil && mismatched.remoteReads == 0,
              "Wrong saved-copy identity fails without displaying or silently redownloading it")

        let corruptURL = root.appendingPathComponent("corrupt.mp4")
        try FileManager.default.copyItem(at: source, to: corruptURL)
        let corruptWriter = try FileHandle(forWritingTo: corruptURL)
        try corruptWriter.seek(toOffset: 0); try corruptWriter.write(contentsOf: Data([0xFF])); try corruptWriter.close()
        let corrupt = LocalPreviewFixture(source: source, root: root, media: media)
        let corruptModel = JarvisFileMediaPreviewModel(api: corrupt, localMediaLookup: { _, _ in saved(url: corruptURL) },
            localFileVerifier: JarvisTemporaryFileVerifier(root: root))
        let corruptOpened = await corruptModel.open(id: id, versionID: version)
        check(!corruptOpened && corruptModel.preview.player == nil && corrupt.remoteReads == 0,
              "Copy verifier hashes actual local bytes instead of trusting a saved receipt")
        check(FileManager.default.fileExists(atPath: corruptURL.path), "A corrupt durable source is not deleted")
        let residue = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        check(residue.allSatisfy { !$0.lastPathComponent.hasPrefix("jarvis-transfer-") },
              "Failed verification leaves no private partial viewing copy")

        let changed = LocalPreviewFixture(source: source, root: root, media: try metadata(versionID: "source/newer"))
        var changedLookups = 0
        let changedModel = JarvisFileMediaPreviewModel(api: changed, localMediaLookup: { _, _ in changedLookups += 1; return saved() })
        let changedOpened = await changedModel.open(id: id, versionID: version)
        check(!changedOpened && changedLookups == 0 && changed.remoteReads == 0,
              "A different metadata version is rejected before local lookup")
        let limited = LocalPreviewFixture(source: source, root: root, media: try metadata(limit: byteCount - 1))
        var limitedLookups = 0
        let limitedModel = JarvisFileMediaPreviewModel(api: limited, localMediaLookup: { _, _ in limitedLookups += 1; return saved() })
        let limitedOpened = await limitedModel.open(id: id, versionID: version)
        check(!limitedOpened && limitedLookups == 0 && limited.remoteReads == 0,
              "The actual transfer byte limit applies equally to local previews")

        for returnedLocal in [false, true] {
            let lateFixture = LocalPreviewFixture(source: source, root: root, media: media)
            var pending: CheckedContinuation<JarvisWorkspaceLocalFileContent?, Never>?
            let lateModel = JarvisFileMediaPreviewModel(api: lateFixture,
                localMediaLookup: { _, _ in await withCheckedContinuation { pending = $0 } },
                localFileVerifier: JarvisTemporaryFileVerifier(root: root))
            let opening = Task { await lateModel.open(id: id, versionID: version) }
            for _ in 0..<1000 where pending == nil { try await Task.sleep(nanoseconds: 1_000_000) }
            check(pending != nil, "Preview reaches held local lookup")
            if returnedLocal { lateFixture.context.invalidate() }
            else { lateModel.deactivate() }
            pending?.resume(returning: returnedLocal ? saved() : nil)
            let opened = await opening.value
            check(!opened && lateModel.preview.player == nil && lateFixture.remoteReads == 0,
                  "Retired view/account rejects late local results and cannot begin remote fallback")
            lateModel.deactivate()
        }
        check(FileManager.default.fileExists(atPath: source.path), "All preview paths preserve the original media fixture")
        print("Phone Files local preview: \(checks) checks passed; no provider, physical playback or baseline rerun")
    }
}
