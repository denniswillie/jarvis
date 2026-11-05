import Foundation
import CoreGraphics
import ImageIO

@MainActor
private final class ResourceFixture: JarvisResourceFileAccess, JarvisThumbnailAccess {
    var resources: [JarvisResourceFile]
    var metadataVersions: [String?] = []
    var contentVersions: [String] = []
    var fallbackReads = 0
    var previewData = Data(), thumbnailReads = 0
    var holdThumbnail = false
    var pendingThumbnail: CheckedContinuation<Void, Never>?
    func getThumbnail(file: JarvisRetainedFile, reference: JarvisResourceReference,
                      access: JarvisWorkspaceLocalAccess) async throws -> Data {
        thumbnailReads += 1
        if holdThumbnail { await withCheckedContinuation { pendingThumbnail = $0 } }
        return previewData
    }
    var firstPage: JarvisResourceFilePage?
    var pages: [String: JarvisResourceFilePage] = [:]
    var pageReads: [(String?, String?)] = []
    var holdContent = false
    var pending: CheckedContinuation<Void, Never>?
    init(_ resources: [JarvisResourceFile]) { self.resources = resources }
    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage {
        .init(files: resources.map(\.file), pageFull: true)
    }
    func getResourceFiles(query: String?) async throws -> JarvisResourceFilePage {
        .init(files: resources, pageFull: true)
    }
    func getResourceFiles(query: String?, cursor: String?) async throws -> JarvisResourceFilePage {
        pageReads.append((query, cursor))
        if let cursor { return pages[cursor]! }
        return firstPage ?? JarvisResourceFilePage(files: resources, pageFull: true)
    }
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        resources.first { $0.file.id == id }!.file
    }
    func getResourceFile(id: UUID, versionID: String?) async throws -> JarvisResourceFile {
        metadataVersions.append(versionID)
        let resource = resources.first { $0.reference.fileID == id }!
        if let versionID, resource.reference.versionID != versionID {
            throw JarvisAPIError(code: "file_identity_changed", message: "Version changed")
        }
        return resource
    }
    func getResourceContent(file: JarvisRetainedFile, reference: JarvisResourceReference) async throws -> Data {
        contentVersions.append(reference.versionID)
        if holdContent { await withCheckedContinuation { pending = $0 } }
        return Data([1, 2, 3])
    }
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        fallbackReads += 1
        return Data([1, 2, 3])
    }
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        .init(deleted: true, storageCleanup: "deleted", fullErasureVerified: false)
    }
}

@main
enum AudiovisualResourceChecks {
    @MainActor static func main() async throws {
        var count = 0
        func check(_ condition: Bool, _ label: String) {
            precondition(condition, label); count += 1
        }
        let idA = "12345678-0000-4000-8000-000000000001"
        let idB = "12345678-0000-4000-8000-000000000002"
        func encoded(id: String, type: String = "image/jpeg", version: String = "document/v1", explicit: String? = nil, size: Int = 3) throws -> Data {
            var value: [String: Any] = ["id": id, "filename": "Same name.jpg", "content_type": type,
                "size_bytes": size, "description": "Captured image", "source": "visual_frame", "created_at": "2026-11-02T19:00:00Z",
                "memory_source": ["source_id": id, "revision": 1, "generation": 1,
                    "document_id": version, "content_digest": "source-digest", "source_type": "file", "trust": "user_provided",
                    "time_basis": "recorded", "recorded_at": "2026-11-02T19:00:00Z", "is_current": true, "ingestion_status": "ready"]]
            if let explicit { value["version_id"] = explicit }
            return try JSONSerialization.data(withJSONObject: value)
        }
        func resource(id: String, type: String = "image/jpeg", version: String = "document/v1") throws -> JarvisResourceFile {
            try JSONDecoder().decode(JarvisResourceFile.self, from: encoded(id: id, type: type, version: version))
        }
        let first = try resource(id: idA), second = try resource(id: idB)
        check(first.reference.versionID == "document/v1", "Canonical source document ID is the resource version")
        do {
            _ = try JSONDecoder().decode(JarvisResourceFile.self, from: encoded(id: idA, explicit: "different-version"))
            preconditionFailure("Conflicting canonical versions accepted")
        } catch { count += 1 }
        let document = try resource(id: UUID().uuidString, type: "text/plain")
        let api = ResourceFixture([first, second, document])
        let model = JarvisFilesViewModel(api: api)
        var updates: [JarvisResourceSelectionContext?] = []
        model.onResourceContextChange = { updates.append($0) }
        model.setResourceContextVisible(true)
        await model.reload()
        check(model.audiovisualFiles.count == 2, "Gallery includes images and excludes unrelated documents")
        let context = model.resourceContext!
        check(context.visibleResources.count == 2 && context.selectedResource == nil, "Browsing does not auto-select a resource")
        check(context.visibleResources[0].displayID == context.visibleResources[1].displayID,
            "Short-ID collision fixture is actually ambiguous")
        check(context.visibleResources[0].fileID != context.visibleResources[1].fileID,
            "Ambiguous names and aliases retain both distinct canonical IDs for clarification")
        let opened = await model.open(first.file)
        check(opened != nil && model.resourceContext?.selectedResource == first.reference,
            "Authenticated opened item establishes exact selected context")
        check(api.metadataVersions.last! == "document/v1" && api.contentVersions.last == "document/v1" && api.fallbackReads == 0,
            "Image metadata and bytes use the selected exact version")
        model.closePreview()
        check(model.resourceContext?.selectedResource == first.reference,
            "Closing preview retains explicit selection while the gallery remains visible")
        model.setResourceContextVisible(false)
        check(model.resourceContext == nil && updates.last! == nil, "Leaving gallery clears worker-facing resource context")
        model.setResourceContextVisible(true)
        api.resources[0] = try resource(id: idA, version: "document/v2")
        _ = await model.open(first.file)
        check(model.contentError != nil && model.resourceContext?.selectedResource == nil,
            "A changed canonical version fails instead of silently retargeting")
        await model.reload()
        api.holdContent = true
        let opening = Task { await model.open(api.resources[0].file) }
        while api.pending == nil { await Task.yield() }
        model.suspend()
        api.pending?.resume(); api.pending = nil
        _ = await opening.value
        check(model.resourceContext == nil && model.content == nil && model.selectedFile == nil,
            "Late bytes after suspension cannot resurrect content or context")
        api.holdContent = false
        model.setResourceContextVisible(true)
        await model.resume()
        model.deactivate()
        check(model.resourceContext == nil && model.files.isEmpty && model.thumbnails.images.isEmpty,
            "Account retirement clears context and thumbnail cache")

        let many = try (0..<40).map { _ in try resource(id: UUID().uuidString) }
        let pagedAPI = ResourceFixture(many)
        pagedAPI.firstPage = .init(files: Array(many.prefix(8)), pageFull: true, nextCursor: "opaque-1")
        for index in 1...4 {
            pagedAPI.pages["opaque-\(index)"] = .init(files: Array(many[(index * 8)..<(index * 8 + 8)]),
                pageFull: index < 4, nextCursor: index < 4 ? "opaque-\(index + 1)" : nil)
        }
        let paged = JarvisFilesViewModel(api: pagedAPI)
        paged.query = "camera"; paged.setResourceContextVisible(true)
        await paged.reload()
        paged.query = "unsent draft"
        for _ in 1...4 { await paged.loadMoreResources() }
        check(paged.files.count == 40 && paged.nextResourceCursor == nil, "Opaque pages reach all media and stop at nil cursor")
        check(pagedAPI.pageReads.allSatisfy { $0.0 == "camera" }, "More pages retain the submitted query until reload")
        check(paged.resourceContext?.visibleResources.count == 32 && paged.resourceContext?.pageFull == true,
            "Large lists expose at most32 references and disclose omitted context")
        paged.setResourceVisible(many[39].file.id, visible: true)
        _ = await paged.open(many[0].file)
        check(paged.resourceContext?.visibleResources == [many[39].reference]
              && paged.resourceContext?.selectedResource == many[0].reference,
            "Viewport references and explicit selection remain distinct exact identities")
        await paged.reload()
        check(paged.files.count == 8 && pagedAPI.pageReads.last!.0 == "unsent draft"
              && pagedAPI.pageReads.last!.1 == nil, "New search resets pagination and selection")
        paged.deactivate()
        check(paged.nextResourceCursor == nil && paged.resourceContext == nil, "Retirement clears pagination and context")
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let size = try source.resourceValues(forKeys: [.fileSizeKey]).fileSize!
        let video = try JSONDecoder().decode(JarvisResourceFile.self,
            from: encoded(id: UUID().uuidString, type: "video/mp4", size: size))
        let videoAPI = ResourceFixture([video])
        let cloudThumbnails = JarvisAudiovisualThumbnailLoader(api: videoAPI)
        await cloudThumbnails.load(video.file, reference: video.reference)
        check(videoAPI.fallbackReads == 0 && cloudThumbnails.images.isEmpty,
            "Browsing a cloud-only video does not download its original for a thumbnail")
        let provider = CGDataProvider(data: Data(repeating: 120, count: 12) as CFData)!
        let previewImage = CGImage(width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: 6,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let jpeg = NSMutableData()
        let destination = CGImageDestinationCreateWithData(jpeg, "public.jpeg" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, previewImage, nil)
        check(CGImageDestinationFinalize(destination), "Synthetic thumbnail encodes")
        videoAPI.previewData = jpeg as Data
        var previewFile = video.file
        previewFile.thumbnail = JarvisThumbnailReference(path: "/files/" + video.reference.fileID.uuidString.lowercased() + "/thumbnail",
            versionID: video.reference.versionID, contentType: "image/jpeg", contentSHA256: String(repeating: "a", count: 64),
            sizeBytes: jpeg.length, width: 2, height: 2)
        let previews = JarvisAudiovisualThumbnailLoader(api: videoAPI)
        await previews.load(previewFile, reference: video.reference)
        check(previews.images[video.file.id] != nil && videoAPI.thumbnailReads == 1 && videoAPI.fallbackReads == 0,
            "Gallery decodes only the small preview without opening the original")
        check(FileManager.default.fileExists(atPath: source.path), "Existing durable original remains untouched")
        previews.clear()
        check(previews.images.isEmpty, "Thumbnail pixels retire with their account/view lifetime")
        videoAPI.holdThumbnail = true
        let loading = Task { await previews.load(previewFile, reference: video.reference) }
        while videoAPI.pendingThumbnail == nil { await Task.yield() }
        previews.clear()
        videoAPI.pendingThumbnail?.resume()
        await loading.value
        check(previews.messages.isEmpty && previews.images.isEmpty && previews.loading.isEmpty,
            "A late preview response cannot repopulate a retired account/view")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ResourceURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel(); ResourceURLProtocol.handler = nil }
        let client = JarvisAPIClient(configuration: .init(backendBaseURL: URL(string: "https://audiovisual.invalid")!,
            staticBearerToken: "fixture-only"), session: session)
        var responseVersion = first.reference.versionID
        ResourceURLProtocol.handler = { request in
            let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!
            check(query.first { $0.name == "version_id" }?.value == first.reference.versionID,
                "Exact image version is sent to the existing Files endpoint")
            check(query.first { $0.name == "session_id" }?.value == "owned-session",
                "Exact image read retains the owned session")
            return (Data([1, 2, 3]), ["Content-Type": "image/jpeg", "X-Jarvis-Version-Id": responseVersion])
        }
        let image = try await client.getRetainedFileContent(file: first.file, sessionID: "owned-session",
            versionID: first.reference.versionID)
        check(image == Data([1, 2, 3]), "Exact-version image response remains usable")
        responseVersion = "changed-version"
        do {
            _ = try await client.getRetainedFileContent(file: first.file, sessionID: "owned-session",
                versionID: first.reference.versionID)
            preconditionFailure("Wrong image response version accepted")
        } catch { count += 1 }
        print("PASS: \(count) audiovisual resource checks")
    }
}

private final class ResourceURLProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (Data, [String: String]))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (bytes, headers) = try Self.handler!(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: headers)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: bytes)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}
