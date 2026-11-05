import AVFoundation
import Combine
import Foundation

@MainActor
final class JarvisFilesViewModel: ObservableObject {
    @Published var query = ""
    @Published private(set) var files: [JarvisRetainedFile] = []
    @Published private(set) var pageFull = false
    @Published private(set) var nextResourceCursor: String?
    @Published private(set) var isLoadingMore = false
    @Published private(set) var isLoading = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var selectedFile: JarvisRetainedFile?
    @Published private(set) var content: Data?
    @Published private(set) var isLoadingContent = false
    @Published private(set) var contentError: String?
    @Published private(set) var isDeleting = false
    @Published private(set) var deletionMessage: String?
    /// Invalidates preview/export UI when the enclosing presentation is retired.
    @Published private(set) var presentationRevision = 0
    @Published private(set) var resourceContext: JarvisResourceSelectionContext?
    /// Core attaches this context beside the unchanged utterance. It is not an action request.
    var onResourceContextChange: ((JarvisResourceSelectionContext?) -> Void)?
    let thumbnails: JarvisAudiovisualThumbnailLoader
    let media: JarvisFileMediaPreviewModel?
    var onAuthorizationFailure: (() -> Void)?

    private let api: any JarvisFileAccess
    private var listGeneration = 0
    private var contentGeneration = 0
    private var active = true
    private var suspended = false
    private var resourceReferences: [String: JarvisResourceReference] = [:]
    private var selectedResource: JarvisResourceReference?
    private var resourceContextVisible = false
    private var visibleResourceIDs: Set<String> = []
    private var loadedResourceQuery: String?

    var audiovisualFiles: [JarvisRetainedFile] {
        files.filter { $0.mediaType.hasPrefix("image/") || $0.mediaType == "video/mp4" }
    }

    func resourceReference(for file: JarvisRetainedFile) -> JarvisResourceReference? {
        resourceReferences[file.id]
    }

    func setResourceContextVisible(_ visible: Bool) {
        resourceContextVisible = visible
        if !visible { selectedResource = nil; visibleResourceIDs = [] }
        publishResourceContext()
    }

    func setResourceVisible(_ fileID: String, visible: Bool) {
        if visible { visibleResourceIDs.insert(fileID) }
        else { visibleResourceIDs.remove(fileID) }
        publishResourceContext()
    }

    private func publishResourceContext() {
        let candidates = audiovisualFiles.filter { visibleResourceIDs.isEmpty || visibleResourceIDs.contains($0.id) }
        let refs = candidates.compactMap { resourceReferences[$0.id] }
        let value = active && !suspended && resourceContextVisible
            ? JarvisResourceSelectionContext(visibleResources: Array(refs.prefix(32)),
                selectedResource: selectedResource, pageFull: pageFull || refs.count > 32) : nil
        guard resourceContext != value else { return }
        resourceContext = value
        onResourceContextChange?(value)
    }

    init(api: any JarvisFileAccess, localMediaLookup: JarvisLocalMediaLookup? = nil) {
        self.api = api
        self.thumbnails = JarvisAudiovisualThumbnailLoader(api: api)
        self.media = (api as? any JarvisFilesMediaAccess).map {
            JarvisFileMediaPreviewModel(api: $0, localMediaLookup: localMediaLookup)
        }
        self.thumbnails.onAuthorizationFailure = { [weak self] in self?.onAuthorizationFailure?() }
        self.media?.onAuthorizationFailure = { [weak self] in self?.onAuthorizationFailure?() }
    }

    func reload() async {
        guard active, !suspended, !Task.isCancelled else { return }
        listGeneration += 1
        let generation = listGeneration
        files = []
        resourceReferences = [:]
        visibleResourceIDs = []
        nextResourceCursor = nil
        isLoadingMore = false
        selectedResource = nil
        thumbnails.clear()
        pageFull = false
        publishResourceContext()
        errorMessage = nil
        isLoading = true
        defer { if generation == listGeneration { isLoading = false } }
        do {
            let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
            if resourceContextVisible, let resources = api as? any JarvisResourceFileAccess {
                let query = text.isEmpty ? nil : text
                let page = try await resources.getResourceFiles(query: query, cursor: nil)
                guard generation == listGeneration, !Task.isCancelled else { return }
                let bounded = Array(page.files.prefix(8))
                files = bounded.map(\.file)
                resourceReferences = Dictionary(bounded.map { ($0.file.id, $0.reference) }, uniquingKeysWith: { first, _ in first })
                pageFull = page.pageFull || page.files.count > bounded.count
                nextResourceCursor = page.nextCursor
                loadedResourceQuery = query
            } else {
                let page = try await api.getRetainedFiles(query: text.isEmpty ? nil : text)
                guard generation == listGeneration, !Task.isCancelled else { return }
                files = page.files
                pageFull = page.pageFull
            }
            publishResourceContext()
        } catch {
            guard generation == listGeneration, !Task.isCancelled else { return }
            errorMessage = error.localizedDescription
            reportAuthorizationFailure(error)
        }
    }

    func loadMoreResources() async {
        guard active, !suspended, resourceContextVisible, !isLoading, !isLoadingMore,
              !Task.isCancelled, let cursor = nextResourceCursor,
              let resources = api as? any JarvisResourceFileAccess else { return }
        let generation = listGeneration
        isLoadingMore = true
        errorMessage = nil
        defer { if listGeneration == generation { isLoadingMore = false } }
        do {
            let page = try await resources.getResourceFiles(query: loadedResourceQuery, cursor: cursor)
            guard active, !suspended, !Task.isCancelled, listGeneration == generation else { return }
            guard page.nextCursor != cursor else { throw JarvisResourceSelectionContext.WireError.invalidContext }
            let known = Set(files.map(\.id))
            let additional = page.files.prefix(8).filter { !known.contains($0.file.id) }
            files.append(contentsOf: additional.map(\.file))
            for resource in additional { resourceReferences[resource.file.id] = resource.reference }
            nextResourceCursor = page.nextCursor
            pageFull = page.pageFull
            publishResourceContext()
        } catch {
            guard listGeneration == generation, !Task.isCancelled else { return }
            errorMessage = error.localizedDescription
            reportAuthorizationFailure(error)
        }
    }

    @discardableResult
    func open(_ file: JarvisRetainedFile) async -> JarvisFileDownload? {
        await open(id: file.id, initialFile: file)
    }

    @discardableResult
    func open(id: String) async -> JarvisFileDownload? {
        await open(id: id, initialFile: nil)
    }

    private func open(id: String, initialFile: JarvisRetainedFile?) async -> JarvisFileDownload? {
        guard active, !suspended, !Task.isCancelled else { return nil }
        contentGeneration += 1
        media?.close()
        let generation = contentGeneration
        selectedFile = initialFile
        selectedResource = nil
        publishResourceContext()
        content = nil
        contentError = nil
        isLoadingContent = true
        defer { if generation == contentGeneration { isLoadingContent = false } }
        do {
            let fresh: JarvisRetainedFile
            let reference: JarvisResourceReference?
            if resourceContextVisible, let resources = api as? any JarvisResourceFileAccess,
               let fileID = UUID(uuidString: id) {
                let resource = try await resources.getResourceFile(id: fileID,
                    versionID: resourceReferences[id]?.versionID)
                fresh = resource.file
                reference = resource.reference
            } else {
                fresh = try await api.getRetainedFile(id: id)
                reference = nil
            }
            guard UUID(uuidString: fresh.id) != nil, UUID(uuidString: fresh.id) == UUID(uuidString: id) else {
                throw JarvisAPIError(code: "file_identity_changed", message: "The requested file changed. Reload the list.")
            }
            guard generation == contentGeneration, !Task.isCancelled else { return nil }
            if fresh.mediaType == "video/mp4" {
                selectedFile = fresh
                guard let media, let fileID = UUID(uuidString: fresh.id) else {
                    throw JarvisAPIError(code: "media_transfer_unavailable", message: "Video previews are not available for this connection.")
                }
                let opened = await media.open(id: fileID, versionID: reference?.versionID)
                guard generation == contentGeneration, !Task.isCancelled else { return nil }
                if opened, let selected = media.selected {
                    guard reference == nil || selected.versionID == reference?.versionID else {
                        media.close()
                        throw JarvisAPIError(code: "file_identity_changed", message: "The selected video changed. Refresh the library.")
                    }
                    selectedFile = selected.file
                    selectedResource = reference
                    publishResourceContext()
                    if let asset = media.preview.player?.currentItem?.asset {
                        await thumbnails.loadOpenedVideo(fileID: selected.file.id, asset: asset)
                    }
                } else {
                    contentError = media.errorMessage ?? "This saved video could not be opened."
                }
                // Media is owned by the private-file player. It never returns
                // document Data or fabricates a durable saved-copy receipt.
                return nil
            }
            let bytes: Data
            if let reference, let resources = api as? any JarvisResourceFileAccess {
                bytes = try await resources.getResourceContent(file: fresh, reference: reference)
            } else {
                bytes = try await api.getRetainedFileContent(file: fresh)
            }
            guard generation == contentGeneration, !Task.isCancelled else { return nil }
            selectedFile = fresh
            content = bytes
            selectedResource = reference
            publishResourceContext()
            return JarvisFileDownload(file: fresh, data: bytes)
        } catch {
            guard generation == contentGeneration, !Task.isCancelled else { return nil }
            contentError = error.localizedDescription
            reportAuthorizationFailure(error)
            return nil
        }
    }

    func remove(_ file: JarvisRetainedFile) async -> Bool {
        guard active, !suspended, !Task.isCancelled, !isDeleting else { return false }
        let revision = presentationRevision
        isDeleting = true
        deletionMessage = nil
        selectedResource = nil
        publishResourceContext()
        closePreview()
        defer { isDeleting = false }
        do {
            let result = try await api.deleteRetainedFile(id: file.id)
            guard active, !suspended, revision == presentationRevision, !Task.isCancelled else { return false }
            deletionMessage = result.deleted
                ? "Removed from Jarvis. Storage cleanup: \(result.storageCleanup). Exported copies are not removed."
                : "Deletion is pending. Refresh to check its status."
            await reload()
            guard active, !suspended, revision == presentationRevision, !Task.isCancelled else { return false }
            return result.deleted
        } catch {
            guard active, !suspended, revision == presentationRevision, !Task.isCancelled else { return false }
            deletionMessage = "Deletion was not confirmed. Refresh before trying again. \(error.localizedDescription)"
            reportAuthorizationFailure(error)
            return false
        }
    }

    func closePreview() {
        contentGeneration += 1
        media?.close()
        selectedFile = nil
        content = nil
        contentError = nil
        isLoadingContent = false
    }

    func clear() {
        presentationRevision += 1
        listGeneration += 1
        files = []
        resourceReferences = [:]
        visibleResourceIDs = []
        nextResourceCursor = nil
        loadedResourceQuery = nil
        isLoadingMore = false
        selectedResource = nil
        thumbnails.clear()
        pageFull = false
        publishResourceContext()
        isLoading = false
        errorMessage = nil
        deletionMessage = nil
        closePreview()
    }

    func suspend() {
        suspended = true
        clear()
    }

    func resume() async {
        guard active, !Task.isCancelled else { return }
        suspended = false
        await reload()
    }

    func deactivate() {
        active = false
        query = ""
        suspend()
        media?.deactivate()
    }

    private func reportAuthorizationFailure(_ error: Error) {
        let status = (error as? JarvisAPIError)?.statusCode
        if status == 401 || status == 403 { onAuthorizationFailure?() }
    }
}
