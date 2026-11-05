import Combine
import Foundation

/// Owns a library video preview and its separately prepared export.
/// Browsing/exporting a retained file has no presentation request to acknowledge.
@MainActor
final class JarvisFileMediaPreviewModel: ObservableObject {
    @Published private(set) var selected: JarvisMediaFile?
    @Published private(set) var isLoading = false
    @Published private(set) var downloadProgress: Double?
    @Published private(set) var errorMessage: String?
    @Published private(set) var isPreparingExport = false
    @Published private(set) var preparedExport: JarvisMediaExport?
    @Published private(set) var exportMessage: String?
    let preview = JarvisMediaPreviewViewModel()
    var onAuthorizationFailure: (() -> Void)?

    private let api: any JarvisFilesMediaAccess
    private let localMediaLookup: JarvisLocalMediaLookup?
    private let localFileVerifier: JarvisTemporaryFileVerifier
    private var active = true
    private var generation = 0
    private var previewAccess = JarvisWorkspaceLocalAccess()
    private var exportAccess: JarvisWorkspaceLocalAccess?
    private var exportAttempt: UUID?
    private let maximumByteCount = JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumByteCount
    private let maximumDurationMilliseconds = JarvisWorkspaceRequest.MediaTransferPolicy.capture.maximumDurationMilliseconds

    init(api: any JarvisFilesMediaAccess, localMediaLookup: JarvisLocalMediaLookup? = nil,
         localFileVerifier: JarvisTemporaryFileVerifier = JarvisTemporaryFileVerifier()) {
        self.api = api
        self.localMediaLookup = localMediaLookup
        self.localFileVerifier = localFileVerifier
    }

    @discardableResult
    func open(id: UUID, versionID: String? = nil) async -> Bool {
        guard active, !Task.isCancelled else { return false }
        close()
        let attempt = generation, access = previewAccess
        isLoading = true
        defer { if generation == attempt { isLoading = false } }
        do {
            let media = try await api.getMediaFile(id: id, versionID: versionID, access: access)
            try requireCurrent(attempt, access: access)
            let limit = try media.validate(fileID: id, versionID: versionID ?? media.versionID,
                maximumByteCount: maximumByteCount, maximumDurationMilliseconds: maximumDurationMilliseconds)
            let file = try await viewingCopy(media, fileID: id, maximumByteCount: limit,
                attempt: attempt, access: access)
            do {
                try requireCurrent(attempt, access: access)
                try requireExactFile(file, media: media)
            } catch {
                file.discard()
                throw error
            }
            // The player model consumes the file on success and failure.
            guard await preview.open(file: file, filename: media.file.exportFilename,
                maximumByteCount: limit, maximumDurationMilliseconds: min(maximumDurationMilliseconds,
                    media.transferPolicy?.maximumDurationMilliseconds ?? 300_000)) else {
                try requireCurrent(attempt, access: access)
                throw JarvisAPIError(code: "video_preview_unavailable",
                    message: preview.errorMessage ?? "This saved video could not be opened.")
            }
            try requireCurrent(attempt, access: access)
            selected = media
            return true
        } catch {
            guard generation == attempt else { return false }
            preview.close()
            if !(error is CancellationError), !Task.isCancelled { errorMessage = error.localizedDescription }
            reportAuthorizationFailure(error)
            return false
        }
    }

    @discardableResult
    func prepareExport() async -> Bool {
        guard active, let selected, !isLoading, !isPreparingExport, !Task.isCancelled,
              let id = UUID(uuidString: selected.file.id) else { return false }
        cancelExport()
        let attempt = generation, operation = UUID(), access = JarvisWorkspaceLocalAccess()
        exportAccess = access
        exportAttempt = operation
        isPreparingExport = true
        exportMessage = nil
        defer {
            if exportAttempt == operation { isPreparingExport = false }
        }
        do {
            // Recheck source access, retaining the displayed exact version. A
            // different current head is never silently substituted for export.
            let fresh = try await api.getMediaFile(id: id, versionID: selected.versionID, access: access)
            try requireCurrent(attempt, access: access)
            try fresh.validate(fileID: id, versionID: selected.versionID,
                maximumByteCount: maximumByteCount, maximumDurationMilliseconds: maximumDurationMilliseconds)
            guard exportAttempt == operation, fresh.versionID == selected.versionID,
                  fresh.contentSHA256 == selected.contentSHA256,
                  fresh.file.sizeBytes == selected.file.sizeBytes else { throw changedFile() }
            let limit = min(maximumByteCount, fresh.transferLimitBytes)
            let file = try await viewingCopy(fresh, fileID: id, maximumByteCount: limit,
                attempt: attempt, access: access)
            do {
                try requireCurrent(attempt, access: access)
                guard exportAttempt == operation else { throw CancellationError() }
                try requireExactFile(file, media: selected)
            } catch {
                file.discard()
                throw error
            }
            preparedExport = JarvisMediaExport(filename: selected.file.exportFilename, file: file)
            return true
        } catch {
            guard generation == attempt, exportAttempt == operation else { return false }
            cancelExport()
            if !(error is CancellationError), !Task.isCancelled {
                exportMessage = "Export could not be prepared: " + error.localizedDescription
            }
            reportAuthorizationFailure(error)
            return false
        }
    }

    /// Called only from the matching native picker's actual completion. Merely
    /// showing the picker or preparing bytes is not a successful export.
    func finishExport(id: UUID, completed: Bool, failureDescription: String? = nil) {
        guard active, preparedExport?.id == id, let exportAccess else { return }
        do {
            try api.checkContext()
            try exportAccess.check()
        } catch {
            cancelExport()
            return
        }
        cancelExport()
        exportMessage = completed ? "Exported a separate copy."
            : failureDescription.map { "Export was not completed: " + $0 }
    }

    func cancelExport() {
        exportAttempt = nil
        exportAccess?.invalidate()
        exportAccess = nil
        preparedExport?.file.discard()
        preparedExport = nil
        isPreparingExport = false
    }

    func close() {
        generation += 1
        previewAccess.invalidate()
        preview.close()
        previewAccess = JarvisWorkspaceLocalAccess()
        cancelExport()
        selected = nil
        downloadProgress = nil
        errorMessage = nil
        exportMessage = nil
        isLoading = false
    }

    func deactivate() {
        active = false
        close()
        previewAccess.invalidate()
    }

    isolated deinit {
        previewAccess.invalidate()
        exportAccess?.invalidate()
        preparedExport?.file.discard()
        preview.deactivate()
    }

    private func requireCurrent(_ attempt: Int, access: JarvisWorkspaceLocalAccess) throws {
        try Task.checkCancellation()
        try api.checkContext()
        try access.check()
        guard active, generation == attempt else { throw CancellationError() }
    }

    private func viewingCopy(_ media: JarvisMediaFile, fileID: UUID, maximumByteCount: Int,
                             attempt: Int, access: JarvisWorkspaceLocalAccess) async throws -> JarvisVerifiedTemporaryFile {
        if let localMediaLookup {
            let reference = JarvisResourceReference(fileID: fileID, versionID: media.versionID,
                filename: media.file.filename, mediaType: media.file.mediaType)
            let local = try await localMediaLookup(reference, access)
            // A stale nil lookup must not fall through into a new remote fetch.
            try requireCurrent(attempt, access: access)
            if let local {
                guard local.url.isFileURL, local.contentType == media.file.mediaType,
                      local.byteCount == media.file.sizeBytes,
                      local.contentSHA256 == media.contentSHA256 else { throw changedFile() }
                // The journal lends its durable original. Only this separately
                // hashed private copy is owned and later discarded by the player.
                return try await localFileVerifier.copyVerifiedFile(from: local.url,
                    expectedByteCount: media.file.sizeBytes, expectedSHA256: media.contentSHA256,
                    maximumByteCount: maximumByteCount, access: access, format: .mp4)
            }
        }
        if let progressive = api as? any JarvisMediaProgressAccess {
            return try await progressive.getMediaContent(media, access: access) { [weak self] fraction in
                Task { @MainActor [weak self] in
                    guard let self, self.generation == attempt, self.isLoading,
                          (try? access.check()) != nil else { return }
                    self.downloadProgress = max(self.downloadProgress ?? 0, min(1, max(0, fraction)))
                }
            }
        }
        return try await api.getMediaContent(media, access: access)
    }

    private func requireExactFile(_ file: JarvisVerifiedTemporaryFile, media: JarvisMediaFile) throws {
        guard file.byteCount == media.file.sizeBytes, file.contentSHA256 == media.contentSHA256 else {
            throw changedFile()
        }
        _ = try file.urlForReading()
    }

    private func changedFile() -> JarvisAPIError {
        JarvisAPIError(code: "file_identity_changed", message: "The selected video changed. Reopen it before exporting.")
    }

    private func reportAuthorizationFailure(_ error: Error) {
        let status = (error as? JarvisAPIError)?.statusCode
        if status == 401 || status == 403 { onAuthorizationFailure?() }
    }
}
