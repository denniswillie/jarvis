import AVFoundation
import Combine
import CoreGraphics
import Foundation
import ImageIO

/// The app supplies its existing account-scoped saved-copy lookup, matching both
/// file and version before calling offlineFile. The URL is borrowed: this loader
/// never deletes a durable copy. Nil means cloud-only; no historical download.
typealias JarvisLocalMediaLookup = @MainActor (JarvisResourceReference, JarvisWorkspaceLocalAccess) async throws -> JarvisWorkspaceLocalFileContent?

@MainActor
final class JarvisAudiovisualThumbnailLoader: ObservableObject {
    @Published private(set) var images: [String: CGImage] = [:]
    @Published private(set) var messages: [String: String] = [:]
    @Published private(set) var loading: Set<String> = []
    var onAuthorizationFailure: (() -> Void)?

    private let api: any JarvisFileAccess
    private var generation = 0
    private var access = JarvisWorkspaceLocalAccess()

    init(api: any JarvisFileAccess) {
        self.api = api
    }

    func load(_ file: JarvisRetainedFile, reference: JarvisResourceReference?) async {
        let key = file.id
        guard images[key] == nil, messages[key] == nil, !loading.contains(key), !Task.isCancelled else { return }
        let attempt = generation, access = access
        loading.insert(key)
        defer { if attempt == generation { loading.remove(key) } }
        do {
            guard let reference, let descriptor = file.thumbnail,
                  let previews = api as? any JarvisThumbnailAccess else {
                messages[key] = "Preview not available yet"
                return
            }
            try descriptor.validate(reference: reference)
            let bytes = try await previews.getThumbnail(file: file, reference: reference, access: access)
            try current(attempt, access: access)
            guard bytes.count == descriptor.sizeBytes,
                  let source = CGImageSourceCreateWithData(bytes as CFData, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                  properties[kCGImagePropertyPixelWidth] as? Int == descriptor.width,
                  properties[kCGImagePropertyPixelHeight] as? Int == descriptor.height,
                  let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: 480,
                    kCGImageSourceShouldCacheImmediately: true
                  ] as CFDictionary) else { throw unavailable() }
            try current(attempt, access: access)
            if images.count >= 96, let oldest = images.keys.sorted().first { images[oldest] = nil }
            images[key] = thumbnail
        } catch {
            guard generation == attempt, !Task.isCancelled, !(error is CancellationError) else { return }
            messages[key] = "Thumbnail unavailable"
            let status = (error as? JarvisAPIError)?.statusCode
            if status == 401 || status == 403 { onAuthorizationFailure?() }
        }
    }

    /// Reuses the already verified player asset; no second media download.
    func loadOpenedVideo(fileID: String, asset: AVAsset) async {
        let attempt = generation, access = access
        do {
            let image = try await Self.frame(from: asset)
            try current(attempt, access: access)
            images[fileID] = image
            messages[fileID] = nil
        } catch { /* Playback remains usable when a thumbnail frame is unavailable. */ }
    }

    func clear() {
        generation += 1
        access.invalidate()
        access = JarvisWorkspaceLocalAccess()
        images = [:]
        messages = [:]
        loading = []
    }

    private func current(_ attempt: Int, access: JarvisWorkspaceLocalAccess) throws {
        try Task.checkCancellation()
        try access.check()
        guard generation == attempt else { throw CancellationError() }
        try (api as? any JarvisFilesMediaAccess)?.checkContext()
    }

    private static func frame(from asset: AVAsset) async throws -> CGImage {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 480, height: 480)
        return try await generator.image(at: .zero).image
    }

    private func unavailable() -> JarvisAPIError {
        JarvisAPIError(code: "thumbnail_unavailable", message: "This thumbnail could not be opened.")
    }
}
