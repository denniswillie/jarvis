import AVFoundation
import Combine
import Foundation

/// Owns a temporary preview and its player. The enclosing account/presentation
/// owner must call deactivate on retirement; an already-held AVPlayer URL cannot
/// be revoked merely by rejecting subsequent reads of the temporary file.
@MainActor
final class JarvisMediaPreviewViewModel: ObservableObject {
    @Published private(set) var player: AVPlayer?
    @Published private(set) var isLoading = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var filename: String?

    private var active = true
    private var generation = 0
    private var file: JarvisVerifiedTemporaryFile?
    private var loadingAsset: AVURLAsset?

    /// Consumes the temporary file on both success and failure. Preparing a
    /// player does not auto-play, emit a display receipt, or save an offline copy.
    @discardableResult
    func open(file incoming: JarvisVerifiedTemporaryFile, filename: String,
              maximumByteCount: Int, maximumDurationMilliseconds: Int) async -> Bool {
        guard active, !Task.isCancelled else { incoming.discard(); return false }
        close()
        let attempt = generation
        file = incoming
        isLoading = true
        do {
            guard incoming.byteCount > 0, incoming.byteCount <= maximumByteCount,
                  maximumDurationMilliseconds > 0 else {
                throw unavailable()
            }
            let asset = AVURLAsset(url: try incoming.urlForReading())
            loadingAsset = asset
            let playable = try await asset.load(.isPlayable)
            let duration = try await asset.load(.duration)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            for track in tracks {
                let formats = try await track.load(.formatDescriptions)
                guard !formats.isEmpty, formats.allSatisfy({ CMFormatDescriptionGetMediaSubType($0) == kCMVideoCodecType_H264 }) else {
                    throw unavailable()
                }
            }
            for track in try await asset.loadTracks(withMediaType: .audio) {
                let formats = try await track.load(.formatDescriptions)
                guard !formats.isEmpty, formats.allSatisfy({ CMFormatDescriptionGetMediaSubType($0) == kAudioFormatMPEG4AAC }) else {
                    throw unavailable()
                }
            }
            guard active, generation == attempt else { incoming.discard(); return false }
            try Task.checkCancellation()
            _ = try incoming.urlForReading()
            let milliseconds = CMTimeGetSeconds(duration) * 1000
            guard playable, !tracks.isEmpty, milliseconds.isFinite, milliseconds > 0,
                  milliseconds <= Double(maximumDurationMilliseconds) else { throw unavailable() }
            let preview = AVPlayer(playerItem: AVPlayerItem(asset: asset))
            preview.actionAtItemEnd = .pause
            player = preview
            self.filename = filename
            loadingAsset = nil
            isLoading = false
            return true
        } catch {
            incoming.discard()
            guard generation == attempt else { return false }
            close()
            if !(error is CancellationError), !Task.isCancelled { errorMessage = error.localizedDescription }
            return false
        }
    }

    func close() {
        generation += 1
        loadingAsset?.cancelLoading()
        loadingAsset = nil
        player?.pause()
        player?.replaceCurrentItem(with: nil)
        player = nil
        file?.discard()
        file = nil
        filename = nil
        errorMessage = nil
        isLoading = false
    }

    func pause() { player?.pause() }

    func deactivate() {
        active = false
        close()
    }

    isolated deinit {
        loadingAsset?.cancelLoading()
        player?.pause()
        player?.replaceCurrentItem(with: nil)
        file?.discard()
    }

    private func unavailable() -> JarvisWorkspaceError {
        JarvisWorkspaceError(code: "video_preview_unavailable", message: "This saved video could not be opened.")
    }
}
