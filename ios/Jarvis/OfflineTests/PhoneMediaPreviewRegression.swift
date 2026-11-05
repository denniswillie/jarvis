import AVFoundation
import CryptoKit
import Foundation

@main
enum PhoneMediaPreviewRegression {
    @MainActor
    static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-media-preview-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let reader = try FileHandle(forReadingFrom: source)
        var hash = SHA256(), count = 0
        while let chunk = try reader.read(upToCount: 65_536), !chunk.isEmpty { hash.update(data: chunk); count += chunk.count }
        try reader.close()
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let verifier = JarvisTemporaryFileVerifier(root: root)
        let access = JarvisWorkspaceLocalAccess()
        let temporary = try await verifier.copyVerifiedFile(from: source, expectedByteCount: count,
            expectedSHA256: digest, maximumByteCount: 32 * 1024 * 1024, access: access, format: .mp4)
        let location = try temporary.urlForReading()
        let preview = JarvisMediaPreviewViewModel()
        let opened = await preview.open(file: temporary, filename: "Synthetic clip.mp4", maximumByteCount: 32 * 1024 * 1024,
                                        maximumDurationMilliseconds: 30_000)
        if !opened {
            FileHandle.standardError.write(Data("Preview failure: \(preview.errorMessage ?? "no error")\n".utf8))
            for diagnosticURL in [source, location] where FileManager.default.fileExists(atPath: diagnosticURL.path) {
                let asset = AVURLAsset(url: diagnosticURL)
                do {
                    let playable = try await asset.load(.isPlayable)
                    let seconds = CMTimeGetSeconds(try await asset.load(.duration))
                    let videos = try await asset.loadTracks(withMediaType: .video).count
                    FileHandle.standardError.write(Data("Asset \(diagnosticURL.lastPathComponent): playable=\(playable), seconds=\(seconds), videos=\(videos)\n".utf8))
                } catch { FileHandle.standardError.write(Data("Asset \(diagnosticURL.lastPathComponent): \(error)\n".utf8)) }
            }
        }
        check(opened && preview.player?.currentItem != nil && preview.errorMessage == nil,
              "A real generated H264 MP4 prepares a local AVPlayer item")
        check(preview.player?.rate == 0 && !preview.isLoading && preview.filename == "Synthetic clip.mp4",
              "Preparation never starts playback")
        let heldPlayer = preview.player
        access.invalidate()
        preview.deactivate()
        check(preview.player == nil && heldPlayer?.currentItem == nil && heldPlayer?.rate == 0,
              "Account owner retirement releases even an externally held player's item")
        check(!FileManager.default.fileExists(atPath: location.path) && FileManager.default.fileExists(atPath: source.path),
              "Retirement removes only the temporary copy")
        let another = try await verifier.copyVerifiedFile(from: source, expectedByteCount: count,
            expectedSHA256: digest, maximumByteCount: 32 * 1024 * 1024, access: JarvisWorkspaceLocalAccess(), format: .mp4)
        let anotherLocation = try another.urlForReading()
        let reopened = await preview.open(file: another, filename: "Synthetic clip.mp4", maximumByteCount: 32 * 1024 * 1024,
                                         maximumDurationMilliseconds: 30_000)
        check(!reopened && !FileManager.default.fileExists(atPath: anotherLocation.path),
              "A retired preview cannot adopt a later file")
        let invalid = Data("This is not an MP4, despite matching an advertised hash.".utf8)
        let invalidURL = root.appendingPathComponent("invalid.mp4")
        try invalid.write(to: invalidURL)
        let invalidHash = SHA256.hash(data: invalid).map { String(format: "%02x", $0) }.joined()
        let invalidCopy = try await verifier.copyVerifiedFile(from: invalidURL, expectedByteCount: invalid.count,
            expectedSHA256: invalidHash, maximumByteCount: 32 * 1024 * 1024, access: JarvisWorkspaceLocalAccess(), format: .mp4)
        let invalidLocation = try invalidCopy.urlForReading()
        let rejected = JarvisMediaPreviewViewModel()
        let accepted = await rejected.open(file: invalidCopy, filename: "Synthetic clip.mp4",
                                          maximumByteCount: 32 * 1024 * 1024, maximumDurationMilliseconds: 30_000)
        check(!accepted && rejected.player == nil && rejected.errorMessage != nil,
              "Matching hashes and labels cannot make invalid MP4 bytes playable")
        check(!FileManager.default.fileExists(atPath: invalidLocation.path), "Invalid media preparation cleans its temporary file")
        let expired = JarvisWorkspaceLocalAccess()
        let expiredCopy = try await verifier.copyVerifiedFile(from: source, expectedByteCount: count,
            expectedSHA256: digest, maximumByteCount: 32 * 1024 * 1024, access: expired, format: .mp4)
        let expiredLocation = try expiredCopy.urlForReading()
        expired.invalidate()
        let locked = JarvisMediaPreviewViewModel()
        let unlocked = await locked.open(file: expiredCopy, filename: "Synthetic clip.mp4", maximumByteCount: 32 * 1024 * 1024,
                                        maximumDurationMilliseconds: 30_000)
        check(!unlocked && locked.player == nil && !FileManager.default.fileExists(atPath: expiredLocation.path),
              "A file from retired access is never passed to AVPlayer")
        let releasedCopy = try await verifier.copyVerifiedFile(from: source, expectedByteCount: count,
            expectedSHA256: digest, maximumByteCount: 32 * 1024 * 1024, access: JarvisWorkspaceLocalAccess(), format: .mp4)
        let releasedLocation = try releasedCopy.urlForReading()
        var owner: JarvisMediaPreviewViewModel? = JarvisMediaPreviewViewModel()
        let ownerOpened = await owner!.open(file: releasedCopy, filename: "Synthetic clip.mp4", maximumByteCount: 32 * 1024 * 1024,
                                           maximumDurationMilliseconds: 30_000)
        check(ownerOpened, "Second independent preview is prepared")
        let retainedPlayer = owner!.player
        owner = nil
        check(retainedPlayer?.currentItem == nil && !FileManager.default.fileExists(atPath: releasedLocation.path),
              "Owner deinitialization releases the externally held player and temporary bytes")
        print("Phone media preview regressions passed: \(checks) checks; synthetic_mp4_bytes=\(count); no play() or audio")
    }
}
