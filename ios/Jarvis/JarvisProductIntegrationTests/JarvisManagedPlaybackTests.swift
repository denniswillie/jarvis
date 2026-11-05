import AVKit
import CryptoKit
import Foundation
import SwiftUI
import UIKit
import XCTest
@testable import Jarvis

/// Source-only candidate. This composes the real file preview with a local,
/// exact-output adapter. It does not qualify HTTP delivery, account access or
/// heard audio. A separately allocated UI runner must perform every control.
@MainActor
final class JarvisManagedPlaybackTests: XCTestCase {
    private var window: UIWindow?
    private var previous: UIWindow?
    private var deadline = Date.distantPast
    private var observations: [[String: Any]] = []

    func testPlayPauseCloseExactAcceptedOutput() async throws {
#if !targetEnvironment(simulator)
        throw PlaybackFailure("Simulator-only composition")
#else
        guard let path = ProcessInfo.processInfo.environment["JARVIS_MANAGED_PLAYBACK_MANIFEST"] else {
            throw XCTSkip("Requires an explicitly allocated exact-output playback manifest")
        }
        continueAfterFailure = false
        executionTimeAllowance = 90
        deadline = Date().addingTimeInterval(90)
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--jarvis-replay-manifest"), i + 1 < args.count,
              args[i + 1] == "/dev/null", !args.contains("--jarvis-replay-run") else {
            throw PlaybackFailure("Safe DEBUG launch required; this case must not start voice")
        }
        let manifest = try JSONDecoder().decode(PlaybackManifest.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        try manifest.validate()
        let folder = URL(fileURLWithPath: manifest.directory, isDirectory: true)
        let resultURL = folder.appendingPathComponent("native-host-result.json")
        guard !FileManager.default.fileExists(atPath: resultURL.path) else {
            throw PlaybackFailure("Never overwrite prior playback evidence")
        }
        let metadataData = try boundedData(manifest.metadata_path, maximum: 65_536)
        guard digest(metadataData) == manifest.metadata_sha256 else { throw PlaybackFailure("Metadata changed") }
        let metadata = try JSONDecoder().decode(PlaybackHistoricalMediaResponse.self, from: metadataData).file
        let id = try XCTUnwrap(UUID(uuidString: manifest.file_id))
        _ = try metadata.validate(fileID: id, versionID: manifest.version_id,
            maximumByteCount: 64 * 1024 * 1024, maximumDurationMilliseconds: 60_000)
        guard metadata.contentSHA256 == manifest.content_sha256, metadata.file.sizeBytes == manifest.size_bytes else {
            throw PlaybackFailure("Presentation metadata does not match the accepted output")
        }
        try manifest.checkAcceptedReport()
        let adapter = PlaybackAcceptedFileAdapter(manifest: manifest, media: metadata)
        let model = JarvisFilesViewModel(api: adapter)
        let media = try XCTUnwrap(model.media)
        var succeeded = false
        defer {
            // Failure cleanup is deliberately after every success assertion.
            model.deactivate()
            window?.isHidden = true
            window?.rootViewController = nil
            window = nil
            previous?.makeKey()
            previous = nil
            let result: [String: Any] = ["run_id": manifest.run_id, "host_assertions_passed": succeeded,
                "accepted_file_id": manifest.file_id, "version_id": manifest.version_id,
                "content_sha256": manifest.content_sha256, "report_sha256": manifest.report_sha256,
                "observations": observations,
                "temporary_file_absent_after_teardown": adapter.temporaryURL.map { !FileManager.default.fileExists(atPath: $0.path) } ?? false,
                "visual_review_required": true, "heard_audio_proven": false,
                "scope": "Existing product preview, exact accepted local bytes, native controls; no HTTP/account qualification"]
            do { try writeJSON(result, to: resultURL) } catch { XCTFail("Cannot retain host result: \(error)") }
        }
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        previous = scene.windows.first(where: \.isKeyWindow)
        let mounted = UIWindow(windowScene: scene)
        mounted.frame = scene.screen.bounds
        mounted.rootViewController = UIHostingController(rootView:
            PlaybackPresentation(model: model, fileID: id).preferredColorScheme(.light))
        window = mounted
        mounted.makeKeyAndVisible()
        try await eventually("Exact product preview did not become ready") {
            media.selected?.versionID == manifest.version_id &&
            self.nativeController()?.isReadyForDisplay == true && media.preview.player?.currentItem?.status == .readyToPlay
        }
        let player = try XCTUnwrap(media.preview.player)
        let item = try XCTUnwrap(player.currentItem)
        let temporary = try XCTUnwrap(adapter.temporaryURL)
        try require(player.rate == 0 && nativeController()?.player === player, "Must prepare without autoplay")
        try require(FileManager.default.fileExists(atPath: temporary.path), "Viewing copy missing before playback")
        let before = player.currentTime().seconds
        try await phase("play", manifest: manifest)
        try await eventually("Native Play did not advance the same player") {
            player.rate > 0 && player.currentTime().seconds >= before + 0.5 && player.currentItem === item
        }
        try await phase("frame-one", manifest: manifest, player: player)
        let firstTime = player.currentTime().seconds
        try await eventually("Playback time did not continue advancing") {
            player.rate > 0 && player.currentTime().seconds >= firstTime + 1.25 && player.currentItem === item
        }
        try await phase("frame-two", manifest: manifest, player: player)
        try require(player.rate > 0 && player.currentTime().seconds > firstTime + 1,
                    "Second native capture must precede end of active playback")
        try await phase("pause", manifest: manifest)
        try await eventually("Native Pause did not stop playback") { player.rate == 0 }
        let paused = player.currentTime().seconds
        try await Task.sleep(nanoseconds: 600_000_000)
        try require(player.rate == 0 && abs(player.currentTime().seconds - paused) < 0.1,
                    "Paused playback must remain stationary")
        try require(player.currentItem === item && FileManager.default.fileExists(atPath: temporary.path),
                    "Pause must retain the playable temporary file")
        observations.append(["phase": "paused", "seconds": paused, "file_present": true])
        try await phase("close", manifest: manifest)
        try await eventually("Actual Close did not release player and viewing copy") {
            media.preview.player == nil && player.currentItem == nil &&
            !FileManager.default.fileExists(atPath: temporary.path) && model.selectedFile == nil
        }
        // Verify the original exact bytes again; close must never remove/change them.
        let original = try await JarvisTemporaryFileVerifier(root: folder).copyVerifiedFile(
            from: URL(fileURLWithPath: manifest.output_path), expectedByteCount: manifest.size_bytes,
            expectedSHA256: manifest.content_sha256, maximumByteCount: 64 * 1024 * 1024,
            access: JarvisWorkspaceLocalAccess(), format: .mp4)
        original.discard()
        try require(adapter.contentReads == 1, "Playback must use one selected viewing copy")
        observations.append(["phase": "closed", "temporary_absent": true, "original_hash_unchanged": true])
        try await phase("finished", manifest: manifest)
        succeeded = true
#endif
    }

    private func phase(_ name: String, manifest: PlaybackManifest, player: AVPlayer? = nil) async throws {
        let folder = URL(fileURLWithPath: manifest.directory)
        let at = Date().timeIntervalSince1970
        var stage: [String: Any] = ["run_id": manifest.run_id, "phase": name, "at": at]
        if let player {
            let controller = try XCTUnwrap(nativeController())
            let window = try XCTUnwrap(window)
            try require(controller.player === player && controller.isReadyForDisplay && player.rate > 0,
                        "Capture requires this ready, playing native controller")
            let video = controller.view.convert(controller.videoBounds, to: window)
            try require(video.width > 80 && video.height > 80 && window.bounds.contains(video),
                        "Video must be visibly inside the captured screen")
            // Upper interior strip excludes native central transport and bottom scrubber.
            // Human review must still establish that the crop is unobscured video.
            let crop = CGRect(x: video.minX + video.width * 0.15, y: video.minY + video.height * 0.18,
                              width: video.width * 0.7, height: video.height * 0.18)
            stage["crop"] = [crop.minX, crop.minY, crop.width, crop.height]
            stage["screen"] = [window.screen.bounds.width, window.screen.bounds.height]
            stage["player_seconds_before_capture"] = player.currentTime().seconds
        }
        try writeJSON(stage, to: folder.appendingPathComponent("stage.json"))
        let progressURL = folder.appendingPathComponent("ui-progress.json")
        try await eventually("UI did not complete stage \(name)", seconds: 12) {
            guard let data = try? Data(contentsOf: progressURL),
                  let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
            return raw["run_id"] as? String == manifest.run_id && raw["phase"] as? String == name &&
                (raw["at"] as? Double ?? 0) >= at
        }
        if let player { stage["player_seconds_after_capture"] = player.currentTime().seconds }
        observations.append(stage)
    }

    private func eventually(_ message: String, seconds: Double = 12, _ check: () -> Bool) async throws {
        let until = min(deadline, Date().addingTimeInterval(seconds))
        while Date() < until {
            if check() { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw PlaybackFailure(message)
    }

    private func nativeController() -> AVPlayerViewController? {
        func find(_ controller: UIViewController) -> AVPlayerViewController? {
            if let player = controller as? AVPlayerViewController { return player }
            if let next = controller.presentedViewController, let result = find(next) { return result }
            return controller.children.lazy.compactMap(find).first
        }
        return window?.rootViewController.flatMap(find)
    }
}

private struct PlaybackPresentation: View {
    let model: JarvisFilesViewModel
    let fileID: UUID
    @State private var presented = true
    var body: some View {
        Color.clear.sheet(isPresented: $presented) { JarvisTaskFilePreview(fileID: fileID, model: model) }
    }
}

private final class PlaybackAcceptedFileAdapter: JarvisFilesMediaAccess {
    let manifest: PlaybackManifest
    let media: JarvisMediaFile
    private(set) var temporaryURL: URL?
    private(set) var contentReads = 0
    init(manifest: PlaybackManifest, media: JarvisMediaFile) { self.manifest = manifest; self.media = media }
    func checkContext() throws {} // Component-only adapter; never asserted as verified sign-in.
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        guard UUID(uuidString: id) == UUID(uuidString: manifest.file_id) else { throw PlaybackFailure("Wrong file") }
        return media.file
    }
    func getMediaFile(id: UUID, versionID: String?, access: JarvisWorkspaceLocalAccess) async throws -> JarvisMediaFile {
        try access.check()
        guard id == UUID(uuidString: manifest.file_id), versionID == nil || versionID == manifest.version_id else {
            throw PlaybackFailure("Wrong media reference")
        }
        return media
    }
    func getMediaContent(_ selected: JarvisMediaFile, access: JarvisWorkspaceLocalAccess) async throws -> JarvisVerifiedTemporaryFile {
        guard selected == media, contentReads == 0 else { throw PlaybackFailure("Unexpected repeat/changed read") }
        contentReads += 1
        let file = try await JarvisTemporaryFileVerifier(root: URL(fileURLWithPath: manifest.directory)).copyVerifiedFile(
            from: URL(fileURLWithPath: manifest.output_path), expectedByteCount: manifest.size_bytes,
            expectedSHA256: manifest.content_sha256, maximumByteCount: 64 * 1024 * 1024, access: access, format: .mp4)
        temporaryURL = try file.urlForReading()
        return file
    }
    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage { throw PlaybackFailure("No list requested") }
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data { throw PlaybackFailure("No document requested") }
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion { throw PlaybackFailure("Deletion prohibited") }
}

private struct PlaybackHistoricalMediaResponse: Decodable {
    let file: JarvisMediaFile
}

private struct PlaybackManifest: Decodable {
    let schema_version: Int
    let run_id: String
    let not_before: Double
    let directory: String
    let report_path: String
    let report_sha256: String
    let output_path: String
    let metadata_path: String
    let metadata_sha256: String
    let file_id: String
    let version_id: String
    let content_sha256: String
    let size_bytes: Int
    func validate() throws {
        guard schema_version == 1, let run = UUID(uuidString: run_id), run.uuidString == run_id,
              directory.hasPrefix("/"), URL(fileURLWithPath: directory).lastPathComponent == "phone-playback-" + run_id,
              [report_path, output_path, metadata_path].allSatisfy({ $0.hasPrefix("/") }),
              Date().timeIntervalSince1970 >= not_before, Date().timeIntervalSince1970 - not_before < 180,
              size_bytes > 0, size_bytes <= 64 * 1024 * 1024 else { throw PlaybackFailure("Invalid/stale manifest") }
    }
    func checkAcceptedReport() throws {
        let bytes = try boundedData(report_path, maximum: 8 * 1024 * 1024)
        guard digest(bytes) == report_sha256,
              let report = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              let qualification = report["qualification"] as? [String: Any], qualification["passed"] as? Bool == true,
              let historical = report["historical_output_metadata"] as? [String: Any],
              historical["path"] as? String == metadata_path, historical["sha256"] as? String == metadata_sha256,
              historical["authenticated_phone_delivery"] as? Bool == false,
              historical["live_source_after_cleanup"] as? Bool == false,
              let accepted = report["accepted_output"] as? [String: Any],
              let ref = accepted["file"] as? [String: Any],
              UUID(uuidString: ref["file_id"] as? String ?? "") == UUID(uuidString: file_id),
              ref["version_id"] as? String == version_id, ref["content_sha256"] as? String == content_sha256,
              accepted["size_bytes"] as? Int == size_bytes,
              let artifacts = report["review_artifacts"] as? [String: [String: Any]],
              let output = artifacts["output.mp4"], output["path"] as? String == output_path,
              output["content_sha256"] as? String == content_sha256, output["size_bytes"] as? Int == size_bytes,
              output["state"] as? String == "retained_exact_unreviewed",
              let inspection = report["independent_output_inspection"] as? [String: Any],
              inspection["full_video_audio_decode"] as? Bool == true,
              inspection["content_sha256"] as? String == content_sha256 else {
            throw PlaybackFailure("Exact successful managed-output evidence is required")
        }
    }
}

private struct PlaybackFailure: Error, CustomStringConvertible {
    let description: String
    init(_ text: String) { description = text }
}
private func require(_ value: Bool, _ message: String) throws { if !value { throw PlaybackFailure(message) } }
private func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
private func boundedData(_ path: String, maximum: Int) throws -> Data {
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? handle.close() }
    let data = try handle.read(upToCount: maximum + 1) ?? Data()
    guard data.count <= maximum else { throw PlaybackFailure("Evidence exceeds bound") }
    return data
}
private func writeJSON(_ value: [String: Any], to url: URL) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
}

/// Native fullscreen controls using a synthetic local video; no account,
/// network, capture or heard-audio claims are made by these cases.
@MainActor
final class JarvisRecordingFullscreenTests: XCTestCase {
    private var window: UIWindow?
    private var previous: UIWindow?
    private var model: JarvisMediaPreviewViewModel?
    private var root: URL?
    private var copy: URL?

    override func tearDown() async throws {
        model?.deactivate()
        window?.isHidden = true; window?.rootViewController = nil; window = nil
        previous?.makeKeyAndVisible(); previous = nil
        if let root { try? FileManager.default.removeItem(at: root) }
        model = nil; root = nil; copy = nil
        try await super.tearDown()
    }

    func testInlineRecordingFullscreenPreservesPlayback() async throws {
        try await exerciseFullscreen(inSheet: false)
    }

    func testSheetRecordingFullscreenPreservesPlayback() async throws {
        try await exerciseFullscreen(inSheet: true)
    }

    func testRetiringRecordingClearsFullscreenPlayer() async throws {
        let player = try await prepare(inSheet: true)
        let controller = try XCTUnwrap(findController(in: window?.rootViewController))
        let coordinator = try XCTUnwrap(controller.delegate as? JarvisAudiovisualVideoPlayer.Coordinator)
        activate(try control(containing: "full", in: controller.view))
        try await eventually { coordinator.isFullscreen }
        print("Recording fullscreen: native presentation began")
        try await Task.sleep(for: .milliseconds(700))
        model?.deactivate()
        try await eventually { !coordinator.isFullscreen }
        XCTAssertNil(player.currentItem)
        XCTAssertNil(model?.player)
        XCTAssertFalse(FileManager.default.fileExists(atPath: try XCTUnwrap(copy).path))
    }

    private func exerciseFullscreen(inSheet: Bool) async throws {
        let player = try await prepare(inSheet: inSheet)
        let controller = try XCTUnwrap(findController(in: window?.rootViewController))
        let coordinator = try XCTUnwrap(controller.delegate as? JarvisAudiovisualVideoPlayer.Coordinator)
        let item = try XCTUnwrap(player.currentItem)
        let originalOrientation = window?.windowScene?.interfaceOrientation
        let sought = await player.seek(to: CMTime(seconds: 2, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        XCTAssertTrue(sought)
        capture("recording-inline-\(inSheet)")
        activate(try control(containing: "full", in: controller.view))
        try await eventually { coordinator.isFullscreen }
        try await Task.sleep(for: .milliseconds(700))
        _ = try XCTUnwrap(model?.player, "Fullscreen transition must keep the preview owner alive")
        let geometry = "fullscreen=\(coordinator.isFullscreen), orientation=\(String(describing: window?.windowScene?.interfaceOrientation)), frame=\(controller.view.frame), video=\(item.presentationSize)"
        print(geometry)
        let info = XCTAttachment(string: geometry); info.name = "expanded-geometry"; info.lifetime = .keepAlways; add(info)
        capture("recording-expanded-before-rotation")
        try await eventually { self.window?.windowScene?.interfaceOrientation.isLandscape == true }
        try await Task.sleep(for: .milliseconds(700))
        print("Recording fullscreen: landscape verified")
        XCTAssertTrue(controller.player === player); XCTAssertTrue(player.currentItem === item)
        XCTAssertEqual(player.rate, 0)
        XCTAssertEqual(player.currentTime().seconds, 2, accuracy: 0.15)
        XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(copy).path))
        let expanded = try XCTUnwrap(topPresentation())
        XCTAssertGreaterThan(expanded.view.bounds.width, expanded.view.bounds.height)
        capture("recording-fullscreen-\(inSheet)")
        player.play()
        try await eventually { player.currentTime().seconds > 2.3 }
        print("Recording fullscreen: playback advanced")
        let before = player.currentTime().seconds
        // AVKit's fullscreen controls expose an exit/close button. It belongs
        // to the same player controller, not the surrounding preview sheet.
        let exit = controls(in: expanded.view).first {
            let label = ($0.accessibilityLabel ?? "").lowercased()
            return label.contains("exit full") || label.contains("minimi") || label == "done" || label == "close"
                || $0.accessibilityIdentifier == "Close Button"
        }
        let button = try XCTUnwrap(exit, controlDescription(expanded.view))
        print("Recording fullscreen controls: \(controlDescription(expanded.view))")
        print("Recording exit selected: \(button.accessibilityLabel ?? "nil") [\(button.accessibilityIdentifier ?? "nil")]")
        activate(button)
        try await Task.sleep(for: .milliseconds(700))
        print("After exit: fullscreen=\(coordinator.isFullscreen), orientation=\(self.window?.windowScene?.interfaceOrientation.rawValue ?? -1), original=\(originalOrientation?.rawValue ?? -1)")
        try await eventually { !coordinator.isFullscreen }
        try await eventually { self.window?.windowScene?.interfaceOrientation == originalOrientation }
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertTrue(controller.player === player); XCTAssertTrue(player.currentItem === item)
        XCTAssertTrue(model?.player === player)
        XCTAssertGreaterThan(player.currentTime().seconds, before)
        XCTAssertGreaterThan(player.rate, 0)
        player.pause()
        capture("recording-returned-\(inSheet)")
    }

    private func prepare(inSheet: Bool) async throws -> AVPlayer {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["JARVIS_FULLSCREEN_FIXTURE"] else { throw XCTSkip("Synthetic video fixture required") }
        let source = URL(fileURLWithPath: path), bytes = try Data(contentsOf: source)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("fullscreen-video-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        root = folder
        let file = try await JarvisTemporaryFileVerifier(root: folder).copyVerifiedFile(from: source,
            expectedByteCount: bytes.count, expectedSHA256: digest(bytes), maximumByteCount: 8 * 1024 * 1024,
            access: JarvisWorkspaceLocalAccess(), format: .mp4)
        copy = try file.urlForReading()
        let model = JarvisMediaPreviewViewModel(); self.model = model
        let opened = await model.open(file: file, filename: "Synthetic recording.mp4",
            maximumByteCount: 8 * 1024 * 1024, maximumDurationMilliseconds: 90_000)
        XCTAssertTrue(opened)
        model.player?.isMuted = true
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive })
        previous = scene.windows.first { $0.isKeyWindow }
        let window = UIWindow(windowScene: scene); self.window = window
        window.frame = scene.coordinateSpace.bounds
        window.rootViewController = UIHostingController(rootView: RecordingFullscreenSurface(model: model, inSheet: inSheet))
        window.makeKeyAndVisible()
        try await eventually { self.findController(in: window.rootViewController)?.isReadyForDisplay == true }
        try await Task.sleep(for: .milliseconds(350))
        return try XCTUnwrap(model.player)
    }

    private func eventually(_ check: () -> Bool) async throws {
        for _ in 0..<120 {
            if check() { return }
            try await Task.sleep(for: .milliseconds(50))
        }
        throw PlaybackFailure("Timed out waiting for recording fullscreen state")
    }
    private func findController(in controller: UIViewController?) -> AVPlayerViewController? {
        guard let controller else { return nil }
        if let player = controller as? AVPlayerViewController { return player }
        if let result = findController(in: controller.presentedViewController) { return result }
        return controller.children.lazy.compactMap { self.findController(in: $0) }.first
    }
    private func topPresentation() -> UIViewController? {
        var controller = window?.rootViewController
        while let presented = controller?.presentedViewController { controller = presented }
        return controller
    }
    private func controls(in view: UIView) -> [UIControl] {
        (view as? UIControl).map { [$0] } ?? view.subviews.flatMap { controls(in: $0) }
    }
    private func controlDescription(_ view: UIView) -> String {
        controls(in: view).map { "\($0.accessibilityLabel ?? "nil") [\($0.accessibilityIdentifier ?? "nil")] hidden=\($0.isHidden)" }.joined(separator: "\n")
    }
    private func control(containing text: String, in view: UIView) throws -> UIControl {
        try XCTUnwrap(controls(in: view).first {
            ($0.accessibilityLabel ?? "").lowercased().contains(text)
                || ($0.accessibilityIdentifier ?? "").lowercased().contains(text)
        }, controlDescription(view))
    }
    private func activate(_ control: UIControl) {
        print("Activating \(control.accessibilityIdentifier ?? "button") events=\(control.allControlEvents.rawValue)")
        control.sendActions(for: control.allControlEvents.contains(.primaryActionTriggered) ? .primaryActionTriggered : .touchUpInside)
    }
    private func capture(_ name: String) {
        guard let window else { return }
        window.layoutIfNeeded()
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
        let attachment = XCTAttachment(image: image); attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
        if let folder = ProcessInfo.processInfo.environment["JARVIS_FULLSCREEN_PREVIEWS"] {
            try? image.pngData()?.write(to: URL(fileURLWithPath: folder).appendingPathComponent(name + ".png"))
        }
    }
}

private struct RecordingFullscreenSurface: View {
    @ObservedObject var model: JarvisMediaPreviewViewModel
    let inSheet: Bool
    @State private var presented = true
    @State private var videoFullscreen = false
    var body: some View {
        Group {
            if inSheet { Color.clear.sheet(isPresented: $presented) { content } }
            else { content }
        }.preferredColorScheme(.light)
    }
    private var content: some View {
        NavigationStack {
            ScrollView {
                if let player = model.player {
                    JarvisAudiovisualVideoPlayer(player: player, onFullscreenChanged: { videoFullscreen = $0 })
                        .frame(height: 300).clipShape(RoundedRectangle(cornerRadius: 18)).padding(20)
                }
            }
            .navigationTitle("Recording")
            .background(JarvisColors.background)
        }
        .onDisappear { if !videoFullscreen { model.close() } }
    }
}
