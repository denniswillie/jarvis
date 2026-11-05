import XCTest
import UIKit

/// Attach-only operator for the exact-output hosted component. Never launches
/// the app, seeks the video or calls a model/player method. No native allocation
/// is implied by preparing this unregistered source.
@MainActor
final class JarvisManagedPlaybackUITests: XCTestCase {
    func testDriveExactOutputPlayPauseClose() async throws {
#if !targetEnvironment(simulator)
        throw PlaybackUIFailure("Simulator-only operator")
#else
        guard let path = ProcessInfo.processInfo.environment["JARVIS_MANAGED_PLAYBACK_MANIFEST"] else {
            throw XCTSkip("Requires explicitly allocated playback manifest")
        }
        continueAfterFailure = false
        executionTimeAllowance = 90
        let manifest = try JSONDecoder().decode(PlaybackUIManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        guard manifest.schema_version == 1, let run = UUID(uuidString: manifest.run_id),
              run.uuidString == manifest.run_id, manifest.directory.hasPrefix("/"),
              URL(fileURLWithPath: manifest.directory).lastPathComponent == "phone-playback-" + manifest.run_id,
              Date().timeIntervalSince1970 >= manifest.not_before,
              Date().timeIntervalSince1970 - manifest.not_before < 180 else {
            throw PlaybackUIFailure("Invalid/stale manifest")
        }
        let folder = URL(fileURLWithPath: manifest.directory)
        let resultURL = folder.appendingPathComponent("native-ui-result.json")
        guard !FileManager.default.fileExists(atPath: resultURL.path) else { throw PlaybackUIFailure("Prior UI result exists") }
        let until = Date().addingTimeInterval(90)
        let app = XCUIApplication(bundleIdentifier: "com.symbiote.jarvis")
        let phases = ["play", "frame-one", "frame-two", "pause", "close", "finished"]
        var events: [[String: Any]] = []
        var finished = false
        defer {
            do {
                try json(["run_id": manifest.run_id, "completed_ui_stages": finished, "events": events,
                          "visual_review_required": true, "heard_audio_proven": false], to: resultURL)
            } catch { XCTFail("Cannot retain UI evidence: \(error)") }
        }
        for phase in phases {
            var stage: PlaybackUIStage?
            while Date() < until {
                if let bytes = try? Data(contentsOf: folder.appendingPathComponent("stage.json")) {
                    let candidate = try JSONDecoder().decode(PlaybackUIStage.self, from: bytes)
                    guard candidate.run_id == manifest.run_id, candidate.at >= manifest.not_before,
                          phases.contains(candidate.phase) else { throw PlaybackUIFailure("Foreign/stale stage") }
                    if candidate.phase == phase { stage = candidate; break }
                    guard phases.firstIndex(of: candidate.phase)! < phases.firstIndex(of: phase)! else {
                        throw PlaybackUIFailure("An actionable stage was skipped")
                    }
                }
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            let current = try XCTUnwrap(stage, "No stage before bounded deadline")
            guard app.state == .runningForeground else { throw PlaybackUIFailure("Host is not foreground; never relaunch") }
            var event: [String: Any] = ["phase": phase, "stage_at": current.at]
            switch phase {
            case "play": try await tapTransport("Play", app: app, until: until)
            case "pause": try await tapTransport("Pause", app: app, until: until)
            case "close":
                try await tap(app.buttons.matching(identifier: "Close"), until: until)
                attach(app.screenshot().image, name: "closed")
            case "frame-one", "frame-two":
                let crop = try XCTUnwrap(current.crop)
                let screen = try XCTUnwrap(current.screen)
                guard crop.count == 4, screen.count == 2, screen.allSatisfy({ $0 > 0 }),
                      crop.allSatisfy({ $0.isFinite }), crop[2] > 0, crop[3] > 0 else {
                    throw PlaybackUIFailure("Invalid observed video bounds")
                }
                // This is a rendered screen capture, not an extracted/decoded frame.
                let screenshot = XCUIScreen.main.screenshot().image
                let source = try XCTUnwrap(screenshot.cgImage)
                let sx = Double(source.width) / screen[0], sy = Double(source.height) / screen[1]
                guard abs(sx - sy) < 0.01, screenshot.imageOrientation == .up else {
                    throw PlaybackUIFailure("Unexpected screenshot geometry/orientation")
                }
                let rect = CGRect(x: crop[0] * sx, y: crop[1] * sy, width: crop[2] * sx, height: crop[3] * sy).integral
                guard CGRect(x: 0, y: 0, width: source.width, height: source.height).contains(rect),
                      let pixels = source.cropping(to: rect) else { throw PlaybackUIFailure("Crop leaves captured screen") }
                let cropped = UIImage(cgImage: pixels)
                try save(screenshot, name: phase + "-screen", folder: folder)
                try save(cropped, name: phase + "-video", folder: folder)
                attach(screenshot, name: phase + "-screen-requires-review")
                attach(cropped, name: phase + "-video-requires-review")
                event["crop_pixels"] = [rect.minX, rect.minY, rect.width, rect.height]
                event["player_seconds_before_capture"] = current.player_seconds_before_capture
            case "finished": finished = true
            default: throw PlaybackUIFailure("Unknown stage")
            }
            let now = Date().timeIntervalSince1970
            event["at"] = now
            events.append(event)
            try json(["run_id": manifest.run_id, "phase": phase, "at": now],
                     to: folder.appendingPathComponent("ui-progress.json"))
        }
#endif
    }

    private func tapTransport(_ name: String, app: XCUIApplication, until: Date) async throws {
        let buttons = app.buttons.matching(identifier: name)
        guard buttons.count <= 1 else { throw PlaybackUIFailure("Ambiguous \(name) control") }
        if !buttons.firstMatch.isHittable {
            let surface = app.descendants(matching: .any).matching(identifier: "libraryVideoPlayer")
            guard surface.count == 1, surface.firstMatch.isHittable else {
                throw PlaybackUIFailure("No observed video surface to reveal native controls")
            }
            // The center can hit the hidden transport button and toggle playback.
            // Reveal controls away from it, then tap the visible named button below.
            surface.firstMatch.coordinate(withNormalizedOffset: CGVector(dx: 0.15, dy: 0.25)).tap()
        }
        try await tap(buttons, until: until)
    }

    private func tap(_ query: XCUIElementQuery, until: Date) async throws {
        let deadline = min(until, Date().addingTimeInterval(8))
        while Date() < deadline {
            guard query.count <= 1 else { throw PlaybackUIFailure("Ambiguous native control") }
            if query.firstMatch.exists && query.firstMatch.isEnabled && query.firstMatch.isHittable {
                query.firstMatch.tap()
                return
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw PlaybackUIFailure("Native control did not become reachable")
    }
    private func save(_ image: UIImage, name: String, folder: URL) throws {
        let path = folder.appendingPathComponent(name + ".png")
        guard !FileManager.default.fileExists(atPath: path.path) else { throw PlaybackUIFailure("Prior capture exists") }
        try XCTUnwrap(image.pngData()).write(to: path, options: .atomic)
    }
    private func attach(_ image: UIImage, name: String) {
        let item = XCTAttachment(image: image); item.name = name; item.lifetime = .keepAlways; add(item)
    }
    private func json(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
}
private struct PlaybackUIManifest: Decodable {
    let schema_version: Int, run_id: String, not_before: Double, directory: String
}
private struct PlaybackUIStage: Decodable {
    let run_id: String, phase: String, at: Double
    let crop: [Double]?, screen: [Double]?, player_seconds_before_capture: Double?
}
private struct PlaybackUIFailure: Error, CustomStringConvertible {
    let description: String
    init(_ text: String) { description = text }
}
