import XCTest

/// Attaches only to the separately launched hosted platform probe. Never calls
/// launch/activate/terminate; success still requires the hosted test to pass.
@MainActor
final class JarvisNativeAttachmentUITests: XCTestCase {
    func testAttachToExistingHostedWindow() async throws {
#if !targetEnvironment(simulator)
        throw AttachmentUIFailure("Simulator only")
#else
        guard let path = ProcessInfo.processInfo.environment["JARVIS_NATIVE_ATTACH_MANIFEST"] else {
            throw XCTSkip("Run only with the allocated attachment manifest")
        }
        continueAfterFailure = false
        executionTimeAllowance = 60
        let manifest = try JSONDecoder().decode(AttachmentUIManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        let directory = URL(fileURLWithPath: manifest.directory)
        guard manifest.directory.hasPrefix("/"), directory.lastPathComponent == "native-attach-" + manifest.runID.uuidString,
              FileManager.default.fileExists(atPath: directory.path),
              !FileManager.default.fileExists(atPath: directory.appendingPathComponent("driver-done.json").path) else {
            throw AttachmentUIFailure("Require a fresh run-bound evidence directory")
        }
        let app = XCUIApplication(bundleIdentifier: "com.symbiote.jarvis")
        let deadline = Date().addingTimeInterval(50)
        let stages = [("ready", "Note", "Attachment Note"),
                      ("note-observed", "Workspace", "Attachment Workspace"),
                      ("workspace-observed", "Home", "Attachment Home")]
        var hostPID: Int?
        for (phase, tab, destination) in stages {
            let stage = try await waitForStage(phase, manifest: manifest, deadline: deadline)
            if let hostPID {
                guard hostPID == stage.hostPID else { throw AttachmentUIFailure("Hosted process changed") }
            } else { hostPID = stage.hostPID }
            guard app.state == .runningForeground,
                  app.staticTexts["nativeAttachmentRun"].label == manifest.runID.uuidString else {
                capture("wrong-hosted-window", app)
                throw AttachmentUIFailure("Expected hosted window is not foreground; do not launch or activate it")
            }
            let query = app.tabBars.buttons.matching(identifier: tab)
            guard query.count == 1 && query.firstMatch.isHittable else {
                capture("missing-native-tab", app)
                throw AttachmentUIFailure("Native hosted tab unavailable")
            }
            query.firstMatch.tap()
            let visible = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
                query.firstMatch.isSelected && app.staticTexts[destination].exists
            }, object: nil)
            guard await XCTWaiter.fulfillment(of: [visible], timeout: 5) == .completed else {
                capture("destination-missing", app)
                throw AttachmentUIFailure("Native tap did not reveal destination")
            }
            capture("attached-" + tab.lowercased(), app)
        }
        let last = try await waitForStage("home-observed", manifest: manifest, deadline: deadline)
        guard last.hostPID == hostPID else { throw AttachmentUIFailure("Hosted process changed on return") }
        let done: [String: Any] = ["run_id": manifest.runID.uuidString, "phase": "driver-completed",
            "at": Date().timeIntervalSince1970, "host_pid": last.hostPID]
        try JSONSerialization.data(withJSONObject: done, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("driver-done.json"), options: .atomic)
#endif
    }

    private func waitForStage(_ phase: String, manifest: AttachmentUIManifest, deadline: Date) async throws -> AttachmentUIStage {
        let url = URL(fileURLWithPath: manifest.directory).appendingPathComponent("stage.json")
        while Date() < deadline {
            if let data = try? Data(contentsOf: url) {
                let stage = try JSONDecoder().decode(AttachmentUIStage.self, from: data)
                guard stage.runID == manifest.runID, stage.hostPID > 0 else {
                    throw AttachmentUIFailure("Wrong hosted run or process")
                }
                if stage.phase == phase { return stage }
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw AttachmentUIFailure("No hosted stage: " + phase)
    }

    private func capture(_ name: String, _ app: XCUIApplication) {
        let hierarchy = XCTAttachment(string: app.debugDescription)
        hierarchy.name = name + "-hierarchy"
        hierarchy.lifetime = .keepAlways
        add(hierarchy)
        if app.state == .runningForeground {
            let screenshot = XCTAttachment(screenshot: app.screenshot())
            screenshot.name = name
            screenshot.lifetime = .keepAlways
            add(screenshot)
        }
    }
}

private struct AttachmentUIManifest: Decodable {
    let runID: UUID
    let directory: String
    enum CodingKeys: String, CodingKey { case runID = "run_id", directory }
}
private struct AttachmentUIStage: Decodable {
    let runID: UUID
    let phase: String
    let hostPID: Int
    enum CodingKeys: String, CodingKey { case runID = "run_id", phase, hostPID = "host_pid" }
}
private struct AttachmentUIFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
