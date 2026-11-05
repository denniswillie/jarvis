import XCTest

/// UI operator for the existing hosted connected-copy tests, not a substitute
/// for its real HTTP, receipt, identity, disk-byte and paused-player assertions.
/// Requires a separately scheduled hosted runner on the same simulator.
/// Never launches, activates or terminates the app, or controls the endpoint.
@MainActor
final class JarvisConnectedMediaUITests: XCTestCase {
    private var deadline = Date.distantPast
    private var events: [[String: String]] = []

    func testDriveExistingConnectedMediaStages() async throws {
        try await drive(.media)
    }

    func testDriveExistingConnectedNoteStages() async throws {
        try await drive(.note)
    }

    private func drive(_ kind: ConnectedCopyKind) async throws {
#if !targetEnvironment(simulator)
        throw ConnectedMediaUIFailure("This adapter is simulator-only")
#else
        guard let path = ProcessInfo.processInfo.environment["JARVIS_CONNECTED_COPY_UI_MANIFEST"] else {
            throw XCTSkip("Requires an explicitly allocated connected-copy UI driver manifest")
        }
        continueAfterFailure = false
        executionTimeAllowance = 240
        let manifest = try JSONDecoder().decode(ConnectedMediaUIManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        try manifest.validate(expectedKind: kind)
        let stageURL = URL(fileURLWithPath: manifest.stagePath)
        let resultURL = stageURL.deletingLastPathComponent().appendingPathComponent("native-ui-driver.json")
        guard !FileManager.default.fileExists(atPath: resultURL.path) else {
            throw ConnectedMediaUIFailure("Refusing to overwrite an earlier UI driver result")
        }
        deadline = Date().addingTimeInterval(240)
        let app = XCUIApplication(bundleIdentifier: "com.symbiote.jarvis")
        var completed = false
        defer {
            let result: [String: Any] = ["run_id": manifest.runID.uuidString,
                "completed_ui_stages": completed, "events": events,
                "scope": "Queried UI controls only; hosted XCTest and server evidence remain required"]
            if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
                do { try data.write(to: resultURL, options: .atomic) }
                catch { XCTFail("Could not retain the UI driver trace") }
            }
        }

        let phases = ["verify-account", "open-workspace"] + (kind == .media ? ["open-video"] : []) + ["save", "take-offline",
            "open-offline", "restore-online", "verify-after-reauthentication",
            "open-after-reauthentication", "finished-awaiting-review"]
        let passivePhases: Set<String> = ["take-offline", "restore-online"]
        var lastIndex = -1
        while Date() < deadline {
            guard let data = try? Data(contentsOf: stageURL) else {
                try await Task.sleep(nanoseconds: 100_000_000)
                continue
            }
            let stage = try JSONDecoder().decode(ConnectedMediaUIStage.self, from: data)
            guard stage.runID == manifest.runID, stage.at >= manifest.notBefore,
                  let index = phases.firstIndex(of: stage.phase), index >= lastIndex else {
                throw ConnectedMediaUIFailure("Wrong run, stale, unknown or regressed hosted stage")
            }
            if index == lastIndex {
                try await Task.sleep(nanoseconds: 100_000_000)
                continue
            }
            if index > lastIndex + 1 {
                guard phases[(lastIndex + 1)..<index].allSatisfy({ passivePhases.contains($0) }) else {
                    throw ConnectedMediaUIFailure("An actionable hosted stage was skipped")
                }
            }
            lastIndex = index
            trace(stage.phase, "observed")
            if stage.phase == "finished-awaiting-review" {
                completed = true
                return
            }
            // Endpoint transitions belong to the external fixture controller.
            if passivePhases.contains(stage.phase) { continue }
            try require(app.state == .runningForeground,
                        "Hosted app is not foreground; attachment must not relaunch it", app: app)
            record(stage.phase + "-before", app: app)
            switch stage.phase {
            case "verify-account", "verify-after-reauthentication":
                try await openCopyList(app)
            case "open-workspace":
                try await tap(app.buttons.matching(identifier: "Done"), "Done", app: app)
                try await tapTab("Workspace", app: app)
            case "open-video":
                try await tap(app.buttons.matching(identifier: "Open"), "Open", app: app)
            case "save":
                try await tap(app.buttons.matching(identifier: "Save offline"), "Save offline", app: app,
                              scrollContainer: app.scrollViews.firstMatch)
            case "open-offline":
                try await openCopyList(app)
                try await tap(app.buttons.matching(identifier: manifest.filename), "Exact saved-copy row", app: app)
            case "open-after-reauthentication":
                try await tap(app.buttons.matching(identifier: manifest.filename), "Exact retained-copy row", app: app)
            default:
                throw ConnectedMediaUIFailure("Unsupported actionable stage")
            }
            trace(stage.phase, "native controls activated; hosted assertions still pending")
            record(stage.phase + "-after", app: app)
        }
        record("driver-deadline", app: app)
        throw ConnectedMediaUIFailure("Connected-copy UI deadline reached")
#endif
    }

    private func openCopyList(_ app: XCUIApplication) async throws {
        try await tapTab("Note", app: app)
        try await tap(app.buttons.matching(identifier: "notebookSavedFilesButton"), "Saved-copy menu", app: app)
        try await tap(app.buttons.matching(identifier: "On this phone"), "On this phone", app: app)
    }

    private func tapTab(_ name: String, app: XCUIApplication) async throws {
        try await tap(app.tabBars.buttons.matching(identifier: name), name + " tab", app: app)
    }

    private func tap(_ query: XCUIElementQuery, _ description: String, app: XCUIApplication,
                     scrollContainer: XCUIElement? = nil) async throws {
        let controlDeadline = min(deadline, Date().addingTimeInterval(10))
        var swipes = 0
        while Date() < controlDeadline {
            try require(app.state == .runningForeground, "Hosted app left foreground", app: app)
            try require(query.count <= 1, "Ambiguous native control: " + description, app: app)
            let element = query.firstMatch
            if element.exists && element.isEnabled && element.isHittable {
                element.tap()
                trace(description, "tap")
                return
            }
            // Save may be below the paused preview. Scroll only an observed
            // scroll view, at most three times; never tap a guessed coordinate.
            if element.exists && !element.isHittable, swipes < 3,
               let container = scrollContainer, container.exists && container.isHittable {
                container.swipeUp()
                swipes += 1
                trace(description, "scroll to control " + String(swipes))
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        record("missing-" + description, app: app)
        throw ConnectedMediaUIFailure("Native control unavailable: " + description)
    }

    private func require(_ condition: Bool, _ message: String, app: XCUIApplication) throws {
        if !condition {
            record("failed-native-attachment", app: app)
            throw ConnectedMediaUIFailure(message)
        }
    }

    private func trace(_ stage: String, _ action: String) {
        events.append(["stage": stage, "action": action, "at": String(Date().timeIntervalSince1970)])
    }

    private func record(_ name: String, app: XCUIApplication) {
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

private struct ConnectedMediaUIManifest: Decodable {
    let schemaVersion: Int
    let kind: ConnectedCopyKind
    let runID: UUID
    let stagePath: String
    let filename: String
    let notBefore: TimeInterval
    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", runID = "run_id", stagePath = "stage_path"
        case kind, filename, notBefore = "not_before"
    }

    func validate(expectedKind: ConnectedCopyKind) throws {
        let stage = URL(fileURLWithPath: stagePath)
        let prefix = kind == .media ? "phone-media-" : "phone-copy-"
        guard schemaVersion == 2, kind == expectedKind, stagePath.hasPrefix("/"), stage.lastPathComponent == "stage.json",
              stage.deletingLastPathComponent().lastPathComponent == prefix + runID.uuidString,
              !filename.isEmpty, filename.count <= 255, !filename.contains("/"),
              notBefore.isFinite, notBefore > 0, notBefore <= Date().timeIntervalSince1970 else {
            throw ConnectedMediaUIFailure("Invalid run-bound UI driver manifest")
        }
    }
}

private enum ConnectedCopyKind: String, Decodable { case media, note }

private struct ConnectedMediaUIStage: Decodable {
    let runID: UUID
    let phase: String
    let at: TimeInterval
    enum CodingKeys: String, CodingKey { case runID = "run_id", phase, at }
}

private struct ConnectedMediaUIFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
