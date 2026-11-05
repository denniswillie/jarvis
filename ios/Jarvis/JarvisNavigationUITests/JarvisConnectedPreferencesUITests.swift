import XCTest

/// Attach-only controls for a separately coordinated real profile-commit journey.
/// The existing session owner supplies the app, backend commits and stage file.
/// This operator never launches voice, fabricates a commit event or calls a model.
/// The controller awaits ui-progress.json for each stage before advancing.
@MainActor
final class JarvisConnectedPreferencesUITests: XCTestCase {
    private var deadline = Date.distantPast
    private var events: [[String: String]] = []

    func testPreserveDraftAcrossCommittedPushAndSaveThroughControls() async throws {
#if !targetEnvironment(simulator)
        throw PreferencesUIFailure("This operator is simulator-only")
#else
        guard let path = ProcessInfo.processInfo.environment["JARVIS_CONNECTED_PREFERENCES_UI_MANIFEST"] else {
            throw XCTSkip("Requires a separately allocated real-session preferences journey")
        }
        continueAfterFailure = false
        executionTimeAllowance = 240
        let manifest = try JSONDecoder().decode(PreferencesUIManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        try manifest.validate()
        let stageURL = URL(fileURLWithPath: manifest.stagePath)
        let resultURL = stageURL.deletingLastPathComponent().appendingPathComponent("native-ui-driver.json")
        guard !FileManager.default.fileExists(atPath: resultURL.path) else {
            throw PreferencesUIFailure("Prior UI evidence must not be overwritten")
        }
        deadline = Date().addingTimeInterval(240)
        let app = XCUIApplication(bundleIdentifier: "com.symbiote.jarvis")
        var completed = false
        defer {
            let result: [String: Any] = ["run_id": manifest.runID.uuidString,
                "completed_ui_stages": completed, "events": events,
                "scope": "Real controls only; actual commit, websocket, owner and provider evidence required separately"]
            do {
                let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: resultURL, options: .atomic)
            } catch { XCTFail("Could not preserve preferences UI evidence") }
        }
        let phases = ["open-preferences", "edit-draft", "review-update"] +
            (manifest.exerciseSaveFailure == true ? ["save-failure"] : []) + ["save-draft",
                      "verify-saved", "verify-clean-push", "finished-awaiting-review"]
        var lastIndex = -1
        while Date() < deadline {
            guard let data = try? Data(contentsOf: stageURL) else {
                try await Task.sleep(nanoseconds: 100_000_000)
                continue
            }
            let stage = try JSONDecoder().decode(PreferencesUIStage.self, from: data)
            guard stage.runID == manifest.runID, stage.at >= manifest.notBefore,
                  let index = phases.firstIndex(of: stage.phase), index >= lastIndex else {
                throw PreferencesUIFailure("Wrong run, stale or regressed preferences stage")
            }
            if index == lastIndex {
                try await Task.sleep(nanoseconds: 100_000_000)
                continue
            }
            guard index == lastIndex + 1 else { throw PreferencesUIFailure("An actionable stage was skipped") }
            lastIndex = index
            trace(stage.phase, "observed")
            if stage.phase == "finished-awaiting-review" { completed = true; return }
            try requireForeground(app)
            record(stage.phase + "-before", app: app)
            switch stage.phase {
            case "open-preferences":
                try await tap(app.tabBars.buttons.matching(identifier: "Home"), app: app)
                try await tap(app.buttons.matching(identifier: "homePreferencesButton"), app: app)
                try await requireViewing("", app: app)
            case "edit-draft":
                try await tap(app.buttons.matching(identifier: "preferencesEditButton"), app: app)
                try await requireEditor("", app: app)
                try await requireKeyboardControl(app.buttons["preferencesSaveButton"], app: app)
                let editor = app.textViews.matching(identifier: "preferencesEditor").firstMatch
                editor.typeText(manifest.draft)
                try await requireEditor(manifest.draft, app: app)
            case "review-update":
                try await requireEditor(manifest.draft, app: app)
                try await wait(app: app) {
                    app.staticTexts["Preferences changed elsewhere"].exists &&
                    app.staticTexts[manifest.pushedWhileDirty].exists &&
                    app.buttons["preferencesSaveButton"].exists &&
                    !app.buttons["preferencesSaveButton"].isEnabled
                }
                try await requireKeyboardControl(app.buttons["Keep my draft"], app: app)
            case "save-failure":
                try await tap(app.buttons.matching(identifier: "Keep my draft"), app: app)
                try await requireKeyboardControl(app.buttons["preferencesSaveButton"], app: app)
                try await tap(app.buttons.matching(identifier: "preferencesSaveButton"), app: app)
                try await wait(app: app) { app.staticTexts["preferencesError"].exists }
                try await requireEditor(manifest.draft, app: app)
                try await requireKeyboardControl(app.buttons["preferencesSaveButton"], app: app)
                guard !app.buttons["preferencesEditButton"].exists,
                      app.buttons["preferencesSaveButton"].isEnabled else {
                    throw PreferencesUIFailure("Failed save must retain editing and allow retry")
                }
            case "save-draft":
                if app.buttons["Keep my draft"].exists {
                    try await tap(app.buttons.matching(identifier: "Keep my draft"), app: app)
                }
                try await requireEditor(manifest.draft, app: app)
                try await requireKeyboardControl(app.buttons["preferencesSaveButton"], app: app)
                try await tap(app.buttons.matching(identifier: "preferencesSaveButton"), app: app)
                if manifest.exerciseSaveFailure == true {
                    try await wait(app: app) {
                        app.keyboards.firstMatch.exists && !app.buttons["preferencesSaveButton"].isEnabled &&
                        !app.buttons["preferencesEditButton"].exists
                    }
                }
            case "verify-saved":
                try await requireViewing(manifest.draft, app: app)
            case "verify-clean-push":
                try await requireViewing(manifest.pushedWhileClean, app: app)
                guard !app.staticTexts["Preferences changed elsewhere"].exists else {
                    throw PreferencesUIFailure("A clean editor incorrectly requires conflict resolution")
                }
            default: throw PreferencesUIFailure("Unsupported stage")
            }
            trace(stage.phase, "controls completed; authoritative assertions remain with session owner")
            record(stage.phase + "-after", app: app)
            let progress: [String: Any] = ["run_id": manifest.runID.uuidString,
                "phase": stage.phase, "at": Date().timeIntervalSince1970]
            try JSONSerialization.data(withJSONObject: progress, options: [.sortedKeys]).write(
                to: stageURL.deletingLastPathComponent().appendingPathComponent("ui-progress.json"),
                options: .atomic)
        }
        throw PreferencesUIFailure("Preferences UI deadline reached")
#endif
    }

    private func requireEditor(_ text: String, app: XCUIApplication) async throws {
        try await wait(app: app) {
            let query = app.textViews.matching(identifier: "preferencesEditor")
            return query.count == 1 && query.firstMatch.value as? String == text
        }
    }

    private func requireViewing(_ text: String, app: XCUIApplication) async throws {
        try await wait(app: app) {
            let saved = app.staticTexts["preferencesSavedText"]
            return saved.exists && saved.label == (text.isEmpty ? "Add your preferences when you’re ready." : text) &&
                !app.keyboards.firstMatch.exists && !app.textViews["preferencesEditor"].exists &&
                app.buttons["preferencesEditButton"].isHittable && !app.buttons["preferencesSaveButton"].exists
        }
    }

    private func requireKeyboardControl(_ control: XCUIElement, app: XCUIApplication) async throws {
        try await wait(app: app) {
            let keyboard = app.keyboards.firstMatch
            return keyboard.exists && control.exists && control.isHittable &&
                control.frame.maxY <= keyboard.frame.minY
        }
        trace("keyboard-control", "\(control.label)=\(control.frame); keyboard=\(app.keyboards.firstMatch.frame)")
    }

    private func tap(_ query: XCUIElementQuery, app: XCUIApplication) async throws {
        try await wait(app: app) {
            guard query.count <= 1 else { throw PreferencesUIFailure("Ambiguous preferences control") }
            let control = query.firstMatch
            guard control.exists && control.isEnabled && control.isHittable else { return false }
            control.tap()
            return true
        }
    }

    private func wait(app: XCUIApplication, condition: () throws -> Bool) async throws {
        let limit = min(deadline, Date().addingTimeInterval(10))
        while Date() < limit {
            try requireForeground(app)
            if try condition() { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        record("missing-control-or-expected-text", app: app)
        throw PreferencesUIFailure("Preferences control or expected text did not become available")
    }

    private func requireForeground(_ app: XCUIApplication) throws {
        guard app.state == .runningForeground else {
            throw PreferencesUIFailure("Existing app must remain foreground; this operator never relaunches it")
        }
    }

    private func trace(_ stage: String, _ action: String) {
        events.append(["stage": stage, "action": action, "at": String(Date().timeIntervalSince1970)])
    }

    private func record(_ name: String, app: XCUIApplication) {
        let hierarchy = XCTAttachment(string: app.debugDescription)
        hierarchy.name = name + "-hierarchy"; hierarchy.lifetime = .keepAlways; add(hierarchy)
        if app.state == .runningForeground {
            let frame = XCTAttachment(screenshot: app.screenshot())
            frame.name = name; frame.lifetime = .keepAlways; add(frame)
        }
    }
}

private struct PreferencesUIManifest: Decodable {
    let schemaVersion: Int
    let runID: UUID
    let stagePath: String
    let notBefore: TimeInterval
    let draft: String
    let pushedWhileDirty: String
    let pushedWhileClean: String
    let exerciseSaveFailure: Bool?
    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", runID = "run_id", stagePath = "stage_path"
        case notBefore = "not_before", draft
        case pushedWhileDirty = "pushed_while_dirty", pushedWhileClean = "pushed_while_clean"
        case exerciseSaveFailure = "exercise_save_failure"
    }
    func validate() throws {
        let stage = URL(fileURLWithPath: stagePath)
        let texts = [draft, pushedWhileDirty, pushedWhileClean]
        guard schemaVersion == 1, stagePath.hasPrefix("/"), stage.lastPathComponent == "stage.json",
              stage.deletingLastPathComponent().lastPathComponent == "phone-preferences-" + runID.uuidString,
              notBefore.isFinite, notBefore > 0, notBefore <= Date().timeIntervalSince1970,
              Set(texts).count == 3, texts.allSatisfy({ !$0.isEmpty && $0.unicodeScalars.count <= 1_000 }) else {
            throw PreferencesUIFailure("Invalid preferences fixture manifest")
        }
    }
}

private struct PreferencesUIStage: Decodable {
    let runID: UUID
    let phase: String
    let at: TimeInterval
    enum CodingKeys: String, CodingKey { case runID = "run_id", phase, at }
}

private struct PreferencesUIFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
