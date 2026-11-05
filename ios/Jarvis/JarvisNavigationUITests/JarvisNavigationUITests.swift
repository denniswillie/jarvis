import XCTest

/// A native control-path smoke, independent of Device Hub's host AX bridge.
/// Run alone on the designated simulator with its existing app configuration.
/// This launches the normal root and taps only the Home, Note and Workspace tabs.
/// A passing result does not qualify backend content, saved copies or audio.
@MainActor
final class JarvisNavigationUITests: XCTestCase {
    func testIdleHomeCanOpenNoteAndWorkspace() throws {
#if !targetEnvironment(simulator)
        throw NavigationFailure("This smoke is simulator-only.")
#else
        guard ProcessInfo.processInfo.environment["JARVIS_NAVIGATION_SMOKE"] == "1" else {
            throw XCTSkip("Schedule explicitly in the shared simulator lane.")
        }
        continueAfterFailure = false
        executionTimeAllowance = 60

        let app = XCUIApplication(bundleIdentifier: "com.symbiote.jarvis")
        // No replay execution, deep link, credentials or testing-only app branch.
        app.launchArguments = []
        app.launchEnvironment = [:]
        app.launch()

        let bar = app.tabBars.firstMatch
        try require(bar.waitForExistence(timeout: 10), "Native tab bar is unavailable", app: app)
        let home = bar.buttons["Home"]
        let note = bar.buttons["Note"]
        let workspace = bar.buttons["Workspace"]
        try require(home.exists && note.exists && workspace.exists,
                    "Native automation did not expose the three expected tab controls", app: app)
        // Restore Home on normal completion or a thrown verification failure.
        // Never tap an unobserved fallback position or an audio control.
        defer {
            if app.state == .runningForeground && home.exists && home.isHittable && !home.isSelected {
                home.tap()
            }
        }
        if !home.isSelected {
            try require(home.isHittable, "Home tab is not hittable", app: app)
            home.tap()
        }
        try wait("Idle Home", app: app) {
            home.isSelected && app.buttons["Start"].isHittable
                && app.buttons["homePreferencesButton"].exists
                && app.buttons["homeRemindersButton"].exists
        }
        try require(!app.buttons["Stop"].exists, "An active voice session is outside this smoke", app: app)
        record("idle-home", app: app)

        try require(note.isHittable, "Note tab is not hittable", app: app)
        note.tap()
        try wait("Note destination", app: app) {
            note.isSelected && app.navigationBars["Note"].exists
                && app.buttons["notebookSavedFilesButton"].exists
        }
        record("note-after-native-tap", app: app)

        try require(workspace.isHittable, "Workspace tab is not hittable", app: app)
        workspace.tap()
        try wait("Cold Workspace destination", app: app) {
            workspace.isSelected && app.staticTexts["Your workspace"].exists
                && app.staticTexts["Start Jarvis to see live results here. You can read your note in the Note tab."].exists
        }
        record("workspace-after-native-tap", app: app)

        try require(home.isHittable, "Home return control is not hittable", app: app)
        home.tap()
        try wait("Returned idle Home", app: app) { home.isSelected && app.buttons["Start"].isHittable }
        try require(!app.buttons["Stop"].exists, "Navigation changed the visible audio state", app: app)
        record("returned-idle-home", app: app)
#endif
    }

    private func wait(_ description: String, app: XCUIApplication, until condition: @escaping () -> Bool) throws {
        let expectation = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in condition() }, object: nil)
        try require(XCTWaiter.wait(for: [expectation], timeout: 5) == .completed,
                    "Timed out waiting for \(description)", app: app)
    }

    private func require(_ condition: Bool, _ message: String, app: XCUIApplication) throws {
        guard condition else {
            record("failed-navigation-boundary", app: app)
            throw NavigationFailure(message)
        }
    }

    private func record(_ name: String, app: XCUIApplication) {
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = name
        screenshot.lifetime = .keepAlways
        add(screenshot)
        let hierarchy = XCTAttachment(string: app.debugDescription)
        hierarchy.name = name + "-native-hierarchy"
        hierarchy.lifetime = .keepAlways
        add(hierarchy)
    }
}

private struct NavigationFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
