import AVKit
import CryptoKit
import Foundation
import SwiftUI
import UIKit
import XCTest
@testable import Jarvis

/// Run alone in a fresh test process, against Core's disposable HTTP fixture.
/// The operator activates real Open/Save/copy-row controls. This test never calls
/// perform/openOffline/didDisplayMedia or starts playback/audio.
/// UI-tool traces and server request logs remain required alongside this result.
@MainActor
final class JarvisConnectedMediaTests: XCTestCase {
    private var app: JarvisAppViewModel?
    private var store: JarvisSessionStore?
    private var defaultsName: String?
    private var window: UIWindow?
    private var previousWindow: UIWindow?
    private var evidenceDirectory: URL?
    private var deadline = Date.distantPast

    func testExactMediaOpenSaveAndVerifiedOfflineReopen() async throws {
        let args = ProcessInfo.processInfo.arguments
        guard let replay = args.firstIndex(of: "--jarvis-replay-manifest"),
              args.indices.contains(replay + 1), args[replay + 1] == "/dev/null",
              !args.contains("--jarvis-replay-run") else {
            throw ConnectedMediaFailure("Require the safe DEBUG launch branch, without audio/replay execution")
        }
        guard let index = args.firstIndex(of: "--jarvis-connected-media-manifest"),
              args.indices.contains(index + 1) else {
            throw XCTSkip("No connected media fixture was supplied")
        }
        let manifestURL = URL(fileURLWithPath: args[index + 1])
        let fixture = try JSONDecoder().decode(ConnectedMediaFixture.self, from: Data(contentsOf: manifestURL))
        try fixture.validate()
        let credentialURL = URL(fileURLWithPath: fixture.credentialPath)
        let attributes = try FileManager.default.attributesOfItem(atPath: credentialURL.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600 else {
            throw ConnectedMediaFailure("Fixture credential must be a regular file with mode 0600")
        }
        let token = try String(contentsOf: credentialURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else { throw ConnectedMediaFailure("Fixture credential is empty") }
        let configuration = JarvisConfiguration(backendBaseURL: fixture.baseURL, staticBearerToken: token)
        let directory = manifestURL.deletingLastPathComponent().appendingPathComponent("phone-media-" + fixture.runID.uuidString)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw ConnectedMediaFailure("Use a fresh run ID; prior copy evidence must not be overwritten")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        evidenceDirectory = directory
        deadline = Date().addingTimeInterval(240)

        // A separate bounded read verifies the real server and expected
        // owner before the app is configured. It never supplies identity to AppVM.
        let transportSettings = URLSessionConfiguration.ephemeral
        transportSettings.timeoutIntervalForRequest = 5
        transportSettings.timeoutIntervalForResource = 10
        let transport = URLSession(configuration: transportSettings)
        defer { transport.invalidateAndCancel() }
        let api = JarvisAPIClient(configuration: configuration, session: transport)
        let context = try await api.fetchAuthenticatedContext(sessionID: fixture.initialSessionID)
        try fixture.validate(context: context)

        let first = try makeApp(fixture: fixture, configuration: configuration, directory: directory)
        try mount(first)
        XCTAssertNil(first.workspace)
        XCTAssertNil(first.productFallbackSessionID)
        try stage("verify-account", instruction: "Tap Note, folder, On this phone. This actual menu action must verify the account without starting voice.", fixture: fixture)
        try await wait("Initial verified account journal") {
            first.workspace?.journalReady == true && !first.isPreparingSavedCopies
                && first.productFallbackSessionID.flatMap(UUID.init(uuidString:)) == fixture.initialSessionID
        }
        let firstWorkspace = try XCTUnwrap(first.workspace)
        XCTAssertEqual(firstWorkspace.sessionID, fixture.initialSessionID)
        XCTAssertEqual(first.state, .ready)
        XCTAssertFalse(first.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(first.audioSessionViewModel.activeSession)
        try await capture("cold-sign-in-verified-copy-list")
        try stage("open-workspace", instruction: "Tap Done, then Workspace to find the prepared copy offer.", fixture: fixture)
        try await wait("Actual Workspace navigation") { first.productTab == .workspace && first.presentedSheet == nil }
        try await wait("Prepared exact-version offer") { firstWorkspace.offers.contains { $0.hasSameTarget(as: fixture.downloadRequest) } }
        XCTAssertTrue(firstWorkspace.canSaveOffline)
        XCTAssertTrue(firstWorkspace.offlineFiles.isEmpty)
        try await wait("Prepared exact-version open offer") { firstWorkspace.offers.contains { $0.hasSameTarget(as: fixture.openRequest) } }
        try await capture("ready-to-open")
        try stage("open-video", instruction: "Tap Open in Workspace. Do not invoke model actions or start playback.", fixture: fixture)
        try await wait("Real Open control and visible native renderer must complete the display receipt") {
            firstWorkspace.deliveries.contains { $0.id == fixture.openRequest.id && $0.phase == .completed && $0.acknowledged }
        }
        let displayed = try XCTUnwrap(firstWorkspace.deliveries.first { $0.id == fixture.openRequest.id })
        XCTAssertTrue(displayed.request.hasSameTarget(as: fixture.openRequest))
        XCTAssertEqual(displayed.acknowledgement?.state, .displayed)
        XCTAssertNil(displayed.acknowledgement?.localReceipt)
        XCTAssertTrue(firstWorkspace.offlineFiles.isEmpty, "Viewing a video must not save it offline")
        let presentation = try XCTUnwrap(firstWorkspace.mediaPresentation)
        XCTAssertEqual(presentation.artifactID, fixture.openRequest.artifactID)
        XCTAssertEqual(presentation.versionID, fixture.openRequest.versionID)
        try validatePreview(firstWorkspace.mediaPreview, fixture: fixture)
        let firstOnlinePlayer = firstWorkspace.mediaPreview.player
        try await capture("exact-video-displayed")
        try stage("save", instruction: "Tap Save offline in Workspace. Do not invoke model actions.", fixture: fixture)
        try await wait("Real Save offline control must produce a completed, acknowledged copy") {
            firstWorkspace.deliveries.contains { $0.id == fixture.downloadRequest.id && $0.phase == .completed && $0.acknowledged }
        }
        let saved = try XCTUnwrap(firstWorkspace.offlineFiles.first { $0.id == fixture.downloadRequest.id })
        XCTAssertTrue(saved.request.hasSameTarget(as: fixture.downloadRequest))
        XCTAssertEqual(saved.acknowledgement?.state, .savedLocal)
        XCTAssertEqual(saved.acknowledgement?.localReceipt?.contentSHA256, fixture.downloadRequest.contentSHA256)
        XCTAssertEqual(saved.acknowledgement?.localReceipt?.byteCount, fixture.downloadRequest.byteCount)
        XCTAssertEqual(saved.acknowledgement?.localReceipt?.versionID, fixture.downloadRequest.versionID)
        XCTAssertEqual(saved.acknowledgement?.localReceipt?.fileID, fixture.downloadRequest.fileID)
        XCTAssertEqual(saved.acknowledgement?.localReceipt?.destination, "app_storage")
        XCTAssertNil(firstWorkspace.offlinePreview)
        try await capture("saved")

        try stage("take-offline", instruction: "Stop only the disposable fixture endpoint, then write control.json phase=offline with this run_id.", fixture: fixture)
        try await waitForControl("offline", fixture: fixture)
        try await requireUnavailableEndpoint(configuration)
        try stage("open-offline", instruction: "Tap Note, folder, On this phone, then the saved-video row. Keep playback paused.", fixture: fixture)
        try await wait("Real saved-copy row must open with the fixture endpoint unavailable") { firstWorkspace.isShowingOfflineMedia && firstWorkspace.offlineMediaPreview.player?.currentItem?.status == .readyToPlay }
        try validatePreview(firstWorkspace.offlineMediaPreview, fixture: fixture)
        let firstOfflinePlayer = firstWorkspace.offlineMediaPreview.player
        XCTAssertEqual(first.presentedSheet, .offlineCopies)
        try await capture("offline-exact-copy")

        try stage("restore-online", instruction: "Restore the same fixture endpoint, then write control.json phase=online with this run_id.", fixture: fixture)
        try await waitForControl("online", fixture: fixture)
        _ = try await api.fetchAuthenticatedContext(sessionID: fixture.initialSessionID)
        first.stopSession()
        try await wait("Close only the disposable initial data session") { first.productFallbackSessionID == nil }
        XCTAssertNil(first.pushNotifications, "Remote push must not become a reset prerequisite")
        first.resetConfiguration()
        guard first.configuration == nil else { throw ConnectedMediaFailure("Real configuration reset did not complete") }
        XCTAssertFalse(firstWorkspace.canSaveOffline)
        XCTAssertNil(firstWorkspace.offlinePreview)
        XCTAssertNil(firstOnlinePlayer?.currentItem)
        XCTAssertNil(firstOfflinePlayer?.currentItem)
        XCTAssertFalse(firstWorkspace.isShowingOfflineMedia)
        XCTAssertNil(firstWorkspace.mediaPresentation)
        clearFixtureCredential()

        let second = try makeApp(fixture: fixture, configuration: configuration, directory: directory)
        try mount(second)
        XCTAssertNil(second.workspace)
        XCTAssertNil(second.productFallbackSessionID)
        try stage("verify-after-reauthentication", instruction: "Tap Note, folder, On this phone. Verify the retained copy appears without starting voice.", fixture: fixture)
        try await wait("New session must independently verify the same account journal") {
            second.workspace?.journalReady == true && !second.isPreparingSavedCopies
                && second.workspace?.sessionID != fixture.initialSessionID
        }
        let secondWorkspace = try XCTUnwrap(second.workspace)
        await secondWorkspace.refreshLocalEntries()
        XCTAssertEqual(secondWorkspace.offlineFiles.map(\.id), [fixture.downloadRequest.id])
        XCTAssertNil(secondWorkspace.offlinePreview)
        XCTAssertEqual(second.state, .ready)
        try await capture("reauthenticated-copy-list")
        try stage("open-after-reauthentication", instruction: "Tap the retained saved-video row in On this phone. Keep playback paused.", fixture: fixture)
        try await wait("Real retained-copy row must reopen after same-owner verification") { secondWorkspace.isShowingOfflineMedia && secondWorkspace.offlineMediaPreview.player?.currentItem?.status == .readyToPlay }
        try validatePreview(secondWorkspace.offlineMediaPreview, fixture: fixture)
        XCTAssertEqual(second.presentedSheet, .offlineCopies)
        XCTAssertFalse(second.audioSessionViewModel.isJarvisActive)
        XCTAssertNil(second.audioSessionViewModel.activeSession)
        try await capture("reauthenticated-exact-copy")
        try stage("finished-awaiting-review", instruction: "Check the terminal XCTest result, screenshots, UI activation trace and server logs. Verify exactly two content downloads (Open and Save), one displayed and one exact saved_local receipt, and no download or new receipt on offline/reauthenticated reopen. Cold offline launch, playback/audio and physical-device behavior remain unqualified.", fixture: fixture)
    }

    override func tearDown() async throws {
        window?.isHidden = true
        window?.rootViewController = nil
        window = nil
        previousWindow?.makeKeyAndVisible()
        previousWindow = nil
        app?.resetConfiguration()
        app = nil
        clearFixtureCredential()
        // Keep this run's journal, screenshots and stage record for inspection.
        try await super.tearDown()
    }

    private func makeApp(fixture: ConnectedMediaFixture, configuration: JarvisConfiguration, directory: URL) throws -> JarvisAppViewModel {
        let name = "jarvis.connected-media." + UUID().uuidString
        defaultsName = name
        let store = JarvisSessionStore(defaults: try XCTUnwrap(UserDefaults(suiteName: name)), keychain: JarvisKeychain(service: name))
        self.store = store
        let app = JarvisAppViewModel(store: store, audioSessionViewModel: JarvisAudioSessionViewModel(),
            workspaceJournalFactory: { context in
                try fixture.validate(context: context)
                return try JarvisWorkspaceLocalStore(ownerKey: JarvisWorkspaceLocalStore.ownerKey(for: context),
                    root: directory.appendingPathComponent("journal", isDirectory: true), allowsFileBytes: true)
            })
        app.saveSetup(backendBaseURL: configuration.backendBaseURL.absoluteString, staticBearerToken: configuration.staticBearerToken)
        guard app.configuration == configuration else { throw ConnectedMediaFailure("Fixture configuration was not accepted") }
        self.app = app
        return app
    }

    private func mount(_ app: JarvisAppViewModel) throws {
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first { $0.activationState == .foregroundActive })
        if previousWindow == nil { previousWindow = scene.windows.first { $0.isKeyWindow } }
        window?.isHidden = true
        window?.rootViewController = nil
        let window = UIWindow(windowScene: scene)
        window.rootViewController = UIHostingController(rootView: JarvisRootView(viewModel: app).environment(\.scenePhase, .active))
        self.window = window
        window.makeKeyAndVisible()
    }

    private func clearFixtureCredential() {
        store?.clearConfiguration()
        store = nil
        if let defaultsName { UserDefaults.standard.removePersistentDomain(forName: defaultsName) }
        defaultsName = nil
    }

    private func validatePreview(_ preview: JarvisMediaPreviewViewModel, fixture: ConnectedMediaFixture) throws {
        let player = try XCTUnwrap(preview.player)
        XCTAssertEqual(player.rate, 0, "Connected qualification does not start playback or audio")
        XCTAssertEqual(preview.filename, fixture.filename)
        let asset = try XCTUnwrap(player.currentItem?.asset as? AVURLAsset)
        let reader = try FileHandle(forReadingFrom: asset.url)
        defer { try? reader.close() }
        var size = 0, hash = SHA256()
        while let chunk = try reader.read(upToCount: 65_536), !chunk.isEmpty {
            size += chunk.count; hash.update(data: chunk)
        }
        XCTAssertEqual(size, fixture.downloadRequest.byteCount)
        XCTAssertEqual(hash.finalize().map { String(format: "%02x", $0) }.joined(), fixture.downloadRequest.contentSHA256)
    }

    private func requireUnavailableEndpoint(_ configuration: JarvisConfiguration) async throws {
        let settings = URLSessionConfiguration.ephemeral
        settings.timeoutIntervalForRequest = 2
        settings.timeoutIntervalForResource = 3
        let session = URLSession(configuration: settings)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: configuration.backendBaseURL.appendingPathComponent("workspace/note"))
        request.setValue("Bearer " + configuration.staticBearerToken, forHTTPHeaderField: "Authorization")
        do {
            _ = try await session.data(for: request)
        } catch let error as URLError where [.cannotConnectToHost, .networkConnectionLost, .notConnectedToInternet, .timedOut].contains(error.code) {
            return
        }
        throw ConnectedMediaFailure("Fixture endpoint is still reachable; an HTTP error is not offline proof")
    }

    private func wait(_ message: String, _ condition: @MainActor () -> Bool) async throws {
        while Date() < deadline {
            if condition() { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw ConnectedMediaFailure("Connected-copy deadline: " + message)
    }

    private func waitForControl(_ phase: String, fixture: ConnectedMediaFixture) async throws {
        let url = try XCTUnwrap(evidenceDirectory).appendingPathComponent("control.json")
        try await wait("Operator phase " + phase) {
            guard let data = try? Data(contentsOf: url),
                  let value = try? JSONDecoder().decode(ConnectedMediaControl.self, from: data) else { return false }
            return value.runID == fixture.runID && value.phase == phase
        }
    }

    private func stage(_ phase: String, instruction: String, fixture: ConnectedMediaFixture) throws {
        let value: [String: Any] = ["run_id": fixture.runID.uuidString, "phase": phase,
            "instruction": instruction, "at": Date().timeIntervalSince1970,
            "session_id": app?.workspace?.sessionID.uuidString ?? "none"]
        let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: try XCTUnwrap(evidenceDirectory).appendingPathComponent("stage.json"), options: .atomic)
    }

    private func capture(_ name: String) async throws {
        // Allow the published state to reach SwiftUI before collecting pixels.
        // This is a review capture, never a substitute for a display receipt.
        try await Task.sleep(nanoseconds: 100_000_000)
        let window = try XCTUnwrap(window)
        window.layoutIfNeeded()
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
        let attachment = XCTAttachment(image: image)
        attachment.name = "connected-media-" + name + "-requires-visual-review"
        attachment.lifetime = .keepAlways
        add(attachment)
        try XCTUnwrap(image.pngData()).write(to: try XCTUnwrap(evidenceDirectory).appendingPathComponent(name + ".png"), options: .atomic)
    }
}

private struct ConnectedMediaFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

private struct ConnectedMediaControl: Decodable {
    let runID: UUID
    let phase: String
    enum CodingKeys: String, CodingKey { case runID = "run_id", phase }
}

private struct ConnectedMediaFixture: Decodable {
    let schemaVersion: Int
    let runID: UUID
    let baseURL: URL
    let credentialPath: String
    let expectedOwnerID: String
    let expectedBackendNamespace: String
    let initialSessionID: UUID
    let openRequest: JarvisWorkspaceRequest
    let downloadRequest: JarvisWorkspaceRequest
    let filename: String
    let contentType: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", runID = "run_id", baseURL = "base_url", credentialPath = "credential_path"
        case expectedOwnerID = "expected_owner_id", expectedBackendNamespace = "expected_backend_namespace"
        case initialSessionID = "initial_session_id", openRequest = "open_request", downloadRequest = "download_request"
        case filename, contentType = "content_type"
    }

    func validate() throws {
        let limit = try downloadRequest.validatedTransferByteLimit()
        guard schemaVersion == 1, baseURL.scheme == "http", ["127.0.0.1", "localhost", "::1"].contains(baseURL.host ?? ""),
              baseURL.user == nil, baseURL.password == nil, baseURL.query == nil, baseURL.fragment == nil,
              credentialPath.hasPrefix("/"), !expectedOwnerID.isEmpty, !expectedBackendNamespace.isEmpty,
              openRequest.action == .open, openRequest.artifactKind == "file", openRequest.contentReady,
              openRequest.state == .pending, openRequest.format == nil, openRequest.transferPolicy == nil,
              openRequest.transferLimitBytes == nil, openRequest.fileID == nil, openRequest.byteCount == nil,
              openRequest.contentSHA256 == nil,
              downloadRequest.action == .download, downloadRequest.artifactKind == "file", downloadRequest.contentReady,
              downloadRequest.state == .pending, downloadRequest.format == nil, downloadRequest.transferPolicy == .output,
              downloadRequest.fileID == downloadRequest.artifactID, openRequest.id != downloadRequest.id,
              openRequest.artifactID == downloadRequest.artifactID, openRequest.versionID == downloadRequest.versionID,
              !downloadRequest.versionID.isEmpty, limit >= 15_431_453,
              downloadRequest.byteCount == 15_431_453,
              downloadRequest.contentSHA256 == "f4f885c568d5bee8310e1dea75ec5b36860c954b03dc37d2654e9dee4a001ce5",
              !filename.isEmpty, contentType == "video/mp4" else {
            throw ConnectedMediaFailure("Invalid local fixture: require paired exact offers for the pinned synthetic H264/AAC MP4")
        }
    }

    func validate(context: JarvisAuthenticatedContext) throws {
        guard context.hasValidIdentity, context.ownerID == expectedOwnerID,
              context.backendNamespace == expectedBackendNamespace else {
            throw ConnectedMediaFailure("The server-verified account/backend does not match this disposable fixture")
        }
    }
}
