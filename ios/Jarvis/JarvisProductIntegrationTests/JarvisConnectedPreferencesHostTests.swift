import CryptoKit
import Foundation
import SwiftUI
import UIKit
import XCTest
@testable import Jarvis

/// Hosts the actual product shell while a separate native UI runner uses its controls.
/// Backend commits and stages belong to the existing frontend-host controller.
/// No callback replacement, profile mutations, model-action calls or physical inputs.
@MainActor
final class JarvisConnectedPreferencesHostTests: XCTestCase {
    private var app: JarvisAppViewModel?
    private var store: JarvisSessionStore?
    private var defaultsName: String?
    private var window: UIWindow?
    private var previousWindow: UIWindow?
    private var output: URL?
    private var verifiedContext: JarvisAuthenticatedContext?
    private var observations: [[String: Any]] = []
    private var observedStates = Set<String>()
    private var lastSnapshot: Data?
    private var lastReadinessSnapshot: Data?

    func testRealProductPreferencesWithExistingReplaySession() async throws {
#if DEBUG && targetEnvironment(simulator)
        let arguments = ProcessInfo.processInfo.arguments
        guard let safe = arguments.firstIndex(of: "--jarvis-replay-manifest"),
              arguments.indices.contains(safe + 1), arguments[safe + 1] == "/dev/null",
              !arguments.contains("--jarvis-replay-run") else {
            throw PreferencesHostFailure("Require the safe DEBUG launch branch")
        }
        guard let path = ProcessInfo.processInfo.environment["JARVIS_CONNECTED_PREFERENCES_HOST_MANIFEST"] else {
            throw XCTSkip("Requires an explicitly allocated existing-host Preferences journey")
        }
        let argumentDefaults = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        guard arguments.filter({ $0 == "-jarvis.leanVoiceEnabled" }).count == 1,
              let lean = arguments.firstIndex(of: "-jarvis.leanVoiceEnabled"),
              arguments.indices.contains(lean + 1), arguments[lean + 1] == "YES",
              argumentDefaults["jarvis.leanVoiceEnabled"] as? String == "YES",
              UserDefaults.standard.bool(forKey: "jarvis.leanVoiceEnabled") else {
            throw PreferencesHostFailure("Require an explicit process-only Lean voice opt-in")
        }
        continueAfterFailure = false
        executionTimeAllowance = 240
        let manifest = try JSONDecoder().decode(PreferencesHostManifest.self, from: read(path))
        let ui = try JSONDecoder().decode(PreferencesHostUIManifest.self, from: read(manifest.uiManifest))
        let resource = try object(read(manifest.resourceManifest))
        let loaded = try JarvisReplayLaunchPlan.load(URL(fileURLWithPath: manifest.replayManifest))
        guard manifest.schemaVersion == 1, ui.schemaVersion == 1, manifest.runID == ui.runID,
              loaded.plan.runID == manifest.runID, loaded.scope.ownerID == manifest.ownerID,
              loaded.route == .realtime, loaded.taskBarrier == nil, loaded.memoryReadiness == nil,
              !loaded.manualInput, loaded.plan.utterances.isEmpty,
              loaded.plan.frames.count == 1, loaded.plan.duration == 180,
              loaded.plan.postMediaCaptureSeconds == 0,
              resource["format"] as? String == "jarvis.frontend_host.resources.v1",
              resource["run_id"] as? String == JarvisReplayScope.wireRunID(manifest.runID),
              resource["owner_id"] as? String == manifest.ownerID,
              resource["area_id"] as? String == loaded.scope.areaID,
              resource["memory_bank_id"] as? String == loaded.scope.memoryBankID,
              resource["status"] as? String == "serving",
              resource["normal_lifespan_qualified"] as? Bool == true,
              resource["capability_profile"] as? String == "native_realtime_media",
              let options = resource["native_options"] as? [String: Any],
              options["voice_route"] as? String == "realtime", options["workload"] as? String == "none",
              let base = resource["backend_base_url"] as? String,
              URL(string: base) == loaded.configuration.backendBaseURL,
              loaded.configuration.backendBaseURL.scheme == "http",
              loaded.configuration.backendBaseURL.host == "127.0.0.1",
              let credential = resource["bearer_token_file"] as? String else {
            throw PreferencesHostFailure("Mismatched host, replay, owner or idle-input scope")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: credential)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600,
              String(data: try read(credential, limit: 4096), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) == loaded.configuration.staticBearerToken else {
            throw PreferencesHostFailure("Isolated host credential mismatch")
        }
        let phone = URL(fileURLWithPath: ui.stagePath).deletingLastPathComponent()
        let directory = URL(fileURLWithPath: manifest.outputDirectory)
        guard phone.lastPathComponent == "phone-preferences-" + manifest.runID.uuidString,
              directory == phone.appendingPathComponent("native-host"),
              !FileManager.default.fileExists(atPath: directory.path),
              !FileManager.default.fileExists(atPath: ui.stagePath),
              ui.notBefore <= Date().timeIntervalSince1970,
              Date().timeIntervalSince1970 - ui.notBefore < 180,
              Set([ui.draft, ui.pushedWhileDirty, ui.pushedWhileClean]).count == 3 else {
            throw PreferencesHostFailure("Fresh run-bound controller/native evidence required")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        output = directory
        try write(["run_id": manifest.runID.uuidString, "host_manifest_sha256": hash(try read(path)),
                   "replay_manifest_sha256": loaded.manifestSHA256,
                   "host_pid": ProcessInfo.processInfo.processIdentifier], "inputs.json")
        let name = "jarvis.connected-preferences." + manifest.runID.uuidString
        defaultsName = name
        let store = JarvisSessionStore(defaults: try XCTUnwrap(UserDefaults(suiteName: name)),
                                       keychain: JarvisKeychain(service: name))
        self.store = store
        let product = JarvisAppViewModel(store: store, audioSessionViewModel: JarvisAudioSessionViewModel(),
            workspaceJournalFactory: { [weak self] context in
                guard context.hasValidIdentity, context.ownerID == manifest.ownerID else {
                    throw PreferencesHostFailure("Normal authenticated context returned the wrong owner")
                }
                self?.verifiedContext = context
                return try JarvisWorkspaceLocalStore(ownerKey: JarvisWorkspaceLocalStore.ownerKey(for: context),
                    root: directory.appendingPathComponent("journal"), allowsFileBytes: false)
            })
        app = product
        try product.bindProductEvaluationRun(loaded.plan.runID)
        product.saveSetup(backendBaseURL: loaded.configuration.backendBaseURL.absoluteString,
                          staticBearerToken: loaded.configuration.staticBearerToken)
        guard product.configuration == loaded.configuration else {
            throw PreferencesHostFailure("Isolated product configuration was not accepted")
        }
        try mount(product)
        let audio = product.audioSessionViewModel
        try await audio.startReplay(plan: loaded.plan, route: loaded.route,
            evaluationConfiguration: loaded.configuration, expectedScope: loaded.scope,
            taskConfiguration: loaded.taskBarrier, launchManifestSHA256: loaded.manifestSHA256,
            manualInput: loaded.manualInput, memoryConfiguration: loaded.memoryReadiness)
        let readyDeadline = Date().addingTimeInterval(40)
        while Date() < readyDeadline {
            if try readiness(product, manifest: manifest).values.allSatisfy({ $0 }) { break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        let readinessChecks = try readiness(product, manifest: manifest)
        guard readinessChecks.values.allSatisfy({ $0 }) else {
            let missing = readinessChecks.filter { !$0.value }.keys.sorted().joined(separator: ", ")
            throw PreferencesHostFailure("Native readiness failed: " + missing)
        }
        let sessionID = try XCTUnwrap((audio.activeSession?.id).flatMap(UUID.init(uuidString:)))
        let audioID = try XCTUnwrap(audio.audioSessionId.flatMap(UUID.init(uuidString:)))
        let identity: [String: Any] = ["run_id": manifest.runID.uuidString,
            "wire_run_id": JarvisReplayScope.wireRunID(manifest.runID), "owner_id": manifest.ownerID,
            "session_id": sessionID.uuidString.lowercased(), "audio_session_id": audioID.uuidString.lowercased(),
            "host_pid": ProcessInfo.processInfo.processIdentifier, "at": Date().timeIntervalSince1970,
            "replay_manifest_sha256": loaded.manifestSHA256,
            // Required configuration only. Core must also match the actual session.created
            // conversation_mode to these session/audio identities before driving the UI.
            "required_conversation_mode": "lean_v1",
            "verified_backend_namespace": verifiedContext!.backendNamespace]
        try observe(product, ui: ui)
        try write(identity, "ready.json")

        let journeyDeadline = Date().addingTimeInterval(175)
        var completed = false
        while Date() < journeyDeadline {
            try observe(product, ui: ui)
            let finalURL = phone.appendingPathComponent("native-ui-driver.json")
            if FileManager.default.fileExists(atPath: finalURL.path) {
                let final = try object(read(finalURL.path))
                guard final["run_id"] as? String == manifest.runID.uuidString,
                      final["completed_ui_stages"] as? Bool == true else {
                    throw PreferencesHostFailure("Native UI operator did not complete")
                }
                completed = true
                break
            }
            guard audio.replayRunID == manifest.runID, audio.isConnected,
                  (audio.activeSession?.id).flatMap(UUID.init(uuidString:)) == sessionID,
                  audio.audioSessionId.flatMap(UUID.init(uuidString:)) == audioID else {
                throw PreferencesHostFailure("The allocated native session ended or changed before the UI journey")
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        guard completed,
              observedStates.isSuperset(of: ["typed-draft", "dirty-push-preserved", "saved-draft", "clean-push"]),
              product.preferences?.savedProfile?.revision == 3,
              product.preferences?.draft == ui.pushedWhileClean else {
            throw PreferencesHostFailure("Missing actual product draft/push/save observations")
        }
        try write(identity.merging(["completed_model_observations": true,
            "observed_states": observedStates.sorted(), "observations": observations]) { _, new in new },
                  "native-product-result.json")
        try await stopOwnedReplay()
#else
        throw PreferencesHostFailure("This bridge requires a DEBUG simulator host")
#endif
    }

    private func readiness(_ product: JarvisAppViewModel, manifest: PreferencesHostManifest) throws -> [String: Bool] {
#if DEBUG
        let audio = product.audioSessionViewModel
        let sessionID = (audio.activeSession?.id).flatMap(UUID.init(uuidString:))
        let checks = [
            "audio_connected": audio.isConnected,
            "session_identity": sessionID != nil,
            "audio_identity": audio.audioSessionId.flatMap(UUID.init(uuidString:)) != nil,
            "replay_run": audio.replayRunID == manifest.runID,
            "verified_owner": verifiedContext?.ownerID == manifest.ownerID,
            "verified_session": sessionID != nil && verifiedContext?.sessionID == sessionID,
            "workspace_session": sessionID != nil && product.workspace?.sessionID == sessionID,
            "journal_ready": product.workspace?.journalReady == true,
            "profile_owner": product.preferences?.savedProfile?.ownerID == manifest.ownerID,
            "profile_revision_zero": product.preferences?.savedProfile?.revision == 0,
            "profile_draft_empty": product.preferences?.draft == ""
        ]
        var snapshot: [String: Any] = ["run_id": manifest.runID.uuidString, "checks": checks,
            "session_id": audio.activeSession?.id as Any? ?? NSNull(),
            "audio_session_id": audio.audioSessionId as Any? ?? NSNull(),
            "verified_owner": verifiedContext?.ownerID as Any? ?? NSNull(),
            "profile_revision": product.preferences?.savedProfile?.revision as Any? ?? NSNull(),
            "profile_error": product.preferences?.errorMessage as Any? ?? NSNull(),
            "profile_locked": product.preferences?.isLocked as Any? ?? NSNull()]
        let key = try JSONSerialization.data(withJSONObject: snapshot, options: [.sortedKeys])
        if key != lastReadinessSnapshot {
            lastReadinessSnapshot = key
            snapshot["at"] = Date().timeIntervalSince1970
            try write(snapshot, "readiness.json")
        }
        return checks
#else
        throw PreferencesHostFailure("DEBUG readiness required")
#endif
    }

    private func observe(_ product: JarvisAppViewModel, ui: PreferencesHostUIManifest) throws {
        guard let model = product.preferences, let saved = model.savedProfile else { return }
        if saved.revision == 0, model.draft == ui.draft, model.isDirty { observedStates.insert("typed-draft") }
        if saved.revision == 1, saved.behaviorInstructions == ui.pushedWhileDirty,
           model.draft == ui.draft, model.baseRevision == 0, model.needsReview, !model.canSave {
            observedStates.insert("dirty-push-preserved")
        }
        if saved.revision == 2, saved.behaviorInstructions == ui.draft, model.draft == ui.draft,
           model.baseRevision == 2, !model.isDirty, !model.needsReview { observedStates.insert("saved-draft") }
        if saved.revision == 3, saved.behaviorInstructions == ui.pushedWhileClean,
           model.draft == ui.pushedWhileClean, !model.isDirty, !model.needsReview { observedStates.insert("clean-push") }
        var snapshot: [String: Any] = ["revision": saved.revision, "owner_id": saved.ownerID,
            "saved": saved.behaviorInstructions, "draft": model.draft,
            "base_revision": model.baseRevision as Any? ?? NSNull(), "needs_review": model.needsReview,
            "can_save": model.canSave, "is_saving": model.isSaving]
        let key = try JSONSerialization.data(withJSONObject: snapshot, options: [.sortedKeys])
        guard key != lastSnapshot else { return }
        guard observations.count < 256 else { throw PreferencesHostFailure("Product observation bound exceeded") }
        lastSnapshot = key
        snapshot["at"] = Date().timeIntervalSince1970
        observations.append(snapshot)
        try write(["observations": observations, "observed_states": observedStates.sorted()], "observations.json")
    }

    private func mount(_ product: JarvisAppViewModel) throws {
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive })
        previousWindow = scene.windows.first { $0.isKeyWindow }
        let window = UIWindow(windowScene: scene)
        window.rootViewController = UIHostingController(rootView:
            JarvisRootView(viewModel: product).environment(\.scenePhase, .active))
        self.window = window
        window.makeKeyAndVisible()
    }

    private func stopOwnedReplay() async throws {
#if DEBUG
        guard let audio = app?.audioSessionViewModel else { return }
        if audio.shouldShowStopJarvis || audio.replayRunID != nil { audio.stopJarvis() }
        let deadline = Date().addingTimeInterval(20)
        while audio.shouldShowStopJarvis && Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        if let evidence = audio.lastReplayAudioEvidence {
            let report = JarvisReplayEvidenceReport.make(evidence: evidence,
                runID: try XCTUnwrap(output.flatMap { UUID(uuidString: $0.deletingLastPathComponent()
                    .lastPathComponent.replacingOccurrences(of: "phone-preferences-", with: "")) }),
                finalUIStatus: audio.voiceStatusText)
            try write(report, "replay-capture.json")
        }
        guard audio.replayRunID == nil, !audio.shouldShowStopJarvis, audio.activeSession == nil else {
            throw PreferencesHostFailure("Owned native replay did not finish cleanup")
        }
        try write(["native_replay_stopped": true, "at": Date().timeIntervalSince1970], "native-stop.json")
#endif
    }

    override func tearDown() async throws {
        defer {
            window?.isHidden = true; window?.rootViewController = nil; window = nil
            previousWindow?.makeKeyAndVisible(); previousWindow = nil
            store?.clearConfiguration(); store = nil
            if let defaultsName { UserDefaults.standard.removePersistentDomain(forName: defaultsName) }
            defaultsName = nil; app = nil
        }
        try await stopOwnedReplay()
        try await super.tearDown()
    }

    private func read(_ path: String, limit: Int = 512 * 1024) throws -> Data {
        guard path.hasPrefix("/"),
              ((try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.intValue ?? Int.max) <= limit else {
            throw PreferencesHostFailure("Expected a bounded absolute local file")
        }
        return try Data(contentsOf: URL(fileURLWithPath: path))
    }
    private func object(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
    private func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func write(_ value: [String: Any], _ name: String) throws {
        guard let output else { return }
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent(name), options: .atomic)
    }
}

private struct PreferencesHostManifest: Decodable {
    let schemaVersion: Int
    let runID: UUID
    let ownerID: String
    let resourceManifest: String
    let replayManifest: String
    let uiManifest: String
    let outputDirectory: String
    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", runID = "run_id", ownerID = "owner_id"
        case resourceManifest = "resource_manifest", replayManifest = "replay_manifest"
        case uiManifest = "ui_manifest", outputDirectory = "output_directory"
    }
}
private struct PreferencesHostUIManifest: Decodable {
    let schemaVersion: Int
    let runID: UUID
    let stagePath: String
    let notBefore: Double
    let draft: String
    let pushedWhileDirty: String
    let pushedWhileClean: String
    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", runID = "run_id", stagePath = "stage_path", notBefore = "not_before", draft
        case pushedWhileDirty = "pushed_while_dirty", pushedWhileClean = "pushed_while_clean"
    }
}
private struct PreferencesHostFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
