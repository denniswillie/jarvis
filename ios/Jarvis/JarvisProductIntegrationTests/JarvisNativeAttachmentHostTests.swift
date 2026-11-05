import SwiftUI
import UIKit
import XCTest

/// Fixture-free platform check: can a separate UI runner attach to a hosted
/// XCTest window while this runner remains alive? These are probe tabs, not
/// product/content qualification. No Jarvis model, HTTP, credentials or audio.
@MainActor
final class JarvisNativeAttachmentHostTests: XCTestCase {
    func testHostedWindowAcceptsSeparateNativeDriver() async throws {
#if !targetEnvironment(simulator)
        throw AttachmentHostFailure("Simulator only")
#else
        let args = ProcessInfo.processInfo.arguments
        guard let safe = args.firstIndex(of: "--jarvis-replay-manifest"),
              args.indices.contains(safe + 1), args[safe + 1] == "/dev/null",
              !args.contains("--jarvis-replay-run") else {
            throw AttachmentHostFailure("Require safe DEBUG root without replay execution")
        }
        guard let path = ProcessInfo.processInfo.environment["JARVIS_NATIVE_ATTACH_MANIFEST"] else {
            throw XCTSkip("Run only with the allocated attachment manifest")
        }
        continueAfterFailure = false
        executionTimeAllowance = 60
        let manifest = try JSONDecoder().decode(AttachmentHostManifest.self,
            from: Data(contentsOf: URL(fileURLWithPath: path)))
        let directory = URL(fileURLWithPath: manifest.directory)
        guard manifest.directory.hasPrefix("/"), directory.lastPathComponent == "native-attach-" + manifest.runID.uuidString,
              FileManager.default.fileExists(atPath: directory.path),
              !FileManager.default.fileExists(atPath: directory.appendingPathComponent("stage.json").path) else {
            throw AttachmentHostFailure("Require a fresh run-bound evidence directory")
        }
        let deadline = Date().addingTimeInterval(50)
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive })
        let prior = scene.windows.first { $0.isKeyWindow }
        let state = AttachmentProbeState()
        let window = UIWindow(windowScene: scene)
        window.rootViewController = UIHostingController(rootView: AttachmentProbeView(state: state, runID: manifest.runID))
        window.makeKeyAndVisible()
        defer {
            window.isHidden = true
            window.rootViewController = nil
            prior?.makeKeyAndVisible()
        }
        try stage("ready", manifest, directory)
        for (selection, phase) in [(1, "note-observed"), (2, "workspace-observed"), (0, "home-observed")] {
            while state.selection != selection && Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            guard state.selection == selection else { throw AttachmentHostFailure("No real UI selection for " + phase) }
            let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
            }
            let attachment = XCTAttachment(image: image)
            attachment.name = phase
            attachment.lifetime = .keepAlways
            add(attachment)
            try stage(phase, manifest, directory)
        }
        while Date() < deadline {
            if let data = try? Data(contentsOf: directory.appendingPathComponent("driver-done.json")),
               let done = try? JSONDecoder().decode(AttachmentHostControl.self, from: data),
               done.runID == manifest.runID && done.phase == "driver-completed" {
                return
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw AttachmentHostFailure("Native driver did not finish its observed Home verification")
#endif
    }

    private func stage(_ phase: String, _ manifest: AttachmentHostManifest, _ directory: URL) throws {
        let value: [String: Any] = ["run_id": manifest.runID.uuidString, "phase": phase,
            "at": Date().timeIntervalSince1970, "host_pid": ProcessInfo.processInfo.processIdentifier]
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("stage.json"), options: .atomic)
    }
}

@MainActor
private final class AttachmentProbeState: ObservableObject {
    @Published var selection = 0
}

private struct AttachmentProbeView: View {
    @ObservedObject var state: AttachmentProbeState
    let runID: UUID
    var body: some View {
        VStack {
            Text(runID.uuidString).accessibilityIdentifier("nativeAttachmentRun")
            TabView(selection: $state.selection) {
                Text("Attachment Home").tabItem { Label("Home", systemImage: "house") }.tag(0)
                Text("Attachment Note").tabItem { Label("Note", systemImage: "note.text") }.tag(1)
                Text("Attachment Workspace").tabItem { Label("Workspace", systemImage: "rectangle.on.rectangle") }.tag(2)
            }
        }
    }
}

private struct AttachmentHostManifest: Decodable {
    let runID: UUID
    let directory: String
    enum CodingKeys: String, CodingKey { case runID = "run_id", directory }
}
private struct AttachmentHostControl: Decodable {
    let runID: UUID
    let phase: String
    enum CodingKeys: String, CodingKey { case runID = "run_id", phase }
}
private struct AttachmentHostFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
