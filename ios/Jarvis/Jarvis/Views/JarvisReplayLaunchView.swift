#if DEBUG
import SwiftUI

/// Separate debug root: no normal bootstrap, saved-session restore, memory
/// refresh, notifications, glasses startup or settings writes.
struct JarvisReplayLaunchView: View {
    @StateObject private var audio = JarvisAudioSessionViewModel()
    @State private var status = "Validating local replay assets. Network disabled."
    @State private var loaded: JarvisReplayLaunchPlan.Loaded?
    @State private var didLoad = false
    @State private var wasStarted = false
    @State private var artifactsWritten = false

    var body: some View {
        VStack(spacing: 18) {
            Text("Jarvis evaluation replay").font(.title)
            Text("Prerecorded inputs. Silent decoded-PCM output. Not live glasses or hearing evidence.")
            Text(status)
            Text(audio.voiceStatusText)
            if wasStarted { Button("Stop evaluation") { audio.stopJarvis() } }
        }
        .padding()
        .task {
            guard !didLoad else { return }
            didLoad = true
            do {
                let args = ProcessInfo.processInfo.arguments
                guard let index = args.firstIndex(of: "--jarvis-replay-manifest"), index + 1 < args.count else {
                    throw JarvisReplayError.invalidPlan
                }
                let loaded = try JarvisReplayLaunchPlan.load(URL(fileURLWithPath: args[index + 1]))
                self.loaded = loaded
                status = "Assets validated. No session opened."
                guard args.contains("--jarvis-replay-run") else { return }
                status = "Waiting for authenticated isolated runtime scope."
                try await audio.startReplay(plan: loaded.plan, route: loaded.route,
                    evaluationConfiguration: loaded.configuration, expectedScope: loaded.scope,
                    taskConfiguration: loaded.taskBarrier, launchManifestSHA256: loaded.manifestSHA256,
                    manualInput: loaded.manualInput, memoryConfiguration: loaded.memoryReadiness)
                wasStarted = true
                status = "Evaluation session starting; no physical audio output."
            } catch {
                status = "Replay refused: asset, isolation or startup validation failed."
            }
        }
        .onChange(of: audio.replayRunID) { old, new in
            if old != nil, new == nil { writeArtifacts() }
        }
    }

    private func writeArtifacts() {
        guard !artifactsWritten, let loaded, let evidence = audio.lastReplayAudioEvidence else { return }
        do {
            let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true)
            let root = loaded.plan.captureRoot(documents: documents)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try evidence.decodedPCM16.write(to: root.appendingPathComponent("decoded.pcm"), options: .atomic)
            var report = JarvisReplayEvidenceReport.make(evidence: evidence, runID: loaded.plan.runID,
                finalUIStatus: audio.voiceStatusText)
            report["voice_route_status"] = audio.voiceStatusEvidence
            if let lineage = audio.lastReplayFrameLineage { report["visual_frame_lineage"] = lineage }
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("capture.json"), options: .atomic)
            artifactsWritten = true
            status = "Local capture exported to its invocation-bound Replay directory."
        } catch { status = "Replay ended; local artifact export failed." }
    }
}
#endif
