import Foundation

#if os(iOS)
// No live audio route or microphone configuration is changed.
final class JarvisNeuralRecordingEchoFilter {
    private let filter: JRNeuralRecordingFilter
    private let diagnosticID: String
    private let startedAt: Date
    private let startedUptime: TimeInterval
    private var nextSnapshotUptime: TimeInterval
    private var checkpoints: [[String: Any]] = []
    private var omittedCheckpoints = 0
    #if DEBUG
    // Explicitly opted-in, bounded capture of the exact neural inputs, not a new audio route.
    private let inputDiagnostic = JarvisRecordingEchoDiagnostic.configured(
        directoryName: "JarvisNeuralEchoDiagnostic",
        channelNames: ["microphone_input", "neural_filter_playback_reference", "microphone_unfiltered_copy", "selected_digital_mix"])
    #endif
    private static let diagnosticQueue = DispatchQueue(label: "jarvis.recording.echo-diagnostics", qos: .utility)

    init?(output: @escaping (JarvisRecordingMixer.Chunk) -> Void) {
        let id = UUID().uuidString
        let now = Date()
        guard let resourceURL = Bundle.main.resourceURL else {
            Self.save(["diagnostic_id": id, "phase": "unavailable", "reason": "bundle_resources_missing"])
            return nil
        }
        let path = resourceURL.appendingPathComponent("NeuralAEC").path
        guard let filter = JRNeuralRecordingFilter(modelsPath: path, output: { start, data in
            let samples = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            output(.init(startFrame: start, samples: samples))
        }) else {
            Self.save(["diagnostic_id": id, "phase": "unavailable", "reason": "neural_models_unavailable"])
            return nil
        }
        self.filter = filter
        diagnosticID = id
        startedAt = now
        startedUptime = ProcessInfo.processInfo.systemUptime
        nextSnapshotUptime = startedUptime + 5
        NSLog("Recording neural filter enabled; live audio route unchanged")
        snapshot(phase: "enabled")
    }

    func offer(startFrame: Int64, microphone: [Float], reference: [Float], direct: [Float]) {
        #if DEBUG
        inputDiagnostic?.append(startFrame: startFrame, microphone: microphone,
                                reference: reference, cleaned: microphone, direct: direct,
                                count: microphone.count)
        #endif
        filter.offer(atFrame: startFrame,
                     microphone: microphone.withUnsafeBytes { Data($0) },
                     reference: reference.withUnsafeBytes { Data($0) },
                     direct: direct.withUnsafeBytes { Data($0) })
        let now = ProcessInfo.processInfo.systemUptime
        if now >= nextSnapshotUptime {
            nextSnapshotUptime = now + 5
            snapshot(phase: "recording")
        }
    }
    func finish() { filter.finish() }
    func reset() { filter.reset(); snapshot(phase: "segment_reset") }
    func finishDiagnostic() {
        #if DEBUG
        inputDiagnostic?.finish()
        #endif
        NSLog("Recording neural filter metrics: %@", filter.metrics)
        snapshot(phase: "finished")
    }

    private func snapshot(phase: String) {
        let metrics = filter.metrics
        if checkpoints.count >= 128 {
            checkpoints.remove(at: 1)
            omittedCheckpoints += 1
        }
        checkpoints.append(["elapsed_seconds": ProcessInfo.processInfo.systemUptime - startedUptime,
                            "phase": phase, "metrics": metrics])
        var report: [String: Any] = [
            "schema_version": 1, "filter_revision": "neural-alignment-v1-diagnostics",
            "diagnostic_id": diagnosticID, "started_at": ISO8601DateFormatter().string(from: startedAt),
            "phase": phase, "metrics": metrics, "checkpoints": checkpoints,
            "omitted_checkpoints": omittedCheckpoints,
            "app_build": Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        ]
        if let error = filter.lastError { report["failure_reason"] = error }
        Self.save(report)
    }

    private static func save(_ report: [String: Any]) {
        var value = report
        value["captured_at"] = ISO8601DateFormatter().string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
              let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else { return }
        let url = directory.appendingPathComponent("JarvisRecordingEchoDiagnostic.json")
        // One bounded, replaceable report. No raw audio and no disk I/O on the recording queue.
        diagnosticQueue.async {
            do { try data.write(to: url, options: .atomic) }
            catch { NSLog("Recording echo diagnostic save failed: %@", String(describing: error)) }
        }
    }
}

#endif
