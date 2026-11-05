import Foundation

typealias JarvisRecordingSession = JarvisRecordingCoordinator<JarvisRecordingCapture.Completed>

enum JarvisRecordingSessionError: Error {
    case audioGenerationMismatch
}

extension JarvisRecordingCoordinator where Completed == JarvisRecordingCapture.Completed {
    /// Uses the already-running native audio device and DAT stream. The caller
    /// must supply durable retention before advertising a recording capability.
    /// No engine, microphone permission, voice turn or camera session is started.
    static func native(context: Context, current: @escaping @MainActor () -> Bool,
                       audio: JarvisDuplexAudioDevice, video: JarvisRecordingVideoSource,
                       prepare: @escaping @MainActor (Command) async throws -> JarvisRecordingWriter.Output,
                       retain: @escaping Retain,
                       write: @escaping @MainActor (JarvisRecordingControl.Outbound, Context) async throws -> Void,
                       changed: @escaping @MainActor (UUID, State) -> Void = { _, _ in },
                       issue: @escaping @MainActor (UUID, Issue) -> Void = { _, _ in }) throws -> JarvisRecordingSession {
        guard audio.voiceGeneration == context.generationID else {
            throw JarvisRecordingSessionError.audioGenerationMismatch
        }
        let source = JarvisRecordingAudioSource(device: audio)
        return JarvisRecordingSession(context: context, current: current, start: { command, stopped in
            // Durable reservation and its bound must succeed before either
            // source is attached. Failure cannot create an untracked capture.
            do {
                let output = try await prepare(command)
                let destination: JarvisRecordingWriter.Destination?
                let streaming: JarvisRecordingWriter.Streaming?
                switch output {
                case .local(let value): destination = value; streaming = nil
                case .stream(let value): destination = nil; streaming = value
                }
                if !command.includeJarvisAudio {
                    audio.setRecordingOutputSuppressed(true, recordingID: command.recordingID)
                }
                if command.recordingMode == "voiceover" {
                    audio.setMediaOutputSuppressed(true, recordingID: command.recordingID)
                }
                let capture = try await JarvisRecordingCapture.start(audio: source, video: video,
                    includeJarvisAudio: command.includeJarvisAudio,
                    includeMediaAudio: command.includeMediaAudio,
                    maximumDurationMilliseconds: command.maxDurationMilliseconds,
                    maxBytes: command.maxBytes, destination: destination, streaming: streaming, onStopped: { _ in stopped() })
                try await capture.waitUntilRecording()
                return Capture(captureID: capture.captureID, finish: {
                    defer {
                        audio.setRecordingOutputSuppressed(false, recordingID: command.recordingID)
                        audio.setMediaOutputSuppressed(false, recordingID: command.recordingID)
                    }
                    do { return try await capture.finish() }
                    catch {
                        persistRecordingFailure(error, phase: "capture_finish", recordingID: command.recordingID)
                        throw error
                    }
                })
            } catch {
                audio.setRecordingOutputSuppressed(false, recordingID: command.recordingID)
                audio.setMediaOutputSuppressed(false, recordingID: command.recordingID)
                persistRecordingFailure(error, phase: "capture_start", recordingID: command.recordingID)
                throw error
            }
        }, retain: retain, write: write, changed: changed, issue: issue)
    }
}

/// Preserve the real failure after relaunch, without recording media or credentials.
private func persistRecordingFailure(_ error: Error, phase: String, recordingID: UUID) {
    let ns = error as NSError
    var fields: [String: Any] = ["phase": phase, "recording_id": recordingID.uuidString,
        "recorded_at": ISO8601DateFormatter().string(from: Date()),
        "error_domain": ns.domain, "error_code": ns.code]
    if let failure = error as? JarvisRecordingCapture.Failure {
        fields["capture_failure"] = String(describing: failure.code)
        fields["coverage"] = String(describing: failure.coverage)
        fields["writer_evidence"] = String(describing: failure.writerEvidence)
    } else if let failure = error as? JarvisRecordingWriter.Failure {
        fields["writer_failure"] = String(describing: failure.reason)
    }
    guard let data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]) else { return }
    print("JarvisRecordingFailure " + String(decoding: data, as: UTF8.self))
    if let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
        try? data.write(to: cache.appendingPathComponent("JarvisRecordingFailure.json"), options: .atomic)
    }
}
