@preconcurrency import AVFoundation
import Foundation

/// Exercises actual writer timeline/accounting and descriptor validation with
/// an injected encoder. These bytes are not media or playback evidence.
@main struct JarvisRecordingDurationChecks {
    final class Encoder: JarvisRecordingEncoding {
        let output: URL
        var completedSuccessfully = false
        init(output: URL) { self.output = output }
        func isReady(for track: JarvisRecordingWriter.Track) -> Bool { true }
        func appendVideo(_ sample: CMSampleBuffer) throws { }
        func appendAudio(_ samples: [Float], at time: CMTime) throws { }
        func finish(at time: CMTime, completion: @escaping @Sendable (Bool) -> Void) {
            do {
                try Data("Injected encoder; not a movie".utf8).write(to: output)
                completedSuccessfully = true; completion(true)
            } catch { completion(false) }
        }
        func cancel() { }
    }

    static func main() async throws {
        var config = JarvisRecordingWriter.Configuration(width: 160, height: 96)
        try config.validate()
        config.maximumDuration = CMTime(value: 1_800_001, timescale: 1000)
        do { try config.validate(); fatalError("More than1800s accepted") }
        catch JarvisRecordingWriter.Reason.invalidConfiguration { }
        config.maximumDuration = CMTime(value: 1800, timescale: 1)

        for includeSource in [false, true] {
            let directory = try JarvisRecordingDirectory()
            let queue = DispatchQueue(label: "duration-accounting")
            let generation = UUID(), recording = UUID()
            let writer = JarvisRecordingWriter(configuration: config, generationID: generation,
                queue: queue, encoder: Encoder(output: directory.file), directory: directory)
            let firstFrame = try video()
            precondition(writer.offerVideo(firstFrame, generationID: generation) == .queued)
            queue.sync { }
            if includeSource {
                let samples = [Float](repeating: 0.125, count: 4800)
                for part in 0..<18000 {
                    precondition(writer.offerAudio(samples, at: CMTime(value: Int64(part * 4800), timescale: 48_000),
                        generationID: generation) == .queued)
                    queue.sync { }
                }
            } else {
                // The writer correctly requires real admitted audio before it
                // pads an explicit remaining timeline gap.
                precondition(writer.offerAudio([Float](repeating: 0.125, count: 4800), at: .zero,
                    generationID: generation) == .queued)
                queue.sync { }
            }
            let completed = try await writer.finish(at: CMTime(value: 1800, timescale: 1), generationID: generation)
            precondition(completed.evidence.sourceAudioFrames == (includeSource ? 86_400_000 : 4800))
            precondition(completed.evidence.paddedAudioFrames == (includeSource ? 0 : 86_395_200))
            let command = try JarvisRecordingControl.Command(values: [
                "type": "recording.command", "version": 1, "action": "start",
                "operation_id": recording.uuidString, "recording_id": recording.uuidString,
                "session_id": UUID().uuidString, "generation_id": generation.uuidString,
                "audio_session_id": "duration-fixture", "include_jarvis_audio": true,
                "max_bytes": 1024 * 1024 * 1024, "max_duration_ms": 1_800_000])
            let binding = try JarvisRecordingManifest.Binding(command: command)
            let descriptor = try JarvisRecordingManifest.make(binding: binding, completed: completed,
                expectedWriterGenerationID: generation)
            let upload = try JarvisRecordingManifest.decode(descriptor, binding: binding)
            precondition(upload.clientReport.timelineEndMS == 1_800_000)
            precondition(upload.clientReport.sourceAudioFrames + upload.clientReport.paddedAudioFrames == 86_400_000)
            var invalid = completed.evidence
            if includeSource { invalid.sourceAudioFrames = 86_400_001 } else { invalid.paddedAudioFrames = 86_400_001 }
            let invalidCompleted = JarvisRecordingWriter.Completed(generationID: generation,
                file: completed.file, requestedEnd: completed.requestedEnd, evidence: invalid)
            do {
                _ = try JarvisRecordingManifest.make(binding: binding, completed: invalidCompleted,
                    expectedWriterGenerationID: generation)
                fatalError("Audio beyond1800s accepted")
            } catch JarvisRecordingManifest.Failure.invalidCompleted { }
            print("PASS1800s actual writer accounting and manifest: source=\(includeSource),86.4Mframes; over-bound rejected")
        }
    }

    static func video() throws -> CMSampleBuffer {
        var pixels: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, 160, 96, kCVPixelFormatType_32BGRA, nil, &pixels) == kCVReturnSuccess,
              let pixels else { throw JarvisRecordingWriter.Reason.invalidFormat }
        var format: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixels,
            formatDescriptionOut: &format) == noErr, let format else { throw JarvisRecordingWriter.Reason.invalidFormat }
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 24),
            presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixels,
            formatDescription: format, sampleTiming: &timing, sampleBufferOut: &sample) == noErr,
              let sample else { throw JarvisRecordingWriter.Reason.invalidFormat }
        return sample
    }
}
