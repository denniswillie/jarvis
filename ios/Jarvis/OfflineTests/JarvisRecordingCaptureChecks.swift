@preconcurrency import AVFoundation
import CoreAudio
import CoreVideo
import Foundation

private final class CaptureClock: @unchecked Sendable {
    let origin = CMClockMakeHostTimeFromSystemUnits(mach_absolute_time())
    private let lock = NSLock()
    private var offset: Int64 = 0
    func ticks(_ milliseconds: Int64) -> UInt64 {
        CMClockConvertHostTimeToSystemUnits(CMTimeAdd(origin, CMTime(value: milliseconds, timescale: 1_000)))
    }
    func now() -> UInt64 { lock.withLock { ticks(offset) } }
    func advance(_ ms: Int64) { lock.withLock { offset = ms } }
}
private final class StopEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [JarvisRecordingCapture.StopReason] = []
    func add(_ reason: JarvisRecordingCapture.StopReason) { lock.withLock { values.append(reason) } }
    var count: Int { lock.withLock { values.count } }
}
private final class CaptureAudioFake: @unchecked Sendable {
    let voiceGeneration = UUID()
    let queue = DispatchQueue(label: "capture-check.audio")
    let delay: Int
    let failsDetach: Bool
    private var id: UUID?
    private var sink: JarvisRecordingAudioSource.Sink?
    private var detachCount = 0
    private var voicePackets = 0
    init(delay: Int = 0, failsDetach: Bool = false) { self.delay = delay; self.failsDetach = failsDetach }
    var source: JarvisRecordingAudioSource {
        JarvisRecordingAudioSource(voiceGeneration: voiceGeneration, attach: { [self] id, sink, done in
            queue.asyncAfter(deadline: .now() + .milliseconds(delay)) { [self] in
                guard self.id == nil else { done(false); return }
                self.id = id; self.sink = sink
                sink(.format(format(rate: 48_000, route: 1)))
                sink(.coverage(.init(routeGeneration: 1, microphoneDroppedPackets: 0, microphoneDroppedFrames: 0,
                    outputDroppedPackets: 0, outputDroppedFrames: 0, microphoneErrors: 0, outputErrors: 0,
                    lastMicrophoneStatus: 0, lastOutputStatus: 0)))
                done(true)
            }
        }, detach: { [self] id, done in queue.async { [self] in
            detachCount += 1
            guard self.id == id else { done(false); return }
            sink?(.boundary(captureID: id, routeGeneration: 1,
                reason: failsDetach ? "capture_fence_failed" : "capture_detached", terminal: true, consumerFenced: !failsDetach))
            self.id = nil; sink = nil; done(!failsDetach)
        } })
    }
    func format(rate: Int, route: UInt64, voice: UUID? = nil) -> JarvisRecordingAudioSource.Format {
        .init(voiceGeneration: voice ?? voiceGeneration, routeGeneration: route,
              sampleRate: Double(rate), inputChannels: 1, outputChannels: 1)
    }
    func send(host: UInt64, sample: Double, rate: Int = 48_000, route: UInt64 = 1,
              output: Bool = false, frequency: Double = 440, voice: UUID? = nil,
              mediaSpans: [JarvisMediaCaptureSpan] = [], mediaCoverage: JarvisMediaCaptureCoverage = .init()) {
        queue.async { [self] in
            voicePackets += 1 // Independent conversation consumer continues.
            guard let id, let sink else { return }
            let frames = rate / 100
            let samples = (0..<frames).map { index in
                Int16((0.2 * sin(2 * .pi * frequency * (sample + Double(index)) / Double(rate)) * 32_767).rounded()).littleEndian
            }
            sink(.pcm(.init(captureID: id, format: format(rate: rate, route: route, voice: voice),
                track: output ? .renderedOutput : .microphone,
                pcm16Interleaved: samples.withUnsafeBytes { Data($0) }, frames: frames,
                hostTime: host, sampleTime: sample,
                timestampFlags: AudioTimeStampFlags.hostTimeValid.rawValue | AudioTimeStampFlags.sampleTimeValid.rawValue,
                softwareGateOpen: true, mediaSpans: mediaSpans, mediaCoverage: mediaCoverage)))
        }
    }
    func barrier() async { await withCheckedContinuation { c in queue.async { c.resume() } } }
    func state() async -> (active: Bool, detaches: Int, voicePackets: Int) {
        await withCheckedContinuation { c in queue.async { [self] in c.resume(returning: (id != nil, detachCount, voicePackets)) } }
    }
}

@main struct RecordingCaptureChecks {
    typealias Capture = JarvisRecordingCapture
    static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError(message) }
    }
    static func frame(_ time: CMTime, duration: CMTime = CMTime(value: 1, timescale: 24),
                      width: Int = 32, height: Int = 32) throws -> CMSampleBuffer {
        var pixel: CVPixelBuffer?
        require(CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixel) == kCVReturnSuccess,
                "Pixel allocation")
        CVPixelBufferLockBaseAddress(pixel!, [])
        memset(CVPixelBufferGetBaseAddress(pixel!)!, 100, CVPixelBufferGetDataSize(pixel!))
        CVPixelBufferUnlockBaseAddress(pixel!, [])
        var format: CMVideoFormatDescription?
        require(CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
            imageBuffer: pixel!, formatDescriptionOut: &format) == noErr, "Frame format")
        var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        require(CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixel!,
            formatDescription: format!, sampleTiming: &timing, sampleBufferOut: &sample) == noErr, "Frame sample")
        return sample!
    }
    private static func scene(_ clock: CaptureClock) throws -> (JarvisRecordingVideoSource, UUID) {
        let video = JarvisRecordingVideoSource(), generation = video.beginGeneration()
        _ = video.offer(try frame(clock.origin), sourceGeneration: generation, receivedHostTime: clock.ticks(1))
        return (video, generation)
    }
    private static func options(_ clock: CaptureClock) -> Capture.Options {
        var options = Capture.Options(); options.nowHost = { clock.now() }
        options.videoClock = .dat07PhoneHostUnqualified // These fixtures explicitly use phone-host PTS.
        options.startupTimeoutMilliseconds = 500; options.fenceTimeoutMilliseconds = 500
        return options
    }
    private static func addFrame(_ video: JarvisRecordingVideoSource, _ generation: UUID, _ clock: CaptureClock,
                         at ms: Int64, duration: CMTime = CMTime(value: 1, timescale: 24)) throws {
        _ = video.offer(try frame(CMTimeAdd(clock.origin, CMTime(value: ms, timescale: 1_000)), duration: duration),
                        sourceGeneration: generation, receivedHostTime: clock.ticks(ms + 20))
    }
    @MainActor static func expectFailure(_ code: Capture.Code, _ body: @MainActor () async throws -> Void) async {
        do { try await body(); fatalError("Expected \(code)") }
        catch let error as Capture.Failure { require(error.code == code, "Expected \(code), got \(error.code)") }
        catch { fatalError("Unexpected error \(error)") }
    }
    static func main() async throws {
        try await selectedMediaChecks()
        let clock = CaptureClock(), audio = CaptureAudioFake(), stopped = StopEvents()
        let (video, generation) = try scene(clock)
        let capture = try await Capture.start(audio: audio.source, video: video, includeJarvisAudio: false,
            maximumDurationMilliseconds: 1_000, onStopped: { stopped.add($0) }, options: options(clock))
        try addFrame(video, generation, clock, at: 100, duration: CMTime(value: 200, timescale: 1_000))
        audio.send(host: clock.ticks(100), sample: 4_800)
        audio.send(host: clock.ticks(100), sample: 4_800, output: true)
        await audio.barrier(); clock.advance(110)
        let done = try await capture.finish(), again = try await capture.finish()
        require(done.writer.file === again.writer.file, "Repeated stop did not reuse result")
        require(CMTimeCompare(done.times.mediaEnd, CMTime(value: 300, timescale: 1_000)) >= 0,
                "Stop compressed admitted video duration")
        require(done.coverage.audioFenced && done.coverage.videoFenced && stopped.count == 0, "Manual stop/fence receipt")
        require(done.mixerEvidence.renderedFrames == 0, "Excluded Jarvis audio leaked")
        let before = await audio.state()
        audio.send(host: clock.ticks(400), sample: 19_200); await audio.barrier()
        try addFrame(video, generation, clock, at: 400)
        let after = await audio.state()
        require(!after.active && after.voicePackets == before.voicePackets + 1 && video.latest?.sourceGeneration == generation,
                "Recording stop affected unrelated audio/video consumers")
        done.writer.file.discard()
        print("PASS 1: shared stop result, admitted-end containment, exclusion and continuing consumers")

        let lateClock = CaptureClock(), delayed = CaptureAudioFake(delay: 150)
        let (lateVideo, _) = try scene(lateClock)
        let pending = Task {
            try await Capture.start(audio: delayed.source, video: lateVideo, includeJarvisAudio: true,
                maximumDurationMilliseconds: 1_000, onStopped: { _ in }, options: options(lateClock))
        }
        try await Task.sleep(for: .milliseconds(30)); pending.cancel()
        do { _ = try await pending.value; fatalError("Cancelled startup succeeded") }
        catch is CancellationError {}
        catch let error as Capture.Failure { require(error.code == .cancelled, "Cancellation code") }
        let delayedState = await delayed.state()
        require(!delayedState.active && delayedState.detaches > 0, "Late cancelled attachment left a sink")
        print("PASS 2: cancellation during delayed startup detaches the eventual attachment")

        let timeoutClock = CaptureClock(), timeoutAudio = CaptureAudioFake(delay: 180)
        let (timeoutVideo, _) = try scene(timeoutClock)
        var short = options(timeoutClock); short.startupTimeoutMilliseconds = 15; short.fenceTimeoutMilliseconds = 20
        await expectFailure(.fenceTimeout) {
            _ = try await Capture.start(audio: timeoutAudio.source, video: timeoutVideo, includeJarvisAudio: false,
                maximumDurationMilliseconds: 1_000, onStopped: { _ in }, options: short)
        }
        try await Task.sleep(for: .milliseconds(250))
        let timeoutState = await timeoutAudio.state()
        require(!timeoutState.active && timeoutState.detaches >= 2, "Post-timeout late attachment was not cleaned up")
        print("PASS 3: bounded startup/fence timeout retains cleanup for a still-later attachment")

        let failureClock = CaptureClock(), badDetach = CaptureAudioFake(failsDetach: true)
        let (failureVideo, _) = try scene(failureClock)
        let failed = try await Capture.start(audio: badDetach.source, video: failureVideo, includeJarvisAudio: false,
            maximumDurationMilliseconds: 1_000, onStopped: { _ in }, options: options(failureClock))
        await expectFailure(.audioFenceFailed) { _ = try await failed.finish() }
        await expectFailure(.audioFenceFailed) { _ = try await failed.finish() }
        print("PASS 4: failed fence cannot become a successful file receipt or a different retry result")

        let replacementClock = CaptureClock(), replacementAudio = CaptureAudioFake(), replacementEvents = StopEvents()
        let (replacementVideo, oldGeneration) = try scene(replacementClock)
        let replacement = try await Capture.start(audio: replacementAudio.source, video: replacementVideo,
            includeJarvisAudio: false, maximumDurationMilliseconds: 1_000,
            onStopped: { replacementEvents.add($0) }, options: options(replacementClock))
        try addFrame(replacementVideo, oldGeneration, replacementClock, at: 0)
        replacementAudio.send(host: replacementClock.ticks(0), sample: 0); await replacementAudio.barrier()
        replacementClock.advance(100); let newGeneration = replacementVideo.beginGeneration()
        try await Task.sleep(for: .milliseconds(50))
        let replaced = try await replacement.finish()
        require(replaced.coverage.videoFenced && replaced.coverage.audioFenced, "Source replacement did not fence")
        require(replacementEvents.count == 1 && newGeneration != oldGeneration, "Source replacement callback")
        replaced.writer.file.discard()
        print("PASS 5: video generation replacement auto-stops once and fences only old consumers")

        let limitAudio = CaptureAudioFake(), limitEvents = StopEvents(), limitVideo = JarvisRecordingVideoSource()
        let limitGeneration = limitVideo.beginGeneration()
        _ = limitVideo.offer(try frame(CMClockMakeHostTimeFromSystemUnits(mach_absolute_time())),
            sourceGeneration: limitGeneration, receivedHostTime: mach_absolute_time())
        let limited = try await Capture.start(audio: limitAudio.source, video: limitVideo,
            includeJarvisAudio: false, maximumDurationMilliseconds: 150, onStopped: { limitEvents.add($0) })
        let now = mach_absolute_time()
        _ = limitVideo.offer(try frame(CMClockMakeHostTimeFromSystemUnits(now), duration: CMTime(value: 10, timescale: 1_000)),
                             sourceGeneration: limitGeneration, receivedHostTime: now)
        limitAudio.send(host: now, sample: 0); await limitAudio.barrier()
        try await Task.sleep(for: .milliseconds(300))
        let limitResult = try await limited.finish()
        require(limitEvents.count == 1 && limitResult.coverage.audioFenced && limitResult.coverage.videoFenced, "Auto limit did not stop")
        require(CMTimeCompare(limitResult.times.mediaEnd, CMTime(value: 150, timescale: 1_000)) == 0, "Auto limit endpoint")
        limitResult.writer.file.discard()
        print("PASS 6: real monotonic max-duration auto-stop fences before successful completion")

        let bytesClock = CaptureClock(), bytesAudio = CaptureAudioFake()
        let (bytesVideo, bytesGeneration) = try scene(bytesClock)
        await expectFailure(.invalidConfiguration) {
            _ = try await Capture.start(audio: bytesAudio.source, video: bytesVideo, includeJarvisAudio: false,
                maximumDurationMilliseconds: 1_000, maxBytes: 0, onStopped: { _ in }, options: options(bytesClock))
        }
        let tiny = try await Capture.start(audio: bytesAudio.source, video: bytesVideo, includeJarvisAudio: false,
            maximumDurationMilliseconds: 1_000, maxBytes: 64, onStopped: { _ in }, options: options(bytesClock))
        try addFrame(bytesVideo, bytesGeneration, bytesClock, at: 0)
        bytesAudio.send(host: bytesClock.ticks(0), sample: 0); await bytesAudio.barrier(); bytesClock.advance(100)
        await expectFailure(.writerFailed) { _ = try await tiny.finish() }
        print("PASS 7: command-specific maxBytes validation and lower writer envelope")
        try await syntheticEncode()
        try await readinessChecks()
        try await resolutionChangeCheck()
    }

    static func resolutionChangeCheck() async throws {
        let clock = CaptureClock(), audio = CaptureAudioFake(), stopped = StopEvents()
        let (video, generation) = try scene(clock)
        let capture = try await Capture.start(audio: audio.source, video: video,
            includeJarvisAudio: true, maximumDurationMilliseconds: 1000,
            onStopped: { stopped.add($0) }, options: options(clock))
        for (ms, width, height) in [(0, 32, 32), (100, 64, 48), (200, 16, 16)] {
            let sample = try frame(CMTimeAdd(clock.origin, CMTime(value: Int64(ms), timescale: 1000)),
                                   width: width, height: height)
            _ = video.offer(sample, sourceGeneration: generation, receivedHostTime: clock.ticks(Int64(ms + 20)))
            audio.send(host: clock.ticks(Int64(ms)), sample: Double(ms * 48))
            audio.send(host: clock.ticks(Int64(ms)), sample: Double(ms * 48), output: true)
        }
        await audio.barrier(); clock.advance(400)
        let result = try await capture.finish()
        require(stopped.count == 0, "Resolution change stopped recording")
        require(result.writer.evidence.video.appended == 3, "Resized camera frames were lost")
        require(result.coverage.rejectedVideo == 0, "Resolution change rejected video")
        let asset = AVURLAsset(url: try result.writer.file.urlForReading())
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let size = try await tracks[0].load(.naturalSize)
        require(size == CGSize(width: 32, height: 32), "Recording canvas changed with camera resolution")
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: tracks[0], outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output); require(reader.startReading(), "Resized video decode start")
        var count = 0
        while let sample = output.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            require(abs(CMTimeGetSeconds(pts) - Double(count) / 10) < 0.001, "Resizing changed video timing")
            count += 1
        }
        require(reader.status == .completed && count == 3, "Resized video failed to decode")
        result.writer.file.discard()
        print("PASS: same-source resolution changes retain all frames, original timing and fixed canvas")
    }

    static func selectedMediaChecks() async throws {
        for selected in [false, true] {
            let clock = CaptureClock(), audio = CaptureAudioFake()
            let (video, generation) = try scene(clock)
            let capture = try await Capture.start(audio: audio.source, video: video,
                includeJarvisAudio: false, includeMediaAudio: selected,
                maximumDurationMilliseconds: 1000, onStopped: { _ in }, options: options(clock))
            try addFrame(video, generation, clock, at: 100)
            let pcm = [Int16](repeating: 4000, count: 960).withUnsafeBytes { Data($0) }
            let span = JarvisMediaCaptureSpan(sourceGeneration: 1, hostTime: clock.ticks(100), frames: 480, pcm16Interleaved: pcm)
            var coverage = JarvisMediaCaptureCoverage()
            coverage.consumedOutputFrames = 160; coverage.resolvedSourceFrames = 480
            audio.send(host: clock.ticks(100), sample: 4800)
            audio.send(host: clock.ticks(100), sample: 1600, rate: 16000, output: true,
                       mediaSpans: [span], mediaCoverage: coverage)
            // No selected media in the next output interval must still permit drain.
            audio.send(host: clock.ticks(110), sample: 5280)
            audio.send(host: clock.ticks(110), sample: 1760, rate: 16000, output: true)
            await audio.barrier(); clock.advance(120)
            try await capture.waitUntilRecording()
            let result = try await capture.finish()
            require(result.selectedMediaAudio == selected, "Selected-media manifest flag")
            require(result.mixerEvidence.renderedFrames == 0, "Excluded voice contribution")
            require(result.mixerEvidence.mediaFrames == (selected ? 480 : 0), "Media dropped before sidecar admission")
            require(result.mediaCoverage.admittedSourceFrames == (selected ? 480 : 0), "Media admission evidence")
            require(result.mediaCoverage.recordedSourceFrames == (selected ? 480 : 0), "Media final coverage evidence")
            require(result.mediaCoverage.rejectedSourceFrames == 0, "Media arrived after premature microphone flush")
            _ = try JSONEncoder().encode(result.mediaCoverage)
            result.writer.file.discard()
        }
        print("PASS: default-compatible media opt-in, sidecars with voice excluded, atomic packet admission/drain and manifest coverage")
    }

    static func readinessChecks() async throws {
        let missingClock = CaptureClock(), missingAudio = CaptureAudioFake()
        let (missingVideo, _) = try scene(missingClock)
        let missing = try await Capture.start(audio: missingAudio.source, video: missingVideo,
            includeJarvisAudio: false, maximumDurationMilliseconds: 1_000,
            onStopped: { _ in }, options: options(missingClock))
        await expectFailure(.startupTimeout) { try await missing.waitUntilRecording() }
        await expectFailure(.startupTimeout) { _ = try await missing.finish() }
        print("PASS: attachment without encoded media never confirms recording")

        let badClock = CaptureClock(), badAudio = CaptureAudioFake()
        let (badVideo, badGeneration) = try scene(badClock)
        let bad = try await Capture.start(audio: badAudio.source, video: badVideo,
            includeJarvisAudio: false, maximumDurationMilliseconds: 1_000,
            onStopped: { _ in }, options: options(badClock))
        _ = badVideo.offer(try frame(CMTime(value: 3, timescale: 1)),
            sourceGeneration: badGeneration, receivedHostTime: badClock.ticks(100))
        await expectFailure(.mappingFailed) { try await bad.waitUntilRecording() }
        await expectFailure(.mappingFailed) { _ = try await bad.finish() }
        print("PASS: first-frame mapping failure reaches startup confirmation")

        let clock = CaptureClock(), audio = CaptureAudioFake(), video = JarvisRecordingVideoSource()
        let generation = video.beginGeneration()
        _ = video.offer(try frame(CMTime(value: 7, timescale: 1)),
            sourceGeneration: generation, receivedHostTime: clock.ticks(0))
        var configured = options(clock); configured.videoClock = .datGlassesReceiptAnchored
        let capture = try await Capture.start(audio: audio.source, video: video,
            includeJarvisAudio: true, maximumDurationMilliseconds: 1_000,
            onStopped: { _ in }, options: configured)
        _ = video.offer(try frame(CMTime(value: 7100, timescale: 1000)),
            sourceGeneration: generation, receivedHostTime: clock.ticks(120))
        audio.send(host: clock.ticks(100), sample: 4_800)
        audio.send(host: clock.ticks(100), sample: 4_800, output: true)
        await audio.barrier(); clock.advance(110)
        try await capture.waitUntilRecording()
        let result = try await capture.finish()
        require(result.writer.evidence.video.appended > 0 && result.writer.evidence.audio.appended > 0,
                "Confirmed before both tracks were written")
        require(result.videoClock == .datGlassesReceiptAnchored, "Lost estimated-clock provenance")
        result.writer.file.discard()
        print("PASS: independent glasses timestamps encode with audio before confirming start")
    }

    static func syntheticEncode() async throws {
        guard let directory = ProcessInfo.processInfo.environment["JARVIS_RECORDING_TEST_OUTPUT"] else { fatalError("Evidence directory") }
        let clock = CaptureClock(), audio = CaptureAudioFake(), (video, generation) = try scene(clock)
        let capture = try await Capture.start(audio: audio.source, video: video, includeJarvisAudio: true,
            maximumDurationMilliseconds: 3_000, onStopped: { _ in }, options: options(clock))
        var nextVideo = 0
        for ms in stride(from: Int64(0), to: 2_000, by: 10) {
            while nextVideo < 48, nextVideo * 1_000 / 24 <= Int(ms) {
                let pts = CMTimeAdd(clock.origin, CMTime(value: Int64(nextVideo), timescale: 24))
                _ = video.offer(try frame(pts), sourceGeneration: generation, receivedHostTime: clock.ticks(ms + 20))
                nextVideo += 1
            }
            if ms >= 50, ms < 1_950, !(800..<850).contains(ms) {
                let rate = ms < 1_000 ? 16_000 : 44_100, route: UInt64 = ms < 1_000 ? 1 : 2
                audio.send(host: clock.ticks(ms), sample: Double(ms) * Double(rate) / 1_000, rate: rate, route: route)
                audio.send(host: clock.ticks(ms), sample: Double(ms) * Double(rate) / 1_000,
                           rate: rate, route: route, output: true, frequency: 880)
            }
            try await Task.sleep(for: .milliseconds(3))
        }
        await audio.barrier(); clock.advance(2_000)
        let result = try await capture.finish()
        let diagnostic = "Synthetic capture video=\(result.writer.evidence.video) coverage=\(result.coverage) intervals=\(result.writer.evidence.intervals)\n"
        FileHandle.standardOutput.write(Data(diagnostic.utf8))
        require(result.writer.evidence.video.appended == 48, "Capture did not retain all synthetic video frames")
        require(result.mixerEvidence.routeChanges == 1 && result.mixerEvidence.renderedFrames > 0, "Audio route/include evidence")
        // Host nanoseconds round to machine ticks; the writer then rounds up
        // to the next whole audio sample so the final frame is not truncated.
        let durationRounding = CMTimeSubtract(result.times.mediaEnd, CMTime(value: 2, timescale: 1))
        require(CMTimeCompare(durationRounding, .zero) >= 0 &&
                CMTimeCompare(durationRounding, CMTime(value: 1, timescale: 48_000)) <= 0,
                "Common-clock duration exceeds one audio sample of rounding")
        let artifact = URL(fileURLWithPath: directory).appendingPathComponent("capture-synthetic.mp4")
        try FileManager.default.copyItem(at: result.writer.file.urlForReading(), to: artifact)
        result.writer.file.discard()
        let asset = AVURLAsset(url: artifact)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: tracks[0], outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMIsFloatKey: true,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsNonInterleaved: false])
        reader.add(output); require(reader.startReading(), "AAC decode start")
        var samples: [Float] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sample) else { fatalError("Decode data") }
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / 4)
            let status = chunk.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!)
            }
            require(status == noErr, "Decode copy"); samples.append(contentsOf: chunk)
        }
        require(reader.status == .completed && samples.count >= 96_000, "Decoded duration")
        let gapRMS = sqrt(samples[39_360..<40_320].reduce(0.0) { $0 + Double($1 * $1) } / 960)
        require(gapRMS < 0.02, "Missing interval compressed or filled with source audio")
        for frequency in [440.0, 880.0] {
            let projection = (9_600..<33_600).reduce(0.0) {
                $0 + Double(samples[$1]) * sin(2 * .pi * frequency * Double($1) / 48_000)
            } / 24_000
            require(projection > 0.07, "Mixed signal missing or shifted: \(frequency)")
        }
        print("PASS 8: actual synthetic H264/AAC capture encode/decode, two audio routes, include mode and retained gap")
        print("Artifact: \(artifact.path)")
        print("PASS: 8 capture groups; mock sources/real encoder, no hardware synchronization or runtime capability claim")
    }
}
