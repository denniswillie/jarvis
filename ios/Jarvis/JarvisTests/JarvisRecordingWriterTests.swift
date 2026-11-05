import AVFoundation
import CryptoKit
import Foundation
import XCTest

/// SOURCE ONLY until Core leases native qualification. Encoder effects below
/// are mocked: passing these tests will not prove H.264/AAC encoding or playback.
final class JarvisRecordingWriterTests: XCTestCase {
    private final class Encoder: JarvisRecordingEncoding {
        let output: URL
        var videoReady = true
        var audioReady = true
        var videoPTS: [CMTime] = []
        var videoDurations: [CMTime] = []
        var audioPTS: [CMTime] = []
        var audioFrames: [Int] = []
        var audioAllZero: [Bool] = []
        var audioFirstValues: [Float] = []
        var audioReadyUntilAppend: Int?
        var audioFailAtAppend: Int?
        var cancelled = false
        var appendFails = false
        var finishSucceeds = true
        var completedSuccessfully: Bool { finishSucceeds }
        var finishHeld = false
        var heldCompletion: (@Sendable (Bool) -> Void)?
        var onFinish: (() -> Void)?
        let bytes = Data("mock encoder bytes; not a playable movie".utf8)
        init(output: URL) { self.output = output }
        func isReady(for track: JarvisRecordingWriter.Track) -> Bool {
            track == .video ? videoReady : audioReady && (audioReadyUntilAppend.map { audioPTS.count < $0 } ?? true)
        }
        func appendVideo(_ sample: CMSampleBuffer) throws {
            if appendFails { throw JarvisRecordingWriter.Reason.appendFailed }
            videoPTS.append(CMSampleBufferGetPresentationTimeStamp(sample))
            videoDurations.append(CMSampleBufferGetDuration(sample))
        }
        func appendAudio(_ samples: [Float], at time: CMTime) throws {
            if appendFails || audioFailAtAppend == audioPTS.count { throw JarvisRecordingWriter.Reason.appendFailed }
            audioPTS.append(time); audioFrames.append(samples.count)
            audioAllZero.append(samples.allSatisfy { $0 == 0 }); audioFirstValues.append(samples[0])
        }
        func finish(at time: CMTime, completion: @escaping @Sendable (Bool) -> Void) {
            if finishHeld { heldCompletion = completion; onFinish?(); return }
            do { try bytes.write(to: output); completion(finishSucceeds) }
            catch { completion(false) }
        }
        func cancel() { cancelled = true }
    }
    private struct Fixture {
        let writer: JarvisRecordingWriter
        let encoder: Encoder
        let queue: DispatchQueue
        let directory: JarvisRecordingDirectory
        func flush() { queue.sync {} }
        func cancel() { writer.cancel(generationID: writer.generationID); flush() }
    }
    private func fixture() throws -> Fixture {
        let directory = try JarvisRecordingDirectory()
        let encoder = Encoder(output: directory.file)
        let queue = DispatchQueue(label: "jarvis.writer.source-test.\(UUID().uuidString)")
        let writer = JarvisRecordingWriter(configuration: .init(width: 64, height: 64), generationID: UUID(),
            queue: queue, encoder: encoder, directory: directory)
        return Fixture(writer: writer, encoder: encoder, queue: queue, directory: directory)
    }
    private func time(_ value: Int64, scale: Int32 = 1000) -> CMTime { CMTime(value: value, timescale: scale) }
    private func video(at time: CMTime, duration: CMTime = CMTime(value: 1, timescale: 24), width: Int = 64) throws -> CMSampleBuffer {
        var pixels: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, 64, kCVPixelFormatType_32BGRA, nil, &pixels) == kCVReturnSuccess,
              let pixels else { throw JarvisRecordingWriter.Reason.invalidFormat }
        var format: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixels,
                                                           formatDescriptionOut: &format) == noErr, let format else {
            throw JarvisRecordingWriter.Reason.invalidFormat
        }
        var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixels,
            formatDescription: format, sampleTiming: &timing, sampleBufferOut: &sample) == noErr, let sample else {
            throw JarvisRecordingWriter.Reason.invalidFormat
        }
        return sample
    }
    private func addBoth(_ fixture: Fixture, at start: CMTime = .zero) throws {
        XCTAssertEqual(fixture.writer.offerVideo(try video(at: start), generationID: fixture.writer.generationID), .queued)
        XCTAssertEqual(fixture.writer.offerAudio([Float](repeating: 0.25, count: 480), at: start,
                                                 generationID: fixture.writer.generationID), .queued)
        fixture.flush()
    }

    func testConfigurationRejectsOddDimensionsAndMoreThanFiveMinutes() throws {
        XCTAssertThrowsError(try JarvisRecordingWriter.Configuration(width: 63, height: 64).validate())
        var config = JarvisRecordingWriter.Configuration(width: 64, height: 64)
        config.maximumDuration = time(300_001)
        XCTAssertThrowsError(try config.validate())
        config.maximumDuration = time(300_000); config.expectedVideoFrameDuration = .invalid
        XCTAssertThrowsError(try config.validate())
    }

    func testLastWriterReleaseDefersEncoderAndFileCleanupToWriterQueue() throws {
        let directory = try JarvisRecordingDirectory()
        let encoder = Encoder(output: directory.file)
        try encoder.bytes.write(to: directory.file)
        let queue = DispatchQueue(label: "jarvis.writer.release-test")
        var writer: JarvisRecordingWriter? = JarvisRecordingWriter(configuration: .init(width: 64, height: 64),
            generationID: UUID(), queue: queue, encoder: encoder, directory: directory)
        weak var released = writer
        queue.suspend()
        writer = nil
        XCTAssertNil(released)
        XCTAssertFalse(encoder.cancelled)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.file.path))
        queue.resume(); queue.sync {}
        XCTAssertTrue(encoder.cancelled)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.file.path))
    }

    func testAdmissionIsNotAppendAndQueueIsBounded() throws {
        let f = try fixture()
        f.queue.suspend()
        for index in 0..<JarvisRecordingWriter.maximumPendingSamples {
            XCTAssertEqual(f.writer.offerAudio([Float](repeating: 0, count: 480), at: time(Int64(index) * 10),
                                                generationID: f.writer.generationID), .queued)
        }
        XCTAssertEqual(f.writer.evidence.audio.appended, 0)
        XCTAssertEqual(f.writer.offerAudio([Float](repeating: 0, count: 480), at: time(640),
                                           generationID: f.writer.generationID), .rejected(.queueFull))
        XCTAssertEqual(f.writer.evidence.audio.queued, 64)
        XCTAssertEqual(f.writer.evidence.audio.dropped, 1)
        f.queue.resume(); f.cancel()
    }

    func testBackpressureLeavesRealGapAndDoesNotRetimeNextFrame() async throws {
        let f = try fixture()
        f.encoder.videoReady = false
        XCTAssertEqual(f.writer.offerVideo(try video(at: .zero), generationID: f.writer.generationID), .queued)
        f.flush(); f.encoder.videoReady = true
        try addBoth(f, at: time(100))
        let completed = try await f.writer.finish(at: time(200), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        XCTAssertEqual(f.encoder.videoPTS, [time(100)])
        XCTAssertEqual(zip(f.encoder.audioPTS, f.encoder.audioAllZero).filter { !$0.1 }.map { $0.0 }, [time(100)])
        XCTAssertEqual(completed.evidence.video.appended, 1)
        XCTAssertEqual(completed.evidence.video.dropped, 1)
        XCTAssertTrue(completed.evidence.intervals.contains { $0.reason == .backpressure })
        XCTAssertTrue(completed.evidence.intervals.contains { $0.track == .video && $0.reason == .missingInput && $0.start == .zero && $0.end == time(100) })
    }

    func testUnknownVideoDurationIsOnlyDiagnosticNotReplacedInEncoderInput() async throws {
        let f = try fixture()
        XCTAssertEqual(f.writer.offerVideo(try video(at: .zero, duration: .invalid), generationID: f.writer.generationID), .queued)
        XCTAssertEqual(f.writer.offerAudio([Float](repeating: 0, count: 480), at: .zero, generationID: f.writer.generationID), .queued)
        let completed = try await f.writer.finish(at: time(100), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        XCTAssertFalse(f.encoder.videoDurations[0].isValid)
        XCTAssertEqual(completed.evidence.video.inferredVideoDurations, 1)
        XCTAssertTrue(completed.evidence.intervals.contains { $0.track == .video && $0.inferredVideoDuration })
    }

    func testOutOfOrderAudioOverlapInvalidFormatAndDurationBoundaryAreRejected() throws {
        let f = try fixture()
        defer { f.cancel() }
        try addBoth(f, at: time(100))
        XCTAssertEqual(f.writer.offerVideo(try video(at: time(100)), generationID: f.writer.generationID), .rejected(.outOfOrder))
        XCTAssertEqual(f.writer.offerAudio([Float](repeating: 0, count: 480), at: time(105), generationID: f.writer.generationID), .rejected(.outOfOrder))
        XCTAssertEqual(f.writer.offerVideo(try video(at: time(200), width: 66), generationID: f.writer.generationID), .rejected(.invalidFormat))
        XCTAssertEqual(f.writer.offerAudio([0], at: .invalid, generationID: f.writer.generationID), .rejected(.invalidTime))
        XCTAssertEqual(f.writer.offerAudio([0], at: time(60_000), generationID: f.writer.generationID), .rejected(.durationLimit))
    }

    func testDiagnosticLimitPreservesDroppedTotalsAndAffectedEnvelope() throws {
        let f = try fixture()
        defer { f.cancel() }
        for index in 0..<256 {
            XCTAssertEqual(f.writer.offerAudio([], at: time(Int64(index)), generationID: f.writer.generationID), .rejected(.invalidFormat))
        }
        let report = f.writer.evidence
        XCTAssertEqual(report.audio.dropped, 256)
        XCTAssertEqual(report.intervals.count, 128)
        XCTAssertEqual(report.truncatedIntervalDetails, 128)
        XCTAssertEqual(report.affectedStart, .zero)
        XCTAssertEqual(report.affectedEnd, time(255))
    }

    func testMissingTrackCannotProduceCompletedAudiovisualRecording() async throws {
        for audioOnly in [false, true] {
            let f = try fixture()
            if audioOnly { _ = f.writer.offerAudio([0], at: .zero, generationID: f.writer.generationID) }
            else { _ = f.writer.offerVideo(try video(at: .zero), generationID: f.writer.generationID) }
            do {
                _ = try await f.writer.finish(at: time(100), generationID: f.writer.generationID)
                XCTFail("Missing track must not return a completed movie")
            } catch let error as JarvisRecordingWriter.Failure {
                XCTAssertEqual(error.reason, audioOnly ? .noVideo : .noAudio)
                XCTAssertGreaterThan(audioOnly ? error.evidence.video.missingIntervals : error.evidence.audio.missingIntervals, 0)
                if !audioOnly {
                    XCTAssertEqual(error.evidence.sourceAudioFrames, 0)
                    XCTAssertEqual(error.evidence.paddedAudioFrames, 0)
                    XCTAssertTrue(f.encoder.audioPTS.isEmpty)
                }
            }
            XCTAssertTrue(f.encoder.cancelled)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.directory.file.path))
        }
    }

    func testNonfiniteAudioIsDroppedAndNeverGivenToEncoder() throws {
        let f = try fixture()
        defer { f.cancel() }
        XCTAssertEqual(f.writer.offerAudio([.nan], at: .zero, generationID: f.writer.generationID), .queued)
        f.flush()
        XCTAssertTrue(f.encoder.audioPTS.isEmpty)
        XCTAssertEqual(f.writer.evidence.audio.dropped, 1)
        XCTAssertEqual(f.writer.evidence.intervals.first?.reason, .invalidFormat)
    }

    func testOneGenerationCancelDoesNotTouchOtherWriterOrCompletedOriginal() async throws {
        let first = try fixture(), second = try fixture()
        try addBoth(first); try addBoth(second)
        first.writer.cancel(generationID: second.writer.generationID)
        XCTAssertEqual(first.writer.offerAudio([0], at: time(20), generationID: second.writer.generationID), .rejected(.wrongGeneration))
        let completed = try await first.writer.finish(at: time(100), generationID: first.writer.generationID)
        let original = try Data(contentsOf: completed.file.urlForReading())
        second.cancel()
        XCTAssertEqual(try Data(contentsOf: completed.file.urlForReading()), original)
        XCTAssertEqual(completed.file.contentSHA256, SHA256.hash(data: original).map { String(format: "%02x", $0) }.joined())
        XCTAssertEqual(first.writer.offerAudio([0], at: time(200), generationID: first.writer.generationID), .rejected(.closed))
        completed.file.discard()
        XCTAssertThrowsError(try completed.file.urlForReading())
    }

    func testAppendFailureAndLateFinishCallbackCannotRestoreCancelledOutput() async throws {
        let failed = try fixture()
        failed.encoder.appendFails = true
        _ = failed.writer.offerAudio([0], at: .zero, generationID: failed.writer.generationID)
        failed.flush()
        do { _ = try await failed.writer.finish(at: time(100), generationID: failed.writer.generationID); XCTFail("Expected failure") }
        catch let error as JarvisRecordingWriter.Failure { XCTAssertEqual(error.reason, .appendFailed) }

        let cancelled = try fixture()
        cancelled.encoder.finishHeld = true
        let started = expectation(description: "Fake encoder entered finish")
        cancelled.encoder.onFinish = { started.fulfill() }
        try addBoth(cancelled)
        let writer = cancelled.writer, generation = writer.generationID, end = time(100)
        let finishing = Task { [writer, generation, end] in try await writer.finish(at: end, generationID: generation) }
        await fulfillment(of: [started], timeout: 1)
        cancelled.cancel()
        do { _ = try await finishing.value; XCTFail("Expected cancellation") }
        catch let error as JarvisRecordingWriter.Failure { XCTAssertEqual(error.reason, .cancelled) }
        cancelled.queue.sync { cancelled.encoder.heldCompletion?(true) }
        cancelled.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: cancelled.directory.file.path))
    }

    func testMiddleGapAddsOnlyExplicitZerosAndKeepsLaterSourceAtOriginalTime() async throws {
        let f = try fixture()
        try addBoth(f)
        XCTAssertEqual(f.writer.offerAudio([Float](repeating: -0.5, count: 480), at: time(60), generationID: f.writer.generationID), .queued)
        let completed = try await f.writer.finish(at: time(100), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        XCTAssertEqual(f.encoder.audioPTS, [time(0), time(10), time(60), time(70)])
        XCTAssertEqual(f.encoder.audioFrames, [480, 2400, 480, 1440])
        XCTAssertEqual(f.encoder.audioFirstValues, [0.25, 0, -0.5, 0])
        XCTAssertEqual(completed.evidence.audio.queued, 2)
        XCTAssertEqual(completed.evidence.audio.appended, 2)
        XCTAssertEqual(completed.evidence.sourceAudioFrames, 960)
        XCTAssertEqual(completed.evidence.paddedAudioFrames, 3840)
        XCTAssertEqual(completed.evidence.paddedAudioBuffers, 2)
        XCTAssertEqual(completed.evidence.paddedAudioDuration, time(80))
        XCTAssertTrue(completed.evidence.intervals.contains { $0.track == .audio && $0.reason == .missingInput
            && $0.start == time(10) && $0.end == time(60) })
    }

    func testLeadingAndTrailingPaddingUsesExplicitZeroAndFinishNotFirstArrival() async throws {
        let f = try fixture()
        _ = f.writer.offerVideo(try video(at: .zero), generationID: f.writer.generationID)
        _ = f.writer.offerAudio([Float](repeating: 0.5, count: 480), at: time(20), generationID: f.writer.generationID)
        let completed = try await f.writer.finish(at: time(100), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        XCTAssertEqual(f.encoder.audioPTS, [time(0), time(20), time(30)])
        XCTAssertEqual(f.encoder.audioFrames, [960, 480, 3360])
        XCTAssertEqual(f.encoder.audioAllZero, [true, false, true])
        XCTAssertEqual(completed.evidence.audio.firstAppendedPTS, time(20))
        XCTAssertEqual(completed.evidence.audio.lastAppendedEnd, time(30))
        XCTAssertEqual(completed.evidence.audio.appended, 1)
        XCTAssertEqual(completed.evidence.sourceAudioFrames, 480)
        XCTAssertEqual(completed.evidence.paddedAudioDuration, time(90))
        XCTAssertEqual(completed.evidence.audio.missingDuration, time(90))
    }

    func testPaddingChunksAndTotalStayInsideExplicitSixtySecondBound() async throws {
        let f = try fixture()
        _ = f.writer.offerVideo(try video(at: .zero), generationID: f.writer.generationID)
        _ = f.writer.offerAudio([Float](repeating: 0.5, count: 480), at: time(59_990), generationID: f.writer.generationID)
        let completed = try await f.writer.finish(at: time(60_000), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        XCTAssertTrue(f.encoder.audioFrames.allSatisfy { $0 > 0 && $0 <= 4800 })
        XCTAssertEqual(f.encoder.audioFrames.reduce(0, +), 60 * 48_000)
        XCTAssertEqual(completed.evidence.sourceAudioFrames, 480)
        XCTAssertEqual(completed.evidence.paddedAudioFrames, 60 * 48_000 - 480)
        XCTAssertEqual(completed.evidence.paddedAudioBuffers, 600)
        XCTAssertEqual(completed.evidence.audio.appended, 1)
    }

    func testPaddingBackpressureOrAppendFailureDiscardsOnlyAffectedGeneration() async throws {
        for appendFailure in [false, true] {
            let f = try fixture(), unrelated = try fixture()
            defer { unrelated.cancel() }
            if appendFailure { f.encoder.audioFailAtAppend = 1 }
            else { f.encoder.audioReadyUntilAppend = 1 }
            _ = f.writer.offerVideo(try video(at: .zero), generationID: f.writer.generationID)
            _ = f.writer.offerAudio([Float](repeating: 0.5, count: 480), at: time(250), generationID: f.writer.generationID)
            f.flush()
            do { _ = try await f.writer.finish(at: time(300), generationID: f.writer.generationID); XCTFail("Expected padding failure") }
            catch let error as JarvisRecordingWriter.Failure {
                XCTAssertEqual(error.reason, appendFailure ? .appendFailed : .backpressure)
                XCTAssertEqual(error.evidence.sourceAudioFrames, 0)
                XCTAssertEqual(error.evidence.audio.appended, 0)
                XCTAssertEqual(error.evidence.paddedAudioFrames, 4800)
                XCTAssertEqual(error.evidence.paddedAudioBuffers, 1)
            }
            XCTAssertTrue(f.encoder.cancelled)
            XCTAssertFalse(unrelated.encoder.cancelled)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.directory.file.path))
        }
    }

    func testTrailingPaddingBackpressureCannotReturnCompletedFile() async throws {
        let f = try fixture()
        try addBoth(f)
        f.encoder.audioReady = false
        do { _ = try await f.writer.finish(at: time(100), generationID: f.writer.generationID); XCTFail("Expected padding failure") }
        catch let error as JarvisRecordingWriter.Failure {
            XCTAssertEqual(error.reason, .backpressure)
            XCTAssertEqual(error.evidence.sourceAudioFrames, 480)
            XCTAssertEqual(error.evidence.audio.appended, 1)
            XCTAssertEqual(error.evidence.paddedAudioFrames, 0)
            XCTAssertEqual(error.evidence.audio.missingDuration, time(90))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.directory.file.path))
    }

    func testHalfSampleStartAndFinishAreRejectedRatherThanRounded() async throws {
        let f = try fixture()
        XCTAssertEqual(f.writer.offerAudio([0.5], at: time(1, scale: 96_000), generationID: f.writer.generationID),
                       .rejected(.audioTimeNotOnSampleGrid))
        try addBoth(f)
        do { _ = try await f.writer.finish(at: time(9601, scale: 96_000), generationID: f.writer.generationID); XCTFail("Expected exact-grid rejection") }
        catch let error as JarvisRecordingWriter.Failure { XCTAssertEqual(error.reason, .audioTimeNotOnSampleGrid) }
        let completed = try await f.writer.finish(at: time(100), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        XCTAssertEqual(completed.evidence.sourceAudioFrames, 480)
        XCTAssertEqual(f.encoder.audioPTS.first, .zero)
    }

    func testExactLargeTimescaleInputKeepsOriginalPTSAndIntegerSampleCoverage() async throws {
        let f = try fixture()
        try addBoth(f)
        let exactSecond = CMTime(value: Int64(Int32.max), timescale: Int32.max)
        XCTAssertEqual(f.writer.offerAudio([0.5], at: exactSecond, generationID: f.writer.generationID), .queued)
        let completed = try await f.writer.finish(at: time(1100), generationID: f.writer.generationID)
        defer { completed.file.discard() }
        let sourceIndices = f.encoder.audioAllZero.indices.filter { !f.encoder.audioAllZero[$0] }
        XCTAssertEqual(sourceIndices.count, 2)
        XCTAssertEqual(f.encoder.audioPTS[sourceIndices[1]].timescale, Int32.max)
        XCTAssertEqual(f.encoder.audioPTS[sourceIndices[1]].value, Int64(Int32.max))
        XCTAssertEqual(completed.evidence.audio.lastAppendedEnd, CMTime(value: 48_001, timescale: 48_000))
        XCTAssertEqual(completed.evidence.sourceAudioFrames, 481)
        XCTAssertEqual(completed.evidence.paddedAudioFrames, 52_800 - 481)
    }
}

// Next separately leased real proof (not run by these mocked tests): ONE 2s
// synthetic encode, same exact video24fps timeline with frame12 absent; audio
// input absent [0,.020), [.800,.850), [1.970,2.000). Supply440Hz before the middle
// gap and880Hz after it, plus a distinct pulse at1.000s matching the video color
// change. Expect91200 source frames +4800 explicit padding frames. Decode actual
// PCM: inspect the gap interior [.822,.828), dominant frequency before/after,
// and audio pulse/video-frame alignment. Predeclare lossy AAC edge tolerance of
// <=1024 samples (21.333ms), gap-interior RMS<.01 versus surrounding tone>.08;
// retain observed values even if a threshold fails. Packet/container length alone
// cannot qualify this change. No physical hearing or capture-clock claim follows.
