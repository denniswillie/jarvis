@preconcurrency import AVFoundation
import CryptoKit
import Darwin
import Foundation
import UniformTypeIdentifiers

/// One M0 recording generation. Core supplies a common timeline whose origin is
/// zero, already-mapped video PTS and already-mixed mono 48 kHz float32 audio.
/// This object never captures, mixes, resamples, discovers clocks or controls a
/// camera/audio session. Missing PCM time is explicitly padded with zeros; this
/// is a representation of absent source input, never evidence of scene silence.
/// Offers take a brief state lock, never wait for the
/// encoder/disk, and report queue admission separately from a successful append.
final class JarvisRecordingWriter: @unchecked Sendable {
    enum Track: Sendable, Equatable { case video, audio }
    enum Reason: String, Error, Sendable {
        case invalidConfiguration, wrongGeneration, closed, queueFull
        case invalidTime, durationLimit, invalidFormat, outOfOrder, backpressure
        case audioTimeNotOnSampleGrid, paddingLimit
        case appendFailed, missingInput, startFailed, noVideo, noAudio
        case invalidFinishTime, alreadyFinishing, cancelled, finishFailed, finishTimeout, fileFailed
    }
    /// Produced by the authenticated pending store before capture admission.
    /// This handle grants a destination, not ownership of its disposal.
    struct Destination: Sendable {
        let directoryURL: URL
        let generationID: UUID
    }
    enum Output: Sendable { case local(Destination), stream(Streaming) }
    struct Streaming: Sendable {
        let generationID: UUID
        var interval = CMTime(seconds: 2, preferredTimescale: 48_000)
        let bytes: JarvisRecordingByteStream
    }
    struct Configuration: Sendable {
        let width: Int
        let height: Int
        var destination: Destination?
        var streaming: Streaming?
        // Only a diagnostic coverage estimate when the video duration is absent.
        // Neither this value nor queue arrival time replaces incoming PTS.
        var expectedVideoFrameDuration = CMTime(value: 1, timescale: 24)
        var maximumDuration = CMTime(seconds: 1800, preferredTimescale: 48_000)
        var maximumFileBytes = JarvisRecordingWriter.maximumFileBytes
        func validate() throws {
            guard (2...3840).contains(width), (2...3840).contains(height), width.isMultiple(of: 2),
                  height.isMultiple(of: 2), Self.positive(expectedVideoFrameDuration),
                  CMTimeCompare(expectedVideoFrameDuration, CMTime(value: 1, timescale: 1)) <= 0,
                  Self.positive(maximumDuration), CMTimeCompare(maximumDuration, CMTime(value: 1800, timescale: 1)) <= 0,
                  (1...JarvisRecordingWriter.maximumFileBytes).contains(maximumFileBytes)
            else { throw Reason.invalidConfiguration }
            if let streaming {
                guard destination == nil, Self.positive(streaming.interval),
                      CMTimeCompare(streaming.interval, CMTime(value: 10, timescale: 1)) <= 0 else {
                    throw Reason.invalidConfiguration
                }
            }
        }
        private static func positive(_ time: CMTime) -> Bool {
            time.isNumeric && time.epoch == 0 && CMTimeCompare(time, .zero) > 0
        }
    }
    enum Admission: Equatable { case queued, rejected(Reason) }
    struct Interval: Sendable {
        let track: Track
        let start: CMTime
        let end: CMTime
        let reason: Reason
        let inferredVideoDuration: Bool
    }
    struct TrackEvidence: Sendable {
        var queued = 0
        var appended = 0
        var dropped = 0
        var missingIntervals = 0
        var missingDuration = CMTime.zero
        var inferredVideoDurations = 0
        var firstAppendedPTS: CMTime?
        var lastAppendedEnd = CMTime.zero
        var lastAppendedEndInferred = false
    }
    struct Evidence: Sendable {
        var video = TrackEvidence()
        var audio = TrackEvidence()
        // Source buffer/frame counters never include generated zero padding.
        var sourceAudioFrames: Int64 = 0
        var paddedAudioFrames: Int64 = 0
        var paddedAudioBuffers = 0
        var paddedAudioDuration: CMTime { CMTime(value: paddedAudioFrames, timescale: 48_000) }
        var intervals: [Interval] = []
        var truncatedIntervalDetails = 0
        // Envelope of all reported affected intervals, NOT a claim that every
        // moment inside the envelope was missing. Totals survive detail truncation.
        var affectedStart: CMTime?
        var affectedEnd: CMTime?
    }
    struct Completed: Sendable {
        let generationID: UUID
        let file: JarvisRecordingLocalFile
        let requestedEnd: CMTime
        let evidence: Evidence
        // Successful local encoder completion only. No playback, hearing, sync,
        // full-decode, upload, source authorization or retained-original claim.
    }
    struct Failure: Error, Sendable { let reason: Reason; let evidence: Evidence }

    static let maximumPendingSamples = 64
    static let maximumPendingBytes = 32 * 1024 * 1024
    static let maximumVideoBufferBytes = 16 * 1024 * 1024
    static let maximumAudioFrames = 4800
    static let maximumPaddingChunkFrames = 4800
    static let maximumAudioTimelineFrames: Int64 = 1800 * 48_000
    static let maximumIntervalDetails = 128
    static let maximumFileBytes = 1024 * 1024 * 1024
    let generationID: UUID
    private let configuration: Configuration
    private let queue: DispatchQueue
    private let encoder: JarvisRecordingEncoding
    private var directory: JarvisRecordingDirectory?
    private let lock = NSLock()
    private enum Phase: Equatable { case recording, finishing, finished, failed, cancelled }
    private var phase = Phase.recording
    private var terminalReason: Reason?
    private var pending: [Entry] = []
    private var pendingBytes = 0
    private var drainScheduled = false
    private var maximumAdmittedEnd = CMTime.zero
    private var lastVideoPTS: CMTime?
    private var lastAudioEnd: CMTime?
    private var report = Evidence()
    private var finishEnd: CMTime?
    private var finishContinuation: CheckedContinuation<Completed, Error>?
    // Encoder lifecycle and directory ownership below are confined to queue.
    private var finalizationStarted = false
    private var lastSizeCheckPTS = CMTime.zero
    // Successfully encoded source + explicit padding frames on Core's timeline.
    // Confined to queue; never advances because of callback/wall-clock time.
    private var audioCursorFrames: Int64 = 0

    private enum Payload { case video(CMSampleBuffer), audio([Float]) }
    private struct Entry: @unchecked Sendable {
        let track: Track
        let start: CMTime
        let end: CMTime
        let bytes: Int
        let inferred: Bool
        let payload: Payload
    }

    /// Call before subscribing to capture. Directory/encoder creation runs away
    /// from capture callbacks and MainActor. The result owns one private file.
    static func make(configuration: Configuration) async throws -> JarvisRecordingWriter {
        try configuration.validate()
        let generation = configuration.streaming?.generationID ?? configuration.destination?.generationID ?? UUID()
        let queue = DispatchQueue(label: "jarvis.recording.\(generation.uuidString)", qos: .utility)
        let writer: JarvisRecordingWriter = try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let directory: JarvisRecordingDirectory?
                    if configuration.streaming != nil {
                        directory = nil
                    } else if let destination = configuration.destination {
                        directory = try JarvisRecordingDirectory(storeOwned: destination)
                    } else {
                        directory = try JarvisRecordingDirectory()
                    }
                    let encoder = try JarvisAssetRecordingEncoder(configuration: configuration,
                        output: directory?.file, generationID: generation)
                    continuation.resume(returning: JarvisRecordingWriter(configuration: configuration,
                        generationID: generation, queue: queue, encoder: encoder, directory: directory))
                } catch { continuation.resume(throwing: Reason.startFailed) }
            }
        }
        if Task.isCancelled { writer.cancel(generationID: generation); throw CancellationError() }
        return writer
    }

    // Internal injection seam for disjoint tests; production uses make().
    init(configuration: Configuration, generationID: UUID, queue: DispatchQueue,
         encoder: JarvisRecordingEncoding, directory: JarvisRecordingDirectory?) {
        self.configuration = configuration; self.generationID = generationID
        self.queue = queue; self.encoder = encoder; self.directory = directory
    }

    /// Retains immutable sample storage until the append returns. Caller must
    /// not modify sample timing, attachments or pixel contents after admission.
    @discardableResult
    func offerVideo(_ sample: CMSampleBuffer, generationID: UUID) -> Admission {
        let start = CMSampleBufferGetPresentationTimeStamp(sample)
        let suppliedDuration = CMSampleBufferGetDuration(sample)
        let inferred = !Self.positive(suppliedDuration)
        let end = CMTimeAdd(start, inferred ? configuration.expectedVideoFrameDuration : suppliedDuration)
        if suppliedDuration.isNumeric && (suppliedDuration.epoch != 0 || CMTimeCompare(suppliedDuration, .zero) < 0) {
            return reject(.invalidTime, track: .video, start: start, end: start, inferred: false, generation: generationID)
        }
        guard CMSampleBufferDataIsReady(sample), CMSampleBufferGetNumSamples(sample) == 1,
              let image = CMSampleBufferGetImageBuffer(sample),
              CVPixelBufferGetWidth(image) == configuration.width, CVPixelBufferGetHeight(image) == configuration.height,
              [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
               kCVPixelFormatType_420YpCbCr8BiPlanarFullRange].contains(CVPixelBufferGetPixelFormatType(image)),
              CVPixelBufferGetDataSize(image) > 0, CVPixelBufferGetDataSize(image) <= Self.maximumVideoBufferBytes
        else { return reject(.invalidFormat, track: .video, start: start, end: end, inferred: inferred, generation: generationID) }
        return offer(Entry(track: .video, start: start, end: end, bytes: CVPixelBufferGetDataSize(image),
                           inferred: inferred, payload: .video(sample)), generation: generationID)
    }

    /// Array storage is retained with Swift value semantics, never a borrowed
    /// callback pointer. Exactly one already-mixed track; no resampling/mixing.
    /// PTS must lie exactly on the 48 kHz grid relative to Core's explicit zero.
    @discardableResult
    func offerAudio(_ monoFloat32: [Float], at presentationTime: CMTime, generationID: UUID) -> Admission {
        let frames = monoFloat32.count
        guard (1...Self.maximumAudioFrames).contains(frames) else {
            return reject(.invalidFormat, track: .audio, start: presentationTime, end: presentationTime,
                          inferred: false, generation: generationID)
        }
        // Canonical integer arithmetic for coverage avoids CMTimeAdd rounding
        // when an exactly representable input uses an unnecessarily large scale.
        // The original presentationTime is still passed unchanged to the encoder.
        let end = Self.audioFrameIndex(presentationTime).map {
            CMTime(value: $0 + Int64(frames), timescale: 48_000)
        } ?? CMTimeAdd(presentationTime, CMTime(value: Int64(frames), timescale: 48_000))
        return offer(Entry(track: .audio, start: presentationTime, end: end, bytes: frames * MemoryLayout<Float>.size,
                           inferred: false, payload: .audio(monoFloat32)), generation: generationID)
    }

    var evidence: Evidence { lock.withLock { report } }

    private func offer(_ entry: Entry, generation: UUID) -> Admission {
        lock.lock(); defer { lock.unlock() }
        guard generation == generationID else { return .rejected(.wrongGeneration) }
        guard phase == .recording else { return .rejected(.closed) }
        let reason: Reason?
        if !Self.nonnegative(entry.start) || !Self.positive(entry.end) || CMTimeCompare(entry.end, entry.start) <= 0 {
            reason = .invalidTime
        } else if CMTimeCompare(entry.start, configuration.maximumDuration) >= 0
                    || (!entry.inferred && CMTimeCompare(entry.end, configuration.maximumDuration) > 0) { reason = .durationLimit
        } else if entry.track == .audio, Self.audioFrameIndex(entry.start) == nil { reason = .audioTimeNotOnSampleGrid
        } else if entry.track == .video, let lastVideoPTS, CMTimeCompare(entry.start, lastVideoPTS) <= 0 { reason = .outOfOrder
        } else if entry.track == .audio, let lastAudioEnd, CMTimeCompare(entry.start, lastAudioEnd) < 0 { reason = .outOfOrder
        } else if pending.count >= Self.maximumPendingSamples || entry.bytes > Self.maximumPendingBytes - pendingBytes {
            reason = .queueFull
        } else { reason = nil }
        if let reason { recordDrop(entry, reason: reason); return .rejected(reason) }
        if entry.track == .video { lastVideoPTS = entry.start; report.video.queued += 1 }
        else { lastAudioEnd = entry.end; report.audio.queued += 1 }
        // Unknown video duration is a coverage estimate only, not a stop-time
        // requirement. Core's explicit finish boundary controls the session end.
        maximumAdmittedEnd = CMTimeMaximum(maximumAdmittedEnd, entry.inferred ? entry.start : entry.end)
        pending.append(entry); pendingBytes += entry.bytes
        scheduleDrainLocked()
        return .queued
    }

    private func reject(_ reason: Reason, track: Track, start: CMTime, end: CMTime,
                        inferred: Bool, generation: UUID) -> Admission {
        lock.withLock {
            guard generation == generationID else { return .rejected(.wrongGeneration) }
            guard phase == .recording else { return .rejected(.closed) }
            if track == .video { report.video.dropped += 1 } else { report.audio.dropped += 1 }
            recordInterval(Interval(track: track, start: start, end: end, reason: reason, inferredVideoDuration: inferred))
            return .rejected(reason)
        }
    }

    private func scheduleDrainLocked() {
        guard !drainScheduled else { return }
        drainScheduled = true
        queue.async { self.drain() }
    }

    private func drain() {
        while true {
            let next: Entry? = lock.withLock {
                guard phase == .recording || phase == .finishing, !pending.isEmpty else {
                    drainScheduled = false; return nil
                }
                let next = pending.removeFirst(); pendingBytes -= next.bytes; return next
            }
            guard let next else { beginFinalizationIfNeeded(); return }
            autoreleasepool { append(next) }
        }
    }

    private func append(_ entry: Entry) {
        guard encoder.isReady(for: entry.track) else {
            lock.withLock { recordDrop(entry, reason: .backpressure) }
            if entry.track == .audio { fail(.backpressure) }
            return
        }
        do {
            switch entry.payload {
            case .video(let buffer): try encoder.appendVideo(buffer)
            case .audio(let samples):
                guard samples.allSatisfy(\.isFinite) else {
                    lock.withLock { recordDrop(entry, reason: .invalidFormat) }; return
                }
                guard let startFrame = Self.audioFrameIndex(entry.start) else { throw Reason.audioTimeNotOnSampleGrid }
                lock.withLock {
                    recordMissing(track: .audio, from: report.audio.lastAppendedEnd,
                                  to: CMTime(value: startFrame, timescale: 48_000), inferred: false)
                }
                try padAudio(to: startFrame)
                guard encoder.isReady(for: .audio) else { throw Reason.backpressure }
                try encoder.appendAudio(samples, at: entry.start)
                audioCursorFrames = startFrame + Int64(samples.count)
                lock.withLock { report.sourceAudioFrames += Int64(samples.count) }
            }
            lock.withLock {
                var track = entry.track == .video ? report.video : report.audio
                if entry.track == .video {
                    recordMissing(track: .video, from: track.lastAppendedEnd, to: entry.start,
                                  inferred: track.lastAppendedEndInferred)
                }
                // Reload after recordMissing updates aggregate counters.
                track = entry.track == .video ? report.video : report.audio
                track.appended += 1
                if entry.inferred { track.inferredVideoDurations += 1 }
                if track.firstAppendedPTS == nil { track.firstAppendedPTS = entry.start }
                if CMTimeCompare(entry.end, track.lastAppendedEnd) > 0 {
                    track.lastAppendedEnd = entry.end; track.lastAppendedEndInferred = entry.inferred
                }
                if entry.track == .video { report.video = track } else { report.audio = track }
            }
            // Encoder/disk checks stay off capture callbacks. The final limit is
            // authoritative; this periodic check additionally bounds runaway output.
            if CMTimeGetSeconds(CMTimeSubtract(entry.end, lastSizeCheckPTS)) >= 1 {
                lastSizeCheckPTS = entry.end
                do {
                    if let directory, try directory.byteCount() > configuration.maximumFileBytes { fail(.fileFailed) }
                } catch { fail(.fileFailed) } // Already appended; do not also count this sample as dropped.
            }
        } catch let reason as Reason {
            lock.withLock { recordDrop(entry, reason: reason) }
            fail(reason)
        } catch {
            lock.withLock { recordDrop(entry, reason: .appendFailed) }
            fail(.appendFailed)
        }
    }

    /// Fill only an explicit gap on the already-mapped source timeline. No
    /// retries, sleeping, callback blocking, borrowed storage or timestamp shifts.
    private func padAudio(to targetFrame: Int64) throws {
        guard targetFrame >= audioCursorFrames else { throw Reason.outOfOrder }
        guard targetFrame <= Self.maximumAudioTimelineFrames else { throw Reason.paddingLimit }
        while audioCursorFrames < targetFrame {
            guard lock.withLock({ phase == .recording || phase == .finishing }) else { throw Reason.cancelled }
            guard encoder.isReady(for: .audio) else { throw Reason.backpressure }
            let frames = min(Self.maximumPaddingChunkFrames, Int(targetFrame - audioCursorFrames))
            let zeros = [Float](repeating: 0, count: frames)
            try encoder.appendAudio(zeros, at: CMTime(value: audioCursorFrames, timescale: 48_000))
            audioCursorFrames += Int64(frames)
            lock.withLock {
                report.paddedAudioFrames += Int64(frames)
                report.paddedAudioBuffers += 1
            }
        }
    }

    /// End must cover all admitted measured sample intervals and fit the <=60s
    /// window. Unknown-duration video requires end > PTS; its estimated duration
    /// affects diagnostics only. No incoming sample is silently retimed.
    /// Explicit finish must also lie exactly on the 48 kHz grid. Core chooses
    /// that endpoint; this writer never silently rounds it to a sample boundary.
    /// No new input is accepted after this call fences intake. Only one finish
    /// waiter is retained; repeated calls fail instead of growing an unbounded list.
    func finish(at end: CMTime, generationID: UUID) async throws -> Completed {
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                lock.withLock {
                    guard generationID == self.generationID else {
                        continuation.resume(throwing: Failure(reason: .wrongGeneration, evidence: report)); return
                    }
                    guard phase == .recording else {
                        continuation.resume(throwing: Failure(reason: terminalReason ?? .alreadyFinishing, evidence: report)); return
                    }
                    guard Self.positive(end), CMTimeCompare(end, configuration.maximumDuration) <= 0,
                          CMTimeCompare(end, maximumAdmittedEnd) >= 0,
                          lastVideoPTS.map({ CMTimeCompare(end, $0) > 0 }) ?? true else {
                        continuation.resume(throwing: Failure(reason: .invalidFinishTime, evidence: report)); return
                    }
                    guard Self.audioFrameIndex(end) != nil else {
                        continuation.resume(throwing: Failure(reason: .audioTimeNotOnSampleGrid, evidence: report)); return
                    }
                    phase = .finishing; finishEnd = end; finishContinuation = continuation
                    scheduleDrainLocked()
                }
            }
        }, onCancel: { self.cancel(generationID: generationID) })
    }

    func cancel(generationID: UUID) {
        let accepted = lock.withLock {
            guard generationID == self.generationID, phase == .recording || phase == .finishing else { return false }
            phase = .cancelled; terminalReason = .cancelled
            for entry in pending { recordDrop(entry, reason: .cancelled) }
            pending.removeAll(); pendingBytes = 0
            return true
        }
        if accepted { queue.async { self.fail(.cancelled) } }
    }

    private func beginFinalizationIfNeeded() {
        let end = lock.withLock { phase == .finishing ? finishEnd : nil }
        guard let end, !finalizationStarted else { return }
        finalizationStarted = true
        lock.withLock {
            if report.video.lastAppendedEndInferred {
                report.video.lastAppendedEnd = CMTimeMinimum(report.video.lastAppendedEnd, end)
            }
            recordMissing(track: .video, from: report.video.lastAppendedEnd, to: end,
                          inferred: report.video.lastAppendedEndInferred)
            if let finalFrame = Self.audioFrameIndex(end) {
                recordMissing(track: .audio, from: report.audio.lastAppendedEnd,
                              to: CMTime(value: finalFrame, timescale: 48_000), inferred: false)
            }
        }
        let counts = lock.withLock { (report.video.appended, report.audio.appended) }
        guard counts.0 > 0 else { fail(.noVideo); return }
        guard counts.1 > 0 else { fail(.noAudio); return }
        do {
            guard let finalFrame = Self.audioFrameIndex(end) else { throw Reason.audioTimeNotOnSampleGrid }
            try padAudio(to: finalFrame)
        } catch let reason as Reason { fail(reason); return }
        catch { fail(.appendFailed); return }
        encoder.finish(at: end) { [weak self] success in
            guard let self else { return }
            self.queue.async { self.finished(success: success, end: end) }
        }
        queue.asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self, self.lock.withLock({ self.phase == .finishing }) else { return }
            self.fail(.finishTimeout)
        }
    }

    private func finished(success: Bool, end: CMTime) {
        guard lock.withLock({ phase == .finishing }) else { return }
        guard success, encoder.completedSuccessfully else { fail(.finishFailed); return }
        do {
            let file: JarvisRecordingEncodedAsset
            if let streamed = encoder.streamedAsset {
                file = streamed
            } else if let directory {
                file = try directory.completedFile(generationID: generationID, maximumBytes: configuration.maximumFileBytes)
            } else { throw Reason.fileFailed }
            let completion: (CheckedContinuation<Completed, Error>, Evidence)? = lock.withLock {
                guard phase == .finishing, let continuation = finishContinuation else { return nil }
                phase = .finished; finishContinuation = nil
                return (continuation, report)
            }
            guard let completion else { file.discard(); return }
            self.directory = nil // Ownership transfers only after successful close + exact local hash.
            completion.0.resume(returning: Completed(generationID: generationID, file: file, requestedEnd: end,
                                                     evidence: completion.1))
        } catch { fail(.fileFailed) }
    }

    private func fail(_ reason: Reason) {
        guard lock.withLock({ phase != .finished }) else { return }
        encoder.cancel()
        directory?.discard(); directory = nil
        let result: (CheckedContinuation<Completed, Error>?, Failure) = lock.withLock {
            let actualReason: Reason = phase == .cancelled ? .cancelled : reason
            phase = actualReason == .cancelled ? .cancelled : .failed; terminalReason = actualReason
            for entry in pending { recordDrop(entry, reason: actualReason) }
            pending.removeAll(); pendingBytes = 0
            let continuation = finishContinuation; finishContinuation = nil
            return (continuation, Failure(reason: actualReason, evidence: report))
        }
        result.0?.resume(throwing: result.1)
    }

    // The following evidence helpers are called only while holding lock.
    private func recordDrop(_ entry: Entry, reason: Reason) {
        if entry.track == .video { report.video.dropped += 1 } else { report.audio.dropped += 1 }
        recordInterval(Interval(track: entry.track, start: entry.start, end: entry.end, reason: reason,
                                inferredVideoDuration: entry.inferred))
    }
    private func recordMissing(track: Track, from: CMTime, to: CMTime, inferred: Bool) {
        guard CMTimeCompare(to, from) > 0 else { return }
        let duration = CMTimeSubtract(to, from)
        if track == .video { report.video.missingIntervals += 1; report.video.missingDuration = CMTimeAdd(report.video.missingDuration, duration) }
        else { report.audio.missingIntervals += 1; report.audio.missingDuration = CMTimeAdd(report.audio.missingDuration, duration) }
        recordInterval(Interval(track: track, start: from, end: to, reason: .missingInput, inferredVideoDuration: inferred))
    }
    private func recordInterval(_ interval: Interval) {
        if Self.nonnegative(interval.start), Self.nonnegative(interval.end) {
            report.affectedStart = report.affectedStart.map { CMTimeMinimum($0, interval.start) } ?? interval.start
            report.affectedEnd = report.affectedEnd.map { CMTimeMaximum($0, interval.end) } ?? interval.end
        }
        if report.intervals.count < Self.maximumIntervalDetails { report.intervals.append(interval) }
        else { report.truncatedIntervalDetails += 1 }
    }
    private static func nonnegative(_ time: CMTime) -> Bool {
        time.isNumeric && time.epoch == 0 && CMTimeCompare(time, .zero) >= 0
    }
    private static func positive(_ time: CMTime) -> Bool { nonnegative(time) && CMTimeCompare(time, .zero) > 0 }
    private static func audioFrameIndex(_ time: CMTime) -> Int64? {
        guard nonnegative(time), CMTimeCompare(time, CMTime(value: 1800, timescale: 1)) <= 0 else { return nil }
        let scaled = CMTimeConvertScale(time, timescale: 48_000, method: .roundTowardZero)
        // Conversion is used to CHECK exact representability, never to accept a
        // rounded timestamp. The supplied PTS still goes unchanged to the encoder.
        guard scaled.isNumeric, CMTimeCompare(time, scaled) == 0 else { return nil }
        return scaled.value
    }
    private struct Cleanup: @unchecked Sendable {
        // Sole cleanup ownership after JRW release; execution stays serialized
        // with prior encoder calls. Never capture a deinitialized writer.
        let encoder: JarvisRecordingEncoding
        let directory: JarvisRecordingDirectory?
        func run() { encoder.cancel(); directory?.discard() }
    }
    deinit {
        let cleanup = Cleanup(encoder: encoder, directory: directory)
        queue.async { cleanup.run() }
    }
}

/// Narrow encoder seam for source-only tests. All methods run on one writer queue.
protocol JarvisRecordingEncoding: AnyObject {
    var completedSuccessfully: Bool { get }
    var streamedAsset: JarvisRecordingEncodedAsset? { get }
    func isReady(for track: JarvisRecordingWriter.Track) -> Bool
    func appendVideo(_ sample: CMSampleBuffer) throws
    func appendAudio(_ samples: [Float], at time: CMTime) throws
    func finish(at time: CMTime, completion: @escaping @Sendable (Bool) -> Void)
    func cancel()
}
extension JarvisRecordingEncoding {
    var streamedAsset: JarvisRecordingEncodedAsset? { nil }
}

private final class JarvisAssetRecordingEncoder: JarvisRecordingEncoding {
    private let writer: AVAssetWriter
    private let video: AVAssetWriterInput
    private let audio: AVAssetWriterInput
    private let audioFormat: CMAudioFormatDescription
    private let segments: JarvisRecordingSegmentSink?
    var completedSuccessfully: Bool { writer.status == .completed && (segments?.succeeded ?? true) }
    var streamedAsset: JarvisRecordingEncodedAsset? { segments?.completedAsset }
    init(configuration: JarvisRecordingWriter.Configuration, output: URL?, generationID: UUID) throws {
        var asbd = AudioStreamBasicDescription(mSampleRate: 48_000, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
        var format: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &asbd, layoutSize: 0, layout: nil,
            magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &format) == noErr,
              let format else { throw JarvisRecordingWriter.Reason.startFailed }
        audioFormat = format
        if let streaming = configuration.streaming {
            let sink = JarvisRecordingSegmentSink(generationID: generationID,
                maximumBytes: configuration.maximumFileBytes, stream: streaming.bytes)
            segments = sink
            writer = AVAssetWriter(contentType: .mpeg4Movie)
            writer.outputFileTypeProfile = .mpeg4AppleHLS
            writer.preferredOutputSegmentInterval = streaming.interval
            writer.initialSegmentStartTime = .zero
            writer.delegate = sink
        } else {
            guard let output else { throw JarvisRecordingWriter.Reason.invalidConfiguration }
            segments = nil
            writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
        }
        video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: configuration.width, AVVideoHeightKey: configuration.height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 4_000_000, AVVideoAllowFrameReorderingKey: false]])
        audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 128_000], sourceFormatHint: format)
        video.expectsMediaDataInRealTime = true; audio.expectsMediaDataInRealTime = true
        guard writer.canAdd(video), writer.canAdd(audio) else { throw JarvisRecordingWriter.Reason.startFailed }
        writer.add(video); writer.add(audio)
        guard writer.startWriting() else { throw JarvisRecordingWriter.Reason.startFailed }
        writer.startSession(atSourceTime: .zero)
    }
    func isReady(for track: JarvisRecordingWriter.Track) -> Bool {
        (track == .video ? video : audio).isReadyForMoreMediaData
    }
    func appendVideo(_ sample: CMSampleBuffer) throws {
        try segments?.check()
        guard video.append(sample) else { throw JarvisRecordingWriter.Reason.appendFailed }
        try segments?.check()
    }
    func appendAudio(_ samples: [Float], at time: CMTime) throws {
        try segments?.check()
        let bytes = samples.count * MemoryLayout<Float>.size
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: bytes,
            blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0, dataLength: bytes,
            flags: 0, blockBufferOut: &block) == kCMBlockBufferNoErr, let block else {
            throw JarvisRecordingWriter.Reason.appendFailed
        }
        let copied = samples.withUnsafeBytes { storage in
            CMBlockBufferReplaceDataBytes(with: storage.baseAddress!, blockBuffer: block, offsetIntoDestination: 0, dataLength: bytes)
        }
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 48_000), presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var size = 4
        var sample: CMSampleBuffer?
        guard copied == kCMBlockBufferNoErr,
              CMSampleBufferCreateReady(allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: audioFormat,
                sampleCount: samples.count, sampleTimingEntryCount: 1, sampleTimingArray: &timing,
                sampleSizeEntryCount: 1, sampleSizeArray: &size, sampleBufferOut: &sample) == noErr,
              let sample, audio.append(sample) else { throw JarvisRecordingWriter.Reason.appendFailed }
        try segments?.check()
    }
    func finish(at time: CMTime, completion: @escaping @Sendable (Bool) -> Void) {
        video.markAsFinished(); audio.markAsFinished(); writer.endSession(atSourceTime: time)
        // The callback only announces completion. Read AVAssetWriter status on
        // the owning serial queue, without sending the non-Sendable writer.
        let sink = segments
        writer.finishWriting { completion(sink?.finish() ?? true) }
    }
    func cancel() {
        if writer.status == .writing { writer.cancelWriting() }
        segments?.cancel()
    }
}

/// One bounded in-memory stream. No full recording or retry journal is written
/// locally. A slow/failed upload exhausts this finite buffer and fails honestly.
final class JarvisRecordingByteStream: @unchecked Sendable {
    struct Summary: Sendable {
        let generationID: UUID
        let byteCount: Int
        let contentSHA256: String
    }
    struct Chunk: Sendable { let offset: Int; let data: Data; let finalByteCount: Int? }
    enum Part: Sendable { case chunk(Chunk), finished(Summary) }
    enum Failure: Error { case failed, busy, invalidAcknowledgement }
    static let chunkBytes = 6 * 1024 * 1024
    static let maximumBufferedBytes = 2 * chunkBytes
    private typealias Waiter = CheckedContinuation<Part, Error>
    private let lock = NSLock()
    private var pending = Data()
    private var inFlight = 0
    private var offered = 0
    private var acknowledged = 0
    private var summary: Summary?
    private var failed = false
    private var waiter: Waiter?
    private var peak = 0
    var peakBufferedBytes: Int { lock.withLock { peak } }
    var isFailed: Bool { lock.withLock { failed } }

    func offer(_ data: Data) -> Bool {
        var ready: (Waiter, Result<Part, Error>)?
        let accepted = lock.withLock {
            guard !failed, summary == nil, !data.isEmpty,
                  data.count <= Self.maximumBufferedBytes - pending.count - inFlight else {
                failed = true; pending = Data(); ready = wakeLocked(); return false
            }
            pending.append(data); offered += data.count
            peak = max(peak, pending.count + inFlight)
            ready = wakeLocked(); return true
        }
        if let ready { ready.0.resume(with: ready.1) }
        return accepted
    }
    func finish(_ value: Summary) -> Bool {
        var ready: (Waiter, Result<Part, Error>)?
        let accepted = lock.withLock {
            guard !failed, summary == nil, value.byteCount == offered, offered > 0 else {
                failed = true; pending = Data(); ready = wakeLocked(); return false
            }
            summary = value; ready = wakeLocked(); return true
        }
        if let ready { ready.0.resume(with: ready.1) }
        return accepted
    }
    func fail() {
        let ready = lock.withLock { () -> (Waiter, Result<Part, Error>)? in
            failed = true; pending = Data(); return wakeLocked()
        }
        if let ready { ready.0.resume(with: ready.1) }
    }
    func next() async throws -> Part {
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                let immediate = lock.withLock { () -> Result<Part, Error>? in
                    if let value = popLocked() { return value }
                    guard waiter == nil else { return .failure(Failure.busy) }
                    waiter = continuation; return nil
                }
                if let immediate { continuation.resume(with: immediate) }
            }
        }, onCancel: { self.fail() })
    }
    func acknowledge(_ chunk: Chunk) throws {
        try lock.withLock {
            guard !failed, chunk.offset == acknowledged, chunk.data.count == inFlight, inFlight > 0 else {
                throw Failure.invalidAcknowledgement
            }
            acknowledged += inFlight; inFlight = 0
        }
    }
    private func popLocked() -> Result<Part, Error>? {
        if failed { return .failure(Failure.failed) }
        guard inFlight == 0 else { return .failure(Failure.busy) }
        if pending.count >= Self.chunkBytes || (summary != nil && !pending.isEmpty) {
            let count = min(Self.chunkBytes, pending.count)
            let data = Data(pending.prefix(count))
            pending = Data(pending.dropFirst(count)); inFlight = count
            return .success(.chunk(.init(offset: acknowledged, data: data, finalByteCount: summary?.byteCount)))
        }
        if let summary { return .success(.finished(summary)) }
        return nil
    }
    private func wakeLocked() -> (Waiter, Result<Part, Error>)? {
        guard let waiter, let result = popLocked() else { return nil }
        self.waiter = nil; return (waiter, result)
    }
}

/// Apple data-output API: initialization then media fragments, in stream order.
/// Each payload enters the bounded upload buffer; only rolling metadata remains
/// here. Unlike outputURL/movieFragmentInterval, finalization rewrites no prefix.
private final class JarvisRecordingSegmentSink: NSObject, AVAssetWriterDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let generationID: UUID
    private let maximumBytes: Int
    private let stream: JarvisRecordingByteStream
    private var digestState = SHA256()
    private var bytes = 0
    private var mediaFragments = 0
    private var initialized = false
    private var failed = false
    private var sealed = false
    private var asset: JarvisRecordingEncodedAsset?

    init(generationID: UUID, maximumBytes: Int, stream: JarvisRecordingByteStream) {
        self.generationID = generationID; self.maximumBytes = maximumBytes; self.stream = stream
        super.init()
    }
    func assetWriter(_ writer: AVAssetWriter, didOutputSegmentData data: Data,
                     segmentType: AVAssetSegmentType) {
        lock.withLock {
            guard !failed, !sealed, !data.isEmpty, data.count <= maximumBytes - bytes else {
                failed = true; stream.fail(); return
            }
            switch segmentType {
            case .initialization:
                guard !initialized else { failed = true; stream.fail(); return }
            case .separable:
                guard initialized else { failed = true; stream.fail(); return }
            @unknown default: failed = true; stream.fail(); return
            }
            guard stream.offer(data) else { failed = true; return }
            digestState.update(data: data); bytes += data.count
            if segmentType == .initialization { initialized = true } else { mediaFragments += 1 }
        }
    }
    func check() throws {
        if lock.withLock({ failed }) || stream.isFailed { throw JarvisRecordingWriter.Reason.fileFailed }
    }
    var succeeded: Bool { lock.withLock { sealed && !failed && asset != nil } }
    var completedAsset: JarvisRecordingEncodedAsset? { lock.withLock { asset } }
    func finish() -> Bool {
        lock.withLock {
            guard !sealed else { return !failed }
            sealed = true
            guard !failed, initialized, mediaFragments > 0 else { failed = true; stream.fail(); return false }
            let summary = JarvisRecordingByteStream.Summary(generationID: generationID, byteCount: bytes,
                contentSHA256: digestState.finalize().map { String(format: "%02x", $0) }.joined())
            guard stream.finish(summary) else { failed = true; return false }
            asset = JarvisRecordingEncodedAsset(streamed: summary)
            return true
        }
    }
    func cancel() {
        lock.withLock { if !sealed { failed = true; sealed = true; stream.fail() } }
    }
}

/// Exact completed encoding metadata. Local legacy captures own a directory;
/// streaming captures have NO local file. Neither form is a remote-save receipt.
final class JarvisRecordingEncodedAsset: @unchecked Sendable {
    let generationID: UUID
    let byteCount: Int
    let contentSHA256: String
    private let directory: JarvisRecordingDirectory?
    var isStoredLocally: Bool { directory != nil }
    fileprivate init(generationID: UUID, byteCount: Int, contentSHA256: String, directory: JarvisRecordingDirectory) {
        self.generationID = generationID; self.byteCount = byteCount; self.contentSHA256 = contentSHA256; self.directory = directory
    }
    fileprivate init(streamed: JarvisRecordingByteStream.Summary) {
        generationID = streamed.generationID; byteCount = streamed.byteCount
        contentSHA256 = streamed.contentSHA256; directory = nil
    }
    func urlForReading() throws -> URL {
        guard let directory else { throw JarvisRecordingWriter.Reason.fileFailed }
        return try directory.urlForReading()
    }
    func discard() { directory?.discard() }
    deinit { discard() }
}
// Compatibility for existing local-custody callers/tests. Streamed assets must
// explicitly use metadata; requesting their nonexistent local URL throws.
typealias JarvisRecordingLocalFile = JarvisRecordingEncodedAsset

final class JarvisRecordingDirectory: @unchecked Sendable {
    private let root: URL
    let file: URL
    private let removesFilesOnDiscard: Bool
    private let lock = NSLock()
    private var discarded = false
    init() throws {
        removesFilesOnDiscard = true
        root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-recording-" + UUID().uuidString, isDirectory: true)
        file = root.appendingPathComponent("original.mp4")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        #if os(iOS)
        do {
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: root.path)
        } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
        #endif
    }
    /// Store-owned bytes survive writer cancellation, failed completion and
    /// object release. Reusing an existing/partial destination is forbidden.
    init(storeOwned destination: JarvisRecordingWriter.Destination) throws {
        root = destination.directoryURL
        file = root.appendingPathComponent("original.mp4")
        removesFilesOnDiscard = false
        guard root.isFileURL else { throw JarvisRecordingWriter.Reason.invalidConfiguration }
        let fd = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw JarvisRecordingWriter.Reason.fileFailed }
        defer { close(fd) }
        var directoryInfo = stat(), fileInfo = stat()
        guard fstat(fd, &directoryInfo) == 0, directoryInfo.st_mode & S_IFMT == S_IFDIR,
              directoryInfo.st_uid == geteuid(), directoryInfo.st_mode & 0o077 == 0,
              fstatat(fd, "original.mp4", &fileInfo, AT_SYMLINK_NOFOLLOW) != 0, errno == ENOENT else {
            throw JarvisRecordingWriter.Reason.fileFailed
        }
    }
    func byteCount() throws -> Int {
        guard FileManager.default.fileExists(atPath: file.path) else { return 0 }
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        guard let size = attributes[.size] as? NSNumber else { throw JarvisRecordingWriter.Reason.fileFailed }
        return size.intValue
    }
    func completedFile(generationID: UUID, maximumBytes: Int) throws -> JarvisRecordingLocalFile {
        let count = try byteCount()
        guard count > 0, count <= maximumBytes else { throw JarvisRecordingWriter.Reason.fileFailed }
        try FileManager.default.setAttributes([.posixPermissions: 0o400], ofItemAtPath: file.path)
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256(), measured = 0
        while let block = try handle.read(upToCount: 64 * 1024), !block.isEmpty {
            measured += block.count
            guard measured <= maximumBytes else { throw JarvisRecordingWriter.Reason.fileFailed }
            hash.update(data: block)
        }
        guard measured == count else { throw JarvisRecordingWriter.Reason.fileFailed }
        return JarvisRecordingLocalFile(generationID: generationID, byteCount: count,
            contentSHA256: hash.finalize().map { String(format: "%02x", $0) }.joined(), directory: self)
    }
    func urlForReading() throws -> URL {
        try lock.withLock { guard !discarded else { throw JarvisRecordingWriter.Reason.closed }; return file }
    }
    func discard() {
        let remove = lock.withLock { if discarded { return false }; discarded = true; return true }
        if remove && removesFilesOnDiscard { try? FileManager.default.removeItem(at: root) }
    }
    deinit { discard() }
}
