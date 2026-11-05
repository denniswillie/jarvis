@preconcurrency import AVFoundation
import CoreAudio
import Foundation

/// Serial recording-worker component, never called from hardware callbacks.
/// Bounded voice/microphone plus optional consumed media share one mono48k output.
/// AVAudioConverter persists across contiguous input; temporary lack of input
/// is not EOS. This performs nominal-rate conversion, not clock calibration.
final class JarvisRecordingMixer {
    var isEchoReductionEnabled: Bool {
        #if os(iOS) || JARVIS_ECHO_TEST
        return echoFilter != nil
        #else
        return false
        #endif
    }

    func enableEchoReduction() {
        #if os(iOS) || JARVIS_ECHO_TEST
        guard echoFilter == nil else { return }
        #if os(iOS)
        echoFilter = JarvisNeuralRecordingEchoFilter { [weak self] in self?.outputSink($0) }
        #else
        echoFilter = JarvisRecordingEchoFilter { [weak self] in self?.outputSink($0) }
        #endif
        #endif
    }

    private func trackEchoMix(microphone: Float, rendered: Float, media: Float) -> Float {
        let direct = (includeOutput ? rendered : 0) + (includeMedia ? media : 0)
        #if os(iOS) || JARVIS_ECHO_TEST
        if echoFilter != nil {
            echoMicrophone.append(microphone)
            echoReference.append(max(-1, min(1, rendered + media)))
            echoDirect.append(direct)
        }
        #endif
        return microphone + direct
    }

    private func deliverEchoChunk(_ chunk: Chunk) {
        #if os(iOS) || JARVIS_ECHO_TEST
        if let echoFilter {
            // The existing mixer owns timestamp alignment and gap handling.
            assert(echoMicrophone.count == chunk.samples.count)
            echoFilter.offer(startFrame: chunk.startFrame, microphone: echoMicrophone,
                             reference: echoReference, direct: echoDirect)
            echoMicrophone.removeAll(keepingCapacity: true)
            echoReference.removeAll(keepingCapacity: true)
            echoDirect.removeAll(keepingCapacity: true)
            return
        }
        #endif
        outputSink(chunk)
    }

    private func finishEchoReduction() {
        #if os(iOS) || JARVIS_ECHO_TEST
        echoFilter?.finish()
        echoFilter?.finishDiagnostic()
        #endif
    }

    private func resetEchoReduction() {
        #if os(iOS) || JARVIS_ECHO_TEST
        echoFilter?.reset()
        #endif
    }
    // Created only by the recording owner; never used by the live audio route.
    #if os(iOS) || JARVIS_ECHO_TEST
    #if os(iOS)
    private var echoFilter: JarvisNeuralRecordingEchoFilter?
    #else
    private var echoFilter: JarvisRecordingEchoFilter?
    #endif
    private var echoMicrophone: [Float] = []
    private var echoReference: [Float] = []
    private var echoDirect: [Float] = []
    #endif
    enum Track: UInt8, Sendable { case microphone = 1, renderedOutput = 2, media = 4 }
    enum MixerError: Error, Equatable {
        case closed, invalidInput, staleRoute, outOfOrder, lateInput, bufferFull
        case converterUnavailable, conversionFailed, conversionBudget, invalidFlush
    }
    enum Admission { case accepted, outsideWindow, excluded }
    struct Input {
        let track: Track
        let routeGeneration: UInt64
        let hostTime: UInt64
        let timestampFlags: UInt32
        let sampleTime: Double
        let sampleRate: Double
        let channels: Int
        let frames: Int
        let pcm16Interleaved: Data
    }
    struct Chunk {
        let startFrame: Int64
        let samples: [Float]
        var presentationTime: CMTime { CMTime(value: startFrame, timescale: 48_000) }
    }
    struct Gap {
        let track: Track
        let startFrame: Int64
        let endFrame: Int64
        let expiredWait: Bool
    }
    struct Evidence {
        var microphoneFrames: Int64 = 0
        var renderedFrames: Int64 = 0
        var mediaFrames: Int64 = 0
        var missingMicrophoneFrames: Int64 = 0
        var missingRenderedFrames: Int64 = 0
        var mixedFrames: Int64 = 0
        var clippedFrames: Int64 = 0
        var lateConvertedFrames: Int64 = 0
        var duplicateConvertedFrames: Int64 = 0
        var conversionShortfallFrames: Int64 = 0
        var routeChanges = 0
        var gaps: [Gap] = []
        var omittedGapDetails = 0
    }
    static let capacity = 96_000 // Two seconds, independent of capture length.
    static let chunkFrames = 4_800
    private let timeline: JarvisRecordingTimeline
    private let includeOutput: Bool
    private let includeMedia: Bool
    private let outputSink: (Chunk) -> Void
    private var sink: (Chunk) -> Void { { [unowned self] chunk in self.deliverEchoChunk(chunk) } }
    private let outputFormat = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
    private var microphone = [Float](repeating: 0, count: capacity)
    private var rendered = [Float](repeating: 0, count: capacity)
    private var media = [Float](repeating: 0, count: capacity)
    private var present = [UInt8](repeating: 0, count: capacity)
    private var segments: [Track: Segment] = [:]
    private var routeGeneration: UInt64?
    private(set) var cursor: Int64 = 0
    private(set) var evidence = Evidence()
    private var closed = false

    private final class Segment {
        let rate: Double
        let channels: Int
        let start: CMTime
        let startFrame: Int64
        let converter: AVAudioConverter?
        let output: AVAudioPCMBuffer
        var inputFrames: Int64 = 0
        var outputCursor: Int64
        var expectedEnd: Int64
        var lastInputEnd: CMTime
        var nextSampleTime: Double?
        init(input: Input, window: JarvisRecordingTimeline.AudioWindow,
             outputFormat: AVAudioFormat) throws {
            rate = input.sampleRate; channels = input.channels; start = window.start
            startFrame = window.outputStartFrame; outputCursor = startFrame
            expectedEnd = startFrame; lastInputEnd = start
            guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat,
                                                frameCapacity: AVAudioFrameCount(JarvisRecordingMixer.chunkFrames)) else {
                throw MixerError.converterUnavailable
            }
            self.output = output
            if rate == 48_000 { converter = nil }
            else {
                guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
                      let converter = AVAudioConverter(from: format, to: outputFormat) else {
                    throw MixerError.converterUnavailable
                }
                // Normal priming waits for lookahead but does not shift output.
                // .none would add through-latency; never reset on every packet.
                converter.primeMethod = .normal
                self.converter = converter
            }
        }
    }

    init(timeline: JarvisRecordingTimeline, includeJarvisAudio: Bool, includeMediaAudio: Bool = false,
         sink: @escaping (Chunk) -> Void) {
        self.timeline = timeline; includeOutput = includeJarvisAudio
        includeMedia = includeMediaAudio; self.outputSink = sink
    }

    @discardableResult func offer(_ input: Input) throws -> Admission {
        guard !closed else { throw MixerError.closed }
        if input.track == .renderedOutput && (!includeOutput && !isEchoReductionEnabled) { return .excluded }
        if input.track == .media && (!includeMedia && !isEchoReductionEnabled) { return .excluded }
        guard input.routeGeneration > 0, (1...2).contains(input.channels),
              (1...19_200).contains(input.frames),
              input.pcm16Interleaved.count == input.frames * input.channels * 2 else {
            throw MixerError.invalidInput
        }
        let hasSampleTime = input.timestampFlags & AudioTimeStampFlags.sampleTimeValid.rawValue != 0
        guard !hasSampleTime || input.sampleTime.isFinite else { throw MixerError.invalidInput }
        guard let window = try timeline.audioWindow(hostTime: input.hostTime, timestampFlags: input.timestampFlags,
            sampleRate: input.sampleRate, frames: input.frames) else { return .outsideWindow }
        guard window.outputStartFrame >= cursor else { throw MixerError.lateInput }
        guard window.outputEndFrame <= cursor + Int64(Self.capacity) else { throw MixerError.bufferFull }
        if let routeGeneration, input.routeGeneration < routeGeneration { throw MixerError.staleRoute }
        if let old = routeGeneration, old != input.routeGeneration {
            try closeSegments(); resetEchoReduction(); evidence.routeChanges += 1
        }
        routeGeneration = input.routeGeneration
        let sourceSampleTime = hasSampleTime ? input.sampleTime + Double(window.sourceFrames.lowerBound) : nil
        if let previous = segments[input.track] {
            guard previous.rate == input.sampleRate, previous.channels == input.channels else {
                throw MixerError.invalidInput
            }
            let delta = CMTimeGetSeconds(CMTimeSubtract(window.start, previous.lastInputEnd))
            let tolerance = 1 / input.sampleRate
            guard delta >= -tolerance else { throw MixerError.outOfOrder }
            let expectedHost = CMTimeAdd(previous.start,
                CMTime(value: previous.inputFrames, timescale: CMTimeScale(previous.rate)))
            let drift = abs(CMTimeGetSeconds(CMTimeSubtract(window.start, expectedHost)))
            let sampleContinuous = sourceSampleTime.flatMap { now in previous.nextSampleTime.map { abs(now - $0) < 0.000_001 } } ?? true
            if abs(delta) > tolerance || drift > tolerance || !sampleContinuous {
                try close(input.track, previous)
                segments.removeValue(forKey: input.track)
            }
        }
        let segment: Segment
        if let previous = segments[input.track] { segment = previous }
        else {
            segment = try Segment(input: input, window: window, outputFormat: outputFormat)
            segments[input.track] = segment
        }
        let count = window.sourceFrames.count
        guard let format = AVAudioFormat(standardFormatWithSampleRate: input.sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)),
              let floats = buffer.floatChannelData?[0] else { throw MixerError.converterUnavailable }
        buffer.frameLength = AVAudioFrameCount(count)
        input.pcm16Interleaved.withUnsafeBytes { bytes in
            for (destination, source) in window.sourceFrames.enumerated() {
                var sum: Float = 0
                for channel in 0..<input.channels {
                    let raw = bytes.loadUnaligned(fromByteOffset: (source * input.channels + channel) * 2, as: Int16.self)
                    sum += Float(Int16(littleEndian: raw)) / 32_768
                }
                floats[destination] = sum / Float(input.channels)
            }
        }
        segment.inputFrames += Int64(count)
        segment.lastInputEnd = window.end
        segment.nextSampleTime = sourceSampleTime.map { $0 + Double(count) }
        // One cumulative ratio, not separately rounded packet lengths.
        let length = CMTimeConvertScale(CMTime(value: segment.inputFrames, timescale: CMTimeScale(segment.rate)),
                                        timescale: 48_000, method: .roundHalfAwayFromZero).value
        segment.expectedEnd = min(segment.startFrame + length, maximumEnd)
        if segment.converter == nil { try place(floats, count: count, track: input.track, segment: segment) }
        else { try convert(segment, track: input.track, input: buffer, ending: false) }
        return .accepted
    }

    /// Ordinary flush waits for both converters' actual produced coverage.
    /// Only a caller's explicit expired wait may finalize missing track data.
    func drain(through frame: Int64, expiringMissing: Bool = false) throws {
        guard !closed else { throw MixerError.closed }
        guard frame >= cursor, frame <= maximumEnd else { throw MixerError.invalidFlush }
        let tracks: [Track] = includeOutput ? [.microphone, .renderedOutput] : [.microphone]
        let finalized = tracks.map { segments[$0].map { min($0.outputCursor, $0.expectedEnd) } ?? cursor }.min() ?? cursor
        flush(through: expiringMissing ? frame : min(frame, max(cursor, finalized)), expired: expiringMissing)
    }

    func finish(through frame: Int64) throws {
        guard !closed else { throw MixerError.closed }
        guard frame >= cursor, frame <= maximumEnd else { throw MixerError.invalidFlush }
        try closeSegments()
        flush(through: frame, expired: false)
        finishEchoReduction(); closed = true
    }

    private var maximumEnd: Int64 {
        CMTimeConvertScale(timeline.end, timescale: 48_000, method: .roundTowardNegativeInfinity).value
    }
    private func closeSegments() throws {
        for track in [Track.microphone, .renderedOutput, .media] {
            if let segment = segments[track] { try close(track, segment) }
        }
        segments.removeAll(keepingCapacity: true)
    }
    private func close(_ track: Track, _ segment: Segment) throws {
        if segment.converter != nil { try convert(segment, track: track, input: nil, ending: true) }
        evidence.conversionShortfallFrames += max(0, segment.expectedEnd - segment.outputCursor)
    }
    private func convert(_ segment: Segment, track: Track, input: AVAudioPCMBuffer?, ending: Bool) throws {
        guard let converter = segment.converter else { return }
        var supplied = false
        // Maximum input packet is19,200frames at>=8k: <=115,200output frames.
        // This fixed budget also bounds draining a converter tail at true EOS.
        for _ in 0..<32 {
            segment.output.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: segment.output, error: &error) { _, state in
                if let input, !supplied { supplied = true; state.pointee = .haveData; return input }
                state.pointee = ending ? .endOfStream : .noDataNow
                return nil
            }
            guard error == nil, status != .error else { throw MixerError.conversionFailed }
            let count = Int(segment.output.frameLength)
            if count > 0, let samples = segment.output.floatChannelData?[0] {
                try place(samples, count: count, track: track, segment: segment)
            }
            if status == .endOfStream || status == .inputRanDry { return }
            guard count > 0 else { throw MixerError.conversionFailed }
        }
        throw MixerError.conversionBudget
    }
    private func place(_ samples: UnsafePointer<Float>, count: Int, track: Track, segment: Segment) throws {
        let start = segment.outputCursor
        let end = min(start + Int64(count), segment.expectedEnd)
        guard end <= cursor + Int64(Self.capacity) else { throw MixerError.bufferFull }
        if end > start {
            for frame in start..<end {
                guard frame >= cursor else { evidence.lateConvertedFrames += 1; continue }
                let index = Int(frame % Int64(Self.capacity))
                guard present[index] & track.rawValue == 0 else { evidence.duplicateConvertedFrames += 1; continue }
                let value = samples[Int(frame - start)]
                guard value.isFinite else { throw MixerError.conversionFailed }
                switch track {
                case .microphone: microphone[index] = value
                case .renderedOutput: rendered[index] = value
                case .media: media[index] = value
                }
                present[index] |= track.rawValue
            }
        }
        // Advance by actual produced frames, including codec tail outside the
        // declared interval. Never assign buffered output to a later packet PTS.
        segment.outputCursor += Int64(count)
    }

    private func flush(through target: Int64, expired: Bool) {
        let tracks: [Track] = includeOutput ? [.microphone, .renderedOutput] : [.microphone]
        let scanEnd = min(target, cursor + Int64(Self.capacity))
        var gapStarts: [Track: Int64] = [:]
        var chunk: [Float] = []
        var chunkStart = cursor
        func emit() {
            if !chunk.isEmpty { sink(Chunk(startFrame: chunkStart, samples: chunk)); chunk.removeAll(keepingCapacity: true) }
        }
        for frame in cursor..<scanEnd {
            let index = Int(frame % Int64(Self.capacity)), mask = present[index]
            for track in tracks {
                if mask & track.rawValue == 0 {
                    if gapStarts[track] == nil { gapStarts[track] = frame }
                } else {
                    if let start = gapStarts.removeValue(forKey: track) { gap(track, start, frame, expired) }
                    if track == .microphone { evidence.microphoneFrames += 1 } else { evidence.renderedFrames += 1 }
                }
            }
            if mask & Track.media.rawValue != 0 { evidence.mediaFrames += 1 }
            if mask == 0 { emit() }
            else {
                if chunk.isEmpty { chunkStart = frame }
            let sum = trackEchoMix(microphone: microphone[index], rendered: rendered[index], media: media[index])
                if sum < -1 || sum > 1 { evidence.clippedFrames += 1 }
                chunk.append(min(1, max(-1, sum))); evidence.mixedFrames += 1
                if chunk.count == Self.chunkFrames { emit() }
            }
            present[index] = 0; microphone[index] = 0; rendered[index] = 0; media[index] = 0
        }
        emit()
        for track in tracks {
            if let start = gapStarts[track] { gap(track, start, target, expired) }
            else if target > scanEnd { gap(track, scanEnd, target, expired) }
        }
        cursor = target
    }
    private func gap(_ track: Track, _ start: Int64, _ end: Int64, _ expired: Bool) {
        guard end > start else { return }
        if track == .microphone { evidence.missingMicrophoneFrames += end - start }
        else { evidence.missingRenderedFrames += end - start }
        if evidence.gaps.count < 128 {
            evidence.gaps.append(Gap(track: track, startFrame: start, endFrame: end, expiredWait: expired))
        } else { evidence.omittedGapDetails += 1 }
    }
}

#if os(iOS) || JARVIS_ECHO_TEST
/// Serial recording-worker adapter. Buffers at most one partial 10 ms block;
/// neither padding nor processing latency changes the media timestamps.
private final class JarvisRecordingEchoFilter {
    #if DEBUG
    private let diagnostic = JarvisRecordingEchoDiagnostic.configured()
    #endif
    private let state: UnsafeMutableRawPointer
    private let output: (JarvisRecordingMixer.Chunk) -> Void
    private var microphone: [Float] = []
    private var reference: [Float] = []
    private var direct: [Float] = []
    private var start: Int64 = 0
    private var expectedNext: Int64?

    init?(output: @escaping (JarvisRecordingMixer.Chunk) -> Void) {
        guard let state = JarvisEchoCreate() else { return nil }
        self.state = state; self.output = output
        microphone.reserveCapacity(480); reference.reserveCapacity(480); direct.reserveCapacity(480)
    }
    deinit { JarvisEchoDestroy(state) }

    func offer(startFrame: Int64, microphone mic: [Float], reference ref: [Float], direct mix: [Float]) {
        guard mic.count == ref.count, mic.count == mix.count else { return }
        if let expectedNext, expectedNext != startFrame { reset() }
        var completed: [Float] = []
        var completedStart: Int64?
        for index in mic.indices {
            if microphone.isEmpty { start = startFrame + Int64(index) }
            microphone.append(mic[index]); reference.append(ref[index]); direct.append(mix[index])
            if microphone.count == 480 {
                if completedStart == nil { completedStart = start }
                completed.append(contentsOf: processBlock())
            }
        }
        expectedNext = startFrame + Int64(mic.count)
        if let completedStart { output(.init(startFrame: completedStart, samples: completed)) }
    }

    func finish() {
        if !microphone.isEmpty { output(.init(startFrame: start, samples: processBlock())) }
        expectedNext = nil
    }

    func reset() { finish(); JarvisEchoReset(state) }

    func finishDiagnostic() {
        #if DEBUG
        diagnostic?.finish()
        #endif
    }

    private func processBlock() -> [Float] {
        let count = microphone.count
        microphone.append(contentsOf: repeatElement(0, count: 480 - count))
        reference.append(contentsOf: repeatElement(0, count: 480 - count))
        var clean = [Float](repeating: 0, count: 480)
        microphone.withUnsafeBufferPointer { mic in
            reference.withUnsafeBufferPointer { ref in
                clean.withUnsafeMutableBufferPointer { dst in
                    JarvisEchoProcess(state, mic.baseAddress!, ref.baseAddress!, dst.baseAddress!)
                }
            }
        }
        #if DEBUG
        diagnostic?.append(startFrame: start, microphone: microphone, reference: reference,
                           cleaned: clean, direct: direct, count: count)
        #endif
        let result = (0..<count).map { max(-1, min(1, clean[$0] + direct[$0])) }
        microphone.removeAll(keepingCapacity: true)
        reference.removeAll(keepingCapacity: true)
        direct.removeAll(keepingCapacity: true)
        return result
    }
}
#endif
