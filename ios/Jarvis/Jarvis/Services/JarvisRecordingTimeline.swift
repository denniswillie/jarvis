@preconcurrency import CoreMedia
import CoreAudio
import Foundation

/// One immutable host-clock origin for a whole recording. DAT 0.8+ video uses
/// a glasses clock: map it once using a paired source/arrival observation, then
/// preserve source frame spacing. This estimate does not prove physical A/V sync.
struct JarvisRecordingTimeline: Sendable {
    enum MappingError: Error, Equatable {
        case invalidConfiguration, invalidTime, missingHostTime
        case incompatibleVideoClock, invalidAudioFormat, invalidVideoSample
        case sampleCopyFailed(Int32)
    }

    /// The pinned DAT 0.7 clock identity is a candidate supported by SDK-owner
    /// commentary, not a measured device guarantee. Never carry it into a new
    /// SDK version automatically. Receipt time is diagnostic, not capture time.
    enum VideoClock: String, Sendable {
        case dat07PhoneHostUnqualified
        case datGlassesReceiptAnchored
    }

    struct AudioWindow: Sendable {
        let sourceFrames: Range<Int>
        let start: CMTime
        let end: CMTime
        // Placement only. These do not perform resampling or certify coverage.
        let outputStartFrame: Int64
        let outputEndFrame: Int64
    }

    static let outputSampleRate: CMTimeScale = 48_000
    let origin: CMTime
    let end: CMTime
    let videoClock: VideoClock
    private let maximumVideoDeliveryDelay: CMTime
    private let videoHostOffset: CMTime

    init(startHostTime: UInt64, maximumDurationMilliseconds: Int,
         videoClock: VideoClock, maximumVideoDeliveryDelay: CMTime,
         videoReference: (presentationTime: CMTime, receivedHostTime: UInt64)? = nil) throws {
        guard startHostTime > 0, (1...1_800_000).contains(maximumDurationMilliseconds),
              Self.valid(maximumVideoDeliveryDelay),
              CMTimeCompare(maximumVideoDeliveryDelay, .zero) >= 0,
              CMTimeCompare(maximumVideoDeliveryDelay, CMTime(value: 60, timescale: 1)) <= 0 else {
            throw MappingError.invalidConfiguration
        }
        let origin = CMClockMakeHostTimeFromSystemUnits(startHostTime)
        guard Self.valid(origin) else { throw MappingError.invalidConfiguration }
        self.origin = origin
        self.end = CMTime(value: Int64(maximumDurationMilliseconds), timescale: 1_000)
        self.videoClock = videoClock
        self.maximumVideoDeliveryDelay = maximumVideoDeliveryDelay
        if videoClock == .datGlassesReceiptAnchored {
            guard let reference = videoReference, Self.valid(reference.presentationTime),
                  reference.receivedHostTime > 0 else { throw MappingError.invalidConfiguration }
            videoHostOffset = CMTimeSubtract(CMClockMakeHostTimeFromSystemUnits(reference.receivedHostTime),
                                             reference.presentationTime)
        } else {
            videoHostOffset = .zero
        }
    }

    func relativeHostTime(_ ticks: UInt64) throws -> CMTime {
        guard ticks > 0 else { throw MappingError.missingHostTime }
        let time = CMClockMakeHostTimeFromSystemUnits(ticks)
        guard Self.valid(time) else { throw MappingError.invalidTime }
        return CMTimeSubtract(time, origin)
    }

    /// A nil result means a valid frame outside the requested capture interval.
    /// Implausible clock values fail instead of being zeroed or arrival-anchored.
    func videoTime(_ presentationTime: CMTime, receivedHostTime: UInt64) throws -> CMTime? {
        guard Self.valid(presentationTime) else { throw MappingError.invalidTime }
        let arrival = try relativeHostTime(receivedHostTime)
        let relative = CMTimeSubtract(CMTimeAdd(presentationTime, videoHostOffset), origin)
        let delay = CMTimeSubtract(arrival, relative)
        // Arrival latency can decrease after the anchor. Do not re-anchor each
        // frame (which would turn transport jitter into uneven video timing).
        let minimumDelay = videoClock == .datGlassesReceiptAnchored
            ? CMTimeMultiply(maximumVideoDeliveryDelay, multiplier: -1) : .zero
        guard Self.valid(relative), Self.valid(delay),
              CMTimeCompare(delay, minimumDelay) >= 0,
              CMTimeCompare(delay, maximumVideoDeliveryDelay) <= 0 else {
            throw MappingError.incompatibleVideoClock
        }
        guard CMTimeCompare(relative, .zero) >= 0, CMTimeCompare(relative, end) < 0 else { return nil }
        return relative
    }

    /// Retimes the wrapper only: the original sample and image bytes remain
    /// unchanged. Its original capture duration is preserved for the writer.
    func videoCopy(_ original: CMSampleBuffer, receivedHostTime: UInt64) throws -> CMSampleBuffer? {
        guard CMSampleBufferDataIsReady(original), CMSampleBufferGetNumSamples(original) == 1,
              CMSampleBufferGetImageBuffer(original) != nil else {
            throw MappingError.invalidVideoSample
        }
        guard let presentation = try videoTime(CMSampleBufferGetPresentationTimeStamp(original),
                                               receivedHostTime: receivedHostTime) else { return nil }
        let originalDecode = CMSampleBufferGetDecodeTimeStamp(original)
        let decode: CMTime
        if originalDecode.isValid {
            guard Self.valid(originalDecode) else { throw MappingError.invalidTime }
            decode = CMTimeSubtract(CMTimeAdd(originalDecode, videoHostOffset), origin)
        } else {
            decode = .invalid
        }
        var timing = CMSampleTimingInfo(duration: CMSampleBufferGetDuration(original),
                                       presentationTimeStamp: presentation, decodeTimeStamp: decode)
        var copy: CMSampleBuffer?
        let status = CMSampleBufferCreateCopyWithNewTiming(allocator: kCFAllocatorDefault,
            sampleBuffer: original, sampleTimingEntryCount: 1, sampleTimingArray: &timing,
            sampleBufferOut: &copy)
        guard status == noErr, let copy else { throw MappingError.sampleCopyFailed(status) }
        return copy
    }

    /// Map the actual AudioTimeStamp host anchor, not callback arrival or a
    /// route-local sampleTime. Whole input samples straddling either boundary
    /// are trimmed; retained samples keep their actual offset on the same H0.
    /// The mixer must preserve gaps and use the actual sample rate to resample.
    func audioWindow(hostTime: UInt64, timestampFlags: UInt32, sampleRate: Double,
                     frames: Int) throws -> AudioWindow? {
        guard timestampFlags & AudioTimeStampFlags.hostTimeValid.rawValue != 0 else {
            throw MappingError.missingHostTime
        }
        guard sampleRate.isFinite, (8_000...192_000).contains(sampleRate),
              sampleRate.rounded() == sampleRate, (1...19_200).contains(frames) else {
            throw MappingError.invalidAudioFormat
        }
        let rate = CMTimeScale(sampleRate)
        let start = try relativeHostTime(hostTime)
        guard Self.valid(start) else { throw MappingError.invalidTime }
        let packetEnd = CMTimeAdd(start, CMTime(value: Int64(frames), timescale: rate))
        guard CMTimeCompare(packetEnd, .zero) > 0, CMTimeCompare(start, end) < 0 else { return nil }
        let first = max(Int64(0), CMTimeConvertScale(CMTimeSubtract(.zero, start),
            timescale: rate, method: .roundTowardPositiveInfinity).value)
        let limit = min(Int64(frames), CMTimeConvertScale(CMTimeSubtract(end, start),
            timescale: rate, method: .roundTowardNegativeInfinity).value)
        guard first < limit else { return nil }
        let retainedStart = CMTimeAdd(start, CMTime(value: first, timescale: rate))
        let retainedEnd = CMTimeAdd(start, CMTime(value: limit, timescale: rate))
        return AudioWindow(sourceFrames: Int(first)..<Int(limit), start: retainedStart, end: retainedEnd,
            outputStartFrame: Self.outputFrame(retainedStart), outputEndFrame: Self.outputFrame(retainedEnd))
    }

    private static func outputFrame(_ time: CMTime) -> Int64 {
        CMTimeConvertScale(time, timescale: outputSampleRate, method: .roundHalfAwayFromZero).value
    }
    private static func valid(_ time: CMTime) -> Bool {
        time.isNumeric && time.timescale > 0 && time.epoch == 0
    }
}
