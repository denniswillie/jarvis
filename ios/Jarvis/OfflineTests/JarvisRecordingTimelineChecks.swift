import CoreAudio
import CoreMedia
import CoreVideo
import Foundation

@main
struct RecordingTimelineChecks {
    typealias Timeline = JarvisRecordingTimeline
    static let origin = CMTime(value: 120, timescale: 1)
    static let flags = AudioTimeStampFlags.hostTimeValid.rawValue

    static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError(message) }
    }
    static func ticks(_ milliseconds: Int64) -> UInt64 {
        CMClockConvertHostTimeToSystemUnits(CMTimeAdd(origin, CMTime(value: milliseconds, timescale: 1_000)))
    }
    static func absolute(_ milliseconds: Int64) -> CMTime {
        CMClockMakeHostTimeFromSystemUnits(ticks(milliseconds))
    }
    static func expect(_ error: Timeline.MappingError, _ action: () throws -> Void) {
        do { try action(); fatalError("Expected \(error)") }
        catch let actual as Timeline.MappingError { require(actual == error, "Wrong error: \(actual)") }
        catch { fatalError("Unexpected error: \(error)") }
    }
    static func near(_ value: CMTime, _ milliseconds: Int64) -> Bool {
        abs(CMTimeGetSeconds(value) - Double(milliseconds) / 1_000) < 0.000_001
    }
    static func frame(_ pts: CMTime) throws -> CMSampleBuffer {
        var pixel: CVPixelBuffer?
        require(CVPixelBufferCreate(kCFAllocatorDefault, 16, 16, kCVPixelFormatType_32BGRA,
                                   nil, &pixel) == kCVReturnSuccess, "Pixel allocation")
        var format: CMVideoFormatDescription?
        require(CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                imageBuffer: pixel!, formatDescriptionOut: &format) == noErr, "Video format")
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 24),
                                       presentationTimeStamp: pts, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        require(CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault,
            imageBuffer: pixel!, formatDescription: format!, sampleTiming: &timing,
            sampleBufferOut: &sample) == noErr, "Sample creation")
        return sample!
    }

    static func main() throws {
        let glasses = try Timeline(startHostTime: ticks(0), maximumDurationMilliseconds: 60_000,
            videoClock: .datGlassesReceiptAnchored, maximumVideoDeliveryDelay: CMTime(value: 2, timescale: 1),
            videoReference: (CMTime(value: 7, timescale: 1), ticks(100)))
        let mapped = try glasses.videoTime(CMTime(value: 7050, timescale: 1000), receivedHostTime: ticks(130))!
        require(near(mapped, 150), "Glasses clock must map to the phone clock despite decreasing transport latency")
        let jittered = try glasses.videoTime(CMTime(value: 7150, timescale: 1000), receivedHostTime: ticks(350))!
        require(near(CMTimeSubtract(jittered, mapped), 100), "Transport jitter changed source frame spacing")
        let glassesCopy = try glasses.videoCopy(frame(CMTime(value: 7050, timescale: 1000)), receivedHostTime: ticks(130))!
        require(near(CMSampleBufferGetPresentationTimeStamp(glassesCopy), 150), "Glasses wrapper mapping")
        expect(.incompatibleVideoClock) {
            _ = try glasses.videoTime(CMTime(value: 1, timescale: 1), receivedHostTime: ticks(200))
        }
        expect(.invalidConfiguration) {
            _ = try Timeline(startHostTime: ticks(0), maximumDurationMilliseconds: 60_000,
                videoClock: .datGlassesReceiptAnchored, maximumVideoDeliveryDelay: CMTime(value: 2, timescale: 1))
        }
        print("PASS: independent glasses clock, fixed offset, jitter and reset rejection")
        let timeline = try Timeline(startHostTime: ticks(0), maximumDurationMilliseconds: 60_000,
            videoClock: .dat07PhoneHostUnqualified, maximumVideoDeliveryDelay: CMTime(value: 2, timescale: 1))
        require(timeline.videoClock == .dat07PhoneHostUnqualified, "Mapping must not claim qualification")
        expect(.invalidConfiguration) {
            _ = try Timeline(startHostTime: 0, maximumDurationMilliseconds: 60_000,
                videoClock: .dat07PhoneHostUnqualified, maximumVideoDeliveryDelay: .zero)
        }
        expect(.invalidConfiguration) {
            _ = try Timeline(startHostTime: ticks(0), maximumDurationMilliseconds: 1_800_001,
                videoClock: .dat07PhoneHostUnqualified, maximumVideoDeliveryDelay: .zero)
        }
        print("PASS 1: explicit unqualified clock identity and bounded configuration")

        let thirtyMinutes = try Timeline(startHostTime: ticks(0), maximumDurationMilliseconds: 1_800_000,
            videoClock: .dat07PhoneHostUnqualified, maximumVideoDeliveryDelay: CMTime(value: 1, timescale: 1))
        let nearEnd = try thirtyMinutes.videoTime(absolute(1_799_900), receivedHostTime: ticks(1_799_950))
        let atEnd = try thirtyMinutes.videoTime(absolute(1_800_000), receivedHostTime: ticks(1_800_000))
        require(nearEnd != nil && atEnd == nil, "Thirty-minute boundary accepts the final interval and rejects later video")
        let lastAudio = try thirtyMinutes.audioWindow(hostTime: ticks(1_799_990), timestampFlags: flags,
            sampleRate: 48_000, frames: 960)
        require(lastAudio?.sourceFrames.count == 480 && lastAudio.map { near($0.end, 1_800_000) } == true,
            "Audio straddling thirty-minutes is trimmed to the same exact endpoint")
        print("PASS: thirty-minute common audio/video endpoint")

        let video = try timeline.videoTime(absolute(100), receivedHostTime: ticks(300))!
        let audio = try timeline.audioWindow(hostTime: ticks(250), timestampFlags: flags,
                                             sampleRate: 48_000, frames: 480)!
        require(near(video, 100) && near(audio.start, 250), "Independent first-sample zeroing")
        require(audio.outputStartFrame == 12_000 && audio.outputEndFrame == 12_480, "48k placement")
        require(near(CMTimeSubtract(audio.start, video), 150), "Cross-stream offset lost")
        print("PASS 2: one origin preserves actual video/audio start offsets")

        let past = try timeline.videoTime(absolute(-50), receivedHostTime: ticks(50))
        let end = try timeline.videoTime(absolute(60_000), receivedHostTime: ticks(60_100))
        require(past == nil && end == nil, "Out-of-window frame not dropped")
        expect(.incompatibleVideoClock) { _ = try timeline.videoTime(.zero, receivedHostTime: ticks(200)) }
        expect(.incompatibleVideoClock) { _ = try timeline.videoTime(absolute(300), receivedHostTime: ticks(200)) }
        expect(.incompatibleVideoClock) { _ = try timeline.videoTime(absolute(100), receivedHostTime: ticks(3_000)) }
        expect(.invalidTime) { _ = try timeline.videoTime(.indefinite, receivedHostTime: ticks(200)) }
        expect(.invalidTime) {
            _ = try timeline.videoTime(CMTime(value: 120, timescale: 1, flags: .valid, epoch: 1),
                                       receivedHostTime: ticks(200))
        }
        print("PASS 3: incompatible clocks fail, never receipt-anchor or silently clamp")

        let leading = try timeline.audioWindow(hostTime: ticks(-5), timestampFlags: flags,
                                               sampleRate: 48_000, frames: 480)!
        let trailing = try timeline.audioWindow(hostTime: ticks(59_995), timestampFlags: flags,
                                                sampleRate: 48_000, frames: 480)!
        require(leading.sourceFrames == 240..<480 && near(leading.start, 0) && near(leading.end, 5), "Leading trim")
        require(trailing.sourceFrames == 0..<240 && near(trailing.end, 60_000), "Trailing trim")
        let before = try timeline.audioWindow(hostTime: ticks(-20), timestampFlags: flags,
                                              sampleRate: 48_000, frames: 480)
        let after = try timeline.audioWindow(hostTime: ticks(60_000), timestampFlags: flags,
                                             sampleRate: 48_000, frames: 480)
        require(before == nil && after == nil, "Out-of-window audio not dropped")
        print("PASS 4: sample-accurate boundary trimming without timeline compression")

        let firstRoute = try timeline.audioWindow(hostTime: ticks(400), timestampFlags: flags,
                                                  sampleRate: 48_000, frames: 480)!
        let nextRoute = try timeline.audioWindow(hostTime: ticks(700), timestampFlags: flags,
                                                 sampleRate: 16_000, frames: 160)!
        require(near(CMTimeSubtract(nextRoute.start, firstRoute.end), 290), "Route gap compressed")
        require(nextRoute.outputStartFrame == 33_600 && nextRoute.outputEndFrame == 34_080, "Negotiated rate ignored")
        let rate441 = try timeline.audioWindow(hostTime: ticks(1_000), timestampFlags: flags,
                                               sampleRate: 44_100, frames: 441)!
        require(rate441.outputEndFrame - rate441.outputStartFrame == 480, "44.1k conversion")
        expect(.missingHostTime) {
            _ = try timeline.audioWindow(hostTime: ticks(200), timestampFlags: AudioTimeStampFlags.sampleTimeValid.rawValue,
                                         sampleRate: 48_000, frames: 480)
        }
        expect(.missingHostTime) {
            _ = try timeline.audioWindow(hostTime: 0, timestampFlags: flags, sampleRate: 48_000, frames: 480)
        }
        for rate in [Double.nan, Double.infinity, 0, 48_000.5, 192_001] {
            expect(.invalidAudioFormat) {
                _ = try timeline.audioWindow(hostTime: ticks(200), timestampFlags: flags, sampleRate: rate, frames: 480)
            }
        }
        print("PASS 5: gaps and actual rate survive route changes; missing host timestamps fail")

        let original = try frame(absolute(500))
        let copy = try timeline.videoCopy(original, receivedHostTime: ticks(600))!
        require(near(CMSampleBufferGetPresentationTimeStamp(copy), 500), "Wrapper PTS")
        require(CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(original), absolute(500)) == 0, "Original mutated")
        require(CMTimeCompare(CMSampleBufferGetDuration(copy), CMTime(value: 1, timescale: 24)) == 0, "Duration changed")
        require(!CMSampleBufferGetDecodeTimeStamp(copy).isValid, "Invented decode timestamp")
        require(CMSampleBufferGetImageBuffer(copy) === CMSampleBufferGetImageBuffer(original), "Pixel payload copied")
        print("PASS 6: real sample-buffer wrapper retiming preserves source and image")
        print("PASS: 6 common-timeline groups; synthetic timing, no hardware synchronization claim")
    }
}
