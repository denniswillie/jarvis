import CoreMedia
import CoreVideo
import Foundation

private final class Received: @unchecked Sendable {
    let lock = NSLock()
    var values: [JarvisRecordingVideoSource.Event] = []
    func append(_ value: JarvisRecordingVideoSource.Event) { lock.withLock { values.append(value) } }
    var snapshot: [JarvisRecordingVideoSource.Event] { lock.withLock { values } }
}
private func sample(_ index: Int64) -> CMSampleBuffer {
    var image: CVPixelBuffer?
    precondition(CVPixelBufferCreate(nil, 16, 16, kCVPixelFormatType_32BGRA, nil, &image) == kCVReturnSuccess)
    var format: CMVideoFormatDescription?
    precondition(CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: image!,
                                                          formatDescriptionOut: &format) == noErr)
    var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 24),
        presentationTimeStamp: CMTime(value: 1000 + index, timescale: 24), decodeTimeStamp: .invalid)
    var result: CMSampleBuffer?
    precondition(CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: image!,
        formatDescription: format!, sampleTiming: &timing, sampleBufferOut: &result) == noErr)
    return result!
}
private func wait(_ signal: DispatchSemaphore) {
    precondition(signal.wait(timeout: .now() + 3) == .success, "bounded delivery timed out")
}

@main
struct RecordingVideoSourceChecks {
    static func main() {
        let source = JarvisRecordingVideoSource(), received = Received(), capture = UUID()
        let generation = source.beginGeneration()
        precondition(source.subscribe(captureID: capture, sourceGeneration: generation, sink: received.append) == nil)
        precondition(source.offer(sample(0), sourceGeneration: generation, receivedHostTime: 500) == .inactive)
        precondition(source.latest?.width == 16 && source.latest?.sequence == 1)
        precondition(source.subscribe(captureID: capture, sourceGeneration: generation, sink: received.append) != nil)
        precondition(source.subscribe(captureID: UUID(), sourceGeneration: generation, sink: received.append) == nil)
        for index in 1...4 {
            precondition(source.offer(sample(Int64(index)), sourceGeneration: generation,
                                      receivedHostTime: UInt64(500 + index)) == .queued)
        }
        let finished = DispatchSemaphore(value: 0)
        precondition(!source.detach(captureID: UUID(), completion: {}))
        precondition(source.detach(captureID: capture, completion: { finished.signal() }))
        wait(finished)
        let first = received.snapshot
        precondition(first.count == 5)
        for index in 0..<4 {
            guard case .frame(let frame) = first[index] else { preconditionFailure() }
            precondition(frame.captureID == capture && frame.snapshot.sourceGeneration == generation)
            precondition(frame.snapshot.sequence == UInt64(index + 2))
            precondition(frame.snapshot.receivedHostTime == UInt64(501 + index))
            precondition(CMTimeCompare(frame.snapshot.presentationTime, CMTime(value: Int64(1001 + index), timescale: 24)) == 0)
            precondition(CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(frame.sampleBuffer), frame.snapshot.presentationTime) == 0)
        }
        guard case .ended(_, _, .detached, let count) = first.last! else { preconditionFailure() }
        precondition(count.observed == 5 && count.queued == 4 && count.overflow == 0)

        // A blocked recorder cannot block intake or grow its frame queue.
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        let overflowEvents = Received(), overflowCapture = UUID()
        precondition(source.subscribe(captureID: overflowCapture, sourceGeneration: generation) { event in
            if case .frame(let frame) = event, frame.snapshot.sequence == 6 { entered.signal(); wait(release) }
            overflowEvents.append(event)
        } != nil)
        precondition(source.offer(sample(5), sourceGeneration: generation, receivedHostTime: 505) == .queued)
        wait(entered)
        for index in 6...13 {
            precondition(source.offer(sample(Int64(index)), sourceGeneration: generation,
                                      receivedHostTime: UInt64(500 + index)) == .queued)
        }
        precondition(source.offer(sample(14), sourceGeneration: generation, receivedHostTime: 514) == .queueFull)
        precondition(source.detach(captureID: overflowCapture, completion: { finished.signal() }))
        release.signal(); wait(finished)
        precondition(overflowEvents.snapshot.count == 10)
        guard case .ended(_, _, _, let overflowCount) = overflowEvents.snapshot.last! else { preconditionFailure() }
        precondition(overflowCount.overflow == 1 && overflowCount.queued == 13)

        // Replacement ends the original subscriber; late SDK callbacks cannot
        // update new metadata or enter a replacement capture.
        let replaced = DispatchSemaphore(value: 0), replacementEvents = Received()
        precondition(source.subscribe(captureID: UUID(), sourceGeneration: generation) { event in
            replacementEvents.append(event)
            if case .ended = event { replaced.signal() }
        } != nil)
        let next = source.beginGeneration()
        precondition(next != generation && source.latest == nil)
        precondition(source.offer(sample(99), sourceGeneration: generation, receivedHostTime: 999) == .retiredSource)
        wait(replaced)
        guard case .ended(_, let old, .sourceReplaced, _) = replacementEvents.snapshot.last! else { preconditionFailure() }
        precondition(old == generation)
        precondition(source.offer(sample(0), sourceGeneration: next, receivedHostTime: 0) == .invalidFrame)
        precondition(source.latest == nil)
        precondition(source.offer(sample(0), sourceGeneration: next, receivedHostTime: 1000) == .inactive)
        let ended = DispatchSemaphore(value: 0)
        precondition(source.subscribe(captureID: UUID(), sourceGeneration: next) { event in
            if case .ended(_, _, .sourceEnded, _) = event { ended.signal() }
        } != nil)
        source.endGeneration(); wait(ended)
        precondition(source.latest == nil)
        precondition(source.offer(sample(1), sourceGeneration: next, receivedHostTime: 1001) == .retiredSource)
        print("PASS: 6 raw-video groups; synthetic buffers, no DAT camera or clock-synchronization claim")
    }
}
