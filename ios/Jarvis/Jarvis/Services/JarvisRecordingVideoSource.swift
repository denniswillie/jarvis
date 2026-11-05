@preconcurrency import CoreMedia
import CoreVideo
import Foundation

/// Bounded fanout of the SAME DAT stream, before MainActor/analyzer sampling.
/// Preserves source PTS and callback host ticks separately. Neither field proves
/// that the SDK's video clock is the microphone clock. No capture or retiming.
final class JarvisRecordingVideoSource: @unchecked Sendable {
    struct Counters: Sendable, Equatable {
        var observed = 0
        var queued = 0
        var overflow = 0
        var invalid = 0
    }
    struct Snapshot: Sendable {
        let sourceGeneration: UUID
        let sequence: UInt64
        let width: Int
        let height: Int
        let pixelFormat: OSType
        let presentationTime: CMTime
        let receivedHostTime: UInt64
        let counters: Counters
    }
    struct Frame: @unchecked Sendable {
        let captureID: UUID
        let snapshot: Snapshot
        // Retained immutable SDK sample, not a borrowed pixel pointer.
        let sampleBuffer: CMSampleBuffer
    }
    enum EndReason: String, Sendable { case detached, sourceEnded, sourceReplaced }
    enum Event: Sendable {
        case frame(Frame)
        case ended(captureID: UUID, sourceGeneration: UUID, reason: EndReason, counters: Counters)
    }
    enum Admission: Equatable { case queued, inactive, retiredSource, invalidFrame, queueFull }
    typealias Sink = @Sendable (Event) -> Void
    typealias Completion = @Sendable () -> Void

    static let maximumPendingFrames = 8
    static let maximumPendingBytes = 16 * 1024 * 1024
    static let maximumOutstandingSubscriptions = 8
    private let lock = NSLock()
    private let delivery = DispatchQueue(label: "jarvis.recording.raw-video", qos: .utility)
    private var generation: UUID?
    private var sequence: UInt64 = 0
    private var current: Snapshot?
    private var counters = Counters()
    private var subscription: Subscription?
    private var outstandingSubscriptions = 0
    private var pending: [Work] = []
    private var pendingFrames = 0
    private var pendingBytes = 0
    private var draining = false

    private struct Subscription: Sendable {
        let id: UUID
        let generation: UUID
        let sink: Sink
    }
    private enum Work: Sendable {
        case frame(Frame, Subscription, bytes: Int)
        case end(Subscription, EndReason, Counters, Completion?)
    }

    var latest: Snapshot? { lock.withLock { current } }

    @discardableResult func beginGeneration() -> UUID {
        lock.withLock {
            terminateSubscriptionLocked(reason: .sourceReplaced, completion: nil)
            let next = UUID()
            generation = next; sequence = 0; current = nil; counters = Counters()
            return next
        }
    }
    func endGeneration() {
        lock.withLock {
            terminateSubscriptionLocked(reason: .sourceEnded, completion: nil)
            generation = nil; current = nil
        }
    }

    /// A real frame must already establish this stream's format. The returned
    /// snapshot is a baseline, not a clock calibration or a capture receipt.
    func subscribe(captureID: UUID, sourceGeneration: UUID, sink: @escaping Sink) -> Snapshot? {
        lock.withLock {
            guard generation == sourceGeneration, let current,
                  subscription == nil, outstandingSubscriptions < Self.maximumOutstandingSubscriptions else { return nil }
            subscription = Subscription(id: captureID, generation: sourceGeneration, sink: sink)
            outstandingSubscriptions += 1
            return current
        }
    }

    /// Fences intake immediately; the completion runs after all accepted frames
    /// and the end event on the delivery queue. It never stops the DAT stream.
    @discardableResult func detach(captureID: UUID, completion: @escaping Completion) -> Bool {
        lock.withLock {
            guard subscription?.id == captureID else { return false }
            terminateSubscriptionLocked(reason: .detached, completion: completion)
            return true
        }
    }

    @discardableResult
    func offer(_ sample: CMSampleBuffer, sourceGeneration: UUID, receivedHostTime: UInt64) -> Admission {
        lock.withLock {
            guard generation == sourceGeneration else { return .retiredSource }
            guard CMSampleBufferDataIsReady(sample), CMSampleBufferGetNumSamples(sample) == 1,
                  receivedHostTime > 0, sequence < UInt64.max,
                  let image = CMSampleBufferGetImageBuffer(sample) else {
                counters.invalid += 1; return .invalidFrame
            }
            let bytes = CVPixelBufferGetDataSize(image)
            let width = CVPixelBufferGetWidth(image), height = CVPixelBufferGetHeight(image)
            guard bytes > 0, bytes <= Self.maximumPendingBytes,
                  width > 0, height > 0, width <= 3840, height <= 3840 else {
                counters.invalid += 1; return .invalidFrame
            }
            sequence += 1; counters.observed += 1
            let metadata = Snapshot(sourceGeneration: sourceGeneration, sequence: sequence,
                width: width, height: height, pixelFormat: CVPixelBufferGetPixelFormatType(image),
                presentationTime: CMSampleBufferGetPresentationTimeStamp(sample),
                receivedHostTime: receivedHostTime, counters: counters)
            current = metadata
            guard let subscription else { return .inactive }
            guard pendingFrames < Self.maximumPendingFrames,
                  bytes <= Self.maximumPendingBytes - pendingBytes else {
                counters.overflow += 1; return .queueFull
            }
            counters.queued += 1
            pending.append(.frame(Frame(captureID: subscription.id, snapshot: metadata, sampleBuffer: sample),
                                  subscription, bytes: bytes))
            pendingFrames += 1; pendingBytes += bytes
            scheduleLocked()
            return .queued
        }
    }

    private func terminateSubscriptionLocked(reason: EndReason, completion: Completion?) {
        guard let previous = subscription else { return }
        subscription = nil
        // One reserved end marker per admitted subscription. Frames and control
        // markers share FIFO ordering; rapid replacements cannot grow forever.
        pending.append(.end(previous, reason, counters, completion))
        scheduleLocked()
    }
    private func scheduleLocked() {
        guard !draining else { return }
        draining = true
        delivery.async { self.drain() }
    }
    private func drain() {
        // Bounded batch, without executing a sink while holding the input lock.
        for _ in 0..<Self.maximumPendingFrames {
            let item: Work? = lock.withLock {
                guard !pending.isEmpty else { return nil }
                let next = pending.removeFirst()
                switch next {
                case .frame(_, _, let bytes): pendingFrames -= 1; pendingBytes -= bytes
                case .end: outstandingSubscriptions -= 1
                }
                return next
            }
            guard let item else { break }
            switch item {
            case .frame(let frame, let subscription, _): subscription.sink(.frame(frame))
            case .end(let subscription, let reason, let counts, let completion):
                subscription.sink(.ended(captureID: subscription.id, sourceGeneration: subscription.generation,
                                         reason: reason, counters: counts))
                completion?()
            }
        }
        lock.withLock {
            if pending.isEmpty { draining = false }
            else { delivery.async { self.drain() } }
        }
    }
}
