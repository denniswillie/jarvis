import Foundation

final class FrameSampler {
    private let minimumFrameInterval: TimeInterval
    private var lastEmittedAt: Date?
    private var nextSequence: Int64 = 1

    init(targetFPS: Double = 1.0) {
        self.minimumFrameInterval = targetFPS > 0 ? 1.0 / targetFPS : 1.0
    }

    func nextSequenceIfAllowed(capturedAt: Date) -> Int64? {
        guard let lastEmittedAt else {
            self.lastEmittedAt = capturedAt
            return consumeSequence()
        }
        guard capturedAt.timeIntervalSince(lastEmittedAt) >= minimumFrameInterval else {
            return nil
        }
        self.lastEmittedAt = capturedAt
        return consumeSequence()
    }

    func reset() {
        lastEmittedAt = nil
        nextSequence = 1
    }

    private func consumeSequence() -> Int64 {
        let sequence = nextSequence
        nextSequence += 1
        return sequence
    }
}
