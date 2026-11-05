/// Bound both application backlog and frames already queued to the server.
/// A socket write is not an ingestion acknowledgement.
struct JarvisVisualFrameFlow<Frame> {
    private var pending: Frame?
    private(set) var inFlightSequence: Int64?

    mutating func enqueue(_ frame: Frame) {
        pending = frame
    }

    mutating func takeNext(sequence: (Frame) -> Int64) -> Frame? {
        guard inFlightSequence == nil, let frame = pending else { return nil }
        pending = nil
        inFlightSequence = sequence(frame)
        return frame
    }

    @discardableResult
    mutating func acknowledge(sequence: Int64) -> Bool {
        guard sequence == inFlightSequence else { return false }
        inFlightSequence = nil
        return true
    }

    mutating func reset() {
        pending = nil
        inFlightSequence = nil
    }
}
