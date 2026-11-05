import Foundation

struct AudioPlaybackReceipt: Equatable, Sendable {
    enum State: String, Sendable { case started, progress, interrupted, completed }
    let responseId: String
    let itemId: String
    let playedSamples: Int
    let state: State
}

struct AudioPlaybackIdentity: Hashable, Sendable {
    let responseId: String?
    let itemId: String?
}

struct AudioPlaybackBuffer: Equatable, Sendable {
    let id: UUID
    let generation: Int
    let identity: AudioPlaybackIdentity
    let startSample: Int
    let sampleCount: Int
}

/// Counts only contiguous buffers acknowledged by dataPlayedBack. Wall-clock
/// time, enqueueing and generation completion never establish played samples.
struct AudioPlaybackLedger {
    private struct Item {
        var scheduled = 0
        var played = 0
        var completedRanges: [Int: Int] = [:]
        var generationFinished = false
        var startedReported = false
        var completedReported = false
        var lastReportedSample = 0
    }
    private var generation = 0
    private var items: [AudioPlaybackIdentity: Item] = [:]
    private var order: [AudioPlaybackIdentity] = []
    private var buffers: [UUID: AudioPlaybackBuffer] = [:]
    private var retired: [AudioPlaybackIdentity] = []
    private let itemLimit = 256
    private let progressSamples = 6_000

    var hasPendingAudio: Bool { !buffers.isEmpty }
    var retainedItemCount: Int { items.count }

    func isRetired(responseId: String?, itemId: String?) -> Bool {
        retired.contains(AudioPlaybackIdentity(responseId: responseId, itemId: itemId))
    }

    mutating func resetConnection() {
        _ = stop()
        retired.removeAll()
    }

    mutating func schedule(responseId: String?, itemId: String?, samples: Int) throws -> AudioPlaybackBuffer {
        guard samples > 0 else { throw JarvisAudioSessionError.serverError("Empty playback buffer.") }
        let identity = AudioPlaybackIdentity(responseId: responseId, itemId: itemId)
        guard !retired.contains(identity) else { throw JarvisAudioSessionError.serverError("Retired playback item.") }
        if items[identity] == nil {
            while items.count >= itemLimit, let expired = order.first(where: { items[$0]?.completedReported == true }) {
                items.removeValue(forKey: expired)
                order.removeAll { $0 == expired }
            }
            guard items.count < itemLimit else { throw JarvisAudioSessionError.serverError("Playback item limit reached.") }
            if let itemId, order.contains(where: { $0.itemId == itemId && $0.responseId != responseId }) {
                throw JarvisAudioSessionError.serverError("Playback item changed its response identity.")
            }
            order.append(identity)
        }
        var item = items[identity] ?? Item()
        guard !item.generationFinished, samples <= 43_200_000 - item.scheduled else {
            throw JarvisAudioSessionError.serverError("Playback data exceeds its completed item or limit.")
        }
        let token = AudioPlaybackBuffer(id: UUID(), generation: generation, identity: identity,
                                        startSample: item.scheduled, sampleCount: samples)
        item.scheduled += samples
        items[identity] = item
        buffers[token.id] = token
        return token
    }

    mutating func playedBack(_ token: AudioPlaybackBuffer) -> [AudioPlaybackReceipt] {
        guard token.generation == generation, buffers[token.id] == token,
              var item = items[token.identity] else { return [] }
        buffers.removeValue(forKey: token.id)
        item.completedRanges[token.startSample] = token.sampleCount
        while let count = item.completedRanges.removeValue(forKey: item.played) {
            item.played += count
        }
        items[token.identity] = item
        return receipts(for: token.identity)
    }

    mutating func generationFinished(responseId: String?, itemId: String?) -> [AudioPlaybackReceipt] {
        let identity = AudioPlaybackIdentity(responseId: responseId, itemId: itemId)
        guard items[identity] != nil else { return [] }
        items[identity]?.generationFinished = true
        return receipts(for: identity)
    }

    private mutating func receipts(for identity: AudioPlaybackIdentity) -> [AudioPlaybackReceipt] {
        guard var item = items[identity], let responseId = identity.responseId, !responseId.isEmpty,
              let itemId = identity.itemId, !itemId.isEmpty else { return [] }
        var result: [AudioPlaybackReceipt] = []
        func receipt(_ state: AudioPlaybackReceipt.State) -> AudioPlaybackReceipt {
            AudioPlaybackReceipt(responseId: responseId, itemId: itemId, playedSamples: item.played, state: state)
        }
        if item.played > 0 && !item.startedReported {
            item.startedReported = true
            item.lastReportedSample = item.played
            result.append(receipt(.started))
        }
        if item.generationFinished && item.played == item.scheduled && item.scheduled > 0 && !item.completedReported {
            item.completedReported = true
            result.append(receipt(.completed))
        } else if item.played - item.lastReportedSample >= progressSamples && !item.completedReported {
            item.lastReportedSample = item.played
            result.append(receipt(.progress))
        }
        items[identity] = item
        return result
    }

    func truncationSnapshot(sampleRate: Double) -> PlaybackTruncationSnapshot? {
        let identity = order.first(where: { key in buffers.values.contains { $0.identity == key } })
            ?? order.last(where: { items[$0]?.completedReported == false })
        guard let identity, let itemId = identity.itemId, let item = items[identity] else { return nil }
        return PlaybackTruncationSnapshot(itemId: itemId, responseId: identity.responseId,
                                         audioEndMs: Int(Double(item.played) * 1_000 / sampleRate))
    }

    mutating func stop() -> [AudioPlaybackReceipt] {
        let result = order.compactMap { identity -> AudioPlaybackReceipt? in
            guard let item = items[identity], !item.completedReported,
                  let responseId = identity.responseId, let itemId = identity.itemId else { return nil }
            return AudioPlaybackReceipt(responseId: responseId, itemId: itemId,
                                        playedSamples: item.played, state: .interrupted)
        }
        // Invalidate before AVAudioPlayerNode.stop invokes outstanding callbacks.
        retired.append(contentsOf: order.filter { !retired.contains($0) })
        if retired.count > itemLimit { retired.removeFirst(retired.count - itemLimit) }
        generation += 1
        items.removeAll()
        order.removeAll()
        buffers.removeAll()
        return result
    }
}
