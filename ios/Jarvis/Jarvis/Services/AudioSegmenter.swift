import Foundation

struct SpeechSegmenterConfiguration {
    var rmsStartThreshold: Float = 0.018
    var rmsEndThreshold: Float = 0.012
    var startWindowSeconds: Double = 0.12
    var endSilenceSeconds: Double = 1.60
    var preRollSeconds: Double = 0.25
    var maxSegmentSeconds: Double = 20.0
}

final class AudioSegmenter {
    enum Event {
        case speechStarted(startedAt: Date)
        case speechChunk(CapturedAudioChunk)
        case speechEnded(startedAt: Date, endedAt: Date)
    }

    private let configuration: SpeechSegmenterConfiguration
    private var isSpeechActive = false
    private var preRollChunks: [CapturedAudioChunk] = []
    private var preRollDuration = 0.0
    private var aboveThresholdDuration = 0.0
    private var silenceDuration = 0.0
    private var activeDuration = 0.0
    private var activeStartedAt: Date?

    init(configuration: SpeechSegmenterConfiguration = SpeechSegmenterConfiguration()) {
        self.configuration = configuration
    }

    func reset() {
        isSpeechActive = false
        preRollChunks.removeAll()
        preRollDuration = 0
        aboveThresholdDuration = 0
        silenceDuration = 0
        activeDuration = 0
        activeStartedAt = nil
    }

    func process(_ chunk: CapturedAudioChunk) -> [Event] {
        let rms = Self.rmsAmplitude(forPCM16Data: chunk.pcm16Data)
        let duration = chunk.durationSeconds
        var events: [Event] = []

        if isSpeechActive {
            activeDuration += duration
            events.append(.speechChunk(chunk))

            if rms < configuration.rmsEndThreshold {
                silenceDuration += duration
            } else {
                silenceDuration = 0
            }

            if silenceDuration >= configuration.endSilenceSeconds || activeDuration >= configuration.maxSegmentSeconds {
                let startedAt = activeStartedAt ?? chunk.capturedAt
                events.append(.speechEnded(startedAt: startedAt, endedAt: chunk.capturedAt))
                reset()
            }
            return events
        }

        appendPreRoll(chunk)

        if rms >= configuration.rmsStartThreshold {
            aboveThresholdDuration += duration
        } else {
            aboveThresholdDuration = max(0, aboveThresholdDuration - duration)
        }

        if aboveThresholdDuration >= configuration.startWindowSeconds {
            isSpeechActive = true
            activeDuration = preRollDuration
            silenceDuration = 0
            activeStartedAt = preRollChunks.first?.capturedAt ?? chunk.capturedAt
            events.append(.speechStarted(startedAt: activeStartedAt ?? chunk.capturedAt))
            events.append(contentsOf: preRollChunks.map { .speechChunk($0) })
            preRollChunks.removeAll()
            preRollDuration = 0
        }

        return events
    }

    private func appendPreRoll(_ chunk: CapturedAudioChunk) {
        preRollChunks.append(chunk)
        preRollDuration += chunk.durationSeconds

        while preRollDuration > configuration.preRollSeconds, !preRollChunks.isEmpty {
            let removed = preRollChunks.removeFirst()
            preRollDuration -= removed.durationSeconds
        }
    }

    private static func rmsAmplitude(forPCM16Data data: Data) -> Float {
        guard data.count >= 2 else { return 0 }
        var sumSquares = Double(0)
        var sampleCount = 0
        var index = data.startIndex

        while index + 1 < data.endIndex {
            let low = UInt16(data[index])
            let high = UInt16(data[data.index(after: index)]) << 8
            let raw = Int16(bitPattern: high | low)
            let normalized = Double(raw) / Double(Int16.max)
            sumSquares += normalized * normalized
            sampleCount += 1
            index = data.index(index, offsetBy: 2)
        }

        guard sampleCount > 0 else { return 0 }
        return Float(sqrt(sumSquares / Double(sampleCount)))
    }
}
