import AVFoundation
import Foundation

final class AudioCaptureService {
    var onAudioChunk: ((CapturedAudioChunk) -> Void)?

    private let audioEngine = AVAudioEngine()
    private let targetSampleRate: Double
    private let tapBufferSize: AVAudioFrameCount
    private var isTapInstalled = false

    init(targetSampleRate: Double = 24_000, tapBufferSize: AVAudioFrameCount = 1_024) {
        self.targetSampleRate = targetSampleRate
        self.tapBufferSize = tapBufferSize
    }

    var isRunning: Bool {
        audioEngine.isRunning
    }

    func start() throws {
        try configureAudioSession()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if !isTapInstalled {
            inputNode.installTap(onBus: 0, bufferSize: tapBufferSize, format: inputFormat) { [weak self] buffer, _ in
                guard let self else { return }
                guard let data = Self.pcm16MonoData(from: buffer, targetSampleRate: self.targetSampleRate) else { return }
                let frameCount = data.count / MemoryLayout<Int16>.size
                let chunk = CapturedAudioChunk(
                    pcm16Data: data,
                    sampleRate: self.targetSampleRate,
                    frameCount: frameCount,
                    capturedAt: Date()
                )
                self.onAudioChunk?(chunk)
            }
            isTapInstalled = true
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    func stop() {
        if isTapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            isTapInstalled = false
        }
        audioEngine.stop()
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)
    }

    private static func pcm16MonoData(from buffer: AVAudioPCMBuffer, targetSampleRate: Double) -> Data? {
        let monoSamples = monoFloatSamples(from: buffer)
        guard !monoSamples.isEmpty else { return nil }

        let sourceRate = buffer.format.sampleRate
        let resampled = resample(samples: monoSamples, sourceSampleRate: sourceRate, targetSampleRate: targetSampleRate)
        guard !resampled.isEmpty else { return nil }

        var data = Data(capacity: resampled.count * MemoryLayout<Int16>.size)
        for sample in resampled {
            let clipped = max(-1.0, min(1.0, sample))
            var intSample = Int16(clipped * Float(Int16.max)).littleEndian
            withUnsafeBytes(of: &intSample) { bytes in
                data.append(contentsOf: bytes)
            }
        }
        return data
    }

    private static func monoFloatSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return [] }
        let channelCount = max(1, Int(buffer.format.channelCount))

        if let floatChannels = buffer.floatChannelData {
            var samples = Array(repeating: Float(0), count: frameLength)
            for channel in 0..<channelCount {
                let channelData = floatChannels[channel]
                for index in 0..<frameLength {
                    samples[index] += channelData[index] / Float(channelCount)
                }
            }
            return samples
        }

        if let int16Channels = buffer.int16ChannelData {
            var samples = Array(repeating: Float(0), count: frameLength)
            for channel in 0..<channelCount {
                let channelData = int16Channels[channel]
                for index in 0..<frameLength {
                    samples[index] += Float(channelData[index]) / Float(Int16.max) / Float(channelCount)
                }
            }
            return samples
        }

        return []
    }

    private static func resample(samples: [Float], sourceSampleRate: Double, targetSampleRate: Double) -> [Float] {
        guard sourceSampleRate > 0, targetSampleRate > 0, !samples.isEmpty else { return [] }
        if abs(sourceSampleRate - targetSampleRate) < 1 {
            return samples
        }

        let outputCount = max(1, Int(Double(samples.count) * targetSampleRate / sourceSampleRate))
        let step = sourceSampleRate / targetSampleRate
        var output: [Float] = []
        output.reserveCapacity(outputCount)

        for outputIndex in 0..<outputCount {
            let sourcePosition = Double(outputIndex) * step
            let lowerIndex = min(samples.count - 1, Int(sourcePosition))
            let upperIndex = min(samples.count - 1, lowerIndex + 1)
            let fraction = Float(sourcePosition - Double(lowerIndex))
            let interpolated = samples[lowerIndex] * (1 - fraction) + samples[upperIndex] * fraction
            output.append(interpolated)
        }
        return output
    }
}
