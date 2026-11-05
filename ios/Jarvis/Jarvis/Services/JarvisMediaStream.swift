import AVFoundation
import Foundation

/// RTP L16/48000/2 over the authenticated backend socket. No audio-session or
/// microphone ownership. Conversion and network work remain outside AudioUnit.
actor JarvisMediaStream {
    enum Failure: Error { case unavailable, packet, discontinuity, backlog, conversion }

    nonisolated static func recordingSourceFrame(outputFrames: Int64, outputRate: Double,
                                                 trailingFrames: UInt32) -> Double {
        Double(outputFrames) * 48_000 / outputRate - Double(trailingFrames)
    }

    struct Packet {
        let sequence: UInt16
        let timestamp: UInt32
        let source: UInt32
        let samples: [Int16]
        var frames: Int { samples.count / 2 }

        init(_ data: Data) throws {
            let bytes = [UInt8](data)
            guard (16...3852).contains(bytes.count), (bytes.count - 12) % 4 == 0,
                  bytes[0] == 0x80, bytes[1] & 0x7f == 96 else { throw Failure.packet }
            func word(_ at: Int) -> UInt16 { UInt16(bytes[at]) << 8 | UInt16(bytes[at + 1]) }
            func long(_ at: Int) -> UInt32 { UInt32(word(at)) << 16 | UInt32(word(at + 2)) }
            sequence = word(2); timestamp = long(4); source = long(8)
            samples = stride(from: 12, to: bytes.count, by: 2).map { Int16(bitPattern: word($0)) }
        }
    }

    private final class NoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                        completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
    }

    func run(request: URLRequest, device: JarvisDuplexAudioDevice, volume: Float,
             receiving: @escaping @Sendable () -> Void,
             playing: @escaping @Sendable () -> Void = {}) async throws {
        try Task.checkCancellation()
        guard let input = device.attachMediaSource(),
              let original = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48_000, channels: 2, interleaved: true),
              let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: input.sampleRate,
                                        channels: AVAudioChannelCount(input.channels), interleaved: true),
              let converter = AVAudioConverter(from: original, to: target) else { throw Failure.unavailable }
        converter.primeMethod = .none
        _ = device.setMediaVolume(volume, muted: false)
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 8
        config.timeoutIntervalForResource = 24 * 60 * 60
        let session = URLSession(configuration: config, delegate: NoRedirect(), delegateQueue: nil)
        let socket = session.webSocketTask(with: request)
        socket.maximumMessageSize = 3852
        defer {
            socket.cancel(with: .goingAway, reason: nil)
            session.invalidateAndCancel()
            _ = device.detachMediaSource(input)
        }
        socket.resume()
        do { try await withTaskCancellationHandler(operation: {
            var previous: Packet?
            var originTimestamp: UInt32 = 0
            var originUptime: TimeInterval = 0
            var reported = false
            var reportedPlaying = false
            var receivedSourceFrames: Int64 = 0
            var convertedOutputFrames: Int64 = 0
            while !Task.isCancelled {
                let message = try await socket.receive()
                try Task.checkCancellation()
                guard case .data(let data) = message else { throw Failure.packet }
                let packet = try Packet(data)
                let now = ProcessInfo.processInfo.systemUptime
                if let previous {
                    // Lost packets are a real coverage gap, not permission to play
                    // old music later or silently time-compress the missing section.
                    guard packet.source == previous.source,
                          packet.sequence == previous.sequence &+ 1,
                          packet.timestamp == previous.timestamp &+ UInt32(previous.frames) else {
                        throw Failure.discontinuity
                    }
                    let mediaElapsed = Double(packet.timestamp &- originTimestamp) / 48_000
                    guard now - originUptime - mediaElapsed < 0.5 else { throw Failure.backlog }
                } else { originTimestamp = packet.timestamp; originUptime = now }
                previous = packet
                guard let source = AVAudioPCMBuffer(pcmFormat: original, frameCapacity: AVAudioFrameCount(packet.frames)),
                      let sourceData = source.int16ChannelData?[0] else { throw Failure.conversion }
                source.frameLength = AVAudioFrameCount(packet.frames)
                packet.samples.withUnsafeBufferPointer { sourceData.update(from: $0.baseAddress!, count: $0.count) }
                let originalPCM = packet.samples.withUnsafeBytes { Data($0) }
                guard device.cacheMediaPCM16(originalPCM, sourceFrame: receivedSourceFrames, input: input) else {
                    throw Failure.backlog
                }
                receivedSourceFrames += Int64(packet.frames)
                let capacity = AVAudioFrameCount(ceil(Double(packet.frames) * input.sampleRate / 48_000)) + 32
                guard let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { throw Failure.conversion }
                var supplied = false
                var error: NSError?
                let status = converter.convert(to: converted, error: &error) { _, state in
                    if supplied { state.pointee = .noDataNow; return nil }
                    supplied = true; state.pointee = .haveData; return source
                }
                guard error == nil, status != .error else { throw Failure.conversion }
                let frames = Int(converted.frameLength)
                if frames > 0 {
                    guard let samples = converted.int16ChannelData?[0] else { throw Failure.conversion }
                    let output = Data(bytes: samples, count: frames * input.channels * MemoryLayout<Int16>.size)
                    // .none has trailingFrames of input-rate through latency.
                    // Cumulative converter output, not this network packet, owns
                    // the phase. Negative priming/uncached source spans are omitted.
                    let sourceFrame = Self.recordingSourceFrame(outputFrames: convertedOutputFrames,
                        outputRate: input.sampleRate, trailingFrames: converter.primeInfo.trailingFrames)
                    convertedOutputFrames += Int64(frames)
                    guard device.enqueueMediaPCM16(output, frames: frames, input: input,
                                                   sourceFrame: sourceFrame) else { throw Failure.backlog }
                    if !reported { reported = true; receiving() }
                    if !reportedPlaying && device.hasRenderedMedia(input) {
                        reportedPlaying = true; playing()
                    }
                }
            }
        }, onCancel: { socket.cancel(with: .goingAway, reason: nil) })
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw error
        }
    }
}
