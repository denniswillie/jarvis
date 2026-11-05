import AVFoundation
import Foundation
import os

@MainActor
final class AudioPlaybackService {
    var onPlaybackStarted: (() -> Void)?
    var onPlaybackStopped: (() -> Void)?
    var onPlaybackReceipt: ((AudioPlaybackReceipt) -> Void)?

    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let outputSampleRate: Double
    private let outputFormat: AVAudioFormat
    private var isEngineConfigured = false
    private var ledger = AudioPlaybackLedger()
    private var observers: [NSObjectProtocol] = []
    private let diagnosticLogger = Logger(subsystem: "com.symbiote.jarvis", category: "playback-timing")
    private var observedRender = false

    init(outputSampleRate: Double = 24_000) {
        precondition(outputSampleRate == 24_000, "Realtime playback uses 24kHz mono PCM16.")
        self.outputSampleRate = outputSampleRate
        self.outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: outputSampleRate,
                                          channels: 1, interleaved: false)!
        observers.append(NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] notification in
            guard let raw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  raw == AVAudioSession.InterruptionType.began.rawValue else { return }
            Task { @MainActor in self?.stop() }
        })
        observers.append(NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: audioEngine, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.stop() }
        })
    }

    deinit {
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
    }

    var isPlaying: Bool { ledger.hasPendingAudio }

    func enqueuePCM16(base64: String, responseId: String?, itemId: String?) throws {
        guard !ledger.isRetired(responseId: responseId, itemId: itemId) else { return }
        guard let data = Data(base64Encoded: base64), !data.isEmpty, data.count % 2 == 0 else {
            throw JarvisAudioSessionError.serverError("Invalid PCM16 playback data.")
        }
        try startEngineIfNeeded()
        let frameCount = data.count / MemoryLayout<Int16>.size
        guard let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(frameCount)),
              let destination = buffer.floatChannelData?[0] else {
            throw JarvisAudioSessionError.serverError("Playback buffer allocation failed.")
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        data.withUnsafeBytes { rawBuffer in
            let source = rawBuffer.bindMemory(to: Int16.self)
            for index in 0..<frameCount {
                let sample = Int16(littleEndian: source[index])
                destination[index] = max(-1.0, min(1.0, Float(sample) / Float(Int16.max)))
            }
        }
        let wasPlaying = isPlaying
        let token = try ledger.schedule(responseId: responseId, itemId: itemId, samples: frameCount)
        playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] kind in
            guard kind == .dataPlayedBack else { return }
            Task { @MainActor in self?.handleBufferPlayedBack(token) }
        }
        if !playerNode.isPlaying {
            playerNode.play()
            logPlayback("playback_requested", boundary: "player_play_called_not_hardware_ack")
        }
        if !wasPlaying { onPlaybackStarted?() }
        observeRenderClock()
    }

    func generationFinished(responseId: String?, itemId: String?) {
        emit(ledger.generationFinished(responseId: responseId, itemId: itemId))
    }

    func stop() {
        let wasPlaying = isPlaying || playerNode.isPlaying
        let receipts = ledger.stop()
        playerNode.stop()
        observedRender = false
        emit(receipts)
        if wasPlaying { onPlaybackStopped?() }
    }

    func resetConnection() {
        stop()
        ledger.resetConnection()
    }

    func truncationSnapshot() -> PlaybackTruncationSnapshot? {
        ledger.truncationSnapshot(sampleRate: outputSampleRate)
    }

    private func handleBufferPlayedBack(_ token: AudioPlaybackBuffer) {
        guard audioEngine.isRunning, playerNode.isPlaying else {
            stop()
            return
        }
        let wasPlaying = isPlaying
        emit(ledger.playedBack(token))
        observeRenderClock()
        if wasPlaying && !isPlaying { onPlaybackStopped?() }
        // Keep the node's timeline alive across a network gap. A later item or
        // chunk must not reset another queued item's completion bookkeeping.
    }

    private func emit(_ receipts: [AudioPlaybackReceipt]) {
        for receipt in receipts {
            if let diagnostic = RealtimeDiagnostics.record("playback_\(receipt.state.rawValue)", fields: [
                "boundary": "av_audio_data_played_back_lower_bound",
                "response_id": receipt.responseId, "item_id": receipt.itemId,
                "played_samples": receipt.playedSamples, "sample_rate": outputSampleRate,
            ]) { diagnosticLogger.info("realtime_trace \(diagnostic, privacy: .public)") }
            onPlaybackReceipt?(receipt)
        }
    }

    private func startEngineIfNeeded() throws {
        if !isEngineConfigured {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
            audioEngine.attach(playerNode)
            audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: outputFormat)
            isEngineConfigured = true
        }
        if !audioEngine.isRunning {
            audioEngine.prepare()
            try audioEngine.start()
        }
    }

    private func logPlayback(_ event: String, boundary: String, renderTime: AVAudioTime? = nil) {
        var fields: [String: Any] = ["boundary": boundary]
        if let snapshot = truncationSnapshot() {
            fields["response_id"] = snapshot.responseId
            fields["item_id"] = snapshot.itemId
        }
        if let renderTime {
            fields["render_sample_time"] = renderTime.sampleTime
            fields["sample_rate"] = renderTime.sampleRate
        }
        if let diagnostic = RealtimeDiagnostics.record(event, fields: fields) {
            diagnosticLogger.info("realtime_trace \(diagnostic, privacy: .public)")
        }
    }

    private func observeRenderClock() {
        guard !observedRender, let nodeTime = playerNode.lastRenderTime,
              let rendered = playerNode.playerTime(forNodeTime: nodeTime),
              rendered.isSampleTimeValid, rendered.sampleTime > 0 else { return }
        observedRender = true
        logPlayback("playback_render_observed", boundary: "player_render_clock_observation", renderTime: rendered)
    }
}
