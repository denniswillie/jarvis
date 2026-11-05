#if DEBUG
import AudioToolbox
import Foundation
import WebRTC

/// Retirement rejects new callbacks immediately. A successful receipt also
/// proves every already-entered callback has returned. Timeout is NOT proof.
final class JarvisReplayCallbackFence {
    private let condition = NSCondition()
    private var retired = false
    private var active = 0
    @discardableResult func perform(_ body: () -> Void) -> Bool {
        condition.lock()
        guard !retired else { condition.unlock(); return false }
        active += 1
        condition.unlock()
        defer {
            condition.lock(); active -= 1; condition.broadcast(); condition.unlock()
        }
        body()
        return true
    }
    func close(timeout: TimeInterval = 0.5) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        retired = true
        let deadline = Date().addingTimeInterval(max(0, timeout))
        while active > 0 {
            guard condition.wait(until: deadline) else { return active == 0 }
        }
        return true
    }
}

/// A session-owned, silent native ADM. No microphone, speaker, AudioUnit or
/// AVAudioSession is opened. The ordinary peer/RTP/provider path is unchanged.
final class JarvisReplayAudioDevice: NSObject, RTCAudioDevice {
    let sceneDuration: TimeInterval
    let captureDuration: TimeInterval
    let recordingCapacityDuration: TimeInterval
    let deviceInputSampleRate = 48_000.0
    let inputIOBufferDuration = 0.01
    let inputNumberOfChannels = 1
    let inputLatency = 0.0
    let deviceOutputSampleRate = 48_000.0
    let outputIOBufferDuration = 0.01
    let outputNumberOfChannels = 1
    let outputLatency = 0.0

    private let lock = NSLock()
    private let callbacks = JarvisReplayCallbackFence()
    private let queue = DispatchQueue(label: "jarvis.debug-replay.audio")
    private var timer: DispatchSourceTimer?
    private var delegate: RTCAudioDeviceDelegate?
    private var initialized = false
    private var playoutInitialized = false
    private var recordingInitialized = false
    private var playing = false
    private var recording = false
    private var sampleTime = 0.0
    private var state: JarvisReplayPCMState
    private let now: () -> TimeInterval

    init(maxDuration: TimeInterval, postMediaCaptureSeconds: TimeInterval = 0,
         now: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) {
        sceneDuration = maxDuration
        captureDuration = maxDuration + postMediaCaptureSeconds
        recordingCapacityDuration = captureDuration + JarvisReplayPlan.startupAllowance
        let valid = JarvisReplayPlan.isValidSceneDuration(maxDuration) && postMediaCaptureSeconds.isFinite
            && postMediaCaptureSeconds >= 0 && postMediaCaptureSeconds <= JarvisReplayPlan.maximumPostMediaCapture
            && JarvisReplayPlan.isValidSceneDuration(captureDuration)
        state = JarvisReplayPCMState(maxDuration: valid ? captureDuration : .nan,
                                    startupAllowance: JarvisReplayPlan.startupAllowance)
        self.now = now
        super.init()
        guard !state.closed else { return }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in self?.processAudio() }
        self.timer = timer
        timer.resume()
    }
    convenience init(plan: JarvisReplayPlan) {
        self.init(maxDuration: plan.duration, postMediaCaptureSeconds: plan.postMediaCaptureSeconds)
    }
    deinit { timer?.cancel() }

    var isInitialized: Bool { lock.withLock { initialized } }
    var isPlayoutInitialized: Bool { lock.withLock { playoutInitialized } }
    var isRecordingInitialized: Bool { lock.withLock { recordingInitialized } }
    var isPlaying: Bool { lock.withLock { playing } }
    var isRecording: Bool { lock.withLock { recording } }
    var evidence: JarvisReplayAudioEvidence { lock.withLock { state.evidence } }
    var inputIdle: Bool { lock.withLock { state.inputIdle } }
    var isRetired: Bool { lock.withLock { state.closed } }
    var latestOutput: JarvisReplayOutputStart? { lock.withLock { state.latestOutput } }

    func initialize(with delegate: RTCAudioDeviceDelegate) -> Bool {
        lock.withLock {
            guard !state.closed else { return false }
            self.delegate = delegate; initialized = true; return true
        }
    }
    func terminateDevice() -> Bool {
        lock.withLock {
            delegate = nil; initialized = false; playing = false; recording = false
            playoutInitialized = false; recordingInitialized = false
            return true
        }
    }
    func initializePlayout() -> Bool {
        lock.withLock { playoutInitialized = initialized && !state.closed; return playoutInitialized }
    }
    func initializeRecording() -> Bool {
        lock.withLock { recordingInitialized = initialized && !state.closed; return recordingInitialized }
    }
    func startPlayout() -> Bool {
        lock.withLock { playing = playoutInitialized && !state.closed; return playing }
    }
    func startRecording() -> Bool {
        lock.withLock { recording = recordingInitialized && !state.closed; return recording }
    }
    func stopPlayout() -> Bool { lock.withLock { playing = false; return true } }
    func stopRecording() -> Bool { lock.withLock { recording = false; return true } }
    func setInputEnabled(_ enabled: Bool) { lock.withLock { state.gateInput(enabled, at: now()) } }
    func setOutputEnabled(_ enabled: Bool, responseID: String? = nil, itemID: String? = nil,
                          connectionID: UUID? = nil, diagnostics: JarvisReplayPlaybackGateDiagnostics? = nil) {
        lock.withLock { state.gateOutput(enabled, at: now(), responseID: responseID,
                                        itemID: itemID, connectionID: connectionID, diagnostics: diagnostics) }
    }
    func responseSnapshot(afterInputOrdinal: Int) -> JarvisReplayResponseSnapshot {
        lock.withLock { state.responseSnapshot(afterInputOrdinal: afterInputOrdinal) }
    }
    func recordLocalInterruption(_ cause: JarvisReplayLocalInterruptionCause) {
        lock.withLock { state.recordLocalInterruption(cause, at: now()) }
    }
    func registerPlan(_ plan: JarvisReplayPlan) throws { try lock.withLock { try state.register(plan) } }
    func recordClockAnchor(_ anchor: JarvisReplayClockAnchor) throws { try lock.withLock { try state.recordAnchor(anchor) } }
    func recordSourceClock(_ data: Data?, failure: JarvisReplayError? = nil) {
        lock.withLock { state.recordSourceClock(data, failure: failure) }
    }
    func beginOutputDrain(scheduled: TimeInterval, observed: TimeInterval, seconds: TimeInterval) {
        lock.withLock { state.beginOutputDrain(scheduled: scheduled, observed: observed, seconds: seconds) }
    }
    func pollOutputDrain(at now: TimeInterval) -> Bool { lock.withLock { state.pollOutputDrain(at: now) } }
    func recordControlEvent(_ stage: JarvisReplayControlStage, fields: [String: Any] = [:],
                            receivedAt: TimeInterval? = nil) {
        let handled = now()
        lock.withLock { state.recordControlEvent(stage, fields: fields, receivedAt: receivedAt, at: handled) }
    }
    func recordTaskBarrierFailure(_ code: String, diagnostic: JarvisReplayTaskFailureDiagnostic? = nil) {
        lock.withLock { state.recordTaskBarrierFailure(code, diagnostic: diagnostic) }
    }
    func recordManualInput(_ value: [String: Any]) { lock.withLock { state.recordManualInput(value) } }
    func armUtterance(_ ordinal: Int, at time: TimeInterval) { lock.withLock { state.armUtterance(ordinal, at: time) } }
    func failUtterance(_ ordinal: Int, code: String, at time: TimeInterval) {
        lock.withLock { state.failUtterance(ordinal, code: code, at: time) }
    }
    func enqueuePCM16(_ data: Data, utteranceOrdinal: Int? = nil, triggeredAt: TimeInterval? = nil,
                      responseSelection: JarvisReplayResponseSelection? = nil, taskClaimSequence: Int? = nil) throws {
        try lock.withLock { try state.enqueue(data, utteranceOrdinal: utteranceOrdinal, requestedAt: now(),
            triggeredAt: triggeredAt, responseSelection: responseSelection, taskClaimSequence: taskClaimSequence) }
    }
    @discardableResult func close() -> Bool {
        lock.withLock { state.close(at: now()); playing = false; recording = false }
        timer?.cancel()
        return callbacks.close()
    }

    private func processAudio() {
        callbacks.perform { processAudioBlock() }
    }

    private func processAudioBlock() {
        let current = lock.withLock { (delegate, recording, playing, state.closed) }
        guard let delegate = current.0, !current.3 else { return }
        var timestamp = AudioTimeStamp()
        timestamp.mSampleTime = sampleTime
        timestamp.mFlags = .sampleTimeValid
        sampleTime += 480
        let boundary = now()
        if current.1 {
            var samples = lock.withLock { state.renderInput(at: boundary) }
            samples.withUnsafeMutableBufferPointer { storage in
                var list = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(
                    mNumberChannels: 1, mDataByteSize: 960, mData: storage.baseAddress))
                var flags = AudioUnitRenderActionFlags()
                let started = now()
                lock.withLock { state.inputCallbackStarted(at: started) }
                let status = delegate.deliverRecordedData(&flags, &timestamp, 0, 480, &list, nil, nil)
                let finished = now()
                lock.withLock { state.inputCallbackCompleted(status: status, at: finished) }
            }
        }
        if current.2 {
            var samples = [Int16](repeating: 0, count: 480)
            let identities = lock.withLock { (state.currentResponseIdentity, state.currentResponseAssociation) }
            let started = now()
            let status = samples.withUnsafeMutableBufferPointer { storage -> OSStatus in
                var list = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(
                    mNumberChannels: 1, mDataByteSize: 960, mData: storage.baseAddress))
                var flags = AudioUnitRenderActionFlags()
                return delegate.getPlayoutData(&flags, &timestamp, 0, 480, &list)
            }
            let finished = now()
            lock.withLock {
                if status == noErr { state.observeDecoded(samples, at: started, callbackEnd: finished,
                                                         responseIdentity: identities.0, responseAssociation: identities.1) }
                else { state.callbackFailed(status) }
            }
        }
    }
}
#endif
