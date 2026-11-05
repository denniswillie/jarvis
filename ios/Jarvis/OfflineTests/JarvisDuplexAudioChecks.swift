import Foundation
import CoreMedia

private final class Events: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [JarvisDuplexAudioDevice.CaptureEvent] = []
    func append(_ event: JarvisDuplexAudioDevice.CaptureEvent) {
        lock.lock(); storage.append(event); lock.unlock()
    }
    func values() -> [JarvisDuplexAudioDevice.CaptureEvent] {
        lock.lock(); defer { lock.unlock() }; return storage
    }
}
private final class ResultBox: @unchecked Sendable {
    var value = false // Synchronized by the semaphore below.
}
private func completed(_ work: (@escaping JarvisDuplexAudioDevice.Completion) -> Void) -> Bool {
    let signal = DispatchSemaphore(value: 0), result = ResultBox()
    work { value in result.value = value; signal.signal() }
    precondition(signal.wait(timeout: .now() + 5) == .success, "completion timed out")
    return result.value
}

@main
struct DuplexAudioChecks {
    static func consumedMedia() {
        JRDTestReset()
        let device = JarvisDuplexAudioDevice(voiceGeneration: UUID(), sessionLease: .init(
            acquire: { _, _ in .init(identity: "consumed-media", sampleRate: 16000,
                inputChannels: 1, outputChannels: 1, ioBufferDuration: 0.01,
                inputLatency: 0, outputLatency: 0) }, release: {}), onStatus: { _ in })
        precondition(device.initialize(with: JRDTestDelegate()))
        precondition(device.initializeRecording() && device.initializePlayout())
        precondition(device.startRecording() && device.startPlayout())
        precondition(completed { device.authorizeSession(completion: $0) })
        device.setGates(microphone: true, output: false)
        let input = device.attachMediaSource()!
        precondition(device.setMediaVolume(1, muted: false))
        for _ in 0..<32 { JRDTestPump(160, mach_absolute_time()) }
        let events = Events(), id = UUID()
        precondition(completed { device.attachSink(captureID: id, sink: events.append, completion: $0) })
        let original = (0..<480).flatMap { _ in [Int16(10000), Int16(-4000)] }.withUnsafeBytes { Data($0) }
        let second = (0..<480).flatMap { _ in [Int16(20000), Int16(-8000)] }.withUnsafeBytes { Data($0) }
        precondition(device.cacheMediaPCM16(original, sourceFrame: 0, input: input))
        precondition(device.cacheMediaPCM16(second, sourceFrame: 480, input: input))
        precondition(device.cacheMediaPCM16(original, sourceFrame: 960, input: input)) // Never consumed.
        let playback = [Int16](repeating: 2000, count: 320).withUnsafeBytes { Data($0) }
        precondition(device.enqueueMediaPCM16(playback, frames: 320, input: input, sourceFrame: 0))
        let origin = CMClockMakeHostTimeFromSystemUnits(mach_absolute_time())
        func host(_ milliseconds: Int64) -> UInt64 {
            CMClockConvertHostTimeToSystemUnits(CMTimeAdd(origin, CMTime(value: milliseconds, timescale: 1000)))
        }
        JRDTestPump(80, host(0)); JRDTestPump(80, host(5))
        precondition(device.setMediaVolume(1, muted: true))
        JRDTestPump(160, host(10))
        // Fence and resolve before intentionally evicting originals.
        precondition(completed { device.detachSink(captureID: id, completion: $0) })
        let packets = events.values().compactMap { event -> JarvisDuplexAudioDevice.PCMBlock? in
            if case .pcm(let packet) = event, packet.track == .renderedOutput { return packet }; return nil
        }
        precondition(packets.count == 3 && packets.allSatisfy { !$0.softwareGateOpen })
        let spans = packets.flatMap(\.mediaSpans)
        precondition(spans.map(\.frames) == [240, 240, 480])
        precondition(spans[0].hostTime == host(0) && spans[1].hostTime == host(5))
        for span in spans.prefix(2) {
            span.pcm16Interleaved.withUnsafeBytes { bytes in
                precondition(bytes.loadUnaligned(as: Int16.self) == 10000)
                precondition(bytes.loadUnaligned(fromByteOffset: 2, as: Int16.self) == -4000)
            }
        }
        precondition(spans[2].pcm16Interleaved.allSatisfy { $0 == 0 })
        precondition(packets.reduce(0) { $0 + $1.mediaCoverage.resolvedSourceFrames } == 960)
        precondition(packets.reduce(0) { $0 + $1.mediaCoverage.zeroGainSourceFrames } == 480)
        let missingEvents = Events(), missingID = UUID()
        precondition(completed { device.attachSink(captureID: missingID, sink: missingEvents.append, completion: $0) })
        for i in 0..<65 { precondition(device.cacheMediaPCM16(original, sourceFrame: 10000 + Int64(i) * 480, input: input)) }
        precondition(device.enqueueMediaPCM16(playback, frames: 320, input: input, sourceFrame: 0))
        JRDTestPump(320, host(20))
        precondition(completed { device.detachSink(captureID: missingID, completion: $0) })
        let missing = missingEvents.values().compactMap { event -> JarvisDuplexAudioDevice.PCMBlock? in
            if case .pcm(let packet) = event, packet.track == .renderedOutput { return packet }; return nil
        }
        precondition(missing.count == 1 && missing[0].mediaSpans.isEmpty)
        precondition(missing[0].mediaCoverage.missingSourceFrames == 960)
        precondition(device.controlSynchronously(.retire))
        JRDTestReset()
        print("PASS: original 48k stereo over 16k mono playback, partial consumption, host anchors, muted voice, applied media gain, unconsumed exclusion and bounded-cache eviction")
    }

    static func main() {
        consumedMedia()
        let lowLevel = JRDRunLowLevelChecks()
        let delegate = JRDTestDelegate()
        let device = JarvisDuplexAudioDevice(voiceGeneration: UUID(), sessionLease: .init(
            acquire: { _, _ in
                precondition(JRDTestAllocations() == JRDTestDisposals() + 1,
                             "AudioUnit must be allocated before session activation")
                return .init(identity: "test-route", sampleRate: 16000,
                inputChannels: 1, outputChannels: 1, ioBufferDuration: 0.01,
                inputLatency: 0.005, outputLatency: 0.005) }, release: {}), onStatus: { _ in })
        precondition(device.initialize(with: delegate))
        precondition(device.initializeRecording() && device.initializePlayout())
        precondition(device.startRecording() && device.startPlayout())
        precondition(JRDTestAllocations() == 0, "hardware before explicit authorization")
        precondition(completed { device.authorizeSession(completion: $0) })
        precondition(JRDTestStarts() == 1 && delegate.parameterNotifications == 2)
        precondition(delegate.notificationViolations == 0)
#if DEBUG
        // Native BOOL fields import as ObjCBool, which JSONSerialization cannot
        // encode and raises an Objective-C exception rather than a Swift error.
        let diagnostic = device.inputDiagnosticSnapshot()
        precondition(JSONSerialization.isValidJSONObject(diagnostic))
        precondition(diagnostic["hardware_present"] as? Bool == true)
        precondition(diagnostic["recording_requested"] as? Bool == true)
        precondition(diagnostic["microphone_gate_open"] as? Bool == false)
        let unit = diagnostic["unit_state"] as? [String: Any]
        precondition(unit?["configured"] as? Bool == true)
        precondition(unit?["input_sample_rate"] as? Double == 16_000)
        precondition((try? JSONSerialization.data(withJSONObject: diagnostic)) != nil)
#endif

        let silentCapture = UUID(), overlappingCapture = UUID()
        device.setGates(microphone: true, output: true)
        device.setRecordingOutputSuppressed(true, recordingID: silentCapture)
        JRDTestPump(64, 12340)
        precondition(!JRDTestLastInputWasSilent() && JRDTestLastOutputWasSilent(),
                     "Silent recording must preserve microphone delivery")
        device.setGates(microphone: true, output: true)
        JRDTestPump(64, 12341)
        precondition(JRDTestLastOutputWasSilent(), "Normal response gates must not unmute a recording")
        device.setRecordingOutputSuppressed(true, recordingID: overlappingCapture)
        device.setRecordingOutputSuppressed(false, recordingID: silentCapture)
        JRDTestPump(64, 12342)
        precondition(JRDTestLastOutputWasSilent(), "One capture cannot release another capture's mute")
        device.setRecordingOutputSuppressed(false, recordingID: overlappingCapture)
        JRDTestPump(64, 12343)
        precondition(!JRDTestLastOutputWasSilent(), "Finishing the last silent capture restores playback")
        let mediaA = device.attachMediaSource()!
        let mediaB = device.attachMediaSource()!
        let mediaPCM = Data(repeating: 0, count: 160 * mediaB.channels * 2)
        precondition(mediaB.sampleRate == 16_000 && mediaB.sourceGeneration != mediaA.sourceGeneration)
        precondition(!device.enqueueMediaPCM16(mediaPCM, frames: 160, input: mediaA))
        precondition(device.enqueueMediaPCM16(mediaPCM, frames: 160, input: mediaB))
        precondition(!device.detachMediaSource(mediaA), "Old source stop must not stop its replacement")
        precondition(!device.setMediaVolume(.nan, muted: false))
        precondition(device.setMediaVolume(0.4, muted: false))
        device.setMediaUserSpeaking(true)
        device.setMediaOutputSuppressed(true, recordingID: silentCapture)
        device.setMediaOutputSuppressed(false, recordingID: silentCapture)
        device.setMediaUserSpeaking(false)
        precondition(device.detachMediaSource(mediaB))
        precondition(!device.enqueueMediaPCM16(mediaPCM, frames: 160, input: mediaB))
        device.setGates(microphone: false, output: false)

        // Failed close must not restart the retained AudioUnit, even through
        // an SDK lifecycle entry rather than an external dispatch wrapper.
        precondition(device.stopPlayout())
        JRDTestSetFailure(1)
        precondition(!device.stopRecording())
        precondition(!device.startRecording())
        precondition(JRDTestStarts() == 1 && JRDTestAllocations() == 1)
        JRDTestSetFailure(0)
        precondition(device.startRecording() && device.startPlayout())
        precondition(JRDTestAllocations() == 2 && JRDTestDisposals() == 1)
        precondition(delegate.interruptionNotifications == 2 && delegate.notificationViolations == 0)

        let events = Events(), capture = UUID(), startCount = JRDTestStarts()
        precondition(completed { device.attachSink(captureID: capture, sink: events.append, completion: $0) })
        device.setGates(microphone: false, output: false)
        JRDTestPump(64, 12345)
        precondition(JRDTestLastInputWasSilent() && JRDTestLastOutputWasSilent())
        precondition(completed { device.detachSink(captureID: capture, completion: $0) })
        precondition(JRDTestStarts() == startCount, "recording attachment restarted voice")
        let packets = events.values().compactMap { event -> JarvisDuplexAudioDevice.PCMBlock? in
            if case .pcm(let packet) = event { return packet }; return nil
        }
        precondition(packets.count == 2 && packets.allSatisfy {
            $0.captureID == capture && $0.hostTime == 12345 && $0.frames == 64 && !$0.softwareGateOpen
        })
        precondition(device.controlSynchronously(.interruptionBegan))
        precondition(JRDTestDisposals() == 2)
        precondition(device.controlSynchronously(.interruptionEnded))
        precondition(JRDTestAllocations() == 3)
        device.setRecordingOutputSuppressed(true, recordingID: silentCapture)
        let oldRouteMedia = device.attachMediaSource()!
        precondition(device.controlSynchronously(.routeChanged))
        precondition(!device.enqueueMediaPCM16(mediaPCM, frames: 160, input: oldRouteMedia),
                     "Route replacement requires freshly negotiated media PCM")
        precondition(JRDTestAllocations() == 4 && JRDTestDisposals() == 3)
        device.setGates(microphone: true, output: true)
        JRDTestPump(64, 12346)
        precondition(!JRDTestLastInputWasSilent() && JRDTestLastOutputWasSilent(),
                     "Recording mute must survive a replacement audio route")
        device.setRecordingOutputSuppressed(false, recordingID: silentCapture)
        JRDTestPump(64, 12347)
        precondition(!JRDTestLastOutputWasSilent())
        print("PASS: silent recording, independent capture owners, and route replacement")
        print("PASS: media source replacement, stale route rejection and independent media controls")
        precondition(device.controlSynchronously(.retire))
        precondition(JRDTestDisposals() == 4 && delegate.notificationViolations == 0)
        precondition(device.controlSynchronously(.retire))
        precondition(!device.controlSynchronously(.authorize))
        precondition(!device.startRecording() && !device.startPlayout())
#if DEBUG
        let retiredDiagnostic = device.inputDiagnosticSnapshot()
        precondition(JSONSerialization.isValidJSONObject(retiredDiagnostic))
        precondition(retiredDiagnostic["hardware_present"] as? Bool == false)
        print("PASS: running and retired microphone diagnostics serialize safely")
#endif
        // Allocation is idempotent, and format negotiation follows activation.
        let allocationBaseline = JRDTestAllocations()
        let prepared = JRDDuplexIO(delegate: delegate, sampleRate: 48_000,
            inputChannels: 1, outputChannels: 1, routeGeneration: 100)!
        precondition(prepared.prepare() == 0 && prepared.prepare() == 0)
        precondition(JRDTestAllocations() == allocationBaseline + 1)
        precondition(prepared.configure(sampleRate: 16_000, inputChannels: 2, outputChannels: 2) == 0)
        precondition(prepared.sampleRate == 16_000 && prepared.inputChannels == 2 && prepared.outputChannels == 2)
        precondition(prepared.configure(sampleRate: 48_000, inputChannels: 1, outputChannels: 1) != 0)
        precondition(prepared.close())

        // Failure before a lease is acquired must dispose the prepared unit,
        // without releasing an audio-session lease that was never obtained.
        let failedBaseline = JRDTestAllocations()
        let failedLease = JarvisDuplexAudioDevice(voiceGeneration: UUID(), sessionLease: .init(
            acquire: { _, _ in
                precondition(JRDTestAllocations() == JRDTestDisposals() + 1)
                throw NSError(domain: "duplex-fixture", code: 1)
            }, release: { preconditionFailure("Unacquired lease released") }), onStatus: { _ in })
        precondition(failedLease.initialize(with: delegate))
        precondition(failedLease.initializeRecording() && failedLease.startRecording())
        precondition(!completed { failedLease.authorizeSession(completion: $0) })
        precondition(JRDTestAllocations() == failedBaseline + 1)
        precondition(JRDTestAllocations() == JRDTestDisposals())
        precondition(failedLease.controlSynchronously(.retire))
        print("PASS: \(lowLevel) low-level groups and 8 adapter groups; AudioUnit hardware substituted, no sensors/provider")
    }
}
