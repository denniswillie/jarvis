import CryptoKit
import Foundation
import WebRTC
import XCTest

@MainActor
final class JarvisReplayTests: XCTestCase {
    private func pcm(_ samples: [Int16]) -> Data {
        Data(samples.flatMap { value -> [UInt8] in
            let bits = UInt16(bitPattern: value)
            return [UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)]
        })
    }
    private func plan(_ utterances: [JarvisReplayPlan.Utterance] = []) throws -> JarvisReplayPlan {
        try JarvisReplayPlan(runID: UUID(), duration: 10,
            frames: [0.0, 1, 2, 3, 4].map { .init(at: $0, image: Data([1])) }, utterances: utterances)
    }
    private func asset(_ data: Data, format: JarvisReplayAudioAsset.Format = .rawPCM16LEMono48000,
                       count: Int = 3, last: Int? = 1) -> JarvisReplayAudioAsset {
        JarvisReplayAudioAsset(fileURL: URL(fileURLWithPath: "/unused-fixture"), format: format,
            sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            sampleCount: count, thresholdPCM16: 16, lastAboveThresholdSampleIndex: last)
    }
    private func wav(_ pcm: Data, rate: Int = 48_000, channels: Int = 1) -> Data {
        func le(_ value: Int, bytes: Int) -> Data { Data((0..<bytes).map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) }) }
        var data = Data("RIFF".utf8); data += le(36 + pcm.count, bytes: 4); data += Data("WAVEfmt ".utf8)
        data += le(16, bytes: 4); data += le(1, bytes: 2); data += le(channels, bytes: 2)
        data += le(rate, bytes: 4); data += le(rate * channels * 2, bytes: 4)
        data += le(channels * 2, bytes: 2); data += le(16, bytes: 2); data += Data("data".utf8)
        data += le(pcm.count, bytes: 4); data += pcm
        return data
    }

    func testAtTimeNeverEmitsFutureSceneOrInput() throws {
        var timeline = JarvisReplayTimeline(plan: try plan([.init(trigger: .atTime(2), pcm16: pcm([100]))]))
        let first = timeline.advance(elapsed: 0, output: nil, inputIdle: true)
        XCTAssertEqual(first.frame, 0); XCTAssertNil(first.utterance)
        XCTAssertNil(timeline.advance(elapsed: 0.5, output: nil, inputIdle: true).frame)
        let next = timeline.advance(elapsed: 2, output: nil, inputIdle: true)
        XCTAssertEqual(next.frame, 2); XCTAssertEqual(next.skippedFrames, 1); XCTAssertEqual(next.utterance, 0)
    }
    func testSceneKeepsAdvancingWhileAwaitingDecodedOutput() throws {
        var timeline = JarvisReplayTimeline(plan: try plan([.init(
            trigger: .outputStart(afterCursor: 0, delay: 0.2, timeout: 5), pcm16: pcm([100]))]))
        XCTAssertEqual(timeline.advance(elapsed: 0, output: nil, inputIdle: true).frame, 0)
        let waiting = timeline.advance(elapsed: 3, output: nil, inputIdle: true)
        XCTAssertEqual(waiting.frame, 3); XCTAssertNil(waiting.utterance)
        let started = JarvisReplayOutputStart(cursor: 1, at: 3.1)
        XCTAssertEqual(timeline.advance(elapsed: 3.4, output: started, inputIdle: true).utterance, 0)
    }
    func testForeignOrOldCursorDoesNotWakeUtterance() throws {
        var timeline = JarvisReplayTimeline(plan: try plan([.init(
            trigger: .outputStart(afterCursor: 2, delay: 0, timeout: 2), pcm16: pcm([100]))]))
        XCTAssertNil(timeline.advance(elapsed: 1, output: .init(cursor: 2, at: 0.5), inputIdle: true).utterance)
        let failure = timeline.advance(elapsed: 2, output: nil, inputIdle: true)
        XCTAssertEqual(failure.failure, .outputWaitExpired); XCTAssertEqual(failure.frame, 2)
    }
    func testPlaybackDrainedIsExplicitlyUnsupported() {
        XCTAssertThrowsError(try plan([.init(trigger: .playbackDrained(timeout: 2), pcm16: pcm([10]))])) {
            XCTAssertEqual($0 as? JarvisReplayError, .unsupportedPlaybackDrain)
        }
    }
    func testOverlappingAudioFailsRatherThanSilentlyChangingSchedule() throws {
        var timeline = JarvisReplayTimeline(plan: try plan([.init(trigger: .atTime(0), pcm16: pcm([100]))]))
        XCTAssertEqual(timeline.advance(elapsed: 0, output: nil, inputIdle: false).failure, .overlappingInput)
    }
    func testEndAndBackwardsClockAreTerminal() throws {
        var timeline = JarvisReplayTimeline(plan: try plan())
        _ = timeline.advance(elapsed: 1, output: nil, inputIdle: true)
        XCTAssertEqual(timeline.advance(elapsed: 0, output: nil, inputIdle: true).failure, .invalidPlan)
        XCTAssertTrue(timeline.advance(elapsed: 1, output: nil, inputIdle: true).finished)
        var other = JarvisReplayTimeline(plan: try plan())
        XCTAssertTrue(other.advance(elapsed: 10, output: nil, inputIdle: true).finished)
    }
    func testPlanRejectsOversizeAndMalformedInput() {
        XCTAssertThrowsError(try plan([.init(trigger: .atTime(0), pcm16: Data([1]))]))
        XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 181,
            frames: [.init(at: 0, image: Data([1]))], utterances: []))
        XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 10,
            frames: [.init(at: 2, image: Data([1])), .init(at: 1, image: Data([1]))], utterances: []))
    }
    func testPreparedSceneDurationPreservesDefaultsAndAccepts180Seconds() throws {
        for duration in [45.0, 120, 165, 180] {
            let value = try JarvisReplayPlan(runID: UUID(), duration: duration,
                frames: [.init(at: duration - 1, image: Data([1]))],
                utterances: [.init(trigger: .atTime(duration - 0.5), pcm16: pcm([100]))])
            XCTAssertEqual(value.duration, duration)
        }
        for duration in [0.0, -1, 180.000_001, 181, Double.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: duration,
                frames: [.init(at: 0, image: Data([1]))], utterances: [])) {
                XCTAssertEqual($0 as? JarvisReplayError, .invalidPlan)
            }
        }
    }
    func testLongSceneDoesNotExtendAnyTriggerTimeoutPast120Seconds() throws {
        func triggers(_ timeout: TimeInterval) -> [JarvisReplayPlan.Trigger] {
            [.outputStart(afterCursor: 0, delay: 0.3, timeout: timeout),
             .responseOutputStart(.init(afterInputOrdinal: 0, delay: 0.3,
                 timeout: timeout, maxPCMAge: 0.05, requireTaskClaim: true)),
             .afterTaskClaim(afterInputOrdinal: 0, timeout: timeout),
             .afterInputComplete(afterInputOrdinal: 0, delay: 1, timeout: timeout)]
        }
        func make(_ trigger: JarvisReplayPlan.Trigger, duration: TimeInterval = 180) throws -> JarvisReplayPlan {
            try JarvisReplayPlan(runID: UUID(), duration: duration,
                frames: [.init(at: 0, image: Data([1]))],
                utterances: [.init(trigger: .atTime(0), pcm16: pcm([100])), .init(trigger: trigger, pcm16: pcm([100]))])
        }
        for trigger in triggers(120) { XCTAssertNoThrow(try make(trigger)) }
        for timeout in [120.000_001, 165, 180, 0, Double.nan, .infinity, -.infinity] {
            for trigger in triggers(timeout) { XCTAssertThrowsError(try make(trigger)) }
        }
        for trigger in triggers(46) { XCTAssertThrowsError(try make(trigger, duration: 45)) }
    }
    func testLongSceneKeepsTotalInputPCMAt60Seconds() throws {
        func make(_ count: Int) throws -> JarvisReplayPlan {
            try JarvisReplayPlan(runID: UUID(), duration: 180,
                frames: [.init(at: 0, image: Data([1]))],
                utterances: [.init(trigger: .atTime(0), pcm16: Data(repeating: 0, count: count))])
        }
        XCTAssertNoThrow(try make(60 * 48_000 * 2))
        XCTAssertThrowsError(try make(60 * 48_000 * 2 + 2))
    }
    func testPCMStartsGatedAndRecordsExactInputSamples() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        XCTAssertThrowsError(try state.enqueue(pcm([0, 100, -200, 0])))
        state.gateInput(true); try state.enqueue(pcm([0, 100, -200, 0]))
        let block = state.renderInput(at: 4)
        XCTAssertEqual(Array(block.prefix(4)), [0, 100, -200, 0]); XCTAssertEqual(block.count, 480)
        XCTAssertEqual(try XCTUnwrap(state.evidence.inputFirstNonSilent.first), 4 + 1.0 / 48_000, accuracy: 0.000001)
        XCTAssertEqual(try XCTUnwrap(state.evidence.inputLastNonSilent.first), 4 + 2.0 / 48_000, accuracy: 0.000001)
        XCTAssertTrue(state.inputIdle)
    }
    func testGateDropsPendingInputAndNeverReplaysItOnReopen() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.gateInput(true); try state.enqueue(pcm([100, 200]))
        state.gateInput(false); state.gateInput(true)
        XCTAssertTrue(state.renderInput(at: 1).allSatisfy { $0 == 0 })
    }
    func testDecodedGateDoesNotClaimHeardUsefulOrProviderAttributedOutput() throws {
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.gateInput(true); try state.enqueue(pcm([100]))
        state.observeDecoded([0, 100], at: 2)
        XCTAssertNil(state.latestOutput); XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
        state.gateOutput(true); state.observeDecoded([0, 100], at: 3)
        XCTAssertEqual(state.latestOutput?.cursor, 1)
        XCTAssertEqual(try XCTUnwrap(state.latestOutput).at, 3 + 1.0 / 48_000, accuracy: 0.000001)
        XCTAssertFalse(state.evidence.physicalPlaybackMeasured); XCTAssertNil(state.evidence.usefulAnswerOnset)
        XCTAssertEqual(state.evidence.boundary, "custom_native_adm_pcm_callback")
    }
    func testDecodedCaptureIsBoundedAndCloseIsPermanent() throws {
        var state = JarvisReplayPCMState(maxDuration: 0.01)
        state.gateOutput(true); state.observeDecoded([Int16](repeating: 100, count: 960), at: 1)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 960); XCTAssertTrue(state.evidence.truncated)
        state.close(); state.close(); state.gateInput(true); state.gateOutput(true)
        XCTAssertThrowsError(try state.enqueue(pcm([100])))
        state.observeDecoded([100], at: 2)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 960)
    }
    func testDecodedPCMUsesSelectedSceneCapacityIncluding180Seconds() {
        let block = [Int16](repeating: 100, count: 48_000)
        for duration in [45.0, 120, 165, 180] {
            var state = JarvisReplayPCMState(maxDuration: duration)
            state.gateOutput(true)
            for second in 0..<Int(duration) { state.observeDecoded(block, at: Double(second)) }
            XCTAssertEqual(state.evidence.decodedPCM16.count, Int(duration) * 96_000)
            XCTAssertFalse(state.evidence.truncated)
            state.observeDecoded([100], at: duration)
            XCTAssertTrue(state.evidence.truncated)
            XCTAssertEqual(state.evidence.decodedPCM16.count, Int(duration) * 96_000)
        }
    }
    func test180SecondDecodedMapIsBoundedIndependentlyOfPCMBytes() {
        var state = JarvisReplayPCMState(maxDuration: 180)
        state.gateOutput(true)
        for index in 0..<18_000 { state.observeDecoded([100], at: Double(index) / 100) }
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 18_000)
        XCTAssertFalse(state.evidence.decodedMappingTruncated)
        state.observeDecoded([100], at: 180)
        XCTAssertTrue(state.evidence.decodedMappingTruncated)
        XCTAssertTrue(state.evidence.truncated)
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 18_000)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 36_000)
    }
    func testDecodedCapacityNeverRetainsPartialPCM16Sample() {
        var state = JarvisReplayPCMState(maxDuration: 0.01002)
        state.gateOutput(true)
        state.observeDecoded([Int16](repeating: 100, count: 481), at: 0)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 960)
        XCTAssertTrue(state.evidence.truncated)
    }
    func testInvalidDurationRetiresSilentADMAndPCMWithoutAllocationOrConversionTrap() {
        for duration in [0.0, -1, 180.000_001, Double.greatestFiniteMagnitude, .nan, .infinity, -.infinity] {
            var state = JarvisReplayPCMState(maxDuration: duration)
            XCTAssertTrue(state.closed)
            state.gateInput(true); state.gateOutput(true)
            XCTAssertThrowsError(try state.enqueue(pcm([100])))
            state.observeDecoded([100], at: 0)
            XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
            let device = JarvisReplayAudioDevice(maxDuration: duration)
            XCTAssertTrue(device.isRetired); XCTAssertFalse(device.startRecording())
            XCTAssertTrue(device.close())
        }
    }
    func testAssetAcceptsExactRawPCMAndRejectsWrongHashOrMetadata() throws {
        let data = pcm([0, 100, 0]); let value = asset(data)
        XCTAssertEqual(try value.validatedPCM16(data), data)
        XCTAssertThrowsError(try value.validatedPCM16(pcm([0, 200, 0])))
        XCTAssertThrowsError(try asset(data, count: 4).validatedPCM16(data))
        XCTAssertThrowsError(try asset(data, last: 2).validatedPCM16(data))
    }
    func testAssetAccepts48kMonoPCM16WAV() throws {
        let data = pcm([0, -100, 0]); let container = wav(data)
        XCTAssertEqual(try asset(container, format: .wavPCM16Mono48000).validatedPCM16(container), data)
    }
    func testWAVRejectsDifferentRateStereoTruncationAndDuplicatePayload() {
        let data = pcm([0, 100, 0])
        for container in [wav(data, rate: 24_000), wav(data, channels: 2), Data(wav(data).dropLast())] {
            XCTAssertThrowsError(try asset(container, format: .wavPCM16Mono48000).validatedPCM16(container))
        }
    }
    func testAssetMetadataNeverBecomesPCMOrModelText() throws {
        let data = pcm([0, 100, 0])
        let loaded = try asset(data).validatedPCM16(data)
        var state = JarvisReplayPCMState(maxDuration: 1)
        state.gateInput(true); try state.enqueue(loaded)
        XCTAssertEqual(Array(state.renderInput(at: 0).prefix(3)), [0, 100, 0])
    }
    func testReplayScopeRequiresExactIsolationAndFreshSessionAttestation() throws {
        let run = UUID()
        let expected = JarvisReplayScopeExpectation(runID: run, ownerID: "eval-owner", areaID: "eval-area", memoryBankID: "eval-bank")
        var fields: [String: Any] = ["run_id": run.uuidString, "owner_id": "eval-owner", "area_id": "eval-area",
            "memory_bank_id": "eval-bank", "provenance": "frontend_replay", "isolation_enforced": true,
            "fresh_session_required": true, "external_notifications_enabled": false, "expires_at": "2030-01-01T00:00:00Z"]
        func scope(_ fields: [String: Any]) throws -> JarvisReplayScope {
            try JSONDecoder().decode(JarvisReplayScope.self, from: JSONSerialization.data(withJSONObject: fields))
        }
        try scope(fields).validate(expected, now: Date(timeIntervalSince1970: 0), remaining: 100)
        for (key, value) in [("run_id", UUID().uuidString), ("owner_id", "personal"),
                             ("area_id", "personal"), ("memory_bank_id", "personal"), ("provenance", "live_glasses")] {
            var wrong = fields; wrong[key] = value
            XCTAssertThrowsError(try scope(wrong).validate(expected, now: Date(timeIntervalSince1970: 0), remaining: 100))
        }
        for (key, value) in [("isolation_enforced", false), ("fresh_session_required", false), ("external_notifications_enabled", true)] {
            var wrong = fields; wrong[key] = value
            XCTAssertThrowsError(try scope(wrong).validate(expected, now: Date(timeIntervalSince1970: 0), remaining: 100))
        }
        fields["expires_at"] = "1970-01-01T00:00:01Z"
        XCTAssertThrowsError(try scope(fields).validate(expected, now: Date(timeIntervalSince1970: 0), remaining: 100))
    }
    func testCustomDeviceCanCreateRealNativeAudioOfferWithoutHardware() async throws {
        RTCInitializeSSL()
        let device = JarvisReplayAudioDevice(maxDuration: 1)
        let factory = RTCPeerConnectionFactory(encoderFactory: nil, decoderFactory: nil, audioDevice: device)
        let config = RTCConfiguration(); config.sdpSemantics = .unifiedPlan; config.iceServers = []
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        let peer = try XCTUnwrap(factory.peerConnection(with: config, constraints: constraints, delegate: nil))
        defer { peer.close(); device.close() }
        let track = factory.audioTrack(with: factory.audioSource(with: constraints), trackId: "fixture-replay")
        track.isEnabled = false; peer.add(track, streamIds: ["fixture"])
        _ = peer.dataChannel(forLabel: "oai-events", configuration: RTCDataChannelConfiguration())
        let offer = try await peer.offer(for: constraints)
        XCTAssertTrue(offer.sdp.contains("m=audio")); XCTAssertTrue(offer.sdp.contains("m=application"))
        XCTAssertFalse(device.isRecording); XCTAssertFalse(device.isPlaying)
    }
    func testTailBudgetPreservesMediaAndTriggerBoundsWithinExistingPCMMaximum() throws {
        for (duration, tail) in [(45.0, 20.0), (120, 20), (165, 15), (180, 0)] {
            let value = try JarvisReplayPlan(runID: UUID(), duration: duration,
                frames: [.init(at: duration - 1, image: Data([1]))], utterances: [])
            XCTAssertEqual(value.duration, duration); XCTAssertEqual(value.postMediaCaptureSeconds, tail)
            let device = JarvisReplayAudioDevice(plan: value)
            defer { device.close() }
            XCTAssertEqual(device.sceneDuration, duration); XCTAssertEqual(device.captureDuration, duration + tail)
            XCTAssertNoThrow(try value.validateAudioCapture(duration: device.sceneDuration, retired: device.isRetired,
                                                           capacity: device.captureDuration))
            XCTAssertThrowsError(try value.validateAudioCapture(duration: duration, retired: false, capacity: duration + tail + 1))
        }
        for tail in [-1.0, 20.0001, .nan, .infinity] {
            XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 45,
                frames: [.init(at: 0, image: Data([1]))], utterances: [], postMediaCaptureSeconds: tail))
        }
        XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 180,
            frames: [.init(at: 0, image: Data([1]))], utterances: [], postMediaCaptureSeconds: 1))
        XCTAssertEqual(JarvisReplayPlan.maximumTriggerTimeout, 120)
    }

    func testRecordingCapacitySharesExistingStartupAllowanceWithoutExtendingScene() throws {
        XCTAssertEqual(JarvisReplayPlan.startupAllowance, 30)
        for (scene, tail) in [(45.0, 20.0), (120, 20), (165, 15), (180, 0)] {
            let value = try JarvisReplayPlan(runID: UUID(), duration: scene,
                frames: [.init(at: 0, image: Data([1]))], utterances: [])
            let device = JarvisReplayAudioDevice(plan: value)
            defer { device.close() }
            XCTAssertEqual(value.duration, scene)
            XCTAssertEqual(value.postMediaCaptureSeconds, tail)
            XCTAssertEqual(value.captureDuration, scene + tail)
            XCTAssertEqual(value.recordingCapacityDuration, scene + tail + 30)
            XCTAssertEqual(device.sceneDuration, scene)
            XCTAssertEqual(device.captureDuration, value.captureDuration)
            XCTAssertEqual(device.recordingCapacityDuration, value.recordingCapacityDuration)
            XCTAssertLessThanOrEqual(device.recordingCapacityDuration, 210)
            XCTAssertNoThrow(try value.validateAudioCapture(duration: device.sceneDuration, retired: device.isRetired,
                capacity: device.captureDuration, recordingCapacity: device.recordingCapacityDuration))
            for wrongCapacity in [value.captureDuration, value.recordingCapacityDuration + 1, .nan, .infinity] {
                XCTAssertThrowsError(try value.validateAudioCapture(duration: scene, retired: false,
                    capacity: value.captureDuration, recordingCapacity: wrongCapacity))
            }
        }
        XCTAssertEqual(JarvisReplayPlan.maximumPostMediaCapture, 20)
        XCTAssertEqual(JarvisReplayPlan.maximumTriggerTimeout, 120)
    }

    func testEarlyOutputRetainsStartupAndExact45Plus20WithoutRebasingOrClaimingDrain() throws {
        let value = try JarvisReplayPlan(runID: UUID(), duration: 45,
            frames: [.init(at: 44, image: Data([1]))], utterances: [], postMediaCaptureSeconds: 20)
        let startup = [Int16](repeating: 111, count: 480)
        let media = [Int16](repeating: 222, count: 480)
        let tail = [Int16](repeating: 333, count: 480)
        // 130 ms reproduces the 7176 boundary; 30 seconds exercises the full
        // existing startup allowance, not a longer runtime or a shifted clock.
        for startupBlocks in [13, 3_000] {
            var state = JarvisReplayPCMState(maxDuration: value.captureDuration,
                                            startupAllowance: JarvisReplayPlan.startupAllowance)
            try state.register(value)
            state.gateOutput(true, at: 100)
            for index in 0..<startupBlocks {
                let at = 100 + Double(index) / 100
                state.observeDecoded(startup, at: at, callbackEnd: at + 0.0001)
            }
            let origin = 100 + Double(startupBlocks) / 100
            let anchor = JarvisReplayClockAnchor(monotonicBefore: origin,
                utc: Date(timeIntervalSince1970: 1_793_471_913), monotonicAfter: origin)
            try state.recordAnchor(anchor)
            XCTAssertEqual(state.evidence.decodedPCM16.count, startupBlocks * 960)
            XCTAssertEqual(state.evidence.decodedSampleMap.first?.callbackStart, 100)
            XCTAssertEqual(state.evidence.decodedSampleMap.first?.sampleOffset, 0)
            XCTAssertEqual(state.evidence.clockAnchor?.referenceUptime, origin)
            for index in 0..<4_500 {
                let at = origin + Double(index) / 100
                state.observeDecoded(media, at: at, callbackEnd: at + 0.0001)
            }
            state.beginOutputDrain(scheduled: origin + 45, observed: origin + 45, seconds: 20)
            for index in 0..<2_000 {
                let at = origin + 45 + Double(index) / 100
                state.observeDecoded(tail, at: at, callbackEnd: at + 0.0001)
            }
            let totalBlocks = startupBlocks + 6_500
            XCTAssertEqual(state.evidence.decodedBlocks, totalBlocks)
            XCTAssertEqual(state.evidence.decodedPCM16.count, totalBlocks * 960)
            XCTAssertEqual(state.evidence.decodedSampleMap.count, totalBlocks)
            XCTAssertEqual(Data(state.evidence.decodedPCM16.prefix(960)), pcm(startup))
            XCTAssertEqual(Data(state.evidence.decodedPCM16.suffix(960)), pcm(tail))
            XCTAssertEqual(state.evidence.decodedSampleMap[startupBlocks].callbackStart, origin)
            XCTAssertEqual(state.evidence.decodedSampleMap[startupBlocks].sampleOffset, startupBlocks * 480)
            XCTAssertEqual(try XCTUnwrap(state.evidence.decodedSampleMap.last).callbackStart,
                           origin + 64.99, accuracy: 0.000001)
            XCTAssertFalse(state.evidence.truncated)
            XCTAssertFalse(state.evidence.decodedMappingTruncated)
            XCTAssertEqual(state.evidence.clockAnchor?.referenceUptime, origin)
            XCTAssertEqual(state.evidence.clockAnchor?.utc, anchor.utc)
            XCTAssertFalse(state.pollOutputDrain(at: origin + 64.999))
            XCTAssertTrue(state.pollOutputDrain(at: origin + 65))
            XCTAssertEqual(state.evidence.outputDrain.status, "timeout")
            XCTAssertEqual(state.evidence.outputDrain.scheduledMediaEnd, origin + 45)
            XCTAssertEqual(state.evidence.outputDrain.deadline, origin + 65)
            XCTAssertEqual(state.evidence.outputDrain.json["decoded_drain_confirmed"] as? Bool, false)
            XCTAssertTrue(state.evidence.decodedSampleMap.allSatisfy { $0.responseIdentity == nil && $0.responseAssociation == nil })
            XCTAssertFalse(state.evidence.physicalPlaybackMeasured)
            XCTAssertNil(state.evidence.usefulAnswerOnset)
            state.close(at: origin + 65)
            XCTAssertEqual(state.evidence.outputDrain.status, "timeout")
            XCTAssertEqual(state.evidence.outputGates.first?.requestedAt, 100)
            XCTAssertEqual(state.evidence.outputGates.last?.requestedAt, origin + 65)
        }
    }

    func testStartupReservePCMRemainsBoundedAndReportsOverflow() {
        var state = JarvisReplayPCMState(maxDuration: 1, startupAllowance: JarvisReplayPlan.startupAllowance)
        state.gateOutput(true, at: 100)
        let block = [Int16](repeating: 100, count: 480)
        for index in 0..<3_100 { state.observeDecoded(block, at: 100 + Double(index) / 100) }
        XCTAssertEqual(state.evidence.decodedPCM16.count, 31 * 96_000)
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 3_100)
        XCTAssertFalse(state.evidence.truncated)
        state.observeDecoded(block, at: 131)
        XCTAssertTrue(state.evidence.truncated)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 31 * 96_000)
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 3_100)
    }

    func testStartupReserveMapHasIndependentFiniteCap() {
        var state = JarvisReplayPCMState(maxDuration: 1, startupAllowance: JarvisReplayPlan.startupAllowance)
        state.gateOutput(true, at: 100)
        for index in 0..<3_100 { state.observeDecoded([100], at: 100 + Double(index) / 100) }
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 3_100)
        XCTAssertFalse(state.evidence.decodedMappingTruncated)
        state.observeDecoded([100], at: 131)
        XCTAssertTrue(state.evidence.decodedMappingTruncated)
        XCTAssertTrue(state.evidence.truncated)
        XCTAssertEqual(state.evidence.decodedSampleMap.count, 3_100)
        XCTAssertEqual(state.evidence.decodedPCM16.count, 6_200)
    }

    func testStartupReserveRejectsUnboundedValuesWithoutRelaxingSceneLimit() {
        for allowance in [-1.0, 30.000_001, Double.greatestFiniteMagnitude, .nan, .infinity, -.infinity] {
            var state = JarvisReplayPCMState(maxDuration: 65, startupAllowance: allowance)
            XCTAssertTrue(state.closed)
            state.gateOutput(true); state.observeDecoded([100], at: 0)
            XCTAssertTrue(state.evidence.decodedPCM16.isEmpty)
        }
        for duration in [180.000_001, .nan, .infinity] {
            XCTAssertTrue(JarvisReplayPCMState(maxDuration: duration,
                startupAllowance: JarvisReplayPlan.startupAllowance).closed)
        }
        XCTAssertFalse(JarvisReplayPCMState(maxDuration: 180,
            startupAllowance: JarvisReplayPlan.startupAllowance).closed)
    }

    func testCaptureTailDoesNotEmitFutureFramesOrExtendStimulusDeadline() throws {
        let value = try JarvisReplayPlan(runID: UUID(), duration: 45,
            frames: [.init(at: 44, image: Data([1]))], utterances: [], postMediaCaptureSeconds: 20)
        var timeline = JarvisReplayTimeline(plan: value)
        XCTAssertEqual(timeline.advance(elapsed: 44, output: nil, inputIdle: true).frame, 0)
        for time in [45.0, 50, 65] {
            let step = timeline.advance(elapsed: time, output: nil, inputIdle: true)
            XCTAssertTrue(step.finished); XCTAssertNil(step.frame); XCTAssertNil(step.utterance)
        }
        var state = JarvisReplayPCMState(maxDuration: value.captureDuration)
        try state.register(value); state.gateInput(true)
        state.beginOutputDrain(scheduled: 45, observed: 45, seconds: 20)
        XCTAssertThrowsError(try state.enqueue(pcm([100])))
        XCTAssertTrue(state.renderInput(at: 46).allSatisfy { $0 == 0 })
        XCTAssertFalse(state.pollOutputDrain(at: 64.999))
        XCTAssertTrue(state.pollOutputDrain(at: 65))
        XCTAssertEqual(state.evidence.outputDrain.status, "timeout")
        XCTAssertEqual(state.evidence.outputDrain.scheduledMediaEnd, 45)
        XCTAssertEqual(state.evidence.outputDrain.finishedAt, 65)
    }

    private func drainState() throws -> (JarvisReplayPCMState, UUID) {
        var state = JarvisReplayPCMState(maxDuration: 30)
        let value = try JarvisReplayPlan(runID: UUID(), duration: 6,
            frames: [.init(at: 0, image: Data([1]))], utterances: [])
        try state.register(value)
        let connection = UUID()
        state.gateOutput(true, at: 1, responseID: "resp_drain", itemID: "item_drain", connectionID: connection)
        return (state, connection)
    }
    private func drainControl(_ state: inout JarvisReplayPCMState, connection: UUID, type: String, at: Double,
                              response: String = "resp_drain") {
        state.recordControlEvent(.providerDataChannelHandled,
            fields: ["connection_id": connection.uuidString.lowercased(), "response_id": response, "event_type": type], at: at)
    }
    private func quiet(_ state: inout JarvisReplayPCMState, from: Double, blocks: Int = 61, gap: Double = 0.01) {
        for index in 0..<blocks {
            let now = from + Double(index) * gap
            state.observeDecoded([Int16](repeating: 0, count: 480), at: now, callbackEnd: now + 0.001,
                responseIdentity: state.currentResponseIdentity, responseAssociation: state.currentResponseAssociation)
        }
    }

    func testNaturalStopAndGenerationDoneNeedActualContinuousAuthorizedQuiet() throws {
        var (state, connection) = try drainState()
        drainControl(&state, connection: connection, type: "response.done", at: 4)
        quiet(&state, from: 5)
        drainControl(&state, connection: connection, type: "output_audio_buffer.stopped", at: 5.61)
        state.gateOutput(false, at: 5.62)
        state.beginOutputDrain(scheduled: 6, observed: 6, seconds: 20)
        XCTAssertTrue(state.pollOutputDrain(at: 6))
        XCTAssertEqual(state.evidence.outputDrain.status, "decoded_quiet_observed")
        XCTAssertEqual(state.evidence.outputDrain.json["decoded_drain_confirmed"] as? Bool, false)
        XCTAssertFalse(state.closed)
        state.close(at: 6.01)
        XCTAssertEqual(state.evidence.outputDrain.status, "decoded_quiet_observed")
    }

    func testDoneOrStopAloneCannotProveDrain() throws {
        for type in ["response.done", "output_audio_buffer.stopped"] {
            var (state, connection) = try drainState()
            quiet(&state, from: 5)
            drainControl(&state, connection: connection, type: type, at: 5.61)
            state.beginOutputDrain(scheduled: 6, observed: 6, seconds: 20)
            XCTAssertFalse(state.pollOutputDrain(at: 7))
            XCTAssertTrue(state.pollOutputDrain(at: 26))
            XCTAssertEqual(state.evidence.outputDrain.status, "timeout")
        }
    }

    func testGatedZerosAndSparseCallbacksCannotProveDrain() throws {
        for gated in [true, false] {
            var (state, connection) = try drainState()
            drainControl(&state, connection: connection, type: "response.done", at: 4)
            if gated { state.gateOutput(false, at: 4.1) }
            quiet(&state, from: 5, gap: gated ? 0.01 : 0.04)
            drainControl(&state, connection: connection, type: "output_audio_buffer.stopped", at: 7.41)
            state.beginOutputDrain(scheduled: 8, observed: 8, seconds: 20)
            XCTAssertFalse(state.pollOutputDrain(at: 9))
            XCTAssertTrue(state.pollOutputDrain(at: 28))
            XCTAssertEqual(state.evidence.outputDrain.status, "timeout")
        }
    }

    func testActiveTailWrongIdentityAndCallbackFailureRemainUnknownUntilTimeout() throws {
        for failure in ["active", "wrong_identity", "callback_failure", "local_fence"] {
            var (state, connection) = try drainState()
            drainControl(&state, connection: connection, type: "response.done", at: 4)
            quiet(&state, from: 5)
            drainControl(&state, connection: connection, type: "output_audio_buffer.stopped", at: 5.61,
                         response: failure == "wrong_identity" ? "resp_other" : "resp_drain")
            if failure == "active" {
                state.observeDecoded([Int16](repeating: 200, count: 480), at: 5.62, callbackEnd: 5.621,
                    responseIdentity: state.currentResponseIdentity, responseAssociation: state.currentResponseAssociation)
            }
            if failure == "callback_failure" { state.callbackFailed(-1) }
            if failure == "local_fence" { state.recordLocalInterruption(.clientInterrupt, at: 5.62) }
            state.beginOutputDrain(scheduled: 6, observed: 6, seconds: 20)
            XCTAssertFalse(state.pollOutputDrain(at: 6.1), failure)
            XCTAssertTrue(state.pollOutputDrain(at: 26), failure)
            XCTAssertEqual(state.evidence.outputDrain.status, "timeout")
        }
    }

    func testTailKeepsLateOutputAndRecordsForcedDeadlineSeparately() throws {
        var (state, _) = try drainState()
        state.beginOutputDrain(scheduled: 6, observed: 6.01, seconds: 20)
        let identity = state.currentResponseIdentity
        state.observeDecoded([200, 201], at: 25.99, callbackEnd: 25.991, responseIdentity: identity)
        XCTAssertEqual(state.evidence.decodedPCM16, pcm([200, 201]))
        XCTAssertTrue(state.pollOutputDrain(at: 26.01))
        XCTAssertEqual(state.evidence.outputDrain.observedMediaEnd, 6.01)
        XCTAssertEqual(state.evidence.outputDrain.deadline, 26)
        XCTAssertEqual(state.evidence.outputDrain.finishedAt, 26.01)
        XCTAssertNil(state.evidence.usefulAnswerOnset)
        var (disabled, _) = try drainState()
        disabled.beginOutputDrain(scheduled: 6, observed: 6, seconds: 0)
        XCTAssertTrue(disabled.pollOutputDrain(at: 6)); XCTAssertEqual(disabled.evidence.outputDrain.status, "disabled")
    }

    func testIncompleteLastInputCannotUseAnEarlierResponseAsDrainProof() throws {
        var state = JarvisReplayPCMState(maxDuration: 30)
        let value = try plan([.init(trigger: .atTime(2), pcm16: pcm([100]))])
        try state.register(value)
        let connection = UUID()
        state.gateOutput(true, at: 1, responseID: "resp_drain", itemID: "item_drain", connectionID: connection)
        drainControl(&state, connection: connection, type: "response.done", at: 4)
        quiet(&state, from: 5)
        drainControl(&state, connection: connection, type: "output_audio_buffer.stopped", at: 5.61)
        state.beginOutputDrain(scheduled: 6, observed: 6, seconds: 20)
        XCTAssertFalse(state.evidence.outputDrain.requiredInputsComplete)
        XCTAssertFalse(state.pollOutputDrain(at: 7))
        state.close(at: 8)
        XCTAssertEqual(state.evidence.outputDrain.status, "closed_early")
    }

    func testSameResponseItemTransitionRetainsResponseEpochButNeverRelabelsSamples() throws {
        var (state, connection) = try drainState()
        let original = try XCTUnwrap(state.currentResponseIdentity)
        let response = try XCTUnwrap(state.currentResponseAssociation)
        state.gateOutput(true, at: 2, responseID: "resp_drain", itemID: "item_final", connectionID: connection)
        XCTAssertNil(state.currentResponseIdentity)
        XCTAssertEqual(state.currentResponseAssociation, response)
        XCTAssertTrue(state.outputEnabled)
        state.observeDecoded([200], at: 2.01, callbackEnd: 2.011,
            responseIdentity: original, responseAssociation: response)
        let block = try XCTUnwrap(state.evidence.decodedSampleMap.last)
        XCTAssertNil(block.responseIdentity); XCTAssertEqual(block.responseAssociation, response)
        XCTAssertEqual(state.evidence.uncorrelatedDecodedBlocks, 1)
        XCTAssertEqual(state.evidence.responseUncorrelatedDecodedBlocks, 0)
        state.gateOutput(true, at: 2.02, responseID: "resp_drain", itemID: "item_drain", connectionID: connection)
        XCTAssertNil(state.currentResponseIdentity)
        XCTAssertEqual(state.currentResponseAssociation?.outputEpoch, original.outputEpoch)
        XCTAssertNil(state.responseSnapshot(afterInputOrdinal: 0).output)
        let report = JarvisReplayEvidenceReport.make(evidence: state.evidence, runID: UUID(), finalUIStatus: "fixture")
        let map = try XCTUnwrap(report["decoded_sample_map"] as? [[String: Any]])
        XCTAssertTrue(map[0]["response_identity"] is NSNull)
        XCTAssertNil((map[0]["response_association"] as? [String: Any])?["item_id"])
        XCTAssertNil((map[0]["response_association"] as? [String: Any])?["phase"])
    }

    func testResponseAssociationCannotSurviveFenceOrCrossCallbackOwnership() throws {
        var (state, connection) = try drainState()
        let old = try XCTUnwrap(state.currentResponseAssociation)
        state.gateOutput(false, at: 2)
        XCTAssertNil(state.currentResponseAssociation)
        state.gateOutput(true, at: 3, responseID: "resp_new", itemID: "item_new", connectionID: connection)
        state.observeDecoded([200], at: 3.01, callbackEnd: 3.011, responseAssociation: old)
        XCTAssertNil(state.evidence.decodedSampleMap.last?.responseAssociation)
        state.gateOutput(false, at: 4)
        state.gateOutput(true, at: 5, responseID: "resp_drain", itemID: "item_drain", connectionID: connection)
        XCTAssertNil(state.currentResponseAssociation)
        state.gateOutput(true, at: 6, responseID: "resp_third", itemID: "item_third", connectionID: UUID())
        XCTAssertNil(state.currentResponseAssociation); XCTAssertTrue(state.evidence.responseEvidenceInvalid)
    }

    private func clockPlan(now: Double = 100, offset: Double = 67) throws -> (JarvisReplayPlan, JarvisReplaySourceAnchor, UUID) {
        let run = UUID(), first = UUID(), second = UUID(), boot = UUID()
        let anchor = JarvisReplaySourceAnchor(runID: run, invocationID: first,
            clock: .init(monotonicBefore: now, utc: Date(timeIntervalSince1970: 1_793_456_000), monotonicAfter: now), bootID: boot)
        let data = try JSONEncoder().encode(anchor)
        let value = try JarvisReplayPlan(runID: run, duration: 28, frames: [],
            utterances: [.init(trigger: .atTime(8), pcm16: pcm([100]))], invocationID: second,
            sourceClock: .init(phaseStart: offset, anchor: anchor, anchorData: data))
        return (value, anchor, boot)
    }

    func testAnchoredZeroCameraUsesOriginalClockAndRefusesMissedQueryInsteadOfRebasing() throws {
        let (value, anchor, boot) = try clockPlan()
        let measured = JarvisReplayClockAnchor(monotonicBefore: 170, utc: Date(timeIntervalSince1970: 1_900_000_000), monotonicAfter: 170)
        let resolved = try XCTUnwrap(value.sourceClock).resolve(plan: value, measured: measured, bootID: boot, now: 170)
        XCTAssertEqual(resolved.phase.referenceUptime, 167)
        XCTAssertEqual(resolved.phase.utc, anchor.clock.utc.addingTimeInterval(67))
        var timeline = JarvisReplayTimeline(plan: value)
        XCTAssertNil(timeline.advance(elapsed: 3, output: nil, inputIdle: true).frame)
        XCTAssertNil(timeline.advance(elapsed: 7.99, output: nil, inputIdle: true).utterance)
        XCTAssertEqual(timeline.advance(elapsed: 8, output: nil, inputIdle: true).utterance, 0)
        XCTAssertThrowsError(try value.sourceClock!.resolve(plan: value, measured: measured, bootID: boot, now: 175)) {
            XCTAssertEqual($0 as? JarvisReplayError, .sourceClockLate)
        }
        XCTAssertThrowsError(try value.sourceClock!.resolve(plan: value, measured: measured, bootID: UUID(), now: 170))
        XCTAssertThrowsError(try anchor.validate(runID: UUID(), bootID: boot, now: 170))
        XCTAssertThrowsError(try anchor.validate(runID: value.runID, bootID: boot, now: 99))
    }

    func testSourceClockAndInvocationOptInPreserveLegacyPathAndRejectUnboundZeroCamera() throws {
        let old = try plan(), documents = URL(fileURLWithPath: "/fixture/Documents")
        XCTAssertEqual(old.captureRoot(documents: documents).path, "/fixture/Documents/Replay/\(old.runID.uuidString.lowercased())")
        let (value, _, _) = try clockPlan()
        XCTAssertEqual(value.captureRoot(documents: documents).lastPathComponent, value.invocationID!.uuidString.lowercased())
        XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 28, frames: [], utterances: [], invocationID: UUID()))
        XCTAssertThrowsError(try JarvisReplayPlan(runID: UUID(), duration: 28, frames: [], utterances: [],
            invocationID: UUID(), sourceClock: .init(phaseStart: 0, anchor: nil, anchorData: nil)))
    }

    func testActualBootIdentityIsAvailableAndStableWithoutWallClockInference() throws {
        let first = try JarvisReplaySourceAnchor.systemBootID()
        XCTAssertEqual(first, try JarvisReplaySourceAnchor.systemBootID())
    }

    func testDeviceShutdownIsIdempotentAndRejectsQueuedInput() {
        let device = JarvisReplayAudioDevice(maxDuration: 1)
        device.close(); device.close(); device.setInputEnabled(true); device.setOutputEnabled(true)
        XCTAssertThrowsError(try device.enqueuePCM16(pcm([100])))
        XCTAssertFalse(device.startRecording()); XCTAssertFalse(device.startPlayout())
    }
}
