#if DEBUG
import Foundation
import Darwin

enum JarvisReplayError: Error, Equatable {
    case invalidPlan, unsupportedPlaybackDrain, overlappingInput, inputNotReady
    case outputWaitExpired, expired, retired, invalidImage, isolationUnavailable
    case unsupportedResponseIdentity, inputCompletionUnavailable, responseCandidateChanged
    case taskClaimUnavailable, taskCompletionUnavailable, providerInputFinalizationUnavailable
    case sourceClockUnavailable, sourceClockMismatch, sourceClockLate
    case manualIdentityMismatch, manualInputIncomplete, manualReadyExpired, memoryReadinessUnavailable
}

/// Test-only immutable measured origin. Continuations never manufacture a new
/// source timestamp, infer a boot identity from UTC, or pause elapsed time.
struct JarvisReplaySourceAnchor: Codable {
    let format: String
    let run_id: String
    let origin_invocation_id: String
    let monotonic_before_seconds: Double
    let monotonic_after_seconds: Double
    let utc_unix_seconds: Double
    let boot_session_id: String

    init(runID: UUID, invocationID: UUID, clock: JarvisReplayClockAnchor, bootID: UUID) {
        format = "jarvis.replay.source_clock.v1"
        run_id = runID.uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        origin_invocation_id = invocationID.uuidString.lowercased()
        monotonic_before_seconds = clock.monotonicBefore
        monotonic_after_seconds = clock.monotonicAfter
        utc_unix_seconds = clock.utc.timeIntervalSince1970
        boot_session_id = bootID.uuidString.lowercased()
    }
    var clock: JarvisReplayClockAnchor {
        .init(monotonicBefore: monotonic_before_seconds, utc: Date(timeIntervalSince1970: utc_unix_seconds),
              monotonicAfter: monotonic_after_seconds)
    }
    func validate(runID: UUID, bootID: UUID, now: TimeInterval) throws {
        guard format == "jarvis.replay.source_clock.v1",
              run_id == runID.uuidString.replacingOccurrences(of: "-", with: "").lowercased(),
              UUID(uuidString: origin_invocation_id) != nil, UUID(uuidString: boot_session_id) == bootID,
              monotonic_before_seconds.isFinite, monotonic_before_seconds >= 0,
              monotonic_after_seconds.isFinite, monotonic_after_seconds >= monotonic_before_seconds,
              monotonic_after_seconds - monotonic_before_seconds <= 1,
              now.isFinite, now >= monotonic_after_seconds,
              utc_unix_seconds.isFinite, utc_unix_seconds > 0, utc_unix_seconds < 4_102_444_800 else {
            throw JarvisReplayError.sourceClockMismatch
        }
    }
    static func systemBootID() throws -> UUID {
        var size = 0
        guard sysctlbyname("kern.bootsessionuuid", nil, &size, nil, 0) == 0, size > 0, size <= 128 else {
            throw JarvisReplayError.sourceClockUnavailable
        }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("kern.bootsessionuuid", &bytes, &size, nil, 0) == 0,
              let id = UUID(uuidString: String(cString: bytes)) else { throw JarvisReplayError.sourceClockUnavailable }
        return id
    }
}

struct JarvisReplaySourceClock {
    let phaseStart: TimeInterval
    let anchor: JarvisReplaySourceAnchor?
    let anchorData: Data?

    func resolve(plan: JarvisReplayPlan, measured: JarvisReplayClockAnchor, bootID: UUID,
                 now: TimeInterval) throws -> (source: JarvisReplaySourceAnchor, phase: JarvisReplayClockAnchor) {
        guard let invocation = plan.invocationID else { throw JarvisReplayError.sourceClockMismatch }
        let source = anchor ?? JarvisReplaySourceAnchor(runID: plan.runID, invocationID: invocation,
                                                       clock: measured, bootID: bootID)
        try source.validate(runID: plan.runID, bootID: bootID, now: now)
        let original = source.clock
        let phase = JarvisReplayClockAnchor(monotonicBefore: original.monotonicBefore + phaseStart,
            utc: original.utc.addingTimeInterval(phaseStart), monotonicAfter: original.monotonicAfter + phaseStart)
        if anchor != nil {
            let firstInput = plan.utterances.compactMap { utterance -> Double? in
                if case .atTime(let at) = utterance.trigger { return at }
                if case .afterManualResponse(let trigger) = utterance.trigger { return trigger.at }; return nil
            }.min() ?? plan.duration
            guard now - phase.referenceUptime < min(firstInput, plan.duration) else {
                throw JarvisReplayError.sourceClockLate
            }
        }
        return (source, phase)
    }
}

/// Local-only scheduling metadata. None of this structure is sent to a model.
struct JarvisReplayPlan {
    static let maximumSceneDuration: TimeInterval = 180
    static let maximumPostMediaCapture: TimeInterval = 20
    // Existing replay authorization/watchdog allowance, not extra scene time.
    // Native output may begin before the visual-ready scene clock is anchored.
    static let startupAllowance: TimeInterval = 30
    // A longer scene does not authorize longer task or response waits.
    static let maximumTriggerTimeout: TimeInterval = 120
    static func isValidSceneDuration(_ duration: TimeInterval) -> Bool {
        duration.isFinite && duration > 0 && duration <= maximumSceneDuration
    }

    // Shared by the capture source and provider-free tests; no camera dependencies.
    func validateAudioCapture(duration: TimeInterval, retired: Bool, capacity: TimeInterval? = nil,
                              recordingCapacity: TimeInterval? = nil) throws {
        guard !retired, Self.isValidSceneDuration(duration), duration == self.duration,
              capacity == nil || capacity == captureDuration,
              recordingCapacity == nil || recordingCapacity == recordingCapacityDuration else {
            throw JarvisReplayError.invalidPlan
        }
    }

    struct Frame { let at: TimeInterval; let image: Data }
    enum Trigger {
        case atTime(TimeInterval)
        // A local decoded-output cursor, NOT a provider response identifier.
        case outputStart(afterCursor: Int, delay: TimeInterval, timeout: TimeInterval)
        case responseOutputStart(JarvisReplayResponseTrigger)
        case playbackDrained(timeout: TimeInterval)
        case afterManualResponse(JarvisReplayManualResponseTrigger)
        case afterTaskClaim(afterInputOrdinal: Int, timeout: TimeInterval)
        case afterInputComplete(afterInputOrdinal: Int, delay: TimeInterval, timeout: TimeInterval)
        case afterProviderInputFinalized(afterInputOrdinal: Int, timeout: TimeInterval)

        var requiresTaskClaim: Bool {
            switch self {
            case .afterTaskClaim: return true
            case .responseOutputStart(let value): return value.requireTaskClaim
            default: return false
            }
        }
        var isTaskTrigger: Bool {
            if case .afterInputComplete = self { return true }
            return requiresTaskClaim
        }
    }
    struct Utterance {
        let trigger: Trigger
        let pcm16: Data
        let clipID: String?
        let assetActivityThreshold: Int?
        let assetFirstActiveSample: Int?
        let assetLastActiveSampleExclusive: Int?
        init(trigger: Trigger, pcm16: Data, clipID: String? = nil, assetActivityThreshold: Int? = nil,
             assetFirstActiveSample: Int? = nil, assetLastActiveSampleExclusive: Int? = nil) {
            self.trigger = trigger; self.pcm16 = pcm16; self.clipID = clipID
            self.assetActivityThreshold = assetActivityThreshold
            self.assetFirstActiveSample = assetFirstActiveSample
            self.assetLastActiveSampleExclusive = assetLastActiveSampleExclusive
        }
    }
    let runID: UUID
    let duration: TimeInterval
    let frames: [Frame]
    let utterances: [Utterance]
    let postMediaCaptureSeconds: TimeInterval
    let invocationID: UUID?
    let sourceClock: JarvisReplaySourceClock?
    var captureDuration: TimeInterval { duration + postMediaCaptureSeconds }
    var recordingCapacityDuration: TimeInterval { captureDuration + Self.startupAllowance }
    func captureRoot(documents: URL) -> URL {
        let root = documents.appendingPathComponent("Replay/\(runID.uuidString.lowercased())")
        return invocationID.map { root.appendingPathComponent($0.uuidString.lowercased()) } ?? root
    }

    init(runID: UUID, duration: TimeInterval, frames: [Frame], utterances: [Utterance],
         postMediaCaptureSeconds: TimeInterval? = nil, invocationID: UUID? = nil,
         sourceClock: JarvisReplaySourceClock? = nil, memoryReadinessDeclared: Bool = false) throws {
        let tail = postMediaCaptureSeconds ?? min(Self.maximumPostMediaCapture, Self.maximumSceneDuration - duration)
        guard Self.isValidSceneDuration(duration),
              tail.isFinite, tail >= 0, tail <= Self.maximumPostMediaCapture, duration + tail <= Self.maximumSceneDuration,
              (!frames.isEmpty || (invocationID != nil && (sourceClock?.anchor != nil
                || (memoryReadinessDeclared && sourceClock != nil)))),
              frames.count <= 240, utterances.count <= 20,
              frames.reduce(0, { $0 + $1.image.count }) <= 64 * 1024 * 1024,
              utterances.reduce(0, { $0 + $1.pcm16.count }) <= 60 * 48_000 * 2 else {
            throw JarvisReplayError.invalidPlan
        }
        if let clock = sourceClock {
            guard invocationID != nil, clock.phaseStart.isFinite, clock.phaseStart >= 0,
                  clock.phaseStart <= Self.maximumSceneDuration,
                  (clock.anchor == nil && clock.anchorData == nil && clock.phaseStart == 0)
                    || (clock.anchor != nil && clock.anchorData != nil && clock.phaseStart > 0),
                  clock.anchor.map({ UUID(uuidString: $0.origin_invocation_id) != invocationID }) ?? true,
                  utterances.allSatisfy({
                      if case .atTime = $0.trigger { return true }
                      if case .afterManualResponse = $0.trigger { return true }
                      return false
                  }) else {
                throw JarvisReplayError.invalidPlan
            }
        }
        var previous: TimeInterval = -1
        for frame in frames {
            guard frame.at.isFinite, frame.at >= 0, frame.at < duration, frame.at > previous,
                  !frame.image.isEmpty, frame.image.count <= 4 * 1024 * 1024 else {
                throw JarvisReplayError.invalidPlan
            }
            previous = frame.at
        }
        for (ordinal, utterance) in utterances.enumerated() {
            guard !utterance.pcm16.isEmpty, utterance.pcm16.count.isMultiple(of: 2) else {
                throw JarvisReplayError.invalidPlan
            }
            if let clipID = utterance.clipID, clipID.isEmpty || clipID.utf8.count > 256 { throw JarvisReplayError.invalidPlan }
            if let threshold = utterance.assetActivityThreshold, !(1...32767).contains(threshold) { throw JarvisReplayError.invalidPlan }
            switch utterance.trigger {
            case .atTime(let value):
                guard value.isFinite, value >= 0,
                      value + Double(utterance.pcm16.count) / 96_000 <= duration else {
                    throw JarvisReplayError.invalidPlan
                }
            case .outputStart(let cursor, let delay, let timeout):
                guard cursor >= 0, delay.isFinite, delay >= 0, timeout.isFinite,
                      timeout > 0, timeout <= min(duration, Self.maximumTriggerTimeout), delay < timeout else {
                    throw JarvisReplayError.invalidPlan
                }
            case .responseOutputStart(let trigger):
                guard trigger.afterInputOrdinal >= 0, trigger.afterInputOrdinal < ordinal,
                      trigger.delay.isFinite, trigger.delay >= 0,
                      trigger.timeout.isFinite, trigger.timeout > 0,
                      trigger.timeout <= min(duration, Self.maximumTriggerTimeout),
                      trigger.delay < trigger.timeout, trigger.maxPCMAge.isFinite,
                      trigger.maxPCMAge > 0, trigger.maxPCMAge <= trigger.timeout,
                      utterance.pcm16.contains(where: { $0 != 0 }),
                      trigger.responseID.map({ JarvisReplayResponseIdentity.safeToken($0, field: "response_id") }) ?? true else {
                    throw JarvisReplayError.invalidPlan
                }
            case .afterTaskClaim(let previous, let timeout), .afterInputComplete(let previous, _, let timeout):
                guard previous >= 0, previous < ordinal, timeout.isFinite, timeout > 0,
                      timeout <= min(duration, Self.maximumTriggerTimeout) else {
                    throw JarvisReplayError.invalidPlan
                }
                if case .afterInputComplete(_, let delay, _) = utterance.trigger {
                    guard delay.isFinite, delay >= 0, delay <= timeout else { throw JarvisReplayError.invalidPlan }
                }
            case .afterProviderInputFinalized(let previous, let timeout):
                guard previous == ordinal - 1, previous >= 0, timeout.isFinite, timeout > 0,
                      timeout <= min(duration, Self.maximumTriggerTimeout) else {
                    throw JarvisReplayError.invalidPlan
                }
            case .afterManualResponse(let trigger):
                try trigger.validate(ordinal: ordinal, duration: duration, pcmSeconds: Double(utterance.pcm16.count) / 96_000)
            case .playbackDrained:
                throw JarvisReplayError.unsupportedPlaybackDrain
            }
        }
        self.runID = runID; self.duration = duration; self.frames = frames; self.utterances = utterances
        self.postMediaCaptureSeconds = tail; self.invocationID = invocationID; self.sourceClock = sourceClock
    }
}

struct JarvisReplayOutputStart: Equatable {
    let cursor: Int
    let at: TimeInterval
}

/// Explicit control identity, not a per-RTP-packet or semantic request attribution.
struct JarvisReplayResponseIdentity: Equatable, Hashable {
    let connectionID: UUID
    let responseID: String
    let itemID: String
    let outputEpoch: Int
    var responseKey: String { connectionID.uuidString.lowercased() + ":" + responseID }
    var association: JarvisReplayResponseAssociation {
        .init(connectionID: connectionID, responseID: responseID, outputEpoch: outputEpoch)
    }
    static func safeToken(_ value: String, field: String) -> Bool {
        (RealtimeDiagnostics.safeFields([field: value])[field] as? String) == value
    }
}

/// Response/epoch control association only, intentionally no item or phase.
struct JarvisReplayResponseAssociation: Equatable {
    let connectionID: UUID
    let responseID: String
    let outputEpoch: Int
    var responseKey: String { connectionID.uuidString.lowercased() + ":" + responseID }
    var json: [String: Any] {
        ["connection_id": connectionID.uuidString.lowercased(), "response_id": responseID,
         "local_output_epoch": outputEpoch]
    }
}

/// Bounded observation of decoded quiescence, not a speaker/drain authority.
/// Gated zero blocks NEVER extend quiet coverage. A natural stop must be
/// adjacent to 500ms of actual, continuously authorized decoded quiet.
struct JarvisReplayOutputDrain {
    var status = "not_started"
    var scheduledMediaEnd: TimeInterval?
    var observedMediaEnd: TimeInterval?
    var deadline: TimeInterval?
    var finishedAt: TimeInterval?
    var requiredInputsComplete = false
    var association: JarvisReplayResponseAssociation?
    var admittedAt: TimeInterval?
    var generationDoneAt: TimeInterval?
    var naturalStopAt: TimeInterval?
    var quietStart: TimeInterval?
    var quietEnd: TimeInterval?
    var quietBlocks = 0
    private var previousQuietStart: TimeInterval?

    mutating func bind(_ value: JarvisReplayResponseAssociation, at now: TimeInterval) {
        guard association != value else { return }
        association = value; admittedAt = now; generationDoneAt = nil; naturalStopAt = nil
        resetQuiet()
    }
    private mutating func resetQuiet() {
        quietStart = nil; quietEnd = nil; previousQuietStart = nil; quietBlocks = 0
    }
    mutating func control(_ stage: JarvisReplayControlStage, fields: [String: String], at now: TimeInterval) {
        guard finishedAt == nil else { return }
        if stage == .localInterruptionRequested || stage == .connectionReset {
            association = nil; admittedAt = nil; generationDoneAt = nil; naturalStopAt = nil; resetQuiet(); return
        }
        guard let bound = association, fields["connection_id"] == bound.connectionID.uuidString.lowercased(),
              fields["response_id"] == bound.responseID else { return }
        if stage == .providerDataChannelHandled && fields["event_type"] == "response.done" { generationDoneAt = now }
        if (stage == .providerDataChannelHandled && fields["event_type"] == "output_audio_buffer.stopped")
            || (stage == .backendControlHandled && fields["event_type"] == "audio.webrtc.stopped") {
            naturalStopAt = now
        }
    }
    mutating func decoded(_ samples: [Int16], at now: TimeInterval, end: TimeInterval?,
                          association current: JarvisReplayResponseAssociation?) {
        guard finishedAt == nil, let bound = association, current == bound else { return }
        guard samples.count == 480, now.isFinite, let end, end.isFinite, end >= now,
              !samples.contains(where: { abs(Int($0)) >= 17 }) else { resetQuiet(); return }
        if let previousQuietStart, now < previousQuietStart || now - previousQuietStart > 0.03 { resetQuiet() }
        if quietStart == nil { quietStart = now }
        previousQuietStart = now; quietEnd = end; quietBlocks += 1
    }
    mutating func begin(scheduled: TimeInterval, observed: TimeInterval, seconds: TimeInterval, inputsComplete: Bool) {
        guard status == "not_started" else { return }
        scheduledMediaEnd = scheduled; observedMediaEnd = observed; deadline = scheduled + seconds
        requiredInputsComplete = inputsComplete
        status = seconds == 0 ? "disabled" : "capturing"
        if seconds == 0 { finishedAt = observed }
    }
    mutating func poll(at now: TimeInterval, lastInputEnd: TimeInterval?, evidenceValid: Bool) -> Bool {
        guard status == "capturing", let deadline else { return finishedAt != nil }
        if now >= deadline { status = "timeout"; finishedAt = now; return true }
        if evidenceValid, requiredInputsComplete, let admittedAt, admittedAt >= (lastInputEnd ?? 0),
           let done = generationDoneAt, done <= now, let stop = naturalStopAt, stop <= now,
           let start = quietStart, let end = quietEnd, end <= now,
           quietBlocks >= 50, end - start >= 0.5, end >= stop - 0.05 {
            status = "decoded_quiet_observed"; finishedAt = now; return true
        }
        return false
    }
    mutating func close(at time: TimeInterval?) {
        if status == "capturing" { status = "closed_early"; finishedAt = time }
    }
    var json: [String: Any] {
        func nullable<T>(_ value: T?) -> Any { value as Any? ?? NSNull() }
        return ["schema_version": 1, "status": status,
            "media_end_scheduled_uptime_seconds": nullable(scheduledMediaEnd),
            "media_end_observed_uptime_seconds": nullable(observedMediaEnd),
            "capture_deadline_uptime_seconds": nullable(deadline), "finished_uptime_seconds": nullable(finishedAt),
            "required_inputs_complete": requiredInputsComplete, "response_association": association.map(\.json) as Any? ?? NSNull(),
            "generation_done_uptime_seconds": nullable(generationDoneAt), "natural_buffer_stop_uptime_seconds": nullable(naturalStopAt),
            "authorized_quiet_start_uptime_seconds": nullable(quietStart), "authorized_quiet_callback_end_uptime_seconds": nullable(quietEnd),
            "authorized_quiet_blocks": quietBlocks, "required_quiet_seconds": 0.5, "maximum_callback_gap_seconds": 0.03,
            "decoded_drain_confirmed": false, "physical_playback_measured": false,
            "limitations": "Bounded capture only. Matching natural stop plus continuous authorized decoded quiet is an observation, not a network-buffer-empty, full-answer, acoustic or hearing receipt. Gated zeros are excluded. Semantic deadlines are unchanged."]
    }
}

struct JarvisReplayInputCompletion: Equatable { let ordinal: Int; let at: TimeInterval }
struct JarvisReplayResponseObservation: Equatable {
    let identity: JarvisReplayResponseIdentity
    let admittedAt: TimeInterval
    let firstActive: JarvisReplayDecodedSample
    let latestActive: JarvisReplayDecodedSample
    let callbackEnd: TimeInterval
}
struct JarvisReplayResponseSnapshot {
    var inputCompletion: JarvisReplayInputCompletion?
    var output: JarvisReplayResponseObservation?
    var providerInputFinalized = false
    var providerInputFailure: String?
}
struct JarvisReplayResponseSelection {
    let input: JarvisReplayInputCompletion
    let output: JarvisReplayResponseObservation
    let selectedAt: TimeInterval
}
struct JarvisReplayResponseTrigger {
    let afterInputOrdinal: Int
    let delay: TimeInterval
    let timeout: TimeInterval
    let maxPCMAge: TimeInterval
    var responseID: String? = nil
    var connectionID: UUID? = nil
    var requireTaskClaim = false

    func accepts(input: JarvisReplayInputCompletion, output: JarvisReplayResponseObservation,
                 at now: TimeInterval) -> Bool {
        now.isFinite && input.at.isFinite && input.ordinal == afterInputOrdinal && now >= input.at
            && now - input.at < timeout && output.callbackEnd.isFinite && now >= output.callbackEnd
            && now - output.callbackEnd <= maxPCMAge
            && output.firstActive.nominalUptime + delay <= now
            && (responseID == nil || responseID == output.identity.responseID)
            && (connectionID == nil || connectionID == output.identity.connectionID)
    }
}

/// Pure clock-driven selection. Scene advancement is independent of a pending reply.
struct JarvisReplayTimeline {
    struct Step {
        var frame: Int?
        var utterance: Int?
        var skippedFrames = 0
        var failure: JarvisReplayError?
        var finished = false
        var failedUtterance: Int?
        var responseSelection: JarvisReplayResponseSelection?
        var taskClaimSequence: Int?
    }
    let plan: JarvisReplayPlan
    private var frameIndex = 0
    private var utteranceIndex = 0
    private var waitBegan: TimeInterval = 0
    private var lastTime: TimeInterval = 0
    private var ended = false
    private var consumedResponses: Set<String> = []
    private var previousResponse: JarvisReplayResponseIdentity?

    init(plan: JarvisReplayPlan) { self.plan = plan }
    var pendingUtteranceOrdinal: Int? { !ended && utteranceIndex < plan.utterances.count ? utteranceIndex : nil }
    var pendingWaitStartElapsed: TimeInterval { waitBegan }
    var pendingResponseInputOrdinal: Int? {
        guard let ordinal = pendingUtteranceOrdinal else { return nil }
        switch plan.utterances[ordinal].trigger {
        case .responseOutputStart(let trigger): return trigger.afterInputOrdinal
        case .afterManualResponse(let trigger): return trigger.afterInputOrdinal
        case .afterTaskClaim(let previous, _), .afterInputComplete(let previous, _, _),
             .afterProviderInputFinalized(let previous, _): return previous
        default: return nil
        }
    }

    mutating func advance(elapsed: TimeInterval, output: JarvisReplayOutputStart?, inputIdle: Bool,
                          responseSnapshot: JarvisReplayResponseSnapshot? = nil, origin: TimeInterval = 0,
                          taskClaimSequence: Int? = nil, manualResponseReady: Bool = false) -> Step {
        guard !ended else { return Step(finished: true) }
        guard elapsed.isFinite, elapsed >= lastTime else {
            ended = true; return Step(failure: .invalidPlan)
        }
        lastTime = elapsed
        guard elapsed < plan.duration else {
            let hasPendingResponseTrigger = pendingResponseInputOrdinal != nil
            ended = true
            if hasPendingResponseTrigger {
                return Step(failure: responseSnapshot?.inputCompletion == nil ? .inputCompletionUnavailable : .expired,
                            failedUtterance: utteranceIndex)
            }
            return Step(finished: true)
        }
        var step = Step()
        while frameIndex < plan.frames.count, plan.frames[frameIndex].at <= elapsed {
            if step.frame != nil { step.skippedFrames += 1 }
            step.frame = frameIndex; frameIndex += 1
        }
        guard utteranceIndex < plan.utterances.count else { return step }
        let next = plan.utterances[utteranceIndex]
        var due = false
        switch next.trigger {
        case .atTime(let offset): due = elapsed >= offset
        case .outputStart(let cursor, let delay, let timeout):
            // Deadline wins, including a late scheduler wakeup with an older
            // matching event. No input may begin at/after this wait's deadline.
            if elapsed - waitBegan >= timeout {
                step.failure = .outputWaitExpired; ended = true; break
            }
            if let output, output.cursor > cursor, output.at >= waitBegan,
               output.at + delay <= elapsed { due = true }
        case .responseOutputStart(let trigger):
            let now = origin + elapsed
            guard origin.isFinite, now.isFinite else { step.failure = .invalidPlan; ended = true; break }
            // The strong deadline starts at the actual full-clip SDK receipt,
            // never at a nominal clip duration or a provider 'done' event.
            guard let input = responseSnapshot?.inputCompletion,
                  input.ordinal == trigger.afterInputOrdinal else { break }
            if now - input.at >= trigger.timeout {
                step.failure = .outputWaitExpired; ended = true; break
            }
            if trigger.requireTaskClaim && taskClaimSequence == nil { break }
            if let candidate = responseSnapshot?.output,
               trigger.accepts(input: input, output: candidate, at: now),
               !consumedResponses.contains(candidate.identity.responseKey),
               previousResponse.map({ $0.connectionID != candidate.identity.connectionID ||
                   candidate.identity.outputEpoch > $0.outputEpoch }) ?? true {
                due = true
                step.responseSelection = .init(input: input, output: candidate, selectedAt: now)
            }
        case .afterTaskClaim(let previous, let timeout), .afterInputComplete(let previous, _, let timeout):
            let now = origin + elapsed
            guard origin.isFinite, now.isFinite else { step.failure = .invalidPlan; ended = true; break }
            guard let input = responseSnapshot?.inputCompletion, input.ordinal == previous,
                  input.at.isFinite, now >= input.at else { break }
            if now - input.at >= timeout { step.failure = .outputWaitExpired; ended = true; break }
            if case .afterInputComplete(_, let delay, _) = next.trigger { due = now >= input.at + delay }
            else { due = taskClaimSequence != nil }
        case .afterProviderInputFinalized(let previous, let timeout):
            let now = origin + elapsed
            guard origin.isFinite, now.isFinite else { step.failure = .invalidPlan; ended = true; break }
            if responseSnapshot?.providerInputFailure != nil {
                step.failure = .providerInputFinalizationUnavailable; ended = true; break
            }
            guard let input = responseSnapshot?.inputCompletion, input.ordinal == previous else { break }
            if now - input.at >= timeout { step.failure = .outputWaitExpired; ended = true; break }
            due = now >= input.at && responseSnapshot?.providerInputFinalized == true
        case .afterManualResponse(let trigger):
            if elapsed >= trigger.deadline { step.failure = .outputWaitExpired; ended = true; break }
            due = elapsed >= trigger.at && manualResponseReady
        case .playbackDrained: step.failure = .unsupportedPlaybackDrain; ended = true
        }
        if due {
            if !inputIdle { step.failure = .overlappingInput; ended = true }
            else {
                if let selected = step.responseSelection {
                    consumedResponses.insert(selected.output.identity.responseKey)
                    previousResponse = selected.output.identity
                }
                if next.trigger.requiresTaskClaim { step.taskClaimSequence = taskClaimSequence }
                step.utterance = utteranceIndex; utteranceIndex += 1; waitBegan = elapsed
            }
        }
        if step.failure != nil { step.failedUtterance = utteranceIndex }
        return step
    }
}

/// PCM sample boundaries at the custom native ADM callback, not speaker/hearing times.
struct JarvisReplayAudioEvidence {
    let provenance = "frontend_replay"
    let boundary = "custom_native_adm_pcm_callback"
    let physicalPlaybackMeasured = false
    let usefulAnswerOnset: TimeInterval? = nil
    var inputBlocks = 0
    var decodedBlocks = 0
    var gatedDecodedBlocks = 0
    var inputFirstNonSilent: [TimeInterval] = []
    var inputLastNonSilent: [TimeInterval] = []
    var outputStarts: [JarvisReplayOutputStart] = []
    var decodedPCM16 = Data()
    var truncated = false
    var callbackFailure: Int32?
    let mediaEpochID = UUID()
    var clockAnchor: JarvisReplayClockAnchor?
    var utterances: [JarvisReplayUtteranceEvidence] = []
    var decodedSampleMap: [JarvisReplayDecodedBlock] = []
    var outputGates: [JarvisReplayOutputGateEvidence] = []
    var inputMappingTruncated = false
    var decodedMappingTruncated = false
    var outputGateHistoryTruncated = false
    var controlEvents: [JarvisReplayControlEvent] = []
    var controlEventsTruncated = false
    var controlEventTimingInvalid = false
    var localInterruptions: [JarvisReplayLocalInterruption] = []
    var responseEvidenceTruncated = false
    var responseEvidenceInvalid = false
    var uncorrelatedDecodedBlocks = 0
    var taskBarrierFailureCode: String?
    var taskBarrierFailureDiagnostic: JarvisReplayTaskFailureDiagnostic?
    var invocationID: UUID?
    var sourceClock: [String: Any]?
    var outputDrain = JarvisReplayOutputDrain()
    var responseUncorrelatedDecodedBlocks = 0
    var manualInput: [String: Any]?
}

/// Same buffer/gate logic used by the real device and deterministic offline tests.
struct JarvisReplayPCMState {
    static let sampleRate = 48_000
    static let blockFrames = 480
    static let maximumInputMapEntries = 6_020
    static let maximumOutputGateEntries = 512
    static let maximumControlEvents = 512
    static let maximumResponseIdentities = 512
    var inputEnabled = false
    var outputEnabled = false
    private(set) var closed = false
    private(set) var evidence = JarvisReplayAudioEvidence()
    private var input: [Int16] = []
    private var inputOffset = 0
    private var inputEvidenceIndex: Int?
    private var activeUtterance: Int?
    private var registeredPCM: [Data] = []
    private var sceneDuration: TimeInterval?
    private var enqueueCount = 0
    private var inputMapCount = 0
    private var outputArmed = false
    private var outputEpoch = 0
    private var lastAdmittedActiveSample: JarvisReplayDecodedSample?
    private var responseIdentity: JarvisReplayResponseIdentity?
    private var responseAssociation: JarvisReplayResponseAssociation?
    private var associationAdmittedAt: TimeInterval?
    private var responseAdmittedAt: TimeInterval?
    private var responseFirstActive: JarvisReplayDecodedSample?
    private var responseLatestActive: JarvisReplayDecodedSample?
    private var responseTail: JarvisReplayDecodedSample?
    private var responseCallbackEnd: TimeInterval?
    private var responseConnection: UUID?
    private var seenResponses: Set<String> = []
    private var consumedResponses: Set<String> = []
    private var pendingLocalInterruption: JarvisReplayLocalInterruption?
    private let maxOutputBytes: Int
    private let maxOutputMapEntries: Int
    private struct PendingInput {
        let ordinal: Int
        let offset: Int
        let count: Int
        let firstActive: Int?
        let lastActive: Int?
        var callbackStart: TimeInterval?
    }
    private var pendingInput: PendingInput?
    var inputIdle: Bool { inputOffset >= input.count && pendingInput == nil }
    var latestOutput: JarvisReplayOutputStart? { evidence.outputStarts.last }
    var currentResponseIdentity: JarvisReplayResponseIdentity? { !closed && outputEnabled ? responseIdentity : nil }
    var currentResponseAssociation: JarvisReplayResponseAssociation? { !closed && outputEnabled ? responseAssociation : nil }

    func responseSnapshot(afterInputOrdinal ordinal: Int) -> JarvisReplayResponseSnapshot {
        var snapshot = JarvisReplayResponseSnapshot()
        guard !closed, !evidence.inputMappingTruncated else { return snapshot }
        if evidence.utterances.indices.contains(ordinal) {
            let input = evidence.utterances[ordinal]
            if input.status == "sdk_input_complete", input.acceptedSamples == input.sampleCount,
               let completed = input.lastCallbackEnd, completed.isFinite {
                snapshot.inputCompletion = .init(ordinal: ordinal, at: completed)
            }
        }
        if evidence.utterances.indices.contains(ordinal + 1),
           let reserved = evidence.utterances[ordinal + 1].providerInputFinalization {
            let proof = reserved.evaluated(evidence)
            snapshot.providerInputFinalized = proof.ready
            snapshot.providerInputFailure = proof.failureCode
        }
        guard outputEnabled, !evidence.truncated, !evidence.decodedMappingTruncated,
              !evidence.outputGateHistoryTruncated, !evidence.responseEvidenceTruncated,
              !evidence.responseEvidenceInvalid, evidence.callbackFailure == nil,
              let identity = responseIdentity, !consumedResponses.contains(identity.responseKey),
              let admitted = responseAdmittedAt, let first = responseFirstActive,
              let last = responseLatestActive, let end = responseCallbackEnd else { return snapshot }
        snapshot.output = .init(identity: identity, admittedAt: admitted, firstActive: first,
                                latestActive: last, callbackEnd: end)
        return snapshot
    }

    init(maxDuration: TimeInterval, startupAllowance: TimeInterval = 0) {
        guard JarvisReplayPlan.isValidSceneDuration(maxDuration), startupAllowance.isFinite,
              startupAllowance >= 0, startupAllowance <= JarvisReplayPlan.startupAllowance else {
            maxOutputBytes = 0; maxOutputMapEntries = 0; closed = true
            return
        }
        let recordingDuration = maxDuration + startupAllowance
        // Allocate only complete PCM16 samples; reject nonfinite values before Int conversion.
        // Startup PCM keeps its original offsets/timestamps and consumes this
        // bounded reserve. Never rebase the scene or reset evidence at readiness.
        maxOutputBytes = Int((recordingDuration * Double(Self.sampleRate)).rounded(.down)) * 2
        maxOutputMapEntries = max(1, Int((recordingDuration * 100).rounded(.up)))
    }

    mutating func register(_ plan: JarvisReplayPlan) throws {
        guard !closed, evidence.utterances.isEmpty, enqueueCount == 0, inputIdle else { throw JarvisReplayError.retired }
        evidence.utterances = plan.utterances.indices.map { JarvisReplayUtteranceEvidence(plan: plan, ordinal: $0) }
        registeredPCM = plan.utterances.map(\.pcm16)
        sceneDuration = plan.duration
        evidence.invocationID = plan.invocationID
        if let clock = plan.sourceClock {
            evidence.sourceClock = ["version": 1, "phase_start_seconds": clock.phaseStart, "status": "pending",
                "anchor_sha256": clock.anchorData.map(JarvisReplayFrameLineage.hash) as Any? ?? NSNull()]
        }
    }
    mutating func recordAnchor(_ anchor: JarvisReplayClockAnchor) throws {
        guard !closed, evidence.clockAnchor == nil, anchor.monotonicBefore.isFinite,
              anchor.monotonicAfter.isFinite, anchor.monotonicAfter >= anchor.monotonicBefore,
              anchor.utc.timeIntervalSince1970.isFinite else { throw JarvisReplayError.invalidPlan }
        evidence.clockAnchor = anchor
    }
    mutating func recordControlEvent(_ stage: JarvisReplayControlStage, fields: [String: Any] = [:],
                                     receivedAt: TimeInterval? = nil, at time: TimeInterval) {
        guard !closed else { return }
        guard time.isFinite, time >= 0,
              receivedAt.map({ $0.isFinite && $0 >= 0 && $0 <= time }) ?? true else {
            evidence.controlEventTimingInvalid = true
            return
        }
        guard evidence.controlEvents.count < Self.maximumControlEvents else {
            evidence.controlEventsTruncated = true
            return
        }
        var safe = RealtimeDiagnostics.safeFields(fields).compactMapValues { $0 as? String }
        if let value = fields["control_connection_id"] as? String, let id = UUID(uuidString: value) {
            safe["control_connection_id"] = id.uuidString.lowercased()
        }
        evidence.outputDrain.control(stage, fields: safe, at: time)
        // Preserve only the actually bound item before authenticated retirement.
        // Never recover an item from response-only history or a later/latest gate.
        let stoppedIdentity: JarvisReplayResponseIdentity?
        if stage == .backendControlHandled, safe["event_type"] == "audio.webrtc.stopped",
           let identity = currentResponseIdentity,
           safe["connection_id"] == identity.connectionID.uuidString.lowercased(),
           safe["response_id"] == identity.responseID { stoppedIdentity = identity }
        else { stoppedIdentity = nil }
        evidence.controlEvents.append(JarvisReplayControlEvent(stage: stage, handledAt: time,
            receivedAt: receivedAt, fields: safe, generations: JarvisReplayControlGenerations.safe(fields),
            boundResponseIdentity: stoppedIdentity))
    }
    mutating func recordManualInput(_ value: [String: Any]) {
        guard !closed else { return }
        evidence.manualInput = value
    }
    mutating func recordTaskBarrierFailure(_ code: String, diagnostic: JarvisReplayTaskFailureDiagnostic? = nil) {
        guard !closed, evidence.taskBarrierFailureCode == nil else { return }
        evidence.taskBarrierFailureCode = code
        if code == "invalidControl", let diagnostic, diagnostic.isValid {
            evidence.taskBarrierFailureDiagnostic = diagnostic
        }
    }
    mutating func recordSourceClock(_ data: Data?, failure: JarvisReplayError? = nil) {
        guard evidence.sourceClock != nil else { return }
        evidence.sourceClock?["status"] = failure == nil ? "bound" : "failed"
        evidence.sourceClock?["failure_code"] = failure.map { String(describing: $0) } as Any? ?? NSNull()
        if let data { evidence.sourceClock?["anchor_sha256"] = JarvisReplayFrameLineage.hash(data) }
    }
    mutating func beginOutputDrain(scheduled: TimeInterval, observed: TimeInterval, seconds: TimeInterval) {
        guard !closed, scheduled.isFinite, observed.isFinite, observed >= scheduled,
              seconds.isFinite, seconds >= 0, seconds <= JarvisReplayPlan.maximumPostMediaCapture else { return }
        let complete = inputIdle && evidence.utterances.allSatisfy { $0.status == "sdk_input_complete" && $0.acceptedSamples == $0.sampleCount }
        evidence.outputDrain.begin(scheduled: scheduled, observed: observed, seconds: seconds, inputsComplete: complete)
        gateInput(false, at: observed) // No additional stimulus, including late queued input.
    }
    mutating func pollOutputDrain(at now: TimeInterval) -> Bool {
        let valid = !closed && now.isFinite && !evidence.truncated && !evidence.decodedMappingTruncated
            && !evidence.controlEventsTruncated && !evidence.controlEventTimingInvalid
            && !evidence.responseEvidenceInvalid && !evidence.responseEvidenceTruncated
            && !evidence.inputMappingTruncated && !evidence.outputGateHistoryTruncated && evidence.callbackFailure == nil
        return evidence.outputDrain.poll(at: now, lastInputEnd: evidence.utterances.compactMap(\.lastCallbackEnd).max(), evidenceValid: valid)
    }
    mutating func armUtterance(_ ordinal: Int, at time: TimeInterval) {
        guard evidence.utterances.indices.contains(ordinal), evidence.utterances[ordinal].triggerArmedAt == nil else { return }
        evidence.utterances[ordinal].triggerArmedAt = time
    }
    mutating func failUtterance(_ ordinal: Int, code: String, at time: TimeInterval) {
        guard evidence.utterances.indices.contains(ordinal),
              evidence.utterances[ordinal].status != "sdk_input_complete", evidence.utterances[ordinal].status != "failed" else { return }
        if let reserved = evidence.utterances[ordinal].providerInputFinalization {
            var proof = reserved.evaluated(evidence)
            proof.failureCode = proof.failureCode ?? code
            evidence.utterances[ordinal].providerInputFinalization = proof
        }
        evidence.utterances[ordinal].status = "failed"
        evidence.utterances[ordinal].failureCode = code
        evidence.utterances[ordinal].responseTrigger?.failureCode = code
        if evidence.utterances[ordinal].triggeredAt == nil { evidence.utterances[ordinal].triggeredAt = time }
    }

    mutating func enqueue(_ data: Data, utteranceOrdinal: Int? = nil, requestedAt: TimeInterval? = nil,
                          triggeredAt: TimeInterval? = nil, responseSelection: JarvisReplayResponseSelection? = nil,
                          taskClaimSequence: Int? = nil) throws {
        if let ordinal = utteranceOrdinal {
            guard evidence.utterances.indices.contains(ordinal), evidence.utterances[ordinal].status == "scheduled" else {
                throw JarvisReplayError.invalidPlan
            }
            evidence.utterances[ordinal].triggeredAt = triggeredAt ?? requestedAt
            evidence.utterances[ordinal].enqueueRequestedAt = requestedAt
            if let responseSelection {
                evidence.utterances[ordinal].responseTrigger = .init(selection: responseSelection)
            }
        }
        do {
            guard !closed else { throw JarvisReplayError.retired }
            guard inputEnabled else { throw JarvisReplayError.inputNotReady }
            guard inputIdle else { throw JarvisReplayError.overlappingInput }
            guard !data.isEmpty, data.count.isMultiple(of: 2), data.count <= 60 * 96_000, enqueueCount < 20 else {
                throw JarvisReplayError.invalidPlan
            }
            if let ordinal = utteranceOrdinal, data != registeredPCM[ordinal] { throw JarvisReplayError.invalidPlan }
            for target in evidence.utterances.indices {
                if var proof = evidence.utterances[target].providerInputFinalization,
                   proof.reservedAt != nil, proof.enqueueRevalidatedAt == nil,
                   utteranceOrdinal != target, utteranceOrdinal != proof.previousOrdinal {
                    proof.failureCode = "provider_input_competing_sdk_input"
                    evidence.utterances[target].providerInputFinalization = proof
                }
            }
            if let ordinal = utteranceOrdinal {
                let trigger = evidence.utterances[ordinal].trigger
                if trigger.requiresTaskClaim {
                    guard let sequence = taskClaimSequence, sequence > 0 else { throw JarvisReplayError.taskClaimUnavailable }
                    evidence.utterances[ordinal].taskClaimSequenceAtEnqueue = sequence
                } else if taskClaimSequence != nil { throw JarvisReplayError.invalidPlan }
                switch trigger {
                case .afterTaskClaim(let previous, let timeout), .afterInputComplete(let previous, _, let timeout):
                    guard let now = requestedAt, now.isFinite,
                          let completed = responseSnapshot(afterInputOrdinal: previous).inputCompletion,
                          now >= completed.at, now - completed.at < timeout else { throw JarvisReplayError.inputCompletionUnavailable }
                    if case .afterInputComplete(_, let delay, _) = trigger, now < completed.at + delay {
                        throw JarvisReplayError.inputCompletionUnavailable
                    }
                case .afterProviderInputFinalized:
                    guard let reserved = evidence.utterances[ordinal].providerInputFinalization,
                          let now = requestedAt, now.isFinite else {
                        throw JarvisReplayError.providerInputFinalizationUnavailable
                    }
                    var proof = reserved.evaluated(evidence)
                    guard proof.ready, let completed = proof.inputCompleteAt,
                          let deadline = proof.deadline, now >= completed, now < deadline else {
                        evidence.utterances[ordinal].providerInputFinalization = proof
                        throw JarvisReplayError.providerInputFinalizationUnavailable
                    }
                    proof.enqueueRevalidatedAt = now
                    evidence.utterances[ordinal].providerInputFinalization = proof
                default: break
                }
            }
            if let ordinal = utteranceOrdinal,
               case .responseOutputStart(let trigger) = evidence.utterances[ordinal].trigger {
                let current = responseSnapshot(afterInputOrdinal: trigger.afterInputOrdinal)
                guard let selected = responseSelection, let now = requestedAt,
                      selected.selectedAt.isFinite, selected.selectedAt <= now,
                      let input = current.inputCompletion, input == selected.input,
                      let output = current.output, output.identity == selected.output.identity,
                      output.firstActive == selected.output.firstActive,
                      trigger.accepts(input: input, output: output, at: now) else {
                    throw JarvisReplayError.responseCandidateChanged
                }
                var receipt = JarvisReplayResponseTriggerEvidence(selection:
                    .init(input: input, output: output, selectedAt: selected.selectedAt))
                receipt.enqueueRevalidatedAt = now
                evidence.utterances[ordinal].responseTrigger = receipt
                consumedResponses.insert(output.identity.responseKey)
            }
            if let ordinal = utteranceOrdinal, let now = requestedAt,
               evidence.utterances.indices.contains(ordinal + 1),
               case .afterProviderInputFinalized(let previous, let timeout) = evidence.utterances[ordinal + 1].trigger,
               previous == ordinal {
                let deadline = evidence.clockAnchor.flatMap { anchor in
                    sceneDuration.map { anchor.referenceUptime + $0 }
                }
                let proof = JarvisReplayProviderInputFinalization.reserve(previous: ordinal, timeout: timeout,
                    evidence: evidence, at: now, sceneDeadline: deadline)
                evidence.utterances[ordinal + 1].providerInputFinalization = proof
            }
            input = stride(from: 0, to: data.count, by: 2).map {
                Int16(bitPattern: UInt16(data[$0]) | UInt16(data[$0 + 1]) << 8)
            }
            inputOffset = 0; inputEvidenceIndex = nil; outputArmed = true
            activeUtterance = utteranceOrdinal
            enqueueCount += 1
            if let ordinal = utteranceOrdinal { evidence.utterances[ordinal].status = "enqueued" }
        } catch {
            if let ordinal = utteranceOrdinal {
                failUtterance(ordinal, code: "enqueue_\(String(describing: error))", at: requestedAt ?? triggeredAt ?? 0)
            }
            throw error
        }
    }

    mutating func renderInput(at now: TimeInterval) -> [Int16] {
        var block = [Int16](repeating: 0, count: Self.blockFrames)
        guard !closed else { return block }
        evidence.inputBlocks += 1
        guard inputEnabled, inputOffset < input.count, pendingInput == nil else { return block }
        let offset = inputOffset
        let count = min(Self.blockFrames, input.count - inputOffset)
        for i in 0..<count { block[i] = input[inputOffset + i] }
        inputOffset += count
        for i in 0..<count where block[i] != 0 {
            let at = now + Double(i) / Double(Self.sampleRate)
            if inputEvidenceIndex == nil {
                inputEvidenceIndex = evidence.inputFirstNonSilent.count
                evidence.inputFirstNonSilent.append(at); evidence.inputLastNonSilent.append(at)
            }
            if let index = inputEvidenceIndex { evidence.inputLastNonSilent[index] = at }
        }
        if let ordinal = activeUtterance {
            let threshold = evidence.utterances[ordinal].activityThreshold
            pendingInput = PendingInput(ordinal: ordinal, offset: offset, count: count,
                firstActive: (0..<count).first { abs(Int(block[$0])) >= threshold },
                lastActive: (0..<count).last { abs(Int(block[$0])) >= threshold })
        }
        if inputOffset >= input.count { input.removeAll(keepingCapacity: false); inputOffset = 0 }
        return block
    }

    mutating func inputCallbackStarted(at time: TimeInterval) {
        guard var pending = pendingInput, pending.callbackStart == nil else { return }
        pending.callbackStart = time; pendingInput = pending
        let index = pending.ordinal
        if evidence.utterances[index].firstCallbackStart == nil { evidence.utterances[index].firstCallbackStart = time }
        evidence.utterances[index].attemptedSamples += pending.count
        if evidence.utterances[index].status != "cancelled" { evidence.utterances[index].status = "sdk_input_in_progress" }
    }
    mutating func inputCallbackCompleted(status: Int32, at time: TimeInterval) {
        if status != 0 { evidence.callbackFailure = status }
        guard let pending = pendingInput, let start = pending.callbackStart else { return }
        // Record an already-entered callback even if close requested retirement
        // while it ran. Successful fencing waits for this completion.
        pendingInput = nil
        let index = pending.ordinal
        evidence.utterances[index].lastCallbackEnd = time
        if inputMapCount < Self.maximumInputMapEntries {
            evidence.utterances[index].callbacks.append(JarvisReplayInputBlock(sourceSampleOffset: pending.offset,
                clipSampleCount: pending.count, transportFrameCount: Self.blockFrames,
                callbackStart: start, callbackEnd: time, status: status))
            inputMapCount += 1
        } else { evidence.inputMappingTruncated = true }
        if status == 0 {
            evidence.utterances[index].acceptedSamples += pending.count
            if let first = pending.firstActive, evidence.utterances[index].firstSuccessfulActivity == nil {
                evidence.utterances[index].firstSuccessfulActivity = start + Double(first) / Double(Self.sampleRate)
            }
            if let last = pending.lastActive {
                evidence.utterances[index].lastSuccessfulActivity = start + Double(last) / Double(Self.sampleRate)
            }
            if evidence.utterances[index].acceptedSamples == evidence.utterances[index].sampleCount {
                evidence.utterances[index].status = "sdk_input_complete"
                evidence.utterances[index].failureCode = nil
            }
        } else {
            evidence.utterances[index].status = "failed"
            evidence.utterances[index].failureCode = "sdk_input_callback_failed"
            if activeUtterance == index { input.removeAll(); inputOffset = 0 }
        }
        if inputOffset >= input.count { activeUtterance = nil }
    }

    mutating func observeDecoded(_ samples: [Int16], at now: TimeInterval, callbackEnd: TimeInterval? = nil,
                                responseIdentity callbackIdentity: JarvisReplayResponseIdentity? = nil,
                                responseAssociation callbackAssociation: JarvisReplayResponseAssociation? = nil) {
        guard !closed else { return }
        evidence.decodedBlocks += 1
        guard outputEnabled else { evidence.gatedDecodedBlocks += 1; return }
        guard now.isFinite, now >= 0, callbackEnd.map({ $0.isFinite && $0 >= now }) ?? true else {
            evidence.responseEvidenceInvalid = true; return
        }
        if outputArmed, let first = samples.firstIndex(where: { abs(Int($0)) > 16 }) {
            evidence.outputStarts.append(JarvisReplayOutputStart(cursor: evidence.outputStarts.count + 1,
                at: now + Double(first) / Double(Self.sampleRate)))
            outputArmed = false
        }
        let available = max(0, (maxOutputBytes - evidence.decodedPCM16.count) / 2)
        let retained = min(available, samples.count)
        if retained < samples.count { evidence.truncated = true }
        guard retained > 0 else { return }
        guard evidence.decodedSampleMap.count < maxOutputMapEntries else {
            evidence.decodedMappingTruncated = true; evidence.truncated = true; return
        }
        let offset = evidence.decodedPCM16.count / 2
        let correlated = callbackIdentity != nil && callbackIdentity == responseIdentity
            && responseAdmittedAt.map({ now >= $0 }) == true && callbackEnd != nil
        let entryAssociation = callbackAssociation ?? callbackIdentity?.association
        let responseCorrelated = entryAssociation != nil && entryAssociation == responseAssociation
            && associationAdmittedAt.map({ now >= $0 }) == true && callbackEnd != nil
        if !correlated { evidence.uncorrelatedDecodedBlocks += 1 }
        if !responseCorrelated { evidence.responseUncorrelatedDecodedBlocks += 1 }
        evidence.outputDrain.decoded(samples, at: now, end: callbackEnd,
                                     association: responseCorrelated ? entryAssociation : nil)
        responseLatestActive = nil; responseCallbackEnd = nil
        evidence.decodedSampleMap.append(JarvisReplayDecodedBlock(sampleOffset: offset, sampleCount: retained,
            callbackStart: now, callbackEnd: callbackEnd, outputEpoch: outputEpoch,
            responseIdentity: correlated ? callbackIdentity : nil,
            responseAssociation: responseCorrelated ? entryAssociation : nil))
        if let last = (0..<retained).last(where: { abs(Int(samples[$0])) > 16 }) {
            lastAdmittedActiveSample = JarvisReplayDecodedSample(sampleOffset: offset + last,
                nominalUptime: now + Double(last) / Double(Self.sampleRate), outputEpoch: outputEpoch)
            if correlated {
                responseLatestActive = lastAdmittedActiveSample
                responseTail = lastAdmittedActiveSample
                responseCallbackEnd = callbackEnd
                if responseFirstActive == nil, let first = (0..<retained).first(where: { abs(Int(samples[$0])) > 16 }) {
                    responseFirstActive = .init(sampleOffset: offset + first,
                        nominalUptime: now + Double(first) / Double(Self.sampleRate), outputEpoch: outputEpoch)
                }
            }
        }
        for sample in samples.prefix(retained) {
            let value = UInt16(bitPattern: sample)
            evidence.decodedPCM16.append(UInt8(truncatingIfNeeded: value))
            evidence.decodedPCM16.append(UInt8(truncatingIfNeeded: value >> 8))
        }
    }

    mutating func callbackFailed(_ status: Int32) { evidence.callbackFailure = status }
    mutating func gateInput(_ enabled: Bool, at time: TimeInterval? = nil) {
        inputEnabled = enabled && !closed
        if !inputEnabled {
            if let ordinal = activeUtterance, !["sdk_input_complete", "failed"].contains(evidence.utterances[ordinal].status) {
                evidence.utterances[ordinal].status = "cancelled"
                evidence.utterances[ordinal].failureCode = "input_gate_closed"
                evidence.utterances[ordinal].stopRequestedAt = time
            }
            input.removeAll(); inputOffset = 0; activeUtterance = nil
        }
    }
    mutating func recordLocalInterruption(_ cause: JarvisReplayLocalInterruptionCause, at time: TimeInterval) {
        recordControlEvent(.localInterruptionRequested, fields: ["reason": cause.rawValue], at: time)
        guard !closed, let identity = currentResponseIdentity else { return }
        guard time.isFinite, time >= 0 else { evidence.responseEvidenceInvalid = true; return }
        guard evidence.localInterruptions.count < Self.maximumControlEvents else {
            evidence.responseEvidenceTruncated = true; return
        }
        let event = JarvisReplayLocalInterruption(
            eventID: "\(evidence.mediaEpochID.uuidString.lowercased())/local-interruption/\(evidence.localInterruptions.count)",
            identity: identity, cause: cause, requestedAt: time)
        evidence.localInterruptions.append(event); pendingLocalInterruption = event
    }

    private mutating func clearResponseBinding() {
        responseIdentity = nil; responseAdmittedAt = nil; responseFirstActive = nil
        responseLatestActive = nil; responseTail = nil; responseCallbackEnd = nil
    }

    mutating func gateOutput(_ enabled: Bool, at time: TimeInterval? = nil,
                              responseID: String? = nil, itemID: String? = nil, connectionID: UUID? = nil,
                              diagnostics: JarvisReplayPlaybackGateDiagnostics? = nil) {
        guard !closed else { return }
        let priorIdentity = responseIdentity, priorTail = responseTail
        let priorAssociation = responseAssociation
        let changed = enabled != outputEnabled
        if enabled && changed { outputEpoch += 1; lastAdmittedActiveSample = nil }
        outputEnabled = enabled
        if enabled, let responseID, let itemID, let connectionID, let time,
           time.isFinite, time >= 0, JarvisReplayResponseIdentity.safeToken(responseID, field: "response_id"),
           JarvisReplayResponseIdentity.safeToken(itemID, field: "item_id") {
            let key = connectionID.uuidString.lowercased() + ":" + responseID
            if let connection = responseConnection, connection != connectionID {
                clearResponseBinding(); evidence.responseEvidenceInvalid = true
            } else if !changed, responseAssociation?.responseKey == key {
                // Same response, potentially buffered earlier item audio. Keep
                // response/epoch only; once ambiguous, never relabel an item.
                if responseIdentity?.itemID != itemID { clearResponseBinding() }
            } else {
                clearResponseBinding()
                responseAssociation = nil; associationAdmittedAt = nil
                if seenResponses.contains(key) {
                    // Retired/consumed responses cannot be relabelled as fresh output.
                } else if seenResponses.count >= Self.maximumResponseIdentities {
                    evidence.responseEvidenceTruncated = true
                } else {
                    if !changed { outputEpoch += 1; lastAdmittedActiveSample = nil }
                    seenResponses.insert(key); responseConnection = connectionID
                    responseIdentity = .init(connectionID: connectionID, responseID: responseID,
                                             itemID: itemID, outputEpoch: outputEpoch)
                    responseAdmittedAt = time
                    responseAssociation = responseIdentity?.association; associationAdmittedAt = time
                    if let responseAssociation { evidence.outputDrain.bind(responseAssociation, at: time) }
                }
            }
            if evidence.responseEvidenceInvalid { responseAssociation = nil; associationAdmittedAt = nil }
        } else { clearResponseBinding(); responseAssociation = nil; associationAdmittedAt = nil }
        if evidence.outputGates.count < Self.maximumOutputGateEntries {
            let event = "\(evidence.mediaEpochID.uuidString.lowercased())/output-gate/\(evidence.outputGates.count)"
            evidence.outputGates.append(JarvisReplayOutputGateEvidence(eventID: event, enabled: enabled,
                changed: changed, requestedAt: time, outputEpoch: outputEpoch,
                lastAdmittedActiveSample: priorIdentity != nil && !enabled ? priorTail : lastAdmittedActiveSample,
                responseIdentity: enabled ? responseIdentity : priorIdentity,
                responseAssociation: enabled ? responseAssociation : priorAssociation,
                localInterruptionEventID: !enabled && changed && pendingLocalInterruption?.identity == priorIdentity
                    ? pendingLocalInterruption?.eventID : nil,
                playbackDiagnostics: diagnostics))
        } else { evidence.outputGateHistoryTruncated = true }
        if !enabled || priorIdentity != responseIdentity { pendingLocalInterruption = nil }
    }
    mutating func close(at time: TimeInterval? = nil) {
        guard !closed else { return }
        evidence.outputDrain.close(at: time)
        gateInput(false, at: time); gateOutput(false, at: time); closed = true
        for index in evidence.utterances.indices where evidence.utterances[index].status == "scheduled" {
            evidence.utterances[index].status = "not_started"
            evidence.utterances[index].failureCode = "run_closed_before_injection"
            evidence.utterances[index].stopRequestedAt = time
        }
        registeredPCM.removeAll()
    }
}
#endif
