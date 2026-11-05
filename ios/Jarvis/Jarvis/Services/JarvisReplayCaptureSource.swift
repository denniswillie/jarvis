#if DEBUG
import Foundation
import UIKit

/// Real adapter and sampler, with only the origin of images replaced.
/// `.glasses` is the emulated wire format; authenticated replay scope MUST mark
/// it as evaluation data server-side. It is never evidence of physical glasses.
@MainActor
final class JarvisReplayCaptureSource: JarvisVisualFrameSource {
    var onFrame: ((CapturedVisualFrame) -> Void)?
    var onFailure: ((Error) -> Void)?
    var onFinished: (() -> Void)?
    var onMediaEnded: (() -> Void)?
    var onManualBegin: (() throws -> JarvisReplayManualSessionIdentity)?
    var onManualRelease: ((JarvisReplayManualSessionIdentity) throws -> Void)?
    var onManualScopeIsCurrent: ((JarvisReplayManualSessionIdentity) -> Bool)?
    private(set) var isRunning = false
    private(set) var coalescedFrames = 0
    private(set) var adapterDroppedFrames = 0
    private let plan: JarvisReplayPlan
    private let audio: JarvisReplayAudioDevice
    private let adapter: GlassesFrameAdapter = {
        var value = GlassesFrameAdapter()
        value.recordsReplayEncoding = true
        return value
    }()
    private let launchManifestSHA256: String?
    private let sampler = FrameSampler(targetFPS: 1)
    private var timeline: JarvisReplayTimeline
    private var task: Task<Void, Never>?
    private var retired = false
    private var mediaEnded = false
    private let taskBarrier: JarvisReplayTaskBarrier?
    private let manualInput: JarvisReplayManualInputController?
    private let memoryGate: JarvisReplayMemoryGate?

    init(plan: JarvisReplayPlan, audio: JarvisReplayAudioDevice,
         taskConfiguration: JarvisReplayTaskConfiguration? = nil,
          launchManifestSHA256: String? = nil, manualInput: Bool = false,
         memoryConfiguration: JarvisReplayMemoryConfiguration? = nil) throws {
        try plan.validateAudioCapture(duration: audio.sceneDuration, retired: audio.isRetired,
                                      capacity: audio.captureDuration, recordingCapacity: audio.recordingCapacityDuration)
        for frame in plan.frames {
            guard let image = UIImage(data: frame.image), image.size.width > 0, image.size.height > 0,
                  image.size.width * image.size.height <= 16_000_000 else { throw JarvisReplayError.invalidImage }
        }
        self.plan = plan; self.audio = audio; timeline = JarvisReplayTimeline(plan: plan)
        self.launchManifestSHA256 = launchManifestSHA256
        try audio.registerPlan(plan)
        if manualInput {
            _ = try JarvisReplayLaunchPlan.validateManualInput("manual", route: .realtime,
                invocationID: plan.invocationID, taskMode: taskConfiguration != nil, utterances: plan.utterances,
                memoryReadiness: memoryConfiguration != nil)
        }
        if let memoryConfiguration {
            guard manualInput, let invocation = plan.invocationID else { throw JarvisReplayError.invalidPlan }
            memoryGate = try JarvisReplayMemoryGate(configuration: memoryConfiguration, invocationID: invocation)
        } else { memoryGate = nil }
        self.manualInput = manualInput
            ? JarvisReplayManualInputController(runID: plan.runID, invocationID: plan.invocationID) : nil
        if let value = self.manualInput { audio.recordManualInput(value.json) }
        taskBarrier = taskConfiguration.map { JarvisReplayTaskBarrier(configuration: $0, plan: plan, evidence: { audio.evidence }) }
        taskBarrier?.onFailure = { [weak self] error in
            guard let self, !self.retired else { return }
            self.audio.recordTaskBarrierFailure((error as? JarvisReplayTaskError).map { String(describing: $0) } ?? "invalid_control",
                                                diagnostic: self.taskBarrier?.failureDiagnostic)
            self.stop(); self.onFailure?(error)
        }
    }

    func bindTaskSession(_ sessionID: String) async throws {
        try await taskBarrier?.bind(sessionID: sessionID)
    }

    func manualReady(segmentID: String, connectionID: UUID, fields: [String: Any]) {
        guard !retired, let manualInput else { return }
        _ = manualInput.ready(segmentID: segmentID, connectionID: connectionID, generations: fields,
                              at: ProcessInfo.processInfo.systemUptime)
        audio.recordManualInput(manualInput.json)
    }

    func manualWrite(_ value: JarvisReplayManualWireObservation) {
        guard !retired, let manualInput else { return }
        manualInput.observe(value)
        audio.recordManualInput(manualInput.json)
    }

    func start() async throws {
        guard !retired, !isRunning else { throw JarvisReplayError.retired }
        isRunning = true
    }

    /// Start the single scene/audio clock only after the normal visual socket
    /// has reported visual.session.created. Reconnects never reset this clock.
    func beginPlayback() {
        guard isRunning, !retired, task == nil else { return }
        let before = ProcessInfo.processInfo.systemUptime
        let measuredUTC = Date()
        let after = ProcessInfo.processInfo.systemUptime
        var anchor = JarvisReplayClockAnchor(monotonicBefore: before, utc: measuredUTC, monotonicAfter: after)
        do {
            if let clock = plan.sourceClock {
                let resolved = try clock.resolve(plan: plan, measured: anchor,
                    bootID: JarvisReplaySourceAnchor.systemBootID(), now: after)
                anchor = resolved.phase
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                let data = try clock.anchorData ?? encoder.encode(resolved.source)
                let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                           appropriateFor: nil, create: true)
                let root = plan.captureRoot(documents: documents)
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                       attributes: [.posixPermissions: 0o700])
                try data.write(to: root.appendingPathComponent("source-clock.json"), options: .withoutOverwriting)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: root.appendingPathComponent("source-clock.json").path)
                audio.recordSourceClock(data)
            }
            try audio.recordClockAnchor(anchor)
        } catch {
            audio.recordSourceClock(nil, failure: error as? JarvisReplayError ?? .sourceClockUnavailable)
            stop(); onFailure?(error); return
        }
        let origin = anchor.referenceUptime
        let wallOrigin = anchor.utc
        let fixedClock = anchor
        task = Task { @MainActor [weak self] in
            while let self, self.isRunning, !Task.isCancelled {
                let now = ProcessInfo.processInfo.systemUptime
                if self.mediaEnded {
                    if self.audio.pollOutputDrain(at: now) { self.stop(); self.onFinished?(); return }
                    try? await Task.sleep(for: .milliseconds(10))
                    continue
                }
                let elapsed = now - origin
                if elapsed < 0 { try? await Task.sleep(for: .milliseconds(10)); continue }
                do {
                    if let manual = self.manualInput {
                        if let gate = self.memoryGate,
                           let index = manual.activeOrdinal ?? self.timeline.pendingUtteranceOrdinal,
                           case .afterManualResponse(let trigger) = self.plan.utterances[index].trigger {
                            let commit = manual.committed(trigger.afterInputOrdinal)
                            do {
                                try gate.poll(target: index, trigger: trigger, commit: commit, evidence: self.audio.evidence,
                                    clock: fixedClock, now: now,
                                    currentScope: commit.map { self.onManualScopeIsCurrent?($0.identity) ?? false } ?? false,
                                    consume: manual.readyOrdinal == index)
                            } catch {
                                gate.recordPollingFailure(error, target: index, at: now) { ordinal, code, time in
                                    self.audio.failUtterance(ordinal, code: code, at: time)
                                }
                                throw error
                            }
                        }
                        manual.memoryReadinessEvidence = self.memoryGate?.json
                        try manual.advance(inputs: self.audio.evidence.utterances, at: now,
                            sceneDeadline: origin + self.plan.duration,
                            canEnqueue: { index in
                                guard case .afterManualResponse(let trigger) = self.plan.utterances[index].trigger else { return true }
                                guard let gate = self.memoryGate, let prior = manual.committed(trigger.afterInputOrdinal) else { return false }
                                return try gate.consume(target: index, now: ProcessInfo.processInfo.systemUptime,
                                    currentScope: self.onManualScopeIsCurrent?(prior.identity) ?? false)
                            },
                            enqueue: { index, triggeredAt in
                                try self.audio.enqueuePCM16(self.plan.utterances[index].pcm16,
                                    utteranceOrdinal: index, triggeredAt: triggeredAt)
                            }, release: { identity in
                                guard let release = self.onManualRelease else { throw JarvisReplayError.manualInputIncomplete }
                                try release(identity)
                            })
                        self.audio.recordManualInput(manual.json)
                    }
                } catch { self.stop(); self.onFailure?(error); return }
                let output = self.audio.latestOutput.map { JarvisReplayOutputStart(cursor: $0.cursor, at: $0.at - origin) }
                let response = self.timeline.pendingResponseInputOrdinal.map {
                    self.audio.responseSnapshot(afterInputOrdinal: $0)
                }
                if let ordinal = self.timeline.pendingUtteranceOrdinal {
                    self.audio.armUtterance(ordinal, at: origin + self.timeline.pendingWaitStartElapsed)
                }
                let step = self.timeline.advance(elapsed: elapsed, output: output,
                                                  inputIdle: self.audio.inputIdle && !(self.manualInput?.isBusy ?? false),
                                                 responseSnapshot: response, origin: origin,
                                                 taskClaimSequence: self.taskBarrier?.eligibleSequence(),
                                                 manualResponseReady: self.memoryGate?.probeReady ?? false)
                let triggeredAt = ProcessInfo.processInfo.systemUptime
                self.coalescedFrames += step.skippedFrames
                do {
                    // Scenes advance even while a response-triggered utterance waits.
                    if let index = step.frame {
                        let item = self.plan.frames[index]
                        let capturedAt = wallOrigin.addingTimeInterval(item.at)
                        if let sequence = self.sampler.nextSequenceIfAllowed(capturedAt: capturedAt),
                           let image = UIImage(data: item.image) {
                            var frame = try self.adapter.makeCapturedFrame(from: image, sequence: sequence,
                                capturedAt: capturedAt, source: .glasses)
                            if let encoding = frame.replayEncoding {
                                frame.replayLineage = JarvisReplayFrameLineage(
                                    runID: self.plan.runID, launchManifestSHA256: self.launchManifestSHA256,
                                    preparedFrameOrdinal: index, scheduledAt: item.at,
                                    sourceJPEG: item.image, encodedJPEG: frame.jpegData,
                                    frameSequence: frame.sequence, capturedAt: frame.capturedAt, encoding: encoding)
                            }
                            self.onFrame?(frame)
                        } else { self.adapterDroppedFrames += 1 }
                    }
                    if let error = step.failure {
                        if let ordinal = step.failedUtterance {
                            self.audio.failUtterance(ordinal, code: "trigger_\(String(describing: error))", at: triggeredAt)
                        }
                        throw error
                    }
                    if let index = step.utterance {
                        do {
                            if let manual = self.manualInput {
                                guard let press = self.onManualBegin else { throw JarvisReplayError.manualInputIncomplete }
                                if case .afterManualResponse(let trigger) = self.plan.utterances[index].trigger {
                                    guard let prior = manual.committed(trigger.afterInputOrdinal),
                                          self.memoryGate?.mayPress(target: index, now: triggeredAt,
                                            currentScope: self.onManualScopeIsCurrent?(prior.identity) ?? false) == true else {
                                        throw JarvisReplayError.memoryReadinessUnavailable
                                    }
                                }
                                try manual.begin(input: self.audio.evidence.utterances[index], at: triggeredAt,
                                                 sourceClockOrigin: origin, press: press)
                                if case .afterManualResponse = self.plan.utterances[index].trigger {
                                    try self.memoryGate?.markPressed(target: index, at: ProcessInfo.processInfo.systemUptime)
                                }
                                self.audio.recordManualInput(manual.json)
                            } else if let barrier = self.taskBarrier {
                                try barrier.enqueue(requiresClaim: self.plan.utterances[index].trigger.requiresTaskClaim) { sequence in
                                    try self.audio.enqueuePCM16(self.plan.utterances[index].pcm16, utteranceOrdinal: index,
                                        triggeredAt: triggeredAt, responseSelection: step.responseSelection, taskClaimSequence: sequence)
                                }
                            } else {
                                try self.audio.enqueuePCM16(self.plan.utterances[index].pcm16, utteranceOrdinal: index,
                                    triggeredAt: triggeredAt, responseSelection: step.responseSelection)
                            }
                        } catch {
                            self.audio.failUtterance(index, code: "task_enqueue_refused", at: triggeredAt)
                            throw error
                        }
                    }
                    if step.finished {
                        if self.manualInput?.isBusy == true || self.manualInput?.failureCode != nil {
                            throw JarvisReplayError.manualInputIncomplete
                        }
                        if let barrier = self.taskBarrier, !barrier.completionPublished { throw JarvisReplayError.taskCompletionUnavailable }
                        self.mediaEnded = true
                        self.taskBarrier?.stop()
                        self.audio.beginOutputDrain(scheduled: origin + self.plan.duration, observed: triggeredAt,
                                                    seconds: self.plan.postMediaCaptureSeconds)
                        self.onMediaEnded?()
                    }
                    try await Task.sleep(for: .milliseconds(10))
                } catch is CancellationError { return }
                catch { self.stop(); self.onFailure?(error); return }
            }
        }
    }

    func stop() {
        isRunning = false; retired = true; task?.cancel(); task = nil; sampler.reset()
        taskBarrier?.stop()
        memoryGate?.stop()
        if let manualInput {
            manualInput.memoryReadinessEvidence = memoryGate?.json
            manualInput.retire(); audio.recordManualInput(manualInput.json)
        }
    }
}
#endif
