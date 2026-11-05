#if DEBUG
import AVFoundation
import MWDATCamera
import MWDATCore
import ObjectiveC
import SwiftUI
import WebRTC

/// Physical-device isolation only. Optional DAT camera; no backend, model, or peer connection.
@MainActor
struct JarvisDuplexLifecycleProbeView: View {
    @StateObject private var probe = JarvisDuplexLifecycleProbe()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text(probe.cameraTransitions ? "Camera/microphone transition test" : "Microphone restart test")
                    .font(.system(.largeTitle, design: .serif))
                Text(probe.cameraTransitions
                     ? "Keep the glasses connected. Tap Run and count aloud throughout: 8 seconds without the camera, 10 seconds streaming, then 8 seconds after stopping the camera/session. Camera setup may add a short wait. One microphone unit stays running. You will not hear Jarvis."
                     : "Keep the glasses connected. Tap Run, then count aloud continuously for about 30 seconds. This runs two captures with a short stop between them. You will not hear Jarvis.")
                Text(probe.cameraTransitions
                     ? "Only numeric diagnostics are saved on this phone. Camera frames are counted and discarded. No audio/video files, backend/model calls, or WebRTC connection. Camera permission must already be granted."
                     : "Only numeric diagnostics are saved on this phone. No recorded audio, camera, network or model calls.")
                    .font(.footnote)
                Text(probe.allocateBeforeActivation
                     ? "Allocation before audio-session activation"
                     : "Current order: allocation after activation")
                    .font(.system(.subheadline, design: .monospaced))
                Button(probe.running ? "Stop test" : (probe.cameraTransitions ? "Run camera transitions" : "Run two captures")) {
                    if probe.running { probe.cancel() } else { probe.start() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!probe.running && (probe.requiresRelaunch || (probe.cameraTransitions && !probe.cameraReady)))
                if probe.requiresRelaunch {
                    Text("Cleanup was not confirmed. Relaunch before another test.")
                }
                Text(probe.status).font(.headline)
                Text(probe.latestCounters)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
            }
            .padding(24)
        }
        .task { await probe.prepareCameraIfNeeded() }
        .onDisappear { probe.cancel() }
        .onChange(of: scenePhase) { phase in
            if phase != .active { probe.cancel() }
        }
    }
}

private final class JarvisDuplexLifecycleProbe: ObservableObject, @unchecked Sendable {
    @Published private(set) var running = false
    @Published private(set) var status = "Ready. No capture is running."
    @Published private(set) var latestCounters = ""
    @Published private(set) var requiresRelaunch = false
    @Published private(set) var cameraReady = false
    let allocateBeforeActivation = ProcessInfo.processInfo.arguments.contains("--allocate-before-activation")
    let cameraTransitions = ProcessInfo.processInfo.arguments.contains("--probe-camera-transitions")

    private let queue = DispatchQueue(label: "jarvis.duplex-lifecycle-probe", qos: .userInitiated)
    private let cancellationLock = NSLock()
    private var cancelled = false
    private var observations: [[String: Any]] = []
    private let cameraService = RayBanDATService()
    private var preparingCamera = false

    @MainActor
    func prepareCameraIfNeeded() async {
        guard cameraTransitions, !cameraReady, !running, !preparingCamera else { return }
        preparingCamera = true
        defer { preparingCamera = false }
        status = "Checking the app's camera service. No microphone capture yet."
        queue.async { self.record("camera_preflight_started", values: ["camera_owner": "RayBanDATService"]) }
        let camera = JarvisDuplexProbeCameraObservations()
        var startTask: Task<Void, Never>?
        var failure: String?

        func waitUntil(_ label: String, cleanup: Bool = false, ready: () -> Bool) async throws {
            let deadline = ProcessInfo.processInfo.systemUptime + (cleanup ? 8 : 12)
            while !ready() {
                if !cleanup {
                    try Task.checkCancellation()
                    if isCancelled { throw CancellationError() }
                    if let error = camera.read().error { throw ProbeError.camera(error) }
                }
                guard ProcessInfo.processInfo.systemUptime < deadline else {
                    throw ProbeError.camera("Camera preflight timed out: \(label)")
                }
                // Cleanup must also get a bounded opportunity if the view's task was cancelled.
                if cleanup {
                    await withCheckedContinuation { continuation in
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { continuation.resume() }
                    }
                } else {
                    try await Task.sleep(nanoseconds: 50_000_000)
                }
            }
        }

        do {
            cameraService.configureIfNeeded()
            guard cameraService.diagnostics.sdkAvailable else {
                throw ProbeError.camera(cameraService.diagnostics.lastError ?? "Camera service configuration failed.")
            }
            let wearables = Wearables.shared
            try await waitUntil("registration") { wearables.registrationState == .registered }
            try await waitUntil("connected glasses") {
                wearables.devices.contains {
                    wearables.deviceForIdentifier($0)?.linkState == .connected
                }
            }
            try await waitUntil("the app's selected camera") { cameraService.captureProbeState.hasSelectedDevice }
            status = "Opening the camera through the app's service and waiting for frames."
            startTask = Task { @MainActor in await self.startCamera(camera) }
            try await waitUntil("actual camera frames") {
                let observed = camera.read()
                return observed.startReturned && observed.frames > 0
            }
            let frames = camera.read().frames
            queue.async { self.record("camera_preflight_frames", values: ["video_frames": frames]) }
        } catch {
            failure = String(describing: error)
        }

        status = "Stopping the preflight camera and checking cleanup."
        startTask?.cancel()
        let stopTask = Task { @MainActor in await self.stopCamera(camera) }
        do {
            try await waitUntil("camera cleanup", cleanup: true) {
                (startTask == nil || camera.read().startReturned) && camera.read().stopReturned
            }
            guard cameraService.captureProbeState.fullyStopped else {
                throw ProbeError.camera("The app's camera session did not fully stop.")
            }
            let frames = camera.read().frames
            queue.async { self.record("camera_preflight_stopped", values: ["video_frames": frames, "fully_stopped": true]) }
        } catch {
            requiresRelaunch = true
            let cleanupError = String(describing: error)
            failure = [failure, "Cleanup: \(cleanupError)"].compactMap { $0 }.joined(separator: "; ")
        }
        stopTask.cancel()
        if let failure {
            status = "Camera preflight failed. The speaking test is disabled."
            latestCounters = failure
            queue.async { self.record("camera_preflight_failed", values: ["error": failure]) }
        } else {
            cameraReady = true
            status = "Camera frames and cleanup confirmed. Ready for the counting test."
            queue.async { self.record("camera_preflight_ready", values: ["real_frames_and_stop_confirmed": true]) }
        }
    }

    @MainActor
    private func startCamera(_ camera: JarvisDuplexProbeCameraObservations) async {
        camera.update { $0.cameraState = "starting" }
        do {
            try Task.checkCancellation()
            try await cameraService.startGlassesFrameStream { _ in
                camera.update { $0.frames += 1; $0.cameraState = "streaming" }
            }
            // A timed-out start must not open a camera after the probe has stopped.
            if Task.isCancelled { await cameraService.stopGlassesFrameStream() }
        } catch {
            camera.update { $0.error = String(describing: error) }
        }
        let state = cameraService.captureProbeState
        camera.update {
            $0.startReturned = true
            $0.deviceSessionState = state.session
            $0.cameraState = state.stream
        }
    }

    @MainActor
    private func stopCamera(_ camera: JarvisDuplexProbeCameraObservations) async {
        camera.update { $0.cameraState = "stopping" }
        await cameraService.stopGlassesFrameStream()
        let state = cameraService.captureProbeState
        camera.update {
            $0.stopReturned = true
            $0.fullyStopped = state.fullyStopped
            $0.deviceSessionState = state.session
            $0.cameraState = state.stream
        }
    }

    func start() {
        guard !running, !requiresRelaunch, !cameraTransitions || cameraReady else { return }
        running = true
        latestCounters = ""
        cancellationLock.lock()
        cancelled = false
        cancellationLock.unlock()
        status = "Starting. Please count aloud."
        queue.async { [self] in
            if !cameraTransitions { observations = [] }
            let delegate = JarvisDuplexProbeDelegate(inspectMicrophone: cameraTransitions)
            var finished = true
            for cycle in 1...(cameraTransitions ? 1 : 2) {
                guard !isCancelled, runCycle(cycle, delegate: delegate) else {
                    finished = false
                    break
                }
                if !cameraTransitions && cycle == 1 {
                    show("Stopped between captures. Keep counting aloud.")
                    for _ in 0..<4 {
                        if isCancelled { break }
                        Thread.sleep(forTimeInterval: 1)
                    }
                }
            }
            record("finished", values: [cameraTransitions ? "completed_all_phases" : "completed_both_cycles": finished,
                "cancelled": isCancelled])
            show(finished
                 ? (cameraTransitions ? "All three phases finished. Signal quality still needs review." : "Both captures finished. Signal quality still needs review.")
                 : "Stopped. See the diagnostic result.")
            DispatchQueue.main.async { self.running = false }
        }
    }

    func cancel() {
        cancellationLock.lock()
        cancelled = true
        cancellationLock.unlock()
    }

    private var isCancelled: Bool {
        cancellationLock.lock()
        defer { cancellationLock.unlock() }
        return cancelled
    }

    private func runCycle(_ cycle: Int, delegate: JarvisDuplexProbeDelegate) -> Bool {
        guard let io = JRDDuplexIO(delegate: delegate, sampleRate: 16_000,
            inputChannels: 1, outputChannels: 1, routeGeneration: UInt64(cycle)) else {
            record("construction_failed", values: ["cycle": cycle])
            return false
        }
        let audio = RTCAudioSession.sharedInstance()
        var activated = false
        var succeeded = false
        do {
            if allocateBeforeActivation {
                try check(io.prepare(), operation: "prepare_before_activation")
                record("prepared_before_activation", values: ["cycle": cycle])
            }
            audio.lockForConfiguration()
            do {
                defer { audio.unlockForConfiguration() }
                audio.useManualAudio = true
                audio.isAudioEnabled = false
                try audio.setCategory(.playAndRecord,
                    mode: .voiceChat,
                    options: [.defaultToSpeaker, .allowBluetoothHFP])
                try? audio.setPreferredSampleRate(48_000)
                try? audio.setPreferredIOBufferDuration(0.02)
                try audio.setActive(true)
                activated = true
                let actual = AVAudioSession.sharedInstance().sampleRate
                if actual != 48_000 { try? audio.setPreferredSampleRate(actual) }
            }
            let session = AVAudioSession.sharedInstance()
            let inputs = session.currentRoute.inputs.map { $0.portType.rawValue }
            let outputs = session.currentRoute.outputs.map { $0.portType.rawValue }
            record("session_active", values: ["cycle": cycle, "sample_rate": session.sampleRate,
                "input_channels": session.inputNumberOfChannels, "output_channels": session.outputNumberOfChannels,
                "input_types": inputs, "output_types": outputs])
            guard session.currentRoute.inputs.contains(where: { $0.portType == .bluetoothHFP }),
                  session.currentRoute.outputs.contains(where: { $0.portType == .bluetoothHFP }),
                  abs(session.sampleRate - 16_000) < 1,
                  session.inputNumberOfChannels == 1, session.outputNumberOfChannels == 1 else {
                throw ProbeError.routeNotMatched
            }
            try check(io.configure(), operation: "configure")
            io.setRequests(recording: true, playout: true)
            io.setGates(microphone: true, output: true)
            try check(io.start(), operation: "start")
            record("capture_started", values: ["cycle": cycle])
            if cameraTransitions {
                try runCameraTransitions(io: io, delegate: delegate)
            } else {
                for second in 1...10 {
                    if isCancelled { break }
                    Thread.sleep(forTimeInterval: 1)
                    let counters = String(describing: io.counters())
                    record("sample", values: ["cycle": cycle, "second": second,
                        "counters": counters, "unit": io.controlDiagnosticSnapshot()])
                    show("Capture \(cycle)/2: \(second)/10 seconds. Keep counting aloud.", counters: counters)
                }
            }
            succeeded = !isCancelled
        } catch {
            record("capture_failed", values: ["cycle": cycle, "error": String(describing: error)])
            if cameraTransitions { show("Transition test stopped.", counters: String(describing: error)) }
        }

        let closed = io.close()
        record("unit_closed", values: ["cycle": cycle, "success": closed])
        if !closed {
            succeeded = false
            if cameraTransitions { DispatchQueue.main.async { self.requiresRelaunch = true } }
        }
        if activated {
            audio.lockForConfiguration()
            defer { audio.unlockForConfiguration() }
            do {
                // No blind retry: RTCAudioSession balances its counter even on an error.
                try audio.setActive(false)
                record("session_deactivated", values: ["cycle": cycle, "success": true])
            } catch {
                succeeded = false
                record("session_deactivated", values: ["cycle": cycle, "success": false,
                    "error": String(describing: error)])
            }
        }
        return succeeded
    }

    private func runCameraTransitions(io: JRDDuplexIO, delegate: JarvisDuplexProbeDelegate) throws {
        // Read the pinned bridge's native pointer without adding a production API.
        // All access is on its control queue, before close; never on an audio callback.
        guard let unitIvar = class_getInstanceVariable(JRDDuplexIO.self, "_unit") else {
            throw ProbeError.camera("Native AudioUnit identity is unavailable.")
        }
        let unitStorage = Unmanaged.passUnretained(io).toOpaque().advanced(by: ivar_getOffset(unitIvar))
        let unitID = unitStorage.load(as: UInt.self)
        guard unitID != 0 else { throw ProbeError.camera("Native AudioUnit is missing.") }
        let bridgeID = UInt(bitPattern: Unmanaged.passUnretained(io).toOpaque())
        let audioSession = AVAudioSession.sharedInstance()
        let inputIDs = audioSession.currentRoute.inputs.map(\.uid)
        let outputIDs = audioSession.currentRoute.outputs.map(\.uid)
        let camera = JarvisDuplexProbeCameraObservations()
        var startTask: Task<Void, Never>?
        var stopTask: Task<Void, Never>?
        var stopAttempted = false
        var stopConfirmed = false
        var sameUnit = true
        var sameRoute = true
        let began = ProcessInfo.processInfo.systemUptime
        var windowBegan = began
        var previousFrames: UInt64 = 0
        _ = delegate.takeMicrophoneWindow()

        func sample(_ phase: String) {
            let now = ProcessInfo.processInfo.systemUptime
            let mic = delegate.takeMicrophoneWindow()
            let observed = camera.read()
            let route = audioSession.currentRoute
            let currentUnitID = unitStorage.load(as: UInt.self)
            let hfp = route.inputs.contains { $0.portType == .bluetoothHFP }
                && route.outputs.contains { $0.portType == .bluetoothHFP }
            let routeUnchanged = hfp && route.inputs.map(\.uid) == inputIDs
                && route.outputs.map(\.uid) == outputIDs
                && abs(audioSession.sampleRate - 16_000) < 1
                && audioSession.inputNumberOfChannels == 1 && audioSession.outputNumberOfChannels == 1
            sameUnit = sameUnit && currentUnitID == unitID
            sameRoute = sameRoute && routeUnchanged
            let native = io.counters()
            record("sample", values: ["phase": phase, "elapsed_seconds": now - began,
                "window_seconds": now - windowBegan,
                "microphone_peak_pcm16": mic.peak, "microphone_samples": mic.samples,
                "microphone_nonzero_samples": mic.nonzero, "microphone_callbacks": mic.callbacks,
                "video_frames": observed.frames - previousFrames, "video_frames_total": observed.frames,
                "bridge_id": bridgeID, "audio_unit_id": currentUnitID, "same_audio_unit": currentUnitID == unitID,
                "route_generation": io.routeGeneration, "bluetooth_hfp": hfp, "route_unchanged": routeUnchanged,
                "sample_rate": audioSession.sampleRate,
                "inputs": route.inputs.map { ["type": $0.portType.rawValue, "name": $0.portName, "uid": $0.uid] },
                "outputs": route.outputs.map { ["type": $0.portType.rawValue, "name": $0.portName, "uid": $0.uid] },
                "device_session_state": observed.deviceSessionState,
                "camera_state": observed.cameraState,
                "camera_owner": "RayBanDATService",
                "native_input_callbacks": native.inputCallbackCount,
                "native_rendered_frames": native.renderedInputFrames,
                "native_deliveries": native.inputDeliveryCount,
                "native_peak_cumulative": native.inputPeakBeforeGate,
                "native_microphone_errors": native.microphoneErrors,
                "native_last_microphone_status": native.lastMicrophoneStatus,
                "unit": io.controlDiagnosticSnapshot()])
            show("\(phase): \(Int(now - began)) seconds. Keep counting aloud.",
                 counters: "Mic peak: \(mic.peak) | samples: \(mic.samples) | video frames: +\(observed.frames - previousFrames) | HFP: \(hfp)")
            windowBegan = now
            previousFrames = observed.frames
        }

        // Poll only on the probe queue. SDK tasks run elsewhere; neither the main
        // actor nor the audio callback is blocked. Include setup/stop in the trace.
        func wait(_ phase: String, timeout: TimeInterval, cleanup: Bool = false,
                  until ready: () -> Bool) throws {
            let deadline = ProcessInfo.processInfo.systemUptime + timeout
            defer { sample(phase) } // Label partial boundary windows, rather than discard their peaks.
            while true {
                if !cleanup {
                    if isCancelled { throw CancellationError() }
                    if let error = camera.read().error { throw ProbeError.camera(error) }
                }
                if ready() { return }
                let now = ProcessInfo.processInfo.systemUptime
                if now - windowBegan >= 1 { sample(phase) }
                if now >= deadline { throw ProbeError.camera("Timed out: \(phase)") }
                Thread.sleep(forTimeInterval: 0.05)
            }
        }

        func phase(_ name: String, seconds: TimeInterval) throws {
            let start = ProcessInfo.processInfo.systemUptime
            let firstFrame = camera.read().frames
            record("phase_started", values: ["phase": name, "audio_unit_id": unitID,
                "planned_seconds": seconds, "video_frames_total": firstFrame])
            try wait(name, timeout: seconds + 2) { ProcessInfo.processInfo.systemUptime - start >= seconds }
            record("phase_finished", values: ["phase": name, "audio_unit_id": unitID,
                "duration_seconds": ProcessInfo.processInfo.systemUptime - start,
                "video_frames": camera.read().frames - firstFrame])
        }

        func stopCamera() -> Bool {
            guard !stopAttempted else { return stopConfirmed }
            stopAttempted = true
            startTask?.cancel()
            record("camera_stop_requested", values: ["audio_unit_id": unitID])
            stopTask = Task { @MainActor in
                await self.stopCamera(camera)
            }
            do {
                try wait("camera_stopping", timeout: 8, cleanup: true) {
                    let observed = camera.read()
                    return (startTask == nil || observed.startReturned) && observed.stopReturned && observed.fullyStopped
                }
                stopConfirmed = true
            } catch {
                record("camera_stop_failed", values: ["error": String(describing: error)])
                DispatchQueue.main.async { self.requiresRelaunch = true }
            }
            record("camera_stop_result", values: ["confirmed": stopConfirmed, "audio_unit_id": unitID,
                "start_returned": camera.read().startReturned, "stop_returned": camera.read().stopReturned])
            return stopConfirmed
        }

        defer {
            // Always request/observe camera cleanup before runCycle closes the audio unit.
            _ = stopCamera()
            startTask?.cancel()
            stopTask?.cancel()
        }

        try phase("no_camera", seconds: 8)
        record("camera_start_requested", values: ["audio_unit_id": unitID])
        startTask = Task { @MainActor in
            await self.startCamera(camera)
        }
        try wait("camera_starting", timeout: 10) {
            let observed = camera.read()
            return observed.startReturned && observed.frames > 0
        }
        try phase("dat_streaming", seconds: 10)
        guard stopCamera() else { throw ProbeError.camera("Camera/session stop was not confirmed; no post-stop phase.") }
        // Production stop owns listener removal; count any callbacks already queued.
        try phase("after_camera_stop", seconds: 8)
        record("transition_result", values: ["same_audio_unit_throughout": sameUnit,
            "same_hfp_route_throughout": sameRoute, "audio_unit_id": unitID,
            "camera_stop_confirmed": stopConfirmed, "video_frames_total": camera.read().frames])
        guard sameUnit else { throw ProbeError.camera("Native AudioUnit identity changed during the test.") }
    }

    private func check(_ status: OSStatus, operation: String) throws {
        if status != noErr { throw ProbeError.audio(operation, status) }
    }

    private func show(_ text: String, counters: String? = nil) {
        DispatchQueue.main.async {
            self.status = text
            if let counters { self.latestCounters = counters }
        }
    }

    private func record(_ event: String, values: [String: Any]) {
        let row = values.merging(["event": event, "at": ISO8601DateFormatter().string(from: Date())]) { _, new in new }
        observations.append(row)
        NSLog("[DuplexLifecycleProbe] %@", String(describing: row))
        do {
            let url = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true).appendingPathComponent("JarvisDuplexLifecycleProbe.json")
            let data = try JSONSerialization.data(withJSONObject: ["schema": 1,
                "allocate_before_activation": allocateBeforeActivation,
                "camera_transitions": cameraTransitions, "observations": observations],
                options: [.sortedKeys])
            try data.write(to: url, options: .atomic)
        } catch {
            NSLog("[DuplexLifecycleProbe] Diagnostic write failed: %@", String(describing: error))
        }
    }

    private enum ProbeError: Error {
        case audio(String, OSStatus)
        case routeNotMatched
        case camera(String)
    }
}

/// Callback-side state contains numbers/status only, never frames or PCM buffers.
private final class JarvisDuplexProbeCameraObservations: @unchecked Sendable {
    struct Snapshot {
        var frames: UInt64 = 0
        var startReturned = false
        var stopReturned = false
        var fullyStopped = false
        var deviceSessionState = "absent"
        var cameraState = "absent"
        var error: String?
    }
    private let lock = NSLock()
    private var value = Snapshot()
    func read() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
    func update(_ body: (inout Snapshot) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        body(&value)
    }
}

/// Runs the normal bridge callbacks without a peer connection or another audio owner.
private final class JarvisDuplexProbeDelegate: NSObject, RTCAudioDeviceDelegate {
    private let inspectMicrophone: Bool
    private let meterLock = NSLock()
    private var peak: Int32 = 0
    private var samples: UInt64 = 0
    private var nonzero: UInt64 = 0
    private var callbacks: UInt64 = 0

    init(inspectMicrophone: Bool = false) {
        self.inspectMicrophone = inspectMicrophone
        super.init()
    }

    func takeMicrophoneWindow() -> (peak: Int32, samples: UInt64, nonzero: UInt64, callbacks: UInt64) {
        meterLock.lock()
        defer { meterLock.unlock() }
        let result = (peak, samples, nonzero, callbacks)
        peak = 0
        samples = 0
        nonzero = 0
        callbacks = 0
        return result
    }

    var preferredInputSampleRate: Double { 48_000 }
    var preferredOutputSampleRate: Double { 48_000 }
    var preferredInputIOBufferDuration: TimeInterval { 0.02 }
    var preferredOutputIOBufferDuration: TimeInterval { 0.02 }
    var deliverRecordedData: RTCAudioDeviceDeliverRecordedDataBlock {
        guard inspectMicrophone else { return { _, _, _, _, _, _, _ in noErr } }
        return { [self] _, _, _, frames, data, _, _ in
            var windowPeak: Int32 = 0
            var sampleCount: UInt64 = 0
            var nonzeroCount: UInt64 = 0
            if let data {
                // JRDDuplexIO delivers interleaved signed PCM16, not Float32.
                for buffer in UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: data)) {
                    guard let pcm = buffer.mData?.assumingMemoryBound(to: Int16.self) else { continue }
                    let count = min(Int(frames) * Int(buffer.mNumberChannels), Int(buffer.mDataByteSize) / 2)
                    for index in 0..<count {
                        let magnitude = abs(Int32(pcm[index])) // INT16_MIN must widen before abs.
                        windowPeak = max(windowPeak, magnitude)
                        if magnitude != 0 { nonzeroCount += 1 }
                    }
                    sampleCount += UInt64(count)
                }
            }
            // No allocation, logging, file I/O, retained PCM, or long work under this lock.
            meterLock.lock()
            peak = max(peak, windowPeak)
            samples += sampleCount
            nonzero += nonzeroCount
            callbacks += 1
            meterLock.unlock()
            return noErr
        }
    }
    var getPlayoutData: RTCAudioDeviceGetPlayoutDataBlock {
        { _, _, _, _, data in
            for buffer in UnsafeMutableAudioBufferListPointer(data) {
                if let bytes = buffer.mData { memset(bytes, 0, Int(buffer.mDataByteSize)) }
            }
            return noErr
        }
    }
    func notifyAudioInputParametersChange() {}
    func notifyAudioOutputParametersChange() {}
    func notifyAudioInputInterrupted() {}
    func notifyAudioOutputInterrupted() {}
    func dispatchAsync(_ block: @escaping () -> Void) { block() }
    func dispatchSync(_ block: () -> Void) { block() }
}
#endif
