import Foundation

@MainActor
protocol JarvisVisualFrameSource: AnyObject {
    var onFrame: ((CapturedVisualFrame) -> Void)? { get set }
    var isRunning: Bool { get }
    func start() async throws
    func stop()
}

@MainActor
final class VisualFrameCaptureCoordinator {
    var onFrame: ((CapturedVisualFrame, JarvisVisualCaptureSource) -> Void)?

    private let glassesSource: any JarvisVisualFrameSource
#if DEBUG
    private var replaySource: (any JarvisVisualFrameSource)?
    var isReplaySelected: Bool { replaySource != nil }
    func selectReplay(_ source: any JarvisVisualFrameSource) throws {
        guard !isRunning, replaySource == nil else { throw JarvisReplayError.retired }
        replaySource = source
        source.onFrame = { [weak self, weak source] frame in
            guard let self, let source, self.replaySource === source, source.isRunning else { return }
            self.onFrame?(frame, .glasses)
        }
    }
#endif

    init(glassesSource: any JarvisVisualFrameSource) {
        self.glassesSource = glassesSource
        bindSource()
    }

    var isRunning: Bool {
#if DEBUG
        if let replaySource { return replaySource.isRunning }
#endif
        return glassesSource.isRunning
    }

    func startGlasses() async throws {
#if DEBUG
        if ProcessInfo.processInfo.arguments.contains("--jarvis-audio-without-camera") {
            NSLog("[audio-isolation] Camera capture intentionally disabled")
            throw NSError(domain: "JarvisAudioIsolation", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Camera deliberately disabled for audio comparison."])
        }
        if let replaySource { try await replaySource.start(); return }
#endif
        try await glassesSource.start()
    }

    func stop() {
#if DEBUG
        replaySource?.stop()
        replaySource?.onFrame = nil
        replaySource = nil
#endif
        glassesSource.stop()
    }

    private func bindSource() {
        glassesSource.onFrame = { [weak self] frame in
#if DEBUG
            guard self?.replaySource == nil else { return }
#endif
            // The source already delivers on MainActor. Forward synchronously
            // so a frame cannot be queued across a Stop/new Start boundary.
            self?.onFrame?(frame, .glasses)
        }
    }
}
