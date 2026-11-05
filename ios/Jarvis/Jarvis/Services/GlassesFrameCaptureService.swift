import Foundation
import MWDATCamera
import UIKit

@MainActor
final class GlassesFrameCaptureService: JarvisVisualFrameSource {
    var onFrame: ((CapturedVisualFrame) -> Void)?

    private let datService: RayBanDATService
    private let adapter: GlassesFrameAdapter
    private let sampler: FrameSampler
    private var running = false
    private var streamGeneration: UUID?
    private var lifecycleTask: Task<Void, Error>?

    init(
        datService: RayBanDATService,
        adapter: GlassesFrameAdapter = GlassesFrameAdapter(),
        sampler: FrameSampler = FrameSampler(targetFPS: 1.0)
    ) {
        self.datService = datService
        self.adapter = adapter
        self.sampler = sampler
    }

    var isRunning: Bool {
        running
    }

    func start() async throws {
        let generation = UUID()
        streamGeneration = generation
        running = false
        sampler.reset()
        let previous = lifecycleTask
        let task = Task<Void, Error> {
            // DAT calls can suspend without observing cancellation. A new start
            // must wait for the old start and its cleanup, not race their Stop.
            _ = await previous?.result
            try Task.checkCancellation()
            guard streamGeneration == generation else { throw CancellationError() }
            do {
                try await datService.startGlassesFrameStream { [weak self] datFrame in
                    Task { @MainActor in
                        self?.handleDATFrame(datFrame, generation: generation)
                    }
                }
                try Task.checkCancellation()
                guard streamGeneration == generation else { throw CancellationError() }
                running = true
            } catch {
                // Keep cleanup in this operation, before the next operation can
                // start. Never stop a newer stream from an old caller's catch.
                await datService.stopGlassesFrameStream()
                if streamGeneration == generation {
                    running = false
                    sampler.reset()
                }
                throw error
            }
        }
        lifecycleTask = task
        defer {
            if streamGeneration == generation {
                lifecycleTask = nil
            }
        }
        try await task.value
    }

    func stop() {
        streamGeneration = nil
        running = false
        sampler.reset()
        let previous = lifecycleTask
        previous?.cancel()
        lifecycleTask = Task<Void, Error> { [datService] in
            _ = await previous?.result
            await datService.stopGlassesFrameStream()
        }
    }

    private func handleDATFrame(_ datFrame: VideoFrame, generation: UUID) {
        guard running, streamGeneration == generation else { return }
        let capturedAt = Date()
        guard let sequence = sampler.nextSequenceIfAllowed(capturedAt: capturedAt) else { return }

        do {
            let frame = try adapter.makeCapturedFrame(
                from: datFrame,
                sequence: sequence,
                capturedAt: capturedAt,
                source: .glasses
            )
            datService.recordGlassesFrame(frame)
            onFrame?(frame)
        } catch {
            datService.recordGlassesStreamError(
                "Ray-Ban frame conversion failed: \(error.localizedDescription)",
                streamFailed: false
            )
        }
    }
}

struct GlassesFrameAdapter {
    private let maxDimension: CGFloat
    private let jpegQuality: CGFloat
#if DEBUG
    var recordsReplayEncoding = false
#endif

    init(maxDimension: CGFloat = 768, jpegQuality: CGFloat = 0.7) {
        self.maxDimension = maxDimension
        self.jpegQuality = jpegQuality
    }

    func makeCapturedFrame(
        from datFrame: VideoFrame,
        sequence: Int64,
        capturedAt: Date,
        source: JarvisVisualCaptureSource
    ) throws -> CapturedVisualFrame {
        guard let image = datFrame.makeUIImage() else {
            throw GlassesFrameAdapterError.imageConversionFailed
        }
        return try makeCapturedFrame(
            from: image,
            sequence: sequence,
            capturedAt: capturedAt,
            source: source
        )
    }

    func makeCapturedFrame(
        from image: UIImage,
        sequence: Int64,
        capturedAt: Date,
        source: JarvisVisualCaptureSource
    ) throws -> CapturedVisualFrame {
        let sourceSize = image.size
        guard sourceSize.width > 0, sourceSize.height > 0 else {
            throw GlassesFrameAdapterError.emptyImage
        }

        let scale = min(1, maxDimension / max(sourceSize.width, sourceSize.height))
        let outputSize = CGSize(
            width: max(1, (sourceSize.width * scale).rounded()),
            height: max(1, (sourceSize.height * scale).rounded())
        )

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: outputSize, format: format)
        let renderedImage = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: outputSize))
        }

        guard let jpegData = renderedImage.jpegData(compressionQuality: jpegQuality) else {
            throw GlassesFrameAdapterError.jpegEncodingFailed
        }

        var frame = CapturedVisualFrame(
            sequence: sequence,
            capturedAt: capturedAt,
            source: source,
            jpegData: jpegData,
            width: Int(outputSize.width),
            height: Int(outputSize.height)
        )
#if DEBUG
        if recordsReplayEncoding {
            frame.replayEncoding = JarvisReplayImageEncoding(
                sourceWidth: Double(sourceSize.width), sourceHeight: Double(sourceSize.height),
                sourceScale: Double(image.scale), sourceOrientation: image.imageOrientation.rawValue,
                sourcePixelWidth: image.cgImage?.width, sourcePixelHeight: image.cgImage?.height,
                maximumDimension: Double(maxDimension), resizeScale: Double(scale),
                outputWidth: Int(outputSize.width), outputHeight: Int(outputSize.height),
                rendererScale: Double(format.scale), rendererOpaque: format.opaque,
                rendererPreferredRange: format.preferredRange.rawValue, jpegQuality: Double(jpegQuality))
        }
#endif
        return frame
    }
}

enum GlassesFrameAdapterError: LocalizedError {
    case imageConversionFailed
    case emptyImage
    case jpegEncodingFailed

    var errorDescription: String? {
        switch self {
        case .imageConversionFailed:
            return "Could not convert Ray-Ban video frame into an image."
        case .emptyImage:
            return "Ray-Ban video frame image was empty."
        case .jpegEncodingFailed:
            return "Could not encode Ray-Ban video frame as JPEG."
        }
    }
}
