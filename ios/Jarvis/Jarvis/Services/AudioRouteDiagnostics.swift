import AVFoundation
import Foundation
import os

final class AudioRouteDiagnostics {
    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "audio-route")

    func currentSnapshot() -> AudioRouteSnapshot {
        let session = AVAudioSession.sharedInstance()
        let input = session.currentRoute.inputs.first
        let output = session.currentRoute.outputs.first
        let inputName = input?.portName ?? "-"
        let outputName = output?.portName ?? "-"
        let appearsToUseGlasses = Self.appearsToBeGlasses(inputName) || Self.appearsToBeGlasses(outputName)

        return AudioRouteSnapshot(
            category: session.category.rawValue,
            mode: session.mode.rawValue,
            options: String(describing: session.categoryOptions),
            inputName: inputName,
            inputType: input?.portType.rawValue ?? "-",
            inputUID: input?.uid ?? "-",
            outputName: outputName,
            outputType: output?.portType.rawValue ?? "-",
            outputUID: output?.uid ?? "-",
            appearsToUseGlasses: appearsToUseGlasses
        )
    }

    func logCurrentRoute(reason: String) -> AudioRouteSnapshot {
        let snapshot = currentSnapshot()
        logger.info("\(reason, privacy: .public) category=\(snapshot.category, privacy: .public) mode=\(snapshot.mode, privacy: .public) options=\(snapshot.options, privacy: .public) input=\(snapshot.inputName, privacy: .public) input_type=\(snapshot.inputType, privacy: .public) output=\(snapshot.outputName, privacy: .public) output_type=\(snapshot.outputType, privacy: .public) glasses=\(snapshot.appearsToUseGlasses, privacy: .public)")
        return snapshot
    }

    static func appearsToBeGlasses(_ routeName: String) -> Bool {
        let normalized = routeName.lowercased()
        return normalized.contains("ray-ban")
            || normalized.contains("rayban")
            || normalized.contains("meta")
            || normalized.contains("rb meta")
    }
}
