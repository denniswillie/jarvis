import Foundation
import MWDATCore

enum JarvisVisualCaptureSource: String, CaseIterable, Identifiable, Codable, Sendable {
    case glasses
    case iphoneCamera = "iphone_camera"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .glasses:
            return "Ray-Ban glasses"
        case .iphoneCamera:
            return "iPhone camera"
        }
    }
}

enum JarvisVisualStreamState: Equatable {
    case stopped
    case starting
    case active
    case failed(String)

    var label: String {
        switch self {
        case .stopped:
            return "Vision stopped"
        case .starting:
            return "Starting vision"
        case .active:
            return "Vision active"
        case .failed:
            return "Vision failed"
        }
    }

    var isActive: Bool {
        if case .active = self { return true }
        return false
    }

    var isStarting: Bool {
        if case .starting = self { return true }
        return false
    }

    var errorMessage: String? {
        if case let .failed(message) = self { return message }
        return nil
    }
}

struct CapturedVisualFrame: Sendable {
    let sequence: Int64
    let capturedAt: Date
    let source: JarvisVisualCaptureSource
    let jpegData: Data
    let width: Int
    let height: Int
#if DEBUG
    // Local replay evidence only. Never serialized into the upload payload.
    var replayEncoding: JarvisReplayImageEncoding?
    var replayLineage: JarvisReplayFrameLineage?
#endif

    var sizeDescription: String {
        "\(width)x\(height)"
    }
}

struct RayBanDATConfiguration: Equatable {
    let sdkVersionDescription: String
    let appLinkURLScheme: String?
    let metaAppID: String?
    let clientToken: String?
    let teamID: String?
    let developerModeExpected: Bool
    let usesMWDATPlistEntry: Bool

    var metaAppIDConfigured: Bool { metaAppID != nil }
    var clientTokenConfigured: Bool { clientToken != nil }
    var teamIDConfigured: Bool { teamID != nil }
    var isConfigured: Bool {
        developerModeExpected || (metaAppIDConfigured && clientTokenConfigured && teamIDConfigured)
    }

    var configurationModeLabel: String {
        if developerModeExpected {
            return usesMWDATPlistEntry ? "Developer Mode + MWDAT callback" : "Developer Mode"
        }
        if metaAppIDConfigured && clientTokenConfigured {
            return "MWDAT app config"
        }
        return "Incomplete"
    }

    static func load(bundle: Bundle = .main) -> RayBanDATConfiguration {
        let datConfig = bundle.object(forInfoDictionaryKey: "MWDAT") as? [String: Any]
        return RayBanDATConfiguration(
            sdkVersionDescription: "MetaWearablesDAT 1.0.1",
            appLinkURLScheme: normalizedString(datConfig?["AppLinkURLScheme"]),
            metaAppID: normalizedString(datConfig?["MetaAppID"]),
            clientToken: normalizedString(datConfig?["ClientToken"]),
            teamID: normalizedString(datConfig?["TeamID"]),
            developerModeExpected: true,
            usesMWDATPlistEntry: datConfig != nil
        )
    }

    private static func normalizedString(_ value: Any?) -> String? {
        guard let value = value as? String else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.hasPrefix("$(") && trimmed.hasSuffix(")") {
            return nil
        }
        guard trimmed.lowercased() != "replace_me" else { return nil }
        return trimmed
    }
}

enum RayBanRegistrationState: String, Equatable {
    case unknown
    case notRegistered
    case registering
    case registered
    case failed

    var label: String {
        switch self {
        case .unknown:
            return "Unknown"
        case .notRegistered:
            return "Not registered"
        case .registering:
            return "Registering"
        case .registered:
            return "Registered"
        case .failed:
            return "Failed"
        }
    }

    static func fromDATState(_ state: RegistrationState) -> RayBanRegistrationState {
        switch state {
        case .registered:
            return .registered
        case .registering:
            return .registering
        case .available, .unavailable:
            return .notRegistered
        @unknown default:
            return .unknown
        }
    }
}

enum RayBanCameraPermissionState: String, Equatable {
    case unknown
    case notRequested
    case requesting
    case granted
    case denied
    case failed

    var label: String {
        switch self {
        case .unknown:
            return "Unknown"
        case .notRequested:
            return "Not requested"
        case .requesting:
            return "Requesting"
        case .granted:
            return "Granted"
        case .denied:
            return "Denied"
        case .failed:
            return "Failed"
        }
    }

    static func fromDATStatus(_ status: PermissionStatus) -> RayBanCameraPermissionState {
        switch status {
        case .granted:
            return .granted
        case .denied:
            return .denied
        @unknown default:
            return .unknown
        }
    }
}

enum RayBanDeviceState: String, Equatable {
    case unknown
    case unavailable
    case disconnected
    case connecting
    case connected
    case failed

    var label: String {
        switch self {
        case .unknown:
            return "Unknown"
        case .unavailable:
            return "Unavailable"
        case .disconnected:
            return "Disconnected"
        case .connecting:
            return "Connecting"
        case .connected:
            return "Connected"
        case .failed:
            return "Failed"
        }
    }
}

enum RayBanStreamState: String, Equatable {
    case stopped
    case starting
    case active
    case paused
    case failed

    var label: String {
        switch self {
        case .stopped:
            return "Stopped"
        case .starting:
            return "Starting"
        case .active:
            return "Active"
        case .paused:
            return "Paused"
        case .failed:
            return "Failed"
        }
    }
}

struct RayBanDATDiagnostics: Equatable {
    var sdkAvailable: Bool
    var configured: Bool
    var registrationState: RayBanRegistrationState
    var permissionState: RayBanCameraPermissionState
    var micPermissionState: RayBanCameraPermissionState
    var deviceState: RayBanDeviceState
    var streamState: RayBanStreamState
    var deviceCount: Int
    var configuration: RayBanDATConfiguration
    var lastFrameReceivedAt: Date?
    var lastFrameSequence: Int64?
    var lastFrameSizeDescription: String?
    var lastError: String?
    var lastRefreshedAt: Date?
    var updatedAt: Date

    static func initial(configuration: RayBanDATConfiguration = .load()) -> RayBanDATDiagnostics {
        RayBanDATDiagnostics(
            sdkAvailable: true,
            configured: configuration.isConfigured,
            registrationState: .unknown,
            permissionState: .unknown,
            micPermissionState: .unknown,
            deviceState: .unknown,
            streamState: .stopped,
            deviceCount: 0,
            configuration: configuration,
            lastFrameReceivedAt: nil,
            lastFrameSequence: nil,
            lastFrameSizeDescription: nil,
            lastError: nil,
            lastRefreshedAt: nil,
            updatedAt: Date()
        )
    }
}
