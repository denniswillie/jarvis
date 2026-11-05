import Foundation

/// Product presentation only. These types are intentionally not wire DTOs or
/// a plugin authority model. The agreed adapter must supply confirmed facts,
/// capability permission and actual operation/recovery references.
struct JarvisPluginCardPresentation: Identifiable, Equatable {
    enum Change: Equatable { case enable, disable, preference }
    enum Phase: Equatable {
        case installing, removing, preparing
        var label: String {
            switch self { case .installing: return "Installing…"; case .removing: return "Removing…"; case .preparing: return "Preparing…" }
        }
    }
    enum Recovery: Equatable {
        case retry(String), recheck(String)
        var label: String { if case .retry = self { return "Retry" }; return "Recheck" }
    }
    enum Operation: Equatable {
        case requesting(Change)
        case running(Phase, reference: String)
        case failed(message: String, recovery: Recovery?)
        case unknown(message: String, recheckReference: String?)
        var isInProgress: Bool {
            switch self { case .requesting, .running: return true; case .failed, .unknown: return false }
        }
        var label: String {
            switch self {
            case .requesting: return "Requesting change…"
            case .running(let phase, _): return phase.label
            case .failed: return "Change failed"
            case .unknown: return "Change not confirmed"
            }
        }
        var message: String? {
            switch self { case .failed(let text, _), .unknown(let text, _): return text; default: return nil }
        }
        var recovery: Recovery? {
            switch self {
            case .failed(_, let recovery): return recovery
            case .unknown(_, let reference): return reference.map(Recovery.recheck)
            default: return nil
            }
        }
    }

    let id: String
    let title: String
    let symbol: String
    let detail: String
    let readinessLabel: String
    let accountLabel: String?
    let isSelectedDefault: Bool
    let confirmedEnabled: Bool?
    let allowsPreferenceChange: Bool
    let operation: Operation?
    var installationLabel: String? = nil

    var canChangePreference: Bool { confirmedEnabled != nil && allowsPreferenceChange && operation == nil }
    var defaultLabel: String {
        guard let confirmedEnabled else { return "Suggested default" }
        return confirmedEnabled ? "Selected default" : "Preferred when enabled"
    }

    /// Local explanatory rows, never backend IDs, inventory or enabled state.
    static let selectedDefaults: [Self] = [
        .init(id: "selected-default-video", title: "Video Editor", symbol: "film",
              detail: "Create and edit videos", readinessLabel: "Availability unknown",
              accountLabel: nil, isSelectedDefault: true, confirmedEnabled: nil, allowsPreferenceChange: false, operation: nil),
        .init(id: "selected-default-browser", title: "Browser", symbol: "globe",
              detail: "Browse websites and find information", readinessLabel: "Availability unknown",
              accountLabel: nil, isSelectedDefault: true, confirmedEnabled: nil, allowsPreferenceChange: false, operation: nil),
        .init(id: "selected-default-files", title: "Files & data", symbol: "doc.on.doc",
              detail: "Work with files, tables, and calculations", readinessLabel: "Availability unknown",
              accountLabel: nil, isSelectedDefault: true, confirmedEnabled: nil, allowsPreferenceChange: false, operation: nil)
    ]
}
