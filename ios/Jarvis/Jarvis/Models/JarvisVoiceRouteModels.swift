import CoreFoundation
import Foundation

enum JarvisVoiceRouteError: LocalizedError {
    case invalidWire, incompatibleDriver, identityMismatch, alreadySelected, outputNotFenced, notReady, capacity

    var errorDescription: String? {
        switch self {
        case .invalidWire: return "Invalid voice-route protocol message."
        case .incompatibleDriver: return "The selected voice route is not supported by this media driver."
        case .identityMismatch: return "The voice event belongs to another route or audio connection."
        case .alreadySelected: return "A voice route cannot change during an active connection."
        case .outputNotFenced: return "The previous audio output could not be fenced."
        case .notReady: return "Voice-route negotiation is not complete."
        case .capacity: return "Voice-route evidence capacity reached; restart the route."
        }
    }
}

/// Strict parsing is separate from authorization, which stays on the backend.
enum JarvisVoiceWire {
    static func keys(_ object: [String: Any], required: Set<String>, optional: Set<String> = []) throws {
        let actual = Set(object.keys)
        guard required.isSubset(of: actual), actual.isSubset(of: required.union(optional)) else {
            throw JarvisVoiceRouteError.invalidWire
        }
    }

    static func object(_ value: Any?) throws -> [String: Any] {
        guard let object = value as? [String: Any] else { throw JarvisVoiceRouteError.invalidWire }
        return object
    }

    static func string(_ value: Any?, limit: Int = 256) throws -> String {
        guard let text = value as? String, !text.isEmpty, text.count <= limit else {
            throw JarvisVoiceRouteError.invalidWire
        }
        return text
    }

    static func integer(_ value: Any?, minimum: Int = 1) throws -> Int {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue.rounded(.towardZero) == number.doubleValue,
              number.doubleValue < Double(Int.max), number.doubleValue >= Double(minimum) else {
            throw JarvisVoiceRouteError.invalidWire
        }
        return number.intValue
    }

    static func uuid(_ value: Any?) throws -> UUID {
        guard let result = UUID(uuidString: try string(value, limit: 36)) else {
            throw JarvisVoiceRouteError.invalidWire
        }
        return result
    }

    static func support(_ value: Any?) throws -> Bool {
        switch try string(value) {
        case "supported": return true
        case "unsupported": return false
        default: throw JarvisVoiceRouteError.invalidWire
        }
    }

    static func timestamp(_ value: Any?) throws {
        let text = try string(value, limit: 64)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if formatter.date(from: text) != nil { return }
        formatter.formatOptions = [.withInternetDateTime]
        guard formatter.date(from: text) != nil else { throw JarvisVoiceRouteError.invalidWire }
    }
}

struct JarvisVoiceRouteIdentity: Equatable, Sendable {
    let ownerID: String
    let sessionID: UUID
    let areaID: UUID?
    let routeID: UUID
    let generation: Int

    init(wire: [String: Any]) throws {
        try JarvisVoiceWire.keys(wire, required: ["scope", "route_id", "route_generation"])
        let scope = try JarvisVoiceWire.object(wire["scope"])
        try JarvisVoiceWire.keys(scope, required: ["owner_id", "session_id"], optional: ["area_id"])
        ownerID = try JarvisVoiceWire.string(scope["owner_id"], limit: 200)
        guard ownerID == ownerID.trimmingCharacters(in: CharacterSet(charactersIn: " ")),
              !ownerID.contains("\0") else { throw JarvisVoiceRouteError.invalidWire }
        sessionID = try JarvisVoiceWire.uuid(scope["session_id"])
        if let value = scope["area_id"], !(value is NSNull) { areaID = try JarvisVoiceWire.uuid(value) }
        else { areaID = nil }
        routeID = try JarvisVoiceWire.uuid(wire["route_id"])
        generation = try JarvisVoiceWire.integer(wire["route_generation"])
    }

    var wire: [String: Any] {
        ["scope": ["owner_id": ownerID, "session_id": sessionID.uuidString.lowercased(),
                   "area_id": areaID.map { $0.uuidString.lowercased() } as Any? ?? NSNull()],
         "route_id": routeID.uuidString.lowercased(), "route_generation": generation]
    }
}

enum JarvisVoiceMediaKind: Sendable {
    case nativeWebRTC, streamedPCM

    var transport: String { self == .nativeWebRTC ? "native_voice" : "chained_voice" }
    var topology: String { self == .nativeWebRTC ? "phone_to_provider" : "phone_to_backend" }
    var codec: String { self == .nativeWebRTC ? "opus" : "pcm16" }
    // WebRTC buffer events and inbound-RTP statistics are NOT render receipts.
    var supportsRenderReceipts: Bool { self == .streamedPCM }

    var clientCapabilities: [String: Any] {
        ["supported_versions": [1], "supported_transports": [transport],
         "supported_media_topologies": [topology], "input_codecs": [codec], "output_codecs": [codec],
         "local_output_fencing": true,
         "client_playback_boundaries": supportsRenderReceipts ? ["client_render_lower_bound"] : []]
    }
}

/// Explicit provider profile, not an automatic codec fallback. Existing native
/// callers retain Opus requirements; only the dedicated Live driver opts in to
/// the backend's SDP-negotiated codec metadata. Media topology is unchanged.
enum JarvisVoiceRouteProfile { case standard, liveWebRTC }

struct JarvisVoiceRouteSelection: Equatable {
    let route: JarvisVoiceRouteIdentity
    let audioSessionID: String
    let mediaKind: JarvisVoiceMediaKind
    let playbackEvidence: Bool
    private let capabilityData: Data

    init(event: JarvisRealtimeServerEvent, mediaKind: JarvisVoiceMediaKind,
         profile: JarvisVoiceRouteProfile = .standard) throws {
        guard profile != .liveWebRTC || mediaKind == .nativeWebRTC else {
            throw JarvisVoiceRouteError.incompatibleDriver
        }
        let expectedCodec = profile == .liveWebRTC ? "webrtc_negotiated" : mediaKind.codec
        let value = event.values
        try JarvisVoiceWire.keys(value, required: ["type", "protocol_version", "route", "audio_session_id", "capabilities"])
        guard event.type == "voice_route.selected", try JarvisVoiceWire.integer(value["protocol_version"]) == 1 else {
            throw JarvisVoiceRouteError.invalidWire
        }
        route = try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(value["route"]))
        audioSessionID = try JarvisVoiceWire.string(value["audio_session_id"])
        let capabilities = try JarvisVoiceWire.object(value["capabilities"])
        try JarvisVoiceWire.keys(capabilities, required: ["transport", "media_topology", "control_owner",
            "credential_mode", "input_codec", "output_codec", "vad_owner", "explicit_input_commit",
            "explicit_response_create", "full_duplex", "image_input", "provider_context_receipts",
            "notification_output_attribution", "client_playback_evidence", "provider_interruption", "operations"])
        guard capabilities["transport"] as? String == mediaKind.transport,
              capabilities["media_topology"] as? String == mediaKind.topology,
              capabilities["input_codec"] as? String == expectedCodec,
              capabilities["output_codec"] as? String == expectedCodec,
              capabilities["control_owner"] as? String == "backend",
              capabilities["credential_mode"] as? String == "server_only" else {
            throw JarvisVoiceRouteError.incompatibleDriver
        }
        guard ["phone", "provider", "route", "none"].contains(try JarvisVoiceWire.string(capabilities["vad_owner"])) else {
            throw JarvisVoiceRouteError.invalidWire
        }
        for key in ["explicit_input_commit", "explicit_response_create", "full_duplex", "image_input",
                    "provider_context_receipts", "provider_interruption"] {
            _ = try JarvisVoiceWire.support(capabilities[key])
        }
        let attribution = try JarvisVoiceWire.support(capabilities["notification_output_attribution"])
        playbackEvidence = try JarvisVoiceWire.support(capabilities["client_playback_evidence"])
        guard !playbackEvidence || (mediaKind.supportsRenderReceipts && attribution) else {
            throw JarvisVoiceRouteError.incompatibleDriver
        }
        guard let operations = capabilities["operations"] as? [[String: Any]], operations.count == 6 else {
            throw JarvisVoiceRouteError.invalidWire
        }
        var names = Set<String>()
        let expected: Set<String> = ["start", "publish_context", "offer_notification", "retire_context", "interrupt_speech", "close"]
        for operation in operations {
            try JarvisVoiceWire.keys(operation, required: ["operation", "support"], optional: ["reason"])
            let name = try JarvisVoiceWire.string(operation["operation"])
            guard expected.contains(name), names.insert(name).inserted else { throw JarvisVoiceRouteError.invalidWire }
            let supported = try JarvisVoiceWire.support(operation["support"])
            if !supported { _ = try JarvisVoiceWire.string(operation["reason"], limit: 512) }
            else if let reason = operation["reason"], !(reason is NSNull) { _ = try JarvisVoiceWire.string(reason, limit: 512) }
            guard supported || !["start", "interrupt_speech", "close"].contains(name) else {
                throw JarvisVoiceRouteError.incompatibleDriver
            }
        }
        self.mediaKind = mediaKind
        capabilityData = try JSONSerialization.data(withJSONObject: capabilities, options: [.sortedKeys])
    }

    func envelope(type: String) -> [String: Any] {
        ["type": type, "protocol_version": 1, "audio_session_id": audioSessionID, "route": route.wire]
    }

    func validateEnvelope(_ event: JarvisRealtimeServerEvent) throws {
        guard try JarvisVoiceWire.integer(event.values["protocol_version"]) == 1,
              event.audioSessionId == audioSessionID,
              try JarvisVoiceRouteIdentity(wire: JarvisVoiceWire.object(event.values["route"])) == route else {
            throw JarvisVoiceRouteError.identityMismatch
        }
    }
}

struct JarvisVoiceOutputBinding: Equatable {
    let offerID: UUID
    let attemptID: UUID
    let attemptGeneration: Int
    let outputID: UUID
    let responseID: String?
    let itemID: String?

    init(event: JarvisRealtimeServerEvent, selection: JarvisVoiceRouteSelection) throws {
        try selection.validateEnvelope(event)
        try JarvisVoiceWire.keys(event.values, required: ["type", "protocol_version", "audio_session_id", "route",
                                                       "offer_id", "attempt_id", "attempt_generation", "output"])
        offerID = try JarvisVoiceWire.uuid(event.values["offer_id"])
        attemptID = try JarvisVoiceWire.uuid(event.values["attempt_id"])
        attemptGeneration = try JarvisVoiceWire.integer(event.values["attempt_generation"])
        let output = try JarvisVoiceWire.object(event.values["output"])
        try JarvisVoiceWire.keys(output, required: ["output_id"], optional: ["provider_response_ref", "provider_item_ref"])
        outputID = try JarvisVoiceWire.uuid(output["output_id"])
        responseID = try Self.reference(output["provider_response_ref"])
        itemID = try Self.reference(output["provider_item_ref"])
    }

    private static func reference(_ value: Any?) throws -> String? {
        guard let value, !(value is NSNull) else { return nil }
        return try JarvisVoiceWire.string(value)
    }
}
