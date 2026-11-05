import Foundation

/// Observation only. Emits into the existing diagnostic/control-event paths;
/// never opens a channel or grants input/output permission.
struct JarvisStartupDiagnostics {
    enum Origin: String, CaseIterable {
        case humanStartButton = "human_start_button"
        case replayAutostart = "replay_autostart"
        case unspecified
    }
    enum Predicate: String {
        case sessionInputReady = "session_input_ready"
        case manualBeginEnabled = "manual_begin_enabled"
        case automaticInputEligible = "automatic_input_eligible"
        case nativeMicrophoneEnabled = "native_microphone_enabled"
    }
    struct Marker {
        let at: TimeInterval
        let fields: [String: Any]
    }
    static let maximumMarkers = 64
    let attemptID: UUID
    let origin: Origin
    private(set) var retired = false
    private(set) var markerCount = 0
    private var startedAt: TimeInterval?
    private var sessionID: UUID?
    private var audioSessionID: UUID?
    private var connectionID: UUID?
    private var opened = false
    private var predicates: [Predicate: Bool] = [:]

    mutating func start(at: TimeInterval) -> Marker? {
        guard startedAt == nil, at.isFinite, at >= 0, !retired else { return nil }
        startedAt = at
        return marker("start_requested", at: at)
    }
    mutating func bindSession(_ id: UUID, at: TimeInterval) -> Marker? {
        guard accepts(at), sessionID == nil else { return nil }
        sessionID = id
        return marker("session_bound", at: at)
    }
    mutating func bindAudioSession(_ id: UUID, at: TimeInterval) -> Marker? {
        guard accepts(at), audioSessionID == nil else { return nil }
        audioSessionID = id
        return marker("audio_session_bound", at: at)
    }
    mutating func bindConnection(_ id: UUID, at: TimeInterval) -> Marker? {
        guard accepts(at), connectionID != id else { return nil }
        connectionID = id; opened = false
        return marker("control_connect_requested", at: at)
    }
    mutating func observedOpen(connection: UUID, at: TimeInterval) -> Marker? {
        guard accepts(at), connectionID == connection, !opened else { return nil }
        opened = true
        return marker("control_websocket_open_observed", at: at)
    }
    mutating func predicate(_ name: Predicate, enabled: Bool, at: TimeInterval) -> Marker? {
        guard accepts(at), predicates[name] != enabled else { return nil }
        predicates[name] = enabled
        return marker("predicate_changed", at: at,
                      extra: ["boundary": name.rawValue, "status": enabled ? "enabled" : "disabled"])
    }
    mutating func retire(at: TimeInterval) -> Marker? {
        guard !retired else { return nil }
        let result = marker("attempt_retired", at: at)
        retired = true
        return result
    }
    private func accepts(_ at: TimeInterval) -> Bool {
        !retired && markerCount < Self.maximumMarkers && at.isFinite
            && startedAt.map { at >= $0 } == true
    }
    private mutating func marker(_ event: String, at: TimeInterval,
                                 extra: [String: String] = [:]) -> Marker? {
        guard accepts(at) else { return nil }
        markerCount += 1
        var fields: [String: Any] = [
            "event_type": "client.startup." + (markerCount == Self.maximumMarkers ? "diagnostics_limit" : event),
            "startup_attempt_id": attemptID.uuidString.lowercased(), "reason": origin.rawValue,
        ]
        if let sessionID { fields["session_id"] = sessionID.uuidString.lowercased() }
        if let audioSessionID { fields["audio_session_id"] = audioSessionID.uuidString.lowercased() }
        if let connectionID { fields["control_connection_id"] = connectionID.uuidString.lowercased() }
        if markerCount < Self.maximumMarkers {
            for (key, value) in extra { fields[key] = value }
        }
        return Marker(at: at, fields: fields)
    }
}

// Closed categories only. Never serialize event contents or Error descriptions.
struct JarvisLiveProtocolFailure {
    enum Stage: String, CaseIterable {
        case admission, selection, envelope, payload, transcript
        case mediaSignaling = "media_signaling", routeControl = "route_control"
        case providerAcknowledgement = "provider_acknowledgement"
        case startup, controlWrite = "control_write"
    }
    enum Field: String, CaseIterable {
        case routeState = "route_state", connectionID = "connection_id", selection
        case protocolVersion = "protocol_version", audioSessionID = "audio_session_id"
        case envelope, payload, route, observedAt = "observed_at", role, text
        case transcriptID = "transcript_id", revision
        case mediaCommand = "media_command", mediaOperation = "media_operation"
        case controlCommand = "control_command", providerAcknowledgement = "provider_acknowledgement"
        case offer, controlWrite = "control_write"
    }
    enum Reason: String, CaseIterable {
        case invalidWire = "invalid_wire", incompatibleDriver = "incompatible_driver"
        case identityMismatch = "identity_mismatch", alreadySelected = "already_selected"
        case outputNotFenced = "output_not_fenced", notReady = "not_ready", capacity
        case cancelled, unexpectedError = "unexpected_error"
        init(_ error: Error) {
            switch error as? JarvisVoiceRouteError {
            case .invalidWire?: self = .invalidWire
            case .incompatibleDriver?: self = .incompatibleDriver
            case .identityMismatch?: self = .identityMismatch
            case .alreadySelected?: self = .alreadySelected
            case .outputNotFenced?: self = .outputNotFenced
            case .notReady?: self = .notReady
            case .capacity?: self = .capacity
            case nil: self = error is CancellationError ? .cancelled : .unexpectedError
            }
        }
    }
    static let outerTypes: Set<String> = [
        "voice_route.selected", "voice_route.event", "voice_route.interrupt", "voice_route.close",
        "voice_route.output_bound", "live.media.prepare", "live.media.activate", "audio.provider.selected",
        "missing", "non_string", "unknown",
    ]
    static let nestedTypes: Set<String> = [
        "connection_state", "context_receipt", "command_receipt", "transcript_revision", "route_error",
        "missing", "non_string", "unknown",
    ]
    let fields: [String: Any]

    init(stage: Stage, field: Field, event: JarvisRealtimeServerEvent? = nil, error: Error) {
        func category(_ value: Any?, allowed: Set<String>) -> String {
            guard let value else { return "missing" }
            guard let text = value as? String else { return "non_string" }
            return allowed.contains(text) ? text : "unknown"
        }
        let payload = event?.values["payload"] as? [String: Any]
        fields = ["boundary": "live_protocol_failure", "protocol_stage": stage.rawValue,
                  "protocol_field": field.rawValue, "protocol_reason": Reason(error).rawValue,
                  "outer_type": category(event?.values["type"], allowed: Self.outerTypes),
                  "nested_type": category(payload?["kind"], allowed: Self.nestedTypes)]
    }
}

// Diagnostic data never changes a wire envelope or drives turn admission.
struct RealtimeDiagnostics {
    static let recentItemLimit = 256
    private static let processID = UUID().uuidString
    private static let tokenCharacters = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    private static let tokenFields: Set<String> = [
        "provider", "audio_session_id", "session_id", "connection_id", "segment_id",
        "response_id", "item_id", "client_event_id", "provider_event_id", "frame_id", "visual_session_id",
        "stream_connection_id", "reason", "boundary", "status", "event_type", "route_id", "command_id",
    ]
    private static let numberFields: Set<String> = [
        "connection_generation", "frame_sequence", "audio_end_ms", "render_sample_time", "sample_rate",
        "route_generation", "receive_monotonic_s", "decode_elapsed_ms", "actor_wait_ms",
        "handler_elapsed_ms", "write_elapsed_ms",
    ]
    private static let protocolFields: [String: Set<String>] = [
        "protocol_stage": Set(JarvisLiveProtocolFailure.Stage.allCases.map(\.rawValue)),
        "protocol_field": Set(JarvisLiveProtocolFailure.Field.allCases.map(\.rawValue)),
        "protocol_reason": Set(JarvisLiveProtocolFailure.Reason.allCases.map(\.rawValue)),
        "outer_type": JarvisLiveProtocolFailure.outerTypes,
        "nested_type": JarvisLiveProtocolFailure.nestedTypes,
    ]
    private var recentItems: [String] = []
    private var seenItems: Set<String> = []

    static func safeFields(_ fields: [String: Any]) -> [String: Any] {
        var result: [String: Any] = [:]
        for (key, value) in fields {
            if ["startup_attempt_id", "control_connection_id"].contains(key),
               let value = value as? String, let id = UUID(uuidString: value) {
                result[key] = id.uuidString.lowercased()
            } else if let allowed = protocolFields[key], let value = value as? String, allowed.contains(value) {
                result[key] = value
            } else if tokenFields.contains(key), let value = value as? String, isToken(value) {
                result[key] = value
            } else if numberFields.contains(key), let value = value as? NSNumber,
                      CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite {
                result[key] = value
            } else if key == "captured_at", let value = value as? String,
                      value.count <= 40 {
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                if let date = formatter.date(from: value) {
                    result[key] = formatter.string(from: date)
                }
            }
        }
        return result
    }

    static func record(_ event: String, fields: [String: Any] = [:]) -> String? {
        guard isToken(event) else { return nil }
        var record = safeFields(fields)
        record["event"] = event
        record["schema"] = 1
        record["process_id"] = processID
        record["monotonic_s"] = ProcessInfo.processInfo.systemUptime
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        record["at"] = formatter.string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    mutating func firstAudio(fields: [String: Any]) -> String? {
        let safe = Self.safeFields(fields)
        guard safe["response_id"] != nil || safe["item_id"] != nil else { return nil }
        let response = safe["response_id"] as? String ?? ""
        let item = safe["item_id"] as? String ?? ""
        let key = "\(response)|\(item)"
        guard !seenItems.contains(key) else { return nil }
        seenItems.insert(key)
        recentItems.append(key)
        if recentItems.count > Self.recentItemLimit {
            seenItems.remove(recentItems.removeFirst())
        }
        return Self.record("first_audio_received", fields: safe)
    }

    var retainedItemCount: Int { seenItems.count }

    private static func isToken(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.count <= 160 && value.rangeOfCharacter(from: tokenCharacters.inverted) == nil
    }
}
