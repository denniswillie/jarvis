import CoreFoundation
import Foundation


/// An explicit reset is scoped to this audio connection and durable conversation.
struct JarvisContextResetNotice: Equatable {
    let id: UUID
    let audioSessionID: String
    let sessionID: UUID

    init?(event: JarvisRealtimeServerEvent, audioSessionID: String?, sessionID: String) {
        guard ["session.context_invalidated", "session.closed"].contains(event.type),
              let expectedSession = UUID(uuidString: sessionID),
              let currentAudioID = audioSessionID, event.audioSessionId == currentAudioID,
              let rawSession = event.values["session_id"] as? String, UUID(uuidString: rawSession) == expectedSession,
              let reset = event.values["context_reset"] as? [String: Any],
              let version = reset["protocol_version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              let preserve = reset["preserve_session"] as? NSNumber,
              CFGetTypeID(preserve) == CFBooleanGetTypeID(), preserve.boolValue,
              let rawID = reset["id"] as? String, let id = UUID(uuidString: rawID) else { return nil }
        self.id = id
        self.audioSessionID = currentAudioID
        self.sessionID = expectedSession
    }
}

struct JarvisContextReconnectBudget {
    private var attempts: [TimeInterval] = []

    mutating func admit(now: TimeInterval = ProcessInfo.processInfo.systemUptime) -> Bool {
        guard now.isFinite, now >= 0, attempts.last.map({ now >= $0 }) ?? true else { return false }
        attempts.removeAll { now-$0 >= 60 }
        guard attempts.count < 3 else { return false }
        attempts.append(now)
        return true
    }
}


enum JarvisAudioProvider: String, CaseIterable, Identifiable {
    case openai
    case gemini

    var id: String { rawValue }

    // Keep Gemini decoding/protocol support dormant, but only select OpenAI for voice.
    var supportedSelection: JarvisAudioProvider { .openai }

    var label: String {
        switch self {
        case .openai: return "OpenAI"
        case .gemini: return "Gemini (experimental)"
        }
    }

    func audioWebSocketURL(from rawURL: String, directAudio: Bool = false) throws -> URL {
        guard var components = URLComponents(string: rawURL),
              let scheme = components.scheme?.lowercased(),
              ["ws", "wss"].contains(scheme),
              let host = components.host, !host.isEmpty,
              components.fragment == nil else {
            throw JarvisAudioProviderError.invalidURL
        }
        var items = (components.queryItems ?? []).filter {
            $0.name != "audio_provider" && $0.name != "audio_transport"
        }
        items.append(URLQueryItem(name: "audio_provider", value: rawValue))
        if directAudio {
            guard self == .openai else { throw JarvisAudioProviderError.invalidURL }
            items.append(URLQueryItem(name: "audio_transport", value: "webrtc"))
        }
        components.queryItems = items
        guard let url = components.url else { throw JarvisAudioProviderError.invalidURL }
        return url
    }
}

enum JarvisDirectOutputPolicy {
    enum BindingAction: Equatable {
        case ignore
        case retainMessage
        case playMessage(itemID: String?)
        case clear
    }

    static func allowsPlayback(item: [String: Any]) -> Bool {
        guard item["type"] as? String == "message" else { return false }
        guard let phase = item["phase"], !(phase is NSNull) else { return true }
        guard let phase = phase as? String else { return false }
        return phase == "final_answer" || phase == "commentary"
    }

    static func bindingAction(item: [String: Any], responseID: String,
                              permittedResponseID: String?, isRetired: Bool,
                              eventItemID: String? = nil) -> BindingAction {
        guard !isRetired, responseID == permittedResponseID else { return .ignore }
        guard allowsPlayback(item: item) else {
            // A tool item is not playable and cannot supply a message identity.
            // It also must not clip buffered audio from an admitted message.
            return item["type"] as? String == "function_call" ? .retainMessage : .clear
        }
        let nestedItemID = item["id"] as? String
        let identitiesAgree = eventItemID == nil || nestedItemID == nil || eventItemID == nestedItemID
        // Preserve legacy playback without an item ID, but never fabricate a
        // correlated replay identity when the supplied IDs disagree.
        return .playMessage(itemID: identitiesAgree ? (nestedItemID ?? eventItemID) : nil)
    }
}

enum JarvisAudioProviderError: LocalizedError {
    case invalidURL
    case invalidAcknowledgement
    case providerMismatch
    case conflictingAcknowledgement
    case geminiUnsupported
    case audioSessionMismatch
    case notReady

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Jarvis returned an invalid audio WebSocket URL."
        case .invalidAcknowledgement:
            return "The backend returned an unsupported audio provider acknowledgement."
        case .providerMismatch:
            return "The backend selected a different audio provider. Stop and choose a provider manually."
        case .conflictingAcknowledgement:
            return "The backend changed its audio provider acknowledgement during this run."
        case .geminiUnsupported:
            return "This backend did not confirm Gemini support. Choose OpenAI in Settings or update the backend."
        case .audioSessionMismatch:
            return "The audio provider acknowledgement belongs to a different audio session."
        case .notReady:
            return "The selected audio provider is not ready."
        }
    }
}

struct JarvisAudioProviderAcknowledgement: Equatable {
    let provider: JarvisAudioProvider
    let model: String?
    let audioSessionID: String

    init(event: JarvisRealtimeServerEvent) throws {
        guard event.type == "audio.provider.selected",
              let version = event.values["protocol_version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              let rawProvider = event.values["provider"] as? String,
              let provider = JarvisAudioProvider(rawValue: rawProvider),
              let audioID = event.audioSessionId, !audioID.isEmpty, audioID.count <= 256 else {
            throw JarvisAudioProviderError.invalidAcknowledgement
        }
        let model: String?
        if let value = event.values["model"], !(value is NSNull) {
            guard let text = value as? String, !text.isEmpty, text.count <= 256 else {
                throw JarvisAudioProviderError.invalidAcknowledgement
            }
            model = text
        } else {
            model = nil
        }
        self.provider = provider
        self.model = model
        self.audioSessionID = audioID
    }
}

/// One Start snapshot. A saved preference is never evidence of backend selection.
struct JarvisAudioProviderRun {
    let id = UUID()
    let requested: JarvisAudioProvider
    private(set) var acknowledgement: JarvisAudioProviderAcknowledgement?
    private(set) var activeProvider: JarvisAudioProvider?
    private var readyAudioSessionID: String?

    init(requested: JarvisAudioProvider) {
        self.requested = requested
    }

    var activeModel: String? { activeProvider == nil ? nil : acknowledgement?.model }

    mutating func acknowledge(_ event: JarvisRealtimeServerEvent) throws {
        let selection = try JarvisAudioProviderAcknowledgement(event: event)
        guard selection.provider == requested else { throw JarvisAudioProviderError.providerMismatch }
        if activeProvider != nil, selection.audioSessionID != readyAudioSessionID {
            throw JarvisAudioProviderError.audioSessionMismatch
        }
        if let acknowledgement, acknowledgement != selection {
            throw JarvisAudioProviderError.conflictingAcknowledgement
        }
        acknowledgement = selection
    }

    mutating func markReady(audioSessionID: String?) throws {
        if let acknowledgement {
            guard acknowledgement.audioSessionID == audioSessionID else {
                throw JarvisAudioProviderError.audioSessionMismatch
            }
        } else if requested == .gemini {
            throw JarvisAudioProviderError.geminiUnsupported
        }
        // Legacy OpenAI backends have no selection control event.
        readyAudioSessionID = audioSessionID
        activeProvider = requested
    }

    func startMedia(
        captureMode: JarvisAudioCaptureMode, unifiedSession: Bool,
        startMicrophone: () throws -> Void, startCamera: () -> Void
    ) throws {
        guard activeProvider != nil else { throw JarvisAudioProviderError.notReady }
        if captureMode == .autoDetect || unifiedSession {
            try startMicrophone()
        }
        if unifiedSession {
            startCamera()
        }
    }

    func shouldClearPlayback(_ event: JarvisRealtimeServerEvent, snapshot: PlaybackTruncationSnapshot?) -> Bool {
        guard let activeProvider, event.type == "audio.output.clear",
              event.values["provider"] as? String == activeProvider.rawValue,
              let responseID = event.responseId, let snapshot else { return false }
        return snapshot.responseId == responseID && snapshot.itemId == event.itemId
    }
}
