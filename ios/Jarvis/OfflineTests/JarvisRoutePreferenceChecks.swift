import Foundation

/// Runs only the production preference declarations, without initializing audio.
@main
struct JarvisRoutePreferenceChecks {
    enum Failure: Error { case assertion(String) }

    @MainActor
    static func main() throws {
        var checks = 0
        func expect(_ condition: Bool, _ message: String) throws {
            guard condition else { throw Failure.assertion(message) }
            checks += 1
            print("PASS: \(message)")
        }

        let suite = "JarvisRoutePreferenceChecks." + UUID().uuidString
        guard let voiceDefaults = UserDefaults(suiteName: suite) else {
            throw Failure.assertion("Cannot create isolated preference suite")
        }
        defer { voiceDefaults.removePersistentDomain(forName: suite) }
        let voiceKey = JarvisVoiceRouteChoice.preferenceKey
        try expect(voiceDefaults.object(forKey: voiceKey) == nil, "Voice preference starts unset")
        try expect(JarvisVoiceRouteChoice.load(from: voiceDefaults) == .live, "Unset voice preference selects Live")
        try expect(voiceDefaults.object(forKey: voiceKey) == nil, "Loading the voice default does not persist a choice")
        for choice in JarvisVoiceRouteChoice.allCases {
            choice.save(to: voiceDefaults)
            try expect(JarvisVoiceRouteChoice.load(from: voiceDefaults) == choice, "Explicit \(choice.rawValue) is respected")
            try expect(voiceDefaults.string(forKey: voiceKey) == choice.rawValue, "Explicit \(choice.rawValue) is not overwritten")
        }
        voiceDefaults.removeObject(forKey: voiceKey)
        try expect(JarvisVoiceRouteChoice.load(from: voiceDefaults) == .live, "Removing the voice preference restores Live")

        // NSArgumentDomain is process-local and volatile. Never set or remove a
        // persistent standard preference, and fail rather than touch existing data.
        let duplexDefaults = UserDefaults.standard
        let duplexKey = JarvisDuplexAudioRoute.preferenceKey
        guard duplexDefaults.object(forKey: duplexKey) == nil else {
            throw Failure.assertion("Host already has a duplex preference; refusing to modify it")
        }
        let arguments = duplexDefaults.volatileDomain(forName: UserDefaults.argumentDomain)
        defer { duplexDefaults.setVolatileDomain(arguments, forName: UserDefaults.argumentDomain) }
        try expect(JarvisDuplexAudioRoute.isSelected, "Unset duplex preference enables capture")
        try expect(duplexDefaults.object(forKey: duplexKey) == nil, "Loading the duplex default does not persist a choice")
        for selected in [false, true] {
            var values = arguments
            values[duplexKey] = selected
            duplexDefaults.setVolatileDomain(values, forName: UserDefaults.argumentDomain)
            try expect(JarvisDuplexAudioRoute.isSelected == selected, "Explicit duplex \(selected) is respected")
            try expect(duplexDefaults.object(forKey: duplexKey) as? Bool == selected, "Explicit duplex \(selected) is not overwritten")
        }
        duplexDefaults.setVolatileDomain(arguments, forName: UserDefaults.argumentDomain)
        try expect(JarvisDuplexAudioRoute.isSelected, "Removing the duplex override restores capture")
        print("PASS: \(checks) preference checks; no audio, app launch, or network calls")
    }
}
