import Foundation
import Security

final class JarvisSessionStore {
    static let shared = JarvisSessionStore()

    private let defaults: UserDefaults
    private let keychain: JarvisKeychain

    private enum Keys {
        static let backendBaseURL = "jarvis.backendBaseURL"
        static let staticBearerToken = "jarvis.staticBearerToken"
        static let phoneCameraDebugEnabled = "jarvis.phoneCameraDebugEnabled"
        static let audioProvider = "jarvis.audioProvider"
    }

    init(defaults: UserDefaults = .standard, keychain: JarvisKeychain = JarvisKeychain()) {
        self.defaults = defaults
        self.keychain = keychain
        if let savedProvider = defaults.string(forKey: Keys.audioProvider) {
            let provider = JarvisAudioProvider(rawValue: savedProvider) ?? .openai
            defaults.set(provider.supportedSelection.rawValue, forKey: Keys.audioProvider)
        }
    }

    func loadConfiguration() -> JarvisConfiguration? {
        if let savedURL = defaults.string(forKey: Keys.backendBaseURL),
           let token = try? keychain.read(account: Keys.staticBearerToken),
           let url = normalizedURL(savedURL),
           !token.isEmpty {
            return JarvisConfiguration(backendBaseURL: url, staticBearerToken: token)
        }

        guard let bundledURL = readBundledOverride(named: "BackendBaseURL.override.txt"),
              let bundledToken = readBundledOverride(named: "StaticBearerToken.override.txt"),
              let url = normalizedURL(bundledURL),
              !bundledToken.isEmpty else {
            return nil
        }

        return JarvisConfiguration(backendBaseURL: url, staticBearerToken: bundledToken)
    }

    func bundledConfigurationAvailable() -> Bool {
        guard let bundledURL = readBundledOverride(named: "BackendBaseURL.override.txt"),
              let bundledToken = readBundledOverride(named: "StaticBearerToken.override.txt") else {
            return false
        }
        return normalizedURL(bundledURL) != nil && !bundledToken.isEmpty
    }

    func saveConfiguration(backendBaseURL: String, staticBearerToken: String) throws -> JarvisConfiguration {
        guard let url = normalizedURL(backendBaseURL) else {
            throw JarvisAPIError(code: "invalid_backend_url", message: "Enter a valid Jarvis backend URL.")
        }
        let token = staticBearerToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw JarvisAPIError(code: "missing_token", message: "Enter a Jarvis access token.")
        }

        defaults.set(url.absoluteString, forKey: Keys.backendBaseURL)
        try keychain.save(token, account: Keys.staticBearerToken)
        return JarvisConfiguration(backendBaseURL: url, staticBearerToken: token)
    }

    func clearConfiguration() {
        defaults.removeObject(forKey: Keys.backendBaseURL)
        defaults.removeObject(forKey: Keys.audioProvider)
        try? keychain.delete(account: Keys.staticBearerToken)
    }

    var audioProvider: JarvisAudioProvider {
        get { (defaults.string(forKey: Keys.audioProvider).flatMap(JarvisAudioProvider.init(rawValue:)) ?? .openai).supportedSelection }
        set { defaults.set(newValue.supportedSelection.rawValue, forKey: Keys.audioProvider) }
    }

    var isPhoneCameraDebugEnabled: Bool {
        get { defaults.bool(forKey: Keys.phoneCameraDebugEnabled) }
        set { defaults.set(newValue, forKey: Keys.phoneCameraDebugEnabled) }
    }

    private func normalizedURL(_ rawValue: String) -> URL? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let withScheme = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        guard let url = URL(string: withScheme),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              url.host != nil else {
            return nil
        }
        return url
    }

    private func readBundledOverride(named resourceName: String) -> String? {
        let parts = resourceName.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 2 else { return nil }
        let name = parts.dropLast().joined(separator: ".")
        let ext = String(parts.last ?? "")
        guard let url = Bundle.main.url(forResource: name, withExtension: ext),
              let value = try? String(contentsOf: url, encoding: .utf8)
        else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

struct JarvisKeychain {
    private let service: String

    init(service: String = "com.symbiote.jarvis") {
        self.service = service
    }

    func read(account: String) throws -> String? {
        var query = baseQuery(account: account)
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnData as String] = true

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess else {
            throw JarvisAPIError(code: "keychain_read_failed", message: "Could not read secure Jarvis settings.")
        }
        guard let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func save(_ value: String, account: String) throws {
        try? delete(account: account)
        var query = baseQuery(account: account)
        query[kSecValueData as String] = Data(value.utf8)
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw JarvisAPIError(code: "keychain_save_failed", message: "Could not save secure Jarvis settings.")
        }
    }

    func delete(account: String) throws {
        let status = SecItemDelete(baseQuery(account: account) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw JarvisAPIError(code: "keychain_delete_failed", message: "Could not clear secure Jarvis settings.")
        }
    }

    private func baseQuery(account: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }
}
