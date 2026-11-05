import CryptoKit
import Foundation

struct JarvisPushStore: JarvisPushPersistence {
    private let keychain: JarvisKeychain
    private let account: String
    private let identityAccount: String

    init(configuration: JarvisConfiguration, keychain: JarvisKeychain = JarvisKeychain()) {
        self.keychain = keychain
        // Consent is credential-scoped. Installation identity is backend-scoped,
        // so a repaired credential can revoke the same owned registration.
        let backend = configuration.backendBaseURL.absoluteString
        let identity = backend + "\n" + configuration.staticBearerToken
        account = "jarvis.push." + SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
        identityAccount = "jarvis.push.installation." + SHA256.hash(data: Data(backend.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    func load() throws -> JarvisPushPreferences {
        if let value = try keychain.read(account: account) {
            return try JSONDecoder().decode(JarvisPushPreferences.self, from: Data(value.utf8))
        }
        var preferences = JarvisPushPreferences()
        if let value = try keychain.read(account: identityAccount) {
            guard let identity = UUID(uuidString: value) else {
                throw JarvisAPIError(code: "push_identity_invalid", message: "Secure notification identity is invalid.")
            }
            preferences.deviceID = identity
        } else {
            try keychain.save(preferences.deviceID.uuidString, account: identityAccount)
        }
        try save(preferences)
        return preferences
    }

    func save(_ preferences: JarvisPushPreferences) throws {
        let data = try JSONEncoder().encode(preferences)
        try keychain.save(String(decoding: data, as: UTF8.self), account: account)
    }
}
