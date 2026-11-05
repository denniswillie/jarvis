import Foundation

extension JarvisAPIClient: JarvisPluginsAccess {
    func fetchPluginCatalog() async throws -> JarvisPluginCatalogSnapshot {
        try await request(path: "/plugins", method: "GET", responseType: JarvisPluginCatalogSnapshot.self)
    }

    func changePluginPolicy(_ change: JarvisPluginPolicyChange) async throws -> JarvisPluginPolicyReceipt {
        let result: JarvisPluginPolicyReceipt = try await request(
            path: "/plugins/policy", method: "PUT", body: try JSONEncoder().encode(change),
            responseType: JarvisPluginPolicyReceipt.self)
        guard result.confirms(change) else {
            throw JarvisAPIError(code: "plugin_policy_receipt_mismatch",
                                 message: "The plugin change could not be confirmed. Check its status before making another change.")
        }
        return result
    }

    func fetchExtensionMarketplace() async throws -> JarvisExtensionMarketplace? {
        do {
            return try await request(
                path: "/plugins/extensions",
                method: "GET",
                responseType: JarvisExtensionMarketplace.self
            )
        } catch let error as JarvisAPIError where error.statusCode == 404 || error.statusCode == 503 {
            return nil
        }
    }

    func installPluginExtension(id: String) async throws -> JarvisExtensionMarketplace {
        let safeID = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        return try await request(
            path: "/plugins/extensions/\(safeID)/install",
            method: "POST",
            responseType: JarvisExtensionMarketplace.self
        )
    }

    func removePluginExtension(id: String) async throws -> JarvisExtensionMarketplace {
        let safeID = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        return try await request(
            path: "/plugins/extensions/\(safeID)/install",
            method: "DELETE",
            responseType: JarvisExtensionMarketplace.self
        )
    }

    func updateLiveLocation(_ update: JarvisLiveLocationUpdate) async throws {
        let _: JarvisLiveLocationReceipt = try await request(
            path: "/live-state/location",
            method: "PUT",
            body: try JSONEncoder().encode(update),
            responseType: JarvisLiveLocationReceipt.self
        )
    }
}
