import Foundation

extension JarvisAPIClient: JarvisPushAccess {
    func getPushStatus() async throws -> JarvisPushStatus {
        try await request(path: "/attention/push-status", method: "GET", responseType: JarvisPushStatus.self)
    }

    func getPushDevices() async throws -> [JarvisPushDevice] {
        try await request(path: "/attention/devices", method: "GET", responseType: [JarvisPushDevice].self)
    }

    func changePushDevice(id: UUID, change: JarvisPendingPushChange) async throws -> JarvisPushReceipt {
        let path = "/attention/devices/" + id.uuidString.lowercased()
        let receipt: JarvisPushReceipt
        switch change {
        case let .register(body):
            receipt = try await request(path: path, method: "PUT", body: JSONEncoder().encode(body),
                                        responseType: JarvisPushReceipt.self)
        case let .revoke(body):
            receipt = try await request(path: path + "/revoke", method: "POST", body: JSONEncoder().encode(body),
                                        responseType: JarvisPushReceipt.self)
        }
        guard receipt.device.id == id,
              receipt.acceptedRevision == change.mutation.expectedRevision + 1,
              receipt.device.revision >= receipt.acceptedRevision else {
            throw JarvisAPIError(code: "push_identity_mismatch", message: "Notification registration returned an unexpected receipt.")
        }
        return receipt
    }
}
