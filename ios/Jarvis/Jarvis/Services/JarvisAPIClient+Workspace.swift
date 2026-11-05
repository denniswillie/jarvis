import Foundation

extension JarvisAPIClient: JarvisPreferencesAccess {

    func fetchBehaviorProfile() async throws -> JarvisBehaviorProfile {
        try await request(path: "/assistant/profile", method: "GET", responseType: JarvisBehaviorProfile.self)
    }

    func updateBehaviorProfile(_ update: JarvisBehaviorProfileUpdate) async throws -> JarvisBehaviorProfile {
        try await request(path: "/assistant/profile", method: "PUT", body: JSONEncoder().encode(update),
                          responseType: JarvisBehaviorProfile.self)
    }
}

/// Product routes share Core's authentication, URLSession and replay headers.
extension JarvisAPIClient: JarvisWorkspaceAccess {



    func listWorkspaceRequests(sessionID: UUID) async throws -> JarvisWorkspaceRequestPage {
        try await request(path: "/workspace/requests", method: "GET", queryItems: workspaceQuery(sessionID),
                          responseType: JarvisWorkspaceRequestPage.self)
    }

    func acknowledgeWorkspaceRequest(requestID: UUID, ack: JarvisWorkspaceAcknowledgement) async throws -> JarvisWorkspaceRequestResponse {
        try await request(path: "/workspace/requests/" + requestID.uuidString.lowercased() + "/ack", method: "POST",
                          body: JSONEncoder().encode(ack), responseType: JarvisWorkspaceRequestResponse.self)
    }

    func getWorkspaceFileContent(fileID: UUID, sessionID: UUID) async throws -> JarvisWorkspaceFileContent {
        let session = sessionID.uuidString.lowercased()
        let file = try await getRetainedFile(id: fileID.uuidString.lowercased(), sessionID: session)
        let bytes = try await getRetainedFileContent(file: file, sessionID: session)
        // Current Files metadata does not expose the opaque source version ID.
        // Prepared downloads verify their bytes separately; exact-version file
        // presentation must remain unavailable until that authority seam exists.
        return JarvisWorkspaceFileContent(data: bytes, filename: file.filename, contentType: file.contentType)
    }

    private func workspaceQuery(_ sessionID: UUID) -> [URLQueryItem] {
        [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())]
    }

}
