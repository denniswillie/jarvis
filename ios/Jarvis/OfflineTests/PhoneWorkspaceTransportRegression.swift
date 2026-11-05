import Foundation

/// Every URL request is intercepted. This exercises the real product extension
/// and existing transport without reaching a server, provider or database.
private final class PhoneWorkspaceURLProtocol: URLProtocol {
    static var responder: ((URLRequest) throws -> (Int, String))?
    static var responseHeaders = ["Content-Type": "application/json"]
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            guard let responder = Self.responder else { preconditionFailure("Unconfigured fixture request") }
            let (status, body) = try responder(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                                           headerFields: Self.responseHeaders)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@main
enum PhoneWorkspaceTransportRegression {
    static func main() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [PhoneWorkspaceURLProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel(); PhoneWorkspaceURLProtocol.responder = nil }
        let api = JarvisAPIClient(configuration: JarvisConfiguration(backendBaseURL: URL(string: "https://fixture.invalid")!,
                                                                    staticBearerToken: "fixture-only"), session: session)
        let sessionID = UUID(), artifactID = UUID(), requestID = UUID()
        var checks = 0
        func check(_ condition: Bool, _ label: String) { precondition(condition, label); checks += 1 }
        func inspect(_ request: URLRequest, path: String, method: String) -> [String: String] {
            check(request.url?.host == "fixture.invalid", "Only the fixture host is used")
            check(request.url?.path == path && request.httpMethod == method, "Agreed C0 route and method")
            check(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-only", "Existing authentication transport reused")
            let items = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems ?? []
            return Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.value ?? "") })
        }
        let namespace = "jarvis-storage-v1:" + String(repeating: "a", count: 64)
        PhoneWorkspaceURLProtocol.responder = { request in
            let query = inspect(request, path: "/client-context", method: "GET")
            check(query["session_id"] == sessionID.uuidString.lowercased(), "Verified context is scoped to the active session")
            return (200, "{\"owner_id\":\"fixture-owner\",\"backend_namespace\":\"\(namespace)\",\"session_id\":\"\(sessionID)\"}")
        }
        let context = try await api.fetchAuthenticatedContext(sessionID: sessionID)
        check(context.hasValidIdentity && context.ownerID == "fixture-owner" && context.backendNamespace == namespace,
              "Core context DTO and transport supply the authenticated journal identity")
        for (owner, backend, returnedSession) in [("", namespace, sessionID), ("fixture-owner", "untrusted", sessionID), ("fixture-owner", namespace, UUID())] {
            PhoneWorkspaceURLProtocol.responder = { _ in
                (200, "{\"owner_id\":\"\(owner)\",\"backend_namespace\":\"\(backend)\",\"session_id\":\"\(returnedSession)\"}")
            }
            do {
                _ = try await api.fetchAuthenticatedContext(sessionID: sessionID)
                preconditionFailure("Invalid journal identity accepted")
            } catch let error as JarvisAPIError {
                check(error.code == "client_context_mismatch", "Invalid owner, namespace or session cannot unlock the journal")
            }
        }
        PhoneWorkspaceURLProtocol.responder = { _ in (403, "{\"error\":{\"code\":\"jarvis_auth_invalid\",\"message\":\"Sign in again\"}}") }
        do {
            _ = try await api.fetchAuthenticatedContext(sessionID: sessionID)
            preconditionFailure("Denied context accepted")
        } catch let error as JarvisAPIError {
            check(error.statusCode == 403, "Context authorization failure preserves the lock signal")
        }
        let presentation = """
        {"request_id":"\(requestID)","action":"open","artifact_kind":"file",
        "artifact_id":"\(artifactID)","version_id":"source_exact","content_ready":true,
        "state":"claimed","created_at":"2026-11-01T22:00:00Z","expires_at":"2026-11-01T22:05:00Z"}
        """
        PhoneWorkspaceURLProtocol.responder = { request in
            let query = inspect(request, path: "/workspace/requests", method: "GET")
            check(query["session_id"] == sessionID.uuidString.lowercased(), "Requests bound to session")
            return (200, "{\"items\":[\(presentation)]}")
        }
        let page = try await api.listWorkspaceRequests(sessionID: sessionID)
        check(page.items.first?.id == requestID, "Flat request contract decodes")

        PhoneWorkspaceURLProtocol.responder = { request in
            _ = inspect(request, path: "/workspace/requests/" + requestID.uuidString.lowercased() + "/ack", method: "POST")
            let data: Data
            if let body = request.httpBody { data = body }
            else {
                let stream = request.httpBodyStream!
                stream.open(); defer { stream.close() }
                var body = Data(), buffer = [UInt8](repeating: 0, count: 1024)
                while stream.hasBytesAvailable {
                    let count = stream.read(&buffer, maxLength: buffer.count)
                    if count <= 0 { break }
                    body.append(contentsOf: buffer.prefix(count))
                }
                data = body
            }
            let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            check(json["state"] as? String == "claimed" && (json["session_id"] as? String)?.lowercased() == sessionID.uuidString.lowercased(),
                  "ACK body has the agreed state and session")
            check(Set(json.keys) == ["session_id", "state"], "No guessed owner, override or filesystem path in ACK")
            return (200, "{\"request\":\(presentation)}")
        }
        _ = try await api.acknowledgeWorkspaceRequest(requestID: requestID, ack: .init(sessionID: sessionID, state: .claimed))

        PhoneWorkspaceURLProtocol.responder = { _ in
            (409, "{\"error\":{\"code\":\"history_changed\",\"message\":\"Restart history\"}}")
        }
        do {
            _ = try await api.listWorkspaceRequests(sessionID: sessionID)
            preconditionFailure("History conflict swallowed")
        } catch let error as JarvisAPIError {
            check(error.code == "history_changed" && error.statusCode == 409, "Existing stable transport error preserved")
        }
        PhoneWorkspaceURLProtocol.responseHeaders["Retry-After"] = "13"
        PhoneWorkspaceURLProtocol.responder = { _ in
            (503, "{\"error\":{\"code\":\"workspace_busy\",\"message\":\"Try later\"}}")
        }
        do {
            _ = try await api.listWorkspaceRequests(sessionID: sessionID)
            preconditionFailure("Busy response swallowed")
        } catch let error as JarvisAPIError {
            check(error.retryAfter == 13 && error.statusCode == 503, "Core's current transport preserves server retry metadata through product extension")
        }
        let exportID = UUID()
        let filePath = "/files/" + exportID.uuidString.lowercased()
        func metadata(id: UUID, payload: String, mime: String) throws -> String {
            let json: [String: Any] = ["file": [
                "id": id.uuidString, "filename": "exact-note", "content_type": mime,
                "size_bytes": Data(payload.utf8).count, "description": "Exact exported note",
                "source": "workspace", "created_at": "2026-11-02T01:00:00Z",
                "memory_source": ["source_id": "separate_exported_file_source", "revision": 1, "generation": 1,
                    "content_digest": "fixture-export-digest", "source_type": "file", "trust": "user",
                    "time_basis": "recorded_at", "recorded_at": "2026-11-02T01:00:00Z",
                    "is_current": true, "ingestion_status": "ready"]]]
            return String(decoding: try JSONSerialization.data(withJSONObject: json), as: UTF8.self)
        }
        for (payload, mime) in [("Exact old note · café", "text/plain"), ("{\"value\":null}", "application/json")] {
            var paths: [String] = []
            PhoneWorkspaceURLProtocol.responder = { request in
                let isContent = request.url?.path == filePath + "/content"
                let query = inspect(request, path: isContent ? filePath + "/content" : filePath, method: "GET")
                check(query == ["session_id": sessionID.uuidString.lowercased()],
                      "Both prepared-file reads carry only the captured session, never the note version")
                paths.append(request.url!.path)
                PhoneWorkspaceURLProtocol.responseHeaders = ["Content-Type": isContent ? mime : "application/json"]
                return (200, isContent ? payload : try metadata(id: exportID, payload: payload, mime: mime))
            }
            let file = try await api.getWorkspaceFileContent(fileID: exportID, sessionID: sessionID)
            check(paths == [filePath, filePath + "/content"] && file.data == Data(payload.utf8) && file.contentType == mime,
                  "Actual shared metadata/content transport returns exact validated \(mime) bytes")
        }
        var wrongIdentityReads = 0
        PhoneWorkspaceURLProtocol.responder = { request in
            _ = inspect(request, path: filePath, method: "GET")
            wrongIdentityReads += 1
            PhoneWorkspaceURLProtocol.responseHeaders = ["Content-Type": "application/json"]
            return (200, try metadata(id: UUID(), payload: "Wrong file", mime: "text/plain"))
        }
        do {
            _ = try await api.getWorkspaceFileContent(fileID: exportID, sessionID: sessionID)
            preconditionFailure("Wrong retained file identity reached content")
        } catch let error as JarvisAPIError {
            check(error.code == "file_identity_changed" && wrongIdentityReads == 1,
                  "Mismatched prepared file identity is rejected before any byte request")
        }
        for deniedPath in [filePath, filePath + "/content"] {
            var paths: [String] = []
            PhoneWorkspaceURLProtocol.responder = { request in
                let path = request.url!.path
                let query = inspect(request, path: paths.isEmpty ? filePath : filePath + "/content", method: "GET")
                check(query["session_id"] == sessionID.uuidString.lowercased(), "Authorization check uses captured session")
                paths.append(path)
                PhoneWorkspaceURLProtocol.responseHeaders = ["Content-Type": "application/json"]
                if path == deniedPath {
                    return (403, "{\"error\":{\"code\":\"jarvis_auth_invalid\",\"message\":\"Sign in again\"}}")
                }
                return (200, try metadata(id: exportID, payload: "Exact note", mime: "text/plain"))
            }
            do {
                _ = try await api.getWorkspaceFileContent(fileID: exportID, sessionID: sessionID)
                preconditionFailure("Denied prepared file was returned")
            } catch let error as JarvisAPIError {
                check(error.statusCode == 403 && paths.last == deniedPath,
                      "Metadata/content denial preserves authorization lock signal and stops acquisition")
            }
        }
        // Compile and exercise the original exact protocol witnesses as well as
        // the new session overloads; older Files callers still omit session_id.
        let legacyFiles: any JarvisFileAccess = api
        let legacyPayload = "Legacy saved file"
        PhoneWorkspaceURLProtocol.responder = { request in
            let isContent = request.url?.path == filePath + "/content"
            let query = inspect(request, path: isContent ? filePath + "/content" : filePath, method: "GET")
            check(query.isEmpty, "Existing Files protocol callers preserve their query behavior")
            PhoneWorkspaceURLProtocol.responseHeaders = ["Content-Type": isContent ? "text/plain" : "application/json"]
            return (200, isContent ? legacyPayload : try metadata(id: exportID, payload: legacyPayload, mime: "text/plain"))
        }
        let legacyMetadata = try await legacyFiles.getRetainedFile(id: exportID.uuidString)
        let legacyBytes = try await legacyFiles.getRetainedFileContent(file: legacyMetadata)
        check(legacyBytes == Data(legacyPayload.utf8), "Original protocol witnesses still retrieve validated bytes")
        print("PhoneWorkspaceTransportRegression: \(checks) checks passed (intercepted HTTP only)")
    }
}
