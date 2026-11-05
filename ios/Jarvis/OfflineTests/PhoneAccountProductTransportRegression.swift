import Foundation

private final class AccountProductProtocol: URLProtocol {
    static var response: ((URLRequest) throws -> (Int, String))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, body) = try Self.response!(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                                           headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@main
enum PhoneAccountProductTransportRegression {
    static func main() async throws {
        var count = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); count += 1 }
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [AccountProductProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel(); AccountProductProtocol.response = nil }
        let api = JarvisAPIClient(configuration: JarvisConfiguration(backendBaseURL: URL(string: "https://account-fixture.invalid")!,
                                                                   staticBearerToken: "fixture-only"), session: session)
        let ownerSession = UUID()
        func inspect(_ request: URLRequest, _ path: String, _ method: String) {
            check(request.url?.host == "account-fixture.invalid", "Only intercepted fixture host")
            check(request.url?.path == path && request.httpMethod == method, "Agreed route and method")
            check(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-only", "Existing authenticated transport")
        }
        let empty = "{\"owner_id\":\"owner\",\"behavior_instructions\":\"\",\"revision\":0,\"created_at\":null,\"updated_at\":null}"
        AccountProductProtocol.response = { request in
            inspect(request, "/assistant/profile", "GET")
            check(request.url?.query == nil && request.httpBody == nil, "Read has no guessed owner, session or write")
            return (200, empty)
        }
        let profile = try await api.fetchBehaviorProfile()
        check(profile.revision == 0 && profile.createdAt == nil && profile.updatedAt == nil && profile.behaviorInstructions.isEmpty,
              "Read-only absent profile preserves revision zero and null timestamps")
        AccountProductProtocol.response = { request in
            inspect(request, "/assistant/profile", "PUT")
            var data = request.httpBody ?? Data()
            if data.isEmpty, let stream = request.httpBodyStream {
                stream.open(); defer { stream.close() }
                var bytes = [UInt8](repeating: 0, count: 1024)
                while stream.hasBytesAvailable {
                    let n = stream.read(&bytes, maxLength: bytes.count)
                    if n <= 0 { break }; data.append(contentsOf: bytes.prefix(n))
                }
            }
            let body = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            check(Set(body.keys) == ["behavior_instructions", "expected_revision"], "Update cannot select another owner or session")
            check(body["behavior_instructions"] as? String == "Use café examples. 🧪" && body["expected_revision"] as? Int == 0,
                  "Unicode instructions and exact expected revision reach real encoder")
            return (200, "{\"owner_id\":\"owner\",\"behavior_instructions\":\"Use café examples. 🧪\",\"revision\":1,\"created_at\":\"now\",\"updated_at\":\"now\"}")
        }
        let updated = try await api.updateBehaviorProfile(JarvisBehaviorProfileUpdate(behaviorInstructions: "Use café examples. 🧪", expectedRevision: 0))
        check(updated.revision == 1, "Flat successful save decodes")
        var attempts = 0
        AccountProductProtocol.response = { _ in
            attempts += 1
            return (409, "{\"error\":{\"code\":\"behavior_profile_revision_conflict\",\"message\":\"Changed\",\"details\":{\"expected_revision\":1,\"current_revision\":2}}}")
        }
        do {
            _ = try await api.updateBehaviorProfile(JarvisBehaviorProfileUpdate(behaviorInstructions: "Draft", expectedRevision: 1))
            preconditionFailure("Conflict accepted")
        } catch let error as JarvisAPIError {
            check(error.statusCode == 409 && error.code == "behavior_profile_revision_conflict", "CAS conflict retains typed error")
            check(attempts == 1, "Transport does not retry conflicting profile writes")
        }
        AccountProductProtocol.response = { _ in (403, "{\"error\":{\"code\":\"denied\",\"message\":\"Sign in\"}}") }
        do {
            _ = try await api.fetchBehaviorProfile(); preconditionFailure("Denied profile accepted")
        } catch let error as JarvisAPIError { check(error.statusCode == 403, "Auth loss is not decoded as absent profile") }
        let event = try JSONDecoder().decode(JarvisBehaviorProfileCommitted.self, from: Data("""
            {"type":"behavior_profile.committed","event_id":"behavior-profile:owner:2","owner_id":"owner",
             "revision":2,"behavior_instructions":"New","committed_at":"now"}
            """.utf8))
        check(event.revision == 2 && event.ownerID == "owner", "Committed profile envelope matches fixed Core callback DTO")
        let frame = JarvisRealtimeServerEvent(type: "behavior_profile.committed", values: [
            "event_id": "profile:3", "owner_id": "owner", "revision": 3,
            "behavior_instructions": "Readable", "committed_at": "now",
            "session_id": ownerSession.uuidString, "audio_session_id": UUID().uuidString
        ])
        check(frame.behaviorProfileCommitted?.revision == 3,
              "Actual server-event wrapper decodes only its matching committed type")
        var malformed = frame.values; malformed["revision"] = true
        check(JarvisRealtimeServerEvent(type: frame.type, values: malformed).behaviorProfileCommitted == nil,
              "Malformed boolean revision cannot enter profile callback")
        malformed = frame.values; malformed.removeValue(forKey: "owner_id")
        check(JarvisRealtimeServerEvent(type: frame.type, values: malformed).behaviorProfileCommitted == nil,
              "Missing owner is rejected before event delivery")
        print("Phone account-product transport passed: \(count) checks")
    }
}
