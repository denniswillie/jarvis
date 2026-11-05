import Foundation

final class JarvisAPIClient: JarvisFileAccess, JarvisWorkAccess {
#if DEBUG
    var evaluationRunID: UUID?
#endif
    private let baseURL: URL
    private let bearerToken: String
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(configuration: JarvisConfiguration, session: URLSession = .shared) {
        self.baseURL = configuration.backendBaseURL
        self.bearerToken = configuration.staticBearerToken
        self.session = session
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
    }

    func startSession(title: String?, resumeActive: Bool) async throws -> StartSessionResponse {
        let body = StartSessionRequest(title: title, resumeActive: resumeActive)
        return try await request(
            path: "/sessions/start",
            method: "POST",
            body: try encoder.encode(body),
            responseType: StartSessionResponse.self
        )
    }

    func sandboxPageRequest(sessionID: UUID, id: UUID) throws -> URLRequest {
        try authenticatedRequest(path: "/sandbox-views/" + id.uuidString.lowercased() + "/page", method: "GET",
            queryItems: [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())], accept: "text/html")
    }

    func mediaStreamRequest(sessionID: UUID, id: UUID) throws -> URLRequest {
        var request = try authenticatedRequest(path: "/media-audio/" + id.uuidString.lowercased() + "/stream", method: "GET",
            queryItems: [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())], accept: "application/octet-stream")
        guard let url = request.url, var components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.scheme == "https" || components.scheme == "http" else {
            throw JarvisAPIError(code: "media_url", message: "Music connection unavailable.")
        }
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        guard let socketURL = components.url else { throw JarvisAPIError(code: "media_url", message: "Music connection unavailable.") }
        request.url = socketURL
        return request
    }

    func reportMediaState(sessionID: UUID, id: UUID, state: String) async throws {
        struct Receipt: Encodable { let session_id: UUID; let state: String }
        struct Ack: Decodable { let accepted: Bool }
        let _: Ack = try await request(path: "/media-audio/" + id.uuidString.lowercased() + "/state", method: "POST",
            body: try encoder.encode(Receipt(session_id: sessionID, state: state)), responseType: Ack.self)
    }

    func getActiveSession() async throws -> JarvisSession? {
        let response: ActiveSessionResponse = try await request(
            path: "/sessions/active",
            method: "GET",
            responseType: ActiveSessionResponse.self
        )
        return response.session
    }

    func fetchAuthenticatedContext(sessionID: UUID) async throws -> JarvisAuthenticatedContext {
        let context: JarvisAuthenticatedContext = try await request(
            path: "/client-context", method: "GET",
            queryItems: [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())],
            responseType: JarvisAuthenticatedContext.self
        )
        guard context.sessionID == sessionID, context.hasValidIdentity else {
            throw JarvisAPIError(code: "client_context_mismatch",
                                 message: "Jarvis could not verify this session's saved-data context.")
        }
        return context
    }

    func closeSession(id: String) async throws {
        let _: SessionDetailResponse = try await request(
            path: "/sessions/\(id)/close",
            method: "POST",
            responseType: SessionDetailResponse.self
        )
    }

    func updateBehaviorInstructions(sessionId: String, instructions: String) async throws -> JarvisSession {
        let body = UpdateBehaviorInstructionsRequest(behaviorInstructions: instructions)
        let response: SessionDetailResponse = try await request(
            path: "/sessions/\(sessionId)/behavior-instructions",
            method: "PATCH",
            body: try encoder.encode(body),
            responseType: SessionDetailResponse.self
        )
        return response.session
    }

    func getWorkTasks(offset: Int = 0) async throws -> [JarvisWorkTask] {
        try await request(path: "/tasks", method: "GET", queryItems: workPage(offset),
                          responseType: [JarvisWorkTask].self)
    }

    func getWorkTask(id: UUID) async throws -> JarvisWorkTask {
        let result: JarvisWorkTask = try await request(path: "/tasks/" + id.uuidString.lowercased(),
                                                       method: "GET", responseType: JarvisWorkTask.self)
        guard result.id == id else { throw workIdentityError() }
        return result
    }

    func getWorkInbox(offset: Int = 0, includeDismissed: Bool = false, taskID: UUID? = nil) async throws -> [JarvisInboxItem] {
        var query = workPage(offset) + [URLQueryItem(name: "include_dismissed", value: String(includeDismissed))]
        if let taskID { query.append(URLQueryItem(name: "task_id", value: taskID.uuidString.lowercased())) }
        return try await request(path: "/attention", method: "GET", queryItems: query,
                                 responseType: [JarvisInboxItem].self)
    }

    func getWorkInboxItem(id: UUID) async throws -> JarvisInboxDetail {
        let result: JarvisInboxDetail = try await request(path: "/attention/" + id.uuidString.lowercased(),
                                                          method: "GET", responseType: JarvisInboxDetail.self)
        guard result.item.id == id else { throw workIdentityError() }
        return result
    }

    func changeWorkTask(id: UUID, operation: JarvisTaskOperation, body: JarvisTaskChange) async throws -> JarvisWorkTask {
        try await request(path: "/tasks/" + id.uuidString.lowercased() + "/" + operation.rawValue,
                          method: "POST", body: try encoder.encode(body), responseType: JarvisWorkTask.self)
    }

    func changeWorkInbox(id: UUID, operation: JarvisInboxOperation, body: JarvisInboxChange) async throws -> JarvisInboxChangeReceipt {
        try await request(path: "/attention/" + id.uuidString.lowercased() + "/" + operation.rawValue,
                          method: "POST", body: try encoder.encode(body), responseType: JarvisInboxChangeReceipt.self)
    }

    private func workPage(_ offset: Int) -> [URLQueryItem] {
        [URLQueryItem(name: "limit", value: "20"),
         URLQueryItem(name: "offset", value: String(min(max(offset, 0), 100000)))]
    }

    private func workIdentityError() -> JarvisAPIError {
        JarvisAPIError(code: "work_identity_mismatch", message: "Jarvis returned a different work item. Refresh to check.")
    }

    func getActiveReminders() async throws -> [JarvisReminderSummary] {
        let response: ActiveRemindersResponse = try await request(
            path: "/reminders/active",
            method: "GET",
            responseType: ActiveRemindersResponse.self
        )
        return response.reminders
    }

    func getReminderEvents(limit: Int = 50) async throws -> [JarvisReminderEvent] {
        let response: ReminderEventsResponse = try await request(
            path: "/reminders/events",
            method: "GET",
            queryItems: [URLQueryItem(name: "limit", value: String(limit))],
            responseType: ReminderEventsResponse.self
        )
        return response.events
    }

    func getRecentMemory(
        limit: Int = 30,
        cursor: String? = nil,
        memoryTypes: [String] = [],
        sessionId: String? = nil
    ) async throws -> JarvisRecentMemoryPage {
        var queryItems = [URLQueryItem(name: "limit", value: String(min(max(limit, 1), 100)))]
        queryItems.append(contentsOf: memoryTypes.map {
            URLQueryItem(name: "memory_types", value: $0)
        })
        if let cursor, !cursor.isEmpty {
            queryItems.append(URLQueryItem(name: "cursor", value: cursor))
        }
        if let sessionId, !sessionId.isEmpty {
            queryItems.append(URLQueryItem(name: "session_id", value: sessionId))
        }
        return try await request(
            path: "/memory/recent",
            method: "GET",
            queryItems: queryItems,
            responseType: JarvisRecentMemoryPage.self
        )
    }

    func getMemoryDetail(
        id: String, revision: Int? = nil, offset: Int = 0, expectedDigest: String? = nil
    ) async throws -> JarvisMemoryDetail {
        var queryItems: [URLQueryItem] = []
        if let revision { queryItems.append(URLQueryItem(name: "revision", value: String(revision))) }
        if offset > 0 { queryItems.append(URLQueryItem(name: "offset", value: String(offset))) }
        if let expectedDigest { queryItems.append(URLQueryItem(name: "expected_digest", value: expectedDigest)) }
        return try await request(
            path: "/memory/\(id)",
            method: "GET",
            queryItems: queryItems,
            responseType: JarvisMemoryDetail.self
        )
    }

    func getMemoryEvidence(id: String, limit: Int = 100) async throws -> JarvisMemoryEvidencePage {
        try await request(
            path: "/memory/\(id)/evidence",
            method: "GET",
            queryItems: [
                URLQueryItem(name: "limit", value: String(min(max(limit, 1), 100)))
            ],
            responseType: JarvisMemoryEvidencePage.self
        )
    }

    func getWorkingMemoryState(sessionId: String? = nil) async throws -> JarvisWorkingMemoryState {
        let queryItems = sessionId.map {
            [URLQueryItem(name: "session_id", value: $0)]
        } ?? []
        return try await request(
            path: "/memory/working-state",
            method: "GET",
            queryItems: queryItems,
            responseType: JarvisWorkingMemoryState.self
        )
    }

    func getRetainedFiles(query: String? = nil) async throws -> JarvisRetainedFilePage {
        var items = [URLQueryItem(name: "limit", value: "8")]
        if let query, !query.isEmpty { items.append(URLQueryItem(name: "query", value: query)) }
        return try await request(path: "/files", method: "GET", queryItems: items,
                                 responseType: JarvisRetainedFilePage.self)
    }

    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        try await getRetainedFile(id: id, sessionID: nil)
    }

    func getRetainedFile(id: String, sessionID: String?) async throws -> JarvisRetainedFile {
        let queryItems = sessionID.map { [URLQueryItem(name: "session_id", value: $0)] } ?? []
        let response: JarvisRetainedFileResponse = try await request(path: filePath(id), method: "GET",
                                                                   queryItems: queryItems,
                                                                   responseType: JarvisRetainedFileResponse.self)
        guard UUID(uuidString: response.file.id) == UUID(uuidString: id) else {
            throw JarvisAPIError(code: "file_identity_changed", message: "Jarvis returned a different file.")
        }
        return response.file
    }

    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        try await getRetainedFileContent(file: file, sessionID: nil)
    }

    func getRetainedFileContent(file: JarvisRetainedFile, sessionID: String?, versionID: String? = nil) async throws -> Data {
        var queryItems = sessionID.map { [URLQueryItem(name: "session_id", value: $0)] } ?? []
        if let versionID { queryItems.append(URLQueryItem(name: "version_id", value: versionID)) }
        let (data, response) = try await responseData(path: filePath(file.id) + "/content",
                                                     method: "GET", queryItems: queryItems, accept: file.mediaType)
        try file.validateContent(data, contentType: response.value(forHTTPHeaderField: "Content-Type"))
        if let versionID, response.value(forHTTPHeaderField: "X-Jarvis-Version-Id") != versionID {
            throw JarvisAPIError(code: "file_identity_changed", message: "The selected file version changed. Refresh the library.")
        }
        return data
    }

    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        try await request(path: filePath(id), method: "DELETE", responseType: JarvisFileDeletion.self)
    }

    private func filePath(_ id: String) throws -> String {
        guard let value = UUID(uuidString: id) else {
            throw JarvisAPIError(code: "invalid_file_id", message: "The saved file identifier is invalid.")
        }
        return "/files/" + value.uuidString.lowercased()
    }

    func getGoogleCalendarStatus() async throws -> GoogleCalendarStatusResponse {
        try await request(
            path: "/oauth/google/status",
            method: "GET",
            responseType: GoogleCalendarStatusResponse.self
        )
    }

    func startGoogleCalendarAuthorization() async throws -> GoogleOAuthStartResponse {
        try await request(
            path: "/oauth/google/start",
            method: "GET",
            responseType: GoogleOAuthStartResponse.self
        )
    }

    func request<Response: Decodable>(
        path: String,
        method: String,
        body: Data? = nil,
        queryItems: [URLQueryItem] = [],
        responseType: Response.Type
    ) async throws -> Response {
        let (data, _) = try await responseData(path: path, method: method, body: body, queryItems: queryItems)
        return try decoder.decode(Response.self, from: data)
    }

    private func responseData(path: String, method: String, body: Data? = nil,
                              queryItems: [URLQueryItem] = [], accept: String = "application/json")
        async throws -> (Data, HTTPURLResponse) {
        let request = try authenticatedRequest(path: path, method: method, body: body,
                                               queryItems: queryItems, accept: accept)
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw JarvisAPIError(code: "invalid_response", message: "Jarvis returned an invalid response.")
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            throw decodeError(
                from: data, statusCode: httpResponse.statusCode,
                retryAfter: Self.retryAfterDelay(from: httpResponse.value(forHTTPHeaderField: "Retry-After"))
            )
        }

        return (data, httpResponse)
    }

    /// File-backed callers supply a progress/access delegate and validate their
    /// exact response contract before adopting the returned temporary location.
    func downloadedResponse(path: String, queryItems: [URLQueryItem], accept: String,
                            delegate: any URLSessionTaskDelegate) async throws -> (URL, HTTPURLResponse, URL) {
        let request = try authenticatedRequest(path: path, method: "GET", queryItems: queryItems, accept: accept)
        let (location, response) = try await session.download(for: request, delegate: delegate)
        guard let httpResponse = response as? HTTPURLResponse, let requestedURL = request.url else {
            try? FileManager.default.removeItem(at: location)
            throw JarvisAPIError(code: "invalid_response", message: "Jarvis returned an invalid response.")
        }
        return (location, httpResponse, requestedURL)
    }

    /// One exact upload or reconciliation attempt. The caller retains its local
    /// original and frozen descriptor until it validates the server outcome.
    /// There is no automatic retry, redirect, new identity or base64 media copy.
    func recordingResponse(recordingID: UUID, sessionID: UUID, file: URL? = nil,
                           descriptor: String? = nil) async throws -> (Data, HTTPURLResponse) {
        if let file {
            guard file.isFileURL, let descriptor, !descriptor.isEmpty,
                  descriptor.utf8.count <= 8192, descriptor.unicodeScalars.allSatisfy({ $0.isASCII }),
                  !descriptor.contains("\r"), !descriptor.contains("\n") else {
                throw JarvisAPIError(code: "recording_manifest_invalid", message: "The recording upload descriptor is invalid.")
            }
        } else if descriptor != nil {
            throw JarvisAPIError(code: "recording_manifest_invalid", message: "Reconciliation does not upload media.")
        }
        let suffix = file == nil ? "/reconcile" : "/content"
        var request = try authenticatedRequest(path: "/recordings/" + recordingID.uuidString.lowercased() + suffix,
            method: file == nil ? "POST" : "PUT",
            queryItems: [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())],
            accept: "application/json")
        request.timeoutInterval = 120
        if let descriptor {
            request.setValue("video/mp4", forHTTPHeaderField: "Content-Type")
            request.setValue(descriptor, forHTTPHeaderField: "X-Jarvis-Recording-Manifest")
        }
        let outgoing = request
        let delegate = JarvisRecordingNoRedirect()
        let result = try await withThrowingTaskGroup(of: (Data, URLResponse).self) { [session] group in
            group.addTask {
                if let file { return try await session.upload(for: outgoing, fromFile: file, delegate: delegate) }
                return try await session.data(for: outgoing, delegate: delegate)
            }
            group.addTask {
                try await Task.sleep(nanoseconds: 120_000_000_000)
                throw URLError(.timedOut)
            }
            defer { group.cancelAll() }
            guard let result = try await group.next() else { throw CancellationError() }
            return result
        }
        guard let response = result.1 as? HTTPURLResponse, response.url == outgoing.url,
              result.0.count <= 65_536 else {
            throw JarvisAPIError(code: "recording_response_invalid", message: "The recording outcome could not be verified.")
        }
        guard response.statusCode == 200 || response.statusCode == 202 else {
            throw decodeError(from: result.0, statusCode: response.statusCode,
                              retryAfter: Self.retryAfterDelay(from: response.value(forHTTPHeaderField: "Retry-After")))
        }
        return (result.0, response)
    }

    func recordingMetadataResponse(recordingID: UUID, sessionID: UUID, operation: String,
                                           body: Data? = nil) async throws -> (Data, HTTPURLResponse) {
        var request = try authenticatedRequest(path: "/recordings/" + recordingID.uuidString.lowercased() + "/" + operation,
            method: "POST", body: body,
            queryItems: [URLQueryItem(name: "session_id", value: sessionID.uuidString.lowercased())], accept: "application/json")
        request.timeoutInterval = 120
        let outgoing = request, delegate = JarvisRecordingNoRedirect()
        let result = try await withThrowingTaskGroup(of: (Data, URLResponse).self) { [session] group in
            group.addTask { try await session.data(for: outgoing, delegate: delegate) }
            group.addTask { try await Task.sleep(nanoseconds: 120_000_000_000); throw URLError(.timedOut) }
            defer { group.cancelAll() }
            guard let result = try await group.next() else { throw CancellationError() }
            return result
        }
        guard let response = result.1 as? HTTPURLResponse, response.url == outgoing.url,
              result.0.count <= 65_536 else { throw JarvisAPIError(code: "recording_response_invalid", message: "The recording outcome could not be verified.") }
        guard [200,202].contains(response.statusCode) else {
            throw decodeError(from: result.0, statusCode: response.statusCode,
                retryAfter: Self.retryAfterDelay(from: response.value(forHTTPHeaderField: "Retry-After")))
        }
        return (result.0, response)
    }

    private func authenticatedRequest(path: String, method: String, body: Data? = nil,
                                      queryItems: [URLQueryItem], accept: String) throws -> URLRequest {
        let resourceURL = baseURL.appendingPathComponent(
            path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        )
        var components = URLComponents(url: resourceURL, resolvingAgainstBaseURL: false)
        components?.queryItems = queryItems.isEmpty ? nil : queryItems
        // Query values use form decoding on the server: a literal plus must not
        // become a space in an opaque version or a user's search text.
        let encodedQuery = components?.percentEncodedQuery?.replacingOccurrences(of: "+", with: "%2B")
        components?.percentEncodedQuery = encodedQuery
        guard let url = components?.url else {
            throw JarvisAPIError(code: "invalid_request_url", message: "Jarvis request URL is invalid.")
        }
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        request.httpMethod = method
        request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
#if DEBUG
        JarvisReplayScope.applyRunHeader(evaluationRunID, to: &request)
#endif
        request.setValue(accept, forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        return request
    }

    // Metadata only: neither this parser nor responseData retries a request.
    static func retryAfterDelay(from header: String?, relativeTo now: Date = Date()) -> TimeInterval? {
        guard let header else { return nil }
        let value = header.trimmingCharacters(in: CharacterSet(charactersIn: " \t"))
        guard !value.isEmpty else { return nil }
        if value.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }) {
            guard let seconds = TimeInterval(value), seconds.isFinite else { return nil }
            return seconds
        }

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        guard let futureBoundary = calendar.date(byAdding: .year, value: 50, to: now),
              let centuryWindow = calendar.date(from: DateComponents(
                year: calendar.component(.year, from: futureBoundary) - 99, month: 1, day: 1
              )) else { return nil }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.isLenient = false
        formatter.twoDigitStartDate = centuryWindow
        // HTTP recipients also accept the two obsolete HTTP-date representations.
        let formats = [
            ("EEE, dd MMM yyyy HH:mm:ss 'GMT'", false),
            ("EEEE, dd-MMM-yy HH:mm:ss 'GMT'", true),
            ("EEE MMM dd HH:mm:ss yyyy", false),
            ("EEE MMM  d HH:mm:ss yyyy", false),
        ]
        for (format, hasShortYear) in formats {
            formatter.dateFormat = format
            guard var date = formatter.date(from: value) else { continue }
            if hasShortYear && date > futureBoundary {
                guard let corrected = calendar.date(byAdding: .year, value: -100, to: date) else { continue }
                date = corrected
            }
            // Reject partial parses, rolled-over dates, and non-HTTP date syntax.
            guard formatter.string(from: date) == value else { continue }
            return max(0, date.timeIntervalSince(now))
        }
        return nil
    }

    private func decodeError(from data: Data, statusCode: Int, retryAfter: TimeInterval?) -> JarvisAPIError {
        if let envelope = try? decoder.decode(JarvisErrorEnvelope.self, from: data) {
            return JarvisAPIError(code: envelope.error.code, message: envelope.error.message,
                                  statusCode: statusCode, retryAfter: retryAfter)
        }
        return JarvisAPIError(
            code: "http_\(statusCode)",
            message: "Jarvis request failed with status \(statusCode).",
            statusCode: statusCode,
            retryAfter: retryAfter
        )
    }
}

private final class JarvisRecordingNoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

private struct StartSessionRequest: Encodable {
    let title: String?
    let resumeActive: Bool

    enum CodingKeys: String, CodingKey {
        case title
        case resumeActive = "resume_active"
    }
}

private struct UpdateBehaviorInstructionsRequest: Encodable {
    let behaviorInstructions: String

    enum CodingKeys: String, CodingKey {
        case behaviorInstructions = "behavior_instructions"
    }
}

private struct JarvisErrorEnvelope: Decodable {
    let error: JarvisErrorBody
}

private struct JarvisErrorBody: Decodable {
    let code: String
    let message: String
}
