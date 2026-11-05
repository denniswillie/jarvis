import Foundation

/// Serial, bounded attempts over the existing durable pending-original store.
/// This owns no recording/task authority and never removes local media. Recovery
/// reconciles the same identity before offering its frozen descriptor again.
@MainActor
final class JarvisRecordingTransfers {
    typealias Entry = JarvisRecordingPendingStore.Entry
    typealias Request = @MainActor (Entry, Bool) async throws -> (Data, Int)
    enum Outcome { case saved(UUID), pending, unavailable }
    enum Failure: Error { case invalidResponse }

    private let store: JarvisRecordingPendingStore
    private let scope: JarvisRecordingPendingStore.Scope
    private let request: Request
    private let current: @MainActor () -> Bool
    private let changed: @MainActor (UUID, Outcome) -> Void
    private var jobs: [UUID: Task<Void, Never>] = [:]
    private var tail: Task<Void, Never>?
    private var retired = false

    init(store: JarvisRecordingPendingStore, scope: JarvisRecordingPendingStore.Scope,
         current: @escaping @MainActor () -> Bool, request: @escaping Request,
         changed: @escaping @MainActor (UUID, Outcome) -> Void) {
        self.store = store; self.scope = scope; self.current = current
        self.request = request; self.changed = changed
    }

    /// Called with entries returned by retain()/pending() for this exact scope.
    /// `recovering` must be true for persisted entries after an unknown outcome.
    func submit(_ entry: Entry, recovering: Bool) {
        guard !retired, current(), jobs[entry.recordingID] == nil,
              jobs.count < JarvisRecordingPendingStore.maximumEntries else { return }
        if let saved = entry.confirmedLinkage {
            changed(entry.recordingID, .saved(saved.fileID)); return
        }
        let previous = tail, id = entry.recordingID
        let task = Task { [self] in
            defer { jobs[id] = nil }
            await previous?.value
            guard !Task.isCancelled, !retired, current() else { return }
            do {
                if recovering {
                    let reply = try await request(entry, true)
                    let result = try await consume(reply, entry: entry, reconcile: true)
                    if result { return }
                }
                // A successful reconciliation saying not_saved permits one
                // same-identity upload, never a replacement recording ID.
                guard !Task.isCancelled, !retired, current() else { return }
                _ = try await consume(try await request(entry, false), entry: entry, reconcile: false)
            } catch {
                if !retired, current() { changed(id, .unavailable) }
            }
        }
        jobs[id] = task; tail = task
    }

    func retire() {
        retired = true
        for job in jobs.values { job.cancel() }
        // Neither cancellation nor a lost acknowledgement deletes originals.
    }

    func drained() async { await tail?.value }

    /// True means no further upload is appropriate during this attempt.
    private func consume(_ reply: (Data, Int), entry: Entry, reconcile: Bool) async throws -> Bool {
        guard reply.0.count <= 65_536, [200, 202].contains(reply.1),
              let value = try JSONSerialization.jsonObject(with: reply.0) as? [String: Any],
              let outcome = value["outcome"] as? String else { throw Failure.invalidResponse }
        if outcome == "saved" {
            guard reply.1 == 200 else { throw Failure.invalidResponse }
            let confirmed = try await store.confirmSaved(receiptData: reply.0,
                recordingID: entry.recordingID, scope: scope)
            guard let saved = confirmed.confirmedLinkage else { throw Failure.invalidResponse }
            if !retired, current() { changed(entry.recordingID, .saved(saved.fileID)) }
            return true
        }
        guard reply.1 == 202, let rawID = value["recording_id"] as? String,
              UUID(uuidString: rawID) == entry.recordingID else { throw Failure.invalidResponse }
        if outcome == "publication_unresolved" {
            if !retired, current() { changed(entry.recordingID, .pending) }
            return true
        }
        guard reconcile, outcome == "not_saved", value["finalization_bound"] is Bool else {
            throw Failure.invalidResponse
        }
        return false
    }
}

/// Successful uninterrupted upload path. Acknowledged chunks are released;
/// no whole local original, checkpoint, retry journal or second recording ID.
actor JarvisRecordingStreamingUpload {
    enum Failure: Error { case invalidTarget, invalidBinding, unsupportedServer, invalidResponse, invalidOffset, alreadyStarted }
    struct Target: Sendable {
        let recordingID: UUID
        let sessionID: UUID
        let uploadID: String
        let endpoint: URL
        let bucket: String
        let object: String
        let signature: String
        let maximumBytes: Int
        let chunkBytes: Int
        func validate() throws {
            guard endpoint.scheme == "https", endpoint.host != nil,
                  endpoint.user == nil, endpoint.password == nil, endpoint.fragment == nil,
                  endpoint.query == nil, endpoint.path == "/storage/v1/upload/resumable/sign",
                  !uploadID.isEmpty, uploadID.utf8.count <= 256,
                  !bucket.isEmpty, !object.isEmpty, bucket.utf8.count <= 256, object.utf8.count <= 2048,
                  !signature.isEmpty, signature.utf8.count <= 8192,
                  !signature.contains("\r"), !signature.contains("\n"),
                  (1...JarvisRecordingWriter.maximumFileBytes).contains(maximumBytes),
                  chunkBytes == 6 * 1024 * 1024 else { throw Failure.invalidTarget }
        }
        var metadata: String {
            [("bucketName", bucket), ("objectName", object), ("contentType", "video/mp4")]
                .map { $0.0 + " " + Data($0.1.utf8).base64EncodedString() }.joined(separator: ",")
        }
        func accepts(_ url: URL) -> Bool {
            url.scheme == endpoint.scheme && url.host == endpoint.host && url.port == endpoint.port
                && url.user == nil && url.password == nil && url.fragment == nil && url.query == nil
                && url.path.hasPrefix(endpoint.path + "/") && url.path.count > endpoint.path.count + 1
        }
    }

    struct Reply: Sendable {
        let status: Int
        let headers: [String: String]
        let bodyBytes: Int
    }
    typealias Send = @Sendable (URLRequest, Data?) async throws -> Reply
    private let target: Target
    private let send: Send
    private var started = false

    init(target: Target, send: @escaping Send) throws {
        try target.validate(); self.target = target; self.send = send
    }
    func run(stream: JarvisRecordingByteStream) async throws -> JarvisRecordingByteStream.Summary {
        guard !started else { throw Failure.alreadyStarted }
        started = true
        do {
            let options = try await request("OPTIONS", at: target.endpoint)
            guard [200,204].contains(options.status) else { throw Failure.unsupportedServer }
            if let raw = options.headers["tus-extension"] {
                let values = Set(raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
                guard values.isSuperset(of: ["creation", "creation-defer-length"]) else { throw Failure.unsupportedServer }
            }
            if let raw = options.headers["tus-version"] {
                guard raw.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) }).contains("1.0.0") else { throw Failure.unsupportedServer }
            }
            if let raw = options.headers["tus-max-size"] {
                guard let size = Int(raw), size >= target.maximumBytes else { throw Failure.unsupportedServer }
            }
            let created = try await request("POST", at: target.endpoint,
                headers: ["Upload-Defer-Length": "1", "Upload-Metadata": target.metadata], data: Data())
            guard created.status == 201, let location = created.headers["location"],
                  let resource = URL(string: location, relativeTo: target.endpoint)?.absoluteURL,
                  target.accepts(resource) else { throw Failure.invalidResponse }
            let head = try await request("HEAD", at: resource)
            guard [200,204].contains(head.status), head.headers["upload-offset"] == "0",
                  head.headers["upload-defer-length"] == "1", head.headers["upload-length"] == nil,
                  Self.metadataMatches(head.headers["upload-metadata"], target: target) else { throw Failure.invalidResponse }
            var offset = 0, declaredLength: Int?
            while true {
                try Task.checkCancellation()
                switch try await stream.next() {
                case .chunk(let chunk):
                    guard chunk.offset == offset, !chunk.data.isEmpty, chunk.data.count <= target.chunkBytes,
                          chunk.data.count <= target.maximumBytes - offset,
                          chunk.finalByteCount != nil || chunk.data.count == target.chunkBytes else { throw Failure.invalidOffset }
                    var headers = ["Content-Type": "application/offset+octet-stream", "Upload-Offset": String(offset)]
                    if let final = chunk.finalByteCount {
                        guard final >= offset + chunk.data.count, final <= target.maximumBytes,
                              declaredLength == nil || declaredLength == final else { throw Failure.invalidOffset }
                        if declaredLength == nil { headers["Upload-Length"] = String(final) }
                    }
                    let reply = try await request("PATCH", at: resource, headers: headers, data: chunk.data)
                    guard reply.status == 204, reply.headers["upload-offset"].flatMap(Int.init) == offset + chunk.data.count else { throw Failure.invalidOffset }
                    offset += chunk.data.count
                    if let final = chunk.finalByteCount { declaredLength = final }
                    try stream.acknowledge(chunk)
                case .finished(let summary):
                    guard summary.byteCount == offset, offset > 0, offset <= target.maximumBytes,
                          declaredLength == nil || declaredLength == offset else { throw Failure.invalidOffset }
                    if declaredLength == nil {
                        let reply = try await request("PATCH", at: resource,
                            headers: ["Content-Type": "application/offset+octet-stream", "Upload-Offset": String(offset), "Upload-Length": String(offset)], data: Data())
                        guard reply.status == 204, reply.headers["upload-offset"].flatMap(Int.init) == offset else { throw Failure.invalidOffset }
                    }
                    return summary // Storage transport only; backend must verify/publish.
                }
            }
        } catch { stream.fail(); throw error }
    }

    private func request(_ method: String, at url: URL, headers: [String:String] = [:], data: Data? = nil) async throws -> Reply {
        try Task.checkCancellation()
        guard url == target.endpoint || target.accepts(url) else { throw Failure.invalidTarget }
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
        request.httpMethod = method
        request.setValue(target.signature, forHTTPHeaderField: "x-signature")
        if method != "OPTIONS" { request.setValue("1.0.0", forHTTPHeaderField: "Tus-Resumable") }
        for (key,value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        if let data { request.setValue(String(data.count), forHTTPHeaderField: "Content-Length") }
        let reply = try await send(request, data)
        try Task.checkCancellation()
        guard (0...65_536).contains(reply.bodyBytes), !(300...399).contains(reply.status),
              method == "OPTIONS" || reply.headers["tus-resumable"] == "1.0.0" else { throw Failure.invalidResponse }
        return reply
    }
    private static func metadataMatches(_ raw: String?, target: Target) -> Bool {
        guard let raw, raw.utf8.count <= 8192 else { return false }
        var fields: [String:String] = [:]
        for pair in raw.split(separator: ",") {
            let parts = pair.trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 1)
            guard parts.count == 2, fields[String(parts[0])] == nil,
                  let data = Data(base64Encoded: String(parts[1])), let text = String(data: data, encoding: .utf8) else { return false }
            fields[String(parts[0])] = text
        }
        return fields["bucketName"] == target.bucket && fields["objectName"] == target.object && fields["contentType"] == "video/mp4"
    }

}

/// Only a single bounded chunk is held by URLSession. This session receives the
/// storage capability, never Jarvis's bearer token, cookies or redirect rights.
final class JarvisRecordingStorageTransport: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let session: URLSession
    override init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpShouldSetCookies = false
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        configuration.httpMaximumConnectionsPerHost = 1
        session = URLSession(configuration: configuration)
        super.init()
    }
    func send(_ request: URLRequest, data: Data?) async throws -> JarvisRecordingStreamingUpload.Reply {
        let result: (Data, URLResponse)
        if let data { result = try await session.upload(for: request, from: data, delegate: self) }
        else { result = try await session.data(for: request, delegate: self) }
        guard let response = result.1 as? HTTPURLResponse, response.url == request.url else {
            throw JarvisRecordingStreamingUpload.Failure.invalidResponse
        }
        var headers: [String:String] = [:]
        for (key,value) in response.allHeaderFields {
            guard let key = key as? String else { continue }
            headers[key.lowercased()] = String(describing: value)
        }
        return .init(status: response.statusCode, headers: headers, bodyBytes: result.0.count)
    }
    func close() { session.invalidateAndCancel() }
    deinit { session.invalidateAndCancel() }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
