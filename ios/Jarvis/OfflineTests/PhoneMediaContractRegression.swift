import Foundation

enum PhoneMediaContractRegression {
    static func run() throws -> Int {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let id = UUID(), version = UUID().uuidString.lowercased(), size = 9_502_720
        let digest = String(repeating: "ab", count: 32)
        let inspection: [String: Any] = ["format": "jarvis.mp4_inspection.v1", "content_sha256": digest,
            "size_bytes": size, "duration_ms": 20_000, "width": 1920, "height": 1080,
            "video_codec": "h264", "audio_codec": "aac"]
        let evidence: [String: Any] = ["source_id": UUID().uuidString, "revision": 1, "generation": 1,
            "content_digest": "source-digest", "source_type": "file", "trust": "user",
            "time_basis": "recorded", "recorded_at": "2026-11-02T00:00:00Z", "is_current": true,
            "ingestion_status": "ready"]
        let payload: [String: Any] = ["id": id.uuidString.lowercased(), "filename": "clip.mp4",
            "content_type": "video/mp4", "size_bytes": size, "description": "Clip", "source": "retained",
            "created_at": "2026-11-02T00:00:00Z", "memory_source": evidence, "version_id": version,
            "content_sha256": digest, "transfer_limit_bytes": 32 * 1024 * 1024, "media_inspection": inspection]
        func decode(_ value: [String: Any]) throws -> JarvisMediaFile {
            try JSONDecoder().decode(JarvisMediaFileResponse.self,
                from: JSONSerialization.data(withJSONObject: ["file": value])).file
        }
        func validate(_ media: JarvisMediaFile, limit: Int = 32 * 1024 * 1024, duration: Int = 30_000) throws -> Int {
            try media.validate(fileID: id, versionID: version, maximumByteCount: limit,
                               maximumDurationMilliseconds: duration)
        }
        let media = try decode(payload)
        check(try validate(media) == 32 * 1024 * 1024, "Exact media envelope accepts the qualified output bounds")
        check(try validate(media, limit: 16 * 1024 * 1024) == 16 * 1024 * 1024, "Client limit can only narrow server policy")
        var lower = payload; lower["transfer_limit_bytes"] = 12 * 1024 * 1024
        check(try validate(decode(lower)) == 12 * 1024 * 1024, "Lower deployed limit wins")
        var silent = payload; var silentInspection = inspection; silentInspection["audio_codec"] = NSNull()
        silent["media_inspection"] = silentInspection
        check(try validate(decode(silent)) > 0, "Silent MP4 is accepted")

        var bad: [[String: Any]] = []
        for (key, value) in [("id", UUID().uuidString as Any), ("version_id", "other" as Any),
                             ("content_type", "text/plain" as Any), ("content_sha256", digest.uppercased() as Any),
                             ("transfer_limit_bytes", size - 1 as Any), ("size_bytes", 0 as Any)] {
            var copy = payload; copy[key] = value; bad.append(copy)
        }
        var revoked = payload; var oldEvidence = evidence; oldEvidence["is_current"] = false
        revoked["memory_source"] = oldEvidence; bad.append(revoked)
        for (key, value) in [("format", "future" as Any), ("content_sha256", String(repeating: "0", count: 64) as Any),
                             ("size_bytes", size - 1 as Any), ("duration_ms", 30_001 as Any),
                             ("width", 0 as Any), ("video_codec", "hevc" as Any), ("audio_codec", "other" as Any)] {
            var copy = payload; var changed = inspection; changed[key] = value
            copy["media_inspection"] = changed; bad.append(copy)
        }
        for value in bad {
            do { _ = try validate(decode(value)); preconditionFailure("Invalid exact media metadata accepted") }
            catch is JarvisAPIError { check(true, "Mismatched metadata is rejected before transfer") }
        }
        do { _ = try validate(media, limit: 8 * 1024 * 1024); preconditionFailure("Client cap exceeded") }
        catch is JarvisAPIError { check(true, "Media cannot exceed the supplied client policy") }

        let url = URL(string: "https://example.invalid/files/\(id)/content?version_id=\(version)")!
        let headers = ["Content-Length": String(size), "Content-Type": "video/mp4", "X-Jarvis-Version-Id": version,
            "X-Content-SHA256": digest, "ETag": "\"\(digest)\"", "Cache-Control": "private, no-store", "Vary": "Authorization"]
        let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: headers)!
        try media.validateResponse(response, requestedURL: url); check(true, "Exact response headers accepted")
        for (key, value) in [("Content-Length", String(size - 1)), ("Content-Type", "application/json"),
                             ("X-Jarvis-Version-Id", "other"), ("X-Content-SHA256", "other"),
                             ("ETag", "W/\"\(digest)\""), ("Cache-Control", "public"), ("Vary", "Accept"),
                             ("Content-Encoding", "gzip"), ("Content-Range", "bytes 0-99/100")] {
            var changed = headers; changed[key] = value
            do {
                try media.validateResponse(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: changed)!, requestedURL: url)
                preconditionFailure("Invalid response header accepted")
            } catch is JarvisAPIError { check(true, "Mismatched response rejected") }
        }
        for code in [206, 401, 403, 404, 500] {
            do {
                try media.validateResponse(HTTPURLResponse(url: url, statusCode: code, httpVersion: nil, headerFields: headers)!, requestedURL: url)
                preconditionFailure("Non-full response accepted")
            } catch is JarvisAPIError { check(true, "Only a complete successful response is usable") }
        }
        do { try media.validateResponse(response, requestedURL: URL(string: "https://example.invalid/other")!); preconditionFailure("Redirect accepted") }
        catch is JarvisAPIError { check(true, "The final response must be from the exact requested URL") }
        // The old document envelope still decodes without media fields and retains its byte ceiling.
        var document = payload
        for key in ["version_id", "content_sha256", "transfer_limit_bytes", "media_inspection"] { document.removeValue(forKey: key) }
        document["content_type"] = "text/plain"
        let old = try JSONDecoder().decode(JarvisRetainedFileResponse.self,
            from: JSONSerialization.data(withJSONObject: ["file": document])).file
        do { try old.validateContent(Data(), contentType: "text/plain"); preconditionFailure("Document ceiling widened") }
        catch is JarvisAPIError { check(true, "Legacy envelope and document rejection stay intact") }
        return checks
    }
}
