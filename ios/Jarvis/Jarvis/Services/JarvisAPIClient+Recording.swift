import Foundation

extension JarvisAPIClient {
    /// Exact signed storage capability; only authenticated Jarvis authorizes its
    /// target. The Jarvis bearer remains on this metadata request.
    func recordingUploadTarget(command: JarvisRecordingControl.Command) async throws -> JarvisRecordingStreamingUpload.Target {
        struct Wire: Decodable {
            let recording_id: UUID
            let session_id: UUID
            let upload_id: String
            let bucket_name: String
            let object_path: String
            let resumable_url: URL
            let request_headers: [String: String]
            let credential_expires_at: String
            let max_bytes: Int
            let chunk_bytes: Int
        }
        let response = try await recordingMetadataResponse(recordingID: command.recordingID,
            sessionID: command.sessionID, operation: "upload-target")
        guard response.1.statusCode == 200 else { throw JarvisRecordingStreamingUpload.Failure.invalidResponse }
        let wire = try JSONDecoder().decode(Wire.self, from: response.0)
        let date = ISO8601DateFormatter()
        date.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var expires = date.date(from: wire.credential_expires_at)
        if expires == nil { date.formatOptions = [.withInternetDateTime]; expires = date.date(from: wire.credential_expires_at) }
        guard wire.recording_id == command.recordingID, wire.session_id == command.sessionID,
              wire.max_bytes == command.maxBytes, Set(wire.request_headers.keys) == ["x-signature"],
              let signature = wire.request_headers["x-signature"], let expires, expires > Date() else {
            throw JarvisRecordingStreamingUpload.Failure.invalidBinding
        }
        let target = JarvisRecordingStreamingUpload.Target(recordingID: wire.recording_id, sessionID: wire.session_id,
            uploadID: wire.upload_id, endpoint: wire.resumable_url, bucket: wire.bucket_name, object: wire.object_path,
            signature: signature, maximumBytes: wire.max_bytes, chunkBytes: wire.chunk_bytes)
        try target.validate()
        return target
    }

    /// No media body: the backend verifies the exact uploaded object, publishes
    /// through Files and returns its real saved receipt/download request.
    func finalizeRecordingUpload(recordingID: UUID, sessionID: UUID, uploadID: String,
                                 descriptor: String) async throws -> (Data, HTTPURLResponse) {
        guard !uploadID.isEmpty, uploadID.utf8.count <= 256, !descriptor.isEmpty,
              descriptor.utf8.count <= 8192,
              let value = try JSONSerialization.jsonObject(with: Data(descriptor.utf8)) as? [String: Any] else {
            throw JarvisRecordingStreamingUpload.Failure.invalidBinding
        }
        let body = try JSONSerialization.data(withJSONObject: ["upload_id": uploadID, "descriptor": value], options: [.sortedKeys])
        return try await recordingMetadataResponse(recordingID: recordingID, sessionID: sessionID,
            operation: "finalize-upload", body: body)
    }

}
