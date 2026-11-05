@preconcurrency import AVFoundation
import CryptoKit
import Foundation
import Darwin

actor SignedFixtureStorage {
    let target: JarvisRecordingStreamingUpload.Target
    private var bytes = Data()
    private var length: Int?
    private var patchSizes: [Int] = []
    private var creations = 0
    init(target: JarvisRecordingStreamingUpload.Target) { self.target = target }
    func state() -> (Data, Int?, [Int], Int) { (bytes, length, patchSizes, creations) }
    func send(_ request: URLRequest, _ data: Data?) throws -> JarvisRecordingStreamingUpload.Reply {
        try require(request.value(forHTTPHeaderField: "Authorization") == nil, "bearer sent to storage")
        try require(request.value(forHTTPHeaderField: "x-signature") == target.signature, "storage signature mismatch")
        let expectedPath = ["OPTIONS", "POST"].contains(request.httpMethod!) ? target.endpoint.path : target.endpoint.path + "/fixture"
        try require(request.url?.path == expectedPath, "Signed endpoint/Location path was not preserved")
        var headers = ["tus-resumable": "1.0.0"]
        switch request.httpMethod! {
        case "OPTIONS": return .init(status: 200, headers: [:], bodyBytes: 0) // observed hosted discovery shape
        case "POST":
            try require(request.value(forHTTPHeaderField: "Upload-Defer-Length") == "1", "length not deferred")
            try require(data?.count == 0, "creation sent unexpected bytes")
            creations += 1; headers["location"] = target.endpoint.absoluteString + "/fixture"
            return .init(status: 201, headers: headers, bodyBytes: 0)
        case "HEAD":
            headers["upload-offset"] = String(bytes.count); headers["upload-defer-length"] = "1"
            headers["upload-metadata"] = target.metadata
            return .init(status: 200, headers: headers, bodyBytes: 0)
        case "PATCH":
            try require(request.url?.host == target.endpoint.host, "cross origin")
            try require(request.value(forHTTPHeaderField: "Upload-Offset").flatMap(Int.init) == bytes.count, "offset mismatch")
            if let final = request.value(forHTTPHeaderField: "Upload-Length").flatMap(Int.init) {
                try require(length == nil || length == final, "changed final length"); length = final
            }
            guard let data else { throw ProofError.failed("missing payload") }
            try require(data.count <= JarvisRecordingByteStream.chunkBytes, "oversized transfer buffer")
            if data.count < JarvisRecordingByteStream.chunkBytes { try require(length != nil, "short chunk sent before final length") }
            bytes.append(data); patchSizes.append(data.count)
            headers["upload-offset"] = String(bytes.count)
            return .init(status: 204, headers: headers, bodyBytes: 0)
        default: throw ProofError.failed("unexpected HTTP method")
        }
    }
}

enum ProofError: Error { case failed(String) }
func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw ProofError.failed(message) }
}
@main struct JarvisRecordingSignedUploadChecks {
    static func target(_ endpoint: String = "https://fixture.invalid/storage/v1/upload/resumable/sign") -> JarvisRecordingStreamingUpload.Target {
        .init(recordingID: UUID(), sessionID: UUID(), uploadID: UUID().uuidString,
            endpoint: URL(string: endpoint)!, bucket: "private-fixture", object: "test/recording.mp4",
            signature: "synthetic-capability", maximumBytes: 256 * 1024 * 1024, chunkBytes: JarvisRecordingByteStream.chunkBytes)
    }
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]), bound = target()
        try bound.validate()
        for endpoint in ["https://fixture.invalid/storage/v1/upload/resumable",
                         "https://fixture.invalid/storage/v1/upload/resumable/sign?token=wrong",
                         "http://fixture.invalid/storage/v1/upload/resumable/sign"] {
            do { try target(endpoint).validate(); throw ProofError.failed("Unsafe/unsigned endpoint accepted") }
            catch JarvisRecordingStreamingUpload.Failure.invalidTarget { }
        }
        try require(bound.accepts(URL(string: bound.endpoint.absoluteString + "/fixture")!), "Signed Location rejected")
        for location in ["https://elsewhere.invalid/storage/v1/upload/resumable/sign/fixture",
                         "https://fixture.invalid/storage/v1/upload/resumable/fixture",
                         "https://fixture.invalid/storage/v1/upload/resumable/sign/fixture?token=wrong",
                         "https://fixture.invalid/storage/v1/upload/resumable/sign"] {
            try require(!bound.accepts(URL(string: location)!), "Wrong Location accepted")
        }
        let stream = JarvisRecordingByteStream(), server = SignedFixtureStorage(target: bound)
        let uploader = try JarvisRecordingStreamingUpload(target: bound, send: { try await server.send($0, $1) })
        let task = Task { try await uploader.run(stream: stream) }
        let first = Data(repeating: 0x41, count: JarvisRecordingByteStream.chunkBytes)
        try require(stream.offer(first), "First chunk rejected")
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while await server.state().0.isEmpty {
            try require(ContinuousClock.now < deadline, "First signed chunk did not upload")
            try await Task.sleep(for: .milliseconds(10))
        }
        let early = await server.state()
        try require(early.0.count == first.count && early.1 == nil, "Pre-finalization signed upload not observed")
        let last = Data([0x42, 0x43]); try require(stream.offer(last), "Tail rejected")
        var hash = SHA256(); hash.update(data: first); hash.update(data: last)
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let summary = JarvisRecordingByteStream.Summary(generationID: UUID(), byteCount: first.count + last.count, contentSHA256: digest)
        try require(stream.finish(summary), "Stream finish rejected")
        let uploaded = try await task.value, received = await server.state()
        try require(uploaded.byteCount == summary.byteCount && received.1 == summary.byteCount
            && received.2 == [first.count, last.count] && received.3 == 1, "Signed resource finalization mismatch")
        let result: [String: Any] = ["passed": true, "endpoint": bound.endpoint.path,
            "preFinishUploadedBytes": early.0.count, "finalBytes": uploaded.byteCount,
            "creations": received.3, "realRemoteService": false, "captureOrEncoderRerun": false]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("proof.json"))
        print("Signed TUS path checks passed: exact sign POST/HEAD/PATCH,6MiB before finish, one resource, no bearer")
    }
}
