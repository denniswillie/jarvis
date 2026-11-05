import CryptoKit
import Foundation

@main
enum PhoneMediaJournalRegression {
    @MainActor
    static func main() async throws {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let reader = try FileHandle(forReadingFrom: source)
        var hash = SHA256(), size = 0
        while let chunk = try reader.read(upToCount: 65_536), !chunk.isEmpty { hash.update(data: chunk); size += chunk.count }
        try reader.close()
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        precondition(size > 8 * 1024 * 1024
                     && size <= JarvisWorkspaceRequest.MediaTransferPolicy.output.maximumByteCount)
        let admittedLimit = max(12 * 1024 * 1024, size)
        let fileID = UUID(), version = UUID().uuidString, session = UUID()
        let payload: [String: Any] = ["request_id": UUID().uuidString, "action": "download", "artifact_kind": "file",
            "artifact_id": fileID.uuidString, "file_id": fileID.uuidString, "version_id": version,
            "content_ready": true, "byte_count": size, "content_sha256": digest, "state": "pending",
            "created_at": "2026-11-02T00:00:00Z", "expires_at": "2026-11-02T01:00:00Z",
            "transfer_policy": "mp4_output_v1", "transfer_limit_bytes": admittedLimit]
        func decode(_ value: [String: Any]) throws -> JarvisWorkspaceRequest {
            try JSONDecoder().decode(JarvisWorkspaceRequest.self, from: JSONSerialization.data(withJSONObject: value))
        }
        let request = try decode(payload)
        check(try request.validatedTransferByteLimit() == admittedLimit && request.transferPolicy == .output,
              "Prepared media offer retains its effective persisted limit")
        var legacy = payload
        legacy.removeValue(forKey: "transfer_policy"); legacy.removeValue(forKey: "transfer_limit_bytes")
        legacy["byte_count"] = 100
        let document = try decode(legacy)
        let encodedDocument = try JSONSerialization.jsonObject(with: JSONEncoder().encode(document)) as! [String: Any]
        check(try document.validatedTransferByteLimit() == 8 * 1024 * 1024 && encodedDocument["transfer_policy"] == nil
              && encodedDocument["transfer_limit_bytes"] == nil, "Legacy envelopes retain their old ceiling and absent policy fields")
        for (key, value) in [("transfer_policy", "unknown" as Any), ("transfer_limit_bytes", 500_000_001 as Any),
                             ("transfer_limit_bytes", size - 1 as Any), ("transfer_limit_bytes", NSNull() as Any),
                             ("transfer_policy", NSNull() as Any), ("artifact_kind", "note" as Any),
                             ("action", "open" as Any), ("content_ready", false as Any), ("byte_count", 0 as Any)] {
            var changed = payload; changed[key] = value
            do { _ = try decode(changed).validatedTransferByteLimit(); preconditionFailure("Invalid transfer policy accepted") }
            catch { check(true, "Malformed or widened media policy is rejected") }
        }
        var changed = request; changed.transferLimitBytes = 16 * 1024 * 1024
        check(!request.hasSameTarget(as: changed), "Claim target equality includes the trusted effective bound")
        var capture = payload; capture["transfer_policy"] = "mp4_capture_v1"; capture["transfer_limit_bytes"] = 64 * 1024 * 1024
        check(try decode(capture).validatedTransferByteLimit() == 64 * 1024 * 1024, "Capture ceiling stays separate from output ceiling")
        capture["transfer_limit_bytes"] = 1024 * 1024 * 1024
        check(try decode(capture).validatedTransferByteLimit() == 1024 * 1024 * 1024,
              "Five-minute capture policy supports its finite byte ceiling")
        capture["transfer_limit_bytes"] = 1024 * 1024 * 1024 + 1
        do { _ = try decode(capture).validatedTransferByteLimit(); preconditionFailure("Capture bound widened") }
        catch { check(true, "Capture policy cannot exceed its approved ceiling") }

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-media-journal-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let context = JarvisAuthenticatedContext(ownerID: "fixture-owner-A",
            backendNamespace: "jarvis-storage-v1:" + String(repeating: "a", count: 64), sessionID: session)
        let owner = try JarvisWorkspaceLocalStore.ownerKey(for: context), access = JarvisWorkspaceLocalAccess()
        // Use the ordinary app's factory, overriding only the test directory.
        let store = try JarvisWorkspaceLocalStore.verifiedAccountJournal(for: context, root: root)
        check(store.allowsFileBytes, "Default verified-account composition enables bounded file bytes")
        let invalidContext = JarvisAuthenticatedContext(ownerID: "", backendNamespace: context.backendNamespace, sessionID: session)
        do {
            _ = try JarvisWorkspaceLocalStore.verifiedAccountJournal(for: invalidContext, root: root)
            preconditionFailure("Unverified owner entered the default composition")
        } catch is JarvisWorkspaceError { check(true, "Default composition rejects invalid owner context") }
        _ = try await store.prepareClaim(request, sessionID: session, access: access)
        do { _ = try await store.prepareClaim(changed, sessionID: session, access: access); preconditionFailure("Changed policy reused a journal") }
        catch is JarvisWorkspaceError { check(true, "A request cannot change policy after journal preparation") }
        try await store.confirmClaim(id: request.id, access: access)
        try await store.beginEffect(id: request.id, access: access)
        let ack = try await store.saveDownloadFile(id: request.id, source: source, filename: "Folder/clip.mp4",
                                                  contentType: "video/mp4", access: access)
        check(ack.state == .savedLocal && ack.localReceipt?.fileID == fileID && ack.localReceipt?.versionID == version
              && ack.localReceipt?.byteCount == size && ack.localReceipt?.contentSHA256 == digest
              && ack.localReceipt?.destination == "app_storage", "Durable success has the unchanged exact saved_local receipt")
        let receiptJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(ack.localReceipt!)) as! [String: Any]
        check(Set(receiptJSON.keys) == ["file_id", "version_id", "byte_count", "content_sha256", "destination"],
              "Transfer policy does not invent a new receipt schema")
        let local = try await store.offlineFile(id: request.id, access: access)
        check(local.byteCount == size && local.contentSHA256 == digest && local.filename == "Folder-clip.mp4"
              && local.contentType == "video/mp4" && local.url != source, "Offline reopen streams and verifies the durable copy")
        check(FileManager.default.fileExists(atPath: source.path), "Saving never moves the source file")
        do { _ = try await store.offlineContent(id: request.id, access: access); preconditionFailure("Media exposed through Data API") }
        catch is JarvisWorkspaceError { check(true, "Media cannot fall back to the small-document Data buffer") }
        let reopened = try JarvisWorkspaceLocalStore(ownerKey: owner, root: root, allowsFileBytes: true)
        check(try await reopened.offlineFile(id: request.id, access: JarvisWorkspaceLocalAccess()).contentSHA256 == digest,
              "A fresh store instance reopens the same exact journal and bytes")
        let other = try JarvisWorkspaceLocalStore(ownerKey: "fixture-owner-B", root: root, allowsFileBytes: true)
        check(try await other.entries().isEmpty, "Another owner has a separate journal")
        do { _ = try await other.offlineFile(id: request.id, access: JarvisWorkspaceLocalAccess()); preconditionFailure("Cross-owner copy exposed") }
        catch is JarvisWorkspaceError { check(true, "Another owner cannot open the saved request") }
        access.invalidate()
        do { _ = try await store.offlineFile(id: request.id, access: access); preconditionFailure("Retired access reopened bytes") }
        catch is CancellationError { check(FileManager.default.fileExists(atPath: local.url.path), "Account lock retains durable bytes without exposing them") }

        // Reproduce the journal boundary after final-file adoption but before completion.
        var interrupted = try await reopened.entry(id: request.id)!
        interrupted.phase = .effectStarted; interrupted.acknowledgement = nil; interrupted.relativePath = nil
        let directory = root.appendingPathComponent(JarvisWorkspaceLocalStore.digest(Data(owner.utf8)))
        let journal = directory.appendingPathComponent(request.id.uuidString.lowercased() + ".json")
        try JSONEncoder().encode(interrupted).write(to: journal, options: .atomic)
        let recovery = try JarvisWorkspaceLocalStore(ownerKey: owner, root: root, allowsFileBytes: true)
        let recovered = try await recovery.reconcileDownload(id: request.id, access: JarvisWorkspaceLocalAccess())
        check(recovered == ack, "Crash reconciliation verifies the existing file and restores the exact receipt without downloading")

        let gated = try JarvisWorkspaceLocalStore(ownerKey: owner, root: root, allowsFileBytes: false)
        do { _ = try await gated.offlineFile(id: request.id, access: JarvisWorkspaceLocalAccess()); preconditionFailure("Closed production gate exposed bytes") }
        catch is JarvisWorkspaceError { check(true, "Production byte gate still blocks media reads") }
        var wrong = payload; wrong["request_id"] = UUID().uuidString; wrong["content_sha256"] = String(repeating: "0", count: 64)
        let mismatch = try decode(wrong), validAccess = JarvisWorkspaceLocalAccess()
        _ = try await recovery.prepareClaim(mismatch, sessionID: session, access: validAccess)
        try await recovery.confirmClaim(id: mismatch.id, access: validAccess)
        try await recovery.beginEffect(id: mismatch.id, access: validAccess)
        do {
            _ = try await recovery.saveDownloadFile(id: mismatch.id, source: source, filename: "clip.mp4", contentType: "video/mp4", access: validAccess)
            preconditionFailure("Wrong hash saved")
        } catch is JarvisWorkspaceError {
            let failed = try await recovery.entry(id: mismatch.id)!
            check(failed.acknowledgement == nil && failed.phase == .effectStarted, "Failed verification cannot record saved_local")
            let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            check(!names.contains(where: { $0.hasSuffix(".partial") }) && !names.contains(mismatch.id.uuidString.lowercased() + ".download"),
                  "Rejected bytes leave neither partial nor final download")
        }
        let writer = try FileHandle(forWritingTo: local.url)
        try writer.write(contentsOf: Data([255])); try writer.close()
        do { _ = try await recovery.offlineFile(id: request.id, access: validAccess); preconditionFailure("Corrupted durable bytes reopened") }
        catch is JarvisWorkspaceError { check(true, "Each reopen detects actual on-disk corruption") }
        print("Phone media journal regressions passed: \(checks) checks; source_bytes=\(size); isolated local storage only")
    }
}
