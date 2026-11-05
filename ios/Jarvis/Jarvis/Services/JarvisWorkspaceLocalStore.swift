import CryptoKit
import Foundation
import Darwin

/// A VM lifetime fence, never an account identity or server permission. Invalidate
/// synchronously before replacing the model so a queued actor call cannot begin
/// another local operation. An already-started synchronous operation may finish
/// its journal; the retired VM must not expose/ACK that result.
final class JarvisWorkspaceLocalAccess: @unchecked Sendable {
    private let lock = NSLock()
    private var active = true
    func invalidate() { lock.lock(); defer { lock.unlock() }; active = false }
    func check() throws {
        lock.lock(); defer { lock.unlock() }
        guard active else { throw CancellationError() }
    }
}

/// Small, account-scoped delivery journal and deliberately saved offline copies.
/// A claim/result is persisted before the corresponding local effect/server ACK.
actor JarvisWorkspaceLocalStore {
    nonisolated let allowsFileBytes: Bool
    struct Entry: Codable, Equatable, Identifiable {
        enum Phase: String, Codable { case prepared, claimed, effectStarted, completed }
        let request: JarvisWorkspaceRequest
        let sessionID: UUID
        var phase: Phase
        var acknowledgement: JarvisWorkspaceAcknowledgement?
        var acknowledged: Bool
        var filename: String?
        var contentType: String?
        var relativePath: String?
        var id: UUID { request.requestID }
    }

    private let directory: URL
    private let manager = FileManager.default
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(ownerKey: String, root: URL? = nil, allowsFileBytes: Bool = true) throws {
        self.allowsFileBytes = allowsFileBytes
        guard !ownerKey.isEmpty else {
            throw JarvisWorkspaceError(code: "missing_owner", message: "Sign in before opening saved Workspace files.")
        }
        let base = try root ?? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                       appropriateFor: nil, create: true).appendingPathComponent("Jarvis/Workspace", isDirectory: true)
        directory = base.appendingPathComponent(Self.digest(Data(ownerKey.utf8)), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var protected = directory
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        try protected.setResourceValues(values)
        encoder.outputFormatting = [.sortedKeys]
    }

    /// Ordinary app composition, created only after authenticated owner/session
    /// verification. Storage remains bounded by each existing transfer policy.
    static func verifiedAccountJournal(for context: JarvisAuthenticatedContext, root: URL? = nil) throws -> JarvisWorkspaceLocalStore {
        try JarvisWorkspaceLocalStore(ownerKey: ownerKey(for: context), root: root)
    }

    /// Both identifiers come from Core's authenticated context response. The
    /// session is deliberately excluded so the same verified owner can reconcile
    /// its durable journal across sessions. JSON prevents ambiguous concatenation.
    static func ownerKey(for context: JarvisAuthenticatedContext) throws -> String {
        guard context.hasValidIdentity else {
            throw JarvisWorkspaceError(code: "invalid_owner_context", message: "The saved-data context could not be verified.")
        }
        let data = try JSONSerialization.data(withJSONObject: ["owner": context.ownerID, "backend": context.backendNamespace], options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    func entry(id: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws -> Entry? {
        try access?.check()
        let path = journalURL(id)
        guard manager.fileExists(atPath: path.path) else { return nil }
        let value = try decoder.decode(Entry.self, from: Data(contentsOf: path))
        guard value.id == id else { throw invalidJournal() }
        return value
    }

    func entries(access: JarvisWorkspaceLocalAccess? = nil) throws -> [Entry] {
        try access?.check()
        return try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .compactMap { path in
                guard let id = UUID(uuidString: path.deletingPathExtension().lastPathComponent) else { return nil }
                return try entry(id: id)
            }
            .sorted { $0.request.createdAt > $1.request.createdAt }
    }

    func prepareClaim(_ request: JarvisWorkspaceRequest, sessionID: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws -> Entry {
        try access?.check()
        _ = try request.validatedTransferByteLimit()
        if let existing = try entry(id: request.id) {
            guard existing.sessionID == sessionID, sameEffect(existing.request, request) else { throw invalidJournal() }
            return existing
        }
        let value = Entry(request: request, sessionID: sessionID, phase: .prepared,
                          acknowledgement: nil, acknowledged: false, filename: nil, contentType: nil, relativePath: nil)
        try persist(value)
        return value
    }

    func confirmClaim(id: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws {
        try access?.check()
        var value = try required(id)
        guard value.phase == .prepared else { return }
        value.phase = .claimed; try persist(value)
    }

    func beginEffect(id: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws {
        try access?.check()
        var value = try required(id)
        guard value.phase == .claimed else {
            throw JarvisWorkspaceError(code: "delivery_requires_reconciliation", message: "This request may already have run. Check its saved result first.")
        }
        value.phase = .effectStarted; try persist(value)
    }

    func completeDisplay(id: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws -> JarvisWorkspaceAcknowledgement {
        try access?.check()
        var value = try required(id)
        guard value.request.action == .open, value.phase == .effectStarted else { throw invalidJournal() }
        let ack = JarvisWorkspaceAcknowledgement(sessionID: value.sessionID, state: .displayed)
        value.phase = .completed; value.acknowledgement = ack
        try persist(value)
        return ack
    }

    /// Saves bytes only after matching the exact prepared export; reopens them before claiming success.
    func saveDownload(id: UUID, file: JarvisWorkspaceFileContent, access: JarvisWorkspaceLocalAccess? = nil) throws -> JarvisWorkspaceAcknowledgement {
        try access?.check()
        try requireFileAccess()
        var value = try required(id)
        guard value.request.action == .download, value.phase == .effectStarted,
              value.request.transferPolicy == nil else { throw invalidJournal() }
        try validate(file.data, request: value.request)
        let filename = Self.safeName(file.filename)
        let relative = id.uuidString.lowercased() + ".download"
        let url = directory.appendingPathComponent(relative)
        // Keep the acquired representation durable before its bytes can exist.
        // A restart after the atomic write must recover the same readable copy.
        value.filename = filename; value.contentType = file.contentType
        try persist(value)
        try write(file.data, to: url)
        try validate(Data(contentsOf: url), request: value.request)
        value.relativePath = relative
        let ack = try savedAcknowledgement(value)
        value.acknowledgement = ack; value.phase = .completed
        try persist(value)
        return ack
    }

    /// Adopts only the exact media offer into the existing account journal. The
    /// source remains temporary and independently owned; it is never moved.
    func saveDownloadFile(id: UUID, source: URL, filename: String, contentType: String,
                          access: JarvisWorkspaceLocalAccess) throws -> JarvisWorkspaceAcknowledgement {
        try Task.checkCancellation()
        try access.check()
        try requireFileAccess()
        var value = try required(id)
        guard value.request.action == .download, value.phase == .effectStarted,
              value.request.transferPolicy != nil, contentType == "video/mp4" else { throw invalidJournal() }
        try validateFile(source, request: value.request, access: access)
        let relative = id.uuidString.lowercased() + ".download"
        let destination = directory.appendingPathComponent(relative)
        guard !manager.fileExists(atPath: destination.path) else {
            throw JarvisWorkspaceError(code: "delivery_requires_reconciliation", message: "A local copy already exists. Check its saved result before trying again.")
        }
        value.filename = Self.safeName(filename); value.contentType = "video/mp4"
        try persist(value)
        let partial = directory.appendingPathComponent(id.uuidString.lowercased() + "." + UUID().uuidString + ".partial")
        defer { try? manager.removeItem(at: partial) }
        try manager.copyItem(at: source, to: partial)
        try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: partial.path)
        #if os(iOS)
        try manager.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: partial.path)
        #endif
        try validateFile(partial, request: value.request, access: access)
        let handle = try FileHandle(forWritingTo: partial)
        do { try handle.synchronize(); try handle.close() }
        catch { try? handle.close(); throw error }
        try Task.checkCancellation()
        try access.check()
        try manager.moveItem(at: partial, to: destination)
        // Reopen the final bytes before recording success. If retirement/failure
        // occurs here, the exact final file remains for explicit reconciliation.
        try validateFile(destination, request: value.request, access: access)
        value.relativePath = relative
        let ack = try savedAcknowledgement(value)
        value.acknowledgement = ack; value.phase = .completed
        try persist(value)
        return ack
    }

    /// Explicit reconciliation after a crash between the atomic file write and journal completion.
    /// It never fetches or exports another copy.
    func reconcileDownload(id: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws -> JarvisWorkspaceAcknowledgement {
        try access?.check()
        try requireFileAccess()
        var value = try required(id)
        if let ack = value.acknowledgement, value.phase == .completed { return ack }
        guard value.phase == .effectStarted, value.request.action == .download else { throw invalidJournal() }
        let relative = id.uuidString.lowercased() + ".download"
        if value.request.transferPolicy != nil {
            guard value.contentType == "video/mp4", value.filename != nil else { throw invalidJournal() }
            try validateFile(directory.appendingPathComponent(relative), request: value.request, access: access)
        } else {
            try validate(Data(contentsOf: directory.appendingPathComponent(relative)), request: value.request)
        }
        value.relativePath = relative
        // Older journals may predate pre-write metadata. Only the known NOTE
        // export formats establish a representation without acquired metadata.
        let noteFormat = value.request.artifactKind == "note" ? value.request.format : nil
        let suffix = noteFormat == "text" ? ".txt" : (noteFormat == "json" ? ".json" : "")
        value.filename = value.filename ?? "Jarvis-\(value.request.artifactID.uuidString.prefix(8))\(suffix)"
        value.contentType = value.contentType ?? (noteFormat == "text" ? "text/plain" :
            (noteFormat == "json" ? "application/json" : "application/octet-stream"))
        let ack = try savedAcknowledgement(value)
        value.acknowledgement = ack; value.phase = .completed
        try persist(value)
        return ack
    }

    func completeWithoutEffect(id: UUID, state: JarvisWorkspacePresentationState, errorCode: String? = nil,
                               access: JarvisWorkspaceLocalAccess? = nil) throws -> JarvisWorkspaceAcknowledgement {
        try access?.check()
        guard state == .dismissed || state == .failed else { throw invalidJournal() }
        var value = try required(id)
        if let ack = value.acknowledgement { return ack }
        let ack = JarvisWorkspaceAcknowledgement(sessionID: value.sessionID, state: state, errorCode: errorCode)
        value.acknowledgement = ack; value.phase = .completed; try persist(value)
        return ack
    }

    func markAcknowledged(id: UUID, acknowledgement: JarvisWorkspaceAcknowledgement, access: JarvisWorkspaceLocalAccess? = nil) throws {
        try access?.check()
        var value = try required(id)
        guard value.acknowledgement == acknowledgement else { throw invalidJournal() }
        value.acknowledged = true; try persist(value)
    }

    /// No network call: the same exact saved bytes are validated on each offline reopen.
    func offlineContent(id: UUID, access: JarvisWorkspaceLocalAccess? = nil) throws -> JarvisWorkspaceFileContent {
        try access?.check()
        try requireFileAccess()
        let value = try required(id)
        guard value.request.transferPolicy == nil, value.acknowledgement?.state == .savedLocal,
              let relative = value.relativePath,
              relative == id.uuidString.lowercased() + ".download" else { throw invalidJournal() }
        let data = try Data(contentsOf: directory.appendingPathComponent(relative))
        try validate(data, request: value.request)
        return JarvisWorkspaceFileContent(data: data, filename: value.filename ?? "Jarvis-file",
                                          contentType: value.contentType ?? "application/octet-stream")
    }

    /// Streaming validation on every offline reopen; no network or Data buffer.
    func offlineFile(id: UUID, access: JarvisWorkspaceLocalAccess) throws -> JarvisWorkspaceLocalFileContent {
        try access.check()
        try requireFileAccess()
        let value = try required(id)
        guard value.phase == .completed, value.request.transferPolicy != nil,
              value.acknowledgement?.state == .savedLocal, value.contentType == "video/mp4",
              let filename = value.filename, let relative = value.relativePath,
              relative == id.uuidString.lowercased() + ".download",
              let count = value.request.byteCount, let hash = value.request.contentSHA256 else { throw invalidJournal() }
        let url = directory.appendingPathComponent(relative)
        try validateFile(url, request: value.request, access: access)
        return JarvisWorkspaceLocalFileContent(url: url, filename: filename, contentType: "video/mp4",
                                               byteCount: count, contentSHA256: hash)
    }

    private func savedAcknowledgement(_ entry: Entry) throws -> JarvisWorkspaceAcknowledgement {
        guard let fileID = entry.request.fileID, let bytes = entry.request.byteCount,
              let hash = entry.request.contentSHA256 else { throw invalidJournal() }
        return JarvisWorkspaceAcknowledgement(sessionID: entry.sessionID, state: .savedLocal,
            localReceipt: JarvisWorkspaceLocalReceipt(fileID: fileID, versionID: entry.request.versionID,
                byteCount: bytes, contentSHA256: hash, destination: "app_storage"))
    }
    private func validate(_ data: Data, request: JarvisWorkspaceRequest) throws {
        _ = try request.validatedTransferByteLimit()
        guard request.transferPolicy == nil, request.contentReady, request.fileID != nil, data.count <= 8 * 1024 * 1024,
              request.byteCount == data.count, request.contentSHA256 == Self.digest(data) else {
            throw JarvisWorkspaceError(code: "download_mismatch", message: "The downloaded bytes do not match the selected saved version.")
        }
    }
    private func validateFile(_ url: URL, request: JarvisWorkspaceRequest, access: JarvisWorkspaceLocalAccess?) throws {
        try Task.checkCancellation()
        try access?.check()
        let limit = try request.validatedTransferByteLimit()
        guard request.transferPolicy != nil, url.isFileURL, let expected = request.byteCount,
              let digest = request.contentSHA256 else { throw invalidJournal() }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw invalidJournal() }
        let reader = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? reader.close() }
        var information = stat()
        guard fstat(descriptor, &information) == 0, information.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              information.st_size == Int64(expected), information.st_size <= Int64(limit) else { throw invalidJournal() }
        var hash = SHA256(), count = 0
        while let chunk = try reader.read(upToCount: 64 * 1024), !chunk.isEmpty {
            try Task.checkCancellation()
            try access?.check()
            guard chunk.count <= expected - count else { throw invalidJournal() }
            hash.update(data: chunk); count += chunk.count
        }
        guard count == expected, hash.finalize().map({ String(format: "%02x", $0) }).joined() == digest else {
            throw JarvisWorkspaceError(code: "download_mismatch", message: "The saved video bytes do not match the selected version.")
        }
        try Task.checkCancellation()
        try access?.check()
    }
    private func required(_ id: UUID) throws -> Entry { guard let value = try entry(id: id) else { throw invalidJournal() }; return value }
    private func journalURL(_ id: UUID) -> URL { directory.appendingPathComponent(id.uuidString.lowercased() + ".json") }
    private func persist(_ entry: Entry) throws { try write(encoder.encode(entry), to: journalURL(entry.id)) }
    private func write(_ data: Data, to url: URL) throws {
        #if os(iOS)
        try data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        #else
        try data.write(to: url, options: .atomic)
        #endif
    }
    private func sameEffect(_ left: JarvisWorkspaceRequest, _ right: JarvisWorkspaceRequest) -> Bool {
        left.id == right.id && left.action == right.action && left.artifactKind == right.artifactKind &&
        left.artifactID == right.artifactID && left.versionID == right.versionID && left.format == right.format &&
        left.fileID == right.fileID && left.byteCount == right.byteCount && left.contentSHA256 == right.contentSHA256 &&
        left.transferPolicy == right.transferPolicy && left.transferLimitBytes == right.transferLimitBytes
    }
    private func invalidJournal() -> JarvisWorkspaceError {
        JarvisWorkspaceError(code: "local_delivery_conflict", message: "The saved delivery record does not match this request. No action was repeated.")
    }
    private func requireFileAccess() throws {
        guard allowsFileBytes else {
            throw JarvisWorkspaceError(code: "offline_files_unavailable", message: "Offline file access is not available yet.")
        }
    }
    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private static func safeName(_ name: String) -> String {
        let result = name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
            .components(separatedBy: .controlCharacters).joined(separator: "-")
        return result.isEmpty || result == "." || result == ".." ? "Jarvis-file" : String(result.prefix(200))
    }
}
