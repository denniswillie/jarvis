import CryptoKit
import Darwin
import Foundation

/// Pre-capture durable custody. A reservation owns its directory and full admitted
/// byte budget BEFORE Writer/Capture attaches. Nothing here deletes media, even
/// after failed finalization, reconnect, cancellation, or a saved reconciliation.
actor JarvisRecordingPendingStore {
    typealias Failure = JarvisRecordingManifest.Failure

    struct Scope: Codable, Equatable, Sendable {
        let ownerID: String
        let backendNamespace: String
        init(context: JarvisAuthenticatedContext) throws {
            guard context.hasValidIdentity, !context.ownerID.contains("\0") else { throw Failure.invalidScope }
            ownerID = context.ownerID; backendNamespace = context.backendNamespace
        }
        fileprivate func validate() throws {
            _ = try Scope(context: JarvisAuthenticatedContext(ownerID: ownerID,
                backendNamespace: backendNamespace, sessionID: UUID()))
        }
        fileprivate var directoryName: String {
            get throws {
                let data = try JSONSerialization.data(withJSONObject: ["owner": ownerID, "backend": backendNamespace], options: [.sortedKeys])
                return JarvisRecordingPendingStore.digest(data)
            }
        }
    }

    struct Reservation: Sendable {
        let binding: JarvisRecordingManifest.Binding
        let destination: JarvisRecordingWriter.Destination
        let reservedAt: Date
        var recordingID: UUID { binding.recordingID }
        var sessionID: UUID { binding.sessionID }
    }

    struct Unresolved: Sendable {
        enum Reason: String, Sendable { case awaitingFinalization, invalidFinalization, invalidReservation }
        let recordingID: UUID
        let reservation: Reservation?
        let reason: Reason
        // A stat observation only, NOT proof of playable, complete or saved media.
        let observedFileBytes: Int64?
    }

    struct Entry: Sendable {
        let binding: JarvisRecordingManifest.Binding
        let writerGenerationID: UUID
        let fileURL: URL
        let descriptor: String
        let byteCount: Int
        let contentSHA256: String
        let retainedAt: Date
        let confirmedLinkage: JarvisRecordingManifest.SavedLinkage?
        var recordingID: UUID { binding.recordingID }
        var sessionID: UUID { binding.sessionID }
        var generationID: UUID { binding.generationID }
        var operationID: UUID { binding.operationID }
        var includeJarvisAudio: Bool { binding.includeJarvisAudio }
        var maxBytes: Int { binding.maxBytes }
        var maxDurationMilliseconds: Int { binding.maxDurationMilliseconds }
    }

    private struct ReservationRecord: Codable, Sendable {
        let format: String
        let scope: Scope
        let binding: JarvisRecordingManifest.Binding
        let writerGenerationID: UUID
        let reservedAt: Date
    }
    private struct Journal: Codable, Sendable {
        let format: String
        let scope: Scope
        let binding: JarvisRecordingManifest.Binding
        let writerGenerationID: UUID
        let descriptor: String
        let descriptorSHA256: String
        let retainedAt: Date
    }

    private let root: URL
    static let maximumEntries = 128
    static let maximumRetainedBytes = 512 * 1024 * 1024
    private static let journalLimit = 32 * 1024

    init(root: URL? = nil) throws {
        let base = try root ?? FileManager.default.url(for: .applicationSupportDirectory,
            in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("Jarvis/RecordingPending", isDirectory: true)
        guard base.isFileURL else { throw Failure.unsafeFile }
        self.root = base
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let fd = try Self.openDirectory(base.path); defer { close(fd) }
        #if os(iOS)
        try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: base.path)
        #endif
        var url = base; var resources = URLResourceValues(); resources.isExcludedFromBackup = true
        try url.setResourceValues(resources)
    }

    /// The returned destination must be passed to Capture.start before any
    /// attachment. Replaying a command returns the SAME custody/generation. It
    /// does not authorize a second capture; Writer refuses an existing original.
    func reserve(command: JarvisRecordingControl.Command, scope: Scope) throws -> Reservation {
        let binding = try JarvisRecordingManifest.Binding(command: command)
        return try withNamespace(scope) { namespace, url in
            try Task.checkCancellation()
            let name = binding.recordingID.uuidString.lowercased()
            let names = try Self.entryNames(namespace)
            if names.contains(name) {
                let fd = try Self.openDirectory(name, at: namespace); defer { close(fd) }
                let prior = try Self.reservationRecord(fd, name: name, scope: scope)
                guard prior.binding == binding else { throw Failure.conflict }
                return Self.reservation(prior, at: url.appendingPathComponent(name))
            }
            guard names.count < Self.maximumEntries else { throw Failure.capacityExceeded }
            // Charge full admitted budgets, including completed/confirmed entries;
            // a future disposal API may release custody, but no upload does so.
            var charged = binding.maxBytes
            for existing in names {
                let fd = try Self.openDirectory(existing, at: namespace); defer { close(fd) }
                charged += try Self.reservationRecord(fd, name: existing, scope: scope).binding.maxBytes
            }
            guard charged <= Self.maximumRetainedBytes else { throw Failure.capacityExceeded }
            let record = ReservationRecord(format: "jarvis.recording_reservation.v1", scope: scope, binding: binding,
                writerGenerationID: UUID(), reservedAt: Date())
            let stage = ".staging-" + UUID().uuidString.lowercased()
            guard mkdirat(namespace, stage, 0o700) == 0 else { throw Failure.ioFailure }
            let fd = try Self.openDirectory(stage, at: namespace)
            var committed = false
            defer {
                if !committed { unlinkat(fd, "reservation.json", 0); unlinkat(namespace, stage, AT_REMOVEDIR) }
                close(fd)
            }
            try Self.writeNew(try JarvisRecordingManifest.encode(record), name: "reservation.json", at: fd, maximum: Self.journalLimit)
            guard fsync(fd) == 0 else { throw Failure.ioFailure }
            try Task.checkCancellation()
            guard renameat(namespace, stage, namespace, name) == 0 else { throw Failure.ioFailure }
            committed = true
            guard fsync(namespace) == 0 else { throw Failure.ioFailure }
            return Self.reservation(record, at: url.appendingPathComponent(name))
        }
    }

    /// Adopt only the exact reserved original. There is no copy, move, temporary
    /// fallback, implicit reservation, or deletion on any failure path.
    func retain(command: JarvisRecordingControl.Command, completed: JarvisRecordingWriter.Completed, scope: Scope) throws -> Entry {
        let binding = try JarvisRecordingManifest.Binding(command: command)
        return try withNamespace(scope) { namespace, url in
            let name = binding.recordingID.uuidString.lowercased()
            guard try Self.exists(name, at: namespace) else { throw Failure.unreservedFile }
            let fd = try Self.openDirectory(name, at: namespace); defer { close(fd) }
            let record = try Self.reservationRecord(fd, name: name, scope: scope)
            guard record.binding == binding else { throw Failure.conflict }
            let directory = url.appendingPathComponent(name, isDirectory: true)
            let originalURL = directory.appendingPathComponent("original.mp4")
            guard completed.generationID == record.writerGenerationID,
                  completed.file.generationID == record.writerGenerationID,
                  try completed.file.urlForReading().standardizedFileURL == originalURL.standardizedFileURL else {
                throw Failure.unreservedFile
            }
            let descriptor = try JarvisRecordingManifest.make(binding: binding, completed: completed,
                expectedWriterGenerationID: record.writerGenerationID)
            if try Self.exists("journal.json", at: fd) {
                let prior = try Self.load(fd, record: record, directory: directory)
                guard prior.descriptor == descriptor else { throw Failure.conflict }
                return prior
            }
            let original = openat(fd, "original.mp4", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
            guard original >= 0 else { throw Failure.unsafeFile }
            defer { close(original) }
            try Self.hashFile(original, size: completed.file.byteCount, expected: completed.file.contentSHA256)
            guard fsync(original) == 0 else { throw Failure.ioFailure }
            let journal = Journal(format: "jarvis.recording_pending.v2", scope: scope, binding: binding,
                writerGenerationID: record.writerGenerationID, descriptor: descriptor,
                descriptorSHA256: Self.digest(Data(descriptor.utf8)), retainedAt: Date())
            try Task.checkCancellation()
            try Self.atomicWrite(try JarvisRecordingManifest.encode(journal), name: "journal.json", at: fd, maximum: Self.journalLimit)
            return Self.entry(journal, at: directory, upload: try JarvisRecordingManifest.decode(descriptor, binding: binding), linkage: nil)
        }
    }

    /// Only validated completed entries. Call unresolved(scope:) separately to
    /// surface partials, failed finalization, damaged metadata or legacy records.
    func pending(scope: Scope) throws -> [Entry] {
        try withNamespace(scope) { namespace, url in
            var entries: [Entry] = []
            for name in try Self.entryNames(namespace) {
                try Task.checkCancellation()
                do {
                    let fd = try Self.openDirectory(name, at: namespace); defer { close(fd) }
                    let record = try Self.reservationRecord(fd, name: name, scope: scope)
                    if try Self.exists("journal.json", at: fd) {
                        entries.append(try Self.load(fd, record: record, directory: url.appendingPathComponent(name)))
                    }
                } catch is CancellationError { throw CancellationError() }
                catch { /* Preserved and surfaced by unresolved; never called valid. */ }
            }
            return entries
        }
    }

    func unresolved(scope: Scope) throws -> [Unresolved] {
        try withNamespace(scope) { namespace, url in
            var result: [Unresolved] = []
            for name in try Self.entryNames(namespace) {
                try Task.checkCancellation()
                var admitted: Reservation?
                var observed: Int64?
                do {
                    let fd = try Self.openDirectory(name, at: namespace); defer { close(fd) }
                    let record = try Self.reservationRecord(fd, name: name, scope: scope)
                    let directory = url.appendingPathComponent(name)
                    admitted = Self.reservation(record, at: directory)
                    var metadata = stat()
                    if fstatat(fd, "original.mp4", &metadata, AT_SYMLINK_NOFOLLOW) == 0,
                       metadata.st_mode & S_IFMT == S_IFREG, metadata.st_size >= 0 {
                        observed = metadata.st_size
                    }
                    if try Self.exists("journal.json", at: fd) {
                        _ = try Self.load(fd, record: record, directory: directory)
                    } else {
                        result.append(Unresolved(recordingID: record.binding.recordingID, reservation: admitted,
                            reason: .awaitingFinalization, observedFileBytes: observed))
                    }
                } catch is CancellationError { throw CancellationError() }
                catch {
                    result.append(Unresolved(recordingID: UUID(uuidString: name)!, reservation: admitted,
                        reason: admitted == nil ? .invalidReservation : .invalidFinalization, observedFileBytes: observed))
                }
            }
            return result
        }
    }

    /// Authenticated matching response required. JSON is not authentication, and
    /// the receipt has no hash: linkage never releases custody or budget.
    func confirmSaved(receiptData: Data, recordingID: UUID, scope: Scope) throws -> Entry {
        try withNamespace(scope) { namespace, url in
            let name = recordingID.uuidString.lowercased()
            let fd = try Self.openDirectory(name, at: namespace); defer { close(fd) }
            let record = try Self.reservationRecord(fd, name: name, scope: scope)
            let current = try Self.load(fd, record: record, directory: url.appendingPathComponent(name))
            let receipt = try JarvisRecordingManifest.savedReceipt(receiptData, binding: current.binding, ownerID: scope.ownerID)
            if let old = current.confirmedLinkage {
                guard old.identifiesSamePublication(as: receipt) else { throw Failure.conflict }
                return current
            }
            try Self.atomicWrite(try JarvisRecordingManifest.encode(receipt), name: "saved-receipt.json", at: fd, maximum: 16_384)
            return Entry(binding: current.binding, writerGenerationID: current.writerGenerationID,
                fileURL: current.fileURL, descriptor: current.descriptor, byteCount: current.byteCount,
                contentSHA256: current.contentSHA256, retainedAt: current.retainedAt, confirmedLinkage: receipt)
        }
    }

    private func withNamespace<T>(_ scope: Scope, _ work: (Int32, URL) throws -> T) throws -> T {
        try scope.validate()
        let rootFD = try Self.openDirectory(root.path); defer { close(rootFD) }
        let name = try scope.directoryName
        if mkdirat(rootFD, name, 0o700) != 0, errno != EEXIST { throw Failure.ioFailure }
        let fd = try Self.openDirectory(name, at: rootFD); defer { close(fd) }
        let lockFD = openat(fd, ".lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0o600)
        guard lockFD >= 0 else { throw Failure.unsafeFile }
        defer { close(lockFD) }
        _ = try Self.regular(lockFD, maximum: 0)
        guard flock(lockFD, LOCK_EX | LOCK_NB) == 0 else { throw Failure.busy }
        defer { flock(lockFD, LOCK_UN) }
        guard fsync(rootFD) == 0 else { throw Failure.ioFailure }
        return try work(fd, root.appendingPathComponent(name, isDirectory: true))
    }

    private static func reservation(_ record: ReservationRecord, at directory: URL) -> Reservation {
        Reservation(binding: record.binding, destination: .init(directoryURL: directory, generationID: record.writerGenerationID), reservedAt: record.reservedAt)
    }
    private static func reservationRecord(_ fd: Int32, name: String, scope: Scope) throws -> ReservationRecord {
        let record = try JSONDecoder().decode(ReservationRecord.self, from: readFile("reservation.json", at: fd, maximum: journalLimit))
        try record.binding.validate()
        guard record.format == "jarvis.recording_reservation.v1", record.scope == scope,
              record.binding.recordingID.uuidString.lowercased() == name, record.reservedAt.timeIntervalSince1970.isFinite else {
            throw Failure.corruptEntry
        }
        return record
    }
    private static func load(_ fd: Int32, record: ReservationRecord, directory: URL) throws -> Entry {
        let journal = try JSONDecoder().decode(Journal.self, from: readFile("journal.json", at: fd, maximum: journalLimit))
        guard journal.format == "jarvis.recording_pending.v2", journal.scope == record.scope,
              journal.binding == record.binding, journal.writerGenerationID == record.writerGenerationID,
              journal.descriptorSHA256 == digest(Data(journal.descriptor.utf8)),
              journal.retainedAt.timeIntervalSince1970.isFinite else { throw Failure.corruptEntry }
        let upload = try JarvisRecordingManifest.decode(journal.descriptor, binding: record.binding)
        let original = openat(fd, "original.mp4", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard original >= 0 else { throw Failure.unsafeFile }
        defer { close(original) }
        try hashFile(original, size: upload.sizeBytes, expected: upload.contentSHA256)
        let receipt = try readFile("saved-receipt.json", at: fd, maximum: 16_384, optional: true)
        let linkage = receipt.isEmpty ? nil : try JarvisRecordingManifest.savedReceipt(receipt, binding: record.binding, ownerID: record.scope.ownerID)
        return entry(journal, at: directory, upload: upload, linkage: linkage)
    }
    private static func entry(_ journal: Journal, at directory: URL, upload: JarvisRecordingManifest.Upload,
                              linkage: JarvisRecordingManifest.SavedLinkage?) -> Entry {
        Entry(binding: journal.binding, writerGenerationID: journal.writerGenerationID,
            fileURL: directory.appendingPathComponent("original.mp4"), descriptor: journal.descriptor,
            byteCount: upload.sizeBytes, contentSHA256: upload.contentSHA256, retainedAt: journal.retainedAt, confirmedLinkage: linkage)
    }

    private static func entryNames(_ fd: Int32) throws -> [String] {
        let duplicate = dup(fd)
        guard duplicate >= 0 else { throw Failure.ioFailure }
        guard let directory = fdopendir(duplicate) else { close(duplicate); throw Failure.ioFailure }
        defer { closedir(directory) }
        var result: [String] = [], inspected = 0
        while true {
            errno = 0
            guard let entry = readdir(directory) else { if errno != 0 { throw Failure.ioFailure }; break }
            inspected += 1
            guard inspected <= 4096 else { throw Failure.capacityExceeded }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            if let id = UUID(uuidString: name), id.uuidString.lowercased() == name {
                result.append(name)
                guard result.count <= maximumEntries else { throw Failure.capacityExceeded }
            } else if name != ".", name != "..", name != ".lock", !name.hasPrefix(".staging-") { throw Failure.corruptEntry }
        }
        return result.sorted()
    }
    private static func exists(_ name: String, at directory: Int32) throws -> Bool {
        var metadata = stat()
        if fstatat(directory, name, &metadata, AT_SYMLINK_NOFOLLOW) == 0 { return true }
        if errno == ENOENT { return false }
        throw Failure.ioFailure
    }
    private static func openDirectory(_ name: String, at parent: Int32 = AT_FDCWD) throws -> Int32 {
        let fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw Failure.unsafeFile }
        var metadata = stat()
        guard fstat(fd, &metadata) == 0, metadata.st_mode & S_IFMT == S_IFDIR,
              metadata.st_uid == geteuid(), metadata.st_mode & 0o077 == 0 else { close(fd); throw Failure.unsafeFile }
        return fd
    }
    private static func regular(_ fd: Int32, maximum: Int) throws -> stat {
        var metadata = stat()
        guard fstat(fd, &metadata) == 0, metadata.st_mode & S_IFMT == S_IFREG,
              metadata.st_uid == geteuid(), metadata.st_mode & 0o077 == 0, metadata.st_nlink == 1,
              metadata.st_size >= 0, metadata.st_size <= maximum else { throw Failure.unsafeFile }
        return metadata
    }
    private static func unchanged(_ before: stat, _ after: stat) -> Bool {
        before.st_dev == after.st_dev && before.st_ino == after.st_ino && before.st_size == after.st_size &&
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec
    }
    private static func hashFile(_ fd: Int32, size: Int, expected: String) throws {
        guard (1...64 * 1024 * 1024).contains(size), JarvisRecordingManifest.validHash(expected) else { throw Failure.unsafeFile }
        let before = try regular(fd, maximum: size)
        guard before.st_size == size else { throw Failure.corruptEntry }
        var hash = SHA256(), total = 0, buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            if count < 0, errno == EINTR { continue }
            guard count >= 0 else { throw Failure.ioFailure }
            if count == 0 { break }
            total += count
            guard total <= size else { throw Failure.corruptEntry }
            buffer.withUnsafeBytes { hash.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[..<count])) }
        }
        guard total == size, unchanged(before, try regular(fd, maximum: size)), hex(hash.finalize()) == expected else { throw Failure.corruptEntry }
    }
    private static func readFile(_ name: String, at directory: Int32, maximum: Int, optional: Bool = false) throws -> Data {
        let fd = openat(directory, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if fd < 0, optional, errno == ENOENT { return Data() }
        guard fd >= 0 else { throw Failure.unsafeFile }
        defer { close(fd) }
        let before = try regular(fd, maximum: maximum)
        guard before.st_size > 0 else { throw Failure.corruptEntry }
        var data = Data(), buffer = [UInt8](repeating: 0, count: min(maximum, 64 * 1024))
        while true {
            let count = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            if count < 0, errno == EINTR { continue }
            guard count >= 0 else { throw Failure.ioFailure }
            if count == 0 { break }
            guard data.count + count <= maximum else { throw Failure.corruptEntry }
            data.append(contentsOf: buffer.prefix(count))
        }
        guard data.count == before.st_size, unchanged(before, try regular(fd, maximum: maximum)) else { throw Failure.corruptEntry }
        return data
    }
    private static func atomicWrite(_ data: Data, name: String, at directory: Int32, maximum: Int) throws {
        let temporary = ".metadata-" + UUID().uuidString.lowercased()
        defer { unlinkat(directory, temporary, 0) }
        try writeNew(data, name: temporary, at: directory, maximum: maximum)
        guard renameat(directory, temporary, directory, name) == 0, fsync(directory) == 0 else { throw Failure.ioFailure }
    }
    private static func writeNew(_ data: Data, name: String, at directory: Int32, maximum: Int) throws {
        guard !data.isEmpty, data.count <= maximum else { throw Failure.capacityExceeded }
        let fd = openat(directory, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw Failure.ioFailure }
        defer { close(fd) }
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                if count < 0, errno == EINTR { continue }
                guard count > 0 else { throw Failure.ioFailure }
                offset += count
            }
        }
        guard fchmod(fd, 0o400) == 0, fsync(fd) == 0 else { throw Failure.ioFailure }
    }
    private static func digest(_ data: Data) -> String { hex(SHA256.hash(data: data)) }
    private static func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}
