import CryptoKit
import Foundation
import Darwin

/// A verified private temporary copy, not a durable save or server receipt.
/// Its original access lifetime remains attached even if another account signs in.
final class JarvisVerifiedTemporaryFile: @unchecked Sendable {
    enum Format: Sendable {
        case binary, mp4
        fileprivate var filename: String { self == .mp4 ? "content.mp4" : "content.verified" }
    }
    let byteCount: Int
    let contentSHA256: String
    private let directory: URL
    private let file: URL
    private let access: JarvisWorkspaceLocalAccess
    private let lock = NSLock()
    private var discarded = false

    fileprivate init(directory: URL, file: URL, byteCount: Int, contentSHA256: String,
                     access: JarvisWorkspaceLocalAccess) {
        self.directory = directory
        self.file = file
        self.byteCount = byteCount
        self.contentSHA256 = contentSHA256
        self.access = access
    }

    func urlForReading() throws -> URL {
        lock.lock(); defer { lock.unlock() }
        try access.check()
        guard !discarded else { throw CancellationError() }
        return file
    }

    /// The presentation owner also stops/releases its player when retiring access.
    func discard() {
        lock.lock()
        guard !discarded else { lock.unlock(); return }
        discarded = true
        lock.unlock()
        try? FileManager.default.removeItem(at: directory)
    }

    deinit { discard() }
}

/// Copies an already acquired local file in bounded chunks. Authorization,
/// exact FILE/version metadata and the effective limit belong to the caller's
/// existing authenticated delivery path; this helper establishes none of them.
actor JarvisTemporaryFileVerifier {
    private let root: URL
    private let manager = FileManager.default

    init(root: URL = FileManager.default.temporaryDirectory) {
        self.root = root
    }

    func copyVerifiedFile(from source: URL, expectedByteCount: Int, expectedSHA256: String,
                          maximumByteCount: Int, access: JarvisWorkspaceLocalAccess,
                          format: JarvisVerifiedTemporaryFile.Format = .binary) throws -> JarvisVerifiedTemporaryFile {
        try Task.checkCancellation()
        try access.check()
        guard source.isFileURL, root.isFileURL, expectedByteCount >= 0,
              maximumByteCount >= 0, expectedByteCount <= maximumByteCount,
              expectedSHA256.utf8.count == 64,
              expectedSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw failure("invalid_file_expectation", "The saved file metadata could not be verified.")
        }

        // Open without following a final symlink, and inspect the actual handle.
        let inputDescriptor = Darwin.open(source.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard inputDescriptor >= 0 else {
            throw failure("local_file_unavailable", "The downloaded file is unavailable.")
        }
        let input = FileHandle(fileDescriptor: inputDescriptor, closeOnDealloc: true)
        defer { try? input.close() }
        var information = stat()
        guard fstat(inputDescriptor, &information) == 0,
              information.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              information.st_size == Int64(expectedByteCount) else {
            throw failure("download_mismatch", "The downloaded file does not match the selected size.")
        }

        let directory = root.appendingPathComponent("jarvis-transfer-" + UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: false,
                                    attributes: [.posixPermissions: 0o700])
        var retained = false
        defer { if !retained { try? manager.removeItem(at: directory) } }
        let partial = directory.appendingPathComponent("content.partial")
        let completed = directory.appendingPathComponent(format.filename)
        let outputDescriptor = Darwin.open(partial.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard outputDescriptor >= 0 else {
            throw failure("local_file_unavailable", "A private copy could not be prepared.")
        }
        let output = FileHandle(fileDescriptor: outputDescriptor, closeOnDealloc: true)
        defer { try? output.close() }
        #if os(iOS)
        try manager.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                                  ofItemAtPath: partial.path)
        #endif

        var hash = SHA256(), count = 0
        while let chunk = try input.read(upToCount: 64 * 1024), !chunk.isEmpty {
            try Task.checkCancellation()
            try access.check()
            guard chunk.count <= expectedByteCount - count else {
                throw failure("download_mismatch", "The downloaded file exceeds the selected size.")
            }
            try output.write(contentsOf: chunk)
            hash.update(data: chunk)
            count += chunk.count
        }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard count == expectedByteCount, digest == expectedSHA256 else {
            throw failure("download_mismatch", "The downloaded bytes do not match the selected saved version.")
        }
        try Task.checkCancellation()
        try access.check()
        try output.synchronize()
        try output.close()
        try manager.moveItem(at: partial, to: completed)
        try Task.checkCancellation()
        try access.check()
        let verified = JarvisVerifiedTemporaryFile(directory: directory, file: completed,
            byteCount: count, contentSHA256: digest, access: access)
        retained = true
        return verified
    }

    private func failure(_ code: String, _ message: String) -> JarvisWorkspaceError {
        JarvisWorkspaceError(code: code, message: message)
    }
}
