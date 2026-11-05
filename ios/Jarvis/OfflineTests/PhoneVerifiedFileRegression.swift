import CryptoKit
import Foundation

@main
enum PhoneVerifiedFileRegression {
    @MainActor
    static func main() async throws {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let manager = FileManager.default
        let root = manager.temporaryDirectory.appendingPathComponent("jarvis-verified-file-test-" + UUID().uuidString)
        try manager.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: root) }
        let source = root.appendingPathComponent("source.bin")
        precondition(manager.createFile(atPath: source.path, contents: nil))
        let writer = try FileHandle(forWritingTo: source)
        let chunk = Data((0..<(64 * 1024)).map { UInt8($0 % 251) })
        var hash = SHA256()
        for _ in 0..<145 { try writer.write(contentsOf: chunk); hash.update(data: chunk) }
        try writer.close()
        let size = 145 * chunk.count
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let verifier = JarvisTemporaryFileVerifier(root: root)
        let access = JarvisWorkspaceLocalAccess()
        let copy = try await verifier.copyVerifiedFile(from: source, expectedByteCount: size,
            expectedSHA256: digest, maximumByteCount: 16 * 1024 * 1024, access: access)
        let url = try copy.urlForReading()
        check(size > 8 * 1024 * 1024 && copy.byteCount == size && copy.contentSHA256 == digest,
              "A real file larger than the document ceiling verifies through the bounded file path")
        check(url != source && manager.fileExists(atPath: source.path), "The private copy never moves or deletes its source")
        let attributes = try manager.attributesOfItem(atPath: url.path)
        check((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600, "Private file permissions are retained")
        // Independently stream the completed file; never materialize the large fixture.
        let reader = try FileHandle(forReadingFrom: url)
        var reread = SHA256(), rereadCount = 0
        while let bytes = try reader.read(upToCount: 91_000), !bytes.isEmpty { reread.update(data: bytes); rereadCount += bytes.count }
        try reader.close()
        check(rereadCount == size && reread.finalize().map { String(format: "%02x", $0) }.joined() == digest,
              "The actual completed file has the exact expected bytes")
        access.invalidate()
        do { _ = try copy.urlForReading(); preconditionFailure("Retired access exposed a file") }
        catch is CancellationError { check(true, "The verified result remains bound to its original account lifetime") }
        copy.discard(); copy.discard()
        check(!manager.fileExists(atPath: url.path) && manager.fileExists(atPath: source.path), "Discard is idempotent and preserves source bytes")

        func transferDirectories() throws -> [URL] {
            try manager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix("jarvis-transfer-") }
        }
        for (expectedSize, expectedHash, limit) in [(size, String(repeating: "0", count: 64), size),
                                                   (size - 1, digest, size), (size, digest, 8 * 1024 * 1024)] {
            do {
                _ = try await verifier.copyVerifiedFile(from: source, expectedByteCount: expectedSize,
                    expectedSHA256: expectedHash, maximumByteCount: limit, access: JarvisWorkspaceLocalAccess())
                preconditionFailure("Invalid size/hash/limit accepted")
            } catch is JarvisWorkspaceError { check(try transferDirectories().isEmpty, "Rejected data leaves no partial or completed copy") }
        }
        let symlink = root.appendingPathComponent("source-link.bin")
        try manager.createSymbolicLink(at: symlink, withDestinationURL: source)
        do {
            _ = try await verifier.copyVerifiedFile(from: symlink, expectedByteCount: size,
                expectedSHA256: digest, maximumByteCount: size, access: JarvisWorkspaceLocalAccess())
            preconditionFailure("Symlink accepted")
        } catch is JarvisWorkspaceError { check(try transferDirectories().isEmpty, "A symlink cannot become a verified source") }
        let retired = JarvisWorkspaceLocalAccess(); retired.invalidate()
        do {
            _ = try await verifier.copyVerifiedFile(from: source, expectedByteCount: size,
                expectedSHA256: digest, maximumByteCount: size, access: retired)
            preconditionFailure("Retired lifetime started a copy")
        } catch is CancellationError { check(try transferDirectories().isEmpty, "Retired lifetime creates no file") }
        let cancelled = Task {
            try await verifier.copyVerifiedFile(from: source, expectedByteCount: size,
                expectedSHA256: digest, maximumByteCount: size, access: JarvisWorkspaceLocalAccess())
        }
        cancelled.cancel()
        do { _ = try await cancelled.value; preconditionFailure("Cancelled copy returned a result") }
        catch is CancellationError { check(try transferDirectories().isEmpty, "Cancellation cannot publish a temporary result") }
        print("Phone verified-file regressions passed: \(checks) checks; fixture_bytes=\(size)")
        print("Phone media-contract regressions passed: \(try PhoneMediaContractRegression.run()) checks")
    }
}
