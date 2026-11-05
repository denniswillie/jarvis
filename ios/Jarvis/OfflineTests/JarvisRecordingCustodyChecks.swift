import Foundation

private struct CheckFailure: Error { let message: String }
private func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw CheckFailure(message: message) }
}

@main
private struct Checks {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-custody-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("synthetic custody bytes, not a qualified MP4".utf8)

        do {
            let temporary = try JarvisRecordingDirectory()
            try bytes.write(to: temporary.file)
            let url = temporary.file
            temporary.discard()
            try require(!FileManager.default.fileExists(atPath: url.path), "isolated temporary cleanup changed")
        }
        print("PASS isolated temporary cleanup remains explicit")

        let owned = root.appendingPathComponent("reserved")
        try FileManager.default.createDirectory(at: owned, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let destination = JarvisRecordingWriter.Destination(directoryURL: owned, generationID: UUID())
        var directory: JarvisRecordingDirectory? = try JarvisRecordingDirectory(storeOwned: destination)
        try bytes.write(to: directory!.file)
        var original: JarvisRecordingLocalFile? = try directory!.completedFile(generationID: destination.generationID, maximumBytes: 1024)
        try require(original!.byteCount == bytes.count, "wrong local file")
        directory = nil
        original = nil
        try require(try Data(contentsOf: owned.appendingPathComponent("original.mp4")) == bytes, "last handle deleted durable bytes")
        print("PASS store-owned completed bytes survive all handle releases")

        do {
            _ = try JarvisRecordingDirectory(storeOwned: destination)
            throw CheckFailure(message: "existing original accepted for overwrite")
        } catch JarvisRecordingWriter.Reason.fileFailed { }
        try require(try Data(contentsOf: owned.appendingPathComponent("original.mp4")) == bytes, "replay changed original")
        print("PASS replay refuses overwriting an existing original")

        let failed = root.appendingPathComponent("failed")
        try FileManager.default.createDirectory(at: failed, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        do {
            let partial = try JarvisRecordingDirectory(storeOwned: .init(directoryURL: failed, generationID: UUID()))
            try bytes.write(to: partial.file)
            do { _ = try partial.completedFile(generationID: UUID(), maximumBytes: 1); throw CheckFailure(message: "limit ignored") }
            catch JarvisRecordingWriter.Reason.fileFailed { }
            partial.discard()
        }
        try require(try Data(contentsOf: failed.appendingPathComponent("original.mp4")) == bytes, "failure erased partial evidence")
        print("PASS failed completion/discard preserves store-owned evidence")

        let link = root.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: owned)
        do {
            _ = try JarvisRecordingDirectory(storeOwned: .init(directoryURL: link, generationID: UUID()))
            throw CheckFailure(message: "symlink destination accepted")
        } catch JarvisRecordingWriter.Reason.fileFailed { }
        print("PASS symlink destination refused")
    }
}
