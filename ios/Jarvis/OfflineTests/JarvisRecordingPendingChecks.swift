@preconcurrency import AVFoundation
import Darwin
import Foundation

@main struct JarvisRecordingPendingChecks {
    typealias Store = JarvisRecordingPendingStore
    typealias Manifest = JarvisRecordingManifest
    @MainActor final class WeakFile { weak var value: JarvisRecordingLocalFile? }
    static func require(_ value: Bool, _ message: String) { if !value { fatalError(message) } }
    // A resumed finish() waiter may run before the writer queue releases its
    // local Completed/file references. Completion is not an ARC cleanup fence.
    // Keep each temporary strong weak-load inside a synchronous scope so this
    // observation cannot itself retain the asset across the suspension below.
    @MainActor @inline(never) static func fileWasReleased(_ file: WeakFile) -> Bool { file.value == nil }
    @MainActor static func waitForFileRelease(_ file: WeakFile) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(1))
        while !fileWasReleased(file), clock.now < deadline {
            try await Task.sleep(for: .milliseconds(2))
        }
    }
    @MainActor static func reject(_ expected: Manifest.Failure? = nil, _ body: () async throws -> Void) async {
        do { try await body(); fatalError("Expected rejection") }
        catch { if let expected { require(error as? Manifest.Failure == expected, "Expected \(expected), got \(error)") } }
    }
    static func context(owner: String = "offline-owner", backend: Character = "a") -> JarvisAuthenticatedContext {
        .init(ownerID: owner, backendNamespace: "jarvis-storage-v1:" + String(repeating: String(backend), count: 64), sessionID: UUID())
    }
    static func command(id: UUID = UUID(), session: UUID = UUID(), generation: UUID = UUID(),
                        include: Bool = true, bytes: Int = 64 * 1024 * 1024, duration: Int = 60_000,
                        action: String = "start") throws -> JarvisRecordingControl.Command {
        try .init(values: ["type": "recording.command", "version": 1, "action": action,
            "operation_id": id.uuidString, "recording_id": id.uuidString, "session_id": session.uuidString,
            "generation_id": generation.uuidString, "audio_session_id": "offline-audio-session",
            "include_jarvis_audio": include, "max_bytes": bytes, "max_duration_ms": duration])
    }
    static func receipt(_ entry: Store.Entry, owner: String = "offline-owner") throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "scope": ["recording_id": entry.recordingID.uuidString, "owner_id": owner,
                "session_id": entry.sessionID.uuidString, "area_id": UUID().uuidString,
                "include_jarvis_audio": entry.includeJarvisAudio, "transfer_policy": "mp4_capture_v1",
                "max_bytes": entry.maxBytes, "max_duration_ms": entry.maxDurationMilliseconds,
                "admission_event_id": UUID().uuidString, "admission_payload_id": UUID().uuidString],
            "operation_id": entry.recordingID.uuidString, "outcome": "saved", "receipt_event_id": UUID().uuidString,
            "receipt_payload_id": UUID().uuidString, "replayed": false, "capture_allowed": false,
            "publication_id": UUID().uuidString, "file_id": UUID().uuidString, "source_id": UUID().uuidString,
            "version_id": "offline-version-1"], options: [.sortedKeys])
    }
    static func inode(_ url: URL) throws -> UInt64 {
        var value = stat()
        guard lstat(url.path, &value) == 0 else { throw Manifest.Failure.ioFailure }
        return UInt64(value.st_ino)
    }

    static func main() async throws {
        guard let output = ProcessInfo.processInfo.environment["JARVIS_RECORDING_TEST_OUTPUT"] else { fatalError("Private output required") }
        let base = URL(fileURLWithPath: output)
        let root = base.appendingPathComponent("pending")
        let store = try Store(root: root)
        let scope = try Store.Scope(context: context())
        let admitted = try command()
        let reserved = try await store.reserve(command: admitted, scope: scope)
        let original = reserved.destination.directoryURL.appendingPathComponent("original.mp4")
        require(!FileManager.default.fileExists(atPath: original.path), "Reservation launched capture")
        require(try await store.pending(scope: scope).isEmpty, "Reservation was called complete")
        let unfinished = try await store.unresolved(scope: scope)
        require(unfinished.count == 1 && unfinished[0].reason == .awaitingFinalization && unfinished[0].observedFileBytes == nil,
            "Reservation recovery invented media")
        let reopened = try Store(root: root)
        let replay = try await reopened.reserve(command: admitted, scope: scope)
        require(replay.destination.directoryURL == reserved.destination.directoryURL &&
            replay.destination.generationID == reserved.destination.generationID && replay.reservedAt == reserved.reservedAt, "Reservation replay minted new custody")
        await reject(.conflict) { _ = try await reopened.reserve(command: command(id: admitted.recordingID), scope: scope) }
        await reject(.conflict) { _ = try await reopened.reserve(command: command(id: admitted.recordingID, session: admitted.sessionID,
            generation: admitted.generationID, bytes: 32 * 1024 * 1024), scope: scope) }
        await reject(.conflict) { _ = try await reopened.reserve(command: command(id: admitted.recordingID, session: admitted.sessionID,
            generation: admitted.generationID, include: false), scope: scope) }
        await reject(.invalidCommand) { _ = try await reopened.reserve(command: command(action: "stop"), scope: scope) }
        let otherOwner = try Store.Scope(context: context(owner: "other-owner"))
        let otherBackend = try Store.Scope(context: context(backend: "b"))
        require(try await reopened.unresolved(scope: otherOwner).isEmpty, "Cross-account reservation")
        require(try await reopened.unresolved(scope: otherBackend).isEmpty, "Cross-backend reservation")
        require(try await reopened.unresolved(scope: .init(context: context())).count == 1, "Reconnect hid custody")
        print("PASS 1: durable pre-capture reservation, exact replay/full binding and stable account/backend scope")

        let full = try Store(root: base.appendingPathComponent("full-budget"))
        for _ in 0..<8 { _ = try await full.reserve(command: command(), scope: scope) }
        await reject(.capacityExceeded) { _ = try await full.reserve(command: command(), scope: scope) }
        let fullRecovery = try await full.unresolved(scope: scope)
        require(fullRecovery.count == 8 && fullRecovery.allSatisfy { $0.observedFileBytes == nil }, "Budget was charged after capture")
        let fullCount = try Store(root: base.appendingPathComponent("full-count"))
        for _ in 0..<128 { _ = try await fullCount.reserve(command: command(bytes: 1), scope: scope) }
        await reject(.capacityExceeded) { _ = try await fullCount.reserve(command: command(bytes: 1), scope: scope) }
        require(try await fullCount.unresolved(scope: scope).count == 128, "Entry cap lost reservations")
        print("PASS 2: 512MiB admitted-budget and128-entry rejection BEFORE any capture/file exists")

        let completed = try await encodedOriginal(destination: reserved.destination)
        require(try completed.file.urlForReading().standardizedFileURL == original.standardizedFileURL, "Writer ignored reservation")
        let before = try inode(original)
        let entry = try await store.retain(command: admitted, completed: completed, scope: scope)
        require(try inode(entry.fileURL) == before, "Finalization copied or replaced the original")
        let same = try await reopened.retain(command: admitted, completed: completed, scope: scope)
        require(same.descriptor == entry.descriptor && same.retainedAt == entry.retainedAt && same.fileURL == entry.fileURL, "Retry changed finalization")
        let listed = try await Store(root: root).pending(scope: scope)
        require(listed.count == 1 && listed[0].descriptor == entry.descriptor, "Restart lost exact descriptor")
        require(try await reopened.unresolved(scope: scope).isEmpty, "Finalized entry still unresolved")
        let report = try Manifest.decode(entry.descriptor, binding: entry.binding).clientReport
        require(report.videoFrames == 1 && report.sourceAudioFrames == 2400 && report.paddedAudioFrames == 2400 && report.timelineEndMS == 100,
            "Writer evidence changed")
        require(entry.generationID == admitted.generationID && entry.writerGenerationID == reserved.destination.generationID, "Voice/encoder identity conflated")
        // External directory already contains bytes: an exact reservation replay
        // must not permit a second Writer to truncate/overwrite that original.
        await reject { _ = try await encodedOriginal(destination: replay.destination) }
        require(try inode(original) == before, "Second writer changed original")
        print("PASS 3: actual H264/AAC writes directly into custody; same inode adoption and exact retry descriptor")

        let failedCommand = try command()
        let failedReservation = try await store.reserve(command: failedCommand, scope: scope)
        let weakBad = WeakFile()
        try await failFinalization(store: store, command: failedCommand, reservation: failedReservation, scope: scope, blockJournal: false, weak: weakBad)
        try await waitForFileRelease(weakBad)
        require(weakBad.value == nil, "Failure fixture still owns completed file")
        let failedFile = failedReservation.destination.directoryURL.appendingPathComponent("original.mp4")
        require(FileManager.default.fileExists(atPath: failedFile.path), "Bad descriptor/deinit deleted owned bytes")
        let blockedCommand = try command()
        let blockedReservation = try await store.reserve(command: blockedCommand, scope: scope)
        let weakBlocked = WeakFile()
        try await failFinalization(store: store, command: blockedCommand, reservation: blockedReservation, scope: scope, blockJournal: true, weak: weakBlocked)
        try await waitForFileRelease(weakBlocked)
        require(weakBlocked.value == nil, "Journal fixture still owns completed file")
        let partialCommand = try command()
        let partialReservation = try await store.reserve(command: partialCommand, scope: scope)
        let partial = partialReservation.destination.directoryURL.appendingPathComponent("original.mp4")
        try Data([1, 2, 3]).write(to: partial)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: partial.path)
        let recovery = try await Store(root: root).unresolved(scope: scope)
        require(recovery.count == 3 && recovery.filter { $0.reason == .awaitingFinalization }.count == 2 &&
            recovery.filter { $0.reason == .invalidFinalization }.count == 1, "Failed/partial recovery misclassified as saved")
        require(recovery.first { $0.recordingID == partialCommand.recordingID }?.observedFileBytes == 3, "Partial bytes were lost")
        require(try await reopened.pending(scope: scope).count == 1, "Unfinalized bytes became uploadable")
        require(try await reopened.unresolved(scope: otherOwner).isEmpty, "Unresolved custody crossed account")
        print("PASS 4: descriptor/journal failure plus actual LocalFile deinit retain originals; partials survive restart separately")

        var details = completed.evidence
        details.intervals = (0..<128).map { index in .init(track: index % 2 == 0 ? .audio : .video,
            start: .zero, end: CMTime(value: 1, timescale: 100), reason: .audioTimeNotOnSampleGrid, inferredVideoDuration: false) }
        details.truncatedIntervalDetails = 7
        details.intervals.append(.init(track: .audio, start: .zero, end: CMTime(value: 1, timescale: 1), reason: .durationLimit, inferredVideoDuration: false))
        let verbose = JarvisRecordingWriter.Completed(generationID: completed.generationID, file: completed.file,
            requestedEnd: completed.requestedEnd, evidence: details)
        let text = try Manifest.make(binding: entry.binding, completed: verbose, expectedWriterGenerationID: reserved.destination.generationID)
        let compact = try Manifest.decode(text, binding: entry.binding).clientReport
        require(text.utf8.count <= 8192 && text.utf8.allSatisfy { $0 < 128 } &&
            compact.omittedIntervalDetails + compact.missingIntervals.count == 136, "Gap truncation lost truthful omitted count")
        await reject(.invalidCompleted) { _ = try Manifest.make(binding: .init(command: command(bytes: 1)), completed: completed, expectedWriterGenerationID: completed.generationID) }
        await reject(.invalidCompleted) { _ = try Manifest.make(binding: .init(command: command(duration: 99)), completed: completed, expectedWriterGenerationID: completed.generationID) }
        await reject(.invalidCompleted) { _ = try Manifest.make(binding: entry.binding, completed: completed, expectedWriterGenerationID: UUID()) }
        let linked = try await store.confirmSaved(receiptData: receipt(entry), recordingID: entry.recordingID, scope: scope)
        require(linked.confirmedLinkage != nil && FileManager.default.fileExists(atPath: original.path), "Saved linkage released custody")
        await reject(.invalidReceipt) { _ = try await store.confirmSaved(receiptData: receipt(entry, owner: "wrong-owner"), recordingID: entry.recordingID, scope: scope) }
        require(try await reopened.pending(scope: scope).first?.descriptor == entry.descriptor, "Confirmation changed exact descriptor")
        print("PASS 5: bounded descriptor, truthful source/padding/gap counts, admitted limits and non-deleting saved linkage")

        let external = try await encodedOriginal(destination: nil)
        await reject(.unreservedFile) { _ = try await store.retain(command: command(), completed: external, scope: scope) }
        let reservedExternalCommand = try command()
        let reservedExternal = try await store.reserve(command: reservedExternalCommand, scope: scope)
        await reject(.unreservedFile) { _ = try await store.retain(command: reservedExternalCommand, completed: external, scope: scope) }
        require(!FileManager.default.fileExists(atPath: reservedExternal.destination.directoryURL.appendingPathComponent("original.mp4").path), "External temp bytes were copied into custody")
        external.file.discard()
        let namespace = reserved.destination.directoryURL.deletingLastPathComponent()
        let lock = open(namespace.appendingPathComponent(".lock").path, O_RDWR | O_NOFOLLOW)
        require(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0, "Fixture lock")
        await reject(.busy) { _ = try await reopened.reserve(command: command(), scope: scope) }
        flock(lock, LOCK_UN); close(lock)
        let cancelled = Task { try await reopened.retain(command: admitted, completed: completed, scope: scope) }
        cancelled.cancel()
        do { _ = try await cancelled.value; fatalError("Cancelled retain returned success") } catch is CancellationError {}
        completed.file.discard()
        require(try await reopened.pending(scope: scope).count == 1, "Cancelled retention/discard erased external-owned original")
        print("PASS 6: unreserved external files strictly rejected, cross-instance lock and cancellation preserve custody")

        // Integrity errors remain recoverable originals, never valid pending entries.
        require(chmod(original.path, 0o600) == 0, "Fixture chmod")
        let handle = try FileHandle(forWritingTo: original)
        try handle.write(contentsOf: Data([0xff])); try handle.close()
        require(try await reopened.pending(scope: scope).isEmpty, "Corrupt original was uploadable")
        require(try await reopened.unresolved(scope: scope).contains { $0.recordingID == entry.recordingID && $0.reason == .invalidFinalization }, "Integrity failure not surfaced")
        let symbolCommand = try command()
        let symbolReservation = try await store.reserve(command: symbolCommand, scope: scope)
        let symbolCompleted = try await encodedOriginal(destination: symbolReservation.destination)
        let symbol = try await store.retain(command: symbolCommand, completed: symbolCompleted, scope: scope)
        try FileManager.default.removeItem(at: symbol.fileURL)
        try FileManager.default.createSymbolicLink(at: symbol.fileURL, withDestinationURL: partial)
        require(try await reopened.pending(scope: scope).isEmpty, "Symlink became pending media")
        require(try await reopened.unresolved(scope: scope).contains { $0.recordingID == symbol.recordingID && $0.reason == .invalidFinalization }, "Symlink failure not surfaced")
        print("PASS 7: bounded hash/no-symlink validation routes damaged originals to unresolved without deletion")
        print("PASS: 7 PRE-CAPTURE CUSTODY groups; actual synthetic encoding, no physical sync/capture/server-save claim")
    }

    @MainActor static func failFinalization(store: Store, command: JarvisRecordingControl.Command,
        reservation: Store.Reservation, scope: Store.Scope, blockJournal: Bool, weak: WeakFile) async throws {
        let completed = try await encodedOriginal(destination: reservation.destination)
        weak.value = completed.file
        if blockJournal {
            try FileManager.default.createDirectory(at: reservation.destination.directoryURL.appendingPathComponent("journal.json"),
                withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
            await reject { _ = try await store.retain(command: command, completed: completed, scope: scope) }
        } else {
            let invalid = JarvisRecordingWriter.Completed(generationID: completed.generationID, file: completed.file,
                requestedEnd: CMTime(value: 1, timescale: 1000), evidence: completed.evidence)
            await reject(.invalidCompleted) { _ = try await store.retain(command: command, completed: invalid, scope: scope) }
        }
        // No returned strong file handle: the caller checks actual deinit and
        // bytes after reopening the store. Writer's external ownership is real.
    }

    static func encodedOriginal(destination: JarvisRecordingWriter.Destination?) async throws -> JarvisRecordingWriter.Completed {
        var configuration = JarvisRecordingWriter.Configuration(width: 32, height: 32)
        configuration.destination = destination
        let writer = try await JarvisRecordingWriter.make(configuration: configuration)
        var pixel: CVPixelBuffer?
        require(CVPixelBufferCreate(kCFAllocatorDefault, 32, 32, kCVPixelFormatType_32BGRA, nil, &pixel) == kCVReturnSuccess, "Pixel allocation")
        CVPixelBufferLockBaseAddress(pixel!, [])
        memset(CVPixelBufferGetBaseAddress(pixel!), 90, CVPixelBufferGetDataSize(pixel!))
        CVPixelBufferUnlockBaseAddress(pixel!, [])
        var format: CMVideoFormatDescription?
        require(CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixel!, formatDescriptionOut: &format) == noErr, "Video format")
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: 10), presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        require(CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixel!, formatDescription: format!, sampleTiming: &timing, sampleBufferOut: &sample) == noErr, "Video sample")
        require(writer.offerVideo(sample!, generationID: writer.generationID) == .queued, "Video offer")
        require(writer.offerAudio([Float](repeating: 0.1, count: 2400), at: CMTime(value: 1, timescale: 20), generationID: writer.generationID) == .queued, "Audio offer")
        return try await writer.finish(at: CMTime(value: 1, timescale: 10), generationID: writer.generationID)
    }
}
