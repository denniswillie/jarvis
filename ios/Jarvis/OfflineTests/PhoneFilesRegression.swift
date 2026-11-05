import Foundation

@MainActor
private final class PhoneFilesFixture: JarvisFileAccess {
    var file: JarvisRetainedFile
    var listReads = 0, metadataReads = 0, byteReads = 0, deletes = 0
    var queries: [String?] = []
    var holdList = false, holdMetadata = false, holdBytes = false, holdDelete = false
    var listWait: CheckedContinuation<JarvisRetainedFilePage, Error>?
    var metadataWait: CheckedContinuation<JarvisRetainedFile, Error>?
    var byteWait: CheckedContinuation<Data, Error>?
    var deleteWait: CheckedContinuation<JarvisFileDeletion, Error>?
    var deleteError: Error?
    var listError: Error?
    let bytes = Data("saved".utf8)
    init(file: JarvisRetainedFile) { self.file = file }

    func getRetainedFiles(query: String?) async throws -> JarvisRetainedFilePage {
        listReads += 1; queries.append(query)
        if holdList { return try await withCheckedThrowingContinuation { listWait = $0 } }
        if let listError { throw listError }
        return JarvisRetainedFilePage(files: [file], pageFull: false)
    }
    func getRetainedFile(id: String) async throws -> JarvisRetainedFile {
        metadataReads += 1
        if holdMetadata { return try await withCheckedThrowingContinuation { metadataWait = $0 } }
        return file
    }
    func getRetainedFileContent(file: JarvisRetainedFile) async throws -> Data {
        byteReads += 1
        if holdBytes { return try await withCheckedThrowingContinuation { byteWait = $0 } }
        return bytes
    }
    func deleteRetainedFile(id: String) async throws -> JarvisFileDeletion {
        deletes += 1
        if holdDelete { return try await withCheckedThrowingContinuation { deleteWait = $0 } }
        if let deleteError { throw deleteError }
        return JarvisFileDeletion(deleted: true, storageCleanup: "pending", fullErasureVerified: false)
    }
}

@main
enum PhoneFilesRegression {
    @MainActor
    static func main() async throws {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        func file(_ id: String = UUID().uuidString, name: String = "note.txt") -> JarvisRetainedFile {
            JarvisRetainedFile(id: id, filename: name, contentType: "text/plain", sizeBytes: 5,
                description: "A retained result", source: "generated", createdAt: "2026-11-01T22:00:00Z",
                memorySource: JarvisSourceEvidence(sourceId: id, revision: 1, generation: 1,
                    contentDigest: "fixture", sourceType: "file", trust: "user_provided", sourceTime: nil,
                    timeBasis: "recorded", recordedAt: "2026-11-01T22:00:00Z", isCurrent: true,
                    ingestionStatus: "ready"))
        }
        @MainActor func waitFor(_ predicate: () -> Bool) async throws {
            for _ in 0..<100 {
                if predicate() { return }
                try await Task.sleep(nanoseconds: 1_000_000)
            }
            preconditionFailure("Fixture operation never began")
        }
        let original = file()
        let fixture = PhoneFilesFixture(file: original)
        let model = JarvisFilesViewModel(api: fixture)
        model.query = "  note  "
        await model.reload()
        check(model.files == [original] && fixture.queries.last! == "note", "List uses the user's trimmed query")

        fixture.holdList = true
        let oldList = Task { await model.reload() }
        try await waitFor { fixture.listWait != nil }
        fixture.holdList = false
        let newer = file(name: "newer.txt"); fixture.file = newer
        await model.reload()
        fixture.listWait?.resume(returning: JarvisRetainedFilePage(files: [original], pageFull: true)); fixture.listWait = nil
        await oldList.value
        check(model.files == [newer] && !model.pageFull, "Late old searches cannot replace a newer list")

        fixture.file = file(original.id, name: "fresh-name.txt")
        let opened = await model.open(original)
        check(opened?.file.filename == "fresh-name.txt" && model.content == fixture.bytes,
              "Opening reauthorizes fresh metadata before fetching bytes")
        fixture.file = newer
        let byteReads = fixture.byteReads
        let mismatched = await model.open(original)
        check(mismatched == nil && fixture.byteReads == byteReads && model.content == nil,
              "Mismatched metadata cannot fetch or export another file")

        fixture.file = original; fixture.holdMetadata = true
        let metadataOpen = Task { await model.open(original) }
        try await waitFor { fixture.metadataWait != nil }
        model.closePreview()
        fixture.metadataWait?.resume(returning: original); fixture.metadataWait = nil; fixture.holdMetadata = false
        let lateMetadata = await metadataOpen.value
        check(lateMetadata == nil && fixture.byteReads == byteReads && model.selectedFile == nil,
              "Closing during metadata read prevents the byte request")

        fixture.holdBytes = true
        let byteOpen = Task { await model.open(original) }
        try await waitFor { fixture.byteWait != nil }
        model.closePreview()
        fixture.byteWait?.resume(returning: fixture.bytes); fixture.byteWait = nil; fixture.holdBytes = false
        let lateBytes = await byteOpen.value
        check(lateBytes == nil && model.content == nil && !model.isLoadingContent,
              "Closing during byte read prevents late preview and export")

        fixture.holdList = true
        let hiddenList = Task { await model.reload() }
        try await waitFor { fixture.listWait != nil }
        let revision = model.presentationRevision
        model.suspend()
        fixture.listWait?.resume(returning: JarvisRetainedFilePage(files: [original], pageFull: false)); fixture.listWait = nil
        await hiddenList.value
        check(model.presentationRevision != revision && model.files.isEmpty && model.content == nil,
              "Background invalidates content and export presentation")
        let readsBeforeHidden = fixture.listReads + fixture.metadataReads + fixture.deletes
        await model.reload(); _ = await model.open(original); _ = await model.remove(original)
        check(fixture.listReads + fixture.metadataReads + fixture.deletes == readsBeforeHidden,
              "Suspended presentation cannot begin reads or deletion")
        fixture.holdList = false
        await model.resume()
        check(model.files == [original] && fixture.listReads + fixture.metadataReads + fixture.deletes == readsBeforeHidden + 1,
              "Foreground returns through a fresh authorized list")

        fixture.holdDelete = true
        let deletion = Task { await model.remove(original) }
        try await waitFor { fixture.deleteWait != nil }
        check(model.content == nil && model.selectedFile == nil && model.isDeleting,
              "Deletion immediately closes any retained preview")
        model.suspend(); await model.resume()
        let readsBeforeLateDelete = fixture.listReads
        let duplicate = await model.remove(original)
        check(!duplicate && fixture.deletes == 1, "Foreground does not duplicate a deletion still in flight")
        fixture.deleteWait?.resume(returning: JarvisFileDeletion(deleted: true, storageCleanup: "pending", fullErasureVerified: false))
        fixture.deleteWait = nil; fixture.holdDelete = false
        let hiddenDeletion = await deletion.value
        check(!hiddenDeletion && fixture.listReads == readsBeforeLateDelete && model.deletionMessage == nil && !model.isDeleting,
              "Retired deletion result cannot reload or claim success in a newer presentation")

        let removed = await model.remove(original)
        check(removed && fixture.listReads == readsBeforeLateDelete + 1,
              "Visible confirmed deletion refreshes the current library once")
        check(model.deletionMessage?.contains("Exported copies are not removed") == true,
              "Deletion receipt preserves the exported-copy distinction")
        fixture.deleteError = URLError(.networkConnectionLost)
        let failed = await model.remove(original)
        check(!failed && model.deletionMessage?.contains("not confirmed") == true,
              "Lost deletion response remains unconfirmed with no automatic retry")

        fixture.holdMetadata = true
        let retiringOpen = Task { await model.open(original) }
        try await waitFor { fixture.metadataWait != nil }
        model.deactivate()
        fixture.metadataWait?.resume(returning: original); fixture.metadataWait = nil
        let retired = await retiringOpen.value
        check(retired == nil && model.query.isEmpty && model.content == nil && model.deletionMessage == nil,
              "Account/view retirement suppresses pending preview and local search state")
        let retiredReads = fixture.listReads + fixture.metadataReads + fixture.deletes
        await model.resume(); await model.reload(); _ = await model.open(original); _ = await model.remove(original)
        check(fixture.listReads + fixture.metadataReads + fixture.deletes == retiredReads,
              "A retired model cannot be resurrected by a queued foreground callback")

        let cancelFixture = PhoneFilesFixture(file: original)
        let cancelModel = JarvisFilesViewModel(api: cancelFixture)
        let cancelled = Task {
            await cancelModel.reload(); _ = await cancelModel.open(original); _ = await cancelModel.remove(original)
        }
        cancelled.cancel(); await cancelled.value
        check(cancelFixture.listReads + cancelFixture.metadataReads + cancelFixture.deletes == 0,
              "UI tasks cancelled before scheduling cannot start effects")

        let retiredDeleteFixture = PhoneFilesFixture(file: original); retiredDeleteFixture.holdDelete = true
        let retiredDeleteModel = JarvisFilesViewModel(api: retiredDeleteFixture)
        let retiringDelete = Task { await retiredDeleteModel.remove(original) }
        try await waitFor { retiredDeleteFixture.deleteWait != nil }
        retiredDeleteModel.deactivate()
        retiredDeleteFixture.deleteWait?.resume(throwing: URLError(.networkConnectionLost)); retiredDeleteFixture.deleteWait = nil
        let retiredDelete = await retiringDelete.value
        check(!retiredDelete && retiredDeleteModel.deletionMessage == nil && retiredDeleteFixture.listReads == 0,
              "A late deletion error cannot repopulate the retired account UI")

        let refreshFixture = PhoneFilesFixture(file: original); refreshFixture.holdList = true
        let refreshModel = JarvisFilesViewModel(api: refreshFixture)
        let deleteRefresh = Task { await refreshModel.remove(original) }
        try await waitFor { refreshFixture.listWait != nil }
        refreshModel.deactivate()
        refreshFixture.listWait?.resume(returning: JarvisRetainedFilePage(files: [], pageFull: false)); refreshFixture.listWait = nil
        let lateRefresh = await deleteRefresh.value
        check(!lateRefresh && refreshModel.deletionMessage == nil && refreshModel.files.isEmpty,
              "Retirement during post-delete reload cannot dismiss a newer destination")

        cancelFixture.listError = URLError(.notConnectedToInternet)
        await cancelModel.reload()
        check(cancelModel.errorMessage != nil && !cancelModel.isLoading,
              "Unavailable library shows a failure rather than a successful empty result")
        print("Phone Files regression: \(checks) checks passed")
    }
}
