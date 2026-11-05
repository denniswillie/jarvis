import CryptoKit
import Foundation

@MainActor
private final class MediaWorkspaceFixture: JarvisWorkspaceAccess, JarvisWorkspaceMediaAccess {
    let offer: JarvisWorkspaceRequest
    let source: URL
    let verifier: JarvisTemporaryFileVerifier
    let byteCount: Int
    let contentSHA256: String
    var acknowledgements: [JarvisWorkspaceAcknowledgement] = []
    var mediaReads = 0, dataReads = 0
    var holdMedia = false, stripPolicy = false
    var mediaError: Error?
    var displayError: Error?
    var beforeDisplay: (() -> Void)?
    var pending: CheckedContinuation<Void, Never>?
    var temporaryURL: URL?
    var playbackDurationMilliseconds = 30_000
    init(offer: JarvisWorkspaceRequest, source: URL, root: URL, byteCount: Int, contentSHA256: String) {
        self.offer = offer; self.source = source; verifier = JarvisTemporaryFileVerifier(root: root)
        self.byteCount = byteCount; self.contentSHA256 = contentSHA256
    }



    func listWorkspaceRequests(sessionID: UUID) async throws -> JarvisWorkspaceRequestPage { .init(items: [offer]) }
    func acknowledgeWorkspaceRequest(requestID: UUID, ack: JarvisWorkspaceAcknowledgement) async throws -> JarvisWorkspaceRequestResponse {
        precondition(requestID == offer.id)
        if ack.state == .displayed {
            beforeDisplay?()
            if let displayError { throw displayError }
        }
        acknowledgements.append(ack)
        var response = JarvisWorkspaceRequest(requestID: offer.id, action: offer.action, artifactKind: offer.artifactKind,
            artifactID: offer.artifactID, versionID: offer.versionID, format: offer.format, contentReady: offer.contentReady,
            fileID: offer.fileID, byteCount: offer.byteCount, contentSHA256: offer.contentSHA256, state: ack.state,
            createdAt: offer.createdAt, expiresAt: offer.expiresAt, errorCode: ack.errorCode, localReceipt: ack.localReceipt)
        if !stripPolicy { response.transferPolicy = offer.transferPolicy; response.transferLimitBytes = offer.transferLimitBytes }
        return .init(request: response)
    }
    func getWorkspaceFileContent(fileID: UUID, sessionID: UUID) async throws -> JarvisWorkspaceFileContent {
        dataReads += 1; throw unexpected()
    }
    func getWorkspaceMediaContent(request: JarvisWorkspaceRequest, sessionID: UUID,
                                  access: JarvisWorkspaceLocalAccess) async throws -> JarvisWorkspaceMediaFileContent {
        precondition(request.hasSameTarget(as: offer))
        mediaReads += 1
        if let mediaError { throw mediaError }
        let file = try await verifier.copyVerifiedFile(from: source, expectedByteCount: byteCount,
            expectedSHA256: contentSHA256,
            maximumByteCount: offer.transferLimitBytes ?? JarvisWorkspaceRequest.MediaTransferPolicy.output.maximumByteCount,
            access: access, format: .mp4)
        temporaryURL = try file.urlForReading()
        if holdMedia { await withCheckedContinuation { pending = $0 } }
        return .init(temporaryFile: file, filename: "Saved clip.mp4", contentType: "video/mp4",
                     maximumDurationMilliseconds: playbackDurationMilliseconds)
    }
    func release() { pending?.resume(); pending = nil }
    private func unexpected() -> JarvisWorkspaceError { .init(code: "unexpected_route", message: "Unexpected document route") }
}

@main
enum PhoneMediaWorkspaceRegression {
    @MainActor
    static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let reader = try FileHandle(forReadingFrom: source)
        var hash = SHA256(), size = 0
        while let chunk = try reader.read(upToCount: 65_536), !chunk.isEmpty { hash.update(data: chunk); size += chunk.count }
        try reader.close()
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("jarvis-media-workspace-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let fileID = UUID(), session = UUID()
        let captureFixture = CommandLine.arguments.contains("--capture")
        let policy: JarvisWorkspaceRequest.MediaTransferPolicy = captureFixture ? .capture : .output
        check(JarvisWorkspaceRequest.MediaTransferPolicy.output.maximumByteCount == 500_000_000,
              "Edited video byte ceiling is 500 decimal MB")
        check(JarvisWorkspaceRequest.MediaTransferPolicy.output.maximumDurationMilliseconds == 1_800_000,
              "Edited video duration ceiling is 30 minutes")
        func boundaryOffer(bytes: Int, limit: Int) -> JarvisWorkspaceRequest {
            .init(requestID: UUID(), action: .download, artifactKind: "file", artifactID: fileID,
                versionID: "boundary", format: nil, contentReady: true, fileID: fileID,
                byteCount: bytes, contentSHA256: digest, state: .pending,
                createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil,
                transferPolicy: .output, transferLimitBytes: limit)
        }
        let exactLimit = try boundaryOffer(bytes: 500_000_000, limit: 500_000_000).validatedTransferByteLimit()
        check(exactLimit == 500_000_000, "Exact output byte ceiling is accepted")
        for offered in [boundaryOffer(bytes: 500_000_001, limit: 500_000_000),
                        boundaryOffer(bytes: 1, limit: 500_000_001),
                        boundaryOffer(bytes: 40_000_000, limit: 32 * 1024 * 1024)] {
            do {
                _ = try offered.validatedTransferByteLimit()
                preconditionFailure("Oversize content or widened persisted limit accepted")
            } catch { checks += 1 }
        }
        func offer(action: JarvisWorkspaceRequest.Action = .download) -> JarvisWorkspaceRequest {
            JarvisWorkspaceRequest(requestID: UUID(), action: action, artifactKind: "file", artifactID: fileID,
                versionID: "exact-video-version", format: nil, contentReady: true, fileID: action == .download ? fileID : nil,
                byteCount: action == .download ? size : nil, contentSHA256: action == .download ? digest : nil,
                state: .pending, createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil,
                transferPolicy: action == .download ? policy : nil, transferLimitBytes: action == .download ? policy.maximumByteCount : nil)
        }
        func build(_ name: String, enabled: Bool = true, action: JarvisWorkspaceRequest.Action = .download) throws -> (MediaWorkspaceFixture, JarvisWorkspaceLocalStore, JarvisWorkspaceViewModel) {
            let fixture = MediaWorkspaceFixture(offer: offer(action: action), source: source, root: root, byteCount: size, contentSHA256: digest)
            fixture.playbackDurationMilliseconds = policy.maximumDurationMilliseconds
            let store = try JarvisWorkspaceLocalStore(ownerKey: name, root: root, allowsFileBytes: enabled)
            let model = JarvisWorkspaceViewModel(api: fixture, sessionID: session, store: store, automaticPolling: false)
            model.setVisibility(true, foreground: true)
            return (fixture, store, model)
        }
        let (api, store, model) = try build("owner-A")
        await model.perform(api.offer)
        check(api.mediaReads == 1 && api.dataReads == 0, "Save action uses the file-based media path exactly once")
        check(api.acknowledgements.map(\.state) == [.claimed, .savedLocal], "Claim precedes the exact durable saved receipt")
        check(api.acknowledgements.last?.localReceipt?.contentSHA256 == digest
              && api.acknowledgements.last?.localReceipt?.byteCount == size, "Receipt describes the actual selected large file")
        check(model.offlineFiles.count == 1 && model.deliveryMessage == "Saved for offline access in Jarvis.",
              "Success appears only after durable adoption and acknowledged completion")
        check(api.temporaryURL.map { !FileManager.default.fileExists(atPath: $0.path) } == true,
              "Downloaded temporary file is discarded after durable adoption")
        let entry = model.offlineFiles[0]
        await model.openOffline(entry)
        check(model.isShowingOfflineMedia && model.offlineMediaPreview.player?.currentItem != nil
              && model.offlineMediaPreview.player?.rate == 0, "Saved copy prepares a paused real MP4 player")
        check(api.mediaReads == 1 && api.dataReads == 0 && api.acknowledgements.count == 2,
              "Offline opening neither fetches nor sends another receipt")
        let durable = try await store.offlineFile(id: entry.id, access: JarvisWorkspaceLocalAccess())
        let exported = await model.prepareOfflineExport()
        check(exported && api.mediaReads == 1 && api.acknowledgements.count == 2,
              "Offline export uses verified journal bytes without network reads or new save receipts")
        let export = model.offlineMediaExport!
        let exportURL = try export.file.urlForReading()
        check(exportURL != durable.url && export.file.contentSHA256 == digest,
              "Offline export owns a distinct exact copy, never the retained original")
        model.finishOfflineExport(id: UUID(), completed: true)
        check(model.offlineMediaExport?.id == export.id && model.offlineExportMessage == nil,
              "Only the matching native export completion can report success")
        model.finishOfflineExport(id: export.id, completed: true)
        check(model.offlineExportMessage == "Exported a separate copy."
              && !FileManager.default.fileExists(atPath: exportURL.path)
              && FileManager.default.fileExists(atPath: durable.url.path),
              "Actual completion discards temporary export and preserves durable source")
        let preparedBeforeLock = await model.prepareOfflineExport()
        check(preparedBeforeLock, "A later independent export can be prepared")
        let heldExport = model.offlineMediaExport!
        let lockedExportURL = try heldExport.file.urlForReading()
        let heldPlayer = model.offlineMediaPreview.player
        model.lockJournal()
        check(model.offlineMediaExport == nil && !FileManager.default.fileExists(atPath: lockedExportURL.path),
              "Account lock cancels and removes an in-flight export")
        do { _ = try heldExport.file.urlForReading(); preconditionFailure("Locked export remained readable") }
        catch { checks += 1 }
        check(!model.isShowingOfflineMedia && heldPlayer?.currentItem == nil && model.offlineFiles.isEmpty,
              "Account lock removes visible entries and releases the player's held item")
        check(FileManager.default.fileExists(atPath: durable.url.path), "Account lock preserves the durable account copy")

        let (gatedAPI, _, gated) = try build("gated", enabled: false)
        await gated.perform(gatedAPI.offer)
        check(gatedAPI.mediaReads == 0 && gatedAPI.acknowledgements.isEmpty, "Closed production gate prevents claim and download")
        let (changedAPI, changedStore, changed) = try build("changed")
        changedAPI.stripPolicy = true
        await changed.perform(changedAPI.offer)
        check(changedAPI.mediaReads == 0 && changedAPI.acknowledgements.map(\.state) == [.claimed],
              "A claim response that drops the trusted policy cannot start the local effect")
        check(try await changedStore.entry(id: changedAPI.offer.id)?.phase == .prepared,
              "Policy mismatch leaves the effect unstarted")

        let (deniedAPI, _, denied) = try build("denied")
        var authorizationFailures = 0
        denied.onAuthorizationFailure = { authorizationFailures += 1 }
        deniedAPI.mediaError = JarvisAPIError(code: "media_transfer_failed", message: "Denied", statusCode: 403)
        await denied.perform(deniedAPI.offer)
        check(!denied.canSaveOffline && authorizationFailures == 1 && deniedAPI.acknowledgements.map(\.state) == [.claimed],
              "Known content authorization loss locks the journal and reports no saved result")

        let (lateAPI, lateStore, late) = try build("retired")
        lateAPI.holdMedia = true
        let pending = Task { await late.perform(lateAPI.offer) }
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while lateAPI.pending == nil && ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        precondition(lateAPI.pending != nil, "Media fixture did not reach its deliberate completion hold")
        await late.refreshLocalEntries()
        check(late.deliveries.count == 1 && late.unresolvedDeliveries.isEmpty,
              "An active media transfer is not listed as interrupted work")
        late.deactivate(); lateAPI.release(); await pending.value
        let lateEntry = try await lateStore.entry(id: lateAPI.offer.id)
        check(lateAPI.acknowledgements.map(\.state) == [.claimed]
              && lateEntry?.acknowledgement == nil,
              "A retired model cannot save or acknowledge a late download")
        check(lateAPI.temporaryURL.map { !FileManager.default.fileExists(atPath: $0.path) } == true,
              "Late temporary download is discarded on retirement")

        let (openAPI, openStore, open) = try build("open", action: .open)
        await open.presentFreshRequest(openAPI.offer.id)
        check(open.mediaPresentation != nil && open.mediaPreview.player?.currentItem != nil,
              "A fresh exact FILE intent prepares the selected video through the existing claim path")
        check(openAPI.acknowledgements.map(\.state) == [.claimed] && open.offlineFiles.isEmpty,
              "Asset preparation is neither display nor offline-save confirmation")
        check(open.deliveries.count == 1 && open.unresolvedDeliveries.isEmpty,
              "A prepared video awaiting its visible frame does not show a recovery card")
        let selected = open.mediaPresentation!
        await open.didDisplayMedia(presentationID: selected.id, requestID: selected.requestID,
                                   artifactID: selected.artifactID, versionID: selected.versionID)
        check(openAPI.acknowledgements.count == 1, "A loaded item without a first-frame signal cannot ACK display")
        open.mediaReadinessChanged(presentationID: UUID(), ready: true)
        await open.didDisplayMedia(presentationID: selected.id, requestID: selected.requestID,
                                   artifactID: selected.artifactID, versionID: selected.versionID)
        check(openAPI.acknowledgements.count == 1, "Readiness from another presentation is ignored")
        open.mediaReadinessChanged(presentationID: selected.id, ready: true)
        await open.didDisplayMedia(presentationID: selected.id, requestID: selected.requestID,
                                   artifactID: selected.artifactID, versionID: "wrong-version")
        check(openAPI.acknowledgements.count == 1, "A visible callback cannot acknowledge another version")
        await open.didDisplayMedia(presentationID: selected.id, requestID: selected.requestID,
                                   artifactID: selected.artifactID, versionID: selected.versionID)
        await open.didDisplayMedia(presentationID: selected.id, requestID: selected.requestID,
                                   artifactID: selected.artifactID, versionID: selected.versionID)
        check(openAPI.acknowledgements.map(\.state) == [.claimed, .displayed] && open.offlineFiles.isEmpty,
              "A matching renderer witness records displayed exactly once without saving bytes")
        let displayEntry = try await openStore.entry(id: selected.requestID)
        check(displayEntry?.acknowledgement?.state == .displayed && displayEntry?.relativePath == nil,
              "The display journal does not become a durable local-copy entry")
        open.setVisibility(false, foreground: true)
        check(open.mediaPresentation == selected && open.mediaPreview.player?.rate == 0,
              "Leaving Workspace preserves the selected preview while pausing it")
        open.setVisibility(true, foreground: true)
        check(openAPI.acknowledgements.count == 2, "Returning to an already-shown preview does not replay its receipt")
        let openPlayer = open.mediaPreview.player
        open.deactivate()
        check(openPlayer?.currentItem == nil && openAPI.temporaryURL.map { !FileManager.default.fileExists(atPath: $0.path) } == true,
              "Account retirement releases the online preview and temporary bytes")

        let (hiddenAPI, _, hidden) = try build("hidden", action: .open)
        await hidden.perform(hiddenAPI.offer)
        let hiddenSelection = hidden.mediaPresentation!
        hidden.mediaReadinessChanged(presentationID: hiddenSelection.id, ready: true)
        hidden.setVisibility(false, foreground: true)
        await hidden.didDisplayMedia(presentationID: hiddenSelection.id, requestID: hiddenSelection.requestID,
                                     artifactID: hiddenSelection.artifactID, versionID: hiddenSelection.versionID)
        check(hiddenAPI.acknowledgements.map(\.state) == [.claimed], "Hidden/retired presentation callbacks cannot ACK display")
        hidden.deactivate()
        let (uncertainAPI, _, uncertain) = try build("uncertain-display", action: .open)
        await uncertain.perform(uncertainAPI.offer)
        let uncertainSelection = uncertain.mediaPresentation!
        uncertain.mediaReadinessChanged(presentationID: uncertainSelection.id, ready: true)
        uncertainAPI.displayError = URLError(.networkConnectionLost)
        uncertainAPI.beforeDisplay = {
            check(uncertain.unresolvedDeliveries.isEmpty, "An in-flight display confirmation stays out of recovery")
        }
        await uncertain.didDisplayMedia(presentationID: uncertainSelection.id, requestID: uncertainSelection.requestID,
            artifactID: uncertainSelection.artifactID, versionID: uncertainSelection.versionID)
        uncertainAPI.beforeDisplay = nil
        check(uncertain.displayRequestID == nil && uncertain.unresolvedDeliveries.count == 1
              && uncertain.unresolvedDeliveries[0].acknowledgement?.state == .displayed,
              "A failed display confirmation becomes visible recovery with its existing displayed receipt")
        uncertainAPI.displayError = nil
        await uncertain.reconcile(uncertain.unresolvedDeliveries[0])
        check(uncertain.unresolvedDeliveries.isEmpty && uncertainAPI.mediaReads == 1,
              "Retrying confirmation resolves recovery without reopening or saving the video")
        uncertain.deactivate()
        let (navigatedAPI, _, navigated) = try build("navigated", action: .open)
        navigatedAPI.holdMedia = true
        let loading = Task { await navigated.perform(navigatedAPI.offer) }
        let navigationDeadline = ProcessInfo.processInfo.systemUptime + 5
        while navigatedAPI.pending == nil && ProcessInfo.processInfo.systemUptime < navigationDeadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        precondition(navigatedAPI.pending != nil)
        navigated.setVisibility(false, foreground: true)
        navigatedAPI.release(); await loading.value
        check(navigated.mediaPresentation == nil && navigated.mediaPreview.player == nil
              && navigatedAPI.acknowledgements.map(\.state) == [.claimed],
              "Navigation during download cannot leave a pending player or report display")
        check(navigatedAPI.temporaryURL.map { !FileManager.default.fileExists(atPath: $0.path) } == true,
              "A superseded video download discards its temporary bytes")
        let (cancelledAPI, _, cancelled) = try build("cancelled", action: .open)
        cancelledAPI.holdMedia = true
        let cancelledLoad = Task { await cancelled.perform(cancelledAPI.offer) }
        let cancellationDeadline = ProcessInfo.processInfo.systemUptime + 5
        while cancelledAPI.pending == nil && ProcessInfo.processInfo.systemUptime < cancellationDeadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        precondition(cancelledAPI.pending != nil)
        cancelledLoad.cancel(); cancelledAPI.release(); await cancelledLoad.value
        check(cancelled.mediaPresentation == nil && cancelled.mediaPreview.player == nil
              && cancelledAPI.acknowledgements.map(\.state) == [.claimed],
              "Cancellation in a still-visible Workspace clears the unadopted presentation without display")
        check(cancelledAPI.temporaryURL.map { !FileManager.default.fileExists(atPath: $0.path) } == true,
              "Cancelled preparation discards its exact temporary file")
        print("Phone media workspace regressions passed: \(checks) checks; fixture_bytes=\(size); no HTTP, playback or audio")
    }
}
