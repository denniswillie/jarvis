import CryptoKit
import Foundation

@MainActor
private final class MediaWorkspaceFixture: JarvisWorkspaceAccess, JarvisWorkspaceMediaAccess {
    let offer: JarvisWorkspaceRequest
    var additional: JarvisWorkspaceRequest?
    var requestReads = 0
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
    init(offer: JarvisWorkspaceRequest, source: URL, root: URL, byteCount: Int, contentSHA256: String) {
        self.offer = offer; self.source = source; verifier = JarvisTemporaryFileVerifier(root: root)
        self.byteCount = byteCount; self.contentSHA256 = contentSHA256
    }



    func listWorkspaceRequests(sessionID: UUID) async throws -> JarvisWorkspaceRequestPage { requestReads += 1; return .init(items: [offer] + [additional].compactMap { $0 }) }
    func acknowledgeWorkspaceRequest(requestID: UUID, ack: JarvisWorkspaceAcknowledgement) async throws -> JarvisWorkspaceRequestResponse {
        let offer = ([self.offer] + [additional].compactMap { $0 }).first { $0.id == requestID }!
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
        precondition(request.hasSameTarget(as: offer) || additional.map { request.hasSameTarget(as: $0) } == true)
        mediaReads += 1
        if let mediaError { throw mediaError }
        let file = try await verifier.copyVerifiedFile(from: source, expectedByteCount: byteCount,
            expectedSHA256: contentSHA256, maximumByteCount: 32 * 1024 * 1024, access: access, format: .mp4)
        temporaryURL = try file.urlForReading()
        if holdMedia && request.id == offer.id { await withCheckedContinuation { pending = $0 } }
        return .init(temporaryFile: file, filename: "Saved clip.mp4", contentType: "video/mp4")
    }
    func release() { pending?.resume(); pending = nil }
    private func unexpected() -> JarvisWorkspaceError { .init(code: "unexpected_route", message: "Unexpected document route") }
}


@main
enum PhoneRecordingDeliveryChecks {
    @MainActor static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let bytes = try Data(contentsOf: source)
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("recording-delivery-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        var checks = 0
        func check(_ ok: Bool, _ why: String) { precondition(ok, why); checks += 1 }
        let session = UUID(), file = UUID()
        func offer() -> JarvisWorkspaceRequest {
            .init(requestID: UUID(), action: .download, artifactKind: "file", artifactID: file,
                versionID: "source-exact", format: nil, contentReady: true, fileID: file,
                byteCount: bytes.count, contentSHA256: digest, state: .pending,
                createdAt: "fixture", expiresAt: "fixture", errorCode: nil, localReceipt: nil,
                transferPolicy: .capture, transferLimitBytes: 256 * 1024 * 1024)
        }
        func build(_ name: String) throws -> (MediaWorkspaceFixture, JarvisWorkspaceLocalStore, JarvisWorkspaceViewModel) {
            let api = MediaWorkspaceFixture(offer: offer(), source: source, root: root,
                byteCount: bytes.count, contentSHA256: digest)
            let store = try JarvisWorkspaceLocalStore(ownerKey: name, root: root, allowsFileBytes: true)
            let model = JarvisWorkspaceViewModel(api: api, sessionID: session, store: store, automaticPolling: false)
            return (api, store, model)
        }
        let (api, store, model) = try build("hidden")
        model.setVisibility(false, foreground: false)
        let saved = await model.downloadConfirmedRecording(requestID: api.offer.id, fileID: file, versionID: "source-exact")
        check(saved && api.mediaReads == 1 && api.dataReads == 0, "Confirmed recording downloads while gallery is hidden")
        check(api.acknowledgements.map(\.state) == [.claimed, .savedLocal], "Actual saved-local receipt follows claim")
        let entry = try await store.entry(id: api.offer.id, access: JarvisWorkspaceLocalAccess())
        check(entry?.acknowledgement?.localReceipt?.contentSHA256 == digest, "Exact bytes bind saved receipt")
        check(!model.isShowingOfflineMedia && model.offlineMediaPreview.player == nil, "Auto-download never presents a player")
        let local = try await model.localMediaFile(fileID: file, versionID: "source-exact", access: JarvisWorkspaceLocalAccess())
        check(local != nil && FileManager.default.fileExists(atPath: local!.url.path), "Gallery can borrow exact durable copy")
        let wrong = await model.downloadConfirmedRecording(requestID: api.offer.id, fileID: file, versionID: "different")
        check(!wrong && api.mediaReads == 1, "Wrong version performs no transfer")
        model.lockJournal()
        let locked = await model.downloadConfirmedRecording(requestID: api.offer.id, fileID: file, versionID: "source-exact")
        check(!locked && FileManager.default.fileExists(atPath: local!.url.path), "Account lock preserves bytes and denies download")

        let (busyAPI, busyStore, busy) = try build("busy")
        busyAPI.additional = offer(); busyAPI.holdMedia = true
        busy.setVisibility(true, foreground: true)
        let foreground = Task { await busy.perform(busyAPI.offer) }
        for _ in 0..<200 where busyAPI.pending == nil { try await Task.sleep(for: .milliseconds(5)) }
        check(busyAPI.pending != nil, "Foreground transfer held at actual private bytes")
        let second = busyAPI.additional!
        let background = Task { await busy.downloadConfirmedRecording(requestID: second.id, fileID: file, versionID: "source-exact") }
        for _ in 0..<200 where busyAPI.requestReads == 0 { try await Task.sleep(for: .milliseconds(5)) }
        try await Task.sleep(for: .milliseconds(25))
        check(busyAPI.mediaReads == 1 && busy.workingRequestID == busyAPI.offer.id, "Automatic transfer waits behind foreground owner")
        busyAPI.release(); await foreground.value
        let secondSaved = await background.value
        check(secondSaved && busyAPI.mediaReads == 2, "Automatic transfer continues after foreground completion")
        let entries = try await busyStore.entries(access: JarvisWorkspaceLocalAccess())
        check(entries.filter { $0.acknowledgement?.state == .savedLocal }.count == 2, "Both exact independent requests have real receipts")

        let (failedAPI, _, failed) = try build("failed")
        failedAPI.mediaError = URLError(.networkConnectionLost)
        let unconfirmed = await failed.downloadConfirmedRecording(requestID: failedAPI.offer.id, fileID: file, versionID: "source-exact")
        check(!unconfirmed && !failedAPI.acknowledgements.contains { $0.state == .savedLocal }, "Failed download never becomes saved")

        let reference = JarvisResourceReference(fileID: file, versionID: "source-exact", filename: "My clip.mp4", mediaType: "video/mp4")
        let context = JarvisResourceSelectionContext(visibleResources: [reference], selectedResource: reference, pageFull: false)
        let payload = try JarvisResourceSelectionContext.envelope(context, sessionID: session, audioSessionID: "audio-fixture", sequence: 1)
        let wire = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let decoded = try JSONSerialization.jsonObject(with: wire) as! [String: Any]
        let projected = decoded["context"] as! [String: Any]
        let selected = projected["selected_resource"] as! [String: Any]
        check(decoded["type"] as? String == "workspace.resource_context" && decoded["client_event_id"] == nil, "Exact Core envelope has no legacy event identifier")
        check(selected["version_id"] as? String == "source-exact" && selected["file_id"] as? String == file.uuidString, "Full canonical identity survives wire encoding")
        let cleared = try JarvisResourceSelectionContext.envelope(nil, sessionID: session, audioSessionID: "audio-fixture", sequence: 2)
        check(cleared["context"] is NSNull, "Clear uses explicit null")
        do {
            _ = try JarvisResourceSelectionContext.envelope(.init(visibleResources: Array(repeating: reference, count: 33), selectedResource: nil, pageFull: true), sessionID: session, audioSessionID: "audio-fixture", sequence: 3)
            preconditionFailure("Unbounded resource context accepted")
        } catch { checks += 1 }
        do {
            _ = try JarvisResourceSelectionContext.envelope(context, sessionID: session, audioSessionID: "audio-fixture", sequence: 0)
            preconditionFailure("Zero sequence accepted")
        } catch { checks += 1 }
        if CommandLine.arguments.count > 2 { try wire.write(to: URL(fileURLWithPath: CommandLine.arguments[2])) }
        model.deactivate(); busy.deactivate(); failed.deactivate()
        print("Recording delivery/context: \(checks) checks passed; actual private-file transfer, no remote service or device")
    }
}
