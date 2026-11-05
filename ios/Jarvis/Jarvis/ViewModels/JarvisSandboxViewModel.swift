import Combine
import Foundation

@MainActor
final class JarvisSandboxViewModel: ObservableObject {
    enum Phase: Equatable { case idle, loadingGallery, choosing, connecting, viewing, reconnecting, failed }
    struct Notice: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var choices: [JarvisSandboxTarget] = []
    @Published private(set) var hasMore = false
    @Published private(set) var lease: JarvisSandboxLease?
    @Published private(set) var pageRequest: URLRequest?
    @Published private(set) var pageID = UUID()
    @Published private(set) var errorMessage: String?
    @Published var notice: Notice?
    @Published var resumeCandidate: JarvisSandboxTarget?
    @Published private(set) var loadingMore = false
    @Published private(set) var thumbnails: [UUID: JarvisSandboxThumbnail] = [:]
    var onPresentationIntent: ((UUID) async -> Void)?
    var onFirstFrame: ((JarvisSandboxLease) -> Void)?
    var onAuthorizationFailure: ((Error) -> Void)?
    private let api: (any JarvisSandboxAccess)?
    private let sessionID: UUID
    private var generation = UUID()
    private var polling: Task<Void, Never>?
    private var reconnecting = false
    private var reconnectCount = 0
    private var retired = false
    private var reportedFrame = false
    private var selectedTarget: (UUID, String)?
    private var connectingSince: Date?
    private var nextCursor: String?
    private var thumbnailTimes: [UUID: Date] = [:]
    private var loadingThumbnails: Set<String> = []

    init(api: (any JarvisSandboxAccess)?, sessionID: UUID) {
        self.api = api; self.sessionID = sessionID
    }
    isolated deinit { polling?.cancel() }
    var isShowingScreen: Bool { lease != nil || (selectedTarget != nil && (phase == .connecting || phase == .failed)) }

    func thumbnail(for target: JarvisSandboxTarget) -> JarvisSandboxThumbnail? {
        guard let value = thumbnails[target.id], value.generation == target.generation else { return nil }
        return value
    }

    func loadThumbnail(_ target: JarvisSandboxTarget) async {
        guard !retired, target.isDesktop, target.stale != true, let api else { return }
        if thumbnail(for: target) != nil, let loaded = thumbnailTimes[target.id], Date().timeIntervalSince(loaded) < 30 { return }
        let key = target.id.uuidString + target.generation
        guard loadingThumbnails.insert(key).inserted else { return }
        defer { loadingThumbnails.remove(key) }
        let current = generation
        do {
            let value = try await api.sandboxThumbnail(sessionID: sessionID, allocationID: target.id, generation: target.generation)
            guard !retired, !Task.isCancelled, current == generation,
                  choices.contains(where: { $0.id == target.id && $0.generation == target.generation }),
                  value.allocationID == target.id, value.generation == target.generation,
                  value.contentType == "image/jpeg", !value.image.isEmpty, value.image.count <= 32768 else { return }
            thumbnails[target.id] = value; thumbnailTimes[target.id] = Date()
            if thumbnails.count > 16, let oldest = thumbnailTimes.min(by: { $0.value < $1.value })?.key {
                thumbnails.removeValue(forKey: oldest); thumbnailTimes.removeValue(forKey: oldest)
            }
        } catch {
            // A preview failure never opens/resumes a computer or blocks its
            // status/explicit View action. The card keeps its honest fallback.
        }
    }

    func choose() async {
        guard !retired, phase == .idle || phase == .choosing || (phase == .failed && selectedTarget == nil) else { return }
        guard let api else {
            notice = Notice(title: "Screen unavailable", message: "Connect to Jarvis to view your sandbox.")
            return
        }
        let current = generation
        phase = .loadingGallery
        do {
            let result = try await api.sandboxChoices(sessionID: sessionID)
            guard current == generation, !retired else { return }
            choices = result.items; hasMore = result.hasMore; nextCursor = result.nextCursor
            if choices.isEmpty {
                phase = .idle
                notice = Notice(title: "No sandbox to view", message: "Ask Jarvis to start a sandbox, then view its screen here.")
            } else { phase = .choosing }
        } catch { handle(error, current: current) }
    }

    func loadMore() async {
        guard !retired, !loadingMore, let nextCursor, let api else { return }
        loadingMore = true
        defer { loadingMore = false }
        let current = generation
        do {
            let result = try await api.sandboxChoices(sessionID: sessionID, after: nextCursor)
            guard current == generation, !retired else { return }
            let existing = Set(choices.map(\.id))
            choices += result.items.filter { !existing.contains($0.id) }
            hasMore = result.hasMore; self.nextCursor = result.nextCursor
        } catch { handle(error, current: current) }
    }

    func requestView(_ target: JarvisSandboxTarget, resume: Bool = false) async {
        guard !retired, let api else { return }
        if target.isPaused && !resume { resumeCandidate = target; return }
        guard target.isDesktop, target.canOpen || (target.isPaused && resume), let onPresentationIntent else {
            notice = Notice(title: "Screen unavailable", message: target.isDesktop
                ? "Refresh this computer's status before opening it." : "This computer supports terminal work, not a desktop screen.")
            return
        }
        let current = generation
        selectedTarget = (target.allocationID, target.generation)
        phase = .connecting
        do {
            let response = try await api.controlApp(sessionID: sessionID, operationID: UUID(), command: [
                "action": "sandbox.start", "allocation_id": target.allocationID.uuidString.lowercased(),
                "generation": target.generation, "resume": resume])
            guard current == generation, !retired else { return }
            guard let request = response.presentation, request.state == "pending" else {
                throw JarvisAPIError(code: "sandbox_presentation_unconfirmed", message: response.summary)
            }
            await onPresentationIntent(request.requestID)
            if current == generation, lease == nil, !retired {
                selectedTarget = nil; phase = .choosing
                notice = Notice(title: "Display requested", message: "The phone has not confirmed opening this screen yet.")
            }
        } catch { handle(error, current: current) }
    }

    func resumeAndView() async {
        guard let target = resumeCandidate else { return }
        resumeCandidate = nil
        await requestView(target, resume: true)
    }

    func stop() async {
        guard let selected = lease, let api else { close(); return }
        close(sendRevoke: false)
        do {
            _ = try await api.controlApp(sessionID: sessionID, operationID: UUID(), command: [
                "action": "sandbox.stop", "view_id": selected.id.uuidString.lowercased()])
        } catch {
            guard !retired else { return }
            notice = Notice(title: "Closed on this phone", message: "The server stop is unconfirmed. The agent's work was not stopped.")
        }
    }

    func receiveStop(viewID: UUID) {
        guard lease?.id == viewID else { return }
        close(sendRevoke: false)
    }

    @discardableResult
    func open(allocationID: UUID, version: String) async -> Bool {
        guard !retired, let api else { return false }
        close()
        selectedTarget = (allocationID, version)
        let current = generation
        phase = .connecting
        connectingSince = Date()
        do {
            let result = try await api.openSandboxView(sessionID: sessionID, allocationID: allocationID, generation: version)
            guard current == generation, !retired else {
                try? await api.closeSandboxView(sessionID: sessionID, id: result.id)
                return false
            }
            guard result.target.allocationID == allocationID, result.target.generation == version else {
                try? await api.closeSandboxView(sessionID: sessionID, id: result.id)
                throw JarvisAPIError(code: "sandbox_view_identity_mismatch", message: "This screen could not be verified.")
            }
            lease = result
            pageRequest = try api.sandboxPageRequest(sessionID: sessionID, id: result.id)
            pageID = UUID()
            polling = Task { [weak self] in
                while !Task.isCancelled {
                    do { try await Task.sleep(for: .seconds(5)) } catch { return }
                    guard let self, current == self.generation else { return }
                    await self.check(current: current)
                }
            }
            return true
        } catch { handle(error, current: current); return false }
    }

    func pageEvent(_ event: String, pageID: UUID) {
        guard !retired, self.pageID == pageID, lease != nil else { return }
        if event == "frame" {
            phase = .viewing; errorMessage = nil; connectingSince = nil
        } else if event == "disconnected" || event == "failed" {
            phase = .reconnecting
            let current = generation
            Task { await check(current: current, reconnect: true) }
        }
    }

    private func check(current: UUID, reconnect: Bool = false) async {
        guard !retired, current == generation, let lease, let api, !reconnecting else { return }
        reconnecting = true
        defer { reconnecting = false }
        do {
            let status = try await api.sandboxViewStatus(sessionID: sessionID, id: lease.id)
            guard current == generation else { return }
            guard status.id == lease.id, status.target == lease.target else {
                throw JarvisAPIError(code: "sandbox_view_replaced", message: "This sandbox view is no longer current.")
            }
            if let since = connectingSince, Date().timeIntervalSince(since) > 20,
               phase == .connecting || phase == .reconnecting {
                pageRequest = nil; phase = .failed
                errorMessage = "The screen couldn't load. Try again."
                return
            }
            if reconnect {
                if reconnectCount < 1 {
                    reconnectCount += 1
                    connectingSince = Date()
                    pageRequest = try api.sandboxPageRequest(sessionID: sessionID, id: lease.id)
                    pageID = UUID()
                } else {
                    pageRequest = nil; phase = .failed
                    errorMessage = "The screen connection was interrupted."
                }
            }
        } catch { handle(error, current: current) }
    }

    private func handle(_ error: Error, current: UUID) {
        guard current == generation, !retired, !(error is CancellationError) else { return }
        let code = (error as? JarvisAPIError)?.code ?? ""
        onAuthorizationFailure?(error)
        guard !retired, current == generation else { return }
        if code == "sandbox_not_running" {
            close()
            notice = Notice(title: "Sandbox not running", message: "Refresh its card. A paused computer can be resumed explicitly.")
        } else if ["sandbox_view_replaced", "sandbox_view_access_ended", "sandbox_view_expired", "sandbox_view_session_mismatch"].contains(code) {
            close()
            notice = Notice(title: "This view has ended", message: error.localizedDescription)
        } else {
            pageRequest = nil; phase = .failed
            errorMessage = error.localizedDescription
        }
    }

    func retry() async {
        if let selectedTarget { await open(allocationID: selectedTarget.0, version: selectedTarget.1) }
        else { close(); await choose() }
    }

    /// The RFB frame is ready, but a phone presentation receipt additionally
    /// requires its renderer to intersect the active Workspace viewport.
    func confirmVisibleFrame(_ visible: Bool) {
        guard visible, !retired, phase == .viewing, !reportedFrame, let lease else { return }
        reportedFrame = true
        onFirstFrame?(lease)
    }

    func close(sendRevoke: Bool = true) {
        generation = UUID(); polling?.cancel(); polling = nil
        if sendRevoke, let lease, let api {
            let session = sessionID
            Task { try? await api.closeSandboxView(sessionID: session, id: lease.id) }
        }
        pageRequest = nil; pageID = UUID(); lease = nil; choices = []
        phase = .idle; errorMessage = nil; reportedFrame = false; reconnectCount = 0
        reconnecting = false; selectedTarget = nil; connectingSince = nil
        resumeCandidate = nil; nextCursor = nil
    }
    func deactivate() {
        close(); retired = true; notice = nil
        thumbnails.removeAll(); thumbnailTimes.removeAll()
    }
}
