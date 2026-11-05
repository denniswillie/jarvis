import Foundation
import os

@MainActor
final class JarvisVisualIngestClient {
#if DEBUG
    var evaluationRunID: UUID?
    var onReplayFrameSocketWritten: ((JarvisReplayFrameUploadReceipt) -> Void)?
#endif
    var onEvent: ((JarvisRealtimeServerEvent) -> Void)?
    var onDisconnect: ((Error?) -> Void)?

    private let logger = Logger(subsystem: "com.symbiote.jarvis", category: "visual-ingest")
    private let urlSession: URLSession
    private let sendQueue = JarvisVisualWebSocketSendQueue()
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var sendTask: Task<Void, Never>?
    private var frameFlow = JarvisVisualFrameFlow<CapturedVisualFrame>()
    private var acknowledgementTimeout: Task<Void, Never>?
    private var isDisconnectingIntentionally = false
    private var diagnosticContext: [String: Any] = [:]

    init(urlSession: URLSession = .shared) {
        self.urlSession = urlSession
    }

    var isConnected: Bool {
        webSocketTask != nil
    }

    func connect(
        visualWebSocketURL: URL,
        bearerToken: String,
        audioSessionId: String?,
        visualSessionId: String
    ) {
        disconnect()
        isDisconnectingIntentionally = false
        diagnosticContext = ["visual_session_id": visualSessionId, "connection_id": UUID().uuidString]
        diagnosticContext["audio_session_id"] = audioSessionId

        let url = urlWithVisualQuery(
            baseURL: visualWebSocketURL,
            audioSessionId: audioSessionId,
            visualSessionId: visualSessionId
        )
        var request = URLRequest(url: url)
        request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
#if DEBUG
        JarvisReplayScope.applyRunHeader(evaluationRunID, to: &request)
#endif
        logger.info("Connecting visual websocket host=\(url.host ?? "-", privacy: .public) path=\(url.path, privacy: .public)")

        let task = urlSession.webSocketTask(with: request)
        webSocketTask = task
        task.resume()
        startReceiveLoop(task: task)
    }

    func disconnect() {
        if webSocketTask != nil {
            logger.info("Disconnecting visual websocket")
        }
        isDisconnectingIntentionally = true
        receiveTask?.cancel()
        receiveTask = nil
        sendTask?.cancel()
        sendTask = nil
        acknowledgementTimeout?.cancel()
        acknowledgementTimeout = nil
        frameFlow.reset()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
    }

    func enqueueLatestVisualFrame(_ frame: CapturedVisualFrame) {
        guard webSocketTask != nil else { return }
        frameFlow.enqueue(frame)
        sendNextFrameIfPossible()
    }

    private func sendNextFrameIfPossible() {
        guard sendTask == nil, let socket = webSocketTask,
              let frame = frameFlow.takeNext(sequence: { $0.sequence }) else { return }
        acknowledgementTimeout?.cancel()
        acknowledgementTimeout = Task { [weak self, weak socket] in
            do { try await Task.sleep(for: .seconds(12)) } catch { return }
            guard let self, let socket, self.webSocketTask === socket,
                  self.frameFlow.inFlightSequence == frame.sequence else { return }
            let error = JarvisAudioSessionError.serverError("Visual upload acknowledgement timed out; reconnecting with a fresh frame.")
            self.disconnect()
            self.onDisconnect?(error)
        }
        sendTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.sendFrame(frame)
                guard !Task.isCancelled, self.webSocketTask === socket else { return }
                self.sendTask = nil
                // An acknowledgement can arrive before URLSession.send returns.
                self.sendNextFrameIfPossible()
            } catch {
                guard !Task.isCancelled, self.webSocketTask === socket else { return }
                self.logger.error("Visual frame send failed: \(error.localizedDescription, privacy: .public)")
                self.disconnect()
                self.onDisconnect?(error)
            }
        }
    }

    private func sendFrame(_ frame: CapturedVisualFrame) async throws {
        guard let webSocketTask else {
            throw JarvisAudioSessionError.realtimeNotConnected
        }

        let payload: [String: Any] = [
            "type": "visual.frame.capture",
            "client_event_id": UUID().uuidString,
            "frame_sequence": frame.sequence,
            "captured_at": frame.capturedAt.jarvisISO8601String(),
            "source": frame.source.rawValue,
            "content_type": "image/jpeg",
            "width": frame.width,
            "height": frame.height,
            "image_base64": frame.jpegData.base64EncodedString(),
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [])
        guard let text = String(data: data, encoding: .utf8) else {
            throw JarvisAudioSessionError.serverError("Failed to encode visual frame event.")
        }
        let context = diagnosticContext
#if DEBUG
        // Snapshot identity/observer before suspension. A late completion must
        // never inherit a later connection or replay run.
        let replayRun = evaluationRunID
        let replayObserver = onReplayFrameSocketWritten
#endif
        try await sendQueue.send(.string(text), task: webSocketTask)
#if DEBUG
        if let replayObserver, let lineage = frame.replayLineage, lineage.runID == replayRun {
            replayObserver(JarvisReplayFrameUploadReceipt(lineage: lineage,
                uploadedJPEG: frame.jpegData, frameSequence: frame.sequence,
                capturedAtWire: payload["captured_at"] as? String,
                clientEventID: payload["client_event_id"] as? String,
                visualSessionID: context["visual_session_id"] as? String,
                connectionID: context["connection_id"] as? String,
                audioSessionID: context["audio_session_id"] as? String))
        }
#endif
        if let diagnostic = RealtimeDiagnostics.record("camera_frame_socket_written", fields: context.merging(payload) { _, new in new }) {
            logger.info("realtime_trace \(diagnostic, privacy: .public)")
        }
    }

    private func startReceiveLoop(task: URLSessionWebSocketTask) {
        receiveTask = Task { [weak self, weak task] in
            guard let task else { return }
            while !Task.isCancelled {
                do {
                    let message = try await task.receive()
                    guard let event = try Self.decode(message: message) else { continue }
                    await MainActor.run {
                        guard let self, self.webSocketTask === task else { return }
                        if ["visual.frame.accepted", "visual.frame.dropped"].contains(event.type),
                           let sequence = event.frameSequence,
                           self.frameFlow.acknowledge(sequence: sequence) {
                            self.acknowledgementTimeout?.cancel()
                            self.acknowledgementTimeout = nil
                            self.sendNextFrameIfPossible()
                        }
                        self.onEvent?(event)
                    }
                } catch {
                    await MainActor.run {
                        guard let self, self.webSocketTask === task,
                              !self.isDisconnectingIntentionally else { return }
                        self.logger.error("Visual websocket disconnected: \(error.localizedDescription, privacy: .public)")
                        self.disconnect()
                        self.onDisconnect?(error)
                    }
                    return
                }
            }
        }
    }

    private func urlWithVisualQuery(
        baseURL: URL,
        audioSessionId: String?,
        visualSessionId: String
    ) -> URL {
        guard var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else {
            return baseURL
        }
        var items = components.queryItems ?? []
        items.removeAll { item in
            item.name == "audio_session_id" || item.name == "visual_session_id"
        }
        if let audioSessionId, !audioSessionId.isEmpty {
            items.append(URLQueryItem(name: "audio_session_id", value: audioSessionId))
        }
        items.append(URLQueryItem(name: "visual_session_id", value: visualSessionId))
        components.queryItems = items
        return components.url ?? baseURL
    }

    private static func decode(message: URLSessionWebSocketTask.Message) throws -> JarvisRealtimeServerEvent? {
        let data: Data
        switch message {
        case .data(let payload):
            data = payload
        case .string(let text):
            guard let payload = text.data(using: .utf8) else { return nil }
            data = payload
        @unknown default:
            return nil
        }

        let object = try JSONSerialization.jsonObject(with: data)
        guard let values = object as? [String: Any], let type = values["type"] as? String else {
            return nil
        }
        return JarvisRealtimeServerEvent(type: type, values: values)
    }
}

private actor JarvisVisualWebSocketSendQueue {
    func send(_ message: URLSessionWebSocketTask.Message, task: URLSessionWebSocketTask) async throws {
        try await task.send(message)
    }
}
