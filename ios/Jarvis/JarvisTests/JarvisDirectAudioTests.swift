import Foundation
import XCTest

final class JarvisDirectAudioTests: XCTestCase {
    func testDirectOutputPolicyAllowsOutwardMessagesButNotAnalysisOrToolCalls() {
        XCTAssertTrue(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message"]))
        XCTAssertTrue(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message", "phase": NSNull()]))
        XCTAssertTrue(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message", "phase": "final_answer"]))
        XCTAssertFalse(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message", "phase": "final"]))
        XCTAssertTrue(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message", "phase": "commentary"]))
        XCTAssertFalse(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message", "phase": "analysis"]))
        XCTAssertFalse(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "function_call", "phase": "final_answer"]))
        XCTAssertFalse(JarvisDirectOutputPolicy.allowsPlayback(item: ["type": "message", "phase": 1]))
    }

    func testFunctionCallRetainsOnlyAnExistingMessageBinding() {
        XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
            item: ["type": "message", "phase": "commentary", "id": "message-1"],
            responseID: "response-1", permittedResponseID: "response-1", isRetired: false),
            .playMessage(itemID: "message-1"))
        XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
            item: ["type": "function_call", "id": "tool-1"], responseID: "response-1",
            permittedResponseID: "response-1", isRetired: false), .retainMessage)
        for item: [String: Any] in [
            ["type": "message", "phase": "analysis"],
            ["type": "message", "phase": "unknown"],
            ["type": "message", "phase": true],
            ["type": "unknown"],
        ] {
            XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
                item: item, responseID: "response-1", permittedResponseID: "response-1",
                isRetired: false), .clear)
        }
    }

    func testStaleRetiredAndUnpermittedItemsCannotMutatePlayback() {
        for item: [String: Any] in [
            ["type": "message", "phase": "commentary"],
            ["type": "message", "phase": "analysis"],
            ["type": "function_call"],
        ] {
            XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
                item: item, responseID: "old", permittedResponseID: "current", isRetired: false), .ignore)
            XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
                item: item, responseID: "current", permittedResponseID: nil, isRetired: false), .ignore)
            XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
                item: item, responseID: "current", permittedResponseID: "current", isRetired: true), .ignore)
        }
    }

    func testOnlyActualMessageIdentifiersBecomeReplayBindings() {
        let item: [String: Any] = ["type": "message", "phase": "commentary", "id": "message-1"]
        XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
            item: item, responseID: "response-1", permittedResponseID: "response-1",
            isRetired: false, eventItemID: "message-1"), .playMessage(itemID: "message-1"))
        XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
            item: item, responseID: "response-1", permittedResponseID: "response-1",
            isRetired: false, eventItemID: "different"), .playMessage(itemID: nil))
        XCTAssertEqual(JarvisDirectOutputPolicy.bindingAction(
            item: ["type": "message"], responseID: "response-1", permittedResponseID: "response-1",
            isRetired: false, eventItemID: "message-2"), .playMessage(itemID: "message-2"))
    }

    func testFrameFlowKeepsOnlyNewestPendingImageUntilAcknowledgement() {
        var flow = JarvisVisualFrameFlow<Int64>()
        for frame in Int64(1)...10_000 { flow.enqueue(frame) }
        XCTAssertEqual(flow.takeNext(sequence: { $0 }), 10_000)
        for frame in Int64(10_001)...20_000 { flow.enqueue(frame) }
        XCTAssertNil(flow.takeNext(sequence: { $0 }))
        XCTAssertFalse(flow.acknowledge(sequence: 9_999))
        XCTAssertNil(flow.takeNext(sequence: { $0 }))
        XCTAssertTrue(flow.acknowledge(sequence: 10_000))
        XCTAssertEqual(flow.takeNext(sequence: { $0 }), 20_000)
    }

    func testFrameFlowResetDiscardsPendingAndInFlightImages() {
        var flow = JarvisVisualFrameFlow<Int64>()
        flow.enqueue(1)
        XCTAssertEqual(flow.takeNext(sequence: { $0 }), 1)
        flow.enqueue(2)
        flow.reset()
        XCTAssertFalse(flow.acknowledge(sequence: 1))
        XCTAssertNil(flow.takeNext(sequence: { $0 }))
        flow.enqueue(3)
        XCTAssertEqual(flow.takeNext(sequence: { $0 }), 3)
    }

    func testFrameFlowPreservesLarge64BitSequenceIdentifiers() {
        var flow = JarvisVisualFrameFlow<Int64>()
        let sequence = Int64.max - 1
        flow.enqueue(sequence)
        XCTAssertEqual(flow.takeNext(sequence: { $0 }), sequence)
        XCTAssertFalse(flow.acknowledge(sequence: sequence - 1))
        XCTAssertTrue(flow.acknowledge(sequence: sequence))
    }

    func testDirectURLReplacesProviderAndTransportWithoutLosingSession() throws {
        let url = try JarvisAudioProvider.openai.audioWebSocketURL(
            from: "wss://example.com/audio?session_id=fixture&audio_provider=gemini&audio_transport=websocket&audio_transport=bad",
            directAudio: true
        )
        let items = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems)
        XCTAssertEqual(items.filter { $0.name == "audio_provider" }.map(\.value), ["openai"])
        XCTAssertEqual(items.filter { $0.name == "audio_transport" }.map(\.value), ["webrtc"])
        XCTAssertEqual(items.first { $0.name == "session_id" }?.value, "fixture")
    }

    func testExplicitRelayDoesNotInheritDirectTransport() throws {
        let url = try JarvisAudioProvider.openai.audioWebSocketURL(
            from: "wss://example.com/audio?audio_transport=webrtc"
        )
        XCTAssertNil(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first {
            $0.name == "audio_transport"
        })
    }

    func testDirectTransportRejectsGeminiAndNonWebsocketURLs() {
        XCTAssertThrowsError(try JarvisAudioProvider.gemini.audioWebSocketURL(
            from: "wss://example.com/audio", directAudio: true
        ))
        XCTAssertThrowsError(try JarvisAudioProvider.openai.audioWebSocketURL(
            from: "https://example.com/audio", directAudio: true
        ))
    }
}
