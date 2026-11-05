import Foundation
import XCTest

// This unhosted target compiles the same production model/store source files.
// It never launches Jarvis, reads its real credentials, or opens a microphone.
final class JarvisAudioProviderTests: XCTestCase {
    private func acknowledgement(
        provider: String = "gemini", audioID: String = "fixture-audio", model: String = "fixture-model",
        version: Any = 1
    ) -> JarvisRealtimeServerEvent {
        JarvisRealtimeServerEvent(type: "audio.provider.selected", values: [
            "provider": provider, "audio_session_id": audioID, "model": model, "protocol_version": version,
        ])
    }

    private func readyRun(_ provider: JarvisAudioProvider) throws -> JarvisAudioProviderRun {
        var run = JarvisAudioProviderRun(requested: provider)
        try run.acknowledge(acknowledgement(provider: provider.rawValue))
        try run.markReady(audioSessionID: "fixture-audio")
        return run
    }

    func testPreferencePersistsAndResetDoesNotTouchRealCredentials() {
        let name = "jarvis.provider-tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let keychain = JarvisKeychain(service: name)
        let store = JarvisSessionStore(defaults: defaults, keychain: keychain)
        XCTAssertEqual(store.audioProvider, .openai)
        store.audioProvider = .gemini
        XCTAssertEqual(JarvisSessionStore(defaults: defaults, keychain: keychain).audioProvider, .openai)
        XCTAssertEqual(defaults.string(forKey: "jarvis.audioProvider"), "openai")
        store.clearConfiguration()
        XCTAssertEqual(store.audioProvider, .openai)
        XCTAssertNil(defaults.object(forKey: "jarvis.audioProvider"))
    }

    func testPreviouslySavedGeminiPreferenceMigratesToOpenAI() {
        let name = "jarvis.provider-tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("gemini", forKey: "jarvis.audioProvider")
        let store = JarvisSessionStore(defaults: defaults, keychain: JarvisKeychain(service: name))
        XCTAssertEqual(store.audioProvider, .openai)
        XCTAssertEqual(defaults.string(forKey: "jarvis.audioProvider"), "openai")
        XCTAssertEqual(JarvisAudioProvider.gemini.supportedSelection, .openai)
    }

    func testUnknownSavedPreferenceFallsBackOnlyLocally() {
        let name = "jarvis.provider-tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("future-provider", forKey: "jarvis.audioProvider")
        XCTAssertEqual(JarvisSessionStore(defaults: defaults, keychain: JarvisKeychain(service: name)).audioProvider, .openai)
        var run = JarvisAudioProviderRun(requested: .openai)
        XCTAssertThrowsError(try run.acknowledge(acknowledgement(provider: "future-provider")))
        XCTAssertNil(run.activeProvider)
    }

    func testURLPreservesSessionAndUnrelatedQueryItemsAndReplacesAllProviderValues() throws {
        let url = try JarvisAudioProvider.gemini.audioWebSocketURL(
            from: "wss://fixture.invalid/audio/path?session_id=session-123&other=a%2Bb&other=c&audio_provider=openai&audio_provider=bad"
        )
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        XCTAssertEqual(components.path, "/audio/path")
        let items = components.queryItems!
        XCTAssertEqual(items.filter { $0.name == "session_id" }.map(\.value), ["session-123"])
        XCTAssertEqual(items.filter { $0.name == "other" }.map(\.value), ["a+b", "c"])
        XCTAssertEqual(items.filter { $0.name == "audio_provider" }.map(\.value), ["gemini"])
        let gpt = try JarvisAudioProvider.openai.audioWebSocketURL(from: url.absoluteString)
        XCTAssertEqual(URLComponents(url: gpt, resolvingAgainstBaseURL: false)?.queryItems?.last?.value, "openai")
    }

    func testInvalidAudioURLsFailWithoutFallback() {
        for url in ["/relative", "https://fixture.invalid/audio", "wss:///audio", "wss://fixture.invalid/audio#fragment"] {
            XCTAssertThrowsError(try JarvisAudioProvider.gemini.audioWebSocketURL(from: url), url)
        }
    }

    func testPreferenceChangesDoNotChangeStartSnapshotOrActiveProvider() throws {
        var preference = JarvisAudioProvider.gemini
        var run = JarvisAudioProviderRun(requested: preference)
        preference = .openai
        XCTAssertEqual(run.requested, .gemini)
        XCTAssertNil(run.activeProvider)
        try run.acknowledge(acknowledgement())
        XCTAssertNil(run.activeProvider)
        XCTAssertNil(run.activeModel)
        try run.markReady(audioSessionID: "fixture-audio")
        XCTAssertEqual(run.activeProvider, .gemini)
        XCTAssertEqual(run.activeModel, "fixture-model")
        let next = JarvisAudioProviderRun(requested: preference)
        XCTAssertEqual(next.requested, .openai)
        XCTAssertNotEqual(next.id, run.id)
    }

    func testLegacyOpenAIBackendCanBecomeReadyWithoutAcknowledgement() throws {
        var run = JarvisAudioProviderRun(requested: .openai)
        try run.markReady(audioSessionID: "legacy-audio")
        XCTAssertEqual(run.activeProvider, .openai)
        XCTAssertNil(run.activeModel)
    }

    func testLegacyBackendCannotSilentlySubstituteGPTForGemini() {
        var run = JarvisAudioProviderRun(requested: .gemini)
        XCTAssertThrowsError(try run.markReady(audioSessionID: "legacy-audio"))
        XCTAssertNil(run.activeProvider)
    }

    func testLateAcknowledgementCannotRelabelAnotherLegacyAudioSession() throws {
        var run = JarvisAudioProviderRun(requested: .openai)
        try run.markReady(audioSessionID: "legacy-audio")
        XCTAssertThrowsError(try run.acknowledge(acknowledgement(provider: "openai")))
        XCTAssertNil(run.activeModel)
        try run.acknowledge(acknowledgement(provider: "openai", audioID: "legacy-audio"))
        XCTAssertEqual(run.activeModel, "fixture-model")
    }

    func testProviderMismatchAndWrongAudioSessionFailBeforeCapture() throws {
        var run = JarvisAudioProviderRun(requested: .gemini)
        XCTAssertThrowsError(try run.acknowledge(acknowledgement(provider: "openai")))
        try run.acknowledge(acknowledgement())
        XCTAssertThrowsError(try run.markReady(audioSessionID: "different-audio"))
        var calls = 0
        XCTAssertThrowsError(try run.startMedia(captureMode: .autoDetect, unifiedSession: true,
                                              startMicrophone: { calls += 1 }, startCamera: { calls += 1 }))
        XCTAssertEqual(calls, 0)
    }

    func testDuplicateAcknowledgementIsIdempotentButConflictingModelFails() throws {
        var run = JarvisAudioProviderRun(requested: .gemini)
        try run.acknowledge(acknowledgement())
        try run.acknowledge(acknowledgement())
        XCTAssertThrowsError(try run.acknowledge(acknowledgement(model: "different-model")))
        XCTAssertEqual(run.acknowledgement?.model, "fixture-model")
    }

    func testInvalidProtocolVersionsIncludingJSONBooleanAreRejected() throws {
        for version: Any in [0, 2, true, "1"] {
            var run = JarvisAudioProviderRun(requested: .gemini)
            let data = try JSONSerialization.data(withJSONObject: acknowledgement(version: version).values)
            let values = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            XCTAssertThrowsError(try run.acknowledge(JarvisRealtimeServerEvent(type: "audio.provider.selected", values: values)))
        }
    }

    func testBothProvidersAndCaptureModesKeepMicrophoneBeforeCamera() throws {
        for provider in JarvisAudioProvider.allCases {
            for mode in JarvisAudioCaptureMode.allCases {
                let run = try readyRun(provider)
                var order: [String] = []
                try run.startMedia(captureMode: mode, unifiedSession: true,
                                   startMicrophone: { order.append("microphone") }, startCamera: { order.append("camera") })
                XCTAssertEqual(order, ["microphone", "camera"], "\(provider) \(mode)")
            }
        }
    }

    func testAudioOnlyHoldToTalkDoesNotStartCaptureUntilManualPress() throws {
        let run = try readyRun(.gemini)
        for mode in JarvisAudioCaptureMode.allCases {
            var order: [String] = []
            try run.startMedia(captureMode: mode, unifiedSession: false,
                               startMicrophone: { order.append("microphone") }, startCamera: { order.append("camera") })
            XCTAssertEqual(order, mode == .autoDetect ? ["microphone"] : [])
        }
    }

    func testCaptureFailureNeverStartsCamera() throws {
        let run = try readyRun(.gemini)
        var cameraStarted = false
        XCTAssertThrowsError(try run.startMedia(captureMode: .holdToTalk, unifiedSession: true,
                                              startMicrophone: { throw JarvisAudioProviderError.notReady },
                                              startCamera: { cameraStarted = true }))
        XCTAssertFalse(cameraStarted)
    }

    func testFailedGeminiRunCanBeDiscardedAndFollowedByLegacyGPT() throws {
        var run: JarvisAudioProviderRun? = JarvisAudioProviderRun(requested: .gemini)
        let oldID = run!.id
        XCTAssertThrowsError(try run?.markReady(audioSessionID: "no-ack"))
        run = nil
        run = JarvisAudioProviderRun(requested: .openai)
        XCTAssertNotEqual(run?.id, oldID)
        try run?.markReady(audioSessionID: "legacy-gpt")
        XCTAssertEqual(run?.activeProvider, .openai)
        var order: [String] = []
        try run?.startMedia(captureMode: .autoDetect, unifiedSession: true,
                           startMicrophone: { order.append("microphone") }, startCamera: { order.append("camera") })
        XCTAssertEqual(order, ["microphone", "camera"])
    }

    func testGeminiPlaybackClearCannotStopGPTOrANewerResponse() throws {
        let clear = JarvisRealtimeServerEvent(type: "audio.output.clear", values: [
            "provider": "gemini", "response_id": "old-response", "item_id": "old-item",
        ])
        let old = PlaybackTruncationSnapshot(itemId: "old-item", responseId: "old-response", audioEndMs: 12)
        let newer = PlaybackTruncationSnapshot(itemId: "new-item", responseId: "new-response", audioEndMs: 12)
        XCTAssertTrue(try readyRun(.gemini).shouldClearPlayback(clear, snapshot: old))
        XCTAssertFalse(try readyRun(.gemini).shouldClearPlayback(clear, snapshot: newer))
        XCTAssertFalse(try readyRun(.openai).shouldClearPlayback(clear, snapshot: old))
        XCTAssertFalse(try readyRun(.gemini).shouldClearPlayback(clear, snapshot: nil))
    }

    func testOpenAIPlaybackClearRequiresMatchingProviderResponseAndItem() throws {
        let clear = JarvisRealtimeServerEvent(type: "audio.output.clear", values: [
            "provider": "openai", "response_id": "old-response", "item_id": "old-item",
        ])
        let old = PlaybackTruncationSnapshot(itemId: "old-item", responseId: "old-response", audioEndMs: 12)
        let newer = PlaybackTruncationSnapshot(itemId: "new-item", responseId: "new-response", audioEndMs: 12)
        let otherItem = PlaybackTruncationSnapshot(itemId: "other-item", responseId: "old-response", audioEndMs: 12)
        XCTAssertTrue(try readyRun(.openai).shouldClearPlayback(clear, snapshot: old))
        XCTAssertFalse(try readyRun(.gemini).shouldClearPlayback(clear, snapshot: old))
        XCTAssertFalse(try readyRun(.openai).shouldClearPlayback(clear, snapshot: newer))
        XCTAssertFalse(try readyRun(.openai).shouldClearPlayback(clear, snapshot: otherItem))
        XCTAssertFalse(try readyRun(.openai).shouldClearPlayback(clear, snapshot: nil))
        XCTAssertFalse(JarvisAudioProviderRun(requested: .openai).shouldClearPlayback(clear, snapshot: old))
    }
}
