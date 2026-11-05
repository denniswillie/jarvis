#if DEBUG
import CryptoKit
import Foundation

/// Local-only manifest. Unknown speech fields (including script text) are not
/// decoded. Only verified image bytes and PCM enter the normal input paths.
struct JarvisReplayLaunchPlan: Decodable {
    struct Frame: Decodable { let at_seconds: Double; let relative_path: String; let sha256: String }
    struct Trigger: Decodable {
        let kind: String
        let at_seconds: Double?
        let after_cursor: Int?
        let delay_seconds: Double?
        let timeout_seconds: Double?
        let after_input_ordinal: Int?
        let max_pcm_age_seconds: Double?
        let response_id: String?
        let connection_id: String?
        let require_task_claim: Bool?
        let canonical_requirement: String?
        let playback_requirement: String?
        private struct WireKey: CodingKey {
            let stringValue: String
            let intValue: Int? = nil
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: WireKey.self)
            func key(_ name: String) -> WireKey { WireKey(stringValue: name)! }
            kind = try values.decode(String.self, forKey: key("kind"))
            at_seconds = try values.decodeIfPresent(Double.self, forKey: key("at_seconds"))
            after_cursor = try values.decodeIfPresent(Int.self, forKey: key("after_cursor"))
            delay_seconds = try values.decodeIfPresent(Double.self, forKey: key("delay_seconds"))
            timeout_seconds = try values.decodeIfPresent(Double.self, forKey: key("timeout_seconds"))
            after_input_ordinal = try values.decodeIfPresent(Int.self, forKey: key("after_input_ordinal"))
            max_pcm_age_seconds = try values.decodeIfPresent(Double.self, forKey: key("max_pcm_age_seconds"))
            response_id = try values.decodeIfPresent(String.self, forKey: key("response_id"))
            connection_id = try values.decodeIfPresent(String.self, forKey: key("connection_id"))
            require_task_claim = try values.decodeIfPresent(Bool.self, forKey: key("require_task_claim"))
            canonical_requirement = try values.decodeIfPresent(String.self, forKey: key("canonical_requirement"))
            playback_requirement = try values.decodeIfPresent(String.self, forKey: key("playback_requirement"))
            if kind == "after_provider_input_finalized",
               Set(values.allKeys.map(\.stringValue)) != Set(["kind", "after_input_ordinal", "timeout_seconds"]) {
                throw JarvisReplayError.invalidPlan
            }
        }
        func value() throws -> JarvisReplayPlan.Trigger {
            guard kind == "response_output_start" || require_task_claim == nil else { throw JarvisReplayError.invalidPlan }
            switch kind {
            case "at_time":
                guard let at_seconds else { throw JarvisReplayError.invalidPlan }
                return .atTime(at_seconds)
            case "output_start":
                guard let after_cursor, let delay_seconds, let timeout_seconds else { throw JarvisReplayError.invalidPlan }
                return .outputStart(afterCursor: after_cursor, delay: delay_seconds, timeout: timeout_seconds)
            case "response_output_start":
                guard let after_input_ordinal, let delay_seconds, let timeout_seconds,
                      let max_pcm_age_seconds else { throw JarvisReplayError.invalidPlan }
                let connection = connection_id.flatMap(UUID.init(uuidString:))
                guard connection_id == nil || connection != nil else { throw JarvisReplayError.invalidPlan }
                return .responseOutputStart(.init(afterInputOrdinal: after_input_ordinal,
                    delay: delay_seconds, timeout: timeout_seconds, maxPCMAge: max_pcm_age_seconds,
                    responseID: response_id, connectionID: connection, requireTaskClaim: require_task_claim ?? false))
            case "after_manual_response":
                guard let after_input_ordinal, let at_seconds, let timeout_seconds,
                      let canonical_requirement, let playback_requirement,
                      after_cursor == nil, delay_seconds == nil, max_pcm_age_seconds == nil,
                      response_id == nil, connection_id == nil else { throw JarvisReplayError.invalidPlan }
                return .afterManualResponse(.init(afterInputOrdinal: after_input_ordinal, at: at_seconds,
                    timeout: timeout_seconds, canonical: canonical_requirement, playback: playback_requirement))
            case "after_task_claim":
                guard let after_input_ordinal, let timeout_seconds else { throw JarvisReplayError.invalidPlan }
                return .afterTaskClaim(afterInputOrdinal: after_input_ordinal, timeout: timeout_seconds)
            case "after_input_complete":
                guard let after_input_ordinal, let delay_seconds, let timeout_seconds else { throw JarvisReplayError.invalidPlan }
                return .afterInputComplete(afterInputOrdinal: after_input_ordinal, delay: delay_seconds, timeout: timeout_seconds)
            case "after_provider_input_finalized":
                guard let after_input_ordinal, let timeout_seconds else { throw JarvisReplayError.invalidPlan }
                return .afterProviderInputFinalized(afterInputOrdinal: after_input_ordinal, timeout: timeout_seconds)
            case "playback_drained": throw JarvisReplayError.unsupportedPlaybackDrain
            default: throw JarvisReplayError.invalidPlan
            }
        }
    }
    struct Utterance: Decodable { let clip_id: String; let trigger: Trigger }
    struct Speech: Decodable {
        struct Clip: Decodable {
            let id: String; let relative_path: String; let sha256: String
            let sample_rate_hz: Int; let channels: Int; let sample_width_bytes: Int; let sample_count: Int
            let first_active_sample: Int?; let last_active_sample_exclusive: Int?
            let energy_threshold_abs_pcm16: Int
        }
        let format: String
        let clips: [Clip]
    }
    struct MemoryReadiness: Decodable {
        let version: Int
        let scope_file: String
        // Absent in prepared packs; mandatory in owned runtime staging.
        let manifest_sha256: String?
    }
    struct TaskBarrier: Decodable {
        let version: Int
        let scope_file: String
        let request_input_ordinal: Int
    }
    struct SourceClock: Decodable {
        let version: Int
        let phase_start_seconds: Double
        let anchor_file: String?
        let anchor_sha256: String?
        func load(under root: URL) throws -> JarvisReplaySourceClock {
            guard version == 1, phase_start_seconds.isFinite, phase_start_seconds >= 0,
                  phase_start_seconds <= 180 else { throw JarvisReplayError.invalidPlan }
            guard let file = anchor_file else {
                guard anchor_sha256 == nil, phase_start_seconds == 0 else { throw JarvisReplayError.invalidPlan }
                return .init(phaseStart: 0, anchor: nil, anchorData: nil)
            }
            guard file == "source-clock.json", let hash = anchor_sha256, hash.count == 64,
                  phase_start_seconds > 0 else { throw JarvisReplayError.invalidPlan }
            let data = try JarvisReplayLaunchPlan.read(JarvisReplayLaunchPlan.localFile(file, under: root), limit: 8192)
            guard JarvisReplayFrameLineage.hash(data) == hash else { throw JarvisReplayError.sourceClockMismatch }
            let object = try JarvisVoiceWire.object(JSONSerialization.jsonObject(with: data))
            try JarvisVoiceWire.keys(object, required: ["format", "run_id", "origin_invocation_id",
                "monotonic_before_seconds", "monotonic_after_seconds", "utc_unix_seconds", "boot_session_id"])
            return .init(phaseStart: phase_start_seconds,
                anchor: try JSONDecoder().decode(JarvisReplaySourceAnchor.self, from: data), anchorData: data)
        }
    }
    struct Loaded {
        let plan: JarvisReplayPlan
        let manifestSHA256: String
        let route: JarvisVoiceRouteChoice
        let configuration: JarvisConfiguration
        let scope: JarvisReplayScopeExpectation
        let taskBarrier: JarvisReplayTaskConfiguration?
        let manualInput: Bool
        let memoryReadiness: JarvisReplayMemoryConfiguration?
    }
    let version: Int
    let run_id: String
    let backend_base_url: URL
    let bearer_token_file: String
    let owner_id: String
    let area_id: String
    let memory_bank_id: String
    let voice_route: String
    let duration_seconds: Double
    let speech_manifest: String
    let frames: [Frame]
    let utterances: [Utterance]
    let task_barrier: TaskBarrier?
    let replay_invocation_id: String?
    let replay_source_clock: SourceClock?
    let post_media_capture_seconds: Double?
    let capture_mode: String?
    let memory_readiness: MemoryReadiness?

    static func localFile(_ path: String, under root: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains(":") else { throw JarvisReplayError.invalidPlan }
        let base = root.resolvingSymlinksInPath().standardizedFileURL
        let url = base.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
        guard url.path.hasPrefix(base.path + "/") else { throw JarvisReplayError.invalidPlan }
        return url
    }
    static func read(_ url: URL, limit: Int) throws -> Data {
        guard url.isFileURL,
              (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= limit else {
            throw JarvisReplayError.invalidPlan
        }
        let data = try Data(contentsOf: url)
        guard data.count <= limit else { throw JarvisReplayError.invalidPlan }
        return data
    }
    static func load(_ manifestURL: URL) throws -> Loaded {
        let decoder = JSONDecoder()
        let manifestData = try read(manifestURL, limit: 512 * 1024)
        let manifest = try decoder.decode(Self.self, from: manifestData)
        let runID = try JarvisReplayScope.parseRunID(manifest.run_id)
        guard manifest.version == 1, manifest.frames.count <= 240, manifest.utterances.count <= 20 else {
            throw JarvisReplayError.invalidPlan
        }
        let route: JarvisVoiceRouteChoice
        switch manifest.voice_route {
        case "realtime": route = .realtime
        case "live": route = .live
        default: throw JarvisReplayError.invalidPlan
        }
        let base = manifestURL.deletingLastPathComponent()
        let invocation = manifest.replay_invocation_id.flatMap(UUID.init(uuidString:))
        guard manifest.replay_invocation_id == nil || invocation?.uuidString.lowercased() == manifest.replay_invocation_id else {
            throw JarvisReplayError.invalidPlan
        }
        let sourceClock = try manifest.replay_source_clock?.load(under: base)
        if sourceClock != nil {
            guard route == .realtime, invocation != nil, manifest.task_barrier == nil else { throw JarvisReplayError.invalidPlan }
        }
        let tokenData = try read(localFile(manifest.bearer_token_file, under: base), limit: 4096)
        guard let token = String(data: tokenData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !token.isEmpty else { throw JarvisReplayError.isolationUnavailable }
        let speechURL = try localFile(manifest.speech_manifest, under: base)
        let speech = try decoder.decode(Speech.self, from: read(speechURL, limit: 512 * 1024))
        guard speech.format == "jarvis.speech_assets.v1", speech.clips.count <= 100,
              Set(speech.clips.map(\.id)).count == speech.clips.count else { throw JarvisReplayError.invalidPlan }
        var inputs: [JarvisReplayPlan.Utterance] = []
        var totalPCM = 0
        for utterance in manifest.utterances {
            guard let clip = speech.clips.first(where: { $0.id == utterance.clip_id }),
                  clip.sample_rate_hz == 48_000, clip.channels == 1, clip.sample_width_bytes == 2,
                  (1...32767).contains(clip.energy_threshold_abs_pcm16) else { throw JarvisReplayError.invalidPlan }
            let url = try localFile(clip.relative_path, under: speechURL.deletingLastPathComponent())
            let asset = JarvisReplayAudioAsset(fileURL: url, format: .wavPCM16Mono48000,
                sha256: clip.sha256, sampleCount: clip.sample_count,
                thresholdPCM16: clip.energy_threshold_abs_pcm16 - 1,
                lastAboveThresholdSampleIndex: clip.last_active_sample_exclusive.map { $0 - 1 })
            let pcm = try asset.loadPCM16()
            let first = (0..<clip.sample_count).first { index in
                let bits = UInt16(pcm[index * 2]) | UInt16(pcm[index * 2 + 1]) << 8
                return abs(Int(Int16(bitPattern: bits))) >= clip.energy_threshold_abs_pcm16
            }
            guard first == clip.first_active_sample else { throw JarvisReplayError.invalidPlan }
            totalPCM += pcm.count
            guard totalPCM <= 60 * 96_000 else { throw JarvisReplayError.invalidPlan }
            inputs.append(.init(trigger: try utterance.trigger.value(), pcm16: pcm, clipID: clip.id,
                assetActivityThreshold: clip.energy_threshold_abs_pcm16,
                assetFirstActiveSample: clip.first_active_sample,
                assetLastActiveSampleExclusive: clip.last_active_sample_exclusive))
        }
        try validateTriggerRoute(route, utterances: inputs)
        try validateTaskMode(manifest.task_barrier, route: route, utterances: inputs)
        let manualInput = try validateManualInput(manifest.capture_mode, route: route,
            invocationID: invocation, taskMode: manifest.task_barrier != nil, utterances: inputs,
            memoryReadiness: manifest.memory_readiness != nil)
        var memoryConfiguration: JarvisReplayMemoryConfiguration?
        if let memory = manifest.memory_readiness {
            let object = try JarvisReplayTaskJSON.object(manifestData, limit: 512 * 1024)
            guard let declaration = object["memory_readiness"] as? [String: Any],
                  let authorizedSHA = memory.manifest_sha256 else { throw JarvisReplayError.invalidPlan }
            try JarvisReplayTaskJSON.keys(declaration, ["version", "scope_file", "manifest_sha256"])
            guard memory.version == 1, memory.scope_file == "memory-readiness-scope.json",
                  let declarations = object["utterances"] as? [[String: Any]] else { throw JarvisReplayError.invalidPlan }
            for entry in declarations {
                guard let trigger = entry["trigger"] as? [String: Any] else { throw JarvisReplayError.invalidPlan }
                if trigger["kind"] as? String == "after_manual_response" {
                    try JarvisReplayTaskJSON.keys(trigger, ["kind", "after_input_ordinal", "at_seconds",
                        "timeout_seconds", "canonical_requirement", "playback_requirement"])
                }
            }
            memoryConfiguration = try JarvisReplayMemoryConfiguration.load(
                scopeData: JarvisReplayMemoryConfiguration.readScope(base.appendingPathComponent(memory.scope_file)),
                manifestSHA256: authorizedSHA, inputsRoot: base, runID: runID,
                ownerID: manifest.owner_id, areaID: manifest.area_id, bankID: manifest.memory_bank_id)
        }
        var taskConfiguration: JarvisReplayTaskConfiguration?
        if let task = manifest.task_barrier {
            taskConfiguration = try JarvisReplayTaskConfiguration.load(
                scopeData: read(localFile(task.scope_file, under: base), limit: 8 * 1024),
                manifestData: manifestData, inputsRoot: base, runID: runID, ownerID: manifest.owner_id)
        }
        var images: [JarvisReplayPlan.Frame] = []
        var imageBytes = 0
        for frame in manifest.frames {
            let data = try read(localFile(frame.relative_path, under: base), limit: 4 * 1024 * 1024)
            guard SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == frame.sha256.lowercased() else {
                throw JarvisReplayError.invalidPlan
            }
            imageBytes += data.count
            guard imageBytes <= 64 * 1024 * 1024 else { throw JarvisReplayError.invalidPlan }
            images.append(.init(at: frame.at_seconds, image: data))
        }
        return Loaded(plan: try JarvisReplayPlan(runID: runID, duration: manifest.duration_seconds,
                frames: images, utterances: inputs, postMediaCaptureSeconds: manifest.post_media_capture_seconds,
                invocationID: invocation, sourceClock: sourceClock, memoryReadinessDeclared: memoryConfiguration != nil),
            manifestSHA256: JarvisReplayFrameLineage.hash(manifestData), route: route,
            configuration: JarvisConfiguration(backendBaseURL: manifest.backend_base_url, staticBearerToken: token),
            scope: JarvisReplayScopeExpectation(runID: runID, ownerID: manifest.owner_id,
                areaID: manifest.area_id, memoryBankID: manifest.memory_bank_id),
            taskBarrier: taskConfiguration, manualInput: manualInput, memoryReadiness: memoryConfiguration)
    }

    static func validateManualInput(_ mode: String?, route: JarvisVoiceRouteChoice, invocationID: UUID?,
                                    taskMode: Bool, utterances: [JarvisReplayPlan.Utterance],
                                    memoryReadiness: Bool = false) throws -> Bool {
        switch mode {
        case nil, "auto":
            guard !memoryReadiness, !utterances.contains(where: {
                if case .afterManualResponse = $0.trigger { return true }; return false
            }) else { throw JarvisReplayError.invalidPlan }
            return false
        case "manual":
            guard route == .realtime, invocationID != nil, !taskMode, !utterances.isEmpty,
                  utterances.allSatisfy({
                      if case .atTime = $0.trigger { return true }
                      if case .afterManualResponse = $0.trigger { return memoryReadiness }
                      return false
                  }) else {
                throw JarvisReplayError.invalidPlan
            }
            return true
        default: throw JarvisReplayError.invalidPlan
        }
    }

    static func validateTaskMode(_ task: TaskBarrier?, route: JarvisVoiceRouteChoice,
                                 utterances: [JarvisReplayPlan.Utterance]) throws {
        guard let task else {
            guard !utterances.contains(where: { $0.trigger.isTaskTrigger }) else { throw JarvisReplayError.invalidPlan }
            return
        }
        guard route == .realtime, task.version == 1, task.scope_file == "task-scope.json",
              task.request_input_ordinal == 0, [4, 6, 8].contains(utterances.count),
              case .atTime = utterances[0].trigger,
              case .afterTaskClaim(let request, _) = utterances[1].trigger, request == 0 else {
            throw JarvisReplayError.invalidPlan
        }
        for index in 2..<utterances.count {
            if index.isMultiple(of: 2) {
                guard case .responseOutputStart(let trigger) = utterances[index].trigger,
                      trigger.requireTaskClaim, trigger.afterInputOrdinal == index - 1 else { throw JarvisReplayError.invalidPlan }
            } else {
                switch utterances[index].trigger {
                case .afterInputComplete(let previous, _, _), .afterProviderInputFinalized(let previous, _):
                    guard previous == index - 1 else { throw JarvisReplayError.invalidPlan }
                default: throw JarvisReplayError.invalidPlan
                }
            }
        }
    }

    static func validateTriggerRoute(_ route: JarvisVoiceRouteChoice,
                                     utterances: [JarvisReplayPlan.Utterance]) throws {
        if case .live = route, utterances.contains(where: {
            if case .responseOutputStart = $0.trigger { return true }
            if case .afterProviderInputFinalized = $0.trigger { return true }; return false
        }) { throw JarvisReplayError.unsupportedResponseIdentity }
    }
}
#endif
