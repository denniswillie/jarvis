import Foundation

/// Host-only regression entry point. No WebRTC, simulator, app or provider.
@main
enum PlaybackGateDiagnosticsRegression {
    static func main() throws {
        var checks = 0
        func check(_ condition: Bool, _ name: String) {
            precondition(condition, name)
            checks += 1
        }
        func snapshot(remote: Bool = true, permit: Bool = true, playable: Bool = true,
                      match: Bool = true, external: Bool = false, held: Bool = false,
                      receiver: Bool = true, replayID: Bool = true,
                      origin: JarvisReplayPlaybackGateDiagnostics.Origin = .speakerPredicate,
                      item: JarvisReplayPlaybackItemDiagnostics? = nil) -> JarvisReplayPlaybackGateDiagnostics {
            .init(origin: origin, remoteDescriptionPresent: remote, permittedIdentityPresent: permit,
                  playableIdentityPresent: playable, playableIdentityMatchesPermit: match,
                  externalPlaybackInactive: !external, routeOutputHeld: held, receiverPresent: receiver,
                  replayItemIdentityPresent: replayID, itemMatchesPermittedResponse: permit && item != nil,
                  item: item)
        }
        let secret = "DO_NOT_RETAIN_RAW_ITEM_TRANSCRIPT_AUDIO_OR_SECRET"
        let categories: [([String: Any], String, String)] = [
            ([:], "missing", "missing"),
            (["type": NSNull(), "phase": NSNull()], "null", "null"),
            (["type": "message", "phase": "final_answer"], "message", "final_answer"),
            (["type": "message", "phase": "commentary"], "message", "commentary"),
            (["type": "message", "phase": "analysis"], "message", "analysis"),
            (["type": "function_call", "phase": "final_answer"], "function_call", "final_answer"),
            (["type": secret, "phase": secret], "other_string", "other_string"),
            (["type": 1, "phase": true], "invalid_type", "invalid_type"),
            (["type": [secret], "phase": ["text": secret]], "invalid_type", "invalid_type")
        ]
        var examples: [[String: Any]] = []
        for (raw, type, phase) in categories {
            var input = raw
            input["text"] = secret; input["audio"] = secret; input["api_key"] = secret
            let item = JarvisReplayPlaybackItemDiagnostics(item: input, policyAllowsPlayback: false)
            check(item.typeCategory == type, "safe item category")
            check(item.phaseCategory == phase, "safe phase category")
            let diagnostic = snapshot(item: item).json
            let encoded = try JSONSerialization.data(withJSONObject: diagnostic, options: [.sortedKeys])
            check(!String(decoding: encoded, as: UTF8.self).contains(secret), "no raw item retained")
            examples.append(diagnostic)
        }
        for mask in 0..<64 {
            let remote = mask & 1 != 0, permit = mask & 2 != 0, playable = mask & 4 != 0
            let external = mask & 8 != 0, held = mask & 16 != 0, receiver = mask & 32 != 0
            let value = snapshot(remote: remote, permit: permit, playable: playable,
                                 match: permit && playable, external: external, held: held, receiver: receiver)
            check(value.predicateAllowsPlayback == (remote && permit && playable && !external && !held),
                  "existing predicate truth table")
            check(value.json["receiver_present"] as? Bool == receiver, "receiver is observational")
        }
        check(!snapshot(match: false).predicateAllowsPlayback, "mismatched identities deny")
        check(snapshot(receiver: false).predicateAllowsPlayback, "receiver does not change replay predicate")
        check(snapshot().json["item_type_category"] as? String == "not_observed", "missing item is explicit")

        let connection = UUID(), run = UUID()
        var policyChecks = 0
        func policyCheck(_ condition: Bool, _ name: String) {
            check(condition, name)
            policyChecks += 1
        }
        let phases: [(Any?, Bool)] = [
            (nil, true), (NSNull(), true), ("commentary", true), ("final_answer", true),
            ("analysis", false), ("final", false), ("unknown", false), ("", false),
            (1, false), (true, false), (["commentary"], false),
        ]
        for (phase, allowed) in phases {
            var item: [String: Any] = ["type": "message"]
            if let phase { item["phase"] = phase }
            policyCheck(JarvisDirectOutputPolicy.allowsPlayback(item: item) == allowed, "outward phase allowlist")
        }
        for type in ["function_call", "function_call_output", "unknown", ""] {
            policyCheck(!JarvisDirectOutputPolicy.allowsPlayback(item: ["type": type, "phase": "commentary"]),
                        "non-message cannot grant playback")
        }
        let commentary: [String: Any] = ["type": "message", "phase": "commentary", "id": "message-1"]
        let tool: [String: Any] = ["type": "function_call", "id": "tool-1", "arguments": secret]
        let analysis: [String: Any] = ["type": "message", "phase": "analysis", "id": "analysis-1"]
        for item in [commentary, tool, analysis, ["type": "unknown"]] {
            policyCheck(JarvisDirectOutputPolicy.bindingAction(item: item, responseID: "old",
                permittedResponseID: "current", isRetired: false) == .ignore, "stale item ignored")
            policyCheck(JarvisDirectOutputPolicy.bindingAction(item: item, responseID: "current",
                permittedResponseID: nil, isRetired: false) == .ignore, "unpermitted item ignored")
            policyCheck(JarvisDirectOutputPolicy.bindingAction(item: item, responseID: "current",
                permittedResponseID: "current", isRetired: true) == .ignore, "retired item ignored")
        }
        policyCheck(JarvisDirectOutputPolicy.bindingAction(item: commentary, responseID: "current",
            permittedResponseID: "current", isRetired: false, eventItemID: "conflict") == .playMessage(itemID: nil),
            "conflicting IDs cannot fabricate correlation")

        // Exercise production item decisions against the real replay PCM state.
        // The host-only driver cannot establish native transport or hearing.
        var commentaryState = JarvisReplayPCMState(maxDuration: 10)
        var permit: String? = "current"
        var playable: String?
        var messageID: String?
        var held = false
        var external = false
        var time = 10.0
        func receive(_ item: [String: Any], response: String = "current", retired: Bool = false) {
            let action = JarvisDirectOutputPolicy.bindingAction(item: item, responseID: response,
                permittedResponseID: permit, isRetired: retired)
            switch action {
            case .ignore: return
            case .retainMessage: break
            case .playMessage(let id): playable = response; messageID = id
            case .clear: playable = nil; messageID = nil
            }
            time += 0.1
            let diagnostic = snapshot(permit: permit != nil, playable: playable != nil,
                match: permit != nil && playable == permit, external: external, held: held,
                replayID: messageID != nil,
                item: .init(item: item, policyAllowsPlayback: JarvisDirectOutputPolicy.allowsPlayback(item: item)))
            commentaryState.gateOutput(diagnostic.predicateAllowsPlayback, at: time,
                responseID: permit, itemID: messageID, connectionID: connection, diagnostics: diagnostic)
        }
        receive(tool)
        policyCheck(!commentaryState.outputEnabled && messageID == nil, "tool first grants nothing")
        receive(commentary)
        policyCheck(commentaryState.outputEnabled && messageID == "message-1", "commentary admitted")
        policyCheck(commentaryState.currentResponseIdentity != nil, "actual message identity bound")
        commentaryState.observeDecoded([32, -32], at: time + 0.01, callbackEnd: time + 0.02,
                                       responseIdentity: commentaryState.currentResponseIdentity)
        for _ in 0..<4 { receive(commentary); receive(tool) }
        policyCheck(commentaryState.outputEnabled && messageID == "message-1", "tools preserve message not tool ID")
        policyCheck(commentaryState.currentResponseIdentity != nil, "duplicates preserve original correlation")
        policyCheck(commentaryState.evidence.outputGates.last?.playbackDiagnostics?.item?.policyAllowsPlayback == false,
                    "tool diagnostics never borrow retained message permission")
        let beforeStale = commentaryState.evidence.outputGates.count
        receive(analysis, response: "old")
        receive(commentary, retired: true)
        policyCheck(commentaryState.evidence.outputGates.count == beforeStale && messageID == "message-1",
                    "stale events do not mutate active binding or gate")
        commentaryState.recordLocalInterruption(.clientInterrupt, at: time + 0.03)
        policyCheck(commentaryState.evidence.localInterruptions.count == 1, "one actual binding supports one cut")
        receive(["type": "message", "phase": "final_answer", "id": "message-2"])
        policyCheck(messageID == "message-2", "later message never inherits earlier item ID")
        policyCheck(commentaryState.currentResponseIdentity == nil,
                    "second item in same response cannot fabricate another cut identity")
        receive(tool)
        policyCheck(messageID == "message-2" && commentaryState.currentResponseIdentity == nil,
                    "tool cannot resurrect consumed message identity")
        commentaryState.recordLocalInterruption(.clientInterrupt, at: time + 0.03)
        policyCheck(commentaryState.evidence.localInterruptions.count == 1, "no extra cut from duplicate response")
        let commentaryReport = JarvisReplayEvidenceReport.make(evidence: commentaryState.evidence,
            runID: run, finalUIStatus: "offline_policy_fixture")
        let commentaryGates = commentaryReport["output_gates"] as! [[String: Any]]
        policyCheck(commentaryGates.last?["local_output_epoch"] as? Int == 1,
                    "messages and tools do not create extra output epochs")
        policyCheck(commentaryState.evidence.decodedPCM16.count == 4, "admitted fixture PCM retained")
        receive(analysis)
        policyCheck(!commentaryState.outputEnabled && playable == nil && messageID == nil, "analysis clears binding")
        receive(tool)
        policyCheck(!commentaryState.outputEnabled && messageID == nil, "tool cannot reopen cleared binding")
        receive(["type": "unknown"])
        policyCheck(!commentaryState.outputEnabled, "unknown item fails closed")
        permit = nil; held = true
        receive(commentary)
        policyCheck(!commentaryState.outputEnabled && messageID == nil, "interrupt without permit stays closed")
        permit = "next"
        receive(commentary, response: "next")
        policyCheck(!commentaryState.outputEnabled, "route hold survives a playable item")
        held = false; external = true
        receive(tool, response: "next")
        policyCheck(!commentaryState.outputEnabled, "external playback fence survives tool item")
        external = false
        receive(["type": "message", "phase": "final_answer", "id": "next-message"], response: "next")
        policyCheck(commentaryState.outputEnabled && messageID == "next-message", "fresh permitted final answer allowed")

        var baseline = JarvisReplayPCMState(maxDuration: 1)
        var instrumented = baseline
        let admitted = snapshot(item: .init(item: ["type": "message", "phase": "final_answer"],
                                            policyAllowsPlayback: true))
        func exercise(_ state: inout JarvisReplayPCMState, diagnostics: Bool) {
            state.gateOutput(false, at: 1, diagnostics: diagnostics ? snapshot(permit: false, match: false) : nil)
            state.observeDecoded([0, 32, -32, 0], at: 1.1, callbackEnd: 1.11)
            state.gateOutput(true, at: 2, responseID: "resp-fixture", itemID: "item-fixture",
                             connectionID: connection, diagnostics: diagnostics ? admitted : nil)
            state.observeDecoded([0, 32, -32, 0], at: 2.01, callbackEnd: 2.02,
                                 responseIdentity: state.currentResponseIdentity)
            state.recordLocalInterruption(.clientInterrupt, at: 2.03)
            state.gateOutput(false, at: 2.04, diagnostics: diagnostics ? snapshot(held: true) : nil)
        }
        exercise(&baseline, diagnostics: false)
        exercise(&instrumented, diagnostics: true)
        let oldReport = JarvisReplayEvidenceReport.make(evidence: baseline.evidence, runID: run, finalUIStatus: "fixture")
        let newReport = JarvisReplayEvidenceReport.make(evidence: instrumented.evidence, runID: run, finalUIStatus: "fixture")
        let newGates = newReport["output_gates"] as! [[String: Any]]
        check(newGates.allSatisfy { $0["playback_diagnostics"] != nil }, "diagnostics reach each observed gate")
        var stripped = newReport
        stripped["output_gates"] = newGates.map { value -> [String: Any] in
            var result = value; result.removeValue(forKey: "playback_diagnostics"); return result
        }
        check(try JSONSerialization.data(withJSONObject: oldReport, options: [.sortedKeys])
                  == JSONSerialization.data(withJSONObject: stripped, options: [.sortedKeys]),
              "diagnostics do not change core evidence or legacy serialization")
        check(instrumented.evidence.gatedDecodedBlocks == 1, "disabled gate still discards")
        check(instrumented.evidence.decodedPCM16.count == 8, "enabled gate still retains")
        check(instrumented.evidence.localInterruptions.count == 1, "interrupt evidence unchanged")
        check(newGates.last?["local_output_epoch"] as? Int == 1, "epoch unchanged")

        var missingIdentity = JarvisReplayPCMState(maxDuration: 1)
        missingIdentity.gateOutput(true, at: 1, responseID: "resp-fixture", connectionID: connection,
                                  diagnostics: snapshot(replayID: false))
        missingIdentity.observeDecoded([32, 0], at: 1.1, callbackEnd: 1.11)
        check(missingIdentity.outputEnabled && missingIdentity.currentResponseIdentity == nil,
              "missing item identity is not a disabled gate")
        check(missingIdentity.evidence.uncorrelatedDecodedBlocks == 1, "missing identity remains uncorrelated")

        var forced = JarvisReplayPCMState(maxDuration: 1)
        forced.gateOutput(false, at: 1, diagnostics: snapshot(origin: .forcedClose))
        check(!forced.outputEnabled && forced.evidence.outputGates[0].playbackDiagnostics!.predicateAllowsPlayback,
              "forced close diagnostics cannot enable output")
        forced.close(at: 2)
        check(forced.evidence.outputGates.last?.playbackDiagnostics == nil,
              "local close does not invent a parent predicate snapshot")

        var bounded = JarvisReplayPCMState(maxDuration: 1)
        for i in 0...JarvisReplayPCMState.maximumOutputGateEntries {
            bounded.gateOutput(false, at: Double(i), diagnostics: admitted)
        }
        check(bounded.evidence.outputGates.count == JarvisReplayPCMState.maximumOutputGateEntries,
              "diagnostic history uses existing gate cap")
        check(bounded.evidence.outputGateHistoryTruncated, "cap remains explicit")
        let output: [String: Any] = ["checks": checks, "diagnostics": examples + [admitted.json],
                                     "gate_records": newGates, "policy_binding_checks": policyChecks,
                                     "commentary_gate_records": commentaryGates]
        print(String(decoding: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]), as: UTF8.self))
    }
}
