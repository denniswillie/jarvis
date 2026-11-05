import Foundation

@main
enum PhonePluginCatalogRegression {
    static func main() throws {
        var checks = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); checks += 1 }
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        let original = try Data(contentsOf: url)
        let snapshot = try JSONDecoder().decode(JarvisPluginCatalogSnapshot.self, from: original)
        let now = snapshot.evaluatedAt.addingTimeInterval(5)
        let base = try JSONSerialization.jsonObject(with: original) as! [String: Any]
        func decode(_ value: [String: Any]) throws -> JarvisPluginCatalogSnapshot {
            try JSONDecoder().decode(JarvisPluginCatalogSnapshot.self, from: JSONSerialization.data(withJSONObject: value))
        }
        func edited(_ fields: [String: Any]) -> [String: Any] {
            var result = base
            var items = base["items"] as! [[String: Any]]
            for (key, value) in fields { items[0][key] = value }
            result["items"] = items
            return result
        }
        func rejected(_ value: [String: Any], _ name: String) {
            do { _ = try decode(value); preconditionFailure(name) }
            catch { checks += 1 }
        }
        let first = snapshot.presentations(at: now)[0]
        let disable = JarvisPluginPolicyChange(mutationID: UUID(), expectedRevision: 7,
            enabledChanges: [.init(pluginID: .videoEditor, enabled: false)], defaultChanges: [])
        let requesting = snapshot.presentations(at: now, pendingChange: disable, isSaving: true)
        check(requesting[0].operation == .requesting(.disable) && requesting[0].confirmedEnabled == true,
              "Per-card request state never optimistically disables the confirmed choice")
        check(requesting[0].isSelectedDefault && requesting[0].installationLabel == first.installationLabel,
              "Pending change retains remembered default and actual installed observation")
        check(Array(requesting.dropFirst()) == Array(snapshot.presentations(at: now).dropFirst()),
              "Unrelated plugin cards do not acquire another card's pending state")
        let uncertain = snapshot.presentations(at: now, pendingChange: disable, isSaving: false)[0]
        check(uncertain.operation?.isInProgress == false && uncertain.operation?.label == "Change not confirmed"
                && uncertain.operation?.recovery == nil && !uncertain.canChangePreference,
              "Lost reply has an explicit unknown state without invented lifecycle retry or endless spinner")
        let preferred = JarvisPluginPolicyChange(mutationID: UUID(), expectedRevision: 7, enabledChanges: [],
            defaultChanges: [.init(category: "video_editing", pluginID: nil)])
        let choosing = snapshot.presentations(at: now, pendingChange: preferred, isSaving: true)[0]
        check(choosing.operation == .requesting(.preference) && choosing.isSelectedDefault,
              "Default-only save shows per-card pending and preserves confirmed default until receipt")
        check(snapshot.presentations(at: now)[0] == first, "Request presentation never mutates the catalog")
        check(snapshot.items.count == 3 && snapshot.policyRevision == 7 && snapshot.policyAvailable,
              "Real pure-backend fixture decodes all three curated cards and confirmed revision")
        check(first.id == "video_editor" && first.confirmedEnabled == true && first.isSelectedDefault && first.readinessLabel == "Ready",
              "Confirmed ready preference maps separately from installation")
        check(first.installationLabel == "Installed version: fixture-1" && first.accountLabel == nil && !first.canChangePreference,
              "Installed version is observed; catalog does not invent an account or mutation permission")
        check(snapshot.presentations(at: now).dropFirst().allSatisfy { $0.confirmedEnabled == nil && !$0.isSelectedDefault && $0.readinessLabel == "Availability unknown" },
              "Missing plugin choices do not inherit defaults or another card's readiness")
        let deadline = snapshot.observationDeadlines[0]
        check(snapshot.presentations(at: deadline.addingTimeInterval(-0.001))[0].readinessLabel == "Ready",
              "Fresh status lasts only through the supplied deadline")
        let expired = snapshot.presentations(at: deadline)[0]
        check(expired.readinessLabel == "Availability unknown" && expired.installationLabel == "Installation status unknown",
              "Exact expiry removes ready and installed-version claims")
        check(expired.confirmedEnabled == true && expired.isSelectedDefault,
              "Expired observations do not erase a confirmed saved preference")
        check(snapshot.presentations(at: snapshot.evaluatedAt.addingTimeInterval(-1))[0].readinessLabel == "Availability unknown",
              "A future server evaluation cannot establish present readiness")

        let disabled = try decode(edited(["desired_enabled": false, "readiness": "unavailable", "reason_code": "plugin_disabled"]))
        let disabledCard = disabled.presentations(at: deadline.addingTimeInterval(100))[0]
        check(disabledCard.confirmedEnabled == false && disabledCard.isSelectedDefault && disabledCard.defaultLabel == "Preferred when enabled",
              "Disabled preferred choice survives expiry without fallback")
        check(disabledCard.readinessLabel == "Disabled" && !disabledCard.canChangePreference,
              "Accepted disabled policy remains unavailable even when observations expire")
        check(JarvisPluginCardPresentation.selectedDefaults.allSatisfy { $0.defaultLabel == "Suggested default" },
              "Local initial suggestions are not presented as authenticated defaults")

        let config = snapshot.items[0].configurationID!.uuidString
        func operation(_ phase: String, recovery: [String: Any]? = nil) -> [String: Any] {
            ["operation_ref": "existing-operation-7", "configuration_id": config, "phase": phase,
             "reason_code": "unsafe_secret_should_not_render", "recovery": recovery as Any? ?? NSNull()]
        }
        let installing = try decode(edited(["operation": operation("installing")]))
        let installingCard = installing.presentations(at: now)[0]
        check(installingCard.operation == .running(.installing, reference: "existing-operation-7") && installingCard.readinessLabel != "Ready",
              "Pending exact operation overrides even an inconsistent ready flag")
        let expiredInstall = installing.presentations(at: deadline)[0]
        check(expiredInstall.operation?.isInProgress == false && expiredInstall.operation?.recovery == nil,
              "Expired installation does not spin forever or manufacture a retry")
        check(installing.presentations(at: now)[1] == snapshot.presentations(at: now)[1],
              "Pending video operation does not change another plugin card")

        let failed = try decode(edited(["operation": operation("failed", recovery: ["kind": "retry", "reference": "original-safe-retry"])]))
        check(failed.presentations(at: now)[0].operation?.recovery == .retry("original-safe-retry"),
              "Fresh failed operation preserves only the explicitly supplied safe recovery")
        check(failed.presentations(at: deadline)[0].operation?.recovery == nil,
              "Expired failure cannot keep an old retry handle active")
        let unknown = try decode(edited(["operation": operation("unknown", recovery: ["kind": "recheck", "reference": "original-status-read"])]))
        check(unknown.presentations(at: now)[0].operation?.recovery == .recheck("original-status-read"),
              "Unknown change rechecks the same existing operation")
        let unknownNoAction = try decode(edited(["operation": operation("unknown")]))
        check(unknownNoAction.presentations(at: now)[0].operation?.recovery == nil,
              "Unknown without safe recovery remains explicit without a fabricated action")
        let safeMessage = failed.presentations(at: now)[0].operation?.message ?? ""
        check(!safeMessage.contains("unsafe_secret"), "Raw operation reason code is not rendered as an exception or credential")
        let unavailable = try decode(edited(["readiness": "unavailable", "reason_code": "unexpected_sensitive_reason"]))
        check(unavailable.presentations(at: now)[0].readinessLabel == "Unavailable",
              "Unrecognized reasons use a safe state-based fallback")

        let contradictory = try decode(edited(["desired_enabled": false]))
        check(contradictory.presentations(at: now)[0].readinessLabel == "Disabled", "Ready input cannot override a confirmed disabled preference")
        let missingVersion = try decode(edited(["installed_version": NSNull()]))
        check(missingVersion.presentations(at: now)[0].readinessLabel == "Availability unknown", "Ready needs actual installed version evidence")
        let noExpiry = try decode(edited(["observations_valid_until": NSNull()]))
        check(noExpiry.presentations(at: now)[0].readinessLabel == "Availability unknown", "Absent expiry cannot establish indefinitely ready software")

        var unknownPolicy = base
        unknownPolicy["policy_available"] = false; unknownPolicy["policy_revision"] = NSNull()
        var unknownItems = base["items"] as! [[String: Any]]
        for index in unknownItems.indices {
            unknownItems[index]["desired_enabled"] = NSNull(); unknownItems[index]["configuration_id"] = NSNull()
            unknownItems[index]["selected_default"] = false
        }
        unknownPolicy["items"] = unknownItems
        let missingPolicy = try decode(unknownPolicy)
        check(missingPolicy.presentations(at: now).allSatisfy { $0.confirmedEnabled == nil && !$0.isSelectedDefault && !$0.canChangePreference && $0.readinessLabel == "Availability unknown" },
              "Unavailable policy never becomes the deployment baseline or retains ready software")

        var wrongSchema = base; wrongSchema["schema_version"] = 2
        rejected(wrongSchema, "Unknown schema must not be silently interpreted")
        var duplicates = base; duplicates["items"] = [unknownItems[0], unknownItems[0]]
        rejected(duplicates, "Duplicate IDs must not alias independent cards")
        var negativeRevision = base; negativeRevision["policy_revision"] = -1
        rejected(negativeRevision, "Negative policy revision is incompatible")
        var falsePolicy = base; falsePolicy["policy_available"] = false
        rejected(falsePolicy, "Unavailable policy cannot still claim a confirmed revision")
        let permitted = try decode(edited(["can_change": true]))
        check(permitted.presentations(at: now)[0].canChangePreference,
              "Only explicit permission from the authenticated catalog can enable a confirmed choice")
        rejected(edited(["can_change": true, "desired_enabled": NSNull(), "configuration_id": NSNull(),
                         "selected_default": false, "operation": NSNull()]),
                 "A permission flag cannot enable an unknown policy choice")
        rejected(edited(["category": "web_browsing"]), "Preference cannot be mapped to the wrong category")
        rejected(edited(["configuration_id": NSNull()]), "Confirmed choice requires its configuration identity")
        rejected(edited(["observations_valid_until": "2026-11-01T23:01:00"]), "Naive local timestamp cannot extend server observation lifetime")
        rejected(edited(["observations_valid_until": 12345]), "Timestamp must not accept an ambiguous numeric unit")
        rejected(edited(["operation": operation("unknown", recovery: ["kind": "retry", "reference": "new-effect"])]),
                 "Unknown response cannot offer a newly inferred retry")
        rejected(edited(["operation": operation("installing", recovery: ["kind": "recheck", "reference": "another-read"])]),
                 "Active phase cannot expose a conflicting recovery action")
        var wrongOperation = operation("failed"); wrongOperation["configuration_id"] = UUID().uuidString
        rejected(edited(["operation": wrongOperation]), "An old configuration cannot control the current card")
        var badReference = operation("unknown"); badReference["operation_ref"] = "https://example.invalid/action"
        rejected(edited(["operation": badReference]), "Opaque operation reference cannot become an arbitrary URL")

        var fractional = base; fractional["evaluated_at"] = "2026-11-02T00:00:00.125+01:00"
        let fraction = try decode(fractional)
        check(abs(fraction.evaluatedAt.timeIntervalSince(snapshot.evaluatedAt) - 0.125) < 0.0001,
              "Offset and fractional timestamp preserve the exact observation time")
        print("Phone plugin catalog regression: \(checks) checks passed (pure payload only)")
    }
}
