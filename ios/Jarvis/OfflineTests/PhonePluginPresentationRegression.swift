import Foundation

@main
enum PhonePluginPresentationRegression {
    static func main() {
        typealias Card = JarvisPluginCardPresentation
        var checks = 0
        func check(_ condition: Bool, _ message: String) { precondition(condition, message); checks += 1 }
        func card(_ enabled: Bool?, operation: Card.Operation? = nil, allowed: Bool = true) -> Card {
            Card(id: "fixture-editor", title: "Video Editor", symbol: "film", detail: "Fixture descriptor",
                readinessLabel: "Availability unknown", accountLabel: nil, isSelectedDefault: false,
                confirmedEnabled: enabled, allowsPreferenceChange: allowed, operation: operation)
        }
        check(Card.selectedDefaults.allSatisfy { $0.confirmedEnabled == nil && !$0.canChangePreference && $0.operation == nil },
              "Selected software defaults never pretend to be enabled, installed or currently operating")
        check(card(true).canChangePreference && !card(nil).canChangePreference,
              "A preference control requires confirmed state and an explicit adapter permission")
        check(!card(true, allowed: false).canChangePreference, "A UI row cannot manufacture backend permission")
        let requesting = card(false, operation: .requesting(.enable))
        check(requesting.confirmedEnabled == false && !requesting.canChangePreference,
              "Submitting enable does not optimistically confirm enablement")
        check(requesting.operation?.label == "Requesting change…" && requesting.operation?.isInProgress == true,
              "Local submission is not mislabelled as an accepted installation")
        let installing = card(false, operation: .running(.installing, reference: "existing-op-1"))
        let unaffected = card(true)
        check(installing.operation?.label == "Installing…" && installing.operation?.isInProgress == true && !installing.canChangePreference,
              "Installation is pending on only that card")
        check(unaffected.canChangePreference, "Another plugin stays interactive during installation")
        let removing = card(true, operation: .running(.removing, reference: "existing-op-2"))
        check(removing.confirmedEnabled == true && removing.operation?.label == "Removing…" && !removing.canChangePreference,
              "Removal progress cannot pretend the final disabled state was confirmed")
        let unknown = card(true, operation: .unknown(message: "Response lost", recheckReference: "existing-op-2"))
        check(unknown.operation?.recovery == .recheck("existing-op-2") && !unknown.canChangePreference,
              "Unknown outcome rechecks its existing operation and blocks conflicting new edits")
        check(unknown.confirmedEnabled == true && unknown.operation?.isInProgress == false,
              "Uncertain change is explicit and never shown as a completed disable or endless spinner")
        let failedUnknownEffect = card(false, operation: .failed(message: "Unavailable", recovery: nil))
        check(failedUnknownEffect.operation?.recovery == nil, "A failure alone is not permission to retry an effect")
        let failedSafe = card(false, operation: .failed(message: "Not started", recovery: .retry("same-request")))
        check(failedSafe.operation?.recovery == .retry("same-request"), "Only adapter-confirmed safe retry exposes the original request reference")
        let remembered = Card(id: "fixture-preferred", title: "Editor", symbol: "film", detail: "Fixture",
            readinessLabel: "Disabled", accountLabel: nil, isSelectedDefault: true, confirmedEnabled: false,
            allowsPreferenceChange: true, operation: nil)
        check(remembered.isSelectedDefault && remembered.confirmedEnabled == false && remembered.defaultLabel == "Preferred when enabled",
              "Disabling preserves the preferred choice while clearly making it inactive")
        let unknownWithoutRoute = card(true, operation: .unknown(message: "Outcome unknown", recheckReference: nil))
        check(unknownWithoutRoute.operation?.recovery == nil && !unknownWithoutRoute.canChangePreference,
              "Unknown outcome without an owner recovery route stays explicit without manufacturing a recheck action")
        let failedRecheck = card(false, operation: .failed(message: "Needs a status check", recovery: .recheck("existing-operation")))
        check(failedRecheck.operation?.recovery == .recheck("existing-operation"), "A failed status can preserve an owner-supplied read-only recheck")
        print("PhonePluginPresentationRegression: \(checks) checks passed (presentation only, no backend lifecycle)")
    }
}
