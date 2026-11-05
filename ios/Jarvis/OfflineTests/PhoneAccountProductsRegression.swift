import Foundation

private final class AccountFixture: JarvisPreferencesAccess {
    var profile = JarvisBehaviorProfile(ownerID: "owner", behaviorInstructions: "Be concise.", revision: 1, createdAt: "created", updatedAt: "updated")
    var getError: Error?, saveError: Error?
    var holdGet = false, holdSave = false
    var getWait: CheckedContinuation<JarvisBehaviorProfile, Error>?
    var saveWait: CheckedContinuation<JarvisBehaviorProfile, Error>?
    var writes: [JarvisBehaviorProfileUpdate] = []

    func fetchBehaviorProfile() async throws -> JarvisBehaviorProfile {
        if holdGet { return try await withCheckedThrowingContinuation { getWait = $0 } }
        if let getError { throw getError }; return profile
    }
    func updateBehaviorProfile(_ update: JarvisBehaviorProfileUpdate) async throws -> JarvisBehaviorProfile {
        writes.append(update)
        if holdSave { return try await withCheckedThrowingContinuation { saveWait = $0 } }
        if let saveError { throw saveError }
        profile = JarvisBehaviorProfile(ownerID: profile.ownerID,
            behaviorInstructions: update.behaviorInstructions.trimmingCharacters(in: .whitespacesAndNewlines),
            revision: profile.revision + 1, createdAt: profile.createdAt, updatedAt: "saved")
        return profile
    }

}

@main
enum PhoneAccountProductsRegression {
    @MainActor
    static func main() async throws {
        var count = 0
        func check(_ value: Bool, _ name: String) { precondition(value, name); count += 1 }
        func profile(_ revision: Int64, _ text: String, owner: String = "owner") -> JarvisBehaviorProfile {
            JarvisBehaviorProfile(ownerID: owner, behaviorInstructions: text, revision: revision, createdAt: "created", updatedAt: "updated")
        }
        func event(_ revision: Int64, _ text: String, owner: String = "owner") -> JarvisBehaviorProfileCommitted {
            JarvisBehaviorProfileCommitted(type: "behavior_profile.committed", eventID: "event-\(revision)", ownerID: owner,
                                           revision: revision, behaviorInstructions: text, committedAt: "committed")
        }
        @MainActor func waitFor(_ predicate: () -> Bool) async throws {
            for _ in 0..<100 {
                if predicate() { return }; try await Task.sleep(nanoseconds: 1_000_000)
            }
            preconditionFailure("Fixture operation did not begin")
        }

        let api = AccountFixture(), model = JarvisPreferencesViewModel(api: AccountFixture())
        // This independent model is never initialized from an unverified local cache.
        check(model.savedProfile == nil && !model.canSave, "Preferences begin unloaded and cannot save")
        await model.committed(event(2, "Unverified event"))
        check(model.savedProfile == nil && model.draft.isEmpty,
              "An event cannot initialize preferences before owner verification")
        let initialAPI = AccountFixture(), initial = JarvisPreferencesViewModel(api: AccountFixture())
        initial.verifyOwner("owner")
        await initial.committed(event(2, "Wrong owner", owner: "other"))
        check(initial.savedProfile == nil, "Initial events retain the verified-owner fence")
        initialAPI.holdGet = true
        let pendingInitial = JarvisPreferencesViewModel(api: initialAPI)
        pendingInitial.verifyOwner("owner")
        let initialRead = Task { await pendingInitial.refresh() }
        try await waitFor { initialAPI.getWait != nil }
        await pendingInitial.committed(event(2, "Committed during the first read"))
        check(pendingInitial.savedProfile?.revision == 2 && pendingInitial.draft == "Committed during the first read",
              "A verified-owner push can initialize an editor while its first read is pending")
        initialAPI.getWait?.resume(returning: profile(1, "Older initial response")); initialAPI.getWait = nil
        await initialRead.value
        check(pendingInitial.savedProfile?.revision == 2 && pendingInitial.baseRevision == 2 && !pendingInitial.isDirty,
              "A late initial read cannot overwrite the newer committed profile")
        let preferences = JarvisPreferencesViewModel(api: api)
        await preferences.refresh()
        check(preferences.draft == "Be concise." && preferences.baseRevision == 1 && !preferences.isDirty,
              "First authoritative read initializes clean editor")
        await preferences.committed(event(2, "Use plain English."))
        check(preferences.draft == "Use plain English." && preferences.baseRevision == 2,
              "A clean editor follows a committed update")
        preferences.draft = "My unsaved instructions"
        await preferences.committed(event(3, "Keep it brief."))
        check(preferences.draft == "My unsaved instructions" && preferences.baseRevision == 2 && preferences.needsReview,
              "Push preserves dirty draft and its original revision")
        let blockedSave = await preferences.save()
        check(!blockedSave && api.writes.isEmpty, "Conflict cannot silently write, rebase or claim confirmation")
        preferences.keepDraftAfterReview()
        check(preferences.baseRevision == 3 && preferences.canSave && preferences.draft == "My unsaved instructions",
              "Explicit review keeps draft against the latest saved version")
        api.profile = profile(3, "Keep it brief.")
        let confirmedSave = await preferences.save()
        check(confirmedSave, "Only an exact acknowledged save reports success to the editing UI")
        check(api.writes.last?.expectedRevision == 3 && api.writes.last?.behaviorInstructions == "My unsaved instructions",
              "Save submits exact draft and reviewed revision")
        check(preferences.baseRevision == 4 && !preferences.isDirty && preferences.statusMessage == "Preferences saved.",
              "Acknowledged save marks only that text clean")
        await preferences.committed(event(2, "Old text"))
        await preferences.committed(event(9, "Other account", owner: "other"))
        check(preferences.savedProfile?.revision == 4 && preferences.draft == "My unsaved instructions",
              "Old revisions and other owners cannot replace visible preferences")

        api.holdSave = true; preferences.draft = "First submitted edit"
        let saving = Task { await preferences.save() }
        try await waitFor { api.saveWait != nil }
        let writesBeforeDuplicate = api.writes.count
        let duplicateSave = await preferences.save()
        check(!duplicateSave && api.writes.count == writesBeforeDuplicate && preferences.isSaving,
              "In-flight Done cannot submit or report a second save")
        preferences.draft = "Typed while saving"
        api.saveWait?.resume(returning: profile(5, "First submitted edit")); api.saveWait = nil
        await saving.value
        check(preferences.draft == "Typed while saving" && preferences.baseRevision == 5 && preferences.isDirty,
              "Typing during a save survives its completion")
        let secondSave = Task { await preferences.save() }
        try await waitFor { api.saveWait != nil }
        await preferences.committed(event(7, "Voice updated after save"))
        api.saveWait?.resume(returning: profile(6, "Typed while saving")); api.saveWait = nil
        await secondSave.value
        check(preferences.savedProfile?.revision == 7 && preferences.baseRevision == 6 && preferences.needsReview,
              "Late save response cannot overwrite a newer committed profile")
        preferences.useSavedVersion()
        check(preferences.draft == "Voice updated after save" && !preferences.isDirty && !preferences.needsReview,
              "Use saved version resolves review explicitly")

        api.holdSave = false; api.profile = profile(8, "External change")
        api.saveError = JarvisAPIError(code: "behavior_profile_revision_conflict", message: "Conflict", statusCode: 409)
        preferences.draft = "Still my draft"
        await preferences.save()
        check(preferences.draft == "Still my draft" && preferences.savedProfile?.revision == 8 && preferences.baseRevision == 7,
              "409 reads latest while preserving draft and old edit base")
        check(preferences.needsReview && !preferences.canSave && !preferences.conflictRequiresRefresh,
              "Conflict requires user review even after successful reload")
        preferences.keepDraftAfterReview()
        api.getError = JarvisAPIError(code: "offline", message: "Offline")
        await preferences.save()
        check(preferences.conflictRequiresRefresh && !preferences.canSave,
              "Failed conflict reload prevents a blind write retry")
        preferences.keepDraftAfterReview()
        check(preferences.conflictRequiresRefresh, "Cannot review unknown latest data")
        api.getError = nil; api.profile = profile(9, "Latest")
        await preferences.refresh()
        check(!preferences.conflictRequiresRefresh && preferences.needsReview && preferences.draft == "Still my draft",
              "Reconnect recovers latest without overwriting unsaved text")

        preferences.useSavedVersion(); api.saveError = nil
        preferences.draft = "  \n "
        await preferences.save()
        check(preferences.draft.isEmpty && preferences.baseRevision == 10 && !preferences.isDirty,
              "Whitespace clearing uses the server-normalized saved text")
        preferences.draft = String(repeating: "🧑‍🔬", count: 4001)
        check(!preferences.canSave, "Length limit counts Unicode scalars like backend code points")

        api.holdGet = true
        let read = Task { await preferences.refresh() }
        try await waitFor { api.getWait != nil }
        preferences.deactivate()
        api.getWait?.resume(returning: profile(11, "Late private data")); api.getWait = nil
        await read.value
        check(preferences.savedProfile == nil && preferences.draft.isEmpty && !preferences.canSave,
              "Retired account cannot repopulate from late GET")
        let deniedAPI = AccountFixture(), denied = JarvisPreferencesViewModel(api: deniedAPI)
        await denied.refresh(); denied.draft = "Private draft"
        deniedAPI.getError = JarvisAPIError(code: "denied", message: "Denied", statusCode: 403)
        await denied.refresh()
        check(denied.isLocked && denied.savedProfile == nil && denied.draft.isEmpty,
              "Authorization loss hides saved text and draft")
        let mismatch = JarvisPreferencesViewModel(api: AccountFixture())
        mismatch.verifyOwner("different-owner"); await mismatch.refresh()
        check(mismatch.isLocked && mismatch.savedProfile == nil, "Verified owner and profile owner must agree")

        let failureAPI = AccountFixture(), failed = JarvisPreferencesViewModel(api: AccountFixture())
        let retrying = JarvisPreferencesViewModel(api: failureAPI)
        await retrying.refresh(); retrying.draft = "Keep this draft after failure."
        failureAPI.saveError = JarvisAPIError(code: "unavailable", message: "Try again", statusCode: 503)
        let failedSave = await retrying.save()
        check(!failedSave && retrying.draft == "Keep this draft after failure." && retrying.isDirty &&
              retrying.canSave && retrying.errorMessage != nil && retrying.statusMessage == nil,
              "Failed save reports no confirmation and preserves retryable draft")
        failureAPI.saveError = nil; failureAPI.holdSave = true
        let invalidSave = Task { await retrying.save() }
        try await waitFor { failureAPI.saveWait != nil }
        failureAPI.saveWait?.resume(returning: profile(2, "Different response text")); failureAPI.saveWait = nil
        let invalidAcknowledgement = await invalidSave.value
        check(!invalidAcknowledgement && retrying.isDirty && retrying.baseRevision == 1,
              "Mismatched acknowledgement cannot close editing as success")
        await failed.refresh(); failed.draft = "Scoped private edit"
        failed.deactivate()
        let retiredSave = await failed.save()
        check(!retiredSave && failed.savedProfile == nil, "Retired account cannot report save confirmation")

        print("Phone account-product regressions passed: \(count) checks")
    }
}
