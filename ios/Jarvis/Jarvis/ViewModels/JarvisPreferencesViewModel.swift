import Combine
import Foundation

/// One backend profile. Drafts stay in memory and never silently rebase onto a push.
@MainActor
final class JarvisPreferencesViewModel: ObservableObject {
    @Published private(set) var savedProfile: JarvisBehaviorProfile?
    @Published var draft = ""
    @Published private(set) var isLoading = false
    @Published private(set) var isSaving = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var statusMessage: String?
    @Published private(set) var isLocked = false
    @Published private(set) var baseRevision: Int64?
    @Published private(set) var conflictRequiresRefresh = false
    var onAuthorizationFailure: (() -> Void)?

    private let api: any JarvisPreferencesAccess
    private var baseInstructions = ""
    private var active = true
    private var lifetime = UUID()
    private var readGeneration = 0
    private var expectedOwnerID: String?

    init(api: any JarvisPreferencesAccess) { self.api = api }

    var isDirty: Bool { draft != baseInstructions }
    var needsReview: Bool {
        if conflictRequiresRefresh { return true }
        guard let savedProfile, let baseRevision else { return false }
        return savedProfile.revision != baseRevision
    }
    var canSave: Bool {
        active && !isLocked && !isSaving && savedProfile != nil && baseRevision != nil &&
        (baseRevision ?? Int64.max) < Int64.max &&
        isDirty && !needsReview && draft.unicodeScalars.count <= 12_000
    }

    func verifyOwner(_ ownerID: String) {
        guard active else { return }
        if let savedProfile, savedProfile.ownerID != ownerID {
            lock(); onAuthorizationFailure?()
        } else { expectedOwnerID = ownerID }
    }

    func refresh(clearError: Bool = true) async {
        guard active, !isLocked, !Task.isCancelled else { return }
        let token = lifetime
        readGeneration += 1
        let generation = readGeneration
        isLoading = true
        if clearError { errorMessage = nil }
        defer { if readGeneration == generation { isLoading = false } }
        do {
            let profile = try await api.fetchBehaviorProfile()
            guard active, !isLocked, !Task.isCancelled, lifetime == token, readGeneration == generation else { return }
            try accept(profile)
            if let baseRevision, profile.revision > baseRevision { conflictRequiresRefresh = false }
        } catch {
            guard active, !isLocked, !Task.isCancelled, lifetime == token, readGeneration == generation else { return }
            handle(error)
        }
    }

    @discardableResult
    func save() async -> Bool {
        guard canSave, let baseRevision, !Task.isCancelled else { return false }
        let token = lifetime, submitted = draft
        let update = JarvisBehaviorProfileUpdate(behaviorInstructions: submitted, expectedRevision: baseRevision)
        isSaving = true; errorMessage = nil; statusMessage = nil
        defer { if lifetime == token { isSaving = false } }
        do {
            let profile = try await api.updateBehaviorProfile(update)
            guard active, !isLocked, !Task.isCancelled, lifetime == token else { return false }
            guard profile.revision == baseRevision + 1,
                  profile.behaviorInstructions == submitted.trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw JarvisAPIError(code: "profile_invalid_revision", message: "Could not verify the saved preferences. Refresh to check them.")
            }
            try accept(profile, updateCleanDraft: false)
            // A push or further typing can arrive while the request is in flight.
            // Advance only to this exact acknowledged save; retain any newer draft.
            self.baseRevision = profile.revision
            baseInstructions = profile.behaviorInstructions
            if draft == submitted { draft = profile.behaviorInstructions }
            statusMessage = needsReview ? "Saved. Preferences also changed elsewhere; review the latest version." : "Preferences saved."
            return true
        } catch {
            guard active, !isLocked, !Task.isCancelled, lifetime == token else { return false }
            if let error = error as? JarvisAPIError, error.code == "behavior_profile_revision_conflict" {
                conflictRequiresRefresh = true
                errorMessage = "Preferences changed elsewhere. Your draft is still here. Review the latest version before saving."
                await refresh(clearError: false)
            } else { handle(error) }
            return false
        }
    }

    func committed(_ event: JarvisBehaviorProfileCommitted) async {
        guard active, !isLocked, event.type == "behavior_profile.committed", !event.eventID.isEmpty,
              event.revision > 0, event.ownerID == expectedOwnerID else { return }
        let profile = JarvisBehaviorProfile(ownerID: event.ownerID, behaviorInstructions: event.behaviorInstructions,
                                           revision: event.revision, createdAt: savedProfile?.createdAt,
                                           updatedAt: event.committedAt)
        do {
            try accept(profile)
            if let baseRevision, profile.revision > baseRevision { conflictRequiresRefresh = false }
        }
        catch { errorMessage = error.localizedDescription }
    }

    /// Explicit conflict resolution; neither action sends a write.
    func useSavedVersion() {
        guard active, !isLocked, !isSaving, !conflictRequiresRefresh, let savedProfile else { return }
        draft = savedProfile.behaviorInstructions
        baseInstructions = savedProfile.behaviorInstructions; baseRevision = savedProfile.revision
        errorMessage = nil; statusMessage = nil
    }

    func keepDraftAfterReview() {
        guard active, !isLocked, !isSaving, !conflictRequiresRefresh, let savedProfile else { return }
        baseInstructions = savedProfile.behaviorInstructions; baseRevision = savedProfile.revision
        errorMessage = nil; statusMessage = "Draft kept. Save when you’re ready."
    }

    func deactivate() {
        active = false; lifetime = UUID(); readGeneration += 1
        savedProfile = nil; draft = ""; baseInstructions = ""; baseRevision = nil
        isLoading = false; isSaving = false; errorMessage = nil; statusMessage = nil
        conflictRequiresRefresh = false
    }

    private func accept(_ profile: JarvisBehaviorProfile, updateCleanDraft: Bool = true) throws {
        guard profile.isValid,
              expectedOwnerID == nil || expectedOwnerID == profile.ownerID,
              savedProfile == nil || savedProfile?.ownerID == profile.ownerID else {
            throw JarvisAPIError(code: "profile_identity_mismatch", message: "Could not verify these preferences. Sign in again.", statusCode: 403)
        }
        if let current = savedProfile {
            if profile.revision < current.revision { return }
            if profile.revision == current.revision {
                guard profile.behaviorInstructions == current.behaviorInstructions else {
                    throw JarvisAPIError(code: "profile_revision_mismatch", message: "Preferences could not be verified. Refresh to try again.")
                }
                return
            }
        }
        let wasClean = !isDirty && !isSaving
        savedProfile = profile
        expectedOwnerID = profile.ownerID
        if baseRevision == nil || (updateCleanDraft && wasClean) {
            draft = profile.behaviorInstructions
            baseInstructions = profile.behaviorInstructions; baseRevision = profile.revision
        }
        statusMessage = nil
    }

    private func handle(_ error: Error) {
        if let error = error as? JarvisAPIError, error.statusCode == 401 || error.statusCode == 403 {
            lock(); onAuthorizationFailure?()
        } else if let error = error as? JarvisAPIError,
                  error.statusCode == 404 || error.code == "behavior_profile_unavailable" {
            errorMessage = "Preferences aren’t available right now. Try again later or check App settings."
            if isDirty { errorMessage? += " Your draft is still here." }
        } else { errorMessage = error.localizedDescription }
    }

    func lockForSignIn() { lock() }

    private func lock() {
        lifetime = UUID(); readGeneration += 1
        isLocked = true; savedProfile = nil; draft = ""; baseInstructions = ""; baseRevision = nil
        isLoading = false; isSaving = false; statusMessage = nil
        errorMessage = "Sign in again to view and edit your preferences."
    }
}
