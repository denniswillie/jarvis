import SwiftUI

struct JarvisPreferencesView: View {
    @ObservedObject var model: JarvisPreferencesViewModel
    var openSettings: () -> Void
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var isEditing = false
    @FocusState private var editorFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Make Jarvis yours")
                            .font(JarvisTypography.pageTitle)
                            .foregroundStyle(JarvisColors.primaryText)
                        Text("Tell Jarvis how you like to be helped. You can change this here or ask Jarvis aloud.")
                            .font(.body)
                            .foregroundStyle(JarvisColors.secondaryText)
                            .lineSpacing(4)
                    }

                    if model.savedProfile != nil {
                        VStack(alignment: .leading, spacing: 12) {
                            if isEditing {
                                TextEditor(text: $model.draft)
                                    .font(.body)
                                    .scrollContentBackground(.hidden)
                                    .padding(12)
                                    .frame(minHeight: dynamicTypeSize.isAccessibilitySize ? 260 : 180)
                                    .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 18))
                                    .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(JarvisColors.border))
                                    .focused($editorFocused)
                                    .accessibilityLabel("Your preferences")
                                    .accessibilityIdentifier("preferencesEditor")
                                    .onAppear { editorFocused = true }
                            } else {
                                Text(model.draft.isEmpty ? "Add your preferences when you’re ready." : model.draft)
                                    .font(.body)
                                    .foregroundStyle(model.draft.isEmpty ? JarvisColors.secondaryText : JarvisColors.primaryText)
                                    .lineSpacing(5)
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, minHeight: 180, alignment: .topLeading)
                                    .accessibilityIdentifier("preferencesSavedText")
                            }
                            if model.draft.unicodeScalars.count > 12_000 {
                                Text("Keep your preferences under 12,000 characters.")
                                    .font(.footnote).foregroundStyle(JarvisColors.clay)
                            }
                        }
                    } else if model.isLoading {
                        ProgressView("Loading preferences")
                            .frame(maxWidth: .infinity, minHeight: 120)
                    }

                }
                .padding(dynamicTypeSize.isAccessibilitySize ? 18 : 24)
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .background(JarvisColors.background)
            .safeAreaInset(edge: .bottom, spacing: 0) { editingDock }
            .navigationTitle("Preferences")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if !isEditing {
                    ToolbarItem(placement: .topBarLeading) {
                        Button(action: openSettings) {
                            Label("Settings", systemImage: "gearshape")
                        }
                        .accessibilityLabel("Open settings")
                        .accessibilityIdentifier("preferencesSettingsButton")
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Close") { dismiss() }.accessibilityLabel("Close preferences")
                    }
                }
            }
            .interactiveDismissDisabled(isEditing)
            .onChange(of: model.isLocked) { _, locked in
                if locked { editorFocused = false; isEditing = false }
            }
            .tint(JarvisColors.accent)
            .task { await model.refresh() }
            .refreshable { await model.refresh() }
        }
    }

    /// Safe-area placement keeps these real controls above the software keyboard.
    /// Saving must be acknowledged before leaving editing; failed writes retain it.
    private var editingDock: some View {
        VStack(alignment: .leading, spacing: 12) {
            if model.needsReview { reviewChanges }
            if let error = model.errorMessage {
                Text(error).font(.subheadline).foregroundStyle(JarvisColors.clay)
                    .accessibilityIdentifier("preferencesError")
            } else if isEditing, let status = model.statusMessage {
                Text(status).font(.subheadline).foregroundStyle(JarvisColors.secondaryText)
            }
            if model.savedProfile != nil && !model.isLocked {
                HStack {
                    Spacer()
                    if isEditing {
                        Button {
                            guard !model.isSaving else { return }
                            Task {
                                let confirmed = model.isDirty ? await model.save() : !model.needsReview
                                guard confirmed, !model.isLocked, model.savedProfile != nil,
                                      !model.isDirty, !model.needsReview else { return }
                                editorFocused = false
                                isEditing = false
                            }
                        } label: {
                            actionIcon("checkmark", busy: model.isSaving)
                        }
                        .disabled(model.isSaving || model.needsReview || (model.isDirty && !model.canSave))
                        .accessibilityLabel(model.isSaving ? "Saving preferences" : "Done")
                        .accessibilityHint("Saves your preferences and closes the keyboard")
                        .accessibilityIdentifier("preferencesSaveButton")
                    } else {
                        Button { isEditing = true } label: { actionIcon("pencil") }
                            .accessibilityLabel("Edit preferences")
                            .accessibilityIdentifier("preferencesEditButton")
                    }
                }
                .buttonStyle(.plain)
            } else if model.errorMessage != nil && !model.isLocked {
                Button("Try again") { Task { await model.refresh() } }.frame(minHeight: 44)
            }
        }
        .padding(.horizontal, dynamicTypeSize.isAccessibilitySize ? 18 : 24)
        .padding(.vertical, 14)
        .frame(maxWidth: 760)
        .frame(maxWidth: .infinity)
        .background(JarvisColors.background)
    }

    private func actionIcon(_ symbol: String, busy: Bool = false) -> some View {
        ZStack {
            Circle().fill(JarvisColors.accent)
            if busy { ProgressView().tint(.white) }
            else { Image(systemName: symbol).font(.title3.weight(.semibold)).foregroundStyle(.white) }
        }
        .frame(width: 54, height: 54)
        .contentShape(Circle())
    }

    private var reviewChanges: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Preferences changed elsewhere")
                .font(.headline)
            if model.conflictRequiresRefresh {
                Button("Load latest preferences") { Task { await model.refresh() } }
                    .frame(minHeight: 44)
            }
            ScrollView {
                Text(model.savedProfile.flatMap { $0.behaviorInstructions.isEmpty ? nil : $0.behaviorInstructions }
                     ?? "No saved preferences.")
                    .font(.subheadline).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: dynamicTypeSize.isAccessibilitySize ? 100 : 64)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { conflictActions }
                VStack(alignment: .leading, spacing: 4) { conflictActions }
            }
            .disabled(model.isSaving || model.conflictRequiresRefresh)
        }
        .padding(14)
        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(JarvisColors.border))
        .accessibilityIdentifier("preferencesConflictReview")
    }

    @ViewBuilder private var conflictActions: some View {
        Button("Use saved preferences") { model.useSavedVersion() }
            .frame(minHeight: 44)
        Button("Keep my draft") { model.keepDraftAfterReview() }
            .frame(minHeight: 44)
    }
}
