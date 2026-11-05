import SwiftUI

private typealias WorkDestination = JarvisWorkDetailSelection

struct JarvisWorkInboxView<FilePreview: View>: View {
    @ObservedObject var model: JarvisWorkInboxViewModel
    var notificationLink: JarvisNotificationLink? = nil
    var productLink: JarvisProductWorkLink? = nil
    let filePreview: (UUID) -> FilePreview
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @State private var section = 0
    @State private var path: [WorkDestination] = []

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Your work")
                        .font(JarvisTypography.headline(26))
                    Text("Background tasks and saved answers.")
                        .font(JarvisTypography.body(14)).foregroundStyle(JarvisColors.softWhite)
                    Picker("Work section", selection: $section) {
                        Text("Tasks").tag(0)
                        Text("Inbox").tag(1)
                    }.pickerStyle(.segmented)
                    WorkMutationStatus(model: model)
                    if section == 1 {
                        Toggle("Include dismissed and unavailable", isOn: $model.includeDismissed)
                            .font(JarvisTypography.body(13))
                            .onChange(of: model.includeDismissed) { _, _ in Task { await model.reload() } }
                    }
                    if model.isLoading {
                        ProgressView("Fetching current state")
                    } else {
                        if let error = section == 0 ? model.tasksError : model.inboxError {
                            Text(error).foregroundStyle(JarvisColors.amber)
                            Button("Refresh") { Task { await model.reload() } }
                        }
                        if section == 0 {
                            if model.tasks.isEmpty { Text("No background tasks yet.").foregroundStyle(JarvisColors.softWhite) }
                            ForEach(model.tasks) { task in
                                NavigationLink(value: WorkDestination.task(task.id)) {
                                    row(task.displayTitle, subtitle: task.statusLabel, icon: "arrow.triangle.branch")
                                }
                                Divider()
                            }
                        } else {
                            if model.inbox.isEmpty { Text("Your inbox is empty.").foregroundStyle(JarvisColors.softWhite) }
                            ForEach(model.inbox) { item in
                                NavigationLink(value: WorkDestination.inbox(item.id)) {
                                    row(title(item), subtitle: item.stateLabel, icon: "tray")
                                }
                                Divider()
                            }
                        }
                        if section == 0 ? model.moreTasks : model.moreInbox {
                            Button(model.isLoadingMore ? "Loading..." : "Load more") {
                                Task { await model.loadMore(tasks: section == 0) }
                            }.disabled(model.isLoadingMore)
                        }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
                    .padding(22).foregroundStyle(JarvisColors.cream)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(JarvisColors.graphite)
            .navigationTitle("Tasks and inbox")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Refresh") { Task { await model.reload() } }.disabled(model.isLoading)
                }
                ToolbarItem(placement: .topBarTrailing) { Button("Close") { dismiss() } }
            }
            .refreshable { await model.reload() }
            .navigationDestination(for: WorkDestination.self) { destination in
                switch destination {
                case let .task(id): WorkTaskDetail(model: model, id: id)
                case let .inbox(id):
                    WorkInboxDetail(model: model, id: id, filePreview: filePreview)
                        .id(notificationLink?.id ?? productLink?.id)
                }
            }
        }
        .tint(JarvisColors.amber)
        .preferredColorScheme(.light)
        .task(id: notificationLink?.id ?? productLink?.id) {
            if let notificationLink {
                model.clearDetail()
                model.beginPresentation()
                section = 1
                path = [.inbox(notificationLink.intentID)]
                await model.openInbox(notificationLink.intentID)
            } else if let productLink {
                model.clearDetail()
                model.beginPresentation()
                switch productLink.destination {
                case .task(let id): section = 0; path = [.task(id)]; await model.openTask(id)
                case .inbox(let id): section = 1; path = [.inbox(id)]; await model.openInbox(id)
                }
            } else {
                await model.present()
            }
        }
        .onChange(of: path) { _, destinations in
            guard scenePhase == .active else { return }
            guard let target = destinations.last else { model.clearDetail(); return }
            guard target != model.detailSelection else { return }
            Task {
                model.beginPresentation()
                switch target { case .task(let id): await model.openTask(id); case .inbox(let id): await model.openInbox(id) }
            }
        }
        .onDisappear {
            if scenePhase == .active { model.hide() }
            else { model.suspend() }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                model.suspend()
            } else {
                Task { await model.resume(selection: path.last) }
            }
        }
    }

    private func title(_ item: JarvisInboxItem) -> String {
        guard let id = item.taskID, let task = model.tasks.first(where: { $0.id == id }) else { return item.sourceLabel }
        return task.displayTitle
    }

    private func row(_ title: String, subtitle: String, icon: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon).foregroundStyle(JarvisColors.amber)
            VStack(alignment: .leading, spacing: 5) {
                Text(verbatim: title).font(JarvisTypography.label(16))
                Text(subtitle).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.softWhite)
            }
            Spacer(minLength: 0)
        }
        .foregroundStyle(JarvisColors.cream).multilineTextAlignment(.leading)
        .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 8)
    }
}

private struct WorkMutationStatus: View {
    @ObservedObject var model: JarvisWorkInboxViewModel
    var body: some View {
        if model.isMutating { ProgressView("Submitting request") }
        if let message = model.mutationMessage {
            Text(verbatim: message).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.amber)
        }
        if model.pendingAction != nil && !model.isMutating {
            if let label = model.pendingActionLabel { Text(label).font(JarvisTypography.label(13)) }
            Button("Retry the same request") { Task { await model.retry() } }
            Text("Resolve this request before making another change.")
                .font(JarvisTypography.body(12)).foregroundStyle(JarvisColors.mutedWhite)
        }
    }
}

private struct WorkTaskDetail: View {
    @ObservedObject var model: JarvisWorkInboxViewModel
    let id: UUID
    @State private var draft = JarvisWorkReplyDraft()
    @State private var confirmCancel = false
    @State private var confirmResume = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                WorkMutationStatus(model: model)
                if model.isLoadingDetail { ProgressView("Fetching task") }
                if let error = model.detailError { Text(error).foregroundStyle(JarvisColors.amber) }
                if let error = model.taskAnswersError { Text(error).font(.footnote).foregroundStyle(JarvisColors.secondaryText) }
                if let task = model.selectedTask, task.id == id {
                    Text(verbatim: task.displayTitle).font(JarvisTypography.headline(26))
                    Text(task.statusLabel).font(JarvisTypography.label(15)).foregroundStyle(JarvisColors.amber)
                    if let date = model.detailRefreshedAt {
                        Text("Status checked \(date.formatted(.relative(presentation: .named)))").font(.caption).foregroundStyle(.secondary)
                    }
                    if let outcome = task.requestedOutcome { Text(verbatim: outcome).font(JarvisTypography.body(15)).textSelection(.enabled) }
                    if let reason = task.reason {
                        DisclosureGroup("More details") { Text(reason).font(.footnote).foregroundStyle(.secondary) }
                    }
                    ForEach(model.taskInbox) { item in
                        NavigationLink(value: WorkDestination.inbox(item.id)) {
                            Label(item.sourceLabel + " / " + item.stateLabel, systemImage: "text.bubble")
                        }
                    }
                    if task.canUpdate {
                        Text(task.status == "waiting_for_input" ? "Reply to the expert" : "Correct or update this task")
                            .font(JarvisTypography.label(16))
                        TextEditor(text: Binding(get: { draft.text }, set: { draft.update($0, for: task) })).frame(minHeight: 110)
                            .scrollContentBackground(.hidden)
                            .padding(8).background(JarvisColors.panelStrong)
                            .accessibilityLabel("Task reply or correction")
                        Text("Describe the change. Jarvis will update the task and replace its previous answer.")
                            .font(JarvisTypography.body(12)).foregroundStyle(JarvisColors.mutedWhite)
                        if draft.needsReview(for: task) {
                            if draft.baseTask?.id == task.id {
                                Text("This task changed while you were writing. Review the updated task before sending.")
                                    .font(.callout).foregroundStyle(JarvisColors.primaryText)
                                Button("Use this task version") { draft.review(task) }
                            } else {
                                Text("This draft belongs to another task.").font(.callout)
                                Button("Discard this draft") { draft = JarvisWorkReplyDraft() }
                            }
                        }
                        Button("Send update") {
                            let target = draft.baseTask ?? task, message = draft.text
                            Task { await model.changeTask(target, operation: .update, message: message) }
                        }.disabled(draft.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || draft.needsReview(for: task) || model.pendingAction != nil || model.isLoadingDetail)
                    }
                    if task.canResume {
                        Button("Resume task") { confirmResume = true }.disabled(model.pendingAction != nil || model.isLoadingDetail)
                    }
                    if task.canCancel {
                        Button("Cancel background work", role: .destructive) { confirmCancel = true }.disabled(model.pendingAction != nil || model.isLoadingDetail)
                    }
                    if task.status == "outcome_unknown" {
                        Text("An execution or action needs reconciliation. Do not assume it failed or repeat it blindly.")
                            .foregroundStyle(JarvisColors.amber)
                    }
                }
                Button("Refresh task") { Task { await model.openTask(id) } }
            }.frame(maxWidth: .infinity, alignment: .leading)
                .padding(22).foregroundStyle(JarvisColors.cream)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(JarvisColors.graphite).navigationTitle("Background task")
        .onChange(of: model.lastAcceptedAction) { _, action in if let action { draft.accepted(action) } }
        .confirmationDialog("Cancel the background work, not just its speech?", isPresented: $confirmCancel) {
            Button("Cancel background work", role: .destructive) {
                if let task = model.selectedTask, task.id == id { Task { await model.changeTask(task, operation: .cancel) } }
            }
        }
        .confirmationDialog("Resume this task as a new revision?", isPresented: $confirmResume) {
            Button("Resume task") {
                if let task = model.selectedTask, task.id == id { Task { await model.changeTask(task, operation: .resume) } }
            }
        }
    }
}

private struct WorkInboxDetail<FilePreview: View>: View {
    @ObservedObject var model: JarvisWorkInboxViewModel
    let id: UUID
    let filePreview: (UUID) -> FilePreview
    @State private var confirmDismiss = false
    @State private var selectedArtifact: JarvisTaskArtifact?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                WorkMutationStatus(model: model)
                if model.isLoadingDetail { ProgressView("Checking current source") }
                if let error = model.detailError { Text(error).foregroundStyle(JarvisColors.amber) }
                if let detail = model.selectedInbox, detail.item.id == id {
                    Text(detail.item.sourceLabel).font(JarvisTypography.headline(26))
                    Text(detail.item.stateLabel).font(JarvisTypography.label(15)).foregroundStyle(JarvisColors.amber)
                    if !detail.item.sourceAvailable {
                        Text("This source is no longer available. Its old content is not displayed.")
                        if let reason = detail.item.sourceUnavailableReason { Text(reason).font(JarvisTypography.mono(11)) }
                    } else if let answer = detail.expertResult {
                        if answer.status == "waiting_for_input" { Text("The expert needs your input.").foregroundStyle(JarvisColors.amber) }
                        Text(verbatim: answer.answer).font(JarvisTypography.body(15)).textSelection(.enabled)
                        ForEach(Array(answer.uncertainties.enumerated()), id: \.offset) { _, uncertainty in
                            Text("Uncertainty: " + uncertainty).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.softWhite)
                        }
                        if !detail.artifacts.isEmpty {
                            Text("Saved files").font(JarvisTypography.label(16))
                            ForEach(detail.artifacts) { artifact in
                                Button { selectedArtifact = artifact } label: {
                                    Label(artifact.filename, systemImage: "doc.text")
                                }
                            }
                        }
                    } else if let notification = detail.notification {
                        if let title = notification.title { Text(verbatim: title).font(JarvisTypography.label(17)) }
                        Text(verbatim: notification.message).font(JarvisTypography.body(15)).textSelection(.enabled)
                        if let captured = notification.sourceCapturedAt { Text("Source captured: " + captured).font(JarvisTypography.mono(11)) }
                    } else {
                        Text("No current readable answer was returned.")
                    }
                    if let taskID = detail.item.taskID {
                        NavigationLink("Open task or send a reply", value: WorkDestination.task(taskID))
                    }
                    if detail.canReplay {
                        Button(detail.item.attemptCount == 0 ? "Read aloud" : "Read aloud again") { Task { await model.changeInbox(detail, operation: .replay) } }
                            .disabled(model.pendingAction != nil || model.isLoadingDetail)
                        Text("Reads this answer in a connected voice session.")
                            .font(JarvisTypography.body(12)).foregroundStyle(JarvisColors.mutedWhite)
                    }
                    if detail.item.canDismiss {
                        Button("Dismiss from inbox") { confirmDismiss = true }.disabled(model.pendingAction != nil || model.isLoadingDetail)
                    }
                }
                Button("Refresh answer") { Task { await model.openInbox(id) } }
            }.frame(maxWidth: .infinity, alignment: .leading)
                .padding(22).foregroundStyle(JarvisColors.cream)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(JarvisColors.graphite).navigationTitle("Inbox item")
        .sheet(item: $selectedArtifact) { artifact in
            filePreview(artifact.fileID)
        }
        .onChange(of: model.selectedInbox?.item.sourceAvailable) { _, available in
            if available != true { selectedArtifact = nil }
        }
        .onChange(of: model.selectedInbox?.item.generation) { _, _ in selectedArtifact = nil }
        .confirmationDialog("Dismiss this delivery, without completing a task or reminder?", isPresented: $confirmDismiss) {
            Button("Dismiss from inbox") {
                if let detail = model.selectedInbox, detail.item.id == id { Task { await model.changeInbox(detail, operation: .dismiss) } }
            }
        }
    }
}
