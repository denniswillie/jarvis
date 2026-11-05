import SwiftUI

struct ReminderInboxView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @ObservedObject var audioSessionViewModel: JarvisAudioSessionViewModel
    var showsAllItems = false

    var body: some View {
        JarvisPanel {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Reminders")
                        .font(JarvisTypography.label(13))
                        .foregroundStyle(JarvisColors.mutedWhite)
                    Text(summaryText)
                        .font(JarvisTypography.headline(22))
                        .foregroundStyle(JarvisColors.cream)
                }

                Spacer()

                Button {
                    viewModel.refreshReminderState()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(JarvisColors.cream)
                        .frame(width: 38, height: 38)
                        .background(Circle().fill(JarvisColors.panelStrong))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Refresh reminders")
            }

            if let speakingMessage = audioSessionViewModel.speakingReminderMessage {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "waveform")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(JarvisColors.graphite)
                        .frame(width: 30, height: 30)
                        .background(Circle().fill(JarvisColors.amber))

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Speaking now")
                            .font(JarvisTypography.label(12))
                            .foregroundStyle(JarvisColors.amber)
                        Text(speakingMessage)
                            .font(JarvisTypography.body(14))
                            .foregroundStyle(JarvisColors.cream)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            } else if isAwaitingFirstLoad {
                ProgressView("Loading reminders")
                    .font(.subheadline)
                    .tint(JarvisColors.accent)
            } else if viewModel.activeReminders.isEmpty, viewModel.reminderEvents.isEmpty,
                      viewModel.reminderRefreshError == nil {
                Text("Nothing is waiting. Ask Jarvis to remind you whenever something matters.")
                    .font(JarvisTypography.body(14))
                    .foregroundStyle(JarvisColors.softWhite)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                VStack(alignment: .leading, spacing: 15) {
                    ForEach(showsAllItems ? viewModel.activeReminders : Array(viewModel.activeReminders.prefix(3))) { reminder in
                        reminderRow(reminder)
                    }

                    ForEach(showsAllItems ? viewModel.reminderEvents : Array(viewModel.reminderEvents.prefix(3))) { event in
                        eventRow(event)
                    }
                }
            }

            if let error = viewModel.reminderRefreshError {
                Text(error)
                    .font(JarvisTypography.body(12))
                    .foregroundStyle(JarvisColors.clay)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var summaryText: String {
        if viewModel.reminderRefreshError != nil,
           viewModel.activeReminders.isEmpty, viewModel.reminderEvents.isEmpty { return "Unable to load" }
        if isAwaitingFirstLoad { return "Loading" }
        let count = viewModel.activeReminders.count
        if count == 0 { return "All clear" }
        return count == 1 ? "1 active" : "\(count) active"
    }

    private var isAwaitingFirstLoad: Bool {
        viewModel.reminderLastRefreshedAt == nil && viewModel.reminderRefreshError == nil &&
        viewModel.activeReminders.isEmpty && viewModel.reminderEvents.isEmpty
    }

    private func reminderRow(_ reminder: JarvisReminderSummary) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: reminder.condition == nil ? "clock" : "eye")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(JarvisColors.amber)
                .frame(width: 30, height: 30)
                .background(Circle().fill(JarvisColors.amber.opacity(0.14)))

            VStack(alignment: .leading, spacing: 4) {
                Text(reminder.title)
                    .font(JarvisTypography.label(15))
                    .foregroundStyle(JarvisColors.cream)
                if !reminder.details.isEmpty {
                    Text(reminder.details)
                        .font(JarvisTypography.body(13))
                        .foregroundStyle(JarvisColors.softWhite)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text(reminder.timingLabel)
                    .font(JarvisTypography.mono(10))
                    .foregroundStyle(JarvisColors.mutedWhite)
            }
        }
    }

    private func eventRow(_ event: JarvisReminderEvent) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: event.eventType == "delivered_to_session" ? "waveform" : "tray.full")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(JarvisColors.cream)
                .frame(width: 30, height: 30)
                .background(Circle().fill(JarvisColors.cream.opacity(0.10)))

            VStack(alignment: .leading, spacing: 4) {
                Text(event.statusLabel)
                    .font(JarvisTypography.label(12))
                    .foregroundStyle(JarvisColors.mutedWhite)
                if !event.message.isEmpty {
                    Text(event.message)
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.cream)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text(event.createdLabel)
                    .font(JarvisTypography.mono(10))
                    .foregroundStyle(JarvisColors.mutedWhite)
            }
        }
    }
}
