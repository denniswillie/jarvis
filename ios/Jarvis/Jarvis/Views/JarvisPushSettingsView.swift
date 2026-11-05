import SwiftUI
import UIKit

struct JarvisPushSettingsView: View {
    @ObservedObject var model: JarvisPushViewModel

    var body: some View {
        JarvisPanel {
            Text("Phone notifications").font(JarvisTypography.label(14)).foregroundStyle(JarvisColors.cream)
            Text("Private inbox links when voice is unavailable. A notification never means you completed a reminder.")
                .font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.mutedWhite)
            Text(model.preferences.wantsEnabled ? "Requested: on" : "Requested: off")
                .font(JarvisTypography.body(14)).foregroundStyle(JarvisColors.cream)
            Text("iOS permission: " + model.permission.rawValue)
                .font(JarvisTypography.mono(11)).foregroundStyle(JarvisColors.softWhite)
            if let provider = model.provider {
                Text("Backend: " + provider.state + " / provider access: " + provider.providerAccess)
                    .font(JarvisTypography.mono(11)).foregroundStyle(JarvisColors.softWhite)
            }
            if let quiet = model.device?.quietUntil {
                Text("Server quiet until: " + quiet).font(JarvisTypography.mono(11)).foregroundStyle(JarvisColors.softWhite)
            }
            if let reason = model.device?.invalidReason {
                Text("Registration needs attention: " + reason).font(JarvisTypography.mono(11)).foregroundStyle(JarvisColors.amber)
            }
            if let message = model.message {
                Text(message).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.amber)
            }
            if model.isBusy { ProgressView("Updating notification state") }
            if model.preferences.wantsEnabled {
                Button("Turn off phone notifications") { Task { await model.disable() } }.disabled(model.isBusy)
            } else {
                Button("Enable phone notifications") { Task { await model.enable() } }
                    .disabled(model.isBusy || model.pending != nil)
            }
            if model.pending != nil {
                Button("Retry the same registration request") { Task { await model.retry() } }.disabled(model.isBusy)
            }
            HStack {
                Button("Quiet for one hour") { Task { await model.setQuiet(until: Date().addingTimeInterval(3600)) } }
                Button("Clear quiet time") { Task { await model.setQuiet(until: nil) } }
            }.disabled(!model.canSetQuietTime)
            Button("Refresh notification status") { Task { await model.refresh() } }.disabled(model.isBusy)
            Button("Open iOS notification settings") {
                if let url = URL(string: UIApplication.openNotificationSettingsURLString) { UIApplication.shared.open(url) }
            }
            if model.requiresDisconnectCleanup {
                Text("Turn notifications off and resolve any pending registration before replacing or resetting backend credentials.")
                    .font(JarvisTypography.body(12)).foregroundStyle(JarvisColors.mutedWhite)
            }
        }.task { await model.refresh() }
    }
}
