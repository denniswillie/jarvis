import SwiftUI

struct WorkingMemorySummaryView: View {
    @ObservedObject var viewModel: JarvisAppViewModel

    var body: some View {
        TimelineView(.periodic(from: .now, by: 15)) { context in
            VStack(alignment: .leading, spacing: 11) {
                Text("Present moment")
                    .font(JarvisTypography.label(12))
                    .foregroundStyle(JarvisColors.mutedWhite)

                Text(currentPhrase(at: context.date))
                    .font(JarvisTypography.headline(21))
                    .foregroundStyle(JarvisColors.cream)
                    .fixedSize(horizontal: false, vertical: true)
                    .contentTransition(.opacity)

                if let focus = activeFocus(at: context.date) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Image(systemName: "scope")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(JarvisColors.amber)
                            .accessibilityHidden(true)
                        Text(focus)
                            .font(JarvisTypography.body(13))
                            .foregroundStyle(JarvisColors.softWhite)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Current focus: \(focus)")
                }
            }
            .padding(.leading, 17)
            .overlay(alignment: .leading) {
                Capsule()
                    .fill(isFresh(at: context.date) ? JarvisColors.amber : JarvisColors.cream.opacity(0.18))
                    .frame(width: 3)
                    .accessibilityHidden(true)
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func currentPhrase(at now: Date) -> String {
        switch viewModel.workingMemoryLoadState {
        case .idle, .loading where viewModel.workingMemory == nil:
            return "Finding the present moment..."
        case .degraded:
            return "Current context is temporarily unavailable."
        default:
            break
        }

        guard isFresh(at: now), let session = viewModel.workingMemory?.session else {
            return "No recent visual context."
        }
        let scene = session.state.jarvisLabel(for: "scene")
        let activity = session.state.jarvisLabel(for: "activity")
        switch (activity, scene) {
        case let (activity?, scene?): return "\(activity) · \(scene)"
        case let (activity?, nil): return activity
        case let (nil, scene?): return scene
        case (nil, nil): return "No recent visual context."
        }
    }

    private func isFresh(at now: Date) -> Bool {
        guard
            let session = viewModel.workingMemory?.session,
            let expiresAt = JarvisMemoryFormatting.date(from: session.expiresAt)
        else { return false }
        return expiresAt > now
    }

    private func activeFocus(at now: Date) -> String? {
        guard let owner = viewModel.workingMemory?.owner else { return nil }
        if let expiresAt = JarvisMemoryFormatting.date(from: owner.expiresAt), expiresAt <= now {
            return nil
        }
        guard let focus = owner.state["active_visual_focus"] else { return nil }
        if let direct = focus.stringValue { return direct }
        return focus.objectValue?["value"]?.stringValue
            ?? focus.objectValue?["label"]?.stringValue
    }
}
