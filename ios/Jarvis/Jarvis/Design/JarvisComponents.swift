import SwiftUI

struct JarvisAtmosphericBackground: View {
    let state: JarvisAppState
    var body: some View {
        JarvisColors.background.ignoresSafeArea()
    }
}

struct JarvisOrb: View {
    let state: JarvisAppState
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false

    private var isEngaged: Bool {
        switch state {
        case .active, .connecting: return true
        default: return false
        }
    }

    private var scale: CGFloat {
        guard !reduceMotion else { return 1.0 }
        switch state {
        case .active: return pulse ? 1.06 : 0.96
        case .connecting: return pulse ? 1.03 : 0.98
        default: return pulse ? 1.01 : 0.99
        }
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(isEngaged ? JarvisColors.softAccent.opacity(0.75) : JarvisColors.panelStrong.opacity(0.65))
                .frame(width: 96, height: 96)
                .overlay(
                    Circle()
                        .strokeBorder(
                            isEngaged ? JarvisColors.accent.opacity(0.28) : JarvisColors.border,
                            lineWidth: 1
                        )
                )
                .scaleEffect(scale)

            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            JarvisColors.surface,
                            isEngaged ? JarvisColors.softAccent : JarvisColors.surface
                        ],
                        center: .topLeading,
                        startRadius: 4,
                        endRadius: 68
                    )
                )
                .frame(width: 72, height: 72)
                .overlay(
                    Circle()
                        .strokeBorder(JarvisColors.border, lineWidth: 1)
                )
                .shadow(color: JarvisColors.primaryText.opacity(0.05), radius: 10, x: 0, y: 4)

            Image(systemName: "waveform")
                .font(.system(size: 22, weight: .medium))
                .foregroundStyle(isEngaged ? JarvisColors.accent : JarvisColors.secondaryText)
        }
        .task(id: reduceMotion) {
            guard !reduceMotion else {
                pulse = false
                return
            }
            withAnimation(JarvisMotion.gentlePulse) {
                pulse = true
            }
        }
        .accessibilityHidden(true)
    }
}

struct JarvisStatePill: View {
    let text: String
    var isActive: Bool = false
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        let shape = RoundedRectangle(
            cornerRadius: dynamicTypeSize.isAccessibilitySize ? 16 : 100,
            style: .continuous
        )
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(isActive ? JarvisColors.surface : JarvisColors.secondaryText)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: dynamicTypeSize.isAccessibilitySize ? .infinity : nil)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(shape.fill(isActive ? JarvisColors.accent : JarvisColors.panelStrong))
            .overlay(
                shape.strokeBorder(isActive ? Color.clear : JarvisColors.border, lineWidth: 1)
            )
    }
}

struct JarvisPrimaryButton: View {
    let title: String
    let isLoading: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if isLoading {
                    ProgressView()
                        .tint(JarvisColors.surface)
                }
                Text(title)
                    .font(.body.weight(.semibold))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(JarvisColors.surface)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 20)
            .padding(.vertical, 15)
            .background(
                Capsule()
                    .fill(JarvisColors.accent)
                    .shadow(color: JarvisColors.accent.opacity(0.14), radius: 10, x: 0, y: 4)
            )
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
    }
}

struct JarvisSecondaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(JarvisColors.primaryText)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 18)
                .padding(.vertical, 13)
                .background(
                    Capsule()
                        .fill(JarvisColors.surface)
                        .overlay(
                            Capsule()
                                .strokeBorder(JarvisColors.border, lineWidth: 1)
                        )
                )
        }
        .buttonStyle(.plain)
    }
}

struct JarvisIconButton: View {
    let systemName: String
    let accessibilityLabel: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(JarvisColors.primaryText)
                .frame(width: 44, height: 44)
                .background(
                    Circle()
                        .fill(JarvisColors.surface)
                        .overlay(
                            Circle()
                                .strokeBorder(JarvisColors.border, lineWidth: 1)
                        )
                )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }
}

struct JarvisInputField: View {
    let title: String
    let placeholder: String
    @Binding var text: String
    var isSecure = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(JarvisTypography.label(12))
                .foregroundStyle(JarvisColors.mutedWhite)

            Group {
                if isSecure {
                    SecureField(placeholder, text: $text)
                        .textContentType(.oneTimeCode)
                } else {
                    TextField(placeholder, text: $text)
                }
            }
            .font(JarvisTypography.body(15))
            .foregroundStyle(JarvisColors.cream)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(JarvisColors.surface)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(JarvisColors.border, lineWidth: 1)
                    )
            )
        }
    }
}

struct JarvisPanel<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            content
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(JarvisColors.panel)
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .strokeBorder(JarvisColors.border, lineWidth: 1)
                )
        )
    }
}

enum JarvisMemoryFormatting {
    static func date(from value: String?) -> Date? {
        guard let value else { return nil }
        if let date = try? Date.ISO8601FormatStyle(includingFractionalSeconds: true).parse(value) {
            return date
        }
        return try? Date.ISO8601FormatStyle().parse(value)
    }

    static func date(for memory: JarvisMemoryItem) -> Date? {
        date(from: memory.occurredAt) ?? date(from: memory.createdAt)
    }

    static func relativeTime(for memory: JarvisMemoryItem, relativeTo now: Date = Date()) -> String {
        guard let date = date(for: memory) else { return "Time unavailable" }
        let relative = date.formatted(.relative(presentation: .named, unitsStyle: .wide))
        return memory.sourceEvidence != nil && memory.occurredAt == nil ? "Recorded \(relative)" : relative
    }

    static func fullTimestamp(_ value: String?) -> String {
        guard let date = date(from: value) else { return "Time unavailable" }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    static func typeLabel(_ value: String) -> String {
        value.replacingOccurrences(of: "_", with: " ").localizedCapitalized
    }

    static func percentage(_ value: Double) -> String {
        "\(Int((min(max(value, 0), 1) * 100).rounded()))%"
    }
}
