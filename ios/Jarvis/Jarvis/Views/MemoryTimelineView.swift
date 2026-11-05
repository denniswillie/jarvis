import SwiftUI

struct MemoryTimelineView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var selectedMemory: JarvisMemoryItem?

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [JarvisColors.graphite, JarvisColors.ink, JarvisColors.moss],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 28) {
                        intro
                        timelineContent
                    }
                    .padding(.horizontal, 22)
                    .padding(.vertical, 24)
                }
                .refreshable {
                    await viewModel.reloadMemoryState()
                }
            }
            .navigationTitle("Memory")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(JarvisColors.graphite.opacity(0.94), for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .font(JarvisTypography.label(14))
                        .foregroundStyle(JarvisColors.cream)
                }
            }
        }
        .sheet(item: $selectedMemory) { memory in
            MemoryDetailView(memory: memory, viewModel: viewModel)
        }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(viewModel.sourceBackedMemory ? "Original evidence" : "A trace of what mattered")
                .font(JarvisTypography.brand(31))
                .foregroundStyle(JarvisColors.cream)
                .accessibilityAddTraits(.isHeader)
            Text(viewModel.sourceBackedMemory
                ? "Retained reports, observations and model output. Their origin is shown; storing a statement does not verify it."
                : "Read-only memories grounded in moments Jarvis observed or you explicitly asked it to remember.")
                .font(JarvisTypography.body(14))
                .foregroundStyle(JarvisColors.softWhite)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var timelineContent: some View {
        if viewModel.recentMemories.isEmpty {
            emptyState
        } else {
            if case .degraded = viewModel.memoryLoadState {
                degradedBanner
            }

            ForEach(dayGroups) { group in
                VStack(alignment: .leading, spacing: 0) {
                    Text(group.title)
                        .font(JarvisTypography.label(12))
                        .foregroundStyle(JarvisColors.amber)
                        .padding(.bottom, 8)
                        .accessibilityAddTraits(.isHeader)

                    ForEach(group.items) { memory in
                        Button {
                            selectedMemory = memory
                        } label: {
                            MemoryTimelineRow(memory: memory)
                        }
                        .buttonStyle(.plain)

                        if memory.id != group.items.last?.id {
                            Divider().overlay(JarvisColors.cream.opacity(0.10))
                        }
                    }
                }
            }

        }
        if viewModel.memoryNextCursor != nil {
            Button {
                Task { await viewModel.loadMoreMemories() }
            } label: {
                HStack(spacing: 10) {
                    if viewModel.isLoadingMoreMemories {
                        ProgressView().tint(JarvisColors.graphite)
                    }
                    Text(viewModel.isLoadingMoreMemories ? "Loading earlier memories" : "Load earlier memories")
                        .font(JarvisTypography.label(14))
                }
                .foregroundStyle(JarvisColors.graphite)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Capsule().fill(JarvisColors.cream))
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isLoadingMoreMemories)
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        switch viewModel.memoryLoadState {
        case .idle, .loading:
            HStack(spacing: 12) {
                ProgressView().tint(JarvisColors.amber)
                Text("Gathering your memory timeline...")
                    .font(JarvisTypography.body(15))
                    .foregroundStyle(JarvisColors.softWhite)
            }
        case .degraded:
            degradedBanner
        case .empty, .loaded:
            VStack(alignment: .leading, spacing: 8) {
                Text("Nothing here yet")
                    .font(JarvisTypography.headline(20))
                    .foregroundStyle(JarvisColors.cream)
                Text(viewModel.sourceBackedMemory
                    ? "No readable source records on this page. This is not proof that an event did not happen."
                    : "Meaningful episodes and things you ask Jarvis to remember will appear here.")
                    .font(JarvisTypography.body(14))
                    .foregroundStyle(JarvisColors.softWhite)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var degradedBanner: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "wifi.exclamationmark")
                .foregroundStyle(JarvisColors.amber)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 5) {
                Text("Memory sync is delayed")
                    .font(JarvisTypography.label(14))
                    .foregroundStyle(JarvisColors.cream)
                Text("Pull down or try again when the connection returns.")
                    .font(JarvisTypography.body(13))
                    .foregroundStyle(JarvisColors.softWhite)
                Button("Try again") { viewModel.refreshMemoryState() }
                    .font(JarvisTypography.label(12))
                    .foregroundStyle(JarvisColors.amber)
            }
        }
        .padding(.vertical, 6)
    }

    private var dayGroups: [MemoryDayGroup] {
        var groups: [MemoryDayGroup] = []
        let calendar = Calendar.autoupdatingCurrent
        for memory in viewModel.recentMemories {
            let date = JarvisMemoryFormatting.date(for: memory)
            let day = date.map(calendar.startOfDay(for:))
            let key = day?.timeIntervalSince1970.description ?? "unknown"
            if groups.last?.id == key {
                groups[groups.count - 1].items.append(memory)
            } else {
                groups.append(
                    MemoryDayGroup(
                        id: key,
                        title: day.map(dayTitle) ?? "Earlier",
                        items: [memory]
                    )
                )
            }
        }
        return groups
    }

    private func dayTitle(_ day: Date) -> String {
        let calendar = Calendar.autoupdatingCurrent
        if calendar.isDateInToday(day) { return "Today" }
        if calendar.isDateInYesterday(day) { return "Yesterday" }
        return day.formatted(.dateTime.weekday(.wide).month(.abbreviated).day())
    }
}

private struct MemoryDayGroup: Identifiable {
    let id: String
    let title: String
    var items: [JarvisMemoryItem]
}

private struct MemoryTimelineRow: View {
    let memory: JarvisMemoryItem
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        HStack(alignment: .top, spacing: 15) {
            VStack(spacing: 5) {
                Circle()
                    .fill(memory.status == "forgotten" ? JarvisColors.mutedWhite : JarvisColors.amber)
                    .frame(width: 9, height: 9)
                Rectangle()
                    .fill(JarvisColors.cream.opacity(0.12))
                    .frame(width: 1, height: 54)
            }
            .padding(.top, 7)
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 6) {
                (dynamicTypeSize.isAccessibilitySize
                    ? AnyLayout(VStackLayout(alignment: .leading, spacing: 5))
                    : AnyLayout(HStackLayout(alignment: .firstTextBaseline))) {
                    Text(memory.title)
                        .font(JarvisTypography.label(16))
                        .foregroundStyle(JarvisColors.cream)
                        .multilineTextAlignment(.leading)
                    if !dynamicTypeSize.isAccessibilitySize {
                        Spacer(minLength: 8)
                    }
                    Text(JarvisMemoryFormatting.relativeTime(for: memory))
                        .font(JarvisTypography.mono(9))
                        .foregroundStyle(JarvisColors.mutedWhite)
                }
                Text(memory.displaySummary)
                    .font(JarvisTypography.body(14))
                    .foregroundStyle(JarvisColors.softWhite)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)
                Text(JarvisMemoryFormatting.typeLabel(memory.memoryType))
                    .font(JarvisTypography.label(10))
                    .foregroundStyle(JarvisColors.amber)
            }
            .padding(.bottom, 15)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityHint("Opens memory details and supporting evidence")
    }
}
