import SwiftUI

struct MemoryRailView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var isShowingTimeline = false
    @State private var isShowingFiles = false
    @State private var filePresentationID = UUID()
    @State private var selectedMemory: JarvisMemoryItem?

    var body: some View {
        JarvisPanel {
            (dynamicTypeSize.isAccessibilitySize
                ? AnyLayout(VStackLayout(alignment: .leading, spacing: 12))
                : AnyLayout(HStackLayout(alignment: .firstTextBaseline))) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Memory")
                        .font(JarvisTypography.label(13))
                        .foregroundStyle(JarvisColors.mutedWhite)

                    Text(viewModel.sourceBackedMemory ? "Recent source records" : "What stayed with me")
                        .font(JarvisTypography.headline(22))
                        .foregroundStyle(JarvisColors.cream)
                }

                if !dynamicTypeSize.isAccessibilitySize {
                    Spacer()
                }

                JarvisIconButton(
                    systemName: "clock.arrow.circlepath",
                    accessibilityLabel: "Open memory timeline"
                ) {
                    isShowingTimeline = true
                }
            }

            memoryContent

            Button {
                isShowingTimeline = true
            } label: {
                HStack {
                    Text("Open memory timeline")
                        .font(JarvisTypography.label(14))
                    Spacer()
                    Image(systemName: "arrow.up.right")
                        .accessibilityHidden(true)
                }
                .foregroundStyle(JarvisColors.cream)
                .padding(.top, 4)
            }
            .buttonStyle(.plain)
            .accessibilityHint("Shows all recent memories in chronological order")

            if viewModel.sourceBackedMemory {
                Button {
                    filePresentationID = UUID()
                    isShowingFiles = true
                } label: {
                    Label("Saved files", systemImage: "doc")
                        .font(JarvisTypography.label(14))
                        .foregroundStyle(JarvisColors.cream)
                }
                .buttonStyle(.plain)
                .accessibilityHint("Browse, preview, export or delete retained originals")
            }
        }
        .sheet(isPresented: $isShowingFiles) {
            if viewModel.configuration != nil {
                JarvisFilesAccessView(viewModel: viewModel)
                    .id(filePresentationID)
            }
        }
        .onChange(of: viewModel.configuration) { _, _ in isShowingFiles = false }
        .onChange(of: viewModel.sourceBackedMemory) { _, selected in
            if !selected { isShowingFiles = false }
        }
        .sheet(isPresented: $isShowingTimeline) {
            MemoryTimelineView(viewModel: viewModel)
        }
        .sheet(item: $selectedMemory) { memory in
            MemoryDetailView(memory: memory, viewModel: viewModel)
        }
    }

    @ViewBuilder
    private var memoryContent: some View {
        if viewModel.recentMemoryHighlights.isEmpty {
            switch viewModel.memoryLoadState {
            case .idle, .loading:
                HStack(spacing: 10) {
                    ProgressView()
                        .tint(JarvisColors.amber)
                    Text("Gathering recent moments...")
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.softWhite)
                }
            case .empty, .loaded:
                Text(viewModel.sourceBackedMemory
                    ? "No readable source records on this page. This does not prove nothing was recorded."
                    : "No episodes yet. Jarvis will keep meaningful moments here as your day unfolds.")
                    .font(JarvisTypography.body(14))
                    .foregroundStyle(JarvisColors.softWhite)
                    .fixedSize(horizontal: false, vertical: true)
            case .degraded:
                degradedMessage
            }
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(viewModel.recentMemoryHighlights.prefix(3).enumerated()), id: \.element.id) { index, memory in
                    if index > 0 {
                        Divider().overlay(JarvisColors.cream.opacity(0.10))
                    }
                    Button {
                        selectedMemory = memory
                    } label: {
                        memoryRow(memory)
                    }
                    .buttonStyle(.plain)
                }
            }

            if case .degraded = viewModel.memoryLoadState {
                degradedMessage
            }
        }
    }

    private var degradedMessage: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "wifi.exclamationmark")
                .foregroundStyle(JarvisColors.amber)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                Text("Memory is temporarily unavailable")
                    .font(JarvisTypography.label(13))
                    .foregroundStyle(JarvisColors.cream)
                Button("Try again") {
                    viewModel.refreshMemoryState()
                }
                .font(JarvisTypography.label(12))
                .foregroundStyle(JarvisColors.amber)
            }
        }
    }

    private func memoryRow(_ memory: JarvisMemoryItem) -> some View {
        HStack(alignment: .top, spacing: 13) {
            Circle()
                .fill(JarvisColors.amber.opacity(0.85))
                .frame(width: 7, height: 7)
                .padding(.top, 8)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 5) {
                Text(memory.title)
                    .font(JarvisTypography.label(15))
                    .foregroundStyle(JarvisColors.cream)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Text(memory.displaySummary)
                    .font(JarvisTypography.body(13))
                    .foregroundStyle(JarvisColors.softWhite)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                Text(JarvisMemoryFormatting.relativeTime(for: memory))
                    .font(JarvisTypography.mono(10))
                    .foregroundStyle(JarvisColors.mutedWhite)
            }

            Spacer(minLength: 8)

            Image(systemName: "chevron.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(JarvisColors.mutedWhite)
                .padding(.top, 7)
                .accessibilityHidden(true)
        }
        .padding(.vertical, 13)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityHint("Opens supporting evidence and memory history")
    }
}
