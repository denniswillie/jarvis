import SwiftUI

struct MemoryDetailView: View {
    let memory: JarvisMemoryItem
    @ObservedObject var viewModel: JarvisAppViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

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
                    VStack(alignment: .leading, spacing: 28) {
                        memoryHeader
                        detailContent
                    }
                    .padding(.horizontal, 22)
                    .padding(.vertical, 24)
                }
            }
            .navigationTitle("Memory detail")
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
        .task(id: memory.id) {
            await viewModel.loadMemoryDetail(id: memory.id)
        }
        .onDisappear {
            viewModel.clearMemoryDetail()
        }
    }

    private var displayedMemory: JarvisMemoryItem {
        viewModel.selectedMemoryDetail?.memory ?? memory
    }

    private var hideCachedSource: Bool {
        memory.sourceEvidence != nil && viewModel.selectedMemoryDetail?.memory.id != memory.id
    }

    private var memoryHeader: some View {
        VStack(alignment: .leading, spacing: 10) {
            (dynamicTypeSize.isAccessibilitySize
                ? AnyLayout(VStackLayout(alignment: .leading, spacing: 5))
                : AnyLayout(HStackLayout(spacing: 9))) {
                Text(JarvisMemoryFormatting.typeLabel(displayedMemory.memoryType))
                    .font(JarvisTypography.label(11))
                    .foregroundStyle(JarvisColors.amber)
                Text(displayedMemory.sourceEvidence != nil && displayedMemory.occurredAt == nil
                     ? "Event time unknown"
                     : JarvisMemoryFormatting.fullTimestamp(displayedMemory.occurredAt ?? displayedMemory.createdAt))
                    .font(JarvisTypography.mono(10))
                    .foregroundStyle(JarvisColors.mutedWhite)
            }

            Text(hideCachedSource ? "Source record" : displayedMemory.title)
                .font(JarvisTypography.brand(30))
                .foregroundStyle(JarvisColors.cream)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)

            Text(hideCachedSource ? "Cached source content is hidden until the original is available."
                                  : displayedMemory.displaySummary)
                .font(JarvisTypography.body(16))
                .foregroundStyle(JarvisColors.softWhite)
                .fixedSize(horizontal: false, vertical: true)

            (dynamicTypeSize.isAccessibilitySize
                ? AnyLayout(VStackLayout(alignment: .leading, spacing: 7))
                : AnyLayout(HStackLayout(spacing: 12))) {
                if let confidence = displayedMemory.confidence {
                    Label(
                        "Confidence \(JarvisMemoryFormatting.percentage(confidence))",
                        systemImage: "checkmark.seal"
                    )
                } else {
                    Label("Confidence not assessed", systemImage: "questionmark.circle")
                }
                if displayedMemory.status == "forgotten" {
                    Label("Forgotten", systemImage: "eye.slash")
                } else if displayedMemory.supersededByMemoryId != nil {
                    Label("Superseded", systemImage: "arrow.triangle.branch")
                }
            }
            .font(JarvisTypography.label(10))
            .foregroundStyle(JarvisColors.mutedWhite)
        }
    }

    @ViewBuilder
    private var detailContent: some View {
        switch viewModel.memoryDetailLoadState {
        case .idle, .loading:
            HStack(spacing: 12) {
                ProgressView().tint(JarvisColors.amber)
                Text("Finding supporting context...")
                    .font(JarvisTypography.body(14))
                    .foregroundStyle(JarvisColors.softWhite)
            }
        case .degraded(let message):
            VStack(alignment: .leading, spacing: 8) {
                Text("Supporting context is unavailable")
                    .font(JarvisTypography.headline(18))
                    .foregroundStyle(JarvisColors.cream)
                Text(message)
                    .font(JarvisTypography.body(13))
                    .foregroundStyle(JarvisColors.softWhite)
                Button("Try again") {
                    Task { await viewModel.loadMemoryDetail(id: memory.id) }
                }
                .font(JarvisTypography.label(13))
                .foregroundStyle(JarvisColors.amber)
            }
        case .empty:
            Text("No additional context is available for this memory.")
                .font(JarvisTypography.body(14))
                .foregroundStyle(JarvisColors.softWhite)
        case .loaded:
            if let detail = viewModel.selectedMemoryDetail {
                loadedDetail(detail)
            }
        }
    }

    private func loadedDetail(_ detail: JarvisMemoryDetail) -> some View {
        VStack(alignment: .leading, spacing: 28) {
            if let source = detail.memory.sourceEvidence {
                detailSection(title: "Original source") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Origin: \(JarvisMemoryFormatting.typeLabel(source.trust))")
                        Text(source.sourceTime == nil ? "Event time unknown" :
                            "Event: \(JarvisMemoryFormatting.fullTimestamp(source.sourceTime))")
                        Text("Recorded: \(JarvisMemoryFormatting.fullTimestamp(source.recordedAt))")
                        Text(source.timeBasis)
                        Text("Revision \(source.revision), \(source.isCurrent ? "Current" : "Historical")")
                        Text("Memory processing: \(JarvisMemoryFormatting.typeLabel(source.ingestionStatus))")
                    }
                    .font(JarvisTypography.body(12))
                    .foregroundStyle(JarvisColors.softWhite)
                    Text(detail.memory.content)
                        .font(JarvisTypography.mono(12))
                        .foregroundStyle(JarvisColors.cream)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                    if let page = detail.contentPage {
                        Text("Original JSON: \(detail.memory.content.count) of \(page.totalCharacters) characters")
                            .font(JarvisTypography.label(10))
                            .foregroundStyle(JarvisColors.mutedWhite)
                        if page.nextOffset != nil {
                            Button(viewModel.isLoadingMemoryContent ? "Loading..." : "Read more original content") {
                                Task { await viewModel.loadMoreMemoryContent() }
                            }
                            .disabled(viewModel.isLoadingMemoryContent)
                            .font(JarvisTypography.label(13))
                            .foregroundStyle(JarvisColors.amber)
                        }
                    }
                }
            }
            if detail.memory.status == "forgotten" {
                detailSection(title: "Memory state") {
                    Text("This derived memory has been forgotten. Its former supporting context may no longer be available.")
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.softWhite)
                }
            }

            detailSection(title: "Supporting evidence") {
                if detail.evidence.isEmpty {
                    Text("No supporting evidence is currently available.")
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.softWhite)
                } else {
                    VStack(alignment: .leading, spacing: 17) {
                        ForEach(detail.evidence) { evidence in
                            evidenceRow(evidence)
                        }
                    }
                }
            }

            if !detail.links.isEmpty || !detail.relatedMemories.isEmpty || detail.memory.supersededByMemoryId != nil {
                detailSection(title: "Memory history") {
                    VStack(alignment: .leading, spacing: 14) {
                        ForEach(detail.links) { link in
                            Label(
                                JarvisMemoryFormatting.typeLabel(link.relationship),
                                systemImage: "arrow.triangle.branch"
                            )
                            .font(JarvisTypography.body(13))
                            .foregroundStyle(JarvisColors.softWhite)
                        }
                        ForEach(detail.relatedMemories) { related in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(related.title)
                                    .font(JarvisTypography.label(13))
                                    .foregroundStyle(JarvisColors.cream)
                                Text(related.summary)
                                    .font(JarvisTypography.body(12))
                                    .foregroundStyle(JarvisColors.softWhite)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
            }

            if let uncertainty = detail.uncertainty, !uncertainty.isEmpty {
                detailSection(title: "Uncertainty") {
                    Text(uncertainty)
                        .font(JarvisTypography.body(13))
                        .foregroundStyle(JarvisColors.softWhite)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func evidenceRow(_ evidence: JarvisMemoryEvidence) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: evidence.sourceAvailable ? "checkmark.circle.fill" : "questionmark.circle")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(evidence.sourceAvailable ? JarvisColors.amber : JarvisColors.mutedWhite)
                .padding(.top, 2)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 5) {
                Text(evidence.excerpt.isEmpty ? "Supporting observation" : evidence.excerpt)
                    .font(JarvisTypography.body(14))
                    .foregroundStyle(JarvisColors.cream)
                    .fixedSize(horizontal: false, vertical: true)
                Text(JarvisMemoryFormatting.fullTimestamp(evidence.evidenceTime))
                    .font(JarvisTypography.mono(10))
                    .foregroundStyle(JarvisColors.mutedWhite)
                Text(evidenceAvailability(evidence))
                    .font(JarvisTypography.label(10))
                    .foregroundStyle(evidence.sourceAvailable ? JarvisColors.softWhite : JarvisColors.amber)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func evidenceAvailability(_ evidence: JarvisMemoryEvidence) -> String {
        guard evidence.sourceAvailable else { return "Original source unavailable" }
        if evidence.retainedImageAvailable == true { return "Retained visual evidence available" }
        if evidence.retainedImageAvailable == false { return "Visual image expired; description retained" }
        return "Supporting source available"
    }

    private func detailSection<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(JarvisTypography.label(12))
                .foregroundStyle(JarvisColors.amber)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .padding(.top, 2)
        .overlay(alignment: .top) {
            Divider().overlay(JarvisColors.cream.opacity(0.10))
                .offset(y: -12)
        }
    }
}
