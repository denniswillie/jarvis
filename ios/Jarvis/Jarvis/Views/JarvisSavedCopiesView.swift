import SwiftUI
import AVKit

/// Existing exact saved copies, independent of any note-taking feature.
struct JarvisSavedCopiesView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let model = viewModel.workspace {
                    JarvisSavedCopiesContent(model: model, isPreparing: viewModel.isPreparingSavedCopies,
                        error: viewModel.savedCopiesPreparationError, retry: viewModel.prepareSavedCopies)
                } else {
                    JarvisSavedCopiesAccessView(isPreparing: viewModel.isPreparingSavedCopies,
                        error: viewModel.savedCopiesPreparationError, retry: viewModel.prepareSavedCopies)
                }
            }
            .background(JarvisColors.background)
            .navigationTitle("On this phone")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .tint(JarvisColors.accent)
        }
    }
}

/// Observe journal changes directly, independently of account product models.
private struct JarvisSavedCopiesContent: View {
    @ObservedObject var model: JarvisWorkspaceViewModel
    let isPreparing: Bool
    let error: String?
    let retry: () -> Void
    @State private var exportDocument: JarvisRetainedDocument?
    @State private var isExporting = false
    @State private var exportType: UTType = .data
    @State private var exportFilename = "Jarvis-file"
    @State private var exportMessage: String?

    var body: some View {
        Group {
            if isPreparing {
                JarvisSavedCopiesAccessView(isPreparing: true, error: nil, retry: retry)
            } else if model.canSaveOffline {
                List {
                    if model.offlineFiles.isEmpty {
                        Text("Copies you save on this phone will appear here.")
                            .foregroundStyle(JarvisColors.secondaryText)
                    }
                    ForEach(model.offlineFiles) { entry in
                        Button {
                            clearDocumentExport()
                            Task { await model.openOffline(entry) }
                        } label: {
                            Label(entry.filename ?? "Saved copy", systemImage: entry.request.transferPolicy == nil ? "doc.text" : "film")
                        }
                    }
                    if let file = model.offlinePreview {
                        Section("Saved version") {
                            JarvisWorkspaceFileView(file: file)
                            Button("Export to Files") {
                                guard model.canSaveOffline else { return }
                                exportDocument = JarvisRetainedDocument(data: file.data)
                                exportType = UTType(mimeType: file.contentType) ?? .data
                                exportFilename = file.filename
                                exportMessage = nil; isExporting = true
                            }
                            if let exportMessage { Text(exportMessage).font(.footnote) }
                        }
                    }
                    if model.isShowingOfflineMedia {
                        Section("Saved video") {
                            JarvisSavedVideoPreview(model: model.offlineMediaPreview, workspace: model)
                        }
                    }
                    if let error = model.deliveryMessage {
                        Text(error).foregroundStyle(JarvisColors.clay)
                    }
                }
                .scrollContentBackground(.hidden)
            } else {
                JarvisSavedCopiesAccessView(isPreparing: isPreparing, error: error, retry: retry)
            }
        }
        .fileExporter(isPresented: $isExporting, document: exportDocument,
                      contentType: exportType, defaultFilename: exportFilename) { result in
            exportDocument = nil
            guard model.canSaveOffline else { return }
            switch result {
            case .success: exportMessage = "Exported a separate copy."
            case .failure(let error): exportMessage = "Export was not completed: " + error.localizedDescription
            }
        }
        .onChange(of: model.canSaveOffline) { _, permitted in
            if !permitted { clearDocumentExport() }
        }
        .onDisappear { clearDocumentExport(); model.closeOfflinePreview() }
    }

    private func clearDocumentExport() {
        exportDocument = nil; isExporting = false; exportMessage = nil
    }
}

private struct JarvisSavedVideoPreview: View {
    @ObservedObject var model: JarvisMediaPreviewViewModel
    @ObservedObject var workspace: JarvisWorkspaceViewModel
    @State private var exportItem: JarvisMediaExport?
    @State private var isExporting = false
    @State private var preparing: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let filename = model.filename {
                Text(filename).font(JarvisTypography.headline(18)).foregroundStyle(JarvisColors.primaryText)
            }
            if let player = model.player {
                VideoPlayer(player: player)
                    .frame(minHeight: 240, idealHeight: 320, maxHeight: 420)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .accessibilityLabel("Saved video player")
                    .accessibilityIdentifier("savedVideoPlayer")
                Button(workspace.isPreparingOfflineExport ? "Preparing export…" : "Export to Files") {
                    preparing?.cancel()
                    preparing = Task {
                        guard await workspace.prepareOfflineExport(), !Task.isCancelled,
                              let item = workspace.offlineMediaExport else { return }
                        exportItem = item; isExporting = true
                    }
                }
                .disabled(workspace.isPreparingOfflineExport)
                .accessibilityIdentifier("savedVideoExportButton")
            } else if let error = model.errorMessage {
                Text(error).font(.callout).foregroundStyle(JarvisColors.clay)
            } else {
                ProgressView("Opening video…").frame(maxWidth: .infinity, minHeight: 120)
            }
            if let message = workspace.offlineExportMessage { Text(message).font(.footnote) }
        }
        .padding(.vertical, 8)
        .fileExporter(isPresented: $isExporting, item: exportItem, contentTypes: [.mpeg4Movie],
                      defaultFilename: exportItem?.filename) { result in
            guard let id = exportItem?.id else { return }
            switch result {
            case .success: workspace.finishOfflineExport(id: id, completed: true)
            case .failure(let error):
                workspace.finishOfflineExport(id: id, completed: false, failureDescription: error.localizedDescription)
            }
            exportItem = nil
        } onCancellation: {
            if let id = exportItem?.id { workspace.finishOfflineExport(id: id, completed: false) }
            exportItem = nil
        }
        .onChange(of: workspace.offlineMediaExport?.id) { _, id in
            if id == nil { isExporting = false; exportItem = nil }
        }
        .onDisappear {
            preparing?.cancel(); preparing = nil
            isExporting = false; exportItem = nil; workspace.cancelOfflineExport()
        }
    }
}

private struct JarvisSavedCopiesAccessView: View {
    let isPreparing: Bool
    let error: String?
    let retry: () -> Void

    var body: some View {
        if isPreparing {
            ProgressView("Opening saved copies")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView {
                Label("Saved copies are locked", systemImage: "lock")
            } description: {
                Text(error ?? "Verify this account online to open copies saved on this phone.")
            } actions: {
                Button("Try again", action: retry)
            }
        }
    }
}
