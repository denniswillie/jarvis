import SwiftUI
import UniformTypeIdentifiers
import CoreTransferable
import AVKit

/// The explicit library action can prepare a verified data-only session without
/// starting voice. AppVM retires the model synchronously on account replacement.
struct JarvisFilesAccessView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    var fileID: UUID? = nil
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Group {
            if let model = viewModel.savedFiles {
                Group {
                    if let fileID {
                        JarvisTaskFilePreview(fileID: fileID, model: model)
                    } else {
                        JarvisFilesView(model: model)
                    }
                }.id(ObjectIdentifier(model))
            } else {
                NavigationStack {
                    VStack(spacing: 18) {
                        if viewModel.isPreparingSavedCopies {
                            ProgressView("Opening saved files")
                        } else {
                            Text(viewModel.savedCopiesPreparationError ?? "Connect online to open saved files.")
                                .font(JarvisTypography.body()).multilineTextAlignment(.center)
                            Button("Try again", action: viewModel.prepareSavedFiles)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity).padding(24)
                    .background(JarvisColors.background)
                    .navigationTitle("Saved files").navigationBarTitleDisplayMode(.inline)
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Close") { dismiss() } } }
                }
            }
        }
        .task { viewModel.prepareSavedFiles() }
        .onDisappear { viewModel.closeSavedFiles() }
    }
}

struct JarvisFilesView: View {
    @StateObject private var model: JarvisFilesViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @State private var showingPreview = false
    @State private var previewLoadTask: Task<Void, Never>?

    init(model: JarvisFilesViewModel) {
        _model = StateObject(wrappedValue: model)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Retained originals")
                        .font(JarvisTypography.headline(26))
                    Text("Your saved images, documents, and videos.")
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.softWhite)
                    HStack {
                        TextField("Search saved files", text: $model.query,
                                  prompt: Text("Search filenames or descriptions")
                                    .foregroundStyle(JarvisColors.mutedWhite))
                            .foregroundStyle(JarvisColors.cream)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .submitLabel(.search)
                            .onSubmit { Task { await model.reload() } }
                            .accessibilityLabel("Search saved files")
                        Button("Search") { Task { await model.reload() } }
                    }
                    .font(JarvisTypography.body(15))
                    if model.isLoading {
                        ProgressView("Loading saved files")
                    } else if let error = model.errorMessage {
                        Text(error).foregroundStyle(JarvisColors.amber)
                        Button("Try again") { Task { await model.reload() } }
                    } else if model.files.isEmpty {
                        Text(model.query.isEmpty ? "Your saved files will appear here." : "Nothing matches this search.")
                            .foregroundStyle(JarvisColors.softWhite)
                    } else {
                        ForEach(model.files) { file in
                            Button {
                                previewLoadTask?.cancel()
                                showingPreview = true
                                previewLoadTask = Task { _ = await model.open(file) }
                            } label: {
                                HStack(alignment: .top, spacing: 14) {
                                    Image(systemName: file.mediaType == "video/mp4" ? "play.rectangle"
                                          : file.mediaType.hasPrefix("image/") ? "photo" : "doc.text")
                                        .foregroundStyle(JarvisColors.amber)
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(file.filename).font(JarvisTypography.label(16))
                                        Text(file.description).font(JarvisTypography.body(13))
                                            .foregroundStyle(JarvisColors.softWhite)
                                        Text(ByteCountFormatter.string(fromByteCount: Int64(file.sizeBytes), countStyle: .file))
                                            .font(JarvisTypography.mono(11))
                                            .foregroundStyle(JarvisColors.mutedWhite)
                                    }
                                    Spacer(minLength: 0)
                                    Image(systemName: "chevron.right").accessibilityHidden(true)
                                }
                                .multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.vertical, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            Divider().overlay(JarvisColors.cream.opacity(0.1))
                        }
                        if model.pageFull {
                            Text("Showing up to eight recent matches. Search for a filename or description to find older files.")
                                .font(JarvisTypography.body(13))
                                .foregroundStyle(JarvisColors.mutedWhite)
                        }
                    }
                    if let message = model.deletionMessage {
                        Text(message).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.softWhite)
                    }
                }
                .foregroundStyle(JarvisColors.cream)
                .padding(22)
            }
            .background(JarvisColors.graphite)
            .navigationTitle("Saved files")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Close") { dismiss() } } }
            .refreshable { await model.reload() }
            .sheet(isPresented: $showingPreview, onDismiss: {
                previewLoadTask?.cancel()
                model.closePreview()
            }) { JarvisFilePreviewView(model: model) }
            .task {
                guard scenePhase == .active else { model.suspend(); return }
                await model.resume()
            }
            .onDisappear { previewLoadTask?.cancel(); model.deactivate() }
            .onChange(of: scenePhase) { _, phase in
                if phase == .background {
                    showingPreview = false
                    previewLoadTask?.cancel()
                    model.suspend()
                } else if phase == .active {
                    Task { await model.resume() }
                }
            }
        }
        .tint(JarvisColors.amber)
    }
}

struct JarvisTaskFilePreview: View {
    @ObservedObject var model: JarvisFilesViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    let fileID: UUID

    init(fileID: UUID, model: JarvisFilesViewModel) {
        self.fileID = fileID
        self.model = model
    }

    var body: some View {
        JarvisFilePreviewView(model: model)
            .task(id: fileID) { await model.open(id: fileID.uuidString.lowercased()) }
            .onDisappear { model.closePreview() }
            .onChange(of: scenePhase) { _, phase in
                if phase == .background { model.suspend(); dismiss() }
            }
    }
}

struct JarvisFilePreviewView: View {
    @ObservedObject var model: JarvisFilesViewModel
    var allowsVideoFullscreen = false
    var onVideoFullscreenChanged: (Bool) -> Void = { _ in }
    @State private var videoFullscreen = false
    @Environment(\.dismiss) private var dismiss
    @State private var confirmingDelete = false
    @State private var exporting = false
    @State private var preparingExport = false
    @State private var document: JarvisRetainedDocument?
    @State private var exportType: UTType = .data
    @State private var exportName = "Jarvis-file"
    @State private var exportMessage: String?
    @State private var operationTask: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let file = model.selectedFile {
                        Text(file.filename).font(JarvisTypography.headline(24)).textSelection(.enabled)
                        if let reference = model.resourceReference(for: file) {
                            Text("ID " + reference.displayID).font(JarvisTypography.mono(12))
                                .foregroundStyle(JarvisColors.mutedWhite).textSelection(.enabled)
                        }
                        Text(file.source == "visual_frame" ? "Saved camera image" : "Generated or retained file")
                            .font(JarvisTypography.label(13)).foregroundStyle(JarvisColors.amber)
                        if let captured = file.memorySource.sourceTime {
                            Text("Source time: " + captured).font(JarvisTypography.mono(11))
                        }
                        if model.isLoadingContent {
                            if file.mediaType == "video/mp4", let media = model.media {
                                JarvisVideoOpeningView(model: media)
                            } else {
                                JarvisOriginalOpeningView(isVideo: false)
                            }
                        } else if let error = model.contentError {
                            Text(error).foregroundStyle(JarvisColors.amber)
                            Button("Try again") {
                                operationTask?.cancel()
                                operationTask = Task { _ = await model.open(file) }
                            }
                        } else if file.mediaType == "video/mp4", let media = model.media {
                            JarvisLibraryMediaView(model: media, allowsVideoFullscreen: allowsVideoFullscreen) { expanded in
                                videoFullscreen = expanded
                                onVideoFullscreenChanged(expanded)
                            }
                        } else if let bytes = model.content {
                            JarvisRetainedContentView(contentType: file.contentType, data: bytes)
                            Button(preparingExport ? "Checking access..." : "Export file") {
                                operationTask?.cancel()
                                operationTask = Task {
                                    exportMessage = nil
                                    preparingExport = true
                                    defer { preparingExport = false }
                                    guard let fresh = await model.open(file), !Task.isCancelled else { return }
                                    document = JarvisRetainedDocument(data: fresh.data)
                                    exportType = UTType(mimeType: fresh.file.mediaType) ?? .data
                                    exportName = fresh.file.exportFilename
                                    exporting = true
                                }
                            }
                            .disabled(preparingExport || model.isDeleting)
                            Text("Export checks access again. Copies you export are not removed by later deletion in Jarvis.")
                                .font(JarvisTypography.body(12)).foregroundStyle(JarvisColors.mutedWhite)
                        }
                        Button("Delete from Jarvis", role: .destructive) { confirmingDelete = true }
                            .disabled(model.isDeleting || preparingExport)
                    } else if model.isLoadingContent {
                        JarvisOriginalOpeningView(isVideo: false)
                    } else if let error = model.contentError {
                        Text(error).foregroundStyle(JarvisColors.amber)
                    } else if let message = model.deletionMessage {
                        Text(message)
                    } else {
                        Text("The original is no longer open.")
                    }
                    if let message = exportMessage { Text(message).font(JarvisTypography.body(13)) }
                }
                .foregroundStyle(JarvisColors.cream)
                .padding(22)
            }
            .background(JarvisColors.graphite)
            .navigationTitle("Original file")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Close") { dismiss() } } }
            .confirmationDialog("Delete this saved original from Jarvis?", isPresented: $confirmingDelete) {
                Button("Delete original", role: .destructive) {
                    guard let file = model.selectedFile else { return }
                    operationTask = Task { if await model.remove(file), !Task.isCancelled { dismiss() } }
                }
            } message: {
                Text("The saved source and its Jarvis storage copy will be removed. Exported copies and independent conversations are separate.")
            }
            .fileExporter(isPresented: $exporting, document: document, contentType: exportType,
                          defaultFilename: exportName) { result in
                document = nil
                switch result {
                case .success: exportMessage = "Exported a separate copy."
                case .failure(let error): exportMessage = "Export was not completed: " + error.localizedDescription
                }
            }
            .onChange(of: model.presentationRevision) { _, _ in clearExport() }
            .onDisappear {
                clearExport()
                if !videoFullscreen { model.closePreview() }
            }
        }
        .tint(JarvisColors.amber)
    }

    private func clearExport() {
        operationTask?.cancel()
        operationTask = nil
        document = nil
        exporting = false
        preparingExport = false
        confirmingDelete = false
        exportMessage = nil
    }
}

private struct JarvisLibraryMediaView: View {
    @ObservedObject var model: JarvisFileMediaPreviewModel
    let allowsVideoFullscreen: Bool
    let onVideoFullscreenChanged: (Bool) -> Void
    @State private var preparing: Task<Void, Never>?
    @State private var isExporting = false
    @State private var exportItem: JarvisMediaExport?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            JarvisLibraryVideoPlayer(model: model.preview, allowsVideoFullscreen: allowsVideoFullscreen,
                onVideoFullscreenChanged: onVideoFullscreenChanged)
            Button(model.isPreparingExport ? "Checking access…" : "Export video") {
                preparing?.cancel()
                preparing = Task {
                    guard await model.prepareExport(), !Task.isCancelled,
                          let item = model.preparedExport else { return }
                    exportItem = item
                    isExporting = true
                }
            }
            .disabled(model.isPreparingExport || model.selected == nil)
            Text("Export saves a separate copy in Files.")
                .font(JarvisTypography.body(12)).foregroundStyle(JarvisColors.mutedWhite)
            if let message = model.exportMessage {
                Text(message).font(JarvisTypography.body(13))
            }
        }
        .fileExporter(isPresented: $isExporting, item: exportItem, contentTypes: [.mpeg4Movie],
                      defaultFilename: exportItem?.filename) { result in
            guard let id = exportItem?.id else { return }
            switch result {
            case .success: model.finishExport(id: id, completed: true)
            case .failure(let error):
                model.finishExport(id: id, completed: false, failureDescription: error.localizedDescription)
            }
            exportItem = nil
        } onCancellation: {
            if let id = exportItem?.id { model.finishExport(id: id, completed: false) }
            exportItem = nil
        }
        .onChange(of: model.preparedExport?.id) { _, id in
            if id == nil { isExporting = false; exportItem = nil }
        }
        .onDisappear {
            preparing?.cancel(); preparing = nil
            isExporting = false; exportItem = nil
            model.cancelExport()
        }
    }
}

private struct JarvisLibraryVideoPlayer: View {
    @ObservedObject var model: JarvisMediaPreviewViewModel
    let allowsVideoFullscreen: Bool
    let onVideoFullscreenChanged: (Bool) -> Void
    var body: some View {
        if let player = model.player {
            Group {
                if allowsVideoFullscreen {
                    JarvisAudiovisualVideoPlayer(player: player, onFullscreenChanged: onVideoFullscreenChanged)
                }
                else { VideoPlayer(player: player) }
            }
                .frame(minHeight: 240, idealHeight: 320, maxHeight: 420)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .accessibilityLabel("Saved file video player")
                .accessibilityIdentifier("libraryVideoPlayer")
        }
    }
}

extension JarvisMediaExport: Transferable {
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .mpeg4Movie, shouldAllowToOpenInPlace: false) { item in
            try Task.checkCancellation()
            return SentTransferredFile(try item.file.urlForReading(), allowAccessingOriginalFile: false)
        }
    }
}

struct JarvisRetainedDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.data] }
    var data: Data
    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws { data = configuration.file.regularFileContents ?? Data() }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper { FileWrapper(regularFileWithContents: data) }
}


private struct JarvisVideoOpeningView: View {
    @ObservedObject var model: JarvisFileMediaPreviewModel
    var body: some View { JarvisOriginalOpeningView(isVideo: true, progress: model.downloadProgress) }
}

struct JarvisOriginalOpeningView: View {
    let isVideo: Bool
    var progress: Double? = nil

    var body: some View {
        VStack(spacing: 20) {
            ZStack {
                Circle().fill(JarvisColors.accent.opacity(0.08)).frame(width: 76, height: 76)
                Image(systemName: isVideo ? "play" : "photo").font(.system(size: 26, weight: .light))
                    .foregroundStyle(JarvisColors.accent)
            }
            VStack(spacing: 8) {
                Text(progress == 1 ? "Preparing your video" : (isVideo ? "Opening your video" : "Opening your image"))
                    .font(JarvisTypography.headline(24))
                Text("Just a moment.").font(JarvisTypography.body(15)).foregroundStyle(JarvisColors.secondaryText)
            }
            if let progress, progress < 1 {
                ProgressView(value: progress).frame(maxWidth: 180)
                Text(progress, format: .percent.precision(.fractionLength(0)))
                    .font(JarvisTypography.mono(12)).foregroundStyle(JarvisColors.secondaryText)
            } else { ProgressView().controlSize(.regular) }
        }
        .frame(maxWidth: .infinity).padding(.vertical, 48)
        .background(JarvisColors.surface, in: RoundedRectangle(cornerRadius: 24))
        .accessibilityIdentifier("originalLoadingView")
    }
}
