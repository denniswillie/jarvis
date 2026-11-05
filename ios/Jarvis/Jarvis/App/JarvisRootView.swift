import SwiftUI

struct JarvisRootView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            JarvisAtmosphericBackground(state: viewModel.state)

            rootContent
                .transition(.opacity.combined(with: .move(edge: .bottom)))
        }
        .animation(JarvisMotion.stateTransition, value: viewModel.state)
        .preferredColorScheme(.light)
        .background(JarvisAppLinkHost(presenter: viewModel.appLinkPresenter).frame(width: 0, height: 0))
        .task {
            await viewModel.bootstrap()
        }
        .sheet(item: $viewModel.presentedSheet, onDismiss: viewModel.sheetDidDismiss) { sheet in
            switch sheet {
            case .settings:
                SettingsView(
                    viewModel: viewModel,
                    audioSessionViewModel: viewModel.audioSessionViewModel
                )
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
            case .preferences:
                if let model = viewModel.preferences {
                    JarvisPreferencesView(model: model) { viewModel.presentedSheet = .settings }
                        .id(ObjectIdentifier(model))
                }
            case .reminders:
                NavigationStack {
                    ScrollView {
                        ReminderInboxView(viewModel: viewModel,
                                          audioSessionViewModel: viewModel.audioSessionViewModel,
                                          showsAllItems: true)
                            .padding(20)
                    }
                    .background(JarvisColors.background)
                    .navigationTitle("Reminders")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { viewModel.presentedSheet = nil }
                        }
                    }
                    .task { viewModel.refreshReminderState() }
                    .refreshable { viewModel.refreshReminderState() }
                }
                .tint(JarvisColors.accent)
            case .work:
                if let model = viewModel.workInbox {
                    JarvisWorkInboxView(model: model, notificationLink: viewModel.notificationInboxLink,
                                        productLink: viewModel.workProductLink) { fileID in
                        JarvisFilesAccessView(viewModel: viewModel, fileID: fileID)
                            .id(viewModel.productConfigurationGeneration)
                    }
                }
            case .files:
                if viewModel.configuration != nil {
                    JarvisFilesAccessView(viewModel: viewModel).id(viewModel.productConfigurationGeneration)
                }
            case .offlineCopies:
                JarvisSavedCopiesView(viewModel: viewModel)
            }
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .active {
                viewModel.refreshForegroundState()
            } else if newPhase == .background {
                viewModel.appLinkPresenter.dismiss()
            }
        }
    }

    @ViewBuilder
    private var rootContent: some View {
        switch viewModel.state {
        case .launching:
            LaunchView(statusText: "Waking Jarvis")
        case .needsSetup:
            SetupTokenView(viewModel: viewModel)
        case .ready, .connecting, .active, .error:
            JarvisProductShellView(viewModel: viewModel, audio: viewModel.audioSessionViewModel)
        }
    }
}

/// This is the single owner of product visibility. Child tabs never start or
/// stop voice and never infer visibility from a scene notification alone.
private struct JarvisProductShellView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @ObservedObject var audio: JarvisAudioSessionViewModel
    @Environment(\.scenePhase) private var scenePhase

    private var sessionID: String? { audio.activeSession?.id ?? viewModel.productFallbackSessionID }

    var body: some View {
        TabView(selection: $viewModel.productTab) {
            AmbientCompanionView(viewModel: viewModel, audioSessionViewModel: audio)
                .tabItem { Label("Home", systemImage: "house") }.tag(JarvisProductTab.home)
            NavigationStack {
                Group {
                    if let model = viewModel.workspace {
                        JarvisWorkspaceView(model: model, openTask: viewModel.showWorkspaceTask,
                                            openInbox: viewModel.showWorkspaceInboxItem,
                                            isActive: viewModel.productTab == .workspace && scenePhase == .active && viewModel.presentedSheet == nil)
                    } else {
                        JarvisSandboxEmptyWorkspace(open: viewModel.openExistingSandbox)
                    }
                }
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("Files", action: viewModel.showSavedFiles)
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button { viewModel.showWorkInbox() } label: { Label("Tasks and inbox", systemImage: "tray") }
                            .disabled(viewModel.workInbox == nil)
                    }
                }
            }
            .tabItem { Label("Workspace", systemImage: "rectangle.on.rectangle") }.tag(JarvisProductTab.workspace)
            Group {
                if let model = viewModel.audiovisual {
                    JarvisAudiovisualView(model: model,
                        isActive: viewModel.productTab == .audiovisual && scenePhase == .active && viewModel.presentedSheet == nil,
                        openSavedCopies: viewModel.showSavedCopies,
                        transferMessage: viewModel.audiovisualTransferMessage)
                        .id(ObjectIdentifier(model))
                } else {
                    NavigationStack {
                        VStack(spacing: 18) {
                            if viewModel.isPreparingSavedCopies {
                                ProgressView("Opening your library")
                            } else {
                                JarvisProductUnavailableView(title: "Audiovisual", symbol: "photo.on.rectangle",
                                    message: viewModel.savedCopiesPreparationError ?? "Your saved images and videos appear here.")
                                Button("Open library") { viewModel.prepareAudiovisual() }
                                Button("On this phone", action: viewModel.showSavedCopies)
                            }
                        }.navigationTitle("Audiovisual")
                    }
                }
            }
            .task(id: viewModel.productTab == .audiovisual && viewModel.audiovisual == nil && scenePhase == .active) {
                if viewModel.productTab == .audiovisual && viewModel.audiovisual == nil && scenePhase == .active {
                    viewModel.prepareAudiovisual()
                }
            }
            .tabItem { Label("Audiovisual", systemImage: "photo.on.rectangle") }.tag(JarvisProductTab.audiovisual)
            Group {
                if let model = viewModel.plugins {
                    JarvisPluginsAccountView(model: model,
                        isActive: viewModel.productTab == .plugins && scenePhase == .active && viewModel.presentedSheet == nil)
                        .id(ObjectIdentifier(model))
                } else {
                    NavigationStack {
                        JarvisProductUnavailableView(title: "Plugins", symbol: "square.grid.2x2",
                            message: "Plugin choices are unavailable. Check your account in Preferences.")
                    }
                }
            }
            .tabItem { Label("Plugins", systemImage: "square.grid.2x2") }.tag(JarvisProductTab.plugins)
        }
        .tint(JarvisColors.amber)
        .preferredColorScheme(.light)
        .toolbarBackground(JarvisColors.surface, for: .tabBar)
        .toolbarBackground(.visible, for: .tabBar)
        .onChange(of: sessionID, initial: true) { _, id in
            viewModel.synchronizeProductSession(id)
            updateVisibility()
        }
        .onChange(of: viewModel.workspace.map { ObjectIdentifier($0) }) { _, identity in
            if identity == nil { viewModel.synchronizeProductSession(sessionID) }
            updateVisibility()
        }
        .onChange(of: viewModel.productTab) { _, _ in updateVisibility() }
        .onChange(of: viewModel.presentedSheet) { _, _ in updateVisibility() }
        .onChange(of: scenePhase) { _, _ in updateVisibility() }
        .onChange(of: audio.isJarvisActive, initial: true) { _, active in
            viewModel.workspace?.setActivity(voiceActive: active)
        }
        .onDisappear {
            viewModel.updateProductScene(foreground: false)
            viewModel.workspace?.setVisibility(false, foreground: false)
        }
    }

    private func updateVisibility() {
        viewModel.updateProductScene(foreground: scenePhase == .active)
        viewModel.workspace?.setActivity(voiceActive: audio.isJarvisActive)
        viewModel.workspace?.setVisibility(
            (viewModel.presentedSheet == nil && viewModel.productTab == .workspace) || viewModel.presentedSheet == .offlineCopies,
            foreground: scenePhase == .active, workVisible: viewModel.productTab == .workspace)
    }
}

struct JarvisProductUnavailableView: View {
    let title: String
    let symbol: String
    let message: String
    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: symbol)
        } description: {
            Text(message)
        }
    }
}
