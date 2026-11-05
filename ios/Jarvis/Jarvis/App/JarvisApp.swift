import SwiftUI

@main
struct JarvisApp: App {
    var body: some Scene {
        WindowGroup {
#if DEBUG
            if ProcessInfo.processInfo.arguments.contains("--jarvis-duplex-probe") {
                JarvisDuplexLifecycleProbeView()
            } else if ProcessInfo.processInfo.arguments.contains("--jarvis-replay-manifest") {
                JarvisReplayLaunchView()
            } else {
                JarvisNormalRoot()
            }
#else
            JarvisNormalRoot()
#endif
        }
    }
}

// Keep normal services out of the isolated audio probe, including DAT setup.
@MainActor
private struct JarvisNormalRoot: View {
    // Remote/APNs notifications are intentionally unsupported in this app build.
    @StateObject private var viewModel = JarvisAppViewModel()

    var body: some View {
            JarvisRootView(viewModel: viewModel)
                .task {
                    viewModel.audioSessionViewModel.configureRayBanIfNeeded()
                }
                .onOpenURL { url in
                    _ = viewModel.audioSessionViewModel.handleOpenURL(url)
                }
    }
}
