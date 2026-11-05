import SwiftUI

struct SetupTokenView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @State private var backendURL = JarvisConfiguration.defaultBackendBaseURLString
    @State private var accessToken = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                header

                VStack(spacing: 18) {
                    JarvisInputField(
                        title: "Backend URL",
                        placeholder: JarvisConfiguration.defaultBackendBaseURLString,
                        text: $backendURL
                    )
                    .keyboardType(.URL)

                    JarvisInputField(
                        title: "Access token",
                        placeholder: "Paste Jarvis bearer token",
                        text: $accessToken,
                        isSecure: true
                    )
                }

                VStack(spacing: 12) {
                    JarvisPrimaryButton(title: "Save and continue", isLoading: false) {
                        viewModel.saveSetup(
                            backendBaseURL: backendURL,
                            staticBearerToken: accessToken
                        )
                    }

                    if viewModel.hasBundledConfiguration {
                        JarvisSecondaryButton(title: "Use bundled config") {
                            viewModel.useBundledConfiguration()
                        }
                    }
                }

                if let errorMessage = viewModel.setupErrorMessage {
                    Text(errorMessage)
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.amber)
                }
            }
            .padding(.horizontal, 26)
            .padding(.top, 70)
            .padding(.bottom, 36)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Jarvis")
                .font(JarvisTypography.brand(62))
                .foregroundStyle(JarvisColors.cream)

            Text("Connect your private companion backend to start voice-first sessions through your glasses.")
                .font(JarvisTypography.body(18))
                .foregroundStyle(JarvisColors.softWhite)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
