import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @ObservedObject var audioSessionViewModel: JarvisAudioSessionViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var backendURL = ""
    @State private var accessToken = ""
    @State private var phoneDebug = false
    @State private var confirmLocalForget = false

    var body: some View {
        NavigationStack {
            ZStack {
                JarvisAtmosphericBackground(state: viewModel.state)

                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        Text("Settings")
                            .font(JarvisTypography.brand(42))
                            .foregroundStyle(JarvisColors.cream)

                        if let error = viewModel.setupErrorMessage {
                            Text(error).font(JarvisTypography.body(13)).foregroundStyle(JarvisColors.amber)
                        }
                        if let notifications = viewModel.pushNotifications {
                            JarvisPushSettingsView(model: notifications)
                        }

                        JarvisPanel {
                            Text("Backend")
                                .font(JarvisTypography.label(13))
                                .foregroundStyle(JarvisColors.mutedWhite)

                            Text(viewModel.backendDisplayText)
                                .font(JarvisTypography.mono(12))
                                .foregroundStyle(JarvisColors.softWhite)
                                .textSelection(.enabled)

                            JarvisInputField(
                                title: "Replace backend URL",
                                placeholder: JarvisConfiguration.defaultBackendBaseURLString,
                                text: $backendURL
                            )
                            .keyboardType(.URL)

                            JarvisInputField(
                                title: "Replace access token",
                                placeholder: "Paste new token",
                                text: $accessToken,
                                isSecure: true
                            )

                            JarvisPrimaryButton(title: "Save backend settings", isLoading: false) {
                                viewModel.saveSetup(
                                    backendBaseURL: backendURL,
                                    staticBearerToken: accessToken
                                )
                                if viewModel.setupErrorMessage == nil { dismiss() }
                            }
                        }

                        JarvisPanel {
                            Text("Audio provider")
                                .font(JarvisTypography.label(14))
                                .foregroundStyle(JarvisColors.cream)

                            Text(JarvisAudioProvider.openai.label)
                                .font(JarvisTypography.body(15))
                                .foregroundStyle(JarvisColors.amber)
                                .accessibilityIdentifier("audioProviderSelection")

                            Text("Voice uses OpenAI. Gemini continues to analyze camera images.")
                                .font(JarvisTypography.body(13))
                                .foregroundStyle(JarvisColors.mutedWhite)

                            Picker("Speech mode", selection: $audioSessionViewModel.captureMode) {
                                ForEach(JarvisAudioCaptureMode.allCases) { mode in
                                    Text(mode.label).tag(mode)
                                }
                            }
                            .pickerStyle(.menu)
                            .tint(JarvisColors.amber)
                            .accessibilityIdentifier("speechModeSelection")
                            .onChange(of: audioSessionViewModel.captureMode) { _, _ in
                                audioSessionViewModel.captureModeDidChange()
                            }

                            Picker("Voice route", selection: Binding(
                                get: { audioSessionViewModel.preferredVoiceRoute },
                                set: { audioSessionViewModel.setPreferredVoiceRoute($0) }
                            )) {
                                ForEach(JarvisVoiceRouteChoice.allCases) { route in
                                    Text(route.label).tag(route)
                                }
                            }
                            .disabled(audioSessionViewModel.shouldShowStopJarvis)
                            .accessibilityIdentifier("voiceRouteSelection")

                            Text("GPT Live requires a Live-enabled backend. Stop the session before changing routes. No automatic fallback; choose Realtime to roll back. Live handles turns automatically; hold-to-talk only gates the microphone. Captions are provisional and cannot prove that audio was heard.")
                                .font(JarvisTypography.body(12))
                                .foregroundStyle(JarvisColors.mutedWhite)

                            Text(audioSessionViewModel.audioProviderStatus)
                                .font(JarvisTypography.mono(11))
                                .foregroundStyle(JarvisColors.softWhite)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        JarvisPanel {
                            HStack(alignment: .firstTextBaseline) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Audio route")
                                        .font(JarvisTypography.label(14))
                                        .foregroundStyle(JarvisColors.cream)
                                    Text(audioSessionViewModel.routeSummary)
                                        .font(JarvisTypography.body(13))
                                        .foregroundStyle(JarvisColors.mutedWhite)
                                }

                                Spacer()

                                JarvisStatePill(
                                    text: audioSessionViewModel.routeUsesGlasses ? "Glasses" : "Fallback",
                                    isActive: audioSessionViewModel.routeUsesGlasses
                                )
                            }

                            JarvisSecondaryButton(title: "Refresh audio route") {
                                audioSessionViewModel.refreshRouteDiagnostics()
                            }
                        }

                        rayBanDiagnosticsPanel

                        JarvisPanel {
                            HStack(alignment: .firstTextBaseline) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Backend reminders")
                                        .font(JarvisTypography.label(14))
                                        .foregroundStyle(JarvisColors.cream)
                                    Text(viewModel.reminderSyncStatusText)
                                        .font(JarvisTypography.body(13))
                                        .foregroundStyle(JarvisColors.mutedWhite)
                                }

                                Spacer()

                                JarvisStatePill(
                                    text: "\(viewModel.activeReminders.count) active",
                                    isActive: !viewModel.activeReminders.isEmpty
                                )
                            }

                            Text("\(viewModel.reminderEvents.count) recent reminder events")
                                .font(JarvisTypography.mono(11))
                                .foregroundStyle(JarvisColors.softWhite)

                            JarvisSecondaryButton(title: "Refresh reminders") {
                                viewModel.refreshReminderState()
                            }
                        }

                        JarvisPanel {
                            HStack(alignment: .firstTextBaseline) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Google Calendar")
                                        .font(JarvisTypography.label(14))
                                        .foregroundStyle(JarvisColors.cream)
                                    Text(calendarStatusDescription)
                                        .font(JarvisTypography.body(13))
                                        .foregroundStyle(JarvisColors.mutedWhite)
                                        .fixedSize(horizontal: false, vertical: true)
                                }

                                Spacer()

                                JarvisStatePill(
                                    text: viewModel.googleCalendarStatusLabel,
                                    isActive: viewModel.googleCalendarConnected
                                )
                            }

                            if let error = viewModel.googleCalendarStatusError {
                                Text(error)
                                    .font(JarvisTypography.mono(11))
                                    .foregroundStyle(JarvisColors.softWhite)
                                    .fixedSize(horizontal: false, vertical: true)
                            }

                            JarvisPrimaryButton(
                                title: viewModel.googleCalendarConnected ? "Reconnect calendar" : "Connect Google Calendar",
                                isLoading: viewModel.isGoogleCalendarConnecting
                            ) {
                                viewModel.connectGoogleCalendar()
                            }

                            JarvisSecondaryButton(title: "Refresh calendar status") {
                                viewModel.refreshGoogleCalendarStatus()
                            }
                        }

                        JarvisPanel {
                            Toggle(isOn: Binding(
                                get: { phoneDebug },
                                set: { newValue in
                                    phoneDebug = newValue
                                    viewModel.setPhoneCameraDebugEnabled(newValue)
                                }
                            )) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Phone camera debug")
                                        .font(JarvisTypography.label(14))
                                        .foregroundStyle(JarvisColors.cream)
                                    Text("Reserved for debug capture. Phase 6A keeps glasses as the only active visual source.")
                                        .font(JarvisTypography.body(13))
                                        .foregroundStyle(JarvisColors.mutedWhite)
                                }
                            }
                            .tint(JarvisColors.amber)
                        }

                        JarvisPanel {
                            Text("Diagnostics")
                                .font(JarvisTypography.label(13))
                                .foregroundStyle(JarvisColors.mutedWhite)

                            if audioSessionViewModel.diagnostics.isEmpty {
                                Text("No audio diagnostics yet.")
                                    .font(JarvisTypography.body(14))
                                    .foregroundStyle(JarvisColors.softWhite)
                            } else {
                                VStack(alignment: .leading, spacing: 8) {
                                    ForEach(audioSessionViewModel.diagnostics.suffix(8), id: \.self) { line in
                                        Text(line)
                                            .font(JarvisTypography.mono(11))
                                            .foregroundStyle(JarvisColors.softWhite)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                }
                            }
                        }

                        JarvisSecondaryButton(title: "Reset local configuration") {
                            viewModel.resetConfiguration()
                            if viewModel.configuration == nil { dismiss() }
                        }
                        Button("Forget local connection without server revocation", role: .destructive) {
                            confirmLocalForget = true
                        }
                        .font(JarvisTypography.body(12))
                    }
                    .padding(.horizontal, 22)
                    .padding(.top, 28)
                    .padding(.bottom, 40)
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundStyle(JarvisColors.cream)
                }
            }
        }
        .confirmationDialog("Forget this local connection? Server notifications may remain enabled. You can also disable Jarvis notifications in iOS Settings. No server revocation will be claimed.", isPresented: $confirmLocalForget) {
            Button("Forget locally", role: .destructive) {
                viewModel.resetConfiguration(forgetWithoutServerRevocation: true)
                dismiss()
            }
        }
        .onAppear {
            backendURL = viewModel.configuration?.backendBaseURL.absoluteString ?? ""
            phoneDebug = viewModel.phoneCameraDebugEnabled
            viewModel.refreshGoogleCalendarStatus()
        }
    }

    private var calendarStatusDescription: String {
        if viewModel.googleCalendarConnected {
            return "Jarvis can read and manage your calendar when it helps with daily plans."
        }
        return "Connect once so Jarvis can read and manage events for you."
    }

    private var rayBanDiagnosticsPanel: some View {
        JarvisPanel {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Ray-Ban DAT")
                        .font(JarvisTypography.label(14))
                        .foregroundStyle(JarvisColors.cream)
                    Text(audioSessionViewModel.visualStatus)
                        .font(JarvisTypography.body(13))
                        .foregroundStyle(JarvisColors.mutedWhite)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                JarvisStatePill(
                    text: audioSessionViewModel.rayBanDiagnostics.streamState.label,
                    isActive: audioSessionViewModel.rayBanDiagnostics.streamState == .active
                )
            }

            VStack(alignment: .leading, spacing: 8) {
                rayBanRow("SDK", audioSessionViewModel.rayBanDiagnostics.configuration.sdkVersionDescription)
                rayBanRow("Configured", audioSessionViewModel.rayBanDiagnostics.configuration.configurationModeLabel)
                rayBanRow("Registration", audioSessionViewModel.rayBanDiagnostics.registrationState.label)
                rayBanRow("Camera permission", audioSessionViewModel.rayBanDiagnostics.permissionState.label)
                rayBanRow("Glasses mic permission", audioSessionViewModel.rayBanDiagnostics.micPermissionState.label)
                rayBanRow("Device", "\(audioSessionViewModel.rayBanDiagnostics.deviceState.label) (\(audioSessionViewModel.rayBanDiagnostics.deviceCount))")
                rayBanRow("Latest frame", latestRayBanFrameDescription)
                rayBanRow("Backend visual", audioSessionViewModel.visualBackendStatus)
                rayBanRow("Frames accepted", "\(audioSessionViewModel.visualFrameAcceptedCount)")
                rayBanRow("Frames dropped", "\(audioSessionViewModel.visualFrameDroppedCount)")
                if let lastError = audioSessionViewModel.rayBanDiagnostics.lastError {
                    rayBanRow("Last error", lastError)
                }
            }

            HStack(spacing: 10) {
                Button {
                    audioSessionViewModel.refreshRayBanDiagnostics()
                } label: {
                    Text("Refresh")
                        .font(JarvisTypography.label(14))
                        .foregroundStyle(JarvisColors.cream)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Capsule().fill(JarvisColors.panelStrong))
                }
                .buttonStyle(.plain)

                Button {
                    audioSessionViewModel.requestRayBanRegistrationOrPermission()
                } label: {
                    Text("Register / permission")
                        .font(JarvisTypography.label(14))
                        .foregroundStyle(JarvisColors.graphite)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Capsule().fill(JarvisColors.amber))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func rayBanRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(JarvisTypography.mono(11))
                .foregroundStyle(JarvisColors.mutedWhite)
            Spacer()
            Text(value)
                .font(JarvisTypography.mono(11))
                .foregroundStyle(JarvisColors.softWhite)
                .multilineTextAlignment(.trailing)
        }
    }

    private var latestRayBanFrameDescription: String {
        guard let sequence = audioSessionViewModel.rayBanDiagnostics.lastFrameSequence,
              let size = audioSessionViewModel.rayBanDiagnostics.lastFrameSizeDescription else {
            return "-"
        }
        return "#\(sequence) · \(size)"
    }
}
