import SwiftUI

struct AmbientCompanionView: View {
    @ObservedObject var viewModel: JarvisAppViewModel
    @ObservedObject var audioSessionViewModel: JarvisAudioSessionViewModel
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var presentedErrorMessage: String?
    @State private var dismissedErrorMessage: String?
    @State private var choosingRecordingAudio = false

    private var outerPadding: CGFloat { dynamicTypeSize.isAccessibilitySize ? 16 : 24 }

    private var isHoldToTalkReady: Bool {
        audioSessionViewModel.captureMode == .holdToTalk && audioSessionViewModel.isConnected
    }

    private var isShowingErrorAlert: Binding<Bool> {
        Binding(
            get: { presentedErrorMessage != nil },
            set: { isPresented in
                if !isPresented {
                    if let current = presentedErrorMessage {
                        dismissedErrorMessage = current
                    }
                    presentedErrorMessage = nil
                }
            }
        )
    }

    var body: some View {
        GeometryReader { proxy in
            ScrollView {
                VStack(spacing: 0) {
                    topBar

                    Spacer(minLength: dynamicTypeSize.isAccessibilitySize ? 28 : 40)

                    centerControls

                    Spacer(minLength: dynamicTypeSize.isAccessibilitySize ? 32 : 48)
                }
                .padding(.horizontal, outerPadding)
                .padding(.top, 16)
                .padding(.bottom, 24)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity, minHeight: max(proxy.size.height, 0))
            }
            .refreshable {
                viewModel.refreshActiveSession()
                viewModel.refreshReminderState()
                await viewModel.reloadMemoryState()
            }
        }
        .background(JarvisColors.background.ignoresSafeArea())
        .onChange(of: audioErrorMessage, initial: true) { _, newError in
            guard let newError else {
                presentedErrorMessage = nil
                dismissedErrorMessage = nil
                return
            }
            if newError != dismissedErrorMessage {
                presentedErrorMessage = newError
            }
        }
        .alert("Jarvis", isPresented: isShowingErrorAlert, presenting: presentedErrorMessage) { _ in
            Button("OK", role: .cancel) {}
        } message: { message in
            Text(message)
        }
    }

    private var topBar: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("Jarvis")
                .font(JarvisTypography.pageTitle)
                .foregroundStyle(JarvisColors.primaryText)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 6) {
                headerIconButton(
                    systemName: "bell",
                    accessibilityLabel: "Reminders",
                    identifier: "homeRemindersButton"
                ) {
                    viewModel.presentedSheet = .reminders
                }

                headerIconButton(
                    systemName: "slider.horizontal.3",
                    accessibilityLabel: "Preferences",
                    identifier: "homePreferencesButton"
                ) {
                    viewModel.presentedSheet = .preferences
                }
            }
            .fixedSize()
        }
    }

    private func headerIconButton(
        systemName: String,
        accessibilityLabel: String,
        identifier: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(JarvisColors.primaryText)
                .frame(width: 38, height: 38)
                .background(
                    Circle()
                        .fill(JarvisColors.surface)
                        .overlay(
                            Circle()
                                .strokeBorder(JarvisColors.border, lineWidth: 1)
                        )
                )
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityIdentifier(identifier)
    }

    private var centerControls: some View {
        VStack(spacing: dynamicTypeSize.isAccessibilitySize ? 24 : 28) {
            waveformControl
            primarySessionButton
            if audioSessionViewModel.rayBanDiagnostics.registrationState == .registered
                && audioSessionViewModel.rayBanDiagnostics.permissionState == .granted
                && audioSessionViewModel.rayBanDiagnostics.micPermissionState != .granted {
                Button {
                    audioSessionViewModel.requestRayBanRegistrationOrPermission()
                } label: {
                    Label("Enable Ray-Ban Glasses Mic", systemImage: "mic.badge.plus")
                        .font(JarvisTypography.label(13))
                        .foregroundStyle(JarvisColors.graphite)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Capsule().fill(JarvisColors.amber))
                }
                .buttonStyle(.plain)
            }
            JarvisHomeRecordingView(phase: audioSessionViewModel.recordingPhase,
                message: audioSessionViewModel.recordingStatus,
                stop: audioSessionViewModel.stopRecording)
            if audioSessionViewModel.canRequestRecording && !audioSessionViewModel.isRecording
                && audioSessionViewModel.recordingPhase != .retaining {
                Button { choosingRecordingAudio = true } label: {
                    Label("Record with glasses", systemImage: "record.circle")
                        .font(JarvisTypography.label(14)).frame(minHeight: 44)
                }
                .disabled(audioSessionViewModel.recordingCommandInFlight)
                .tint(JarvisColors.accent)
                .confirmationDialog("Include Jarvis's voice?", isPresented: $choosingRecordingAudio, titleVisibility: .visible) {
                    Button("Include Jarvis audio") { audioSessionViewModel.startRecording(includeJarvisAudio: true) }
                    Button("Without Jarvis audio") { audioSessionViewModel.startRecording(includeJarvisAudio: false) }
                    Button("Cancel", role: .cancel) { }
                }
            }
            if let message = audioSessionViewModel.recordingCommandMessage {
                Text(message).font(.caption).foregroundStyle(JarvisColors.secondaryText)
                    .multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private var waveformControl: some View {
        let baseIcon = Image(systemName: "waveform")
            .font(
                .system(
                    size: dynamicTypeSize.isAccessibilitySize ? 44 : 52,
                    weight: (isHoldToTalkReady && audioSessionViewModel.isManualRecording) ? .medium : .regular
                )
            )
            .foregroundStyle(waveformColor)
            .frame(minWidth: 72, minHeight: 72)
            .contentShape(Rectangle())

        if isHoldToTalkReady {
            baseIcon
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in audioSessionViewModel.beginManualSpeech() }
                        .onEnded { _ in audioSessionViewModel.endManualSpeech() }
                )
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(audioSessionViewModel.isManualRecording ? "Release to send" : "Hold to talk")
                .accessibilityValue(stateSummaryLabel)
                .accessibilityHint("Press and hold to speak, or activate to toggle manual speech.")
                .accessibilityAddTraits(.isButton)
                .accessibilityAction {
                    toggleManualSpeechForAccessibility()
                }
                .accessibilityAction(named: audioSessionViewModel.isManualRecording ? "Release to send" : "Hold to talk") {
                    toggleManualSpeechForAccessibility()
                }
        } else {
            baseIcon
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Jarvis audio, \(stateSummaryLabel)")
        }
    }

    private var waveformColor: Color {
        if audioSessionViewModel.isConnected || audioSessionViewModel.isJarvisActive {
            return JarvisColors.accent
        }
        if audioSessionViewModel.isJarvisStarting {
            return JarvisColors.accent.opacity(0.68)
        }
        return JarvisColors.secondaryText
    }

    private var stateSummaryLabel: String {
        let turnLabel = audioSessionViewModel.voiceRouteFailure == nil
            ? audioSessionViewModel.turnState.label
            : "Voice unavailable"
        return "\(audioSessionViewModel.connectionState.label), \(turnLabel)"
    }

    private func toggleManualSpeechForAccessibility() {
        if audioSessionViewModel.isManualRecording {
            audioSessionViewModel.endManualSpeech()
        } else {
            audioSessionViewModel.beginManualSpeech()
        }
    }

    private var primarySessionButton: some View {
        let isStopping = audioSessionViewModel.shouldShowStopJarvis
        let buttonShape = RoundedRectangle(
            cornerRadius: dynamicTypeSize.isAccessibilitySize ? 26 : 100,
            style: .continuous
        )

        return Button {
            if isStopping {
                audioSessionViewModel.stopJarvis()
            } else {
                dismissedErrorMessage = nil
                audioSessionViewModel.startJarvis(startupOrigin: .humanStartButton)
                if let currentError = audioErrorMessage {
                    presentedErrorMessage = currentError
                }
            }
        } label: {
            Text(isStopping ? "Stop" : "Start")
                .font(.body.weight(.semibold))
                .foregroundStyle(JarvisColors.surface)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, dynamicTypeSize.isAccessibilitySize ? 24 : 38)
                .padding(.vertical, 15)
                .frame(
                    minWidth: dynamicTypeSize.isAccessibilitySize ? nil : 164,
                    maxWidth: dynamicTypeSize.isAccessibilitySize ? .infinity : nil,
                    minHeight: 50
                )
                .background(
                    buttonShape
                        .fill(isStopping ? JarvisColors.primaryText : JarvisColors.accent)
                        .shadow(
                            color: (isStopping ? JarvisColors.primaryText : JarvisColors.accent).opacity(0.12),
                            radius: 10,
                            x: 0,
                            y: 4
                        )
                )
                .contentShape(buttonShape)
        }
        .buttonStyle(.plain)
    }

    private var audioErrorMessage: String? {
        if let failure = audioSessionViewModel.voiceRouteFailure { return failure.message }
        if case let .failed(message) = audioSessionViewModel.connectionState {
            return message
        }
        if case let .failed(message) = audioSessionViewModel.turnState {
            return message
        }
        return viewModel.errorMessage
    }
}


/// Reads the capture controller's state. Touch and voice share its stop operation.
struct JarvisHomeRecordingView: View {
    let phase: JarvisRecordingSession.State?
    let message: String
    let stop: () -> Void
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        if let phase {
            VStack(spacing: 14) {
                HStack(spacing: 9) {
                    if phase == .recording {
                        Circle().fill(JarvisColors.clay).frame(width: 7, height: 7)
                    } else if [.starting, .stopping, .retaining].contains(phase) {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: phase == .retained ? "checkmark.circle" : "exclamationmark.circle")
                            .foregroundStyle(phase == .retained ? JarvisColors.accent : JarvisColors.clay)
                    }
                    Text(phase == .recording ? "Recording" : message)
                        .font(JarvisTypography.body(14))
                        .foregroundStyle(JarvisColors.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("homeRecordingStatus")
                // Latest instruction: idle/finalizing never shows a Stop button.
                if phase == .recording {
                    Button(action: stop) {
                        Label("Stop recording", systemImage: "stop.fill")
                            .font(JarvisTypography.label(14))
                            .padding(.horizontal, 18).frame(minHeight: 44)
                            .foregroundStyle(JarvisColors.primaryText)
                            .background(JarvisColors.surface, in: Capsule())
                            .overlay(Capsule().strokeBorder(JarvisColors.border, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("homeStopRecordingButton")
                }
            }
            .frame(maxWidth: typeSize.isAccessibilitySize ? .infinity : 280)
            .padding(.top, 4)
        }
    }
}
