import SwiftUI

struct LaunchView: View {
    let statusText: String

    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            JarvisOrb(state: .launching)

            VStack(spacing: 10) {
                Text("Jarvis")
                    .font(JarvisTypography.brand(58))
                    .foregroundStyle(JarvisColors.cream)

                Text(statusText)
                    .font(JarvisTypography.body(16))
                    .foregroundStyle(JarvisColors.mutedWhite)
            }

            Spacer()
        }
        .padding(28)
    }
}
