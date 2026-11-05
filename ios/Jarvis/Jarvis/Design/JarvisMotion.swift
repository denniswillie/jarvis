import SwiftUI

struct JarvisMotion {
    static let stateTransition = Animation.spring(response: 0.55, dampingFraction: 0.86)
    static let gentlePulse = Animation.easeInOut(duration: 1.8).repeatForever(autoreverses: true)
    static let slowDrift = Animation.easeInOut(duration: 7.5).repeatForever(autoreverses: true)
    static let quickFade = Animation.easeOut(duration: 0.22)
}
