import SwiftUI

struct JarvisColors {
    static let background = Color(hex: 0xF7F7F4)
    static let surface = Color.white
    static let primaryText = Color(hex: 0x161D1C)
    static let secondaryText = Color(hex: 0x505B59)
    static let accent = Color(hex: 0x23584B)
    static let border = Color(hex: 0xE2E6E4)
    static let softAccent = Color(hex: 0xE6F0EC)

    // Existing product and reserved screens already consume these tokens.
    // Keep compatibility while moving their background/foreground roles to light.
    static let graphite = background
    static let ink = surface
    static let moss = softAccent
    static let fern = Color(hex: 0x4A6B5D)
    static let amber = accent
    static let cream = primaryText
    static let porcelain = Color(hex: 0x10241B)
    static let clay = Color(hex: 0xA14635)
    static let softWhite = secondaryText
    static let mutedWhite = Color(hex: 0x576360)
    static let panel = surface
    static let panelStrong = Color(hex: 0xEEF0ED)
}

extension Color {
    init(hex: UInt, alpha: Double = 1.0) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: alpha
        )
    }
}
