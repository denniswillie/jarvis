import SwiftUI
import UIKit

struct JarvisTypography {
    /// Editorial display type paired with quiet sans-serif controls and data.
    /// Semantic sizing follows the view's Dynamic Type environment.
    static let pageTitle: Font = .system(.largeTitle, design: .serif).weight(.regular)

    static func brand(_ size: CGFloat) -> Font {
        scaledSystemFont(
            size: size,
            weight: .regular,
            design: .serif,
            relativeTo: .largeTitle
        )
    }

    static func headline(_ size: CGFloat) -> Font {
        scaledSystemFont(
            size: size,
            weight: .medium,
            design: .serif,
            relativeTo: .title2
        )
    }

    static func body(_ size: CGFloat = 16) -> Font {
        scaledSystemFont(
            size: size,
            weight: .regular,
            design: .default,
            relativeTo: .body
        )
    }

    static func label(_ size: CGFloat = 13) -> Font {
        scaledSystemFont(
            size: size,
            weight: .semibold,
            design: .default,
            relativeTo: .caption1
        )
    }

    static func mono(_ size: CGFloat = 12) -> Font {
        scaledSystemFont(
            size: size,
            weight: .medium,
            design: .monospaced,
            relativeTo: .caption2
        )
    }

    private static func scaledSystemFont(
        size: CGFloat,
        weight: UIFont.Weight,
        design: UIFontDescriptor.SystemDesign,
        relativeTo textStyle: UIFont.TextStyle
    ) -> Font {
        let baseFont = UIFont.systemFont(ofSize: size, weight: weight)
        let descriptor = baseFont.fontDescriptor.withDesign(design) ?? baseFont.fontDescriptor
        let designedFont = UIFont(descriptor: descriptor, size: size)
        let scaledFont = UIFontMetrics(forTextStyle: textStyle).scaledFont(for: designedFont)
        return Font(scaledFont)
    }
}
