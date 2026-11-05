#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/jarvis-route-preferences.XXXXXX")}"
SERVICES="$ROOT/ios/Jarvis/Jarvis/Services"
CHECKS="$ROOT/ios/Jarvis/OfflineTests/JarvisRoutePreferenceChecks.swift"
mkdir -p "$OUT"
printf 'Evidence: %s\n' "$OUT"
shasum -a 256 "$SERVICES/JarvisLiveAppRoute.swift" \
    "$SERVICES/JarvisDuplexAudioRoute.swift" "$CHECKS" > "$OUT/sources.sha256"

# Compile exact production preference declarations, stopping before URL/audio
# dependencies. Fail closed if either boundary changes. The signed iOS build
# separately checks the complete source files against the real frameworks.
printf 'import Foundation\n' > "$OUT/Preferences.swift"
awk '
    /^enum JarvisVoiceRouteChoice:/ { emit = 1 }
    emit && /^    func audioURL/ { print "}"; found = 1; exit }
    emit { print }
    END { if (!found) exit 1 }
' "$SERVICES/JarvisLiveAppRoute.swift" >> "$OUT/Preferences.swift"
awk '
    /^final class JarvisDuplexAudioRoute/ { print "@MainActor"; emit = 1 }
    emit && /^    let device:/ { print "}"; found = 1; exit }
    emit { print }
    END { if (!found) exit 1 }
' "$SERVICES/JarvisDuplexAudioRoute.swift" >> "$OUT/Preferences.swift"
xcrun swiftc -swift-version 6 -module-cache-path "$OUT/ModuleCache" \
    "$OUT/Preferences.swift" "$CHECKS" -o "$OUT/checks" > "$OUT/compile.log" 2>&1
"$OUT/checks" | tee "$OUT/checks.log"
