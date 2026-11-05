#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/recording-timeline-host.XXXXXX")"
SOURCE="$ROOT/ios/Jarvis/Jarvis/Services/JarvisRecordingTimeline.swift"
CHECK="$ROOT/ios/Jarvis/OfflineTests/JarvisRecordingTimelineChecks.swift"
printf 'Evidence: %s\n' "$OUT"
shasum -a 256 "$SOURCE" "$CHECK" > "$OUT/sources.sha256"
xcrun swiftc -swift-version 6 -framework Foundation -framework CoreMedia -framework CoreVideo -framework CoreAudio \
    "$SOURCE" "$CHECK" -o "$OUT/checks" > "$OUT/compiler.log" 2>&1
"$OUT/checks" | tee "$OUT/checks.log"
shasum -a 256 -c "$OUT/sources.sha256" > "$OUT/source-check.log"
