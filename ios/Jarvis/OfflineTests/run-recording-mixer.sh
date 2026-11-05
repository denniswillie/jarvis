#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/recording-mixer-host.XXXXXX")"
SERVICES="$ROOT/ios/Jarvis/Jarvis/Services"
CHECK="$ROOT/ios/Jarvis/OfflineTests/JarvisRecordingMixerChecks.swift"
SOURCES=("$SERVICES/JarvisRecordingTimeline.swift" "$SERVICES/JarvisRecordingMixer.swift" "$SERVICES/JarvisRecordingWriter.swift" "$CHECK")
printf 'Evidence: %s\n' "$OUT"
shasum -a 256 "${SOURCES[@]}" > "$OUT/sources.sha256"
xcrun swiftc -swift-version 6 -framework Foundation -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreAudio \
    "${SOURCES[@]}" -o "$OUT/checks" > "$OUT/compiler.log" 2>&1
JARVIS_RECORDING_TEST_OUTPUT="$OUT" "$OUT/checks" | tee "$OUT/checks.log"
shasum -a 256 -c "$OUT/sources.sha256" > "$OUT/source-check.log"
