#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/recording-control-host.XXXXXX")"
SOURCE="$ROOT/ios/Jarvis/Jarvis/Services/JarvisRecordingControl.swift"
CHECK="$ROOT/ios/Jarvis/OfflineTests/JarvisRecordingControlChecks.swift"
printf 'Evidence: %s\n' "$OUT"
shasum -a 256 "$SOURCE" "$CHECK" > "$OUT/sources.sha256"
xcrun swiftc -swift-version 6 -framework Foundation "$SOURCE" "$CHECK" -o "$OUT/checks" > "$OUT/compiler.log" 2>&1
"$OUT/checks" | tee "$OUT/checks.log"
shasum -a 256 -c "$OUT/sources.sha256" > "$OUT/source-check.log"
