#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/media-phase-host.XXXXXX")"
printf 'Evidence: %s\n' "$OUT"
xcrun swiftc -swift-version 6 -framework AVFoundation -framework Foundation \
  "$ROOT/ios/Jarvis/Jarvis/Services/JarvisMediaStream.swift" \
  "$ROOT/ios/Jarvis/OfflineTests/JarvisMediaTransportChecks.swift" \
  -o "$OUT/checks" > "$OUT/compiler.log" 2>&1
# No arguments selects synthetic-only checks. Never connect to a socket here.
"$OUT/checks" 2>&1 | tee "$OUT/checks.log"
