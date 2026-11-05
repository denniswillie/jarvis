#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT=${1:-$(mktemp -d "${TMPDIR:-/tmp}/jarvis-recording-coordinator.XXXXXX")}
mkdir -p "$OUT"
/usr/bin/xcrun swiftc -swift-version 5 -parse-as-library \
  "$ROOT/Jarvis/Services/JarvisRecordingControl.swift" \
  "$ROOT/Jarvis/Services/JarvisRecordingCoordinator.swift" \
  "$ROOT/OfflineTests/JarvisRecordingCoordinatorChecks.swift" \
  -o "$OUT/checks"
"$OUT/checks"
