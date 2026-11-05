#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT=${1:-$(mktemp -d "${TMPDIR:-/tmp}/jarvis-recording-custody.XXXXXX")}
mkdir -p "$OUT"
/usr/bin/xcrun swiftc -swift-version 5 -parse-as-library \
  "$ROOT/Jarvis/Services/JarvisRecordingWriter.swift" \
  "$ROOT/OfflineTests/JarvisRecordingCustodyChecks.swift" -o "$OUT/checks"
"$OUT/checks"
