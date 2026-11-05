#!/bin/bash
set -euo pipefail
# One process group and one total 175-second offline budget. Logs/artifacts are
# retained on timeout/failure. This runner never builds/launches the iOS app.
if [[ "${JARVIS_CAPTURE_BUDGET_ACTIVE:-}" != 1 ]]; then
    exec /usr/bin/python3 - "$0" <<'PY'
import os, signal, subprocess, sys
env = dict(os.environ, JARVIS_CAPTURE_BUDGET_ACTIVE="1")
p = subprocess.Popen(["/bin/bash", sys.argv[1]], env=env, start_new_session=True)
try:
    sys.exit(p.wait(timeout=175))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    print("Capture offline budget expired; preserve the printed evidence directory.", file=sys.stderr)
    sys.exit(124)
PY
fi
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/recording-capture-host.XXXXXX")"
SERVICES="$ROOT/ios/Jarvis/Jarvis/Services"
SOURCES=("$SERVICES/JarvisRecordingTimeline.swift" "$SERVICES/JarvisRecordingMixer.swift"
    "$SERVICES/JarvisRecordingWriter.swift" "$SERVICES/JarvisRecordingVideoSource.swift"
    "$SERVICES/JarvisRecordingAudioSource.swift" "$SERVICES/JarvisRecordingCapture.swift"
    "$ROOT/ios/Jarvis/OfflineTests/JarvisRecordingCaptureChecks.swift")
printf 'Evidence: %s\n' "$OUT"
shasum -a 256 "${SOURCES[@]}" > "$OUT/sources.sha256"
if ! xcrun swiftc -swift-version 6 -D JARVIS_RECORDING_CAPTURE_HOST_TESTS \
    -framework Foundation -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreAudio \
    "${SOURCES[@]}" -o "$OUT/checks" > "$OUT/compiler.log" 2>&1; then
    cat "$OUT/compiler.log"
    exit 1
fi
JARVIS_RECORDING_TEST_OUTPUT="$OUT" "$OUT/checks" 2>&1 | tee "$OUT/checks.log"
shasum -a 256 -c "$OUT/sources.sha256" > "$OUT/source-check.log"
