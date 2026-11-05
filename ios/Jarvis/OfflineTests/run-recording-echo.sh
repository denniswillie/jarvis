#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
work="$(mktemp -d "$root/.jarvis-dev/recording-echo-host.XXXXXX")"
echo "Evidence: $work"
src=ios/Jarvis/Jarvis/Services
xcrun clang -O2 -c "$src/AudioEcho/JarvisEchoDSP.c" -o "$work/echo.o"
xcrun swiftc -O -D JARVIS_ECHO_TEST -import-objc-header "$src/AudioEcho/JarvisEchoDSP.h" \
  "$src/JarvisRecordingTimeline.swift" "$src/JarvisRecordingMixer.swift" \
  ios/Jarvis/OfflineTests/JarvisRecordingEchoChecks.swift "$work/echo.o" \
  -framework Foundation -framework AVFoundation -framework CoreMedia -framework CoreAudio \
  -o "$work/checks"
"$work/checks" | tee "$work/results.txt"
xcrun swiftc -O -D JARVIS_ECHO_TEST -import-objc-header "$src/AudioEcho/JarvisEchoDSP.h" \
  "$src/JarvisRecordingTimeline.swift" "$src/JarvisRecordingMixer.swift" \
  "$src/JarvisRecordingWriter.swift" ios/Jarvis/OfflineTests/JarvisRecordingMixerChecks.swift \
  "$work/echo.o" -framework Foundation -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework CoreAudio -o "$work/encode-checks"
JARVIS_RECORDING_ECHO_TEST=1 JARVIS_RECORDING_TEST_OUTPUT="$work" \
  "$work/encode-checks" | tee "$work/encode-results.txt"
