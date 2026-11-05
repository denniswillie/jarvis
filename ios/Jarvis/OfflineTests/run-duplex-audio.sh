#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/duplex-audio-host.XXXXXX")"
SDK_ROOT="$ROOT/.jarvis-dev/xcode-lean-note/SourcePackages/artifacts/webrtc/WebRTC/WebRTC.xcframework"
SIM_FRAMEWORK="$SDK_ROOT/ios-x86_64_arm64-simulator"
HEADERS="$SDK_ROOT/ios-arm64/WebRTC.framework/Headers"
SERVICES="$ROOT/ios/Jarvis/Jarvis/Services"
CHECKS="$ROOT/ios/Jarvis/OfflineTests"
printf 'Evidence: %s\n' "$OUT"
shasum -a 256 "$SERVICES/JarvisDuplexAudioDevice.swift" "$SERVICES/JarvisDuplexAudioIO.h" \
    "$SERVICES/JarvisDuplexAudioIO.m" "$CHECKS/JarvisDuplexTestSupport.h" \
    "$SERVICES/JarvisMediaPCM.h" "$SERVICES/JarvisRecordingAudioSource.swift" "$CHECKS/JarvisMediaPCMChecks.c" \
    "$CHECKS/JarvisDuplexTestSupport.m" "$CHECKS/JarvisDuplexAudioChecks.swift" \
    "$HEADERS/RTCAudioDevice.h" "$HEADERS/RTCMacros.h" > "$OUT/sources.sha256"
# The macOS bundle does not export the custom device protocol. Import its
# exact pinned iOS declarations as a header-only host module; do not pretend
# the host checks exercise WebRTC's native ADM or real iOS hardware.
mkdir -p "$OUT/include/WebRTC"
ln -s "$HEADERS/RTCAudioDevice.h" "$OUT/include/WebRTC/RTCAudioDevice.h"
ln -s "$HEADERS/RTCMacros.h" "$OUT/include/WebRTC/RTCMacros.h"
printf 'module WebRTC [system] { umbrella "WebRTC" export * }\n' > "$OUT/include/module.modulemap"
xcrun clang -std=c11 -Wall -Wextra "$CHECKS/JarvisMediaPCMChecks.c" \
    -o "$OUT/media-checks" > "$OUT/media-compile.log" 2>&1
"$OUT/media-checks" | tee "$OUT/media-checks.log"
xcrun clang -DDEBUG=1 -fobjc-arc -fblocks -I "$OUT/include" \
    -c "$CHECKS/JarvisDuplexTestSupport.m" -o "$OUT/checks.o" > "$OUT/objc.log" 2>&1
xcrun swiftc -swift-version 6 -D DEBUG -I "$OUT/include" \
    -framework Foundation -framework AudioToolbox -framework AVFoundation \
    -import-objc-header "$CHECKS/JarvisDuplexTestSupport.h" \
    "$SERVICES/JarvisDuplexAudioDevice.swift" "$SERVICES/AudioEcho/JarvisRecordingEchoDiagnostic.swift" "$SERVICES/JarvisRecordingAudioSource.swift" "$CHECKS/JarvisDuplexAudioChecks.swift" \
    "$OUT/checks.o" -o "$OUT/checks" > "$OUT/swift.log" 2>&1
"$OUT/checks" | tee "$OUT/checks.log"
SIM_SDK="$(xcrun --sdk iphonesimulator --show-sdk-path)"
xcrun clang -target arm64-apple-ios17.0-simulator -isysroot "$SIM_SDK" \
    -DDEBUG=1 -fobjc-arc -fblocks -F "$SIM_FRAMEWORK" -fsyntax-only \
    "$SERVICES/JarvisDuplexAudioIO.m" > "$OUT/ios-objc.log" 2>&1
xcrun swiftc -swift-version 6 -D DEBUG -target arm64-apple-ios17.0-simulator \
    -sdk "$SIM_SDK" -F "$SIM_FRAMEWORK" -typecheck \
    -import-objc-header "$SERVICES/JarvisDuplexAudioIO.h" \
    "$SERVICES/JarvisDuplexAudioDevice.swift" "$SERVICES/AudioEcho/JarvisRecordingEchoDiagnostic.swift" "$SERVICES/JarvisRecordingAudioSource.swift" > "$OUT/ios-swift.log" 2>&1
shasum -a 256 -c "$OUT/sources.sha256" > "$OUT/source-check.log"
printf 'PASS: host checks and real iOS SDK typecheck; not hardware or app integration acceptance\n'
