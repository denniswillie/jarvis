#!/bin/bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SOURCE/../../.." && pwd)"
APP="$ROOT/.jarvis-dev/soloist/JarvisSoloistCapture.app"
mkdir -p "$APP/Contents/MacOS"
install -m 644 "$SOURCE/Info.plist" "$APP/Contents/Info.plist"
xcrun --sdk macosx clang -std=c11 -fobjc-arc -O2 -Wall -Wextra -Werror \
  -mmacosx-version-min=14.2 -framework Foundation -framework CoreAudio \
  "$SOURCE/capture.m" -o "$APP/Contents/MacOS/JarvisSoloistCapture"
codesign --force --sign - --timestamp=none "$APP"
printf '%s\n' "$APP/Contents/MacOS/JarvisSoloistCapture"
