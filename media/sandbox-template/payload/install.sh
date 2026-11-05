#!/bin/bash
# Build-time only. No credentials, media, runtime bootstrap or npx installers.
set -euo pipefail
test "$(id -u)" = 0
test "$(dpkg --print-architecture)" = amd64
cd /opt/jarvis-render
value() { python3 -c 'import json,sys; print(json.load(open("pins.json"))[sys.argv[1]][sys.argv[2]])' "$1" "$2"; }
NODE_URL=$(value node url)
NODE_SHA=$(value node sha256)
CHROME_URL=$(value chrome url)
CHROME_SHA=$(value chrome sha256)
CHROME_GENERATION=$(value chrome generation)
test "${#NODE_SHA}" = 64
test "${#CHROME_SHA}" = 64
curl --proto '=https' --fail --silent --show-error --connect-timeout 10 --max-time 120 --retry 0 --max-filesize 60000000 "$NODE_URL" -o /tmp/jarvis-node.tar.xz
printf '%s  %s\n' "$NODE_SHA" /tmp/jarvis-node.tar.xz | sha256sum --check --status
mkdir node
tar -xJf /tmp/jarvis-node.tar.xz --directory node --strip-components=1 --no-same-owner
export PATH=/opt/jarvis-render/node/bin:/usr/bin:/bin
test "$(node --version)" = "v$(value node version)"
curl --proto '=https' --fail --silent --show-error --connect-timeout 10 --max-time 120 --retry 0 --max-filesize 210000000 "$CHROME_URL?generation=$CHROME_GENERATION" -o /tmp/jarvis-chrome.zip
test "$(stat -c %s /tmp/jarvis-chrome.zip)" = "$(value chrome size_bytes)"
printf '%s  %s\n' "$CHROME_SHA" /tmp/jarvis-chrome.zip | sha256sum --check --status
mkdir chrome
unzip -q /tmp/jarvis-chrome.zip -d chrome
cd clip
npm ci --ignore-scripts --include=optional --no-audit --no-fund --fetch-retries=0 --fetch-timeout=30000 --cache=/tmp/jarvis-npm-cache
npm test
cd ..
python3 inventory.py --verify-source
test -x clip/node_modules/@remotion/compositor-linux-x64-gnu/remotion
test -x clip/node_modules/@remotion/compositor-linux-x64-gnu/ffmpeg
test -x clip/node_modules/@remotion/compositor-linux-x64-gnu/ffprobe
test -x chrome/chrome-linux64/chrome
ldd chrome/chrome-linux64/chrome > /tmp/jarvis-chrome-libraries.txt
! grep -q 'not found' /tmp/jarvis-chrome-libraries.txt
ldd clip/node_modules/@remotion/compositor-linux-x64-gnu/ffmpeg > /tmp/jarvis-ffmpeg-libraries.txt
! grep -q 'not found' /tmp/jarvis-ffmpeg-libraries.txt
# E2B finalization recursively makes /usr/local writable. Trusted execution stays
# inside our sealed tree; the harness already uses absolute bundled tool paths.
chmod 0555 /opt/jarvis-render/render-clip
rm -rf /tmp/jarvis-npm-cache
rm -f /tmp/jarvis-node.tar.xz /tmp/jarvis-chrome.zip /tmp/jarvis-chrome-libraries.txt /tmp/jarvis-ffmpeg-libraries.txt
# Normalize the entire pinned payload, including distribution metadata. Removing
# write bits alone leaves upstream0600 files unreadable to the worker.
/usr/bin/timeout 20s /usr/bin/python3 -I -S -B /opt/jarvis-render/package_access.py --seal
/usr/bin/python3 -I -S -B /opt/jarvis-render/inventory.py --write
# Separate worker-equivalent access proof, still inside this install's420s cap.
/usr/bin/timeout 35s /usr/bin/python3 -I -S -B /opt/jarvis-render/package_access.py --worker-check
