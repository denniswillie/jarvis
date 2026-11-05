#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$(mktemp -d "$ROOT/.jarvis-dev/recording-pending-host.XXXXXX")"
printf 'Evidence: %s\n' "$OUT"
# One bounded host compile/run; never invokes a simulator, device or service.
/usr/bin/python3 - "$ROOT" "$OUT" <<'PY'
import os
import signal
import subprocess
import sys
import time

root, out = sys.argv[1:]
services = root + '/ios/Jarvis/Jarvis/Services/'
sources = [root + '/ios/Jarvis/Jarvis/Models/JarvisModels.swift'] + [
    services + name for name in ('JarvisRecordingControl.swift', 'JarvisRecordingWriter.swift',
                                'JarvisRecordingManifest.swift', 'JarvisRecordingPendingStore.swift')]
sources += [root + '/ios/Jarvis/OfflineTests/JarvisRecordingPendingChecks.swift']
deadline = time.monotonic() + 175

def run(args, log, env=None):
    with open(out + '/' + log, 'wb') as output:
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit('Host checks timed out; evidence retained: ' + out)
    with open(out + '/' + log) as output:
        print(output.read(), end='')
    if code:
        raise SystemExit(code)

run(['xcrun', 'swiftc', '-swift-version', '6', '-framework', 'Foundation', '-framework', 'AVFoundation',
     '-framework', 'CoreMedia', '-framework', 'CoreVideo', '-framework', 'CoreAudio', *sources,
     '-o', out + '/checks'], 'compiler.log')
run([out + '/checks'], 'checks.log', dict(os.environ, JARVIS_RECORDING_TEST_OUTPUT=out))
PY
