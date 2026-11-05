#!/usr/bin/env bash
# Paid, isolated acceptance probes using the existing backend's configured keys.
set -euo pipefail
production_adapter=false
if [ "${1:-}" = "--production-adapter" ]; then
  production_adapter=true
  shift
fi
if [ "$#" -lt 3 ]; then
  printf 'Usage: bash %s [--production-adapter] IMAGE_DIR SPEECH_DIR NEW_OUTPUT_DIR [GROUPS] [REPEATS]\n' "$0" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
images=$(cd "$1" && pwd)
speech=$(cd "$2" && pwd)
output=$3
groups=${4:-all}
repeats=${5:-3}
container=${JARVIS_BACKEND_CONTAINER:-deployment-jarvis-backend-1}
if [ -e "$output" ]; then
  printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
output_parent=$(cd "$(dirname "$output")" && pwd)
output_name=$(basename "$output")
image=$(docker inspect "$container" --format '{{.Image}}')
entry=(--entrypoint python "$image" -B -m evals.gemini_live.runner)
if [ "$production_adapter" = true ]; then
  # Pillow is an evaluation dependency, installed only in this disposable runner.
  entry=(--entrypoint sh "$image" -c '
python -m pip install --no-input --disable-pip-version-check --no-cache-dir --quiet --root-user-action=ignore --target /tmp/eval-deps Pillow==11.3.0 &&
PYTHONPATH=/tmp/eval-deps:$PYTHONPATH exec python -B -m evals.gemini_live.paired_probe "$@"
' --)
fi
# Keep credentials off argv, reports and disk. Never enable shell xtrace here.
docker exec "$container" python -B -c '
import json, sys
from app.settings import JarvisSettings
s = JarvisSettings()
keys = {"gemini": s.GEMINI_API_KEY, "openai": s.OPENAI_API_KEY,
        "openai_model": s.OPENAI_REALTIME_MODEL}
if sys.argv[1] == "true":
    keys["database"] = s.SUPABASE_DB_URL
    keys["settings"] = {name: getattr(s, name) for name in (
        "OPENAI_REALTIME_VOICE", "OPENAI_REALTIME_TRANSCRIPTION_MODEL",
        "GEMINI_LIVE_MODEL", "GEMINI_LIVE_VOICE", "GEMINI_LIVE_PREPARATION_TIMEOUT_SECONDS",
        "JARVIS_AUDIO_SAMPLE_RATE", "JARVIS_AUDIO_FORMAT",
    ) if hasattr(s, name)}
print(json.dumps(keys))
' "$production_adapter" | docker run --rm -i --read-only --tmpfs /tmp:rw,exec,nosuid,nodev \
  --mount "type=bind,source=$repo_root,target=/workspace,readonly" \
  --mount "type=bind,source=$images,target=/images,readonly" \
  --mount "type=bind,source=$speech,target=/speech,readonly" \
  --mount "type=bind,source=$output_parent,target=/results" \
  --workdir /workspace/backend --env PYTHONDONTWRITEBYTECODE=1 \
  "${entry[@]}" --keys-stdin \
  --images /images/frame-211-before-looks-correct.jpg /images/frame-272-before-final-diagnosis.jpg \
  --speech-dir /speech --output "/results/$output_name" --groups "$groups" --repeats "$repeats"
