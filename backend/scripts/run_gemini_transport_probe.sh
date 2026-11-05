#!/usr/bin/env bash
# Paid native-media probe; credentials stay on stdin, not argv, reports, or disk.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  printf 'Usage: bash %s SPEECH_DIR NEW_OUTPUT_DIR [CASES] [REPEATS]\n' "$0" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
speech=$(cd "$1" && pwd)
output=$2
cases=${3:-marker_early,marker_end,marker_pause,explanation}
repeats=${4:-1}
container=${JARVIS_BACKEND_CONTAINER:-deployment-jarvis-backend-1}
if [ -e "$output" ]; then
  printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
output_parent=$(cd "$(dirname "$output")" && pwd)
output_name=$(basename "$output")
image=$(docker inspect "$container" --format '{{.Image}}')
docker exec "$container" python -B -c '
import json
from app.settings import JarvisSettings
settings = JarvisSettings()
print(json.dumps({"gemini": settings.GEMINI_API_KEY}))
' | docker run --rm -i --read-only --tmpfs /tmp:rw,nosuid,nodev \
  --mount "type=bind,source=$repo_root,target=/workspace,readonly" \
  --mount "type=bind,source=$speech,target=/speech,readonly" \
  --mount "type=bind,source=$output_parent,target=/results" \
  --workdir /workspace/backend --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python "$image" -B -m evals.gemini_live.transport_probe \
  --speech-dir /speech --output "/results/$output_name" --cases "$cases" --repeats "$repeats"
