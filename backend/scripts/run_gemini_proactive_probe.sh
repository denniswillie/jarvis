#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ] || [ -e "$1" ]; then
  printf 'Usage: %s NEW_OUTPUT_DIRECTORY SPEECH_DIRECTORY\n' "$0" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
output=$1
speech=$(cd "$2" && pwd)
mkdir -p "$(dirname "$output")"
parent=$(cd "$(dirname "$output")" && pwd)
name=$(basename "$output")
container=${JARVIS_BACKEND_CONTAINER:-deployment-jarvis-backend-1}
image=$(docker inspect "$container" --format '{{.Image}}')
# Keys travel directly to the isolated runner's stdin, never to reports or argv.
docker exec "$container" python -B -c '
import json
from app.settings import JarvisSettings
s = JarvisSettings()
print(json.dumps({"database": s.SUPABASE_DB_URL, "gemini": s.GEMINI_API_KEY}))
' | docker run --rm -i --read-only --tmpfs /tmp:rw,nosuid,nodev \
  --mount "type=bind,source=$root,target=/workspace,readonly" \
  --mount "type=bind,source=$parent,target=/results" \
  --mount "type=bind,source=$speech,target=/speech,readonly" \
  --workdir /workspace/backend --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python "$image" -B -m evals.gemini_live.proactive_probe \
  --output "/results/$name" --speech-dir /speech
