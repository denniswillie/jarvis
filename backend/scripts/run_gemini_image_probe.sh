#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ] || [ -e "$1" ]; then
  printf 'Usage: %s NEW_OUTPUT_DIRECTORY SPEECH_WAV\n' "$0" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
output=$1
speech_parent=$(cd "$(dirname "$2")" && pwd)
speech_name=$(basename "$2")
mkdir -p "$(dirname "$output")"
parent=$(cd "$(dirname "$output")" && pwd)
name=$(basename "$output")
container=${JARVIS_BACKEND_CONTAINER:-deployment-jarvis-backend-1}
image=$(docker inspect "$container" --format '{{.Image}}')
docker exec "$container" python -B -c '
import json
from app.settings import JarvisSettings
s = JarvisSettings()
print(json.dumps({"database": s.SUPABASE_DB_URL, "gemini": s.GEMINI_API_KEY}))
' | docker run --rm -i --read-only --tmpfs /tmp:rw,exec,nosuid,nodev \
  --mount "type=bind,source=$root,target=/workspace,readonly" \
  --mount "type=bind,source=$parent,target=/results" \
  --mount "type=bind,source=$speech_parent,target=/speech,readonly" \
  --workdir /workspace/backend --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint sh "$image" -c \
  'python -m pip install --disable-pip-version-check --no-cache-dir --quiet --no-input --root-user-action=ignore --target /tmp/image-deps Pillow==11.3.0 && PYTHONPATH=/tmp/image-deps:$PYTHONPATH python -B -m evals.gemini_live.image_probe "$@"' \
  sh --output "/results/$name" --speech "/speech/$speech_name"
