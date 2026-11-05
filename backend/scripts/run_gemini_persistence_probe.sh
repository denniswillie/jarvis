#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ] || [ -e "$1" ]; then
  printf 'Usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
output=$1
mkdir -p "$(dirname "$output")"
parent=$(cd "$(dirname "$output")" && pwd)
name=$(basename "$output")
container=${JARVIS_BACKEND_CONTAINER:-deployment-jarvis-backend-1}
image=$(docker inspect "$container" --format '{{.Image}}')
# The configured credential goes directly from the running backend to stdin.
docker exec "$container" python -B -c '
import json
from app.settings import JarvisSettings
print(json.dumps({"database": JarvisSettings().SUPABASE_DB_URL}))
' | docker run --rm -i --read-only --tmpfs /tmp:rw,nosuid,nodev \
  --mount "type=bind,source=$root,target=/workspace,readonly" \
  --mount "type=bind,source=$parent,target=/results" \
  --workdir /workspace/backend --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python "$image" -B -m evals.gemini_live.persistence_probe --output "/results/$name"
