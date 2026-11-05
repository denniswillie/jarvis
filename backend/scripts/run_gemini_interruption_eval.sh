#!/usr/bin/env bash
# Uses disposable owners in the configured database. Never deploys application code.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  printf 'Usage: %s NEW_OUTPUT_DIRECTORY [REPEATS]\n' "$0" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
output=$1
repeats=${2:-2}
if [ -e "$output" ]; then
  printf 'Output directory must not already exist.\n' >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
parent=$(cd "$(dirname "$output")" && pwd)
name=$(basename "$output")
container=${JARVIS_BACKEND_CONTAINER:-deployment-jarvis-backend-1}
image=$(docker inspect "$container" --format '{{.Image}}')
# Secrets travel directly through stdin, never shell arguments or result files.
docker exec "$container" python -B -c '
import json
from app.settings import JarvisSettings
s = JarvisSettings()
print(json.dumps({"gemini": s.GEMINI_API_KEY, "openai": s.OPENAI_API_KEY,
                  "openai_model": s.OPENAI_REALTIME_MODEL, "database": s.SUPABASE_DB_URL}))
' | docker run --rm -i --read-only --tmpfs /tmp:rw,nosuid,nodev \
  --mount "type=bind,source=$root,target=/workspace,readonly" \
  --mount "type=bind,source=$parent,target=/results" \
  --workdir /workspace/backend --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python "$image" -B -m evals.gemini_live.interruption_runner \
  --keys-stdin --output "/results/$name" --repeats "$repeats"
