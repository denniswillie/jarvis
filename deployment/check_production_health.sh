#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${1:-${JARVIS_BASE_URL:-http://127.0.0.1:9000}}"
ENV_FILE="$SCRIPT_DIR/../backend/.env"

TOKEN="${JARVIS_STATIC_BEARER_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$ENV_FILE" ]]; then
  TOKEN="$(sed -n 's/^JARVIS_STATIC_BEARER_TOKEN=//p' "$ENV_FILE" | tail -n 1)"
fi
if [[ -z "$TOKEN" || "$TOKEN" == "replace_me" ]]; then
  echo "JARVIS_STATIC_BEARER_TOKEN is required for diagnostics." >&2
  exit 1
fi

echo "Waiting for Jarvis at $BASE_URL ..."
for _ in {1..45}; do
  if curl -fsS "$BASE_URL/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "$BASE_URL/health" >/dev/null
curl -fsS "$BASE_URL/ready" >/dev/null
DIAGNOSTICS="$(curl -fsS -H "Authorization: Bearer $TOKEN" "$BASE_URL/diagnostics/config")"
printf '%s' "$DIAGNOSTICS" | python3 -c '
import json, sys
payload = json.load(sys.stdin)
if payload.get("status") != "ready":
    print("Jarvis diagnostics are degraded:", ", ".join(payload.get("failed_checks", [])), file=sys.stderr)
    raise SystemExit(1)
print("Jarvis health, readiness, storage, and model configuration checks passed.")
'
