#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BASE_URL="${1:-${JARVIS_BASE_URL:-https://your-jarvis.example.invalid}}"
ENV_FILE="$SCRIPT_DIR/../backend/.env"

TOKEN="${JARVIS_STATIC_BEARER_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$ENV_FILE" ]]; then
  TOKEN="$(sed -n 's/^JARVIS_STATIC_BEARER_TOKEN=//p' "$ENV_FILE" | tail -n 1)"
fi
if [[ -z "$TOKEN" || "$TOKEN" == "replace_me" ]]; then
  echo "JARVIS_STATIC_BEARER_TOKEN is required." >&2
  exit 1
fi

START_RESPONSE="$(curl -fsS -X POST "$BASE_URL/sessions/start" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Production websocket check","resume_active":false}')"
SESSION_ID="$(printf '%s' "$START_RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["session"]["id"])')"
WS_URL="$(printf '%s' "$START_RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["realtime"]["audio_websocket_url"])')"
cleanup() {
  curl -fsS -X POST "$BASE_URL/sessions/$SESSION_ID/close" \
    -H "Authorization: Bearer $TOKEN" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose -f docker-compose.production.yml exec -T \
  -e JARVIS_CHECK_WS_URL="$WS_URL&client_kind=phone_audio" \
  jarvis-backend python - <<'PY'
import json
import os
from websockets.sync.client import connect

url = os.environ["JARVIS_CHECK_WS_URL"]
token = os.environ["JARVIS_STATIC_BEARER_TOKEN"]
with connect(
    url,
    additional_headers={"Authorization": f"Bearer {token}"},
    open_timeout=15,
    close_timeout=5,
) as websocket:
    event = json.loads(websocket.recv(timeout=20))
    if event.get("type") == "error":
        raise SystemExit(f"Realtime check failed: {event.get('code')}")
    print(f"Realtime websocket connected; first event={event.get('type', 'unknown')}")
PY
