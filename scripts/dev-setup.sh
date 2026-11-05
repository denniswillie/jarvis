#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/backend/.env"
PRODUCTION_COMPOSE_FILE="$ROOT_DIR/deployment/docker-compose.production.yml"
DEVELOPMENT_COMPOSE_FILE="$ROOT_DIR/deployment/docker-compose.development.yml"
DEV_CONFIG_DIR="$ROOT_DIR/.jarvis-dev"
IOS_OVERRIDE_FILE="$DEV_CONFIG_DIR/ios-overrides.env"
NGROK_PID_FILE="/tmp/jarvis-ngrok.pid"
NGROK_LOG_FILE="/tmp/jarvis-ngrok.log"
LOCAL_BASE_URL="http://127.0.0.1:9000"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

stop_ngrok() {
  if [[ -f "$NGROK_PID_FILE" ]]; then
    local pid
    pid="$(cat "$NGROK_PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$NGROK_PID_FILE"
  fi

  pkill -f 'ngrok http 9000' 2>/dev/null || true
}

compose() {
  docker compose \
    --env-file "$ENV_FILE" \
    -f "$PRODUCTION_COMPOSE_FILE" \
    -f "$DEVELOPMENT_COMPOSE_FILE" \
    "$@"
}

read_static_bearer_token() {
  python3 - "$ENV_FILE" <<'PY'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() != "JARVIS_STATIC_BEARER_TOKEN":
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(value, end="")
    break
PY
}

wait_for_url() {
  local label="$1"
  local url="$2"
  shift 2

  for _ in {1..60}; do
    if curl --silent --show-error --fail "$@" "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  printf 'Timed out waiting for %s at %s\n' "$label" "$url" >&2
  return 1
}

discover_ngrok_url() {
  python3 <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as response:
    payload = json.load(response)

for tunnel in payload.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    config = tunnel.get("config", {})
    if public_url.startswith("https://") and str(config.get("addr", "")).endswith(":9000"):
        print(public_url, end="")
        break
PY
}

cleanup_after_failure() {
  local exit_code=$?
  printf '\nJarvis development setup failed. Cleaning up partial state...\n' >&2
  compose down >/dev/null 2>&1 || true
  stop_ngrok
  rm -f "$IOS_OVERRIDE_FILE"
  exit "$exit_code"
}

trap cleanup_after_failure ERR

require_command curl
require_command docker
require_command ngrok
require_command pkill
require_command python3

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing %s. Create it from backend/.env.example first.\n' "$ENV_FILE" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  printf 'Docker is not running. Start Docker Desktop and retry.\n' >&2
  exit 1
fi

STATIC_BEARER_TOKEN="$(read_static_bearer_token)"
if [[ -z "$STATIC_BEARER_TOKEN" || "$STATIC_BEARER_TOKEN" == "replace_me" ]]; then
  printf 'JARVIS_STATIC_BEARER_TOKEN must be set in %s.\n' "$ENV_FILE" >&2
  exit 1
fi

printf 'Restarting the Jarvis ngrok tunnel on port 9000...\n'
stop_ngrok
rm -f "$NGROK_LOG_FILE"
ngrok http 9000 --log stdout >"$NGROK_LOG_FILE" 2>&1 &
NGROK_PID=$!
printf '%s\n' "$NGROK_PID" >"$NGROK_PID_FILE"

PUBLIC_BASE_URL=""
for _ in {1..30}; do
  PUBLIC_BASE_URL="$(discover_ngrok_url 2>/dev/null || true)"
  if [[ -n "$PUBLIC_BASE_URL" ]]; then
    break
  fi
  if ! kill -0 "$NGROK_PID" 2>/dev/null; then
    printf 'ngrok exited before creating a tunnel. See %s.\n' "$NGROK_LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done

if [[ -z "$PUBLIC_BASE_URL" ]]; then
  printf 'Timed out waiting for the ngrok tunnel. See %s.\n' "$NGROK_LOG_FILE" >&2
  exit 1
fi

export JARVIS_DEV_PUBLIC_BASE_URL="$PUBLIC_BASE_URL"

printf 'Starting Jarvis development Docker services...\n'
compose up -d --build

wait_for_url 'local Jarvis health' "$LOCAL_BASE_URL/health"
wait_for_url 'local Jarvis readiness' "$LOCAL_BASE_URL/ready"
wait_for_url \
  'public Jarvis health' \
  "$PUBLIC_BASE_URL/health" \
  --header 'ngrok-skip-browser-warning: true'
wait_for_url \
  'authenticated Jarvis diagnostics' \
  "$PUBLIC_BASE_URL/diagnostics/config" \
  --header 'ngrok-skip-browser-warning: true' \
  --header "Authorization: Bearer $STATIC_BEARER_TOKEN"

mkdir -p "$DEV_CONFIG_DIR"
umask 077
{
  printf 'BACKEND_BASE_URL=%s\n' "$PUBLIC_BASE_URL"
  printf 'STATIC_BEARER_TOKEN=%s\n' "$STATIC_BEARER_TOKEN"
} >"$IOS_OVERRIDE_FILE"

trap - ERR

printf '\nJarvis development setup is ready.\n'
printf 'Local backend:          %s\n' "$LOCAL_BASE_URL"
printf 'Public backend:         %s\n' "$PUBLIC_BASE_URL"
printf 'Realtime WebSocket:     %s\n' "${PUBLIC_BASE_URL/https:/wss:}/realtime/ws"
printf 'Calendar callback URL:  %s\n' "$PUBLIC_BASE_URL/oauth/google/callback"
printf 'iOS Debug config:       %s\n' "$IOS_OVERRIDE_FILE"
printf 'ngrok PID file:         %s\n' "$NGROK_PID_FILE"
printf 'ngrok log file:         %s\n' "$NGROK_LOG_FILE"
printf '\nRebuild and run the Debug iOS app so Xcode bundles the generated local config.\n'
