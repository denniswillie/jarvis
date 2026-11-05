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
  rm -f "$NGROK_LOG_FILE"
}

printf 'Stopping Jarvis development services...\n'

if command -v docker >/dev/null 2>&1 && [[ -f "$ENV_FILE" ]]; then
  docker compose \
    --env-file "$ENV_FILE" \
    -f "$PRODUCTION_COMPOSE_FILE" \
    -f "$DEVELOPMENT_COMPOSE_FILE" \
    down
fi

stop_ngrok
rm -f "$IOS_OVERRIDE_FILE"
rmdir "$DEV_CONFIG_DIR" 2>/dev/null || true

printf 'Jarvis development services are stopped and the generated iOS config is removed.\n'
printf 'The next Debug build will clear local backend credentials from the app bundle.\n'
