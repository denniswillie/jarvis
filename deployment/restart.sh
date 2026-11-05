#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

docker compose -f docker-compose.production.yml up -d --no-deps --force-recreate jarvis-backend
./check_production_health.sh http://127.0.0.1:9000
