#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ../backend/.env ]]; then
  echo "Missing ../backend/.env. Create it from env.production.example." >&2
  exit 1
fi

docker compose -f docker-compose.production.yml config --quiet
docker compose -f docker-compose.production.yml up -d --build
./check_production_health.sh http://127.0.0.1:9000
