#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TAIL_LINES="${TAIL_LINES:-200}"
docker compose -f docker-compose.production.yml logs -f --tail="$TAIL_LINES" jarvis-backend
