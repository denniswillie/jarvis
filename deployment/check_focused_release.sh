#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${JARVIS_BASE_URL:-http://127.0.0.1:9000}"

if [[ -z "${JARVIS_STATIC_BEARER_TOKEN:-}" ]]; then
  echo "JARVIS_STATIC_BEARER_TOKEN is required" >&2
  exit 2
fi

cd "$ROOT_DIR/backend"
"${JARVIS_PYTHON:-python}" scripts/smoke_focused_release.py \
  --base-url "$BASE_URL" "$@"
