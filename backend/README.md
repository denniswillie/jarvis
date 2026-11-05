# Jarvis Backend

Jarvis is a standalone personal companion backend. It is intentionally separate
from GitBio while reusing the same practical architecture style: a small
FastAPI service, explicit settings, Docker deployment, and phase-by-phase
feature growth.

## Phase 1 scope

Phase 1 includes only:

- FastAPI app bootstrap.
- Core settings and console logging.
- `/health` and `/ready` endpoints.
- Dockerfile.
- Minimal EC2 deployment scaffolding.

It does not include database schema, authentication, realtime audio, visual
ingestion, model calls, tools, or iOS app code.

## Local setup

```bash
cd jarvis/backend
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 9000
```

Health check:

```bash
curl http://127.0.0.1:9000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "jarvis-backend",
  "environment": "development"
}
```

## Docker

```bash
cd jarvis/backend
docker build -t jarvis-backend .
docker run --rm --env-file .env -p 9000:9000 jarvis-backend
```

## Environment

Only core settings are required in Phase 1. Provider-specific values such as
Supabase, OpenAI, Gemini, and Google OAuth may remain unset or set to
`replace_me` until the feature that needs them is implemented.

## Perception and memory baseline

The migration baseline is isolated under `evals/perception_memory`. It freezes
current session, realtime transport, reminder, and visual-memory tool contracts
without changing production behavior.

Install the optional fixture dependency and regenerate deterministic JPEGs when
fixture definitions intentionally change:

```bash
cd jarvis/backend
pip install -e '.[eval]'
python -m evals.perception_memory.generate_fixtures
```

Run the deterministic baseline:

```bash
python -m evals.perception_memory.baseline_runner \
  --output evals/perception_memory/reports/pre_memory_architecture.json
```

This run exercises production visual validation, temporal perception, context,
interruption dedupe, and the hierarchical memory-retrieval service with provider
and storage boundaries replaced by deterministic in-memory fixtures. Historical
recall is represented as retrieval of a durable experience event rather than a
search over raw frame captions. The preserved comparison baseline intentionally
continues to report `omitted_caption_detail` and `voice_only_memory`. The
post-cutover architecture runner disables those two simulated legacy gaps while
using the same production boundaries, fixtures, scoring, and frozen report for
regression comparison.

Live results are opt-in. Capture the documented current-view, historical-recall,
and current-frame-capture turns through the running Jarvis app, record their
observations using the schema described in the eval README, then run:

```bash
python -m evals.perception_memory.baseline_runner \
  --live \
  --live-results /path/to/live-observations.json \
  --output evals/perception_memory/reports/live-baseline.json
```

The runner itself never starts a production session or mutates production state
in deterministic mode. Live inputs use structured provider/tool trace events as
described in `evals/perception_memory/README.md`.

## Post-cutover perception and memory evaluation

Stage 16A adds deterministic architecture gates and compares them with the frozen
pre-memory-architecture report:

```bash
python -m evals.perception_memory.new_architecture_runner \
  --output evals/perception_memory/reports/latest_architecture.json
python -m evals.perception_memory.compare_reports \
  evals/perception_memory/reports/pre_memory_architecture.json \
  evals/perception_memory/reports/latest_architecture.json
```

The architecture runner fails when any scenario fails, measured correctness drops
below 100%, interruption dedupe regresses, the bounded sensory backlog is exceeded,
or deterministic latency exceeds its explicit budget. Real-provider latency and
cost gates are intentionally deferred to Stage 16B.

Score a complete, explicitly captured live trace with the same architecture gates:

```bash
python -m evals.perception_memory.new_architecture_runner \
  --live \
  --live-results /tmp/jarvis-live-observations.json \
  --output /tmp/jarvis-live-architecture-report.json
```

Live mode never starts a session implicitly. The input must be a disposable trace
captured through the real iOS, OpenAI Realtime, Gemini, Supabase storage, and
memory-tool paths. Reports retain provider model IDs, usage, tool calls, trace ID,
timestamps, and measured latency without recording credentials.

With a backend running, inspect aggregate memory health without returning memory
content, credentials, image bytes, or storage paths:

```bash
python scripts/smoke_perception_memory.py \
  --base-url http://127.0.0.1:9000 \
  --token "$JARVIS_STATIC_BEARER_TOKEN" \
  --attempts 10 \
  --interval-seconds 2
python scripts/inspect_memory_state.py
```

Failure recovery remains covered by production-boundary tests in the complete test
suite. Run it with `JARVIS_TEST_DB_URL` set to a disposable or development database:

```bash
JARVIS_TEST_DB_URL="$SUPABASE_DB_URL" python -m pytest -q
```
