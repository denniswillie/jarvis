# Jarvis production operations

The maintained personal deployment is the focused release: one API, one frontier
worker and the existing Hindsight service. Follow
[focused operations](../planning_docs/second_brain_focused/OPERATIONS.md) for the
complete Compose invocation, selected images, migrations and recovery.

Do not use the former single-backend deploy/restart instructions for this
installation. They omit the worker and selected-memory overlays. Old memory-job
and embedding counts are historical diagnostics, not readiness gates for
Hindsight. The legacy memory pipeline smoke script and wrapper are retired.

## Readiness

From the repository root, with the existing bearer token in the environment:

```sh
bash deployment/check_focused_release.sh
bash deployment/check_focused_release.sh --require-push
```

Set JARVIS_BASE_URL for the intended host and JARVIS_PYTHON for the backend virtual
environment if necessary. The check verifies HTTP/configuration readiness, absence
of the legacy memory writers and a live frontier worker lease. It reports disabled
or unverified push explicitly. It does not prove model accuracy, Hindsight
ingestion, physical playback or APNs delivery; use the focused acceptance evidence.

## Public endpoint and TLS

The public endpoint remains https://jarvis.trygitbio.com. Keep the configured
Nginx websocket upgrade headers and the intended tunnel/upstream. The focused
personal stack currently runs on the Mac; do not deploy a second independent
writer stack to EC2. After changing the proxy, use check_production_health.sh and
check_realtime_ws.sh against the public URL as connectivity checks only.

## Recovery

Keep the source/task database, private Hindsight data and release selectors.
Restart the same selected API/worker images with the focused Compose command.
Do not re-enable old memory jobs, rerun the fresh-install schema, or roll back
to a pre-migration image as an automatic repair. Preserve backend/.env and private
secrets. The focused runbook describes intentional downtime and scoped recovery.
