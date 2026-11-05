# Source-backed Hindsight deployment

Current personal deployment: the private service was prepared and checked on 2026-09-15. Exact image identities, synthetic restart/cleanup evidence, private release paths, operating commands and remaining API/worker connectivity are recorded in ../../planning_docs/second_brain/reports/M04_PRIVATE_HINDSIGHT_DEPLOYMENT_REPORT.md. The live Jarvis backend has not been switched to it.

This opt-in stack is separate from the live Jarvis backend. Starting it does not enable knowledge intake, apply Jarvis's migrations, backfill personal history, or switch the live memory authority. Those remain explicit M03/M04 gates.

## Privacy and readiness

The image pins hindsight-api-slim 0.9.2 and runs Jarvis's small checked launcher before the maintained Hindsight CLI. The launcher checks the installed version, disables payload tracing/export and unsupported vendor history snapshots, requires API-key authentication and dedicated PostgreSQL, and verifies bounded worker/model capacity.

The Jarvis adapter separately reads effective bank configuration before retention admission and recall: audit logging must be false and original document storage must be true. Saved mental models remain forbidden. Deletion is not blocked merely because an unsafe bank policy needs correction.

Hindsight's bank configuration API deliberately excludes static server fields. Therefore a successful API-key check or bank check does not certify an arbitrary remote service's tracing policy. Only the checked, reviewed deployment is supported for real intake; do not bypass the launcher or point Jarvis at an unverified instance.

The disabled settings prevent new vendor history/log copies; they do not erase pre-existing rows or backups. Use a new empty vendor database for this migration. Previously populated services require an explicit erasure/rebuild plan. Cloud model-provider retention is a separate operational boundary, not disabled by these switches.

## Secrets and startup

Use a permission-restricted, untracked absolute directory containing three one-line files: db-password, api-key (at least 24 characters), and openai-api-key. Keep the host directory mode 0700. The mounted files must be readable by service UID 10001; for local Compose bind-mounted secrets, readable files inside that private directory are appropriate. Compose does not remap bind-mounted secret-file ownership, so host-only mode 0600 files owned by a different UID can prevent startup. No secret value belongs in YAML, shell history, model context, or a committed env file.

Set JARVIS_HINDSIGHT_SECRET_DIR to that directory, then use an explicit project name:

```sh
docker compose -p jarvis-knowledge -f deployment/docker-compose.knowledge.yml build
docker compose -p jarvis-knowledge -f deployment/docker-compose.knowledge.yml up -d
docker compose -p jarvis-knowledge -f deployment/docker-compose.knowledge.yml ps
```

Only the authenticated API is published, on 127.0.0.1:8888. PostgreSQL has no published port. Host access can use the loopback origin. A containerized backend must use a reviewed private network connection or secured host route; localhost inside the backend container is not this API. Do not expose port 8888 publicly to work around that boundary.

The health check confirms authenticated API availability, not model quality, complete memory ingestion, or a live cutover. Record the built image ID and native acceptance evidence before release. The base-image tag and transitive package resolution may move; do not equate the version label alone with a tested image.


## Admission and overload

Jarvis limits each owner's outstanding native retention admissions with SECOND_BRAIN_KNOWLEDGE_NATIVE_INFLIGHT_LIMIT (default 8; range 1-64). This is separate from Hindsight's execution slots. A claim reserves capacity before sending a request, so concurrent workers cannot all fill the same slot.

At capacity, new originals and their durable intents remain in Jarvis; current-source reads do not wait for Hindsight. Existing operation reconciliation and deletion are still eligible. The canonical source journal is not discarded or artificially capped to hide a backlog. Its storage growth needs operational monitoring.

Knowledge diagnostics expose pending counts, oldest pending time, recorded failures, and native_admissions with limit/reserved/available/at_capacity. Reservations are conservative: an uncertain dispatched operation still consumes capacity after automatic retries stop. Investigate its recorded operation and source state or complete source cleanup; do not reset dispatch counts or mark unknown work complete merely to unblock ingestion. A known native terminal failure or verified source cleanup releases the reservation.

## Stop and rollback boundary

```sh
docker compose -p jarvis-knowledge -f deployment/docker-compose.knowledge.yml stop
```

Stop/drain Jarvis knowledge intake before stopping an active memory service. Do not remove hindsight-data as a routine rollback. The canonical source journal, corrections and tombstones remain authoritative for a rebuild. Never replay an older vendor backup before applying the current privacy barriers. M04 establishes the actual source/image rollback tuple; M05 defines retirement.

Raw provider request tracing, OpenTelemetry payload export, observation history and mental-model history stay disabled until their full deletion/restore lifecycle is explicitly implemented and accepted. Jarvis's canonical original history is independent of these vendor snapshots.

## Private container clients

The stack creates the internal network selected by JARVIS_KNOWLEDGE_NETWORK, default jarvis-knowledge-clients. Only hindsight-api joins it; hindsight-db remains on the separate database/egress network. The host-loopback diagnostic port is unchanged.

Combine docker-compose.api-knowledge.yml with the production API base, or docker-compose.frontier-knowledge.yml with the worker base. Each retains its normal egress network and joins the external client network. Start the Hindsight stack first and use the same network name and JARVIS_HINDSIGHT_SECRET_DIR for the consumers.

The overlays use the exact http://hindsight-api:8888 origin plus its explicit private HTTP opt-in, and mount the existing API key. Do not also configure an inline SECOND_BRAIN_HINDSIGHT_API_KEY. This trusts the reviewed local Docker network, not arbitrary remote DNS; use HTTPS outside that boundary.

JARVIS_KNOWLEDGE_ENABLED defaults to false. Set the actual owned JARVIS_KNOWLEDGE_AREA_ID and finish role/source readiness before enabling. The overlays do not enable top-level capabilities or select memory authority. Container connection acceptance, candidate image identities and remaining activation gates are recorded in ../../planning_docs/second_brain/reports/M04_PRIVATE_CONTAINER_CONNECTION_REPORT.md.
