# Explicit schema administration

Run from the backend directory with the intended release installed:

```bash
python -m app.migrations check
python -m app.migrations apply
```

`check` is read-only. It checks the version/name/checksum ledger, baseline table,
and required foundation table presence. It is not a full drift auditor, provider
test, or proof of end-user capability readiness. `apply` takes an advisory lock
and commits all pending migrations and their ledger entries atomically. A changed
applied migration, missing earlier version, or unknown newer version fails closed.
Repair drift through a reviewed forward migration, never by editing a ledger to
pretend a change ran. Run readiness after applying a migration.

`init` installs into an empty schema only. The original baseline portion of
`app/schema.sql` precedes its generated marker; versioned SQL follows it.
`render` produces the complete fresh-install SQL with exact migration checksums.
Always render to a temporary file and then replace the canonical artifact.
Migrations are packaged in wheels together with `app/schema.sql`.

The CLI returns sanitized error types/SQLSTATE rather than potentially sensitive
database exception text. No migration executes during import, startup, or a
model tool call. Capabilities remain disabled by default.

## Deployment and pooled-connection lifecycle

Schema administration and runtime pools have separate owners. The migration
CLI opens a dedicated connection and closes it when finished. Backend and worker
startup create their own pools; neither startup nor voice delivery applies
migrations. Install reviewed migrations before starting the new runtime.

For a deployment replacing a running release:

1. Stop new admission and quiesce the old backend/workers. Settle or preserve
   delivery attempts through their existing shutdown/recovery paths. A submitted
   or partially forwarded attempt stays ambiguous unless actual delivery evidence
   resolves it; do not turn it into a safe replay or cancel its background task.
2. Finish or roll back open transactions and release every borrowed connection.
   Await pool closure before migration. Pool closure waits for outstanding
   borrowers; a timeout is a deployment failure, not permission to migrate under
   active voice/work transactions. Stop or drain every process using that schema.
3. Use the explicit administrative CLI to apply migrations and run readiness.
   A failed or incomplete readiness result must not enable delivery.
4. Start the new runtime with fresh pools and repositories. Never reuse references
   to the closed pre-upgrade pools. Recover durable attempts under the existing
   source, generation and quiet-policy fences; do not replay arbitrary failed work.

If an administrative harness deliberately retains a quiesced pool across DDL,
it must release all pre-upgrade borrowers and explicitly expire connections with
`await pool.expire_connections()` before acquiring post-upgrade connections.
Expiry is not a substitute for draining: it does not reclaim a connection still
borrowed by active work. Closing/recreating pools or restarting their owning
processes is the normal deployment contract. The migration runner does not locate
or expire runtime pools, and there is no automatic hot-migration/retry hook in
voice handling. In particular, never catch `InvalidCachedStatementError` and
blindly replay an arbitrary in-transaction operation after DDL changes a row shape.

Migration 42 adds attention-intent columns, so warm pre-upgrade `SELECT *`
statements require this lifecycle even when the migration ledger is ready.
Schema-41 quiet release remains fail closed. The isolated quiet-resume SQL
fixture retains that assertion, drains a deliberately borrowed old connection,
closes the old pool, applies 42, and creates a fresh runtime before recovery.
This fixture does not touch an already-started native integration service or
apply migrations to public tables.

## Foundation ownership and privacy

Context sequence allocation serializes per owner in the database. It is monotonic,
not necessarily contiguous after deduplicated attempts. Corrections have a new
event ID, the same source identity, and the immediately preceding source revision.
Payloads are immutable snapshots. Removing one irreversibly clears its content,
attachment references, and digest while retaining a minimal tombstone so historical
events can represent the gap. This is not yet the full S11 derivative-deletion flow.
Session closure does not delete context. The composite session foreign key prevents
cross-owner references and restricts physical session deletion until dependent owner
data is handled explicitly; it never cascades a voice-session deletion into tasks.

Grants and preference versions require a finalized current user-request record,
with exact owner/request revision and immutable scope payload. SQL only establishes
that provenance, not that an interpreted scope matches the user's words. The later
authorization/action service must validate scope, expiry, revocation, current
request revision, and payload availability again immediately before execution.
An existing grant row is not permission to replay an effect after a correction.
Evidence model records similarly are snapshots, not current access decisions.

## Real database tests

```bash
python -m evals.second_brain.run_tests --live-db --manifest-dir /tmp/jarvis-s01-fixtures
```

This explicitly reuses the configured database but creates unique disposable
schemas. It does not migrate public tables or activate the running app's workers.
Manifests precede creation, enumerate exact test owners/schemas, and omit secrets.
Cleanup rejects unrelated owners and cross-schema foreign keys, then uses a new
connection to check schema removal and zero fixture-owner rows in public tables.
After an interrupted test, retry a specific manifest with:

```bash
python -m evals.second_brain.fixtures /tmp/jarvis-s01-fixtures/second-brain-RUN_ID.json
```

Fixtures are synthetic test requests, never authorization for real email or other
external actions. PostgreSQL 15+ and the existing pgcrypto/vector extensions are
required for schema parity testing; no paid model or sandbox API is called.

## File preview rollout (0053)

Apply migration 0053 before activating the gallery API release; recycle the existing API pool through its runtime owner afterward. It adds bounded optional JPEG previews to the publication ledger, separate from immutable source originals. New publication reuses verified bytes to prepare a preview; historical files receive one only when an original is explicitly opened. List and thumbnail routes never fetch original storage. JPEG previews are at most 480 pixels per edge and 64 KiB, use the existing owner/session/version access checks, and are cleared when the original is marked for deletion or quarantined. Existing originals and phone copies are not backfilled, downloaded, or deleted by this migration. Video frame extraction uses the runtime's existing ffmpeg executable; absent/failed previews leave tappable tiles.
