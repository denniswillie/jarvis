"""Run a private synthetic source restore using native PostgreSQL base backup/WAL.

No settings/production DSN, model calls, native Hindsight, app workers, or object
storage. Invoke: python -m evals.second_brain.rehearse_recovery --root /tmp/jarvis-s11-restore.<unique>
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import traceback

import asyncpg

from app.knowledge.lifecycle_journal import SourceLifecycleJournal
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.outbox import OutboxRepository
from app.knowledge.recovery import capture_checkpoint, verify_paused_recovery, RecoveryNotReady
from app.knowledge.sources import SourceRepository
from app.migrations.runner import quote_schema
from evals.second_brain.hindsight_service import save
from evals.second_brain.postgres_recovery import postgres_recovery_fixture
from evals.second_brain.source_fixture import source_fixture


@asynccontextmanager
async def scoped_pool(dsn, schema):
    async def setup(connection):
        await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,public,extensions")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10,
                                     command_timeout=30, setup=setup)
    try:
        yield pool
    finally:
        await pool.close()


async def denied(operation, expected_code):
    try:
        await operation
    except SourceError as error:
        assert error.code == expected_code, "Unexpected source refusal"
        return True
    raise AssertionError("Deleted source unexpectedly readable or publishable")


async def run(fixture, report):
    async with source_fixture(fixture.dsn("source"), fixture.root / "source-manifests",
                              test_name="native_physical_restore") as source:
        schema = source.fixture.manifest["schemas"][0]
        original = await source.publish("Synthetic private coding progress", source_key="private")
        weight = await source.publish("I lifted 80 kg", source_key="weight")
        control = await source.publish("Unrelated control remains", source_key="control")
        async with source.pool.acquire() as connection:
            copy = await connection.fetchval(
                """insert into jarvis_context_payloads(owner_id,availability,content)
                    values($1,'available','{"text":"Synthetic private coding progress"}') returning id""",
                source.owner,
            )
            await connection.execute(
                "insert into jarvis_context_payload_dependencies values($1,$2,$3)",
                source.owner, original.payload_id, copy,
            )
            await connection.execute(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other owner','test')",
                source.other,
            )
        # Preserve an actual recorded uncertain retain receipt, not a real send.
        outbox = OutboxRepository(source.sources)
        claim = await outbox.claim(source.owner, "restore-fixture")
        assert claim.document_id == original.document_id
        await outbox.prepare_retain(claim)
        assert await outbox.failure(claim, "synthetic_dispatch_ambiguity", retryable=True) == "outcome_unknown"
        fixture.backup()
        await source.sources.delete_source(source.owner, source.area, original.source_id, expected_revision=1)
        corrected = await source.publish("Correction: I lifted 75 kg", source_key="weight", expected_revision=1)
        newer = await source.publish("New post-backup business progress", source_key="new-business")
        ephemeral = await source.publish("Post-backup temporary private fact", source_key="new-deleted")
        await source.sources.delete_source(source.owner, source.area, ephemeral.source_id, expected_revision=1)
        async with source.pool.acquire() as connection:
            pin = await capture_checkpoint(connection, schema=schema, owners=[source.owner, source.other])
            receipt = dict(await connection.fetchrow("select * from jarvis_memory_outbox where id=$1", claim.id))
            assert receipt["status"] == "outcome_unknown" and receipt["dispatch_count"] == 1
            target_wal = await connection.fetchval("select pg_walfile_name($1::text::pg_lsn)", pin.target_lsn)
            await connection.fetchval("select pg_switch_wal()")
        save(fixture.root / "checkpoint.json", pin.model_dump(mode="json"))
        fixture.await_archive(target_wal)
        journal = SourceLifecycleJournal(source.pool)
        for index, owner_pin in enumerate(pin.journals):
            await journal.export(owner_pin.owner_id, fixture.root / f"journal-{index}", checkpoint=owner_pin)

        fixture.restore("old")
        async with scoped_pool(fixture.dsn("old"), schema) as old:
            async with old.acquire() as connection:
                assert await connection.fetchval(
                    "select content->>'text' from jarvis_context_payloads where id=$1", original.payload_id
                ) == "Synthetic private coding progress"
                assert not await connection.fetchval(
                    "select exists(select 1 from jarvis_knowledge_sources where id=$1)", newer.source_id
                )
                try:
                    await verify_paused_recovery(connection, pin)
                except RecoveryNotReady as error:
                    assert str(error) == "recovery_not_paused"
                else:
                    raise AssertionError("Stale base-only restore passed")
        fixture.remove("old")
        report["checks"]["old_backup_contains_deleted_bytes_but_cannot_pass_verifier"] = True
        save(fixture.root / "result.json", report)

        fixture.restore("missing", pin)
        report["missing_wal"] = fixture.wait_missing_target_failure()
        fixture.remove("missing")
        report["checks"]["missing_wal_target_fails_closed"] = True

        fixture.restore("recovered", pin)
        async with scoped_pool(fixture.dsn("recovered"), schema) as restored:
            async with restored.acquire() as connection:
                deadline = time.monotonic() + 30
                while await connection.fetchval("select pg_get_wal_replay_pause_state()") != "paused":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Recovery target did not pause")
                    await asyncio.sleep(0.1)
                proof = await verify_paused_recovery(connection, pin)
                assert proof["database_checkpoint_matches"] and not proof["application_release_authorized"]
                report["database_proof"] = proof
                from uuid import uuid4
                mismatches = (
                    ({"system_identifier": "1"}, "recovery_cluster_or_timeline_mismatch"),
                    ({"timeline": pin.timeline + 1}, "recovery_cluster_or_timeline_mismatch"),
                    ({"target_lsn": "0/1"}, "recovery_wal_target_mismatch"),
                    ({"migration_digest": "0" * 64}, "recovery_migration_ledger_mismatch"),
                    ({"journals": (pin.journals[0].model_copy(update={"epoch": uuid4()}),) + pin.journals[1:]},
                     "recovery_journal_checkpoint_mismatch"),
                    ({"journals": (pin.journals[0].model_copy(update={"through_sequence": pin.journals[0].through_sequence + 1}),) + pin.journals[1:]},
                     "recovery_journal_checkpoint_mismatch"),
                    ({"journals": (pin.journals[0].model_copy(update={"owner_id": "absent-fixture-owner"}),)},
                     "recovery_journal_scope_missing"),
                )
                for changes, expected_code in mismatches:
                    try:
                        await verify_paused_recovery(connection, pin.model_copy(update=changes))
                    except RecoveryNotReady as error:
                        assert str(error) == expected_code
                    else:
                        raise AssertionError("Mismatched checkpoint passed verification")
                report["checks"]["seven_native_checkpoint_mismatches_rejected"] = True
                # Inspection does not itself promote or enable a worker.
                assert await connection.fetchval("select pg_is_in_recovery()")
                for payload in (original.payload_id, copy, ephemeral.payload_id):
                    row = await connection.fetchrow(
                        "select availability,content from jarvis_context_payloads where id=$1", payload)
                    assert row["availability"] == "deleted" and row["content"] is None
                assert dict(await connection.fetchrow(
                    "select * from jarvis_memory_outbox where id=$1", claim.id)) == receipt
                report["checks"]["deleted_original_and_dependent_copy_revoked_before_promotion"] = True
                report["checks"]["uncertain_retain_identity_fence_and_receipt_preserved"] = True
                # Only this owned disposable DB is promoted. No app is started.
                assert await connection.fetchval("select pg_promote(true, 30)")
            repository = SourceRepository(restored)
            assert await denied(repository.read(source.owner, source.area, original.source_id),
                                "knowledge_source_unavailable")
            assert await denied(repository.read(source.owner, source.area, original.source_id, revision=1),
                                "knowledge_source_unavailable")
            current = await repository.read(source.owner, source.area, weight.source_id)
            historical = await repository.read(source.owner, source.area, weight.source_id, revision=1)
            assert current.content == corrected.content and current.is_current
            assert historical.content == weight.content and not historical.is_current
            assert (current.source_time, current.trust, current.payload_id) == (
                corrected.source_time, corrected.trust, corrected.payload_id)
            recovered_new = await repository.read(source.owner, source.area, newer.source_id)
            assert recovered_new == newer
            assert (await repository.read(source.owner, source.area, control.source_id)).content == control.content
            assert await denied(repository.publish(source.owner, SourcePublication(
                area_id=source.area, namespace="fixture", source_key="new-deleted",
                payload_id=control.payload_id, source_type="experience", trust="user_input",
                source_time=control.source_time, time_basis=control.time_basis, expected_revision=0,
            )), "knowledge_source_deleted")
            assert await denied(repository.read(source.other, source.area, control.source_id),
                                "knowledge_source_not_found")
            recovered_journal = SourceLifecycleJournal(restored)
            for owner_pin in pin.journals:
                assert await recovered_journal.checkpoint(owner_pin.owner_id) == owner_pin
            report["checks"].update(
                deleted_current_and_history_denied=True,
                corrected_current_and_explicit_history_preserved=True,
                new_source_original_provenance_identity_preserved=True,
                created_then_deleted_identity_cannot_resurrect=True,
                unrelated_control_and_owner_scope_preserved=True,
                journal_epochs_and_heads_match=True,
            )
        fixture.remove("recovered")
    report["source_fixture_cleanup_completed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = {"format": "jarvis.postgres_restore_rehearsal.v1", "status": "running", "checks": {},
              "full_restore_accepted": False, "production_database_used": False,
              "native_hindsight_tested": False, "object_storage_tested": False,
              "real_external_effect_sent": False}
    started = time.monotonic()
    report_root = None
    try:
        with postgres_recovery_fixture(args.root) as fixture:
            report_root = fixture.root
            save(fixture.root / "result.json", report)
            asyncio.run(run(fixture, report))
        report["status"] = "passed"
        report["resource_cleanup_completed"] = True
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        report["failure_stack"] = [
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)
        ]
        report["earlier_errors"] = []
        cause = error.__context__
        while cause is not None and len(report["earlier_errors"]) < 4:
            report["earlier_errors"].append({
                "error_type": type(cause).__name__,
                "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                          for frame in traceback.extract_tb(cause.__traceback__)],
            })
            cause = cause.__context__
        # No arbitrary exception text, DSN, originals or provider credentials.
        if isinstance(error, (RecoveryNotReady, AssertionError, TimeoutError)):
            report["failure_code"] = str(error)[:200]
        report["seconds"] = round(time.monotonic() - started, 2)
        if report_root is not None:
            save(report_root / "result.json", report)
        print(json.dumps(report))
        raise SystemExit(1)
    report["seconds"] = round(time.monotonic() - started, 2)
    save(report_root / "result.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
