"""Manifested PostgreSQL qualification of v39; never migrate the public schema.

Requires the existing explicit DB/manifest environment gates. All source data is
synthetic and lives in one disposable schema. No provider/service is contacted.
"""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg

from app.attention.models import AttentionError
from app.attention.repository import AttentionRepository
from app.attention.notification_ledger.repository import VoiceNotificationLedger
from app.live_state.models import SessionScope
from app.migrations.runner import apply_migrations, load_migrations, quote_schema, read_schema_readiness
from app.runtime.feature_config import TaskLimits
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.voice_routes.contracts import (
    ClientPlaybackObservation, DeliveryEvidence, DeliveryEvidenceState as State,
    OperationCapability, OutputIdentity, RouteCapabilities, RouteIdentity, RouteOperation,
)
from evals.second_brain.fixtures import DatabaseFixture


def native_capabilities():
    return RouteCapabilities(transport="native_voice", media_topology="phone_to_provider",
        control_owner="backend", credential_mode="server_only", input_codec="opus", output_codec="opus",
        vad_owner="provider", explicit_input_commit="supported", explicit_response_create="supported",
        full_duplex="supported", image_input="unsupported", provider_context_receipts="supported",
        notification_output_attribution="supported", client_playback_evidence="unsupported",
        provider_interruption="supported", operations=tuple(
            OperationCapability(operation=value, support="supported") for value in RouteOperation))


class FixtureSpeechPreparation:
    def prepare(self, text):
        return SimpleNamespace(digest=sha256(text.encode()).hexdigest(), model="fixture-tts",
                               voice="fixture-voice", instructions="Synthetic nonexecuted request")


async def cleanup_fixture_resources(fixture, report, save, *, pool=None, connection=None):
    """Always reach manifested cleanup, even when a connection cannot close."""
    close_errors = []
    try:
        for name, resource, timeout in (("pool", pool, 15), ("connection", connection, 5)):
            if resource is None:
                continue
            try:
                await asyncio.wait_for(resource.close(), timeout=timeout)
            except Exception as error:
                close_errors.append({"resource": name, "operation": "close", "error_type": type(error).__name__})
                try:
                    resource.terminate()
                except Exception as terminate_error:
                    close_errors.append({"resource": name, "operation": "terminate",
                                         "error_type": type(terminate_error).__name__})
    finally:
        if close_errors:
            report.update(status="failed", connection_close_errors=close_errors)
        try:
            report["cleanup"] = await fixture.cleanup()
        except Exception as error:
            report.update(status="failed", cleanup_error_type=type(error).__name__)
            raise AssertionError(f"Fixture cleanup failed: {type(error).__name__}") from None
        finally:
            save()
    if close_errors:
        raise AssertionError("Fixture connection teardown failed; manifested cleanup was attempted")


class FixtureCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_teardown_preserves_independent_cleanup_result(self):
        cleanup = {"remaining_schemas": 0, "remaining_public_owner_rows": 0}
        fixture = SimpleNamespace(cleanup=AsyncMock(return_value=cleanup))
        pool = SimpleNamespace(close=AsyncMock(), terminate=Mock())
        connection = SimpleNamespace(close=AsyncMock(), terminate=Mock())
        report, save = {"status": "passed"}, Mock()
        await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
        pool.close.assert_awaited_once()
        connection.close.assert_awaited_once()
        pool.terminate.assert_not_called()
        connection.terminate.assert_not_called()
        fixture.cleanup.assert_awaited_once()
        save.assert_called_once()
        self.assertEqual(report, {"status": "passed", "cleanup": cleanup})

    async def test_close_and_terminate_failures_cannot_bypass_cleanup(self):
        for resource_name in ("pool", "connection"):
            for error_type in (RuntimeError, TimeoutError):
                for terminate_fails in (False, True):
                    with self.subTest(resource=resource_name, error=error_type, terminate_fails=terminate_fails):
                        cleanup = {"remaining_schemas": 0, "remaining_public_owner_rows": 0}
                        fixture = SimpleNamespace(cleanup=AsyncMock(return_value=cleanup))
                        resources = {name: SimpleNamespace(close=AsyncMock(), terminate=Mock())
                                     for name in ("pool", "connection")}
                        failing = resources[resource_name]
                        failing.close.side_effect = error_type("sensitive error detail")
                        if terminate_fails:
                            failing.terminate.side_effect = RuntimeError("sensitive termination detail")
                        report, save = {"status": "passed"}, Mock()
                        with self.assertRaisesRegex(AssertionError, "Fixture connection teardown failed"):
                            await cleanup_fixture_resources(fixture, report, save, **resources)
                        for resource in resources.values():
                            resource.close.assert_awaited_once()
                        failing.terminate.assert_called_once()
                        fixture.cleanup.assert_awaited_once()
                        save.assert_called_once()
                        self.assertEqual(report["status"], "failed")
                        self.assertEqual(report["cleanup"], cleanup)
                        self.assertEqual(len(report["connection_close_errors"]), 2 if terminate_fails else 1)
                        self.assertEqual(report["connection_close_errors"][0], {
                            "resource": resource_name, "operation": "close", "error_type": error_type.__name__})
                        self.assertNotIn("sensitive", json.dumps(report))

    async def test_cleanup_failure_is_saved_even_after_connection_close_failure(self):
        fixture = SimpleNamespace(cleanup=AsyncMock(side_effect=RuntimeError("sensitive cleanup detail")))
        connection = SimpleNamespace(close=AsyncMock(side_effect=TimeoutError()), terminate=Mock())
        report, save = {"status": "passed"}, Mock()
        with self.assertRaisesRegex(AssertionError, "Fixture cleanup failed: RuntimeError"):
            await cleanup_fixture_resources(fixture, report, save, connection=connection)
        fixture.cleanup.assert_awaited_once()
        connection.terminate.assert_called_once()
        save.assert_called_once()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cleanup_error_type"], "RuntimeError")
        self.assertEqual(report["connection_close_errors"][0]["error_type"], "TimeoutError")
        self.assertNotIn("sensitive", json.dumps(report))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class VoiceLedgerDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_v39_upgrade_native_ledger_and_concurrency(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"v39-results-{fixture.run_id}.json"
        report = {"format": "jarvis.v39.qualification.v1", "run_id": fixture.run_id,
                  "status": "running", "checks": [], "cleanup": None}
        failures = []
        connection = pool = None

        def save():
            descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation):
            try:
                await operation()
            except Exception as error:
                result = {"name": name, "status": "failed", "error_type": type(error).__name__,
                          "sqlstate": getattr(error, "sqlstate", None)}
                failures.append(result)
            else:
                result = {"name": name, "status": "passed"}
            report["checks"].append(result)
            save()

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(item for item in load_migrations() if item.version <= 39)
            self.assertEqual(migrations[-1].version, 39)
            self.assertEqual(migrations[-1].name, "attention_voice_routes")
            await apply_migrations(connection, schema=schema, fresh=True,
                                   migrations=tuple(item for item in migrations if item.version <= 38))
            # After migrations, exclude public from every fixture application's
            # search path. Missing fixture relations cannot fall through to prod.
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='45s'")

            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=10,
                                            command_timeout=60, setup=setup)
            owner, other = fixture.owners
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            attention = AttentionRepository(tasks, lease_seconds=300)
            ledger = VoiceNotificationLedger(attention)
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", other)
            scope = SessionScope(owner_id=owner, session_id=session)
            route = RouteIdentity(scope=scope, route_id=uuid4(), route_generation=5)
            audio_session = uuid4()

            async def complete():
                task = await tasks.create(owner, TaskCreate(request_id=uuid4(), title="Synthetic v39 qualification",
                    requested_outcome="Reason about a synthetic switch; do not act externally.", session_id=session),
                    model="fixture-model")
                claim = await tasks.claim_next("v39-fixture-worker", ("frontier",), owner_id=owner)
                self.assertIsNotNone(claim)
                self.assertEqual(claim.task.id, task.id)
                context = TaskRunner(tasks, {}).context(claim)
                outcome = await HermesTaskResult(status="succeeded", answer="The switch state is unconfirmed.",
                    speech=[{"text": "The switch state is unconfirmed."}], evidence=[],
                    completed_operation_refs=[], uncertainties=["No real observation was collected."]
                ).verified(context, SimpleNamespace())
                self.assertTrue(await tasks.finish(claim, outcome))
                identity = await connection.fetchval("select id from jarvis_attention_intents where owner_id=$1 and task_id=$2",
                                                     owner, task.id)
                return await tasks.get(owner, task.id), identity

            async def claim_for(identity, worker="v39-delivery"):
                claim = await attention.claim_next(owner, session, worker,
                    admission=lambda row, output: None if row["id"] == identity else "fixture_other_intent")
                self.assertIsNotNone(claim)
                self.assertEqual(claim.intent_id, identity)
                return claim

            async def prepare(claim):
                options = dict(route=route, audio_session_id=audio_session, capabilities=native_capabilities(),
                    offer_id=uuid4(), expires_at=datetime.now(UTC)+timedelta(minutes=4),
                    submission_boundary="fixture.nonexecuted.speech_eligible_write")
                return await ledger.prepare(claim, **options), options

            def event(offer, state, **extra):
                return DeliveryEvidence(evidence_id=uuid4(), route=offer.route, transport="native_voice",
                    observed_at=datetime.now(UTC), state=state, attribution="explicit", offer_id=offer.offer_id,
                    attempt=offer.attempt, **extra)

            async def rejected(sql, *args, expected=("23514", "23503", "23505")):
                # Savepoint/transaction rollback preserves fixture state even if
                # an unexpectedly accepted constraint probe raises AssertionError.
                try:
                    async with connection.transaction():
                        await connection.execute(sql, *args)
                        self.fail("Database accepted an invalid ledger mutation")
                except asyncpg.PostgresError as error:
                    self.assertIn(error.sqlstate, expected)

            old_task, old_intent = await complete()
            legacy = await claim_for(old_intent)
            await attention.prepare(legacy, FixtureSpeechPreparation())
            old_attempt = dict(await connection.fetchrow("select * from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                         owner, legacy.attempt_id))
            old_chunks = [dict(item) for item in await connection.fetch(
                "select * from jarvis_attention_speech_chunks where owner_id=$1 and attempt_id=$2 order by chunk_index",
                owner, legacy.attempt_id)]

            async def upgrade():
                self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), (39,))
                # The v38 repositories prepared statements on other connections.
                # Retire those connections before any post-DDL transaction;
                # never retry a failed transaction or disable production caches.
                await pool.expire_connections()
                self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), ())
                upgraded = dict(await connection.fetchrow(
                    "select * from jarvis_attention_attempts where owner_id=$1 and id=$2", owner, legacy.attempt_id))
                self.assertEqual({key: upgraded[key] for key in old_attempt}, old_attempt)
                self.assertEqual(upgraded["delivery_transport"], "legacy_pcm_tts")
                for key in ("voice_route_id", "voice_route_generation", "voice_offer_id",
                            "voice_route_protocol_version", "voice_audio_session_id"):
                    self.assertIsNone(upgraded[key])
                chunks = [dict(item) for item in await connection.fetch(
                    "select * from jarvis_attention_speech_chunks where owner_id=$1 and attempt_id=$2 order by chunk_index",
                    owner, legacy.attempt_id)]
                self.assertEqual(chunks, old_chunks)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_intents where owner_id=$1", owner), 1)
                self.assertTrue((await read_schema_readiness(connection, schema=schema, migrations=migrations)).ready)

            await check("legacy_upgrade_preservation_and_idempotent_migration", upgrade)
            if failures:
                raise AssertionError("Upgrade gate failed; later writes are unsafe")

            async def legacy_compatibility():
                with self.assertRaises(AttentionError):
                    await prepare(legacy)
                await attention.submitted(legacy)
                await attention.forwarded(legacy, samples=2400)
                await attention.generation_finished(legacy)
                self.assertEqual(await connection.fetchval("select phase from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                          owner, legacy.attempt_id), "awaiting_receipt")
                await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2",
                                         owner, legacy.attempt_id)
                await ledger.recover(owner)
                self.assertEqual(await connection.fetchval("select phase from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                          owner, legacy.attempt_id), "unknown")
                self.assertEqual(await connection.fetchval("select state from jarvis_attention_intents where owner_id=$1 and id=$2",
                                                          owner, old_intent), "unknown")

            await check("legacy_pcm_writer_and_submitted_recovery_preserved", legacy_compatibility)

            async def concurrent_claims():
                await complete()
                await complete()
                claims = await asyncio.wait_for(asyncio.gather(
                    attention.claim_next(owner, session, "concurrent-a"),
                    attention.claim_next(owner, session, "concurrent-b")), timeout=30)
                winners = [item for item in claims if item is not None]
                self.assertEqual(len(winners), 1)
                self.assertEqual(await connection.fetchval("""select count(*) from jarvis_attention_attempts
                    where owner_id=$1 and channel='voice' and phase in ('claimed','submitted','forwarding','awaiting_receipt')""", owner), 1)
                await attention.settle(winners[0], reason="fixture_concurrency_finished")
                remaining = await attention.claim_next(owner, session, "remaining-fixture")
                self.assertIsNotNone(remaining)
                await attention.settle(remaining, reason="fixture_remaining_finished")

            await check("concurrent_claims_share_one_canonical_voice_lane", concurrent_claims)
            native_task, native_intent = await complete()
            native = await claim_for(native_intent)
            offer, options = await prepare(native)
            ticket = None

            async def roots_and_dedupe():
                self.assertEqual(await ledger.prepare(native, **options), offer)
                stored = await connection.fetchrow("select * from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                    owner, native.attempt_id)
                self.assertEqual(stored["delivery_transport"], "native_voice")
                self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_payload_dependencies
                    where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3""",
                    owner, native.source_payload_id, stored["plan_payload_id"]), 1)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_route_evidence where owner_id=$1 and attempt_id=$2",
                                                          owner, native.attempt_id), 1)
                await connection.execute("update jarvis_background_tasks set result_payload_id=result_payload_id where owner_id=$1 and id=$2",
                                         owner, native_task.id)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_intents where owner_id=$1 and task_id=$2",
                                                          owner, native_task.id), 1)

            await check("canonical_intent_dedupe_and_plan_source_roots", roots_and_dedupe)

            async def constraint_guards():
                await rejected("update jarvis_attention_attempts set phase='played' where owner_id=$1 and id=$2", owner, native.attempt_id)
                await rejected("update jarvis_attention_attempts set forwarded_samples=1 where owner_id=$1 and id=$2", owner, native.attempt_id)
                await rejected("update jarvis_attention_attempts set voice_route_generation=voice_route_generation+1 where owner_id=$1 and id=$2", owner, native.attempt_id)
                await rejected("update jarvis_attention_attempts set plan_digest=$3 where owner_id=$1 and id=$2", owner, native.attempt_id, "0"*64)
                await rejected("""insert into jarvis_attention_speech_chunks
                    (owner_id,attempt_id,chunk_index,input_digest,response_id,item_id) values($1,$2,0,$3,'fixture-response','fixture-item')""",
                    owner, native.attempt_id, "0"*64)
                with self.assertRaises(AttentionError):
                    await attention.claim_next(owner, other_session, "foreign-session")
                with self.assertRaises(AttentionError):
                    await ledger.prepare(replace(native, worker_id="wrong-worker"), **options)
                with self.assertRaises(AttentionError):
                    await ledger.prepare(replace(native, owner_id=other, session_id=other_session), **options)

            await check("native_constraints_immutability_and_foreign_claim_rejection", constraint_guards)

            async def guarded_submission():
                nonlocal ticket
                ticket = await ledger.commit_submission(native, offer)
                self.assertIsNotNone(ticket)
                # This separate connection sees the committed submission, not
                # merely an uncommitted update on the writer's connection.
                stored = await connection.fetchrow("select phase,submitted_at from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                    owner, native.attempt_id)
                self.assertEqual(stored["phase"], "submitted")
                self.assertIsNotNone(stored["submitted_at"])
                self.assertIsNone(await ledger.commit_submission(native, offer))
                async with ledger.guarded_write(ticket) as allowed:
                    self.assertEqual(allowed, offer)
                with self.assertRaises(AttentionError):
                    async with ledger.guarded_write(ticket):
                        self.fail("One-use write ticket was reused")

            await check("submission_commits_before_guarded_one_use_write", guarded_submission)

            async def evidence_idempotency():
                received = event(offer, State.PROVIDER_RECEIVED)
                receipts = await asyncio.wait_for(asyncio.gather(
                    ledger.record(native, received), ledger.record(native, received)), timeout=30)
                self.assertEqual(sorted(item.inserted for item in receipts), [False, True])
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_route_evidence where owner_id=$1 and evidence_id=$2",
                                                          owner, received.evidence_id), 1)
                with self.assertRaises(AttentionError):
                    await ledger.record(native, received.model_copy(update={"reason": "conflicting-reuse"}))
                with self.assertRaises(AttentionError):
                    await ledger.record(native, event(offer, State.PROVIDER_RECEIVED).model_copy(update={
                        "route": route.model_copy(update={"route_generation": 6})}))

            await check("concurrent_evidence_idempotency_and_conflicting_reuse", evidence_idempotency)
            output = OutputIdentity(output_id=uuid4(), provider_response_ref="fixture-actual-output")
            final_event = event(offer, State.OUTPUT_FINISHED, output=output)

            async def honest_native_completion():
                await ledger.record(native, event(offer, State.OUTPUT_STARTED, output=output))
                with self.assertRaises(AttentionError):
                    await ledger.record(native, event(offer, State.CLIENT_PLAYBACK_OBSERVED, output=output,
                        playback=ClientPlaybackObservation(receipt_id=uuid4(), observed_at=datetime.now(UTC))))
                result = await ledger.record(native, final_event)
                self.assertEqual(result.attempt_phase, "unknown")
                self.assertEqual(result.hearing, "not_established")
                row = await connection.fetchrow("select phase,forwarded_samples,played_samples,reason from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                owner, native.attempt_id)
                self.assertEqual(dict(row), {"phase": "unknown", "forwarded_samples": 0, "played_samples": None,
                                            "reason": "native_completion_unconfirmed"})
                self.assertEqual(await connection.fetchval("select state from jarvis_attention_intents where owner_id=$1 and id=$2",
                                                          owner, native_intent), "unknown")
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_speech_chunks where owner_id=$1 and attempt_id=$2",
                                                          owner, native.attempt_id), 0)
                self.assertEqual((await tasks.get(owner, native_task.id)).result_payload_id, native_task.result_payload_id)

            await check("native_generation_finishes_unknown_without_fake_playback", honest_native_completion)

            async def evidence_constraints():
                stored = await connection.fetchrow("select * from jarvis_attention_route_evidence where owner_id=$1 and evidence_id=$2",
                                                    owner, final_event.evidence_id)
                await rejected("update jarvis_attention_route_evidence set evidence_state='failed' where owner_id=$1 and evidence_id=$2",
                               owner, final_event.evidence_id, expected=("23514", "P0001"))
                await rejected("""insert into jarvis_attention_route_evidence
                    (owner_id,evidence_id,attempt_id,voice_route_id,voice_route_generation,offer_id,
                     evidence_state,observed_at,evidence_payload_id,evidence_digest)
                    values($1,$2,$3,$4,$5,$6,'provider_received',clock_timestamp(),$7,$8)""",
                    owner, uuid4(), native.attempt_id, uuid4(), route.route_generation, offer.offer_id,
                    stored["evidence_payload_id"], "0"*64)
                await rejected("""insert into jarvis_attention_route_evidence
                    (owner_id,evidence_id,attempt_id,voice_route_id,voice_route_generation,offer_id,
                     evidence_state,observed_at,evidence_payload_id,evidence_digest)
                    values($1,$2,$3,$4,$5,$6,'provider_received',clock_timestamp(),$7,$8)""",
                    other, uuid4(), native.attempt_id, route.route_id, route.route_generation, offer.offer_id,
                    stored["evidence_payload_id"], "0"*64)

            await check("evidence_foreign_keys_and_append_only_constraint", evidence_constraints)

            async def late_evidence():
                # No state resurrection even though a legitimate old callback
                # has a new evidence UUID and may still add historical metadata.
                await ledger.record(native, event(offer, State.OUTPUT_FINISHED, output=output))
                self.assertEqual(await connection.fetchval("select phase from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                          owner, native.attempt_id), "unknown")
                self.assertEqual(await connection.fetchval("select generation from jarvis_attention_intents where owner_id=$1 and id=$2",
                                                          owner, native_intent), native.generation)

            await check("late_valid_evidence_keeps_original_terminal_attempt", late_evidence)

            async def withdrawn_before_write():
                task, identity = await complete()
                claim = await claim_for(identity)
                prepared, _ = await prepare(claim)
                accepted = await ledger.commit_submission(claim, prepared)
                await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2",
                                         owner, task.result_payload_id)
                with self.assertRaises(AttentionError):
                    async with ledger.guarded_write(accepted):
                        self.fail("Withdrawn source entered write boundary")
                row = await connection.fetchrow("select a.phase,p.availability from jarvis_attention_attempts a join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.plan_payload_id where a.owner_id=$1 and a.id=$2",
                                                owner, claim.attempt_id)
                self.assertEqual(row["phase"], "revoked")
                self.assertNotEqual(row["availability"], "available")

            await check("source_withdrawal_revokes_committed_ticket_and_plan", withdrawn_before_write)

            async def native_recovery():
                for submitted in (False, True):
                    _, identity = await complete()
                    claim = await claim_for(identity)
                    prepared, _ = await prepare(claim)
                    if submitted:
                        await ledger.commit_submission(claim, prepared)
                    await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2",
                                             owner, claim.attempt_id)
                    await ledger.recover(owner)
                    self.assertEqual(await connection.fetchval("select phase from jarvis_attention_attempts where owner_id=$1 and id=$2",
                                                              owner, claim.attempt_id), "unknown" if submitted else "failed")
                    self.assertEqual(await connection.fetchval("select state from jarvis_attention_intents where owner_id=$1 and id=$2",
                                                              owner, identity), "unknown" if submitted else "pending")
                    self.assertEqual(await connection.fetchval("select generation from jarvis_attention_intents where owner_id=$1 and id=$2",
                                                              owner, identity), claim.generation+1)

            await check("native_lease_recovery_preserves_possible_exposure", native_recovery)

            async def known_replay_regression():
                _, identity = await complete()
                another = await claim_for(identity)
                evidence_count = await connection.fetchval(
                    "select count(*) from jarvis_attention_route_evidence where owner_id=$1", owner)
                try:
                    with self.assertRaises(AttentionError):
                        await ledger.record(another, final_event)
                    self.assertEqual(await connection.fetchval(
                        "select phase from jarvis_attention_attempts where owner_id=$1 and id=$2",
                        owner, another.attempt_id), "claimed")
                    self.assertEqual(await connection.fetchval(
                        "select count(*) from jarvis_attention_route_evidence where owner_id=$1", owner), evidence_count)
                finally:
                    await attention.settle(another, reason="fixture_wrong_claim_finished")

            await check("known_replay_identity_must_match_claim", known_replay_regression)

            async def withdrawn_replay():
                await connection.execute(
                    "update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2",
                    owner, native_task.result_payload_id)
                before = dict(await connection.fetchrow(
                    "select * from jarvis_attention_attempts where owner_id=$1 and id=$2", owner, native.attempt_id))
                evidence_count = await connection.fetchval(
                    "select count(*) from jarvis_attention_route_evidence where owner_id=$1", owner)
                result = await ledger.record(native, final_event)
                self.assertFalse(result.inserted)
                self.assertEqual(result.attempt_phase, before["phase"])
                self.assertEqual(dict(await connection.fetchrow(
                    "select * from jarvis_attention_attempts where owner_id=$1 and id=$2", owner, native.attempt_id)), before)
                self.assertEqual(await connection.fetchval(
                    "select count(*) from jarvis_attention_route_evidence where owner_id=$1", owner), evidence_count)

            await check("identical_replay_survives_source_and_plan_withdrawal", withdrawn_replay)

            async def readiness_relation_gate():
                # DDL rolls back regardless of the assertion result. Only this
                # manifested fixture table is touched, never a public relation.
                transaction = connection.transaction()
                await transaction.start()
                try:
                    await connection.execute(f"drop table {quote_schema(schema)}.jarvis_attention_route_evidence")
                    readiness = await read_schema_readiness(connection, schema=schema, migrations=migrations)
                    self.assertFalse(readiness.ready)
                    self.assertIn("missing_table:jarvis_attention_route_evidence", readiness.issues)
                finally:
                    await transaction.rollback()

            await check("readiness_requires_v39_evidence_relation", readiness_relation_gate)
            report["status"] = "failed" if failures else "passed"
            save()
        except Exception as error:
            report.update(status="failed", fatal_error_type=type(error).__name__,
                          fatal_sqlstate=getattr(error, "sqlstate", None))
            save()
            raise AssertionError(f"Qualification stopped: {type(error).__name__}; SQLSTATE={getattr(error, 'sqlstate', None)}") from None
        finally:
            # Close active fixture connections before independent cleanup. The
            # existing helper checks owners/dependencies and audits from a new
            # connection. Cleanup failure is preserved and never called success.
            await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
            print(json.dumps({"v39_report": str(report_path), "checks": report["checks"],
                              "cleanup": {key: report["cleanup"][key] for key in
                                  ("remaining_schemas", "remaining_public_owner_rows")}}))
        self.assertFalse(failures, json.dumps(failures))
