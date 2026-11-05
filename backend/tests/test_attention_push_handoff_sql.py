"""Canonical push handoff qualification in disposable schemas and MockTransport.

Synthetic signing keys only. No APNs, provider, server process, Swift or public
migration. HTTP-client entry and simulated partial progress are not physical
wire, display or hearing evidence. Each check and independent cleanup is saved.
"""
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import httpx

from app.attention.models import InboxMutation
from app.attention.push_models import DeviceRegistration, PushOutcome
from app.attention.push_provider import APNsProvider
from app.attention.push_repository import PushRepository
from app.attention.push_service import PushService
from app.attention.quiet import OWNER_LOCK, OwnerQuietRepository, QuietCommand
from app.attention.repository import AttentionRepository
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskMutation, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tools.invocation import ToolInvocationContext
from evals.second_brain.fixtures import DatabaseFixture
from test_attention_push_handoff import ControlledHTTP, ControlledManager
from test_attention_quiet_resume_sql import record_qualification_check
from test_notification_ledger_database import FixtureSpeechPreparation, cleanup_fixture_resources
from test_phone_push import config


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database required")
class PushHandoffSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_authorization_and_controlled_sender(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"push-handoff-results-{fixture.run_id}.json"
        report = {"format": "jarvis.push_handoff.sql.v1", "run_id": fixture.run_id, "status": "running",
                  "checks": [], "cleanup": None, "actual_apns_calls": 0, "actual_provider_calls": 0,
                  "hardware_evidence": False, "wire_evidence": "controlled_simulation_only"}
        failures, resources = [], []
        connection = pool = None

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation):
            return await record_qualification_check(report, failures, save, name, operation)

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            await apply_migrations(connection, schema=schema, fresh=True,
                migrations=tuple(m for m in load_migrations() if m.version <= 42))
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)
            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='45s'")
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=10, command_timeout=60, setup=setup)
            owner, other = fixture.owners
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            contexts, quiet = ContextRepository(tasks), OwnerQuietRepository(pool)
            attention = AttentionRepository(tasks, lease_seconds=300)
            configuration = config(fallback_seconds=0, http_timeout_seconds=20, lease_seconds=60)
            push = PushRepository(attention, configuration)
            manager = ControlledManager()
            device = uuid4()
            await push.change_device(owner, device, DeviceRegistration(request_id=uuid4(), expected_revision=0,
                environment="sandbox", topic=configuration.topic, token="ab" * 32, permission="authorized", enabled=True))

            def sender():
                control = ControlledHTTP()
                provider = APNsProvider(configuration,
                    client=httpx.AsyncClient(transport=httpx.MockTransport(control.handle)))
                resources.append((provider, control))
                return PushService(push, provider, manager, owner, configuration), control

            async def change(action="activate", **kwargs):
                source = await contexts.append(owner, ContextIngress(session_id=session,
                    source_namespace="push_handoff_fixture", source_event_id=uuid4().hex, receipt_id=uuid4().hex,
                    kind="user_transcript", trust="user_input", finality="final",
                    payload={"text": "Resume interruptions." if action == "resume" else "Please do not disturb."}))
                return await quiet.save(owner, session, ToolInvocationContext(request_event_id=source.id,
                    request_revision=source.revision), QuietCommand(action=action,
                    expected_revision=(await quiet.get(owner)).revision, **kwargs))

            async def work():
                task = await tasks.create(owner, TaskCreate(request_id=uuid4(), title="Synthetic push handoff work",
                    requested_outcome="Return a synthetic read-only result without external actions.", session_id=session),
                    model="fixture-model")
                claim = await tasks.claim_next("handoff-work", ("frontier",), owner_id=owner)
                self.assertIsNotNone(claim)
                self.assertEqual(claim.task.id, task.id)
                result = await HermesTaskResult(status="succeeded", answer="Synthetic work remains durable.",
                    speech=[{"text": "Synthetic work remains durable."}], evidence=[], completed_operation_refs=[],
                    uncertainties=[]).verified(TaskRunner(tasks, {}).context(claim), SimpleNamespace())
                self.assertTrue(await tasks.finish(claim, result))
                identity = await connection.fetchval("select id from jarvis_attention_intents where owner_id=$1 and task_id=$2", owner, task.id)
                return await tasks.get(owner, task.id), identity

            async def reset():
                manager.connected = False
                async with tasks.transaction(owner) as c:
                    await c.execute("""update jarvis_attention_attempts set
                        phase=case when submitted_at is null then 'failed' else 'unknown' end,ended_at=clock_timestamp()
                        where owner_id=$1 and phase in ('claimed','submitted','forwarding','awaiting_receipt')""", owner)
                    await c.execute("update jarvis_attention_intents set state='dismissed' where owner_id=$1", owner)
                if (await quiet.get(owner)).effective_state == "quiet":
                    await change("resume")
                current = (await push.devices(owner))[0]
                await push.change_device(owner, device, DeviceRegistration(request_id=uuid4(), expected_revision=current["revision"],
                    environment="sandbox", topic=configuration.topic, token="ab" * 32, permission="authorized", enabled=True,
                    quiet_until=None))

            async def attempt(identity):
                return await connection.fetchrow("""select a.*,p.provider_status,p.retry_at from jarvis_attention_attempts a
                    join jarvis_push_submissions p on p.owner_id=a.owner_id and p.attempt_id=a.id
                    where a.owner_id=$1 and a.intent_id=$2 order by a.created_at desc,a.id desc limit 1""", owner, identity)

            async def assert_durable(task, identity):
                self.assertEqual(await tasks.get(owner, task.id), task)
                detail = await attention.detail(owner, identity)
                self.assertIsNotNone(detail["expert_result"])
                self.assertEqual(detail["hearing"], "not_established_by_inbox_or_generation_state")

            async def preparation():
                await reset()
                task, identity = await work()
                claim = await push.claim_next(owner, "prepare-only")
                first = await push.prepare_submission(claim)
                self.assertIsNotNone(first)
                self.assertEqual(await push.prepare_submission(claim), first)
                row = await attempt(identity)
                self.assertEqual(row["phase"], "claimed")
                self.assertIsNone(row["submitted_at"])
                self.assertIsNone(row["provider_status"])
                await assert_durable(task, identity)
            await check("preparation_is_nonconsuming_and_not_wire_or_provider_acceptance", preparation)

            async def quiet_before_claim():
                await reset()
                await change()
                task, identity = await work()
                service, control = sender()
                for _ in range(3):
                    self.assertFalse(await service.run_once())
                self.assertIsNone(await attempt(identity))
                self.assertEqual(control.requests, [])
                await assert_durable(task, identity)
            await check("quiet_before_eligibility_never_claims_or_invokes_client", quiet_before_claim)

            async def late_quiet(count):
                await reset()
                task, identity = await work()
                service, control = sender()
                entered, release = asyncio.Event(), asyncio.Event()
                original = push.authorize_send
                async def paused(request):
                    entered.set()
                    await release.wait()
                    return await original(request)
                with patch.object(push, "authorize_send", paused):
                    pending = asyncio.create_task(service.run_once())
                    try:
                        await asyncio.wait_for(entered.wait(), 30)
                        before = await attempt(identity)
                        self.assertEqual(before["phase"], "claimed")
                        self.assertIsNone(before["submitted_at"])
                        for _ in range(count):
                            await change()
                    finally:
                        release.set()
                        await asyncio.wait_for(pending, 30)
                row = await attempt(identity)
                self.assertEqual(row["phase"], "revoked")
                self.assertEqual(row["reason"], "owner_quiet_quiet")
                self.assertIsNone(row["submitted_at"])
                self.assertEqual(control.requests, [])
                await assert_durable(task, identity)
                for _ in range(count):
                    await change("resume")
                self.assertTrue(await service.run_once())
                self.assertEqual(len(control.requests), 1)
                self.assertEqual((await attention.list(owner))[0]["source_namespace"], "quiet_resume_summary")
                manager.connected = True
                self.assertIsNone(await attention.claim_next(owner, session, "no-summary-voice-duplicate"))
                manager.connected = False
                self.assertFalse(await service.run_once())
                self.assertEqual(len(control.requests), 1)
            for count in (1, 2, 3):
                await check(f"{count}_quiet_controls_after_prepare_before_authorization_then_one_resume_summary",
                            lambda count=count: late_quiet(count))

            async def reconnect():
                await reset()
                task, identity = await work()
                service, control = sender()
                original = push.prepare_submission
                async def prepared(claim):
                    request = await original(claim)
                    manager.connected = True
                    return request
                revision = (await quiet.get(owner)).revision
                with patch.object(push, "prepare_submission", prepared):
                    self.assertTrue(await service.run_once())
                row = await attempt(identity)
                self.assertEqual(row["reason"], "voice_reconnected")
                self.assertIsNone(row["submitted_at"])
                self.assertEqual(control.requests, [])
                self.assertEqual((await quiet.get(owner)).revision, revision)
                voice = await attention.claim_next(owner, session, "direct-route-available")
                self.assertEqual(voice.intent_id, identity)
                await attention.settle(voice, reason="fixture_speech_stop", interrupted=True)
                await assert_durable(task, identity)
            await check("reconnect_after_prepare_prevents_handoff_without_resuming_quiet_or_canceling_work", reconnect)

            async def final_source_fences(mode):
                await reset()
                task, identity = await work()
                service, control = sender()
                original = push.authorize_send
                async def changed(request):
                    if mode == "device":
                        current = (await push.devices(owner))[0]
                        await push.change_device(owner, device, DeviceRegistration(request_id=uuid4(),
                            expected_revision=current["revision"], environment="sandbox", topic=configuration.topic,
                            token="cd" * 32, permission="authorized", enabled=True))
                    elif mode == "source":
                        await connection.execute("update jarvis_context_payloads set availability='unavailable',content=null where owner_id=$1 and id=$2",
                                                 owner, task.result_payload_id)
                    elif mode == "goal":
                        revised = await tasks.mutate(owner, task.id, "update", TaskMutation(request_id=uuid4(),
                            expected_revision=task.revision, message="A different synthetic goal."))
                        await tasks.mutate(owner, task.id, "cancel", TaskMutation(request_id=uuid4(), expected_revision=revised.revision))
                    else:
                        await connection.execute("update jarvis_attention_intents set not_before=clock_timestamp()+interval '1 hour' where owner_id=$1 and id=$2",
                                                 owner, identity)
                    return await original(request)
                revision = (await quiet.get(owner)).revision
                with patch.object(push, "authorize_send", changed):
                    self.assertTrue(await service.run_once())
                self.assertEqual(control.requests, [])
                self.assertIsNone((await attempt(identity))["submitted_at"])
                self.assertEqual((await quiet.get(owner)).revision, revision)
            for mode in ("device", "source", "goal", "defer"):
                await check(f"{mode}_change_after_prepare_is_rechecked_before_handoff",
                            lambda mode=mode: final_source_fences(mode))

            async def late_after_authorization(mode):
                await reset()
                task, identity = await work()
                service, control = sender()
                control.mode = mode
                original = push.authorize_send
                async def committed_then_quiet(request):
                    receipt = await original(request)
                    self.assertTrue(receipt.allowed)
                    # This deliberately lands AFTER the irrevocable durable
                    # authorization but BEFORE adapter/client invocation.
                    for _ in range(3):
                        await change()
                    return receipt
                async def client_boundary(request):
                    self.assertFalse(manager.lock.locked())
                    async with pool.acquire() as probe, probe.transaction():
                        self.assertTrue(await probe.fetchval(OWNER_LOCK.replace(
                            "pg_advisory_xact_lock", "pg_try_advisory_xact_lock"), owner),
                            "No canonical owner lock may be held over HTTP I/O.")
                    # Actual authorized work and quiet mutations proceed while
                    # the controlled transport is active, without task cancel.
                    background, notice = await work()
                    await change()
                    self.assertEqual((await tasks.get(owner, background.id)).status, TaskStatus.SUCCEEDED)
                    detail = await attention.detail(owner, notice)
                    await attention.mutate(owner, notice, "dismiss", InboxMutation(request_id=uuid4(),
                        expected_generation=detail["generation"]))
                    row = await attempt(identity)
                    self.assertEqual(row["phase"], "submitted")
                    self.assertIsNotNone(row["submitted_at"])
                    self.assertIsNone(row["provider_status"])
                control.callback = client_boundary
                with patch.object(push, "authorize_send", committed_then_quiet):
                    self.assertTrue(await service.run_once())
                row = await attempt(identity)
                self.assertEqual(row["phase"], "accepted" if mode == "accepted" else "unknown")
                self.assertEqual(row["provider_status"], 200 if mode == "accepted" else None)
                self.assertIsNone(row["retry_at"])
                self.assertEqual(len(control.requests), 1)
                self.assertEqual(service.status()["delivery"], "not_proven_by_provider_acceptance")
                self.assertTrue((await quiet.get(owner)).suppresses_unsolicited)
                for _ in range(3):
                    await change("resume")
                    manager.connected = True
                    self.assertFalse(await service.run_once())
                    self.assertIsNone(await attention.claim_next(owner, session, "no-automatic-voice-copy"))
                    manager.connected = False
                    self.assertFalse(await service.run_once())
                await assert_durable(task, identity)
                self.assertEqual(len(control.requests), 1)
            for mode in ("accepted", "partial", "unknown"):
                await check(f"postauthorization_quiet_{mode}_honest_boundary_no_locks_work_continues_no_cross_channel_replay",
                            lambda mode=mode: late_after_authorization(mode))

            async def cancellation(committed):
                await reset()
                task, identity = await work()
                service, control = sender()
                entered = asyncio.Event()
                original = push.authorize_send
                async def blocked(request):
                    if committed:
                        self.assertTrue((await original(request)).allowed)
                    entered.set()
                    await asyncio.Future()
                revision = (await quiet.get(owner)).revision
                with patch.object(push, "authorize_send", blocked):
                    pending = asyncio.create_task(service.run_once())
                    try:
                        await asyncio.wait_for(entered.wait(), 30)
                    finally:
                        pending.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await pending
                row = await attempt(identity)
                self.assertEqual(row["phase"], "unknown" if committed else "failed")
                self.assertEqual(row["submitted_at"] is not None, committed)
                if committed:
                    self.assertIsNone(row["retry_at"])
                    self.assertIsNone(await attention.claim_next(owner, session, "cancelled-handoff-no-voice-retry"))
                    self.assertIsNone(await push.claim_next(owner, "cancelled-handoff-no-phone-retry"))
                else:
                    voice = await attention.claim_next(owner, session, "safe-unsubmitted-route")
                    self.assertEqual(voice.intent_id, identity)
                    for _ in range(3):
                        await attention.settle(voice, reason="speech_only_stop", interrupted=True)
                self.assertEqual((await quiet.get(owner)).revision, revision)
                self.assertEqual(control.requests, [])
                await assert_durable(task, identity)
            for committed in (False, True):
                await check(f"cancellation_{'after_durable_commit_before_callback_return' if committed else 'before_authorization'}_uses_canonical_evidence",
                            lambda committed=committed: cancellation(committed))

            async def expiry():
                await reset()
                task, identity = await work()
                service, control = sender()
                original = push.authorize_send
                async def expired(request):
                    now = await connection.fetchval("select clock_timestamp()")
                    deadline = now + timedelta(seconds=2)
                    await change(expires_at=deadline, timezone="UTC")
                    await asyncio.sleep(max(0, (deadline - await connection.fetchval("select clock_timestamp()")).total_seconds()) + .05)
                    return await original(request)
                with patch.object(push, "authorize_send", expired):
                    self.assertTrue(await service.run_once())
                self.assertEqual(control.requests, [])
                self.assertIsNone((await attempt(identity))["submitted_at"])
                self.assertEqual((await quiet.get(owner)).effective_state, "expired")
                self.assertTrue(await service.run_once())
                self.assertEqual(len(control.requests), 1)
                self.assertFalse(await service.run_once())
                self.assertIsNone(await attention.claim_next(owner, session, "expiry-no-cross-channel-replay"))
                await assert_durable(task, identity)
            await check("expiry_coalesces_prepared_original_instead_of_flushing_it", expiry)

            async def duplicate_and_explicit_replay():
                await reset()
                task, identity = await work()
                claim = await push.claim_next(owner, "one-use")
                request = await push.prepare_submission(claim)
                service, control = sender()
                outcomes = await asyncio.gather(*(service.provider.send(request, authorize=push.authorize_send) for _ in range(3)))
                accepted = [outcome for outcome in outcomes if outcome.state == "accepted"]
                self.assertEqual(len(accepted), 1)
                self.assertEqual(len(control.requests), 1)
                self.assertEqual(sum(outcome.state == "revoked" for outcome in outcomes), 2)
                await push.finish(claim, accepted[0])
                self.assertIsNone(await attention.claim_next(owner, session, "no-unsolicited-duplicate"))
                detail = await attention.detail(owner, identity)
                request = InboxMutation(request_id=uuid4(), expected_generation=detail["generation"])
                await attention.mutate(owner, identity, "replay", request)
                self.assertTrue((await attention.mutate(owner, identity, "replay", request))["replayed_request"])
                self.assertEqual((await attention.claim_next(owner, session, "explicit-original-reply")).intent_id, identity)
                await assert_durable(task, identity)
            await check("one_use_authorization_across_competing_senders_explicit_replay_still_available", duplicate_and_explicit_replay)

            async def crash_reconstruction():
                await reset()
                task, identity = await work()
                claim = await push.claim_next(owner, "crashed-after-auth")
                request = await push.prepare_submission(claim)
                self.assertTrue((await push.authorize_send(request)).allowed)
                await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2",
                                         owner, claim.attempt_id)
                fresh_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, setup=setup, timeout=10, command_timeout=60)
                try:
                    fresh_attention = AttentionRepository(TaskRepository(fresh_pool, TaskLimits(lease_seconds=300)))
                    fresh_push = PushRepository(fresh_attention, configuration)
                    await fresh_push.recover(owner)
                    self.assertIsNone(await fresh_push.claim_next(owner, "reconstructed-phone"))
                    self.assertIsNone(await fresh_attention.claim_next(owner, session, "reconstructed-voice"))
                finally:
                    await asyncio.wait_for(fresh_pool.close(), 10)
                row = await attempt(identity)
                self.assertEqual(row["phase"], "unknown")
                self.assertIsNone(row["provider_status"])
                self.assertIsNone(row["retry_at"])
                await assert_durable(task, identity)
            await check("authorized_crash_recovery_on_new_pool_never_asserts_wire_or_retries_another_channel", crash_reconstruction)

            async def voice_first():
                await reset()
                task, identity = await work()
                voice = await attention.claim_next(owner, session, "voice-first")
                self.assertEqual(voice.intent_id, identity)
                self.assertIsNone(await push.claim_next(owner, "no-phone-while-voice-claimed"))
                requests = await attention.prepare(voice, FixtureSpeechPreparation())
                await attention.submit_chunk(voice, 0, requests[0])
                await attention.settle(voice, reason="fixture_ambiguous_voice")
                self.assertIsNone(await push.claim_next(owner, "no-phone-after-unknown-voice"))
                await assert_durable(task, identity)
            await check("claimed_and_ambiguous_voice_attempts_cannot_retry_through_phone", voice_first)
            report["status"] = "failed" if failures else "passed"
        except Exception as error:
            report.update(status="failed", setup_error_type=type(error).__name__, sqlstate=getattr(error, "sqlstate", None))
            raise AssertionError("Push handoff fixture setup/execution failed: " + type(error).__name__) from None
        finally:
            report["controlled_http_client_entries"] = sum(len(control.requests) for _, control in resources)
            try:
                for provider, _ in resources:
                    await provider.close()
            finally:
                await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
            print("PUSH_HANDOFF_SQL_REPORT=" + str(report_path))
            print("PUSH_HANDOFF_SQL_SUMMARY=" + json.dumps(report))
        self.assertEqual(failures, [], "Named push handoff checks failed; see sanitized report.")
