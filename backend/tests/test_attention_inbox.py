"""Durable task-result inbox, not live-glasses delivery acceptance."""
from dataclasses import replace
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.attention.models import AttentionError, InboxMutation
from app.attention.repository import AttentionRepository
from app.attention.service import AttentionService, initialize_attention_runtime
from app.attention.tts import OpenAISpeechProvider
from app.runtime.feature_config import Capability, SecondBrainSettings, TaskLimits
from app.tasks.hermes import HermesTaskProvider
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskMutation, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture


class AttentionConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_bad_attention_configuration_is_inert(self):
        flags = SecondBrainSettings(_env_file=None, attention_lease_seconds="not-an-int")
        state = flags.resolve(Capability.ATTENTION)
        self.assertFalse(state.enabled)
        services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL=None), db_pool=None)
        with patch("app.attention.service.get_second_brain_settings", return_value=flags):
            await initialize_attention_runtime(services)
        self.assertEqual(services.attention_issue, "attention_disabled")
        self.assertFalse(hasattr(services, "attention_service"))

    def test_selected_limits_are_lazy_and_bounded(self):
        flags = SecondBrainSettings(_env_file=None, enabled="true", attention_enabled="true",
                                    attention_lease_seconds="4")
        state = flags.resolve(Capability.ATTENTION, database_url="synthetic-database")
        self.assertEqual(state.issue, "invalid_configuration")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class AttentionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.inbox = AttentionRepository(self.tasks)
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)
        self.tts = OpenAISpeechProvider(SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace()))

    async def complete(self, *, status="succeeded", runtime=False, corrupt=False):
        task = await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(),
            title="Synthetic expert", requested_outcome="Only reason; no external actions.",
            session_id=self.session), model="fixture-model")
        claim = await self.tasks.claim_next("fixture-worker", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        ctx = TaskRunner(self.tasks, {}).context(claim)
        outcome = HermesTaskProvider.incomplete("synthetic_pause") if runtime else await HermesTaskResult(
            status=status, answer="I cannot confirm the switch is off.",
            speech=[{"text":"I cannot confirm the switch is off."}, {"text":"Please check it directly."}],
            evidence=[], completed_operation_refs=[], uncertainties=["No current observation."],
        ).verified(ctx, SimpleNamespace())
        if corrupt:
            # Simulate a bad publisher before immutable storage, not an update
            # that the production source-immutability trigger correctly forbids.
            outcome = WorkerOutcome(outcome.status, outcome.payload | {"answer":"Tampered expert answer"})
        self.assertTrue(await self.tasks.finish(claim, outcome))
        async with self.f.pool.acquire() as connection:
            intent = await connection.fetchval("select id from jarvis_attention_intents where owner_id=$1 and task_id=$2", self.f.owner, task.id)
        return await self.tasks.get(self.f.owner, task.id), intent

    async def claim(self):
        result = await self.inbox.claim_next(self.f.owner, self.session, "fixture-delivery")
        self.assertIsNotNone(result)
        return result

    async def expire(self, claim):
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()-interval '1 second' where id=$1", claim.attempt_id)

    async def test_publication_is_atomic_deduplicated_and_survives_new_repository(self):
        task, identity = await self.complete()
        self.assertIsNotNone(identity)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_background_tasks set result_payload_id=result_payload_id where owner_id=$1 and id=$2", self.f.owner, task.id)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_intents"), 1)
        restarted = AttentionRepository(self.tasks)
        detail = await restarted.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "pending")
        self.assertEqual(detail["expert_result"]["answer"], "I cannot confirm the switch is off.")
        self.assertTrue(detail["expert_speech_prepared"])
        self.assertEqual(detail["attempts"], [])
        self.assertEqual((await restarted.list(self.f.owner))[0]["id"], identity)
        self.assertEqual(await restarted.list(self.f.other), [])
        with self.assertRaises(AttentionError):
            await restarted.detail(self.f.other, identity)

    async def test_runtime_pause_creates_no_fake_expert_intent(self):
        task, identity = await self.complete(runtime=True)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertIsNone(identity)

    async def test_clarification_is_an_expert_result_not_a_completed_task(self):
        task, identity = await self.complete(status="waiting_for_input")
        self.assertEqual(task.status, TaskStatus.WAITING)
        self.assertTrue((await self.inbox.detail(self.f.owner, identity))["expert_speech_prepared"])

    async def test_one_owner_channel_claim_and_foreign_preferred_session_rejected(self):
        _, first = await self.complete()
        await self.complete()
        with self.assertRaises(AttentionError):
            await self.inbox.claim_next(self.f.owner, self.other_session, "wrong-session")
        claim = await self.claim()
        self.assertEqual(claim.intent_id, first)
        self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "second-worker"))
        with self.assertRaises(AttentionError):
            await self.inbox.prepare(replace(claim, worker_id="different-worker"), self.tts)

    async def test_prepare_persists_exact_identity_without_copying_expert_words(self):
        _, identity = await self.complete()
        claim = await self.claim()
        requests = await self.inbox.prepare(claim, self.tts)
        self.assertEqual([r.text for r in requests], ["I cannot confirm the switch is off.", "Please check it directly."])
        self.assertEqual(await self.inbox.prepare(claim, self.tts), requests)
        async with self.f.pool.acquire() as connection:
            raw = await connection.fetchval("""select p.content::text from jarvis_attention_attempts a
                join jarvis_context_payloads p on p.id=a.plan_payload_id and p.owner_id=a.owner_id where a.id=$1""", claim.attempt_id)
        self.assertNotIn(requests[0].text, raw)
        self.assertIn(requests[0].digest, raw)
        await self.inbox.submitted(claim)
        with self.assertRaises(AttentionError):
            await self.inbox.submitted(claim)
        await self.inbox.forwarded(claim, samples=2400)
        await self.inbox.generation_finished(claim)
        detail = await self.inbox.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "awaiting_receipt")
        self.assertEqual(detail["attempts"][0]["forwarded_samples"], 2400)
        self.assertIsNone(detail["attempts"][0]["played_samples"])
        self.assertIn("not_established", detail["hearing"])

    async def test_revision_and_payload_removal_revoke_inflight_and_hide_old_answer(self):
        for change in ("revision", "delete"):
            with self.subTest(change=change):
                task, identity = await self.complete()
                claim = await self.claim()
                await self.inbox.prepare(claim, self.tts)
                await self.inbox.submitted(claim)
                if change == "revision":
                    updated = await self.tasks.mutate(self.f.owner, task.id, "update", TaskMutation(
                        request_id=uuid4(), expected_revision=task.revision, message="Corrected synthetic assignment."))
                    # Keep this test's next completed task independently claimable.
                    await self.tasks.mutate(self.f.owner, task.id, "cancel", TaskMutation(
                        request_id=uuid4(), expected_revision=updated.revision))
                else:
                    async with self.f.pool.acquire() as connection:
                        await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2", self.f.owner, task.result_payload_id)
                with self.assertRaises(AttentionError):
                    await self.inbox.forwarded(claim, samples=1200)
                detail = await self.inbox.detail(self.f.owner, identity)
                self.assertEqual(detail["state"], "revoked")
                self.assertIsNone(detail["expert_result"])
                self.assertEqual(detail["attempts"][0]["phase"], "revoked")

    async def test_restart_retries_only_unsubmitted_attempts(self):
        _, identity = await self.complete()
        old = await self.claim()
        await self.expire(old)
        replacement = await self.claim()
        self.assertNotEqual(old.attempt_id, replacement.attempt_id)
        with self.assertRaises(AttentionError):
            await self.inbox.prepare(old, self.tts)
        await self.inbox.prepare(replacement, self.tts)
        await self.inbox.submitted(replacement)
        await self.expire(replacement)
        self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "restart"))
        detail = await self.inbox.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "unknown")
        self.assertEqual([a["phase"] for a in detail["attempts"]], ["failed", "unknown"])

    async def test_explicit_replay_is_retry_stable_and_does_not_rerun_the_expert(self):
        task, identity = await self.complete()
        claim = await self.claim()
        await self.inbox.prepare(claim, self.tts)
        await self.inbox.submitted(claim)
        await self.expire(claim)
        self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "restart"))
        detail = await self.inbox.detail(self.f.owner, identity)
        request = InboxMutation(request_id=uuid4(), expected_generation=detail["generation"])
        first = await self.inbox.mutate(self.f.owner, identity, "replay", request)
        second = await self.inbox.mutate(self.f.owner, identity, "replay", request)
        self.assertEqual(first["generation"], second["generation"])
        new = await self.claim()
        self.assertNotEqual(new.attempt_id, claim.attempt_id)
        current_task = await self.tasks.get(self.f.owner, task.id)
        self.assertEqual(current_task.attempt_count, 1)
        self.assertEqual(current_task.result_payload_id, task.result_payload_id)
        dismiss = InboxMutation(request_id=uuid4(), expected_generation=first["generation"])
        await self.inbox.mutate(self.f.owner, identity, "dismiss", dismiss)
        with self.assertRaises(AttentionError):
            await self.inbox.prepare(new, self.tts)
        self.assertEqual(await self.inbox.list(self.f.owner), [])

    async def test_corrupt_output_is_unavailable_without_hiding_other_inbox_items(self):
        task, identity = await self.complete(corrupt=True)
        _, other = await self.complete()
        items = {item["id"]:item for item in await self.inbox.list(self.f.owner)}
        self.assertEqual(items[identity]["source_unavailable_reason"], "task_result_unavailable")
        self.assertTrue(items[other]["source_available"])
        self.assertIsNone((await self.inbox.detail(self.f.owner, identity))["expert_result"])
        self.assertEqual((await self.claim()).intent_id, other)

    async def test_replay_at_attempt_limit_is_not_falsely_accepted(self):
        self.inbox = AttentionRepository(self.tasks, max_attempts=1)
        _, identity = await self.complete()
        claim = await self.claim()
        await self.expire(claim)
        self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "restart"))
        detail = await self.inbox.detail(self.f.owner, identity)
        with self.assertRaises(AttentionError) as error:
            await self.inbox.mutate(self.f.owner, identity, "replay", InboxMutation(
                request_id=uuid4(), expected_generation=detail["generation"]))
        self.assertEqual(error.exception.code, "attention_attempt_limit")

    async def test_source_update_waits_for_guard_then_invalidates_the_next_write(self):
        task, _ = await self.complete()
        claim = await self.claim()
        entered, release, started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def writing():
            async with self.inbox.guard(claim):
                entered.set()
                await release.wait()
        async def updating():
            started.set()
            return await self.tasks.mutate(self.f.owner, task.id, "update", TaskMutation(
                request_id=uuid4(), expected_revision=task.revision, message="Corrected evidence."))
        writer = asyncio.create_task(writing())
        updater = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            updater = asyncio.create_task(updating())
            await started.wait()
            await asyncio.sleep(.05)
            self.assertFalse(updater.done())
            release.set()
            await asyncio.wait_for(writer, 5)
            await asyncio.wait_for(updater, 5)
            with self.assertRaises(AttentionError):
                await self.inbox.prepare(claim, self.tts)
        finally:
            release.set()
            for pending in (writer, updater):
                if pending is not None and not pending.done():
                    pending.cancel()
            await asyncio.gather(*(p for p in (writer, updater) if p is not None), return_exceptions=True)

    async def test_authenticated_inbox_api_reads_actual_written_result_and_dismisses(self):
        from app.main import create_app
        from app.settings import JarvisSettings, get_settings
        task, identity = await self.complete()
        app = create_app()
        app.state.runtime_services = SimpleNamespace(attention_service=AttentionService(self.inbox))
        app.dependency_overrides[get_settings] = lambda: JarvisSettings(_env_file=None,
            JARVIS_OWNER_ID=self.f.owner, JARVIS_STATIC_BEARER_TOKEN="fixture-token")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            self.assertEqual((await client.get("/attention")).status_code, 401)
            headers = {"Authorization":"Bearer fixture-token"}
            detail = await client.get(f"/attention/{identity}", headers=headers)
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["task_id"], str(task.id))
            body = {"request_id":str(uuid4()), "expected_generation":1}
            dismissed = await client.post(f"/attention/{identity}/dismiss", headers=headers, json=body)
            self.assertEqual(dismissed.status_code, 200)
            self.assertEqual((await client.get("/attention", headers=headers)).json(), [])
