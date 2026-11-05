"""Actual disposable SQL and domain effects, without a model or voice claim."""
import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.live_state.models import GuardedPayloadBinding, SessionScope, StateCursor
from app.realtime.live_delegation_input import LiveDelegationInput
from app.realtime.live_domain_commands import LiveDomainCommands
from app.runtime.feature_config import TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.conversation import TaskConversation
from app.tasks.live_command_exchange import LiveCommandExchange
from app.tasks.live_commands import LiveCommandTaskAdmission, WORK_PROFILE
from app.tasks.models import TaskStatus, WorkerOutcome
from app.tasks.readonly import FRAGMENT_FORMAT
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from app.voice_routes.contracts import RouteIdentity, VoiceContextUpdate
from app.voice_routes.live_adapter import LiveDelegationWindow, LiveTranscriptPart
from app.voice_routes.live_protocol import LiveDelegationRequest, LiveTranscriptFragment
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested SQL fixture required")
class LiveWorkSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_live_request_preferences_tool_busy_input_and_idle_continuation(self):
        f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        tasks = TaskRepository(f.pool, TaskLimits(lease_seconds=300))
        async with f.pool.acquire() as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
        scope = SessionScope(owner_id=f.owner, session_id=session, area_id=f.area)
        route = RouteIdentity(scope=scope, route_id=uuid4(), route_generation=1)
        sources = ContextRepository(tasks)
        preferences = BehaviorProfileService(BehaviorProfileRepository(f.pool))
        commands = LiveDomainCommands(SimpleNamespace(behavior_profile_service=preferences), SimpleNamespace(),
            SimpleNamespace(owner_id=f.owner, jarvis_session_id=session))
        provider = ConfiguredTaskProvider("fixture-live-worker")
        registry = TaskWorkerRegistry(f.pool)
        service = TaskService(tasks, {"frontier": provider}, registry=registry)
        await registry.advertise(worker_id="fixture-live-worker", owner_id=f.owner, host_id="fixture-host",
            providers={"frontier": provider}, ttl_seconds=300)
        admission = LiveCommandTaskAdmission(service, work=True)
        mailbox = TaskConversation(tasks)
        originals, parts = [], []

        async def packet(text):
            part = LiveTranscriptPart(uuid4(), 1, LiveTranscriptFragment(uuid4().hex, "user", text, 10, 100))
            body = {"format": FRAGMENT_FORMAT, "voice_route": "live", "route": route.model_dump(mode="json"),
                "audio_provider": "openai", "audio_session_id": "live-work-sql", "caption_id": str(part.transcript_id),
                "caption_revision": 1, "provider_event_id": part.fragment.event_id, "speaker": "user", "delta": text,
                "start_ms": 10, "end_ms": 100, "speech_finality": "provisional", "request_boundary": "unavailable"}
            record = await sources.append(f.owner, ContextIngress(session_id=session,
                source_namespace="fixture-live-work", source_event_id=part.fragment.event_id,
                receipt_id=uuid4().hex, expected_revision=0, kind="observation", trust="derived", payload=body))
            originals.append(record)
            parts.append(part)
            now = datetime.now(timezone.utc)
            update = VoiceContextUpdate(route=route, update_id=uuid4(), cursor=StateCursor(epoch=uuid4(), revision=1),
                mode="snapshot", text="Synthetic task test, no visual claims.",
                guard_binding=GuardedPayloadBinding(scope=scope, payload_root_ids=(record.payload_id,)),
                created_at=now, expires_at=now + timedelta(seconds=90))
            window = LiveDelegationWindow(LiveDelegationRequest(uuid4().hex, uuid4().hex, 110), tuple(parts), len(parts))
            return LiveDelegationInput.from_window(route, "live-work-sql", window, tuple(originals), update)

        first = await packet("Use concise responses.")
        async with tasks.transaction(f.owner) as connection:
            receipt = await admission.admit(connection, first, command_catalog=commands.catalog())
        self.assertEqual(receipt.source.content["profile"], WORK_PROFILE)
        self.assertEqual(receipt.task.budget, tasks.limits)
        self.assertEqual(len([e for e in await tasks.events(f.owner, receipt.task.id) if e["event_type"] == "work_input"]), 1)
        claim = await tasks.claim_next("fixture-live-worker", ("frontier",), owner_id=f.owner,
                                      provider_models={"frontier": provider.model})
        self.assertIsNotNone(claim)
        context = TaskRunner(tasks, {}).context(claim)
        prepared = await WorkerContextFactory(f.pool)(context)
        self.assertIn("edit_preferences", {tool["name"] for tool in prepared.tools.catalog()})
        self.assertNotIn("live_note", {tool["name"] for tool in prepared.tools.catalog()})
        exchange = LiveCommandExchange(tasks)

        async def tool(name, arguments, call):
            pending = asyncio.create_task(prepared.tools.invoke(name, arguments, provider_call_id=call))
            try:
                async with asyncio.timeout(5):
                    while True:
                        rows = await tasks.events(f.owner, receipt.task.id)
                        row = next((r for r in rows if r["event_type"] == "live_command_requested"
                                    and r["content"]["provider_call_id"] == call), None)
                        if row is not None:
                            break
                        await asyncio.sleep(.01)
                    await exchange.dispatch(row["id"], route=route, is_current=lambda: True,
                                            execute=commands.execute, invocation_context=True)
                    return await pending
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

        read = await tool("read_preferences", {}, "read-first")
        self.assertEqual(read["preferences"]["revision"], 0)
        result = await tool("edit_preferences", {"expected_revision": 0, "behavior_instructions": "Use concise responses."}, "write-first")
        self.assertTrue(result["effect_committed"], result)
        reference = result["task_operation"]["operation_ref"]
        self.assertTrue((await prepared.tools.verified_operation(reference))["effect_committed"])
        saved = await preferences.get(owner_id=f.owner)
        self.assertEqual(saved["behavior_instructions"], "Use concise responses.")
        correction = await packet("Actually, explain unfamiliar terms.")
        await mailbox.append_live(receipt.task.id, correction)
        messages = await mailbox.read_inputs(claim)
        self.assertEqual(len(messages), 2)
        current = await tasks.get(f.owner, receipt.task.id)
        self.assertEqual((current.current_attempt_id, current.fence, current.revision),
                         (claim.attempt_id, claim.fence, claim.task_revision))
        updated = await tool("edit_preferences", {"expected_revision": saved["revision"],
            "behavior_instructions": "Explain unfamiliar terms."}, "write-correction")
        self.assertTrue(updated["effect_committed"], updated)
        await mailbox.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Preferences updated."}),
                             through_sequence=messages[-1].sequence)
        later = await packet("Continue: also use plain English.")
        await mailbox.append_live(receipt.task.id, later)
        second = await tasks.claim_next("fixture-live-worker", ("frontier",), owner_id=f.owner,
                                       provider_models={"frontier": provider.model})
        self.assertEqual(second.task.id, receipt.task.id)
        self.assertNotEqual(second.attempt_id, claim.attempt_id)
        history = await mailbox.history(second)
        self.assertTrue(any(item["kind"] == "succeeded" for item in history))
        self.assertEqual(sum(item["kind"] == "live_command_replied" for item in history), 3)
        self.assertEqual((await preferences.get(owner_id=f.owner))["behavior_instructions"], "Explain unfamiliar terms.")
        await mailbox.finish(second, WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Fixture complete."}),
                             through_sequence=(await mailbox.read_inputs(second))[-1].sequence)
