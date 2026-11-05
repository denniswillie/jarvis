"""Actual admission/capture/worker/route code with scoped external boundaries.

No provider, DB, native Hermes process, glasses, or hearing acceptance claims.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.models import SessionScope
from app.realtime.live_session_capture import LiveCaptionCapture
from app.realtime.live_delegation import current_result
from app.runtime.feature_config import TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, TaskStatus
from app.tasks.readonly import CAPABILITY, ReadonlyTaskAdmission, readonly_source, request_identity
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from app.voice_routes.contracts import RouteIdentity, TranscriptRevision, DeliveryEvidence
from app.voice_routes.live_adapter import LiveTranscriptPart
from app.voice_routes.live_protocol import LiveDelegationRequest, LiveTranscriptFragment
from app.voice_routes.live_results import LiveDelegationResult
from readonly_live_store import ReadonlyStore
from test_live_voice_adapter import LiveFixture, NOW


class ReadonlyOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = ReadonlyStore()
        self.scope = SessionScope(owner_id="readonly-fixture", session_id=uuid4())
        self.route = RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1)
        self.db.sessions.add((self.scope.owner_id, self.scope.session_id))
        self.tasks = TaskRepository(self.db, TaskLimits())
        self.sources = ContextRepository(self.tasks)
        self.registry = SimpleNamespace(require=AsyncMock())  # External live-worker heartbeat/readiness boundary.
        self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-no-provider")}, registry=self.registry)
        self.admission = ReadonlyTaskAdmission(self.service)
        self.context = SimpleNamespace(owner_id=self.scope.owner_id, jarvis_session_id=self.scope.session_id, audio_session_id="fixture-audio")
        self.capture = SessionContextCapture(self.sources, owner_id=self.scope.owner_id, session_id=self.scope.session_id,
                                             provider="openai", audio_session_id=self.context.audio_session_id)
        self.captions = LiveCaptionCapture(self.capture, self.context, observation_backed=True)
        self.caption_id = uuid4()
        self.parts = (LiveTranscriptPart(self.caption_id, 1, LiveTranscriptFragment("speech-1", "user", "What number is visible?", 10, 90)),)
        self.event = TranscriptRevision(route=self.route, transcript_id=self.caption_id, revision=1,
            observed_at=datetime.now(timezone.utc), role="user", text=self.parts[0].fragment.delta, final=False)
        self.request = LiveDelegationRequest("delegate-1", "d1", 80)
        self.assertTrue(self.captions.transcript(self.event, self.parts))
        self.observations = await self.captions.observations(self.parts)
        await self.capture.flush()

    async def asyncTearDown(self):
        await self.captions.close()
        await self.capture.close()

    async def admit(self, **changes):
        options = dict(route=self.route, audio_session_id=self.context.audio_session_id,
            delegation=self.request, observations=self.observations, visual_context="No current visible number is confirmed.")
        options.update(changes)
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            return await self.admission.admit(connection, **options)

    async def prepare(self, receipt):
        attempt_id = self.db.running(receipt.task)
        task = await self.tasks.get(self.scope.owner_id, receipt.task.id)
        claim = self.tasks._claim(task, self.db.attempts[attempt_id])
        context = TaskRunner(self.tasks, {}).context(claim)
        prepared = await WorkerContextFactory(self.db)(context)
        return context, prepared

    async def test_actual_capture_has_removable_originals_but_no_final_user_turn(self):
        records = self.db.rows(self.scope.owner_id)
        captions = [r for r in records if r["kind"] == "user_transcript"]
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0]["finality"], "provisional")
        self.assertFalse(self.observations[0].authorizes_request)
        self.assertEqual(self.observations[0].content["start_ms"], 10)
        self.assertEqual(self.observations[0].content["caption_revision"], 1)
        self.assertIn(self.observations[0].payload_id, self.db.parents[captions[0]["payload_id"]])
        self.db.payloads[self.observations[0].payload_id]["availability"] = "deleted"
        self.assertFalse(self.db.available(self.scope.owner_id, captions[0]["payload_id"]))

    async def test_real_admission_is_idempotent_and_cannot_authorize_grants(self):
        a, b = await asyncio.gather(self.admit(), self.admit())
        self.assertEqual(a.task.id, b.task.id)
        self.assertEqual(len(self.db.tasks), 1)
        self.assertFalse(a.source.authorizes_request)
        self.assertFalse(await self.tasks._source_available(self.db, self.scope.owner_id, a.source.id, a.source.revision))
        self.assertIsNone(await self.tasks.context_problem(self.db, a.task))
        self.assertFalse(self.db.grants)
        self.assertEqual(self.registry.require.await_args.kwargs["required_capabilities"], (CAPABILITY,))
        self.assertEqual(a.task.mode, "consult")

    async def test_stable_identity_is_route_scoped_not_offset_or_revision(self):
        key = request_identity(self.route, self.request.delegation_id)
        self.assertEqual(key, request_identity(self.route, self.request.delegation_id))
        self.assertNotEqual(key, request_identity(self.route.model_copy(update={"route_generation": 2}), self.request.delegation_id))
        await self.admit()
        with self.assertRaises(TaskError):
            await self.admit(delegation=replace(self.request, event_id="different-event"))
        self.assertEqual(len(self.db.tasks), 1)

    async def test_unavailable_worker_and_foreign_scope_create_no_task(self):
        self.registry.require.side_effect = TaskError("worker_not_ready", "No live worker")
        with self.assertRaises(TaskError):
            await self.admit()
        self.assertFalse(self.db.tasks)
        self.registry.require.side_effect = None
        other = self.route.model_copy(update={"scope": SessionScope(owner_id="foreign", session_id=uuid4())})
        with self.assertRaises(TaskError):
            await self.admit(route=other)
        self.assertFalse(self.db.tasks)

    async def test_source_correction_withdrawal_and_grants_fail_owner_guard(self):
        receipt = await self.admit()
        original = self.observations[0]
        corrected = await self.sources.append(self.scope.owner_id, ContextIngress(session_id=self.scope.session_id,
            source_namespace=original.source_namespace, source_event_id=original.source_event_id, receipt_id="correction",
            expected_revision=1, kind="observation", trust="derived", payload=original.content | {"delta": "Different words"}))
        self.assertEqual(corrected.revision, 2)
        self.assertEqual(await self.tasks.context_problem(self.db, receipt.task), "readonly_observation_superseded")
        self.db.payloads[receipt.source.payload_id]["availability"] = "deleted"
        self.assertEqual(await self.tasks.context_problem(self.db, receipt.task), "readonly_admission_unavailable")

    async def test_no_user_fragment_and_expired_context_are_not_admitted(self):
        with self.assertRaises(TaskError):
            await self.admit(observations=())
        with self.assertRaises(TaskError):
            await self.admit(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        self.assertFalse(self.db.tasks)

    async def test_later_speech_continuation_does_not_fabricate_completion_or_revoke_original(self):
        receipt = await self.admit()
        more = LiveTranscriptPart(self.caption_id, 2, LiveTranscriptFragment("speech-2", "user", " Also read the label.", 100, 180))
        event = self.event.model_copy(update={"revision": 2, "text": self.event.text + more.fragment.delta})
        self.captions.transcript(event, self.parts + (more,))
        await self.captions.observations(self.parts + (more,))
        await self.capture.flush()
        self.assertIsNone(await self.tasks.context_problem(self.db, receipt.task))
        self.assertEqual(receipt.source.content["speech_sources"][0]["revision"], 1)
        self.assertFalse(any(r["kind"] == "user_transcript" and r["finality"] == "final" for r in self.db.rows(self.scope.owner_id)))

    async def test_cancellation_revises_real_owner_without_a_fake_user_request(self):
        receipt = await self.admit()
        before = len(self.db.events)
        self.assertTrue(await self.admission.cancel(receipt, reason="cancelled"))
        task = await self.tasks.get(self.scope.owner_id, receipt.task.id)
        self.assertEqual((task.status, task.revision), (TaskStatus.CANCELLED, 2))
        self.assertFalse(await self.admission.cancel(receipt, reason="cancelled"))
        self.assertEqual(len(self.db.events), before + 1)  # Derived revocation, not a user request.
        self.assertEqual(list(self.db.events.values())[-1]["kind"], "task_update")
        self.assertEqual(list(self.db.events.values())[-1]["trust"], "derived")
        self.assertEqual(task.request_event_id, receipt.source.id)

    async def test_actual_worker_manifest_and_tool_catalog_are_strictly_readonly(self):
        receipt = await self.admit()
        context, prepared = await self.prepare(receipt)
        self.assertEqual(prepared.manifest.content["shared_context"]["admission_profile"], CAPABILITY)
        self.assertFalse(prepared.tools.authorized_operations())
        names = {t["name"] for t in prepared.tools.catalog()}
        self.assertEqual(names, {"read_task_context_archive", "read_task_context_source", "read_background_task_status"})
        for name in ("remember", "create_reminder", "terminal", "browser", "create_note"):
            with self.subTest(name=name):
                result = await prepared.tools.invoke(name, {}, provider_call_id="forbidden-" + name)
                self.assertEqual(result["status"], "failed")
        self.assertTrue(self.db.manifest_sources)
        self.assertTrue(all(m["finality"] != "provisional" for m in prepared.manifest.content["messages"]))

    async def test_grant_injection_is_rejected_by_actual_owner_before_worker_build(self):
        receipt = await self.admit()
        self.db.grants.add((self.scope.owner_id, receipt.task.id))
        self.assertEqual(await self.tasks.context_problem(self.db, receipt.task), "readonly_task_has_grants")

    async def test_current_result_requires_real_task_revision_and_manifest(self):
        receipt = await self.admit()
        context, prepared = await self.prepare(receipt)
        # Controlled external model/worker completion rows; real result reader.
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            payload = await self.tasks.payload(connection, self.scope.owner_id, {"answer": "The visible number is uncertain."})
            row = self.db.tasks[receipt.task.id]
            row.update(status="succeeded", current_attempt_id=None, result_payload_id=payload)
            await self.tasks.emit(connection, await self.tasks.load(connection, self.scope.owner_id, receipt.task.id),
                                  "succeeded", payload_id=payload, attempt_id=context.claim.attempt_id)
            result = await current_result(self.tasks, connection, receipt, self.route)
            self.assertEqual(result.text, "The visible number is uncertain.")
            self.assertIn(payload, result.roots)
            self.db.manifests[prepared.manifest.id]["status"] = "superseded"
            with self.assertRaises(TaskError):
                await current_result(self.tasks, connection, receipt, self.route)

    async def test_legacy_finalized_user_source_still_uses_original_validator(self):
        record = await self.sources.append(self.scope.owner_id, ContextIngress(session_id=self.scope.session_id,
            source_namespace="voice:legacy", source_event_id="original", receipt_id="one",
            kind="user_transcript", trust="user_input", payload={"text": "Please save this exact note."}))
        self.assertTrue(record.authorizes_request)
        self.assertTrue(await self.tasks._source_available(self.db, self.scope.owner_id, record.id, record.revision))


class CaptionRolloverTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_interleaved_fragments_keep_capturing_after_chunk_limit(self):
        f = LiveFixture()
        db = ReadonlyStore()
        scope = f.identity.scope
        db.sessions.add((scope.owner_id, scope.session_id))
        sources = ContextRepository(TaskRepository(db, TaskLimits()))
        context = SimpleNamespace(owner_id=scope.owner_id, jarvis_session_id=scope.session_id,
                                  audio_session_id="fixture-audio")
        capture = SessionContextCapture(sources, owner_id=scope.owner_id,
            session_id=scope.session_id, provider="openai", audio_session_id=context.audio_session_id)
        captions = LiveCaptionCapture(capture, context, observation_backed=True)
        identities = {"user": set(), "assistant": set()}
        try:
            await f.start()
            for index in range(270):
                role = "user" if index % 2 == 0 else "assistant"
                f.route._transcript(LiveTranscriptFragment(
                    f"fragment-{index}", role, "word ", index * 10, index * 10 + 5))
                event = f.route._current_transcript[role]
                identities[role].add(event.transcript_id)
                self.assertFalse(event.final)
                self.assertLessEqual(event.revision, 128)
                self.assertTrue(captions.transcript(event, tuple(f.route._parts)))
                await captions.observations(tuple(f.route._parts[-1:]))
            # A later delegation still has the actual recent speech, including
            # both sides of each rollover, rather than losing it at fragment129.
            window = tuple(f.route._parts[-64:])
            observations = await captions.observations(window)
            self.assertEqual(len(observations), 64)
            self.assertEqual([o.content["start_ms"] for o in observations],
                             [p.fragment.start_ms for p in window])
            self.assertTrue(all(len(ids) == 2 for ids in identities.values()))
            self.assertIsNone(captions.issue)
        finally:
            await captions.close()
            await capture.close()
            await f.close()


class ReadonlyResultTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = LiveFixture()
        await self.f.start()
        await self.f.inject(type="session.delegation.created", event_id="delegate", offset_ms=20,
                            delegation={"id": "d1", "type": "delegation", "target": "client"})
        # The provider protocol accepts the actual event vocabulary below; the
        # exact provider reader remains the sole source of delegation identity.

    async def asyncTearDown(self):
        await self.f.close()

    def result(self, text="The visible value is uncertain."):
        return LiveDelegationResult(self.f.identity, "d1", uuid4(), 1, uuid4(), text,
                                    NOW + timedelta(minutes=1), (uuid4(),))

    @asynccontextmanager
    async def guard(self, result):
        self.guarded.append(result)
        if not self.valid:
            raise PermissionError("withdrawn source")
        yield

    async def ready(self):
        await self.f.until(lambda: "d1" in self.f.route._delegations)
        self.guarded, self.valid = [], True

    async def test_actual_protocol_reply_ack_is_not_notification_or_hearing(self):
        await self.ready()
        result = self.result()
        receipt = await self.f.route.submit_delegation_result(result, guard=self.guard)
        self.assertTrue(receipt.provider_received)
        writes = [r for r in self.f.wire.socket.sent if r["type"] == "session.commentary.append"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["delegation_id"], "d1")
        self.assertEqual(await self.f.route.submit_delegation_result(result, guard=self.guard), receipt)
        self.assertFalse(any(isinstance(e, DeliveryEvidence) for e in self.f.events))
        self.assertFalse(hasattr(receipt, "attempt_id"))

    async def test_long_unicode_answer_is_lossless_and_only_final_part_is_commentary(self):
        await self.ready()
        result = self.result("Exact code: low = mid + 1; λ. " * 45)
        chunks = result.chunks(len)
        self.assertEqual("".join(json.loads(c)["text"] for c in chunks), result.text)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        receipt = await self.f.route.submit_delegation_result(result, guard=self.guard)
        self.assertTrue(receipt.provider_received)
        self.assertEqual(len(self.guarded), len(chunks))
        self.assertEqual(sum(r["type"] == "session.commentary.append" for r in self.f.wire.socket.sent), 1)

    async def test_no_action_result_is_delivered_as_quiet_context(self):
        await self.ready()
        before = len(self.f.wire.socket.sent)
        result = replace(self.result("No backend work was started for this delegation."),
                         kind="delegation_result", speak=False)
        receipt = await self.f.route.submit_delegation_result(result, guard=self.guard)
        self.assertTrue(receipt.provider_received)
        writes = self.f.wire.socket.sent[before:]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["type"], "session.thinking.append")
        self.assertEqual(writes[0]["delegation_id"], "d1")
        self.assertIn("No backend work was started", writes[0]["content"])
        self.assertFalse(any(isinstance(e, DeliveryEvidence) for e in self.f.events))

    async def test_source_withdrawal_before_write_submits_no_result(self):
        await self.ready()
        self.valid = False
        receipt = await self.f.route.submit_delegation_result(self.result(), guard=self.guard)
        self.assertFalse(receipt.provider_received)
        self.assertFalse(any(r["type"] == "session.commentary.append" for r in self.f.wire.socket.sent))

    async def test_missing_ack_is_unknown_and_not_replayed(self):
        await self.ready()
        self.f.wire.socket.auto_ack = False
        result = self.result()
        receipt = await self.f.route.submit_delegation_result(result, guard=self.guard)
        self.assertEqual(receipt.status, "unknown")
        self.assertTrue(self.f.route.requires_fenced_replacement)
        before = len(self.f.wire.socket.sent)
        self.assertEqual(await self.f.route.submit_delegation_result(result, guard=self.guard), receipt)
        self.assertEqual(len(self.f.wire.socket.sent), before)
