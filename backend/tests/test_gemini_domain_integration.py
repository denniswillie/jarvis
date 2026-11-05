from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.assistant.tool_limits import ToolCallBudget
from app.experience.evidence import evidence_from_message
from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.realtime.gemini.client import GeminiTransportError
from app.realtime.gemini.context import GeminiContext, PreparedImage, bounded_context
from app.realtime.gemini.operations import GeminiOperations, native_declarations
from app.realtime.gemini.persistence import FinalTranscript, GeminiConversationJournal
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.events import DomainEvent, DomainEventName, WorkingMemoryUpdatedPayload
from app.sessions.models import JarvisMessage, JarvisSession
from app.settings import JarvisSettings
from app.tools.module import FunctionToolModule
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from test_gemini_audio_session import ENCODED, Phone, Wire, until


class MessageRepository:
    def __init__(self):
        self.rows = {}
        self.attached = set()
        self.create_count = 0
        self.lose_insert_ack = False

    async def create_or_reuse_gemini_message(self, **values):
        self.create_count += 1
        key = (values["owner_id"], values["session_id"], values["audio_session_id"],
               values["logical_item_id"], values["role"])
        if key not in self.rows:
            self.rows[key] = JarvisMessage(
                id=uuid4(), session_id=values["session_id"], role=values["role"], modality="audio",
                content=values["content"], provider_item_id=values["provider_item_id"],
                provider_response_id=values["provider_response_id"], created_at=values["occurred_at"],
                metadata={**values["metadata"], "audio_provider": "gemini", "audio_session_id": values["audio_session_id"]},
            )
        if self.lose_insert_ack:
            self.lose_insert_ack = False
            raise ConnectionError("fixture lost insert acknowledgement")
        return self.rows[key]

    async def gemini_message_has_experience_evidence(self, **values):
        return values["message_id"] in self.attached

    async def get_recent_messages(self, owner_id, session_id, **kwargs):
        return [message for key, message in self.rows.items() if key[0:2] == (owner_id, session_id)]


class Experience:
    def __init__(self, repository):
        self.repository = repository
        self.attachments = []
        self.fail_before = 0
        self.lose_ack = False

    @asynccontextmanager
    async def activity(self, **kwargs):
        yield

    async def attach_tool_execution(self, execution):
        return SimpleNamespace(attached=True)

    async def attach_message(self, *, owner_id, message):
        if self.fail_before:
            self.fail_before -= 1
            raise ConnectionError("fixture attachment unavailable")
        self.attachments.append(message)
        self.repository.attached.add(message.id)
        if self.lose_ack:
            self.lose_ack = False
            raise ConnectionError("fixture lost attachment acknowledgement")
        return SimpleNamespace(attached=True)


class ExecutionRepository:
    def __init__(self):
        self.rows = {}
        self.read_failure = False

    async def start_execution_with_status(self, **values):
        key = (values["owner_id"], values["session_id"], values["provider_call_id"])
        if key in self.rows:
            return self.rows[key], False
        row = SimpleNamespace(**values, id=uuid4(), status="started", result=None, error=None, completed_at=None)
        self.rows[key] = row
        return row, True

    async def finalize_execution(self, **values):
        key = (values["owner_id"], values["session_id"], values["provider_call_id"])
        row = self.rows[key]
        row.status, row.completed_at = values["status"], values["completed_at"]
        row.result, row.error = values.get("result"), values.get("error")
        return row

    async def get_executions_by_call_ids(self, *, owner_id, session_id, provider_call_ids):
        if self.read_failure:
            raise ConnectionError("fixture durable readback unavailable")
        return [row for key, row in self.rows.items()
                if key[:2] == (owner_id, session_id) and key[2] in provider_call_ids]


def context():
    now, session_id = datetime.now(UTC), uuid4()
    session = JarvisSession(id=session_id, title="Gemini domain fixture", status="active",
                            created_at=now, updated_at=now, closed_at=None, last_activity_at=now,
                            behavior_instructions="", metadata={})
    return RealtimeSessionContext("fixture-audio", "fixture-owner", session_id, "phone_audio", session)


def settings():
    with patch.dict(os.environ, {}, clear=True):
        return JarvisSettings(_env_file=None, GEMINI_API_KEY="fixture-key")


class OperationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = context()
        self.repository = ExecutionRepository()
        self.writes = []

        async def remember(arguments):
            self.writes.append(dict(arguments))
            return {"status": "success", "memory_id": str(uuid4())}

        self.handler = remember
        self.operations = self.make_operations()

    def make_operations(self, *, owner=None):
        registry = JarvisToolRegistry()
        registry.register(ToolDefinition("remember", "Remember", {"type": "object"}, self.handler, True))
        return GeminiOperations(registry=registry, repository=self.repository,
                                owner_id=owner or self.context.owner_id,
                                session_id=self.context.jarvis_session_id, audio_session_id="fixture-audio")

    def receive(self, *, native_id="save-1", namespace=1, operations=None):
        return (operations or self.operations).receive(
            {"id": native_id, "name": "remember", "args": {"content": "I prefer morning walks."}},
            namespace=namespace, turn_number=1,
        )[0]

    async def execute(self, call, *, operations=None):
        return await (operations or self.operations).execute(
            call, still_current=lambda: True, budget=ToolCallBudget(12, 3),
        )

    async def test_before_admission_cancellation_has_a_valid_not_started_status(self):
        call = self.receive()
        self.operations.cancel(namespace=1)
        await self.execute(call)
        status = await self.operations.read_status({"status_ref": call.status_ref})
        self.assertEqual(status["status"], "success")
        self.assertEqual(status["operations"][0]["state"], "not_started")
        self.assertEqual(self.writes, [])
        self.assertEqual(self.repository.rows, {})

    async def test_same_execution_replay_reads_durable_result_without_second_write(self):
        call = self.receive()
        await self.execute(call)
        rebuilt = self.make_operations()
        replay = self.receive(operations=rebuilt)
        await self.execute(replay, operations=rebuilt)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(call.status_ref, replay.status_ref)
        self.assertEqual(replay.state, "succeeded")

    async def test_budget_rejection_is_not_misreported_as_an_admitted_unknown_mutation(self):
        call = self.receive()
        await self.operations.execute(call, still_current=lambda: True, budget=ToolCallBudget(0, 0))
        self.assertFalse(call.admitted)
        self.assertEqual(call.state, "not_started")
        self.assertEqual(self.writes, [])
        self.assertEqual(self.repository.rows, {})

    async def test_fresh_namespace_and_distinct_call_are_not_semantically_deduplicated(self):
        for call in (self.receive(), self.receive(namespace=2), self.receive(native_id="save-2")):
            await self.execute(call)
        self.assertEqual(len(self.writes), 3)
        self.assertEqual(len(self.repository.rows), 3)

    async def test_cancel_after_commit_preserves_receipt_and_new_action_is_allowed(self):
        original = self.receive()
        await self.execute(original)
        self.operations.cancel(namespace=1)
        await self.execute(self.receive(native_id="new-request", namespace=2))
        self.assertEqual(len(self.writes), 2)
        self.assertEqual(original.state, "succeeded")
        self.assertTrue(original.delivery_cancelled)

    async def test_durable_read_failure_does_not_turn_a_completed_mutation_into_failure(self):
        call = self.receive()
        self.repository.read_failure = True
        with self.assertRaises(GeminiTransportError):
            await self.execute(call)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(call.state, "unknown")
        self.repository.read_failure = False
        status = await self.operations.read_status({})
        self.assertEqual(status["operations"][0]["state"], "succeeded")
        self.assertEqual(len(self.writes), 1)

    async def test_other_owner_cannot_use_reference_even_with_same_native_id(self):
        call = self.receive()
        await self.execute(call)
        foreign = self.make_operations(owner="other-owner")
        self.receive(operations=foreign)
        result = await foreign.read_status({"status_ref": call.status_ref})
        self.assertEqual(result["error"]["code"], "operation_not_found")

    async def test_conflicting_native_id_fails_without_reexecuting(self):
        self.receive()
        with self.assertRaises(GeminiTransportError):
            self.operations.receive({"id": "save-1", "name": "remember", "args": {"content": "Different"}},
                                    namespace=1, turn_number=1)
        self.assertEqual(self.writes, [])

    async def test_receipt_has_only_the_short_status_identity(self):
        call = self.receive()
        await self.execute(call)
        receipt = self.operations.receipt(call)
        encoded = json.dumps(receipt)
        self.assertNotIn(call.execution_id, encoded)
        self.assertNotIn(call.result["memory_id"], encoded)
        self.assertNotIn("provider_call_id", encoded)
        self.assertEqual(receipt["state"], "succeeded")

    def test_export_preserves_schema_and_does_not_mutate_registry(self):
        definition = self.operations.registry.get("remember")
        definition.input_schema.update({"required": ["content"], "additionalProperties": False,
                                        "properties": {"content": {"type": "string", "enum": ["one", "two"]}}})
        exported = native_declarations(self.operations.registry)
        self.assertEqual(exported[0]["parametersJsonSchema"], definition.input_schema)
        exported[0]["parametersJsonSchema"]["required"].append("bad")
        self.assertEqual(definition.input_schema["required"], ["content"])


class JournalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = context()
        self.repository = MessageRepository()
        self.experience = Experience(self.repository)
        self.errors = []
        self.journal = GeminiConversationJournal(
            repository=self.repository, experience_service=self.experience,
            owner_id=self.context.owner_id, session_id=self.context.jarvis_session_id,
            audio_session_id=self.context.audio_session_id,
            on_failure=AsyncMock(side_effect=self.errors.append),
        )

    def transcript(self, *, item="turn:1:user", role="user", eligible=True):
        return FinalTranscript(item, role, "I usually work best in the morning.", datetime.now(UTC),
                               "provider-item", eligible_for_experience=eligible)

    async def finish(self):
        await self.journal.drain(timeout_seconds=1)

    async def test_ordinary_speech_is_persisted_and_attached_without_save_tool(self):
        self.journal.submit(self.transcript())
        await self.finish()
        self.assertEqual(len(self.repository.rows), 1)
        self.assertEqual(len(self.experience.attachments), 1)
        self.assertEqual(self.experience.attachments[0].role, "user")
        self.assertEqual(self.journal.diagnostics()["states"], {"attached": 1})

    async def test_duplicate_final_and_legitimate_repeated_words_have_distinct_semantics(self):
        first = self.transcript()
        self.journal.submit(first)
        self.journal.submit(first)
        self.journal.submit(self.transcript(item="turn:2:user"))
        await self.finish()
        self.assertEqual(len(self.repository.rows), 2)
        self.assertEqual(len(self.experience.attachments), 2)

    async def test_insert_ack_loss_reuses_row(self):
        self.repository.lose_insert_ack = True
        self.journal.submit(self.transcript())
        await self.finish()
        self.assertEqual(self.repository.create_count, 2)
        self.assertEqual(len(self.repository.rows), 1)
        self.assertEqual(len(self.experience.attachments), 1)

    async def test_attachment_failure_does_not_insert_a_second_message(self):
        self.experience.fail_before = 1
        self.journal.submit(self.transcript())
        await self.finish()
        self.assertEqual(self.repository.create_count, 1)
        self.assertEqual(len(self.experience.attachments), 1)
        self.assertFalse(self.journal.diagnostics()["degraded"])

    async def test_attachment_ack_loss_reuses_cross_event_evidence_readback(self):
        self.experience.lose_ack = True
        self.journal.submit(self.transcript())
        await self.finish()
        self.assertEqual(self.repository.create_count, 1)
        self.assertEqual(len(self.experience.attachments), 1)

    async def test_partial_assistant_record_is_diagnostic_only(self):
        self.journal.submit(self.transcript(item="turn:1:assistant", role="assistant", eligible=False))
        await self.finish()
        self.assertEqual(len(self.repository.rows), 1)
        self.assertEqual(self.experience.attachments, [])
        self.assertEqual(self.journal.diagnostics()["states"], {"diagnostic_only": 1})

    async def test_persistent_attachment_failure_is_visible_and_bounded(self):
        self.experience.fail_before = 99
        self.journal.submit(self.transcript())
        await self.finish()
        self.assertEqual(self.repository.create_count, 1)
        self.assertEqual(self.errors, ["gemini_memory_persistence_failed"])
        self.assertTrue(self.journal.diagnostics()["degraded"])

    async def test_failure_diagnostics_expose_phase_but_not_speech_or_exception_text(self):
        self.experience.attach_message = AsyncMock(return_value=SimpleNamespace(
            attached=False, reason="PRIVATE-UNTRUSTED-REASON"))
        self.journal.submit(self.transcript())
        await self.finish()
        failure = self.journal.diagnostics()["failures"][0]
        self.assertEqual(failure, {"role": "user", "state": "failed", "phase": "evidence_attachment",
            "attempt": 3, "error_type": "GeminiTransportError",
            "attachment_reason": "unclassified_attachment_rejection"})
        self.assertNotIn("PRIVATE", json.dumps(self.journal.diagnostics()))
        self.assertNotIn("morning", json.dumps(self.journal.diagnostics()))

    async def test_recovered_failure_remains_diagnostic_without_claiming_degraded_retention(self):
        self.experience.fail_before = 1
        self.journal.submit(self.transcript())
        await self.finish()
        diagnostic = self.journal.diagnostics()
        self.assertFalse(diagnostic["degraded"])
        self.assertEqual(diagnostic["failures"][0]["phase"], "evidence_attachment")
        self.assertEqual(diagnostic["failures"][0]["state"], "attached")
        self.assertEqual(diagnostic["failures"][0]["error_type"], "ConnectionError")

    async def test_drain_timeout_does_not_claim_retention_or_leave_worker_running(self):
        self.repository.create_or_reuse_gemini_message = AsyncMock(side_effect=lambda **kw: None)

        async def blocked(**kwargs):
            await asyncio.Event().wait()

        self.repository.create_or_reuse_gemini_message.side_effect = blocked
        self.journal.submit(self.transcript())
        await self.journal.drain(timeout_seconds=.01)
        self.assertIn("gemini_memory_drain_timeout", self.errors)
        self.assertFalse(self.journal.diagnostics()["worker_running"])
        self.assertEqual(self.journal.diagnostics()["states"], {"incomplete": 1})

    async def test_assistant_evidence_distinguishes_generated_from_heard_and_preserves_gpt_shape(self):
        self.journal.submit(replace(self.transcript(role="assistant"), metadata={
            "completion_state": "completed", "delivery_status": "forwarded", "playback_status": "unconfirmed",
        }))
        await self.finish()
        message = self.experience.attachments[0]
        evidence = evidence_from_message(message)
        self.assertIn("playback not confirmed", evidence.summary)
        self.assertEqual(evidence.metadata["role"], "assistant")
        legacy = evidence_from_message(message.model_copy(update={"metadata": {}}))
        self.assertEqual(legacy.summary, message.content)
        self.assertEqual(legacy.metadata, {"role": "assistant", "source_modality": "audio"})


class SessionDomainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.phone, self.wires = Phone(), []
        self.context, self.settings = context(), settings()
        self.repository, self.execution_repository = MessageRepository(), ExecutionRepository()
        self.experience = Experience(self.repository)
        self.writes, self.entered = [], asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

        async def remember(arguments):
            self.entered.set()
            await self.release.wait()
            self.writes.append(dict(arguments))
            return {"status": "success", "memory_id": str(uuid4())}

        def factory(**kwargs):
            wire = Wire(**kwargs)
            self.wires.append(wire)
            return wire

        self.session = GeminiAudioSession(
            websocket=self.phone, context=self.context, settings=self.settings,
            repository=self.repository, connection_factory=factory,
            tool_modules=(FunctionToolModule("fixture", lambda _: [
                ToolDefinition("remember", "Save a fact", {"type": "object"}, remember, True),
            ]),),
            context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
            tool_execution_repository=self.execution_repository, experience_event_service=self.experience,
        )
        self.task = asyncio.create_task(self.session.run())
        await until(lambda: any(e["type"] == "session.created" for e in self.phone.events))

    async def asyncTearDown(self):
        self.release.set()
        await self.session.close("test_finished")
        await self.task

    async def turn(self, name):
        await self.phone.send("audio.input.append", segment_id=name, audio_base64=ENCODED)
        await until(lambda: name in self.session._turns)
        await self.phone.send("audio.input.commit", segment_id=name)
        await until(lambda: self.session._turns[name].released)

    async def call(self, *, call_id="save-1", name="remember", args=None):
        await self.wires[-1].input.put({"toolCall": {"functionCalls": [
            {"id": call_id, "name": name, "args": args if args is not None else {"content": "Morning walks"}},
        ]}})

    def responses(self):
        return [response for wire in self.wires for event in wire.sent
                for response in event.get("toolResponse", {}).get("functionResponses", [])]

    async def test_real_executor_receipt_roundtrip_and_duplicate_native_event(self):
        await self.turn("a")
        await self.call()
        await until(lambda: len(self.responses()) == 1)
        await self.call()
        await self.call(call_id="status-1", name="get_operation_status", args={})
        await until(lambda: len(self.responses()) == 2)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.responses()[0]["id"], "save-1")
        self.assertEqual(self.responses()[1]["response"]["operations"][0]["state"], "succeeded")

    async def test_barge_in_does_not_cancel_admitted_mutation_or_send_obsolete_receipt(self):
        self.release.clear()
        await self.turn("a")
        await self.call()
        await self.entered.wait()
        await self.phone.send("audio.input.append", segment_id="b", audio_base64=ENCODED)
        await until(lambda: "b" in self.session._turns)
        self.release.set()
        await until(lambda: bool(self.writes) and not self.session._domain_tasks)
        await self.phone.send("audio.input.commit", segment_id="b")
        await until(lambda: self.session._turns["b"].released)
        self.assertEqual(self.responses(), [])
        contexts = [json.loads(event["realtimeInput"]["text"]) for wire in self.wires for event in wire.sent
                    if "text" in event.get("realtimeInput", {})]
        self.assertEqual(contexts[-1]["operation_context"]["operations"][0]["state"], "succeeded")
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(len(self.wires), 2)
        self.assertEqual(self.session.operations.calls[(1, "save-1")].state, "succeeded")

    async def test_native_tool_cancellation_is_delivery_cancellation_not_rollback(self):
        self.release.clear()
        await self.turn("a")
        await self.call()
        await self.entered.wait()
        await self.wires[0].input.put({"toolCallCancellation": {"ids": ["save-1"]}})
        await until(lambda: self.session.operations.calls[(1, "save-1")].delivery_cancelled)
        self.release.set()
        await until(lambda: not self.session._domain_tasks)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.responses(), [])
        self.assertEqual(self.session.operations.calls[(1, "save-1")].state, "succeeded")

    async def test_close_drains_admitted_work_while_resources_are_still_available(self):
        self.release.clear()
        await self.turn("a")
        await self.call()
        await self.entered.wait()
        closing = asyncio.create_task(self.session.close("client_stop"))
        await until(lambda: self.session._closed.is_set())
        self.assertFalse(closing.done())
        self.release.set()
        await closing
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.session.operations.calls[(1, "save-1")].state, "succeeded")
        self.assertEqual(self.session.diagnostics()["domain_tasks"], 0)

    async def test_late_user_transcription_is_attached_to_original_logical_turn(self):
        await self.turn("a")
        await self.wires[0].publish({"outputTranscription": {"text": "A suggestion"}, "generationComplete": True, "turnComplete": True})
        await until(lambda: self.session._turns["a"].output.completed)
        await self.turn("b")
        await self.wires[0].publish({"inputTranscription": {"text": "I work best in the morning."}})
        await until(lambda: bool(self.experience.attachments))
        user = next(message for message in self.experience.attachments if message.role == "user")
        self.assertEqual(user.metadata["segment_id"], "a")
        self.assertEqual(user.provider_item_id, self.session._turns["a"].input_item_id)

    async def test_fallback_keeps_running_operation_receipt(self):
        self.release.clear()
        await self.turn("a")
        await self.call()
        await self.entered.wait()
        prepared = self.session._fallback_preparation(self.session._turns["a"])
        operations = json.loads(prepared.context_text)["operation_context"]["operations"]
        self.assertEqual(operations[0]["state"], "running")

    async def test_context_does_not_reinject_commands_from_native_audio_history(self):
        await self.turn("a")
        await self.wires[0].publish({"inputTranscription": {"text": "Remember this fact."}})
        await until(lambda: bool(self.experience.attachments))
        current = await self.session._context_service.shared_payload()
        restarted = await self.session._context_service.shared_payload(include_history=True)
        self.assertEqual(current["recent_messages"], [])
        self.assertEqual(restarted["recent_messages"], [])

    async def test_manager_notification_refreshes_next_turn_without_unsolicited_native_input(self):
        manager = RealtimeSessionManager()
        await manager.register(audio_session_id=self.context.audio_session_id, owner_id=self.context.owner_id,
                               jarvis_session_id=self.context.jarvis_session_id, session=self.session)
        current = {"working_memory": {"available": True, "session": {
            "revision": 3, "expires_at": None, "state": {"activity": {"label": "coding"}},
        }, "owner": None}}
        build = AsyncMock(return_value=current)
        self.session.context_updates._build = build
        self.session._context_service.shared_payload = build
        before = len(self.wires[0].sent)
        await manager.handle_working_memory_updated(DomainEvent(
            DomainEventName.WORKING_MEMORY_UPDATED,
            WorkingMemoryUpdatedPayload(self.context.owner_id, self.context.jarvis_session_id, 3),
        ))
        await self.session.context_updates.coordinator.wait_idle()
        self.assertEqual(len(self.wires[0].sent), before)
        self.assertEqual([e["revision"] for e in self.phone.events if e["type"] == "working_memory.updated"], [3])
        await self.turn("a")
        sent = [json.loads(e["realtimeInput"]["text"]) for e in self.wires[0].sent
                if "text" in e.get("realtimeInput", {})]
        self.assertEqual(sent[-1]["memory_context"]["working_memory"]["session"]["revision"], 3)
        await manager.unregister(audio_session_id=self.context.audio_session_id, owner_id=self.context.owner_id,
                                 jarvis_session_id=self.context.jarvis_session_id)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import time
        self.context, self.settings = context(), settings()
        self.registry = JarvisToolRegistry()
        now = datetime.now(UTC)
        reference = ImageReference(
            frame_id=uuid4(), owner_id=self.context.owner_id, session_id=self.context.jarvis_session_id,
            source="current", captured_at=now, content_type="image/jpeg", width=32, height=32,
            raw_image_expires_at=now + timedelta(seconds=10), visual_session_id=uuid4(), frame_sequence=1,
            stream_connection_id=uuid4(), server_received_at=now, sampled_at=now, sampled_monotonic=time.monotonic(),
        )
        self.resolved = ResolvedImage(reference, b"fixture-pixels", revalidate=AsyncMock())
        self.selection = {"status": "success", "image_status": "selected", "image": reference.to_metadata()}
        self.handler = AsyncMock(return_value=self.selection)
        self.resolver = AsyncMock(return_value=self.resolved)
        self.registry.register(ToolDefinition("get_current_camera_view", "Current frame", {"type": "object"},
                                             self.handler, image_resolver=self.resolver))
        self.service = GeminiContext(context=self.context, settings=self.settings, repository=MessageRepository(),
                                     providers=(), registry=self.registry)

    async def test_delivered_fresh_image_is_reused_without_another_selection(self):
        image, state = await self.service.select_current(namespace=1)
        self.service.mark_delivered(image, namespace=1)
        reused, state = await self.service.select_current(namespace=1)
        self.assertIs(reused, image)
        self.assertEqual(state, "reuse")
        self.assertEqual(self.handler.await_count, 1)

    async def test_new_connection_requires_a_new_image_delivery(self):
        image, _ = await self.service.select_current(namespace=1)
        self.service.mark_delivered(image, namespace=1)
        _, state = await self.service.select_current(namespace=2)
        self.assertEqual(state, "selected")
        self.assertEqual(self.handler.await_count, 2)

    async def test_source_change_is_checked_immediately_before_write(self):
        image, _ = await self.service.select_current(namespace=1)
        self.resolved.revalidate.side_effect = ImageUnavailableError("source_changed", "Camera changed")
        with self.assertRaises(ImageUnavailableError):
            await self.service.revalidate(image)

    async def test_foreign_owner_image_is_rejected(self):
        self.resolver.return_value = replace(self.resolved, reference=replace(self.resolved.reference, owner_id="other-owner"))
        with self.assertRaises(ImageUnavailableError):
            await self.service.resolve_tool_image("get_current_camera_view", self.selection)

    def test_budget_drops_optional_context_without_losing_receipts(self):
        payload = {"memory_context": {"large": "x" * 5000}, "operation_context": {"operations": [{"state": "running"}]}}
        bounded = json.loads(bounded_context(payload, max_characters=200))
        self.assertEqual(bounded["operation_context"], payload["operation_context"])
        self.assertEqual(bounded["memory_context"], "omitted_due_to_context_budget")
