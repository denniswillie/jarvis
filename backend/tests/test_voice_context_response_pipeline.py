"""Composed backend regression for native e228's adapter-stage prewrite expiry.

Real session loops, Live State producer/publisher, canonical binder/source guards,
route coordinator/adapter, and OpenAI serialized writer. Only storage and external
transport are controlled. SQL predicates, perception/model quality, RTP, decoded
PCM, playback and hearing are NOT qualified by this offline composition.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder, memory_dependency
from app.live_state.models import GuardedPayloadBinding, ObservationEntry, SessionScope
from app.live_state.producers import CanonicalStateProducer
from app.live_state.service import LiveStateService
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.live_state_bridge import install_responsive_voice
from app.realtime.models import RealtimeSessionContext
from app.settings import JarvisSettings
from app.voice_routes.contracts import CommandReceipt
from app.voice_routes.coordinator import CoordinationPhase
import test_live_state_canonical_adapters as canonical_fixtures
import test_response_preparation_source_race as source_fixtures
from test_voice_route_runtime import CLIENT_CAPS


class Storage(source_fixtures.Connection):
    """Existing fake payload store, extended only for canonical binder SQL I/O."""

    def is_in_transaction(self):
        return self.entered > self.exited

    async def execute(self, query, *args):
        assert self.is_in_transaction()
        assert "pg_advisory_xact_lock" in query
        return "SELECT 1"

    async def fetchval(self, query, owner, value=None):
        if "pg_try_advisory_xact_lock" in query:
            assert self.is_in_transaction()
            return True  # Controlled SQL result, not real lock qualification.
        if query.startswith("select true from jarvis_sessions"):
            return (owner, value) == (self.scope.owner_id, self.scope.session_id)
        result = await super().fetchval(query, owner, value)
        if query.startswith("select lock_jarvis_copy_sources"):
            # A controlled storage predicate, not a replacement production
            # authority or an assertion that PostgreSQL cascades were executed.
            roots = set(value)
            for _ in range(len(self.dependencies) + 1):
                previous = set(roots)
                for group in self.dependencies:
                    roots.update(parent for _, parent, child in group if child in roots)
                if roots == previous:
                    break
            return result and not roots.intersection(self.rejected)
        return result


class ProviderSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.writes = []
        self.closed = False

    async def inject(self, kind, **fields):
        await self.incoming.put(json.dumps({"type": kind, **fields}))

    async def send(self, raw):
        payload = json.loads(raw)
        self.writes.append(payload)
        if payload["type"] == "session.update":
            await self.inject("session.updated")
        elif payload["type"] == "conversation.item.create":
            await self.inject("conversation.item.done", item=payload["item"])
        elif payload["type"] == "conversation.item.delete":
            await self.inject("conversation.item.deleted", item_id=payload["item_id"])

    async def recv(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True


class AppSocket:
    def __init__(self, clock):
        self.clock = clock
        self.incoming = asyncio.Queue()
        self.events = []

    async def send_json(self, event):
        await self.send_text(json.dumps(event))

    async def send_text(self, raw):
        event = json.loads(raw)
        self.events.append(event)
        kind = event["type"]
        if kind == "voice_route.selected":
            reply = dict(type="voice_route.ready", protocol_version=1,
                         audio_session_id=event["audio_session_id"], route=event["route"],
                         client_event_id=uuid4().hex)
        elif kind in {"voice_route.interrupt", "voice_route.close"}:
            command = event["payload"]
            receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                operation="close" if kind == "voice_route.close" else "interrupt_speech",
                observed_at=self.clock(), status="completed", output_fenced=True)
            reply = dict(type="voice_route.event", protocol_version=1,
                         audio_session_id=event["audio_session_id"],
                         payload=receipt.model_dump(mode="json"), client_event_id=uuid4().hex)
        elif kind == "session.context_invalidated":
            reply = dict(type="session.context_cleared", protocol_version=1,
                         audio_session_id=event["audio_session_id"],
                         reset_id=event["context_reset"]["id"])
        else:
            return
        # Explicit controlled peer receipts; no claim of actual client fencing.
        await self.incoming.put(json.dumps(reply))

    async def iter_text(self):
        while (raw := await self.incoming.get()) is not None:
            yield raw

    async def close(self, **_):
        await self.incoming.put(None)


class VoiceContextResponsePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime(2026, 10, 31, 13, 25, 55, tzinfo=timezone.utc)
        self.scope = SessionScope(owner_id="offline-response-pipeline", session_id=uuid4(), area_id=uuid4())
        self.storage = Storage()
        self.storage.scope = self.scope
        self.pool = source_fixtures.Pool(self.storage)
        self.records = {}
        self.expiry_target = None
        self.expiry_locks = 0
        self.attachment = self.running = None
        self.provider = ProviderSocket()
        self.app = AppSocket(lambda: self.now)
        self.service = LiveStateService(self.scope, clock=lambda: self.now)
        self.producer = CanonicalStateProducer(self.service, producer_id="controlled-visual", clock=lambda: self.now)

        @asynccontextmanager
        async def transaction(owner, *, wait_for_owner=True):
            self.assertEqual(owner, self.scope.owner_id)
            if not wait_for_owner:
                async with SourceRepository.transaction(sources, owner, wait_for_owner=False) as connection:
                    yield connection
                return
            async with self.pool.acquire() as connection, connection.transaction():
                yield connection

        async def area(connection, owner, area_id):
            self.assertEqual((owner, area_id), (self.scope.owner_id, self.scope.area_id))

        async def read_many(connection, owner, area_id, refs):
            await area(connection, owner, area_id)
            return [self.records[ref] for ref in refs]

        sources = SimpleNamespace(pool=self.pool, transaction=transaction, _area=area, _read_many=read_many)
        sources._owner_transaction = lambda owner, **options: SourceRepository._owner_transaction(sources, owner, **options)
        self.binder = CanonicalContextBinder(sources, self.scope, clock=lambda: self.now)

        async def start_live():
            await self.service.start()
            self.producer.start()
            await self.observation("initial-as-of-view", seconds=1)
            self.service.mark_ready()

        self.live = SimpleNamespace(service=self.service, binder=self.binder,
            requires_reconstruction=False, bind=self.binder, guard=self.binder.guard,
            scope_guard=self.binder._transaction, start=start_live, close=self.service.close)
        self.repository = SimpleNamespace(_pool=self.pool,
            get_recent_messages=AsyncMock(return_value=[]),
            create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())))
        settings = JarvisSettings(_env_file=None, RESPONSIVE_VOICE_ENABLED=True,
                                  OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=2)
        self.session = DirectRealtimeAudioSession(
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", websocket=self.app,
            context=RealtimeSessionContext(audio_session_id="offline-response-pipeline",
                owner_id=self.scope.owner_id, jarvis_session_id=self.scope.session_id,
                client_kind="rayban_audio", session=SimpleNamespace()),
            settings=settings, repository=self.repository, tool_modules=(),
            context_providers=(CurrentTimeContextProvider(),),
            now=lambda: self.now)
        # Replace acquisition/wire I/O only. Keep the actual WebRTC wrapper,
        # FIFO send lock, factories, guards, receive loop and response ownership.
        self.session.openai.connect = AsyncMock()
        self.session.openai.answer = "controlled-offline-sdp"
        self.session.openai._websocket = self.provider
        self.addAsyncCleanup(self.cleanup)
        for name, target in (("readiness", "app.realtime.voice_route_runtime._READINESS_LOGGER.warning"),
                             ("publication", "app.voice_routes.bridge._PUBLICATION_LOGGER.warning")):
            capture = patch(target)
            setattr(self, name, capture.start())
            self.addCleanup(capture.stop)
        services = SimpleNamespace(knowledge_runtime=SimpleNamespace(area_selections=SimpleNamespace(
            read=AsyncMock(return_value={"area_status": "active", "area_id": str(self.scope.area_id)}))))
        with patch("app.realtime.live_state_bridge.create_session_live_state", return_value=self.live):
            self.attachment = await install_responsive_voice(self.session, services, settings, CLIENT_CAPS,
                                                              clock=lambda: self.now)
        self.runtime = self.session._voice_route_runtime
        self.running = asyncio.create_task(self.session.run(), name="offline-real-session")
        await self.until(lambda: self.runtime.ready or self.session._closed.is_set())
        self.assertFalse(self.session._closed.is_set(), self.session._close_reason)
        self.assertTrue(self.runtime.channel.ready)
        self.assertTrue(any(event["type"] == "session.created" for event in self.app.events))
        self.storage.on_lock = self.expire_inside_adapter_guard

    async def cleanup(self):
        self.storage.on_lock = None
        if self.attachment is not None:
            await asyncio.wait_for(self.attachment.close(), 5)
        else:
            await asyncio.wait_for(self.session.close("offline-cleanup"), 5)
            await self.service.close()
        if self.running is not None:
            await asyncio.wait_for(self.running, 5)
        self.assertTrue(self.provider.closed)
        self.assertEqual(self.pool.entered, self.pool.exited)
        self.assertEqual(self.storage.entered, self.storage.exited)
        self.assertTrue(self.running is None or self.running.done())
        self.assertTrue(self.attachment is None or all(task.done() for task in self.attachment._tasks))

    async def until(self, predicate):
        try:
            async with asyncio.timeout(3):
                while not predicate():
                    await asyncio.sleep(0)
        except TimeoutError:
            print("pipeline_timeout=" + json.dumps({
                "provider_ready": self.session._realtime_context_ready.is_set(),
                "phase": self.runtime.coordinator.status.phase.value,
                "runtime_failure": self.runtime.failure,
                "channel_ready": self.runtime.channel.ready,
                "provider_events": [event["type"] for event in self.provider.writes],
                "app_events": [(event["type"], event.get("code")) for event in self.app.events],
                "tasks": [(task.get_coro().__qualname__,
                           [frame.f_code.co_name for frame in task.get_stack()])
                          for task in asyncio.all_tasks() if task is not asyncio.current_task()],
            }, sort_keys=True))
            raise

    def rows(self, logger, stage):
        return [row for call in logger.call_args_list
                if len(call.args) > 1 and (row := json.loads(call.args[1])).get("stage") == stage]

    async def observation(self, summary, *, seconds=60, expire_in_adapter=False):
        source = canonical_fixtures.memory(self.scope, source_type="observation", trust="sensor",
            source_time=self.now, recorded_at=self.now, content={"text": summary})
        self.records[source.source_id, source.revision] = source
        dependency = memory_dependency(source)
        expiry = self.now + timedelta(seconds=seconds)
        if expire_in_adapter:
            self.expiry_target = (source.source_id, expiry)
        entry = ObservationEntry(entry_id="current-visual", scope=self.scope,
            received_at=self.now, observed_at=source.source_time, fresh_until=expiry,
            sources=(dependency,), guard_binding=GuardedPayloadBinding(scope=self.scope,
                payload_root_ids=(source.payload_id,), dependencies=(dependency,)),
            evidence_kind="observed", summary=summary)
        self.assertTrue(self.producer.publish(entry).queued)
        await self.service.wait_idle()
        return source

    async def expire_inside_adapter_guard(self, roots):
        if self.expiry_target is None:
            return
        source_id, expiry = self.expiry_target
        combined = [body for identity, body in self.storage.payloads if identity in roots
                    and body.get("format") == "jarvis.live_state.voice.v1"
                    and any(item["source_id"] == str(source_id) for item in body["dependencies"])]
        if not combined:
            return
        self.expiry_locks += 1
        if self.expiry_locks == 2:
            # First guard was successful bridge validity. Expire only inside
            # the adapter's awaited canonical guard, before it can yield/write.
            self.now = expiry + timedelta(milliseconds=1)
            self.expiry_target = None

    def mutations(self):
        return [event for event in self.provider.writes
                if event["type"] in {"conversation.item.create", "conversation.item.delete"}]

    def creates(self):
        return [event for event in self.provider.writes if event["type"] == "response.create"]

    async def question_and_accept(self):
        await self.provider.inject("input_audio_buffer.speech_started", item_id="question")
        await self.provider.inject("input_audio_buffer.speech_stopped", item_id="question")
        await self.provider.inject("input_audio_buffer.committed", item_id="question")
        await self.provider.inject("conversation.item.input_audio_transcription.completed",
                                   item_id="question", transcript="What was visible in that observation?")
        await self.until(lambda: len(self.creates()) == 1)
        request = self.creates()[0]
        await self.provider.inject("response.created", response={"id": "answer",
                                                              "metadata": request["response"]["metadata"]})
        await self.provider.inject("response.output_item.added", response_id="answer", output_index=0,
            item={"id": "answer-item", "type": "message", "role": "assistant", "phase": "final_answer"})
        await self.provider.inject("output_audio_buffer.started", response_id="answer")
        await self.until(lambda: any(event["type"] == "audio.webrtc.started" for event in self.app.events))
        state = self.session._active_responses["answer"]
        self.assertFalse(state.interrupted)
        self.assertIs(state.source_snapshot, self.session._source_context.snapshot_for(request["event_id"]))
        self.assertIsNotNone(state.source_snapshot)
        return state.source_snapshot

    async def test_adapter_expiry_recovers_through_real_publisher_turn_writer_and_completion(self):
        before = len(self.mutations())
        await self.observation("expiry-candidate-before-question", seconds=2, expire_in_adapter=True)
        await self.until(lambda: bool(self.rows(self.readiness, "source_guard_rejected")))
        rejected = self.rows(self.readiness, "source_guard_rejected")[-1]
        # Aged prior context can make awaiting_fresh_context true before this
        # publication returns. Await the exact terminal receipt being asserted.
        await self.until(lambda: any(row.get("update_id") == rejected["update_id"]
                         and row.get("owned_work_joined") is True
                         for row in self.rows(self.publication, "adapter_return")) and (
                         self.runtime.coordinator._bridge.status.fenced or any(
                         row.get("update_id") == rejected["update_id"]
                         for row in self.rows(self.publication, "context_refresh_wait"))))
        returned = next(row for row in self.rows(self.publication, "adapter_return")
                        if row.get("update_id") == rejected["update_id"])
        print("adapter_expiry_pipeline_evidence=" + json.dumps({
            "source_guard": rejected, "adapter_return": returned,
            "phase": self.runtime.coordinator.status.phase.value,
            "reason": self.runtime.coordinator.status.reason, "expiry_lock_count": self.expiry_locks,
            "context_mutations": len(self.mutations()) - before}, sort_keys=True))
        self.assertEqual(rejected["error_code"], "voice_context_expired")
        self.assertFalse(rejected["guard_yielded"])
        self.assertTrue(returned["adapter_entered"])
        # The pre-delete check now authorizes scope only. This second full
        # candidate check is the actual create guard, AFTER owned deletion.
        self.assertEqual(returned["write_state"], "write_possible")
        self.assertEqual(returned["first_possible_operation"], "context_delete")
        self.assertTrue(returned["deleted_context_acknowledged"])
        self.assertFalse(returned["replacement_create_possible"])
        self.assertTrue(returned["owned_work_joined"])
        self.assertEqual(self.expiry_locks, 2)
        self.assertEqual([event["type"] for event in self.mutations()[before:]],
                         ["conversation.item.delete"])
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(self.runtime.context_publication_ready)
        self.assertFalse(self.session._context_revoked)

        await self.observation("fresh-view-before-question")
        await self.until(lambda: self.runtime.ready and
                         "fresh-view-before-question" in self.runtime.adapter._context.text)
        pin = await self.question_and_accept()
        before = len(self.mutations())
        await self.observation("ordinary-update-during-answer")
        self.assertFalse(self.runtime.context_publication_idle)
        await self.observation("newest-update-during-answer")
        await asyncio.sleep(0.12)  # Publisher tick must leave active evidence alone.
        self.assertFalse(self.runtime.context_publication_idle)
        self.assertTrue(any(getattr(entry, "summary", None) == "newest-update-during-answer"
                            for entry in self.service.snapshot().entries))
        self.assertEqual(len(self.mutations()), before)
        self.assertIs(self.session._active_responses["answer"].source_snapshot, pin)
        self.assertFalse(self.session._context_revoked)
        await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
                                   item_id="answer-item", transcript="A controlled as-of answer.")
        await self.provider.inject("response.done", response={"id": "answer", "status": "completed",
            "output": [{"id": "answer-item", "type": "message", "phase": "final_answer",
                        "content": [{"type": "output_audio"}]}]})
        await self.until(lambda: any(event["type"] == "response.done" for event in self.app.events))
        self.assertIn("answer", self.session._direct_output_pending)
        self.assertFalse(self.runtime.context_publication_idle)
        await asyncio.sleep(0.12)  # One actual publisher tick, not a latency target.
        self.assertEqual(len(self.mutations()), before)
        await self.provider.inject("output_audio_buffer.stopped", response_id="answer")
        await self.until(lambda: "newest-update-during-answer" in self.runtime.adapter._context.text)
        self.assertEqual(len(self.mutations()), before + 2)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.session._context_revoked)
        assistant = [call.kwargs for call in self.repository.create_message.await_args_list
                     if call.kwargs["role"] == "assistant"]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["source_payload_id"], pin.payload_id)
        self.assertFalse(any(event.get("code") == "no_final_answer" for event in self.app.events))

    async def test_acknowledged_delete_then_create_guard_expiry_recovers_without_redelete(self):
        adapter = self.runtime.adapter
        old_item_id = adapter._context_item
        self.assertIsNotNone(old_item_id)
        old_item = adapter._items[old_item_id]
        self.assertEqual(old_item.state, "created")
        before = len(self.mutations())
        self.storage.on_lock = None
        original_inject = self.provider.inject
        delete_acks = []

        async def inject_at_owned_delete_ack(kind, **fields):
            if kind == "conversation.item.deleted" and fields.get("item_id") == old_item_id:
                self.assertEqual(old_item.state, "delete_pending")
                self.assertIsNotNone(self.expiry_target)
                delete_acks.append(fields["item_id"])
                self.now = self.expiry_target[1] + timedelta(milliseconds=1)
                self.expiry_target = None
            await original_inject(kind, **fields)

        self.provider.inject = inject_at_owned_delete_ack
        self.addCleanup(setattr, self.provider, "inject", original_inject)

        # Expire at the exact owned-item delete ACK enqueue. The real receive
        # loop must consume it and release deletion before replacement's actual
        # source guard runs; that guard can reject before any binder/root lock.
        await self.observation("expired-after-acknowledged-delete", seconds=2,
                               expire_in_adapter=True)
        await self.until(lambda: bool(self.rows(self.readiness, "source_guard_rejected")))
        rejected = self.rows(self.readiness, "source_guard_rejected")[-1]
        await self.until(lambda: any(
            row.get("update_id") == rejected["update_id"] and row.get("owned_work_joined") is True
            for row in self.rows(self.publication, "adapter_return")) and (
                self.runtime.coordinator._bridge.status.fenced
                or self.runtime.coordinator._bridge.status.reason == "context_expired_awaiting_snapshot"))
        returned = next(row for row in self.rows(self.publication, "adapter_return")
                        if row.get("update_id") == rejected["update_id"])
        mutations = self.mutations()[before:]
        print("acknowledged_delete_expiry_pipeline_evidence=" + json.dumps({
            "source_guard": rejected, "adapter_return": returned,
            "phase": self.runtime.coordinator.status.phase.value,
            "reason": self.runtime.coordinator.status.reason,
            "expiry_lock_count": self.expiry_locks,
            "targeted_delete_ack_count": len(delete_acks),
            "old_item_state": old_item.state,
            "old_item_still_owned": old_item_id in adapter._items,
            "replacement_item": adapter._context_item,
            "mutation_types": [event["type"] for event in mutations],
        }, sort_keys=True))
        self.assertEqual(delete_acks, [old_item_id])
        self.assertEqual(rejected["error_code"], "voice_context_expired")
        self.assertFalse(rejected["guard_yielded"])
        self.assertEqual(old_item.state, "deleted")
        self.assertNotIn(old_item_id, adapter._items)
        self.assertIsNone(adapter._context_item)
        self.assertEqual([event["type"] for event in mutations], ["conversation.item.delete"])
        self.assertEqual(mutations[0]["item_id"], old_item_id)
        self.assertTrue(returned["adapter_entered"])
        self.assertEqual(returned["write_state"], "write_possible")
        self.assertEqual(returned["first_possible_operation"], "context_delete")
        self.assertTrue(returned["owned_work_joined"])
        # The pre-fix counterexample reaches the assertions above, then fails
        # here with context_receipt_unconfirmed. An ACK is not an unwritten try.
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(returned.get("deleted_context_acknowledged"))
        self.assertIs(returned.get("replacement_create_possible"), False)
        self.assertTrue(any(row.get("update_id") == rejected["update_id"]
                            and row.get("outcome") == "expired_after_acknowledged_delete"
                            for row in self.rows(self.publication, "context_refresh_wait")))
        self.assertTrue(self.runtime.coordinator._bridge.awaiting_fresh_context)
        self.assertTrue(self.runtime.context_publication_ready)
        self.assertFalse(self.session._context_revoked)
        self.assertFalse(self.creates())

        await self.observation("fresh-replacement-after-acknowledged-delete")
        await self.until(lambda: self.runtime.ready and
                         "fresh-replacement-after-acknowledged-delete" in adapter._context.text)
        mutations = self.mutations()[before:]
        self.assertEqual([event["type"] for event in mutations],
                         ["conversation.item.delete", "conversation.item.create"])
        self.assertNotEqual(mutations[1]["item"]["id"], old_item_id)
        self.assertIn("fresh-replacement-after-acknowledged-delete", adapter._context.text)
        self.assertNotIn("expired-after-acknowledged-delete", adapter._context.text)
        pin = await self.question_and_accept()
        await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
                                   item_id="answer-item", transcript="A controlled recovered answer.")
        await self.provider.inject("response.done", response={"id": "answer", "status": "completed",
            "output": [{"id": "answer-item", "type": "message", "phase": "final_answer",
                        "content": [{"type": "output_audio"}]}]})
        await self.until(lambda: any(event["type"] == "response.done" for event in self.app.events))
        await self.provider.inject("output_audio_buffer.stopped", response_id="answer")
        await self.until(lambda: "answer" not in self.session._direct_output_pending)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.session._context_revoked)
        assistant = [call.kwargs for call in self.repository.create_message.await_args_list
                     if call.kwargs["role"] == "assistant"]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["source_payload_id"], pin.payload_id)

    async def test_original_withdrawal_fences_actual_accepted_output_after_additive_update(self):
        pin = await self.question_and_accept()
        before = len(self.mutations())
        await self.observation("ordinary-addition-not-a-withdrawal")
        await asyncio.sleep(0.12)
        self.assertFalse(self.runtime.context_publication_idle)
        self.assertEqual(len(self.mutations()), before)
        self.assertFalse(self.session._context_revoked)
        self.storage.rejected.add(pin.sources[0])
        await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
                                   item_id="answer-item", transcript="Must not be forwarded.")
        await self.until(lambda: self.session._context_revoked)
        await self.until(lambda: self.session._closed.is_set())
        self.assertFalse(any(event["type"] == "transcript.done" for event in self.app.events))
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(any(call.kwargs["role"] == "assistant"
                             for call in self.repository.create_message.await_args_list))
        self.assertEqual(self.session._close_reason, "live_context_rebuild_required")
