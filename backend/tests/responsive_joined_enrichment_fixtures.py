"""Controlled I/O for joined production voice, memory and perception execution.

No route, bridge, composer, pipeline, analyzer, input capture or source-binding
output is fabricated. SQL storage, retrieval replies and wire endpoints are
fixtures. This proves composition, not provider latency or audible playback.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError, SourceVersion
from app.live_state.runtime import create_session_live_state
from app.live_state.projection import prepare_voice_context, select_voice_context
from app.perception.frame_window import SensoryVisualFrame
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.live_state_bridge import SessionLiveStateBridge
from app.realtime.models import RealtimeSessionContext
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.runtime.feature_config import MemoryAuthority
from app.settings import JarvisSettings
from app.tasks.models import digest
from app.voice_routes.contracts import CommandReceipt, RouteIdentity, RouteStart
from tests import test_responsive_context_scenarios as scenarios
from test_realtime_shared_view import _Wire
from test_voice_route_runtime import CLIENT_CAPS


A_MARKER = "HISTORICAL_A_41_5"
B_MARKER = "HISTORICAL_B_12_75"


class JoinedStore(scenarios.ScenarioStore):
    """Extend the established canonical frame store with scoped memory rows."""
    def __init__(self):
        super().__init__()
        self.originals, self.memory_edges = {}, {}
        self.transaction_task = None

    @asynccontextmanager
    async def transaction(self, **kwargs):
        async with super().transaction(**kwargs):
            self.transaction_task = asyncio.current_task()
            try:
                yield self
            finally:
                self.transaction_task = None

    def available(self, owner, identity, seen=None):
        for row in self.memory_edges.values():
            if row[0] == owner and row[1] == identity:
                source = self.originals.get(row[3])
                if source is None or (source.revision, source.generation, source.payload_id,
                                     source.content_digest) != (row[4], row[5], row[6], row[7]):
                    return False
        return super().available(owner, identity, seen)

    def ancestors(self, identity, seen=None):
        seen = set() if seen is None else set(seen)
        if identity in seen:
            return seen
        seen.add(identity)
        for parent in self.copies.get(identity, {}).get("parents", ()):
            seen.update(self.ancestors(parent, seen))
        return seen

    async def fetch(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("select id from jarvis_knowledge_sources"):
            self.sql(sql)
            return [{"id": identity} for identity in args[2] if identity in self.originals
                and (self.originals[identity].owner_id, self.originals[identity].area_id) == args[:2]]
        if "from jarvis_visual_focuses f" in normalized:
            self.sql(sql)
            return []
        return await super().fetch(sql, *args)

    async def fetchval(self, sql, *args):
        if " ".join(sql.lower().split()).startswith("insert into jarvis_context_payloads"):
            self.sql(sql)
            return await self.payload(self, args[0], json.loads(args[1]))
        return await super().fetchval(sql, *args)

    async def executemany(self, sql, rows):
        if "insert into jarvis_context_memory_sources" in " ".join(sql.lower().split()):
            self.sql(sql)
            for row in rows:
                assert self.available(row[0], row[6])
                assert self.copies[row[1]]["owner"] == row[0]
                self.memory_edges.setdefault((row[0], row[1], row[3], row[4]), tuple(row))
            return
        return await super().executemany(sql, rows)


class ClientWire:
    """Actual client-channel envelopes and explicit simulated fence receipts."""
    def __init__(self, clock):
        self.clock, self.session = clock, None
        self.messages = []
        self.closed = False

    async def send_json(self, payload):
        self.messages.append(payload)
        if payload["type"] == "voice_route.selected":
            await self.session._handle_client_event({"type": "voice_route.ready", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "route": payload["route"],
                "client_event_id": str(uuid4())})
        elif payload["type"] in {"voice_route.interrupt", "voice_route.close"}:
            command = payload["payload"]
            receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                operation="close" if payload["type"] == "voice_route.close" else "interrupt_speech",
                observed_at=self.clock(), status="completed", output_fenced=True)
            await self.session._handle_client_event({"type": "voice_route.event", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "payload": receipt.model_dump(mode="json"),
                "client_event_id": str(uuid4())})
        elif payload["type"] == "session.context_invalidated":
            await self.session._handle_client_event({"type": "session.context_cleared", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "reset_id": payload["context_reset"]["id"]})

    async def send_text(self, text):
        await self.send_json(json.loads(text))

    async def close(self, *args, **kwargs):
        self.closed = True


class ProviderWire(_Wire):
    async def send(self, raw):
        await super().send(raw)
        event = json.loads(raw)
        if event["type"] == "response.cancel" and event.get("response_id"):
            self.incoming.put_nowait({"type": "response.done", "response": {
                "id": event["response_id"], "status": "cancelled", "output": []}})


class JoinedEnrichmentFixture:
    def __init__(self, test):
        self.test = test
        self.case = scenarios.ResponsiveContextScenarios(methodName="runTest")
        self.release_a, self.entered_a = asyncio.Event(), asyncio.Event()
        self.calls, self.active, self.max_active = [], 0, 0
        self.attachment = self.session = self.receiver = self.live = None

    async def setup(self, *, wait_for_retrieval=True):
        with patch.object(scenarios, "ScenarioStore", JoinedStore):
            await self.case.asyncSetUp()
        self.db = self.case.db
        self.case.scope = self.case.scope.model_copy(update={"area_id": uuid4()})
        self.scope = self.case.scope
        self.db.areas.add((self.scope.owner_id, self.scope.area_id))
        self.a = self.memory(A_MARKER + ": experiment A requires 41.5 units, not 415.")
        self.b = self.memory(B_MARKER + ": experiment B requires 12.75 units, not 1275.")
        self.intent_a = self.intent("Help with experiment A")

        @asynccontextmanager
        async def transaction(owner):
            self.test.assertEqual(owner, self.scope.owner_id)
            async with self.db.transaction():
                yield self.db

        async def area(connection, owner, identity):
            if connection is not self.db or (owner, identity) not in self.db.areas:
                raise SourceError("knowledge_area_unavailable", "Fixture area unavailable")

        async def read_many(connection, owner, identity, requests):
            await area(connection, owner, identity)
            values = []
            for source_id, revision in requests:
                source = self.db.originals.get(source_id)
                if (source is None or (source.owner_id, source.area_id) != (owner, identity)
                        or revision not in {None, source.revision}
                        or not self.db.available(owner, source.payload_id)):
                    raise SourceError("knowledge_source_unavailable", "Original unavailable")
                values.append(source)
            return values

        async def resolve(owner, area_id, document):
            for source in self.db.originals.values():
                if (source.owner_id, source.area_id, source.document_id) == (owner, area_id, document):
                    if self.db.available(owner, source.payload_id):
                        return source
            raise SourceError("knowledge_source_unavailable", "Original unavailable")

        async def revalidate(source):
            current = await resolve(source.owner_id, source.area_id, source.document_id)
            if current != source:
                raise SourceError("knowledge_source_changed", "Original changed")
            return current

        async def recall(owner, area_id, query):
            self.test.assertEqual((owner, area_id), (self.scope.owner_id, self.scope.area_id))
            self.test.assertIsNot(self.db.transaction_task, asyncio.current_task())
            self.calls.append(query)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            selected = self.b if "experiment B" in query else self.a
            try:
                if len(self.calls) == 1:
                    self.entered_a.set()
                    await self.release_a.wait()
                return {"status": "available", "sources": [selected.original_evidence() | {"content_included": True}]}
            finally:
                self.active -= 1

        self.sources = SimpleNamespace(pool=self.db, transaction=transaction, _area=area,
            _read_many=read_many, resolve_document=resolve, revalidate=revalidate,
            current_snapshot=AsyncMock(side_effect=AssertionError("Relevant lookup must replace inventory loading")))
        selection = {"area_id": str(self.scope.area_id), "area_status": "active",
                     "selection_revision": 0, "selection_event_id": None}
        self.services = self.case.services
        self.services.memory_authority = MemoryAuthority.HINDSIGHT
        self.services.working_memory_service = self.case.working
        self.services.knowledge_runtime = SimpleNamespace(sources=self.sources,
            area_selections=SimpleNamespace(read_in_transaction=AsyncMock(return_value=selection)),
            context_for=lambda owner: self.case.context if owner == self.scope.owner_id else None,
            retrieval=SimpleNamespace(recall=recall))
        self.live = create_session_live_state(self.services, scope=self.scope, enabled=True,
            perception_brief_enabled=True, clock=lambda: self.case.now, interval_seconds=.02,
            timeout_seconds=1, enrichment_options={"debounce_seconds": 0, "deadline_seconds": 5})
        self.db.bind_io(self.live.binder)
        self.case.lives.append(self.live)
        await self.live.start()
        self.activity = "cooking"
        self.lane = await self.case.pipeline(self.live, lambda **kwargs: self.case.output(self.activity))
        # Camera ownership is the real per-connection lease, not a test boolean.
        self.lease = self.lane.harness.frame_window.begin_stream(self.scope.owner_id, self.scope.session_id,
                                                               self.lane.harness.visual_session_id)
        await self.setup_voice()
        # Voice and the minimal perception path do not depend on selected
        # history. Individual retrieval scenarios may still wait for their
        # deterministic barrier, but only after the actual route is available.
        await self.until(lambda: self.live.perception_brief_bridge._binding is not None)
        if wait_for_retrieval:
            await asyncio.wait_for(self.entered_a.wait(), 2)

    def memory(self, text):
        content = {"text": text}
        value = SourceVersion(owner_id=self.scope.owner_id, area_id=self.scope.area_id, source_id=uuid4(),
            revision=1, generation=1, payload_id=uuid4(), document_id=str(uuid4()), bank_name="offline",
            content_digest=digest(content), source_type="note", trust="user_input", source_time=self.case.now - timedelta(days=1),
            time_basis="user report", recorded_at=self.case.now, content=content,
            is_current=True, ingestion_status="not_queued")
        self.db.originals[value.source_id] = value
        self.db.copies[value.payload_id] = {"owner": value.owner_id, "content": content, "parents": set()}
        return value

    def intent(self, text):
        source = self.db.event(self.scope.owner_id, self.scope.session_id, at=self.case.now, content={"text": text})
        record = self.db.record(source.request_event_id)
        self.case.context.snapshot.return_value = {"recent": [record], "pinned": [], "watermark": record.sequence}
        return record

    async def setup_voice(self):
        self.client = ClientWire(lambda: self.case.now)
        self.session = DirectRealtimeAudioSession(offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=self.client, context=RealtimeSessionContext(audio_session_id="joined-offline",
                owner_id=self.scope.owner_id, jarvis_session_id=self.scope.session_id,
                client_kind="rayban_audio", session=SimpleNamespace()), settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
                get_recent_messages=AsyncMock(return_value=[])), tool_modules=(), context_providers=())
        self.client.session = self.session
        self.session._source_context.pool = self.db
        self.wire = ProviderWire(self.session)
        self.session.openai._websocket = self.wire
        identity = RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1)
        initial = await prepare_voice_context(select_voice_context(self.live.service.snapshot(), now=self.case.now),
            identity, bind=self.live.bind, clock=lambda: self.case.now)
        start = RouteStart(route=identity, command_id=uuid4(), requested_at=self.case.now,
                           authorization_ref=self.scope.session_id, initial_context=initial)

        @asynccontextmanager
        async def guard(update):
            if update.route != identity:
                raise SourceError("voice_context_not_admitted", "Foreign route")
            manager = self.live.scope_guard() if update.guard_binding is None else self.live.guard(update.guard_binding)
            async with manager as connection:
                if update.expires_at is not None and update.expires_at <= self.case.now:
                    raise SourceError("voice_context_expired", "Expired update")
                yield connection

        async def validity(update):
            try:
                async with guard(update):
                    return True
            except (SourceError, TimeoutError):
                return False

        async def authorize(candidate):
            async with self.live.scope_guard():
                return candidate == start and not self.session._closed.is_set()

        self.route = RealtimeVoiceRouteRuntime(self.session, start, CLIENT_CAPS, atomic_guard=guard,
            payload_roots=lambda update: update.guard_binding.payload_root_ids if update.guard_binding else (),
            validity=validity, authorize_start=authorize, clock=lambda: self.case.now,
            ready_timeout=2, control_timeout=.5)
        self.session.install_voice_route_runtime(self.route)
        self.attachment = SessionLiveStateBridge(self.session, self.live, self.route,
            clock=lambda: self.case.now, visual_pipeline=self.lane.pipeline)
        self.attachment.registered()
        self.attachment.start()
        self.receiver = asyncio.create_task(self.receive())
        self.session._realtime_context_ready.set()
        await self.until(lambda: self.route.ready)

    async def receive(self):
        self.session._provider_receive_task = asyncio.current_task()
        while True:
            event = await self.wire.incoming.get()
            try:
                await self.session._handle_openai_event(event)
            finally:
                self.wire.incoming.task_done()

    async def emit(self, kind, **payload):
        self.wire.incoming.put_nowait({"type": kind, **payload})
        await asyncio.wait_for(self.wire.incoming.join(), 1)
        if self.receiver.done():
            await self.receiver

    async def ask(self, item):
        await self.emit("input_audio_buffer.speech_started", item_id=item)
        await self.emit("input_audio_buffer.committed", item_id=item)

    async def frame(self, sequence):
        """Exercise real camera ownership before the normal frame pipeline."""
        self.case.now = datetime.now(UTC)
        frame = self.lane.harness.frame(sequence=sequence, captured_at=self.case.now)
        image_bytes = b"synthetic-image-not-a-real-photo"
        accepted = self.lane.harness.frame_window.put(SensoryVisualFrame(
            frame_id=frame.id, owner_id=frame.owner_id, session_id=frame.session_id,
            audio_session_id=frame.audio_session_id, visual_session_id=frame.visual_session_id,
            frame_sequence=frame.frame_sequence, source=frame.source, captured_at=frame.captured_at,
            server_received_at=frame.server_received_at, content_type=frame.content_type,
            width=frame.width, height=frame.height, storage_bucket=frame.storage_bucket,
            storage_path=frame.storage_path, raw_image_expires_at=frame.raw_image_expires_at,
            image_bytes=image_bytes), stream_lease=self.lease)
        self.test.assertTrue(accepted, "Fixture camera lease did not admit its frame")
        self.db.frames[frame.id] = frame

        async def diagnostic(value):
            self.lane.diagnostics.append(value)

        await self.lane.pipeline.enqueue(frame=frame, image_bytes=image_bytes, send_diagnostic=diagnostic)
        await self.lane.pipeline.drain()
        return frame

    async def until(self, predicate):
        async with asyncio.timeout(3):
            while not predicate():
                if self.receiver is not None and self.receiver.done():
                    await self.receiver
                await asyncio.sleep(.002)

    def brief_text(self):
        composition = self.live.perception_brief_bridge.composition
        return " ".join(item.finding for item in composition.brief.prior_evidence) if composition else ""

    def voice_text(self):
        return "\n".join(part.get("text", "") for event in self.wire.events
            if event["type"] == "conversation.item.create" for part in event.get("item", {}).get("content", ()))

    async def close(self):
        self.release_a.set()
        try:
            if self.attachment is not None:
                await self.attachment.close("joined_fixture_complete")
            elif self.session is not None:
                await self.session.close("joined_fixture_complete")
        finally:
            if self.receiver is not None:
                self.receiver.cancel()
                await asyncio.gather(self.receiver, return_exceptions=True)
            if hasattr(self.case, "pipelines"):
                await self.case.asyncTearDown()
