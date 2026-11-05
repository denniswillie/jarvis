"""P2 offline contextual scenarios, not a visual detection or hardware benchmark.

Actual EpisodeCheckOwner, Live State producers/binders, SessionPerceptionRuntime,
composer, input capture, analyzer/parser, pipeline, working-memory reducer and
PerceptionEffectGuard execute. Only storage and visual-model responses are fixtures.

Coverage limits deliberately NOT converted into passing acceptance:
- Conditional relevance uses original activity evidence, not fixture activation.
- Frozen question/check handle transport is an independent requirement probe.
- Visual/trusted resolution and focus-owned check/episode resolution remain
  unqualified; non-resolution assertions establish safety, not those capabilities.
- Delayed intent-specific Hindsight enrichment, notification-route deduplication,
  and remote provider-context retirement need their missing/integrated adapters.
- Restart covers durable identity, not recovery of actual retained camera pixels.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.images import ImageReference
from app.knowledge.observations import OBSERVATION_NAMESPACE, ObservationPublication
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import context_dependency
from app.live_state.models import AggregateRef, SessionScope, TaskEntry
from app.live_state.projection import select_voice_context
from app.live_state.runtime import LiveStateRuntime, create_session_live_state
from app.live_state.perception import CheckResolution, EvidenceReference
from app.perception.applicability import PerceptionEffectUnavailable
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.models import SessionWorkingMemoryState, WorkingMemoryWriteResult
from app.perception.references import PerceptionReferences
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy
from app.reminders.duties import ResponsibilityRepository
from app.reminders.episode_checks import EpisodeCheckOwner, reference
from app.reminders.episode_models import AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName, OwnerWorkingMemoryUpdatedPayload
from app.runtime.feature_config import MemoryAuthority
from app.tools.inputs import ToolInputError
from app.visual_memory.pipeline import VisualAnalysisPipeline
from tests.test_live_state_episode_context import IntegrationStore
from test_fast_perception_pipeline import _Harness, _BehaviorRepository


class ScenarioStore(IntegrationStore):
    """External persistence boundary for canonical capture and working-state CAS."""

    def __init__(self):
        super().__init__()
        self.observations, self.observation_roots = {}, {}
        self.working_state = None
        self.applied = []
        self.areas = set()
        self.frames, self.observation_events = {}, {}

    def record(self, identity):
        record = super().record(identity)
        observation_id = next((key for key, value in self.observation_events.items() if value == identity), None)
        return (record.model_copy(update={"source_namespace": OBSERVATION_NAMESPACE,
                                          "source_event_id": str(observation_id)})
                if observation_id is not None else record)

    def available(self, owner, identity, seen=None):
        if not super().available(owner, identity, seen):
            return False
        # Canonical observation events share their removable derived payload.
        # Preserve that payload's input parents as well as event availability.
        if identity in self.observation_roots.values():
            seen = set() if seen is None else seen
            return all(self.available(owner, parent, seen | {identity})
                       for parent in self.copies[identity]["parents"])
        return True

    async def fetch(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if "select id,availability from jarvis_context_payloads" in normalized:
            self.sql(sql)
            return [{"id":identity, "availability":"available" if self.available(args[0], identity) else "deleted"}
                    for identity in args[1] if identity in self.copies or any(e["payload_id"] == identity for e in self.events.values())]
        if "jarvis_knowledge_source_versions" in normalized and "jarvis_observation_sources" not in normalized:
            self.sql(sql)
            return []
        if "select observation_id,payload_id from jarvis_observation_sources" in normalized:
            self.sql(sql)
            return [{"observation_id":identity, "payload_id":self.observation_roots[identity]}
                    for identity in args[1] if identity in self.observation_roots]
        if "select id,primary_visual_frame_id from perception_observations" in normalized:
            self.sql(sql)
            return [{"id":o.id, "primary_visual_frame_id":o.primary_visual_frame_id}
                    for o in self.observations.values() if o.owner_id == args[0] and o.session_id == args[1]
                    and o.primary_visual_frame_id in args[2]]
        return await super().fetch(sql, *args)

    async def fetchval(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("select e.id from jarvis_observation_sources"):
            # PerceptionLiveAdapter performs this read outside a write transaction.
            self.queries.append(normalized)
            observation = self.observations.get(args[2])
            return (self.observation_events.get(args[2]) if observation is not None
                    and (observation.owner_id, observation.session_id) == args[:2]
                    and args[3] == OBSERVATION_NAMESPACE else None)
        if "select lock_jarvis_copy_source(" in normalized:
            self.sql(sql)
            return "available" if self.available(args[0], args[1]) else "unavailable"
        if "select payload_id from jarvis_observation_sources" in normalized:
            self.sql(sql)
            return self.observation_roots.get(args[2])
        if normalized.startswith("select status from jarvis_knowledge_areas"):
            self.sql(sql)
            return "active" if (args[0], args[1]) in self.areas else None
        return await super().fetchval(sql, *args)

    async def execute(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("insert into jarvis_context_event_dependencies"):
            return await self.executemany(sql, (args,))
        if normalized.startswith("update jarvis_context_payloads"):
            self.sql(sql)
            if args[1] in self.copies and not any(args[1] in row["parents"] for row in self.copies.values()):
                self.copies[args[1]]["content"] = None
            return
        return await super().execute(sql, *args)

    async def get_session_working_memory(self, *, owner_id, session_id):
        value = self.working_state
        return value if value is None or (value.owner_id, value.session_id) == (owner_id, session_id) else None

    async def compare_and_swap_session_working_memory(self, *, owner_id, session_id, expected_revision,
                                                     observation_id, state, expires_at, effect_guard=None):
        @asynccontextmanager
        async def lease():
            if effect_guard is not None:
                async with effect_guard() as connection:
                    assert connection is self
                    yield
            else:
                async with self.transaction():
                    yield

        async with lease():
            current = self.working_state
            assert expected_revision == (current.revision if current is not None else 0)
            observation = self.observations[observation_id]
            self.working_state = SessionWorkingMemoryState(owner_id=owner_id, session_id=session_id,
                revision=expected_revision + 1, last_observation_id=observation_id,
                last_observation_ended_at=observation.ended_at, state=deepcopy(state), expires_at=expires_at,
                created_at=current.created_at if current is not None else observation.ended_at,
                updated_at=observation.ended_at)
            self.applied.append(observation_id)
            return WorkingMemoryWriteResult(True, self.working_state)


class ResponsiveContextScenarios(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime.now(UTC)
        self.db = ScenarioStore()
        self.scope = SessionScope(owner_id="owner", session_id=uuid4())
        self.db.sessions.add((self.scope.owner_id, self.scope.session_id))
        self.repo = ResponsibilityRepository(self.db)
        self.owner = EpisodeCheckOwner(self.repo, enabled=True, clock=lambda:self.now)
        await self.owner.initialize()
        self.bus = DomainEventBus()
        self.working = WorkingMemoryService(repository=self.db,
            policy=WorkingMemoryPolicy(ttl_seconds=120, entity_ttl_seconds=120, visible_text_ttl_seconds=60,
                max_entities=20, max_recent_changes=10), domain_event_bus=self.bus, now=lambda:self.now)
        self.context = SimpleNamespace(pool=self.db, snapshot=AsyncMock(return_value={"recent":[], "pinned":[]}))
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY, _closed=False,
            background_task_context_service=SimpleNamespace(repository=self.context), background_task_service=None,
            db_pool=self.db, domain_event_bus=self.bus, responsibility_issue=None,
            attention_service=SimpleNamespace(_closed=False, responsibilities=SimpleNamespace(episode_checks=self.owner)))
        self.lives, self.pipelines, self.releases = [], [], []
        self.spec = EpisodeCheckSpec(title="Kitchen shutdown")
        self.draft = EpisodeCheckDraft(question="What state do the stove controls indicate?",
            object_association="kitchen:stove", observation_window_expires_at=self.now + timedelta(minutes=10))
        self.admitted, self.source = await self.admit()

    async def asyncTearDown(self):
        for release in self.releases:
            release.set()
        for pipeline in self.pipelines:
            await pipeline.close()
        for live in self.lives:
            await live.close()

    def source_event(self, *, kind="user_transcript", trust="user_input"):
        return self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now, kind=kind, trust=trust)

    async def admit(self, **kwargs):
        source = self.source_event()
        request = AdmitEpisodeCheck(spec=kwargs.pop("spec", self.spec), check=kwargs.pop("check", self.draft), **kwargs)
        return await self.owner.admit(self.scope, source, request), source

    def command(self, result, operation, **kwargs):
        check = result.check
        return ChangeEpisodeCheck(responsibility_id=check.canonical_owner_ref.source_id,
            expected_revision=check.canonical_owner_ref.revision, occurrence_id=check.id,
            expected_check_revision=check.revision, episode_id=check.episode_or_object_ref.source_id,
            expected_episode_revision=check.episode_or_object_ref.revision, operation=operation, **kwargs)

    async def change(self, operation, **kwargs):
        return await self.owner.change(self.scope, self.source_event(), self.command(self.admitted, operation, **kwargs))

    async def until(self, predicate, detail=lambda:"condition", timeout=2):
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.002)
        except TimeoutError:
            self.fail("Scenario did not reach boundary: " + str(detail()))

    async def live(self):
        # The same production adapter supplied by the Hindsight factory, with
        # scoped external storage fixtures and no historical-memory backend.
        live = LiveStateRuntime(scope=self.scope, sources=SourceRepository(self.db), context=self.context,
            area_selections=None, access_check=lambda owner:None, working_memory=self.working,
            episode_owner=self.owner, bus=self.bus,
            clock=lambda:self.now, interval_seconds=0.02, timeout_seconds=2, perception_brief_enabled=True)
        self.db.bind_io(live.binder)
        # There is no explicit visual-focus row in these storage fixtures.
        live.perception_runtime.focus_inputs.read = AsyncMock(return_value=None)
        self.lives.append(live)
        await live.start()
        await self.until(lambda: live.perception_brief_bridge._binding is not None
            and bool(live.perception_brief_bridge.composition.brief.questions),
            lambda:(live.worker.last_error, live.episode_context.worker.last_error, live.perception_runtime.worker.last_error))
        return live

    async def pipeline(self, live, generate):
        harness = _Harness(static=True)
        await harness.pipeline.close()
        harness.session_id = self.scope.session_id
        harness.event_bus = self.bus
        captured, requests, diagnostics = [], [], []
        model = SimpleNamespace(provider="offline", model="synthetic-scene",
            generate_temporal_json=AsyncMock(side_effect=generate))

        @asynccontextmanager
        async def transaction(tasks, owner):
            assert tasks is live.binder.tasks and owner == self.scope.owner_id
            async with self.db.acquire() as connection, connection.transaction():
                yield connection

        publication_context = SimpleNamespace(tasks=live.binder.tasks,
            publication=SimpleNamespace(sources=live.sources, area_id=live.scope.area_id, transaction=transaction),
            voice_history_in_transaction=AsyncMock(return_value={"records":[], "total":0}))
        publisher = ObservationPublication(publication_context, owner_id=self.scope.owner_id)
        prepare = publisher.prepare_inputs

        async def capture(request, context):
            prepared, receipt = await prepare(request, context)
            requests.append(prepared)
            captured.append(receipt)
            return prepared, receipt

        publisher.prepare_inputs = capture

        async def insert(*, observation, frame_links, analysis_input=None):
            assert analysis_input is not None
            async with publisher.transaction(observation.owner_id) as connection:
                await publisher.inputs.check(connection, analysis_input, owner_id=observation.owner_id,
                    session_id=observation.session_id, frame_ids=tuple(link.visual_frame_id for link in frame_links))
                self.db.observations[observation.id] = observation
                root = await self.db.payload(connection, observation.owner_id,
                    {"observation_id":str(observation.id), "summary":observation.summary},
                    source_payload_ids=(analysis_input.payload_id,))
                self.db.observation_roots[observation.id] = root
                images = []
                for link in frame_links:
                    frame = self.db.frames[link.visual_frame_id]
                    images.append(ImageReference(frame_id=frame.id, owner_id=frame.owner_id,
                        session_id=frame.session_id, source="historical", captured_at=frame.captured_at,
                        content_type=frame.content_type, width=frame.width, height=frame.height,
                        visual_session_id=frame.visual_session_id, frame_sequence=frame.frame_sequence).to_metadata())
                original = self.db.event(observation.owner_id, observation.session_id, at=observation.ended_at,
                    kind="observation", trust="derived", payload_id=root,
                    content={"origin":"perception_observations",
                        "record":json.loads(json.dumps(asdict(observation), default=str)), "images":images,
                        "frame_links":[{"frame_id":str(link.visual_frame_id)} for link in frame_links]})
                self.db.observation_events[observation.id] = original.request_event_id

        working = self.working
        session_repository = SimpleNamespace(
            get_session=AsyncMock(return_value=SimpleNamespace(id=self.scope.session_id, status="active",
                behavior_instructions="", metadata={})), get_recent_messages=AsyncMock(return_value=[]))
        pipeline = VisualAnalysisPipeline(settings=harness.pipeline._settings, repository=harness.visual_repository,
            session_repository=session_repository, behavior_repository=_BehaviorRepository(),
            interrupt_service=harness.interrupt_service, model_limiter=PriorityAsyncLimiter(1),
            fast_analyzer=FastPerceptionAnalyzer(model),
            perception_repository=SimpleNamespace(insert_observation=insert, observation_publication=None),
            sensory_frame_window=harness.frame_window, domain_event_bus=self.bus, working_memory_service=working)
        pipeline.set_observation_publication(publisher)
        pipeline.register_live_brief(live.perception_brief_bridge)
        self.pipelines.append(pipeline)
        return SimpleNamespace(pipeline=pipeline, harness=harness, model=model, requests=requests,
            captured=captured, diagnostics=diagnostics, publisher=publisher, working=working)

    async def frame(self, lane, sequence):
        self.now = datetime.now(UTC)
        frame = lane.harness.frame(sequence=sequence, captured_at=self.now)
        self.db.frames[frame.id] = frame
        lane.harness.put(frame, b"synthetic-image-not-a-real-photo")

        async def diagnostic(value):
            lane.diagnostics.append(value)

        await lane.pipeline.enqueue(frame=frame, image_bytes=b"synthetic-image-not-a-real-photo",
                                   send_diagnostic=diagnostic)
        await lane.pipeline.drain()
        return frame

    @staticmethod
    def output(activity, *, summary=None, entities=()):
        return json.dumps({"summary":summary or activity, "scene":{"label":"home", "confidence":0.8},
            "activity":{"label":activity, "confidence":0.75}, "entities":list(entities),
            "starts_new_event":True, "boundary_reason":"Synthetic activity transition", "confidence":0.75})

    async def test_cooking_cleaning_departure_preserves_unconfirmed_shutdown(self):
        live = await self.live()
        states = iter(("cooking", "cleaning", "departing"))
        lane = await self.pipeline(live, lambda **kwargs:self.output(next(states)))
        for sequence in range(1, 4):
            await self.frame(lane, sequence)
        self.assertFalse(lane.harness.visual_repository.failed)
        self.assertEqual([o.activity for o in self.db.observations.values()], ["cooking", "cleaning", "departing"])
        self.assertEqual(len(self.db.applied), 3)
        self.assertIn("departing", str(self.db.working_state.state))
        row = self.db.occurrences[self.admitted.check.id]
        self.assertEqual((row["status"], row["report_state"], row["check_lifecycle"]), ("open", "unknown", "tracking"))
        self.assertIsNone(row["check_resolution"])
        self.assertTrue(all(request.perception_brief.scene_change_cues for request in lane.requests))

    async def test_old_shutdown_report_cannot_complete_new_cooking_episode(self):
        report = self.source_event()
        episode = self.admitted.check.episode_or_object_ref
        resolution = CheckResolution(basis="user_report", resolved_at=self.now,
            applicable_episode_or_object_ref=episode, source_refs=(EvidenceReference(
                source=reference("context_event", report.request_event_id, 1), captured_at=self.now,
                episode_or_object_ref=episode),))
        done = await self.owner.change(self.scope, report, self.command(self.admitted, "resolve", resolution=resolution))
        next_episode, _ = await self.admit(responsibility_id=done.check.canonical_owner_ref.source_id,
                                          expected_revision=done.check.canonical_owner_ref.revision)
        live = await self.live()
        self.assertNotEqual(next_episode.check.episode_or_object_ref.source_id, episode.source_id)
        with self.assertRaises(ToolInputError) as caught:
            await self.owner.change(self.scope, self.source_event(), self.command(next_episode, "resolve", resolution=resolution))
        self.assertEqual(caught.exception.code, "resolution_episode_conflict")
        self.assertIsNone(self.db.occurrences[next_episode.check.id]["check_resolution"])
        self.assertEqual(live.perception_brief_bridge.composition.brief.questions[0].check_ref.source_id, next_episode.check.id)

    async def test_belongings_out_of_view_and_unclear_do_not_complete_or_erase_last_seen(self):
        live = await self.live()
        calls = 0

        async def generate(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return self.output("placing belongings", summary="An umbrella is on the table.", entities=({
                    "new_local_key":"umbrella", "type":"object", "name":"umbrella", "state":"on table",
                    "attributes":{}, "last_visible_frame_id":"frame:0", "confidence":0.9},))
            return self.output("departing", summary="The umbrella is not visible; the controls are unclear.")

        lane = await self.pipeline(live, generate)
        await self.frame(lane, 1)
        self.assertFalse(lane.harness.visual_repository.failed)
        before = deepcopy(self.db.working_state.state["entities"])
        original = next(iter(self.db.observations.values()))
        await self.frame(lane, 2)
        self.assertFalse(lane.harness.visual_repository.failed)
        self.assertEqual(self.db.working_state.state["entities"], before)
        self.assertEqual(self.db.observations[original.id].ended_at, original.ended_at)
        self.assertEqual(self.db.occurrences[self.admitted.check.id]["report_state"], "unknown")
        self.assertIsNone(self.db.occurrences[self.admitted.check.id]["check_resolution"])

    async def during_analysis(self, action):
        live = await self.live()
        entered, release = asyncio.Event(), asyncio.Event()
        self.releases.append(release)

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return self.output("checking controls", summary="Controls appear visible; shutdown is unconfirmed.")

        lane = await self.pipeline(live, generate)
        work = asyncio.create_task(self.frame(lane, 1))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await action(live, lane)
        finally:
            release.set()
            await work
        return live, lane

    async def test_replacement_during_analysis_keeps_only_historical_result(self):
        async def replace_goal(live, lane):
            await self.change("revise", check=EpisodeCheckDraft(question="Inspect the other appliance",
                object_association="different:appliance"), rebind_episode=True)

        _, lane = await self.during_analysis(replace_goal)
        self.assertFalse(lane.harness.visual_repository.failed)
        self.assertEqual(len(self.db.observations), 1)
        self.assertEqual(self.db.applied, [])
        self.assertTrue(any(d.get("current_applicability") == "historical_only" for d in lane.diagnostics))
        self.assertEqual(self.db.occurrences[self.admitted.check.id]["episode_revision"], 2)

    async def test_cancel_during_analysis_does_not_resurrect_or_speak(self):
        async def cancel(live, lane):
            await self.change("cancel")

        _, lane = await self.during_analysis(cancel)
        self.assertFalse(lane.harness.visual_repository.failed)
        self.assertEqual(self.db.applied, [])
        self.assertEqual(lane.harness.interrupt_service.calls, [])
        self.assertEqual(self.db.occurrences[self.admitted.check.id]["check_lifecycle"], "dismissed")

    async def test_snooze_during_analysis_does_not_apply_or_complete(self):
        async def snooze(live, lane):
            await self.change("snooze", until=self.now + timedelta(minutes=2))

        _, lane = await self.during_analysis(snooze)
        self.assertFalse(lane.harness.visual_repository.failed)
        self.assertEqual(self.db.applied, [])
        self.assertEqual(self.db.occurrences[self.admitted.check.id]["status"], "open")
        self.assertIsNone(self.db.occurrences[self.admitted.check.id]["check_resolution"])

    async def test_unrelated_task_revision_does_not_invalidate_fixed_assignment(self):
        async def task_update(live, lane):
            original = lane.requests[0].perception_brief
            source = self.source_event(kind="task_update", trust="derived")
            record = self.db.record(source.request_event_id)
            # Stored task-update content is the external fixture; the actual
            # canonical binder/producer/reducer handle its new projection. This
            # does not independently qualify TaskRepository result semantics.
            binding = await live.binder.bind_record(record, "An unrelated background calculation completed.")
            receipt = live.producer.publish(TaskEntry(entry_id="task:unrelated", scope=self.scope,
                received_at=self.now, observed_at=record.source_time, evidence_kind="canonical_record",
                sources=(context_dependency(record),), guard_binding=binding,
                task=AggregateRef(namespace="background_task", aggregate_id=uuid4(), revision=2),
                status="succeeded", summary="Unrelated calculation completed."))
            self.assertTrue(receipt.queued)
            await live.service.wait_idle()
            self.assertGreater(live.service.snapshot().cursor.revision, original.based_on_live_revision)
            self.assertEqual(lane.requests[0].perception_brief, original)

        _, lane = await self.during_analysis(task_update)
        self.assertFalse(lane.harness.visual_repository.failed)
        self.assertEqual(len(self.db.applied), 1)
        self.assertFalse(any(d.get("current_applicability") == "historical_only" for d in lane.diagnostics))

    async def test_withdrawn_source_suppresses_queued_assignment_and_voice_write(self):
        live = await self.live()
        selection = select_voice_context(live.service.snapshot(), now=self.now)
        binding = await live.bind(selection)
        lane = await self.pipeline(live, lambda **kwargs:self.output("checking controls"))
        self.db.events[self.source.request_event_id].update(availability="deleted", content=None)
        await self.frame(lane, 1)
        lane.model.generate_temporal_json.assert_not_awaited()
        self.assertEqual(self.db.applied, [])
        provider_write = AsyncMock()
        with self.assertRaises(SourceError):
            async with live.guard(binding):
                await provider_write()
        provider_write.assert_not_awaited()

    async def test_source_withdrawal_while_analyzing_blocks_publication(self):
        async def withdraw(live, lane):
            self.db.events[self.source.request_event_id].update(availability="deleted", content=None)

        _, lane = await self.during_analysis(withdraw)
        self.assertEqual(len(lane.harness.visual_repository.failed), 1)
        self.assertFalse(self.db.observations)
        self.assertFalse(self.db.applied)

    async def test_restart_retains_episode_identity_and_original_time_without_completion(self):
        live = await self.live()
        before, = (await self.owner.read(self.scope)).snapshots
        await live.close()
        later = await self.live()
        after, = (await self.owner.read(self.scope)).snapshots
        self.assertEqual(after.check.id, before.check.id)
        self.assertEqual(after.payload.episode_started_at, before.payload.episode_started_at)
        self.assertEqual(after.check.lifecycle, "tracking")
        self.assertNotEqual(later.service.snapshot().cursor.epoch, live.service.snapshot().cursor.epoch)
        self.assertIsNone(after.check.resolution)

    async def test_slow_memory_must_not_block_minimal_brief_or_frame_analysis(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.releases.append(release)
        area = uuid4()
        scope = self.scope.model_copy(update={"area_id":area})
        self.db.areas.add((scope.owner_id, area))
        sources = SourceRepository(self.db)

        async def memory(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"sources":[]}

        sources.current_snapshot = AsyncMock(side_effect=memory)
        live = LiveStateRuntime(scope=scope, sources=sources, context=self.context,
            area_selections=SimpleNamespace(read_in_transaction=AsyncMock(return_value={
                "area_id":str(area), "area_status":"active", "selection_revision":0, "selection_event_id":None})),
            bus=self.bus, access_check=lambda owner:None, episode_owner=self.owner,
            clock=lambda:self.now, interval_seconds=0.02, timeout_seconds=2, perception_brief_enabled=True)
        self.db.bind_io(live.binder)
        live.perception_runtime.focus_inputs.read = AsyncMock(return_value=None)
        self.lives.append(live)
        await live.start()
        await asyncio.wait_for(entered.wait(), 1)
        await self.until(lambda: live.perception_runtime.worker.refresh_count >= 1)
        lane = await self.pipeline(live, lambda **kwargs:self.output("cooking"))
        await self.frame(lane, 1)
        self.assertFalse(lane.harness.visual_repository.failed)
        lane.model.generate_temporal_json.assert_awaited_once()
        self.assertFalse(release.is_set())
        self.assertIsNotNone(live.perception_brief_bridge._binding,
            "P2 minimal-brief acceptance: delayed memory keeps the whole snapshot warming and no brief is bound")

    async def test_refresh_storm_coalesces_and_snapshot_reads_do_no_io(self):
        live = await self.live()
        entered, release = asyncio.Event(), asyncio.Event()
        self.releases.append(release)
        active, maximum, calls = 0, 0, 0

        async def slow_context(*args, **kwargs):
            nonlocal active, maximum, calls
            calls += 1
            active += 1
            maximum = max(maximum, active)
            try:
                entered.set()
                await release.wait()
                return {"recent":[], "pinned":[]}
            finally:
                active -= 1

        self.context.snapshot.side_effect = slow_context
        await self.bus.publish(DomainEvent(DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
            OwnerWorkingMemoryUpdatedPayload(self.scope.owner_id, 1)))
        await asyncio.wait_for(entered.wait(), 1)
        reads = self.db.acquisitions
        for revision in range(2, 302):
            await self.bus.publish(DomainEvent(DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
                OwnerWorkingMemoryUpdatedPayload(self.scope.owner_id, revision)))
            selection = select_voice_context(live.service.snapshot(), now=self.now)
            self.assertIn("Unresolved episode check", selection.text)
        self.assertEqual(self.db.acquisitions, reads)
        self.assertEqual((maximum, calls), (1, 1))
        before = live.worker.refresh_count
        release.set()
        await self.until(lambda: live.worker.refresh_count >= before + 2)
        self.assertLessEqual(calls, 3)
        self.assertFalse(live.service.needs_resync)
        brief = live.perception_brief_bridge.composition.brief
        self.assertLessEqual(len(brief.questions), 4)
        self.assertTrue(brief.scene_change_cues)

    async def test_departure_cue_requires_actual_activation_not_only_safe_omission(self):
        cue, _ = await self.admit(spec=EpisodeCheckSpec(title="Take the umbrella"),
            check=EpisodeCheckDraft(question="Is the umbrella with the user?", activation_cues=("When departing",),
                                    object_association="office:umbrella"))
        live = await self.live()
        self.assertNotIn(cue.check.id, self.assigned_checks(live))
        lane = await self.pipeline(live, lambda **kwargs:self.output("departing"))
        await self.frame(lane, 1)
        await self.until(lambda: cue.check.id in self.assigned_checks(live),
            lambda:(live.episode_context.omissions, live.perception_runtime.worker.last_error,
                    live.perception.worker.last_error))
        questions = live.perception_brief_bridge.composition.brief.questions
        self.assertIn(cue.check.id, {q.check_ref.source_id for q in questions if q.check_ref is not None},
            "The actual current departure observation must select the conditional check")
        observation_id, = self.db.observations
        source = reference("context_event", self.db.observation_events[observation_id], 1)
        self.assertIn(source, live.perception_brief_bridge.composition.brief.dependencies)
        check = next(item.check for item in (await self.owner.read(self.scope)).snapshots if item.check.id == cue.check.id)
        self.assertFalse(check.relevance_now, "Ephemeral relevance must not mutate canonical owner state")
        self.assertIsNone(check.resolution)
        self.assertEqual(check.evidence_availability, "unknown")

    @staticmethod
    def assigned_checks(live):
        composition = live.perception_brief_bridge.composition
        return {q.check_ref.source_id for q in composition.brief.questions if q.check_ref is not None} if composition else set()

    async def test_cue_is_not_activated_by_other_activity_or_low_confidence(self):
        cue, _ = await self.admit(check=EpisodeCheckDraft(question="Inspect the laundry basket",
            activation_cues=("When washing clothes",)))
        live = await self.live()

        async def generate(**kwargs):
            output = json.loads(self.output("washing clothes"))
            output["activity"]["confidence"] = 0.4
            return json.dumps(output)

        lane = await self.pipeline(live, generate)
        await self.frame(lane, 1)
        await self.until(lambda: any("activity_uncertain" in item for item in live.episode_context.omissions))
        self.assertNotIn(cue.check.id, self.assigned_checks(live))
        self.assertIsNone(self.db.occurrences[cue.check.id]["check_resolution"])

    async def test_activity_cue_original_withdrawal_blocks_followup_assignment(self):
        cue, _ = await self.admit(check=EpisodeCheckDraft(question="Inspect the bag",
            activation_cues=("While packing",)))
        live = await self.live()
        lane = await self.pipeline(live, lambda **kwargs:self.output("packing"))
        await self.frame(lane, 1)
        await self.until(lambda: cue.check.id in self.assigned_checks(live))
        original, = self.db.observation_events.values()
        self.db.events[original].update(availability="deleted", content=None)
        await self.frame(lane, 2)
        lane.model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(len(self.db.applied), 1)
        self.assertIsNone(self.db.occurrences[cue.check.id]["check_resolution"])

    async def test_cue_guard_rechecks_original_time_and_activity_not_global_revision(self):
        cue, _ = await self.admit(check=EpisodeCheckDraft(question="Inspect the bag",
            activation_cues=("While packing",)))
        live = await self.live()
        lane = await self.pipeline(live, lambda **kwargs:self.output("packing"))
        await self.frame(lane, 1)
        await self.until(lambda: cue.check.id in self.assigned_checks(live))
        brief = live.perception_brief_bridge.composition.brief

        async def validate():
            async with live.binder._transaction() as connection:
                episodes = await live.binder._validate_dependencies(connection, brief.dependencies)
                await live.episode_context.validate_assignment(connection, brief, episodes)

        await validate()
        # Repeated current evidence for the SAME activity need not invalidate
        # an otherwise valid fixed assignment or replace its original time.
        await self.frame(lane, 2)
        await self.until(lambda: live.perception._last_revision == 2)
        await validate()
        self.now += timedelta(seconds=11)
        with self.assertRaises(SourceError) as caught:
            await validate()
        self.assertEqual(caught.exception.code, "live_episode_cue_inapplicable")
        self.assertIsNone(self.db.occurrences[cue.check.id]["check_resolution"])

    async def test_fixed_assignment_has_frozen_question_and_check_reference_handles(self):
        live = await self.live()
        lane = await self.pipeline(live, lambda **kwargs:self.output("checking controls"))
        await self.frame(lane, 1)
        request, = lane.requests
        registry = PerceptionReferences(request)
        self.assertTrue(hasattr(registry, "questions") and hasattr(registry, "checks"),
            "P0/P2 requires question:0 and check:0 registries; legacy frame/focus registries cannot carry typed check findings")
        self.assertEqual(registry.questions["question:0"], str(request.perception_brief.questions[0].id))
        self.assertEqual(registry.checks["check:0"], str(self.admitted.check.id))


if __name__ == "__main__":
    unittest.main()
