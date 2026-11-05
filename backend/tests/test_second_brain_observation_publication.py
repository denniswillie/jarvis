"""Ongoing visual originals through the real canonical publication transaction."""
from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.knowledge.images import SourceImageService
from app.knowledge.observation_inputs import request_document, working_observation_ids
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.prompts import build_fast_perception_prompt
from app.sessions.models import JarvisMessage
from app.tasks.context_events import ContextIngress
from app.knowledge.models import SourceError
from app.knowledge.observations import ObservationPublication, UnavailableObservationPublication
from app.knowledge.publication import CanonicalSourcePublication
from app.perception.models import ObservationFrameLink, VisualObservation, VisibleTextObservation
from app.perception.repository import PerceptionRepository
from app.runtime.feature_config import TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.visual_memory.pipeline import VisualAnalysisPipeline
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.visual_observation_fixture import run_visual_fixture


class ObservationPublicationContractTests(unittest.IsolatedAsyncioTestCase):
    def test_requires_a_selected_canonical_publisher(self):
        with self.assertRaises(ValueError):
            ObservationPublication(SimpleNamespace(publication=None), owner_id="fixture")

    async def test_unavailable_selected_memory_does_not_silently_store_legacy_only(self):
        with self.assertRaises(SourceError):
            async with UnavailableObservationPublication().transaction("fixture"):
                self.fail("Unavailable publication admitted a transaction")

    async def test_selected_path_does_not_write_an_unbound_frame_summary(self):
        pipeline = object.__new__(VisualAnalysisPipeline)
        pipeline._live_briefs = {}
        pipeline._perception_evidence_sessions = {}
        pipeline._observation_publication = None
        pipeline._perception_repository = SimpleNamespace(observation_publication=None)
        pipeline._repository = SimpleNamespace(apply_analysis_result=AsyncMock())
        frame = SimpleNamespace(owner_id="fixture", id=uuid4(), frame_sequence=0)
        output = SimpleNamespace(summary="Fixture analyzer original", memory_tags=["fixture"],
                                 interrupt_candidate=None, prospective_matches=())
        for selected in (False, True):
            pipeline.set_observation_publication(object() if selected else None)
            await pipeline._apply_completed_result(
                frame=frame, output=output, send_diagnostic=None, observation_id=uuid4())
            written = pipeline._repository.apply_analysis_result.call_args.kwargs
            self.assertEqual(written["frame_description"], None if selected else output.summary)
            self.assertEqual(written["memory_tags"], [] if selected else ["fixture"])

    def test_pipeline_configures_the_existing_repository(self):
        pipeline = object.__new__(VisualAnalysisPipeline)
        pipeline._live_briefs = {}
        pipeline._perception_evidence_sessions = {}
        pipeline._observation_publication = None
        pipeline._perception_repository = SimpleNamespace(observation_publication=None)
        marker = object()
        pipeline.set_observation_publication(marker)
        self.assertIs(pipeline._perception_repository.observation_publication, marker)
        pipeline.set_observation_publication(None)
        self.assertIsNone(pipeline._perception_repository.observation_publication)


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class ObservationPublicationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=60))
        self.context = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.publisher = ObservationPublication(self.context, owner_id=self.f.owner)
        self.repository = PerceptionRepository(self.f.pool, observation_publication=self.publisher)
        self.when = datetime(2026, 9, 14, 12, tzinfo=UTC)
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
        self.frame = await self.frame_for(self.f.owner, self.session)

    async def frame_for(self, owner, session, *, sequence=0):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval(
                """insert into visual_frames(owner_id,session_id,visual_session_id,frame_sequence,source,
                    captured_at,content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
                    values($1,$2,$3,$4,'glasses',$5,'image/jpeg',432,768,'fixture','not-uploaded',
                           clock_timestamp()+interval '1 hour') returning id""",
                owner, session, uuid4(), sequence, self.when)

    def observation(self, **changes):
        return replace(VisualObservation(
            id=uuid4(), owner_id=self.f.owner, session_id=self.session, source="glasses",
            started_at=self.when, ended_at=self.when, primary_visual_frame_id=self.frame,
            summary="Synthetic analyzer reads an unfinished binary search.",
            scene="fixture desk", activity="coding", entities=(),
            visible_text=(VisibleTextObservation("mid = (low + high) // 2", "fixture editor", 0.8),),
            state_changes=(), memory_tags=("coding",), keyframe_candidates=(), interrupt_candidate=None,
            starts_new_event=False, boundary_reason=None, confidence=0.8, analyzer_provider="gemini",
            analyzer_model="fixture-only", analyzer_contract_version=1, compacted_at=None, created_at=self.when,
        ), **changes)

    async def insert(self, observation=None):
        observation = observation or self.observation()
        return await self.repository.insert_observation(
            observation=observation,
            frame_links=(ObservationFrameLink(observation.primary_visual_frame_id, 0, "primary"),))

    async def source(self, observation):
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow(
                """select s.source_id,b.payload_id from jarvis_observation_sources b
                    join jarvis_knowledge_source_versions s on s.owner_id=b.owner_id and s.payload_id=b.payload_id
                    where b.owner_id=$1 and b.observation_id=$2""", self.f.owner, observation.id)
        self.assertIsNotNone(row)
        return await self.f.sources.read(self.f.owner, self.f.area, row["source_id"])

    async def counts(self):
        async with self.f.pool.acquire() as connection:
            return {table: await connection.fetchval("select count(*) from " + table) for table in (
                "perception_observations", "perception_observation_frames", "jarvis_context_events",
                "jarvis_context_payloads", "jarvis_knowledge_sources", "jarvis_memory_outbox",
                "jarvis_observation_sources")}

    async def test_original_images_and_interpretation_publish_atomically_without_provider_io(self):
        # The knowledge-area fixture already owns its metadata payload.
        before = await self.counts()
        observation = await self.insert()
        source = await self.source(observation)
        self.assertEqual(source.trust, "derived")
        self.assertEqual(source.source_time, self.when)
        self.assertEqual(source.context_session_id, self.session)
        self.assertEqual(source.ingestion_status, "pending")
        self.assertEqual(source.content["record"]["summary"], observation.summary)
        self.assertEqual(source.content["record"]["visible_text"][0]["text"], "mid = (low + high) // 2")
        self.assertNotIn("storage_path", json.dumps(source.content))
        async with self.f.pool.acquire() as connection:
            images = await SourceImageService(self.f.sources, None)._direct(connection, source)
        self.assertEqual([(image.frame_id, image.session_id, image.captured_at, image.source) for image in images],
                         [(self.frame, self.session, self.when, "historical")])
        self.assertEqual(await self.counts(), {key: value + 1 for key, value in before.items()})

    async def test_publication_failure_rolls_back_observation_and_frame_links(self):
        bad = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, uuid4()))
        self.repository.observation_publication = ObservationPublication(bad, owner_id=self.f.owner)
        before = await self.counts()
        with self.assertRaises(SourceError):
            await self.insert()
        self.assertEqual(await self.counts(), before)

    async def test_archived_area_keeps_canonical_capture_without_new_memory_ingestion(self):
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        await self.insert()
        counts = await self.counts()
        self.assertEqual(counts["perception_observations"], 1)
        self.assertEqual(counts["jarvis_context_events"], 1)
        self.assertEqual(counts["jarvis_observation_sources"], 1)
        self.assertEqual(counts["jarvis_knowledge_sources"], 0)
        self.assertEqual(counts["jarvis_memory_outbox"], 0)

    async def test_repeated_or_foreign_observation_cannot_add_a_second_source(self):
        observation = await self.insert()
        before = await self.counts()
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.insert(observation)
        with self.assertRaises(SourceError):
            await self.insert(self.observation(owner_id=self.f.other))
        self.assertEqual(await self.counts(), before)
        async with self.f.pool.acquire() as connection:
            foreign_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)
        foreign_frame = await self.frame_for(self.f.other, foreign_session)
        with self.assertRaises(asyncpg.PostgresError):
            await self.insert(self.observation(primary_visual_frame_id=foreign_frame))
        self.assertEqual(await self.counts(), before)

    async def test_binding_rejects_a_different_original_payload(self):
        observation = await self.insert()
        unrelated = await self.f.publish("Unrelated synthetic original")
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """insert into jarvis_observation_sources(owner_id,observation_id,session_id,payload_id)
                        values($1,$2,$3,$4)""",
                    self.f.owner, observation.id, self.session, unrelated.payload_id)

    async def test_false_capture_clock_rolls_back_the_whole_write(self):
        before = await self.counts()
        with self.assertRaises(SourceError):
            await self.insert(self.observation(
                started_at=self.when + timedelta(seconds=1), ended_at=self.when + timedelta(seconds=1)))
        self.assertEqual(await self.counts(), before)

    async def seed_operational_copies(self, observation):
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                """insert into jarvis_session_working_memory(owner_id,session_id,revision,last_observation_id,
                    last_observation_ended_at,state,expires_at) values($1,$2,1,$3,$4,$5::jsonb,
                    clock_timestamp()+interval '1 hour')""",
                self.f.owner, self.session, observation.id, self.when, json.dumps({"summary": observation.summary}))
            await connection.execute(
                "update visual_frames set frame_description=$2,memory_tags=array['coding'] where id=$1",
                self.frame, observation.summary)
            event = await connection.fetchval(
                """insert into experience_events(owner_id,session_id,event_kind,started_at,summary,state)
                    values($1,$2,'visual',$3,$4,$5::jsonb) returning id""",
                self.f.owner, self.session, self.when, observation.summary, json.dumps({"summary": observation.summary}))
            await connection.execute(
                """insert into experience_event_evidence(event_id,evidence_type,evidence_ref_id,occurred_at)
                    values($1,'observation',$2,$3)""", event, observation.id, self.when)
        return event

    async def test_delete_erases_original_and_operational_copies_and_blocks_resurrection(self):
        observation = await self.insert()
        source = await self.source(observation)
        event = await self.seed_operational_copies(observation)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from perception_observations"), 0)
            self.assertEqual(await connection.fetchval("select count(*) from perception_observation_frames"), 0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_observation_sources"), 1)
            working = await connection.fetchrow("select * from jarvis_session_working_memory where session_id=$1", self.session)
            self.assertEqual(json.loads(working["state"]), {})
            self.assertGreater(working["revision"], 1)
            self.assertIsNone(working["last_observation_id"])
            self.assertIsNone(await connection.fetchval("select frame_description from visual_frames where id=$1", self.frame))
            self.assertEqual(await connection.fetchval("select summary from experience_events where id=$1", event), "")
            self.assertEqual(await connection.fetchval("select count(*) from experience_event_evidence"), 0)
            self.assertIsNone(await connection.fetchval("select content from jarvis_context_payloads where id=$1", source.payload_id))
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, source.source_id)
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.insert(observation)
        new = await self.insert()
        stale = await self.repository.compare_and_swap_session_working_memory(
            owner_id=self.f.owner, session_id=self.session, expected_revision=1, observation_id=new.id,
            state={"stale_deleted_text": observation.summary}, expires_at=self.when + timedelta(days=3))
        self.assertFalse(stale.applied)
        self.assertEqual(stale.state.state, {})

    async def test_direct_payload_removal_also_invalidates_copies(self):
        observation = await self.insert()
        source = await self.source(observation)
        await self.seed_operational_copies(observation)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where id=$1", source.payload_id)
            self.assertEqual(await connection.fetchval("select count(*) from perception_observations"), 0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_observation_sources"), 1)

    async def test_logical_forgetting_keeps_canonical_original_but_drops_current_cache(self):
        observation = await self.insert()
        source = await self.source(observation)
        await self.seed_operational_copies(observation)
        async with self.f.sources.transaction(self.f.owner) as connection:
            await connection.execute(
                "update jarvis_knowledge_sources set status='forgotten',generation=generation+1 where id=$1", source.source_id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from perception_observations"), 0)
            self.assertIsNotNone(await connection.fetchval("select content from jarvis_context_payloads where id=$1", source.payload_id))
            self.assertEqual(json.loads(await connection.fetchval(
                "select state from jarvis_session_working_memory where session_id=$1", self.session)), {})

    async def prepare_visual_input(self, *, state=None, messages=(), interrupts=()):
        # Explicit fixture capture after canonical ingress, not a physical clock claim.
        self.when = datetime.now(UTC) + timedelta(seconds=2)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update visual_frames set captured_at=$2 where id=$1", self.frame, self.when)
        request = FastPerceptionRequest(
            owner_id=self.f.owner, session_id=self.session, window_id=uuid4(),
            frames=(FastPerceptionFrame(self.frame, 0, self.when, "glasses", "image/jpeg", b"synthetic-pixels"),),
            previous_working_state=state or {}, existing_entity_ids=frozenset((state or {}).get("entities", {})),
            recent_audio_transcripts=(), behavior_instructions="", active_reminders=(), active_focus=None,
            recent_interrupts=tuple({"message": item.message, "reason": item.reason, "confidence": item.confidence,
                                    "created_at": item.created_at.isoformat()} for item in interrupts),
            requested_at=self.when)
        context = SimpleNamespace(owner_id=self.f.owner, session_id=self.session,
                                  recent_audio_transcripts=list(messages), recent_visual_interrupts=list(interrupts))
        prepared, snapshot = await self.publisher.prepare_inputs(request, context)
        return prepared, snapshot

    async def voice_input(self, text="The robot calibration is unfinished.", **changes):
        return await self.context.append(self.f.owner, ContextIngress(**(dict(
            session_id=self.session, source_namespace="voice:visual-input-test",
            source_event_id="input-one", receipt_id=str(uuid4()), kind="user_transcript",
            trust="user_input", source_time=datetime.now(UTC),
            payload={"text": text, "provider_item_id": "user-item", "transcript_status": "finalized"},
        ) | changes)))

    async def voice_source(self, event):
        async with self.f.pool.acquire() as connection:
            source_id = await connection.fetchval(
                "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                self.f.owner, event.id)
        return await self.f.sources.read(self.f.owner, self.f.area, source_id)

    async def input_payload(self, snapshot):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where id=$1", snapshot.payload_id)

    async def publish_prepared(self, prepared, snapshot):
        observation = self.observation()
        await self.repository.insert_observation(
            observation=observation, frame_links=(ObservationFrameLink(self.frame, 0, "primary"),),
            analysis_input=snapshot)
        return observation, await self.source(observation)

    async def test_visual_input_captures_canonical_voice_and_hashes_pixels_not_bytes(self):
        event = await self.voice_input()
        prepared, snapshot = await self.prepare_visual_input()
        await self.publisher.validate_input(prepared, snapshot)
        payload = await self.input_payload(snapshot)
        content = json.loads(payload["content"])
        self.assertEqual(prepared.voice_context["records"][0]["content"]["text"], event.content["text"])
        self.assertIn("canonical voice_context", build_fast_perception_prompt(prepared))
        self.assertNotIn("synthetic-pixels", json.dumps(content))
        self.assertEqual(len(content["request"]["frames"][0]["sha256"]), 64)
        self.assertEqual(content["request"], request_document(prepared))
        observation, source = await self.publish_prepared(prepared, snapshot)
        async with self.f.pool.acquire() as connection:
            dependencies = await connection.fetch(
                "select source_payload_id,dependent_payload_id from jarvis_context_payload_dependencies")
        self.assertIn((event.payload_id, snapshot.payload_id),
                      {(row["source_payload_id"], row["dependent_payload_id"]) for row in dependencies})
        self.assertIn((snapshot.payload_id, source.payload_id),
                      {(row["source_payload_id"], row["dependent_payload_id"]) for row in dependencies})
        self.assertEqual(source.content["analysis_input_id"], str(snapshot.payload_id))
        self.assertEqual(source.content["input_provenance"], "captured")

    async def test_deletion_during_analysis_prevents_publishing_the_result(self):
        event = await self.voice_input()
        source = await self.voice_source(event)
        prepared, snapshot = await self.prepare_visual_input()
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual((await self.input_payload(snapshot))["availability"], "deleted")
        with self.assertRaises(SourceError):
            await self.publisher.validate_input(prepared, snapshot)
        with self.assertRaises(SourceError):
            await self.publish_prepared(prepared, snapshot)
        self.assertEqual((await self.counts())["perception_observations"], 0)

    async def test_source_correction_during_analysis_invalidates_captured_inputs(self):
        await self.voice_input()
        prepared, snapshot = await self.prepare_visual_input()
        await self.voice_input("The motor is calibrated; the controller remains unfinished.",
                               finality="revised", expected_revision=1)
        self.assertNotEqual((await self.input_payload(snapshot))["availability"], "available")
        with self.assertRaises(SourceError):
            await self.publish_prepared(prepared, snapshot)

    async def test_deletion_after_analysis_revokes_derived_observation_and_original_copies(self):
        event = await self.voice_input()
        original = await self.voice_source(event)
        prepared, snapshot = await self.prepare_visual_input()
        observation, derived = await self.publish_prepared(prepared, snapshot)
        await self.seed_operational_copies(observation)
        await self.f.sources.delete_source(self.f.owner, self.f.area, original.source_id, expected_revision=1)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, derived.source_id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from perception_observations"), 0)
            self.assertIsNone(await connection.fetchval(
                "select content from jarvis_context_payloads where id=$1", derived.payload_id))

    async def test_working_memory_keeps_every_explicit_observation_parent(self):
        first = await self.insert()
        first_source = await self.source(first)
        second = await self.insert(self.observation())
        second_source = await self.source(second)
        state = {"last_observation_id": str(second.id), "summary": second.summary,
                 "entities": {"entity-one": {"source_observation_id": str(first.id), "name": "fixture robot"}}}
        prepared, snapshot = await self.prepare_visual_input(state=state)
        self.assertEqual(prepared.previous_working_state, state)
        self.assertEqual(set(working_observation_ids(state)), {first.id, second.id})
        async with self.f.pool.acquire() as connection:
            parents = await connection.fetch(
                "select source_payload_id from jarvis_context_payload_dependencies where dependent_payload_id=$1",
                snapshot.payload_id)
        self.assertEqual({row["source_payload_id"] for row in parents}, {first_source.payload_id, second_source.payload_id})
        await self.f.sources.delete_source(self.f.owner, self.f.area, first_source.source_id, expected_revision=1)
        with self.assertRaises(SourceError):
            await self.publisher.validate_input(prepared, snapshot)

    async def test_missing_working_evidence_and_unbound_conversation_are_explicit_gaps(self):
        message = JarvisMessage(id=uuid4(), session_id=self.session, role="user", modality="audio",
            content="Private unbound text", provider_item_id="unbound", provider_response_id=None, created_at=self.when)
        prepared, snapshot = await self.prepare_visual_input(
            state={"summary": "Private old working text", "last_observation_id": str(uuid4())}, messages=[message])
        self.assertEqual(prepared.previous_working_state, {})
        self.assertEqual(prepared.recent_audio_transcripts, ())
        encoded = json.dumps(request_document(prepared))
        self.assertNotIn("Private unbound text", encoded)
        self.assertNotIn("Private old working text", encoded)
        self.assertEqual({gap["component"] for gap in prepared.input_context_gaps},
                         {"previous_working_state", "recent_conversation"})

    async def test_changed_request_or_foreign_snapshot_is_rejected(self):
        prepared, snapshot = await self.prepare_visual_input()
        with self.assertRaises(SourceError):
            await self.publisher.validate_input(replace(prepared, behavior_instructions="Changed after snapshot"), snapshot)
        with self.assertRaises(SourceError):
            await self.publish_prepared(prepared, replace(snapshot, owner_id=self.f.other))

    async def test_unused_input_discard_does_not_delete_successfully_published_input(self):
        prepared, unused = await self.prepare_visual_input()
        await self.publisher.discard_input(unused)
        self.assertEqual((await self.input_payload(unused))["availability"], "deleted")
        prepared, used = await self.prepare_visual_input()
        await self.publish_prepared(prepared, used)
        await self.publisher.discard_input(used)
        self.assertEqual((await self.input_payload(used))["availability"], "available")

    async def test_archived_area_still_captures_new_canonical_input_without_retention(self):
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        event = await self.voice_input()
        prepared, snapshot = await self.prepare_visual_input()
        self.assertEqual(prepared.voice_context["records"][0]["content"]["text"], event.content["text"])
        await self.repository.insert_observation(
            observation=self.observation(), frame_links=(ObservationFrameLink(self.frame, 0, "primary"),),
            analysis_input=snapshot)
        self.assertEqual((await self.counts())["jarvis_knowledge_sources"], 0)

    async def test_logical_forgetting_also_revokes_inputs_and_derived_observation(self):
        event = await self.voice_input()
        original = await self.voice_source(event)
        prepared, snapshot = await self.prepare_visual_input()
        _, derived = await self.publish_prepared(prepared, snapshot)
        async with self.f.sources.transaction(self.f.owner) as connection:
            await connection.execute(
                "update jarvis_knowledge_sources set status='forgotten',generation=generation+1 where id=$1",
                original.source_id)
        self.assertNotEqual((await self.input_payload(snapshot))["availability"], "available")
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, derived.source_id)

    async def test_deleted_canonical_text_is_not_restored_from_an_independent_message_copy(self):
        event = await self.voice_input("Private canonical words")
        source = await self.voice_source(event)
        unrelated = await self.f.publish("Independent response context")
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        stale = JarvisMessage(id=uuid4(), session_id=self.session, role="assistant", modality="audio",
            content="Private canonical words", provider_item_id="user-item", provider_response_id="response",
            created_at=self.when, source_payload_id=unrelated.payload_id)
        prepared, _ = await self.prepare_visual_input(messages=[stale])
        self.assertNotIn("Private canonical words", json.dumps(request_document(prepared)))

    async def test_canonical_correction_while_archived_still_revokes_visual_input(self):
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        await self.voice_input("Old unindexed words")
        prepared, snapshot = await self.prepare_visual_input()
        await self.voice_input("Corrected unindexed words", finality="revised", expected_revision=1)
        with self.assertRaises(SourceError):
            await self.publisher.validate_input(prepared, snapshot)

    async def test_actual_pipeline_uses_canonical_inputs_and_publishes_bound_result(self):
        event = await self.voice_input()
        await self.prepare_visual_input()
        result = await run_visual_fixture(self.f, self.publisher, self.frame, "Fixture camera sees a robot.")
        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(result["requests"][0].voice_context["records"][0]["content"]["text"], event.content["text"])
        self.assertEqual(result["diagnostics"][-1]["type"], "visual.frame.analyzed")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from perception_observations"), 1)
            self.assertIsNone(await connection.fetchval("select frame_description from visual_frames where id=$1", self.frame))
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_event_dependencies where event_id=$1", event.id), 2)

    async def test_actual_pipeline_discards_result_if_source_changes_during_analysis(self):
        event = await self.voice_input()
        source = await self.voice_source(event)
        await self.prepare_visual_input()
        async def remove_during_analysis(request):
            await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        result = await run_visual_fixture(
            self.f, self.publisher, self.frame, "Result must never become memory.",
            on_analyze=remove_during_analysis)
        self.assertEqual(result["diagnostics"][-1]["type"], "visual.frame.analysis_failed")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from perception_observations"), 0)
            self.assertEqual(await connection.fetchval(
                """select count(*) from jarvis_context_payloads where availability='available'
                    and content->>'format'='jarvis.visual_input.v1'"""), 0)

    async def test_working_memory_cannot_borrow_another_session_observation(self):
        async with self.f.pool.acquire() as connection:
            other_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
        other_frame = await self.frame_for(self.f.owner, other_session)
        observation = self.observation(session_id=other_session, primary_visual_frame_id=other_frame)
        await self.repository.insert_observation(
            observation=observation, frame_links=(ObservationFrameLink(other_frame, 0, "primary"),))
        prepared, _ = await self.prepare_visual_input(state={
            "summary": "Scene from another session", "last_observation_id": str(observation.id)})
        self.assertEqual(prepared.previous_working_state, {})
        self.assertIn("original_evidence_unavailable", {gap["reason"] for gap in prepared.input_context_gaps})

    async def test_canonical_input_binding_rejects_an_unrelated_event_original(self):
        event = await self.voice_input()
        _, snapshot = await self.prepare_visual_input()
        other = await self.voice_input("Independent words", source_event_id="different")
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """insert into jarvis_context_event_dependencies(owner_id,dependent_payload_id,event_id,
                        event_revision,source_payload_id,source_usage) values($1,$2,$3,$4,$5,'current')""",
                    self.f.owner, snapshot.payload_id, other.id, other.revision, event.payload_id)
