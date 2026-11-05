"""Real published speech capture versus one already-running perception write.

No models, provider sessions, images, external memory workers or hardware. This
qualifies the persistence boundary, not cancellation or audible response latency.
"""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
from time import monotonic
import unittest
from uuid import uuid4

from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture
from test_live_state_refresh_contention_sql import LANE, Trace, audit_residue


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class PublishedCaptureContentionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorizing_transcript_survives_one_perception_publication(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        trace, fixture = Trace(), None
        report = {"format": "jarvis.published_capture.contention.v1",
            "observation_count": 1, "snapshot_read_count": 4,
            "pool_max_size": 3, "artificial_lock_delay": False,
            "models_or_providers": False, "latency_acceptance": False}
        try:
            async with source_fixture(dsn, directory, test_name=self.id(),
                                      pool_init=trace.pool_init) as f:
                fixture = f.fixture
                report["fixture_manifest"] = str(fixture.path)
                tasks = TaskRepository(f.pool, TaskLimits())
                selections = SessionKnowledgeAreas(f.sources, tasks, f.area)
                publication = CanonicalSourcePublication(f.sources, f.area, selections)
                history = ContextRepository(tasks, publication=publication)
                async with f.pool.acquire() as connection:
                    session = await connection.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
                capture = SessionContextCapture(history, owner_id=f.owner, session_id=session,
                    provider="openai", audio_session_id=str(uuid4()))
                actors = []
                request = None
                try:
                    # Native speech first has a pending canonical identity; its
                    # finalized revision later authorizes the requested tool.
                    capture.input_admitted("fixture-input")
                    self.assertTrue(await capture.flush())
                    self.assertEqual(capture.write_timeout, 3.0)
                    report["capture_deadline_seconds"] = capture.write_timeout

                    async def publish_observation():
                        token = LANE.set("observation_1")
                        try:
                            return await history.append(f.owner, ContextIngress(
                                session_id=session, source_namespace="perception:analyzer:v1",
                                source_event_id="synthetic-observation", receipt_id="synthetic-final",
                                kind="observation", trust="derived", finality="final",
                                payload={"description": "Synthetic desk layout; no personal evidence."}))
                        finally:
                            LANE.reset(token)

                    async def read_snapshots():
                        token = LANE.set("context_reader")
                        try:
                            for _ in range(report["snapshot_read_count"]):
                                snapshot = await selections.read(f.owner, session)
                                self.assertEqual(snapshot["area_id"], str(f.area))
                                await asyncio.sleep(0.1)
                        finally:
                            LANE.reset(token)

                    with ExitStack() as stack:
                        trace.transactions(stack, f.sources)
                        trace.wrap(stack, history, "append", "canonical_append")
                        observation = asyncio.create_task(publish_observation())
                        actors.append(observation)
                        # Synchronize on actual lock ownership, not a guessed
                        # sleep. The publisher otherwise runs unchanged.
                        await asyncio.wait_for(trace.observation_entered.wait(), 10)
                        readers = asyncio.create_task(read_snapshots())
                        actors.append(readers)
                        token = LANE.set("finalized_transcript")
                        start = monotonic()
                        try:
                            capture.input_final("fixture-input", "Compare two hypothetical desk layouts.")
                            capture.bind_call("fixture-call", "fixture-input")
                            request = await capture.request_for_call("fixture-call")
                        finally:
                            report["request_wait_ms"] = round((monotonic() - start) * 1000, 3)
                            LANE.reset(token)
                        await asyncio.wait_for(asyncio.gather(*actors), 12)
                        await capture.close()
                finally:
                    for actor in actors:
                        if not actor.done():
                            actor.cancel()
                    await asyncio.gather(*actors, return_exceptions=True)
                    await capture.close()
                    await asyncio.sleep(0)
                    report.update(request_authorized=request is not None,
                        capture_issue=capture.issue, failed_events=capture.failed_events,
                        actors_joined=all(actor.done() for actor in actors))
                self.assertIsNone(capture.issue)
                self.assertIsNotNone(request)
                self.assertTrue(request.authorizes_request)
                self.assertEqual((request.owner_id, request.session_id), (f.owner, session))
        finally:
            report.update(stages=trace.stages, queries=trace.queries, dropped=trace.dropped)
            if fixture is not None:
                report["cleanup_audit"] = await audit_residue(dsn, fixture.manifest)
            path = directory / ("published-contention-" + uuid4().hex + ".json")
            path.write_text(json.dumps(report, indent=2) + "\n")
