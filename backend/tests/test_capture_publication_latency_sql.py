"""Opt-in production-path capture qualification, not a model or audio benchmark.

Use the default capture deadline, real publication, session-area binding and
conversation projection. Query timings include database round trips; they do
not distinguish server execution from network latency. Never record parameters.
"""
import asyncio
import hashlib
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
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class PublishedCaptureLatencySQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_deadline_preserves_published_request_authority(self):
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        queries = []
        recording = False

        def query_completed(record):
            if recording:
                queries.append({
                    "query_sha256": hashlib.sha256(record.query.encode()).hexdigest(),
                    "operation": record.query.split(None, 1)[0].lower(),
                    "elapsed_ms": round(record.elapsed * 1000, 3),
                    "error_type": type(record.exception).__name__ if record.exception else None,
                })

        async def initialize(connection):
            connection.add_query_logger(query_completed)

        async with source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], directory,
            test_name=self.id(), pool_init=initialize,
        ) as fixture:
            tasks = TaskRepository(fixture.pool, TaskLimits())
            selections = SessionKnowledgeAreas(fixture.sources, tasks, fixture.area)
            publication = CanonicalSourcePublication(fixture.sources, fixture.area, selections)
            history = ContextRepository(tasks, publication=publication)
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            capture = SessionContextCapture(history, owner_id=fixture.owner,
                session_id=session, provider="openai", audio_session_id=str(uuid4()))
            request = None
            failure = None
            recording = True
            started = monotonic()
            try:
                capture.input_final("fixture-input", "Compare two hypothetical desk layouts.")
                capture.bind_call("fixture-call", "fixture-input")
                request = await capture.request_for_call("fixture-call")
            except Exception as error:
                failure = type(error).__name__
                raise
            finally:
                request_elapsed_ms = (monotonic() - started) * 1000
                try:
                    await capture.close()
                finally:
                    # asyncpg schedules synchronous query observers with call_soon.
                    await asyncio.sleep(0)
                    recording = False
                    report = {
                        "format": "jarvis.published_capture.sql_timing.v1",
                        "capture_deadline_seconds": capture.write_timeout,
                        "request_elapsed_ms": round(request_elapsed_ms, 3),
                        "request_authorized": request is not None,
                        "capture_issue": capture.issue,
                        "failure_type": failure,
                        "query_count": len(queries), "queries": queries,
                        "meaning": "Isolated real SQL; no models, workers, audio or hardware acceptance.",
                    }
                    path = directory / ("published-capture-" + uuid4().hex + ".json")
                    path.write_text(json.dumps(report, indent=2) + "\n")
            self.assertIsNone(capture.issue)
            self.assertIsNotNone(request)
            self.assertTrue(request.authorizes_request)
            async with fixture.pool.acquire() as connection:
                mapped = await connection.fetchrow("""select s.area_id,v.payload_id
                    from jarvis_knowledge_context_sources cs
                    join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                    join jarvis_knowledge_source_versions v on v.owner_id=cs.owner_id
                        and v.source_id=cs.source_id and v.revision=cs.source_revision
                    where cs.owner_id=$1 and cs.event_id=$2""", fixture.owner, request.id)
                message_payload = await connection.fetchval("""select source_payload_id
                    from jarvis_messages where owner_id=$1 and session_id=$2 and role='user'""",
                    fixture.owner, session)
            self.assertIsNotNone(mapped)
            self.assertEqual((mapped["area_id"], mapped["payload_id"]), (fixture.area, request.payload_id))
            self.assertEqual(message_payload, request.payload_id)
