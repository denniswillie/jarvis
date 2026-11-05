"""Native Hindsight retention/recall of a synthetic ongoing visual observation.

The analyzer output is scripted; this is not an OCR/model-quality or glasses test.
The existing owned-service fixture creates and removes its own database and bank.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4

import asyncpg

from app.knowledge.hindsight import HindsightClient
from app.knowledge.images import SourceImageService
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.observations import ObservationPublication
from app.knowledge.outbox import OutboxRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.perception.repository import PerceptionRepository
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_repository import ContextRepository
from app.tasks.context_events import ContextIngress
from app.tasks.repository import TaskRepository
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.visual_observation_fixture import run_visual_fixture


async def exercise(root, service):
    report = {"checks": {}, "worker_events": [], "synthetic_only": True,
              "production_database_used": False, "full_stage_accepted": False,
              "native_visual_analyzer_tested": False, "pixels_uploaded": False}
    started = time.monotonic()
    marker = "OBS-" + uuid4().hex[:12]
    try:
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "database-manifests",
                test_name="native-ongoing-observation-publication"))
            client = HindsightClient(service["url"], service["token"], timeout=15)
            stack.push_async_callback(client.close)
            tasks = TaskRepository(fixture.pool, TaskLimits())
            context = ContextRepository(tasks, publication=CanonicalSourcePublication(fixture.sources, fixture.area))
            publisher = ObservationPublication(context, owner_id=fixture.owner)
            perception = PerceptionRepository(fixture.pool, observation_publication=publisher)
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            user_event = await context.append(fixture.owner, ContextIngress(
                session_id=session, source_namespace="voice:native-visual-input", source_event_id="user-input",
                receipt_id="user-input", kind="user_transcript", trust="user_input", source_time=datetime.now(UTC),
                payload={"text": "I paused binary search project " + marker + " and will continue tomorrow; "
                                 "do not mark the work complete.", "provider_item_id": "fixture-user-input"}))
            async with fixture.pool.acquire() as connection:
                user_source_id = await connection.fetchval(
                    "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                    fixture.owner, user_event.id)
            user_source = await fixture.sources.read(fixture.owner, fixture.area, user_source_id)
            when = datetime.now(UTC)
            async with fixture.pool.acquire() as connection:
                frame = await connection.fetchval(
                    """insert into visual_frames(owner_id,session_id,visual_session_id,frame_sequence,source,
                        captured_at,content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
                        values($1,$2,$3,0,'glasses',$4,'image/jpeg',432,768,'fixture','not-uploaded',
                        clock_timestamp()+interval '1 hour') returning id""",
                    fixture.owner, session, uuid4(), when)
            summary = ("Synthetic project " + marker + ": binary search remains incomplete; "
                       "the midpoint is parenthesized, but the search-bound updates are still missing.")
            pipeline_result = await run_visual_fixture(fixture, publisher, frame, summary)
            require_check(report, "actual_pipeline_consumes_canonical_conversation",
                          len(pipeline_result["requests"]) == 1
                          and pipeline_result["requests"][0].voice_context["records"][0]["content"]["text"]
                              == user_event.content["text"])
            require_check(report, "actual_pipeline_publishes_successfully",
                          pipeline_result["diagnostics"][-1]["type"] == "visual.frame.analyzed")
            observations = await perception.list_session_observations(owner_id=fixture.owner, session_id=session)
            require_check(report, "one_operational_observation_is_stored", len(observations) == 1)
            observation = observations[0]
            async with fixture.pool.acquire() as connection:
                source_id = await connection.fetchval(
                    """select v.source_id from jarvis_observation_sources b
                        join jarvis_knowledge_source_versions v on v.owner_id=b.owner_id and v.payload_id=b.payload_id
                        where b.owner_id=$1 and b.observation_id=$2""", fixture.owner, observation.id)
            source = await fixture.sources.read(fixture.owner, fixture.area, source_id)
            require_check(report, "canonical_original_locally_pending_before_native_ingestion",
                          source.ingestion_status == "pending" and source.trust == "derived"
                          and source.context_session_id == session and source.source_time == when
                          and source.content["record"]["summary"] == observation.summary)
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            await drain(fixture, worker, report, label="ongoing-observation-retain")
            client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "native_document_matches_original_and_source_metadata", True)
            retrieval = SourceMemoryRetrieval(fixture.sources, client, timeout_seconds=30)
            recalled = await retrieval.recall(
                fixture.owner, fixture.area, "What did the camera observe was missing from binary search project " + marker + "?")
            require_check(report, "native_recall_resolves_the_actual_observation_source",
                          source.document_id in {item["document_id"] for item in recalled["sources"]})
            report["recall"] = recalled
            async with fixture.pool.acquire() as connection:
                images = await SourceImageService(fixture.sources, None)._direct(connection, source)
            require_check(report, "canonical_observation_keeps_exact_image_identity_and_clock",
                          len(images) == 1 and images[0].frame_id == frame and images[0].captured_at == when)
            # Delete the actual conversation parent, not merely the derived result.
            await fixture.sources.delete_source(fixture.owner, fixture.area, user_source_id, expected_revision=1)
            try:
                await fixture.sources.read(fixture.owner, fixture.area, source_id)
            except SourceError:
                blocked = True
            else:
                blocked = False
            require_check(report, "local_read_barrier_closes_before_vendor_cleanup", blocked)
            require_check(report, "parent_deletion_revokes_the_derived_visual_source", blocked)
            await drain(fixture, worker, report, label="ongoing-observation-delete")
            require_check(report, "native_observation_document_is_removed",
                          await client.document(source.bank_name, source.document_id) is None)
            require_check(report, "native_conversation_parent_is_removed",
                          await client.document(user_source.bank_name, user_source.document_id) is None)
            after = await retrieval.recall(fixture.owner, fixture.area, "What do you know about " + marker + "?")
            require_check(report, "deleted_observation_is_not_recalled",
                          all(item["source_id"] != str(source_id) for item in after["sources"]))
            async with fixture.pool.acquire() as connection:
                remaining = await connection.fetchval("select count(*) from perception_observations")
                tombstones = await connection.fetchval("select count(*) from jarvis_observation_sources")
            require_check(report, "operational_original_removed_but_replay_identity_retained",
                          remaining == 0 and tombstones == 1)
            report["transport"] = client.diagnostics()
        report["source_fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
                          "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                    for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["limits"] = [
            "Scripted visual interpretation, not actual Gemini analysis, OCR quality or live glasses.",
            "Image identities and clocks are tested; no pixels were uploaded or resolved.",
            "No live API authority switch, complete notification/policy-input lifecycle or migration acceptance.",
        ]
        save(root / "observation-publication-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    args = parser.parse_args()
    with hindsight_service(args.root, runtime=args.runtime, api_key=JarvisSettings().OPENAI_API_KEY,
                           max_starts=1) as service:
        report = asyncio.run(exercise(args.root, service))
    print(json.dumps({"checks": report["checks"], "error": report.get("error"),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "source_fixture_cleanup_returned": report.get("source_fixture_cleanup_returned")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
