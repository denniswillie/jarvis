"""Native archive/reactivation acceptance using only an owned local Hindsight fixture."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
import time
from uuid import uuid4

import asyncpg

from app.knowledge.areas import KnowledgeAreaService
from app.knowledge.area_tools import KnowledgeAreaToolModule
from app.knowledge.hindsight import HindsightClient
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture


async def exercise(root, service):
    report = {"checks": {}, "worker_events": [], "stage_accepted": False, "synthetic_only": True,
              "production_database_used": False, "tool_selection_by_model_tested": False}
    started = time.monotonic()
    try:
        # Match the native restore fixture: extensions belong to the separate
        # source database's public schema, not a disposable application schema.
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "database-manifests", test_name="native-area-archive"))
            client = HindsightClient(service["url"], service["token"], timeout=5)
            stack.push_async_callback(client.close)
            tasks = TaskRepository(fixture.pool, TaskLimits())
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            module = KnowledgeAreaToolModule(KnowledgeAreaService(fixture.sources, tasks), fixture.owner)
            tools = {tool.name: tool for tool in module.build_tools(ToolBuildContext(None, fixture.owner, session))}

            async def state(target, revision):
                async with tasks.transaction(fixture.owner) as connection:
                    payload = await tasks.payload(connection, fixture.owner,
                                                  {"text": "Set the synthetic project area to " + target})
                    request = await connection.fetchrow("""insert into jarvis_context_events(
                        owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                        values($1,$2,'fixture-native-area',$3,'user_request','final','user_input',$4)
                        returning id,revision""", fixture.owner, session, str(uuid4()), payload)
                with bind_tool_invocation(provider_call_id=str(uuid4()),
                                         request_event_id=request["id"], request_revision=request["revision"]):
                    result = await tools["set_knowledge_area_state"].handler({
                        "area_id": str(fixture.area), "state": target, "expected_revision": revision})
                require_check(report, target + "_tool_accepted", result["status"] == "success")
                return result

            async def intent(source):
                async with fixture.pool.acquire() as connection:
                    return await connection.fetchrow("""select id,status,dispatch_count from jarvis_memory_outbox
                        where owner_id=$1 and document_id=$2 and operation='retain'""",
                        fixture.owner, source.document_id)

            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            retained = await fixture.publish(
                "Synthetic project Atlas is paused, not completed. Its robot still needs motor calibration.",
                source_key="archived-existing")
            await drain(fixture, worker, report, label="archive-positive-control")
            client.verify_document(retained, await client.document(retained.bank_name, retained.document_id))
            retained_intent = await intent(retained)
            require_check(report, "native_original_and_completed_operation_exist",
                (await client.operation(retained.bank_name, retained_intent["id"]))["status"] == "completed")
            pending = await fixture.publish(
                "Synthetic project Atlas will use the existing calibration fixture when work resumes.",
                source_key="archived-pending")
            pending_intent = await intent(pending)
            archived = await state("archived", 1)
            await fixture.ready_jobs()
            require_check(report, "archive_does_not_dispatch_pending_original",
                          await worker.step(fixture.owner, "archived-no-dispatch") is None
                          and (await intent(pending))["dispatch_count"] == 0)
            hidden = False
            try:
                await fixture.sources.read(fixture.owner, fixture.area, retained.source_id)
            except SourceError:
                hidden = True
            require_check(report, "ordinary_source_read_is_hidden", hidden)
            client.verify_document(retained, await client.document(retained.bank_name, retained.document_id))
            require_check(report, "archive_preserves_actual_native_document", True)

            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            client.verify_document(retained, await client.document(retained.bank_name, retained.document_id))
            require_check(report, "native_document_and_operation_survive_archived_restart",
                (await client.operation(retained.bank_name, retained_intent["id"]))["status"] == "completed")
            require_check(report, "pending_intent_stays_paused_after_restart",
                          await worker.step(fixture.owner, "archived-restart") is None
                          and (await intent(pending))["id"] == pending_intent["id"])

            await state("active", archived["area"]["revision"])
            await drain(fixture, worker, report, label="archive-reactivation")
            for source in (retained, pending):
                current = await fixture.sources.read(fixture.owner, fixture.area, source.source_id)
                require_check(report, "original_identity_restored_" + source.document_id,
                              current.payload_id == source.payload_id and current.document_id == source.document_id)
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            after_pending, after_retained = await intent(pending), await intent(retained)
            require_check(report, "reactivation_reuses_existing_operation_ids_without_reindexing_old_document",
                after_pending["id"] == pending_intent["id"] and after_pending["dispatch_count"] == 1
                and after_retained["id"] == retained_intent["id"]
                and after_retained["dispatch_count"] == retained_intent["dispatch_count"])
            report["source_ids"] = [str(source.source_id) for source in (retained, pending)]
            report["operation_ids"] = [str(row["id"]) for row in (after_retained, after_pending)]

            for source in (retained, pending):
                await fixture.sources.delete_source(fixture.owner, fixture.area, source.source_id, expected_revision=1)
            await drain(fixture, worker, report, label="archive-fixture-cleanup")
            absent = [await client.document(source.bank_name, source.document_id) is None for source in (retained, pending)]
            require_check(report, "explicit_fixture_deletion_removes_both_native_documents", all(absent))
            report["transport"] = client.diagnostics()
        report["source_fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["limits"] = [
            "Native retention and document/operation preservation were exercised, not live audio or model tool selection.",
            "Archive is not forgetting, backup erasure, source correction or task cancellation.",
            "This service restart is not an older-backup restore or full migration acceptance.",
        ]
        save(root / "area-archive-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY, max_starts=2) as service:
        report = asyncio.run(exercise(args.root, service))
    import json
    print(json.dumps({"root": str(args.root), "checks": report["checks"], "error": report.get("error"),
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
