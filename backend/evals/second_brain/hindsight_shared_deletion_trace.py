"""Native Hindsight shared-original deletion using the selected conversational tools."""
import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import json
from pathlib import Path
import secrets
import time
import traceback
from uuid import uuid4

import asyncpg

from app.knowledge.hindsight import HindsightClient
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.outbox import OutboxRepository
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture


async def exercise(root, service):
    report = {"checks": {}, "worker_events": [], "synthetic_only": True,
              "production_database_used": False, "full_stage_accepted": False,
              "model_tool_selection_tested": False}
    started = time.monotonic()
    marker = "MAPLE-" + secrets.token_hex(6)
    try:
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "database-manifests", test_name="native-shared-source-deletion"))
            client = HindsightClient(service["url"], service["token"], timeout=10)
            stack.push_async_callback(client.close)
            tasks = TaskRepository(fixture.pool, TaskLimits())
            history = ContextRepository(tasks)
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                other_area = await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Second fixture','robotics') returning id",
                    fixture.owner)
            tools = {tool.name: tool for tool in SourceLifecycleToolModule(
                fixture.sources, tasks, fixture.owner, fixture.area,
                selections=SessionKnowledgeAreas(fixture.sources, tasks, fixture.area)).build_tools(
                    ToolBuildContext(None, fixture.owner, session))}
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            retrieval = SourceMemoryRetrieval(fixture.sources, client, timeout_seconds=30)
            original = await fixture.publish("Synthetic robotics project " + marker + " uses VX4 motors.",
                                             source_key="shared-target")
            sibling = await fixture.sources.publish(fixture.owner, SourcePublication(
                area_id=other_area, namespace="fixture", source_key="shared-copy",
                payload_id=original.payload_id, source_type=original.source_type, trust=original.trust,
                source_time=original.source_time, time_basis=original.time_basis, expected_revision=0))
            current = await fixture.publish("Synthetic robotics project " + marker + " now uses VX5 motors.",
                                            source_key="shared-target", expected_revision=1)
            control = await fixture.publish("The synthetic user keeps a blue umbrella in the office.",
                                            source_key="unrelated-control")
            await drain(fixture, worker, report, label="shared-positive-controls")
            for source in (current, sibling, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "both_shared_current_documents_and_control_really_retained", True)
            for source in (current, sibling):
                recall = await retrieval.recall(fixture.owner, source.area_id, "What motors does project " + marker + " use?")
                require_check(report, "positive_recall_" + ("primary" if source is current else "second_area"),
                              source.document_id in {item["document_id"] for item in recall["sources"]})
            scope = await tools["inspect_memory_source_deletion"].handler({"source_id": str(current.source_id)})
            require_check(report, "inspection_finds_historical_shared_original_across_areas",
                {item["source_id"] for item in scope["deletion"]["shared_original_sources"]["items"]} == {str(sibling.source_id)})
            try:
                await fixture.sources.delete_source(fixture.owner, fixture.area, current.source_id,
                                                    expected_revision=current.revision)
            except SourceError as error:
                refused = error.code == "knowledge_source_shared_original"
            else:
                refused = False
            require_check(report, "single_source_scope_refused_before_original_removal", refused)

            request = await history.append(fixture.owner, ContextIngress(
                session_id=session, source_namespace="voice:native-shared-deletion",
                source_event_id=uuid4().hex, receipt_id=uuid4().hex, kind="user_transcript",
                trust="user_input", source_time=datetime.now(UTC),
                payload={"text": "Delete the named robotics originals and their shared copy in the second area, including old versions."}))
            arguments = {"sources": [{"source_id": str(source.source_id), "expected_revision": source.revision}
                                      for source in (current, sibling)]}
            with bind_tool_invocation(provider_call_id=uuid4().hex, request_event_id=request.id, request_revision=request.revision):
                result = await tools["delete_memory_source_group"].handler(arguments)
            require_check(report, "authorized_group_tool_accepts_exact_scope_without_erasure_claim",
                          result["status"] == "success" and result["scope_expanded"] is False and
                          all(item["read_barrier"] == "deleted" and not item["full_erasure_verified"]
                              for item in result["deletions"]))
            async with fixture.pool.acquire() as connection:
                remaining = await connection.fetchval("""select count(*) from jarvis_context_payloads
                    where owner_id=$1 and id=any($2::uuid[]) and content is not null""",
                    fixture.owner, [original.payload_id, current.payload_id])
            require_check(report, "current_and_shared_historical_original_bytes_removed", remaining == 0)
            for source in (current, sibling):
                try:
                    await fixture.sources.read(fixture.owner, source.area_id, source.source_id)
                except SourceError:
                    denied = True
                else:
                    denied = False
                require_check(report, "immediate_read_barrier_" + ("primary" if source is current else "second_area"), denied)
            await drain(fixture, worker, report, label="native-shared-group-deletion")
            absent = [await client.document(source.bank_name, source.document_id) is None
                      for source in (original, current, sibling)]
            require_check(report, "all_group_native_documents_absent", all(absent))
            for source in (current, sibling):
                state = await tools["inspect_memory_source_deletion"].handler({"source_id": str(source.source_id)})
                require_check(report, "recorded_scoped_cleanup_" + ("primary" if source is current else "second_area"),
                              state["deletion"]["state"] == "scoped_cleanup_recorded")
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "unrelated_native_control_preserved", True)
            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            for source in (current, sibling):
                recall = await retrieval.recall(fixture.owner, source.area_id, "What motors does project " + marker + " use?")
                require_check(report, "no_recall_or_document_after_restart_" + ("primary" if source is current else "second_area"),
                    all(item["source_id"] != str(source.source_id) for item in recall["sources"])
                    and await client.document(source.bank_name, source.document_id) is None)
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "unrelated_control_survives_restart", True)
            await fixture.sources.delete_source(fixture.owner, fixture.area, control.source_id, expected_revision=1)
            await drain(fixture, worker, report, label="shared-fixture-disposal")
            require_check(report, "control_fixture_document_disposed",
                          await client.document(control.bank_name, control.document_id) is None)
            report["transport"] = client.diagnostics()
        report["source_fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
                          "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                    for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["limits"] = [
            "Actual tools and Hindsight exercised; no native model tool selection, live registration or glasses test.",
            "Known native document absence and guarded recall are not proof of universal paraphrase or backup erasure.",
            "No external repository, delivered message, backup or unrelated source was authorized for removal.",
            "This restart is not an older-backup restoration or full S11 acceptance.",
        ]
        save(root / "shared-deletion-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    args = parser.parse_args()
    with hindsight_service(args.root, runtime=args.runtime, api_key=JarvisSettings().OPENAI_API_KEY, max_starts=2) as service:
        report = asyncio.run(exercise(args.root, service))
    print(json.dumps({"checks": report["checks"], "error": report.get("error"),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "source_fixture_cleanup_returned": report.get("source_fixture_cleanup_returned")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())

