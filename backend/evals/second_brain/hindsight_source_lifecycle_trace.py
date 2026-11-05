"""Native source correction/forgetting through actual tools and owned Hindsight fixtures."""
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
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from app.knowledge.retrieval import SourceMemoryRetrieval
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
    marker = "CEDAR-" + secrets.token_hex(6)
    try:
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "database-manifests", test_name="native-source-lifecycle"))
            client = HindsightClient(service["url"], service["token"], timeout=10)
            stack.push_async_callback(client.close)
            tasks = TaskRepository(fixture.pool, TaskLimits())
            history = ContextRepository(tasks)
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            tools = {tool.name: tool for tool in SourceLifecycleToolModule(
                fixture.sources, tasks, fixture.owner, fixture.area).build_tools(
                    ToolBuildContext(None, fixture.owner, session))}
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            retrieval = SourceMemoryRetrieval(fixture.sources, client, timeout_seconds=30)

            async def invoke(name, source, text, **arguments):
                request = await history.append(fixture.owner, ContextIngress(
                    session_id=session, source_namespace="voice:native-source-lifecycle",
                    source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript",
                    trust="user_input", source_time=datetime.now(UTC), payload={"text": text}))
                with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=request.id,
                                          request_revision=request.revision):
                    return await tools[name].handler({
                        "source_id": str(source.source_id), "expected_revision": source.revision, **arguments})

            original = await fixture.publish("Synthetic robotics project " + marker + " is complete.",
                                             source_key="correction-target")
            control = await fixture.publish("The synthetic user keeps a blue umbrella at the office.",
                                            source_key="unrelated-control")
            await drain(fixture, worker, report, label="lifecycle-positive-controls")
            for source in (original, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            before = await retrieval.recall(fixture.owner, fixture.area, "What is the status of robotics project " + marker + "?")
            require_check(report, "original_is_actually_retrieved_before_correction",
                          original.document_id in {item["document_id"] for item in before["sources"]})

            corrected = await invoke(
                "correct_memory_source", original, "Correct that: the robot is paused, not complete; motor calibration remains.",
                replacement="Synthetic robotics project " + marker + " is paused, not complete; motor calibration remains.")
            require_check(report, "actual_correction_tool_accepts_current_user_request", corrected["status"] == "success")
            current = await fixture.sources.read(fixture.owner, fixture.area, original.source_id)
            historical = await fixture.sources.read(fixture.owner, fixture.area, original.source_id, revision=1)
            require_check(report, "old_original_is_explicit_history_and_correction_has_honest_provenance",
                          historical.content == original.content and not historical.is_current
                          and current.trust.value == "derived" and current.context_event_id is not None
                          and current.source_id == original.source_id and current.revision == 2)
            await drain(fixture, worker, report, label="native-correction")
            require_check(report, "old_native_document_removed_after_correction",
                          await client.document(original.bank_name, original.document_id) is None)
            client.verify_document(current, await client.document(current.bank_name, current.document_id))
            after = await retrieval.recall(fixture.owner, fixture.area, "What work remains for robotics project " + marker + "?")
            identifiers = {item["document_id"] for item in after["sources"]}
            require_check(report, "native_recall_returns_correction_not_old_current_fact",
                          current.document_id in identifiers and original.document_id not in identifiers)

            forgotten = await invoke("forget_memory_source", current,
                                     "Stop recalling this robotics source, but keep the original evidence.")
            require_check(report, "actual_forgetting_tool_suppresses_without_claiming_original_deletion",
                          forgotten["status"] == "success" and forgotten["source_state"]["state"] == "forgotten"
                          and forgotten["source_state"]["originals_deleted"] is False)
            await drain(fixture, worker, report, label="native-logical-forgetting")
            async with fixture.pool.acquire() as connection:
                originals_retained = await connection.fetchval("""select count(*) from jarvis_context_payloads
                    where owner_id=$1 and id=any($2::uuid[]) and availability='available' and content is not null""",
                    fixture.owner, [original.payload_id, current.payload_id])
            require_check(report, "both_local_originals_remain_after_provider_cleanup", originals_retained == 2)
            absent = [await client.document(source.bank_name, source.document_id) is None
                      for source in (original, current)]
            require_check(report, "all_forgotten_native_versions_are_absent", all(absent))
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "unrelated_native_original_is_preserved", True)

            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            recalled = await retrieval.recall(fixture.owner, fixture.area, "What is known about robotics project " + marker + "?")
            require_check(report, "forgotten_source_not_recalled_after_native_restart",
                          all(item["source_id"] != str(original.source_id) for item in recalled["sources"])
                          and await client.document(current.bank_name, current.document_id) is None)
            state = await tools["inspect_memory_source_state"].handler({"source_id": str(original.source_id)})
            require_check(report, "suppression_and_cleanup_receipt_survive_restart",
                          state["source_state"]["state"] == "forgotten"
                          and state["source_state"]["provider_cleanup"]["cleanup_recorded"] == 2)
            try:
                await fixture.publish("Automatic differently worded replay of " + marker,
                                      source_key="correction-target", expected_revision=2)
            except SourceError as error:
                refused = error.code == "knowledge_source_forgotten"
            else:
                refused = False
            require_check(report, "same_source_paraphrased_replay_cannot_unforget", refused)

            # Explicit owned-fixture disposal, separately authorized from logical forgetting.
            for source in (current, control):
                await fixture.sources.delete_source(
                    fixture.owner, fixture.area, source.source_id, expected_revision=source.revision)
            await drain(fixture, worker, report, label="source-lifecycle-fixture-disposal")
            require_check(report, "final_fixture_document_disposal",
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
            "Actual native retention/recall and tools were exercised, not model selection of tools or live glasses.",
            "Source-based suppression is not a semantic ban on all independently supplied mentions of a topic.",
            "Retained originals, shared evidence, external files, delivered messages and backups are not logically erased.",
            "This restart is not an older-backup restore or complete migration acceptance.",
        ]
        save(root / "source-lifecycle-result.json", report)
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
