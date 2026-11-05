"""Native cleanup recovery with explicit before-delete and lost-response fault injection."""
import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import json
from pathlib import Path
import secrets
import time
import traceback
from unittest.mock import patch
from uuid import uuid4

import asyncpg

from app.knowledge.hindsight import HindsightClient, HindsightError
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
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
              "model_tool_selection_tested": False, "faults_are_injected": True}
    started = time.monotonic()
    try:
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "database-manifests", test_name="native-cleanup-recovery"))
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
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1, max_failures=1), client)

            async def invoke(name, source, text, **arguments):
                request = await history.append(fixture.owner, ContextIngress(
                    session_id=session, source_namespace="voice:native-cleanup-recovery",
                    source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript",
                    trust="user_input", source_time=datetime.now(UTC), payload={"text": text}))
                with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=request.id,
                                          request_revision=request.revision):
                    return await tools[name].handler({"source_id": str(source.source_id), **arguments})

            async def row(source, operation):
                async with fixture.pool.acquire() as connection:
                    return await connection.fetchrow("""select * from jarvis_memory_outbox
                        where owner_id=$1 and document_id=$2 and operation=$3""",
                        fixture.owner, source.document_id, operation)

            async def recover(source, generation, label):
                retained = await row(source, "retain")
                native_delete = client.delete_document

                async def unavailable(bank, document):
                    if (bank, document) != (source.bank_name, source.document_id):
                        raise RuntimeError("Unexpected fixture deletion target")
                    raise HindsightError("fixture_before_delete_failure", retryable=True)

                with patch.object(client, "delete_document", side_effect=unavailable):
                    status = await worker.step(fixture.owner, "native-failure")
                failed = await row(source, "delete")
                require_check(report, label + "_failure_before_delete_keeps_native_document",
                    status == "failed" and failed["status"] == "failed"
                    and await client.document(source.bank_name, source.document_id) is not None)
                inventory = tuple(failed["cleanup_operation_ids"] or ())
                require_check(report, label + "_prepared_inventory_binds_original_retention",
                    bool(inventory) and inventory[0] == retained["id"] and retained["dispatch_count"] == 1)

                async def retry():
                    result = await invoke("retry_memory_source_cleanup", source,
                        "Finish the cleanup already requested for this source.", expected_generation=generation)
                    require_check(report, label + "_retry_accepted_" + str(len(report["worker_events"])),
                        result["status"] == "success" and result["source_state"]["requeued"] == 1
                        and result["source_state"]["cleanup_completion_verified"] is False
                        and result["source_state"]["provider_contacted"] is False)

                await retry()
                async def lost_response(bank, document):
                    if (bank, document) != (source.bank_name, source.document_id):
                        raise RuntimeError("Unexpected fixture deletion target")
                    await native_delete(bank, document)
                    raise HindsightError("fixture_lost_delete_response", retryable=True)

                with patch.object(client, "delete_document", side_effect=lost_response):
                    status = await worker.step(fixture.owner, "native-lost-response")
                lost = await row(source, "delete")
                require_check(report, label + "_lost_response_is_failed_not_false_success",
                    status == "failed" and lost["id"] == failed["id"] and lost["status"] == "failed"
                    and tuple(lost["cleanup_operation_ids"]) == inventory
                    and await client.document(source.bank_name, source.document_id) is None)
                report["worker_events"].append({"phase": label, "status": "injected_lost_response"})
                await retry()
                await drain(fixture, worker, report, label=label + "-recovery")
                done = await row(source, "delete")
                require_check(report, label + "_recovery_preserves_id_inventory_and_fences",
                    done["id"] == failed["id"] and done["status"] == "done"
                    and done["fence"] > lost["fence"] > failed["fence"]
                    and tuple(done["cleanup_operation_ids"]) == inventory)
                require_check(report, label + "_native_document_and_operations_are_gone",
                    await client.document(source.bank_name, source.document_id) is None
                    and all([await client.operation(source.bank_name, operation_id) is None
                             for operation_id in inventory]))
                final_retain = await row(source, "retain")
                require_check(report, label + "_no_duplicate_retention",
                    final_retain["id"] == retained["id"]
                    and final_retain["dispatch_count"] == retained["dispatch_count"])
                result = await invoke("retry_memory_source_cleanup", source,
                    "Check whether cleanup still needs retrying.", expected_generation=generation)
                require_check(report, label + "_completed_cleanup_not_requeued",
                    result["status"] == "success" and result["source_state"]["requeued"] == 0)

            original = await fixture.publish("Synthetic robotics project is complete.", source_key="recovery-target")
            control = await fixture.publish("Synthetic blue umbrella stays at the office.", source_key="control")
            await drain(fixture, worker, report, label="recovery-positive-controls")
            for source in (original, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "both_native_positive_controls_exist", True)
            corrected = await invoke("correct_memory_source", original,
                "Correct that: calibration is still needed.", expected_revision=1,
                replacement="Synthetic robotics project still needs motor calibration.")
            require_check(report, "actual_correction_accepted", corrected["status"] == "success")
            current = await fixture.sources.read(fixture.owner, fixture.area, original.source_id)
            await recover(original, current.generation, "obsolete")
            client.verify_document(current, await client.document(current.bank_name, current.document_id))
            forgotten = await invoke("forget_memory_source", current, "Stop recalling this source; keep its originals.",
                                     expected_revision=current.revision)
            require_check(report, "actual_forgetting_accepted",
                          forgotten["status"] == "success" and forgotten["source_state"]["state"] == "forgotten")
            await recover(current, forgotten["source_state"]["generation"], "forgotten")
            async with fixture.pool.acquire() as connection:
                preserved = await connection.fetchval("""select count(*) from jarvis_context_payloads
                    where owner_id=$1 and id=any($2::uuid[]) and availability='available' and content is not null""",
                    fixture.owner, [original.payload_id, current.payload_id])
            require_check(report, "cleanup_did_not_physically_delete_preserved_originals", preserved == 2)
            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            absent = [await client.document(source.bank_name, source.document_id) is None
                      for source in (original, current)]
            state = await tools["inspect_memory_source_state"].handler({"source_id": str(original.source_id)})
            require_check(report, "restart_keeps_control_and_completed_cleanup",
                all(absent) and state["source_state"]["state"] == "forgotten"
                and state["source_state"]["provider_cleanup"]["cleanup_recorded"] == 2)
            for source in (current, control):
                await fixture.sources.delete_source(
                    fixture.owner, fixture.area, source.source_id, expected_revision=source.revision)
            await drain(fixture, worker, report, label="recovery-fixture-disposal")
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
            "Failures were injected around real native deletion, not an observed spontaneous outage.",
            "Actual tools and durable recovery were exercised, not model tool selection or live glasses.",
            "Recovery does not imply erasure of preserved originals, independent copies, logs or backups.",
            "A native service restart is not an older-backup restore or full migration acceptance.",
        ]
        save(root / "cleanup-recovery-result.json", report)
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
