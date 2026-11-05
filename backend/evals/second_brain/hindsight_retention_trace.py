"""Native automatic source expiry through real runtime, tools and Hindsight."""
import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import secrets
import time
import traceback
from uuid import uuid4

import asyncpg
from pydantic import SecretStr

from app.knowledge.hindsight import HindsightClient
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from app.knowledge.retention_tools import SourceRetentionToolModule
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import KnowledgeConfiguration, TaskLimits
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
              "model_tool_selection_tested": False, "physical_glasses": False}
    started = time.monotonic()
    marker = "CITRINE-" + secrets.token_hex(6)
    try:
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "database-manifests", test_name="native-source-retention"))
            tasks = TaskRepository(fixture.pool, TaskLimits())
            history = ContextRepository(tasks)
            runtime = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
                configuration=KnowledgeConfiguration(base_url=service["url"],
                    api_key=SecretStr(service["token"]), area_id=fixture.area, poll_seconds=0.1,
                    shutdown_seconds=10))
            stack.push_async_callback(runtime.close)
            client = HindsightClient(service["url"], service["token"], timeout=10)
            stack.push_async_callback(client.close)
            retrieval = SourceMemoryRetrieval(fixture.sources, client, timeout_seconds=30)
            async with fixture.pool.acquire() as db:
                session = await db.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",
                                            fixture.owner)
            tools = {tool.name: tool for tool in SourceRetentionToolModule(
                fixture.sources, tasks, fixture.owner).build_tools(ToolBuildContext(None, fixture.owner, session))}
            target = await fixture.publish(
                "Synthetic temporary project " + marker + " uses a red test enclosure.",
                source_key="expiry-target")
            held = await fixture.publish(
                "Synthetic retained project " + marker + "-HELD uses four servo motors.",
                source_key="referenced-source")
            control = await fixture.publish("The synthetic user keeps a blue umbrella at the office.",
                                            source_key="unrelated-control")
            await drain(fixture, runtime.worker, report, label="retention-positive-controls")
            for source in (target, held, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "all_three_native_originals_exist_before_expiry", True)
            before = await retrieval.recall(
                fixture.owner, fixture.area, "What enclosure does temporary project " + marker + " use?")
            require_check(report, "target_is_actually_recalled_before_expiry",
                          target.document_id in {row["document_id"] for row in before["sources"]})
            async with tasks.transaction(fixture.owner) as db:
                await tasks.payload(db, fixture.owner, {"text": "A retained dependent of the held project."},
                                    source_payload_ids=(held.payload_id,))
            request = await history.append(fixture.owner, ContextIngress(
                session_id=session, source_namespace="voice:native-retention", source_event_id=str(uuid4()),
                receipt_id=str(uuid4()), kind="user_transcript", trust="user_input",
                source_time=datetime.now(UTC),
                payload={"text": "Set the two selected synthetic sources to expire at the specified time, "
                                  "subject to retained-evidence holds. Keep the unrelated umbrella record."}))
            expiry = datetime.now(UTC) + timedelta(seconds=5)
            arguments = {}
            for source in (target, held):
                arguments[source.source_id] = {
                    "area_id": str(fixture.area), "source_id": str(source.source_id),
                    "expected_source_revision": source.revision,
                    "expected_source_generation": source.generation, "expected_policy_revision": 0,
                    "expires_at": expiry.isoformat()}
                with bind_tool_invocation(provider_call_id=uuid4().hex,
                        request_event_id=request.id, request_revision=request.revision):
                    accepted = await tools["set_memory_retention"].handler(arguments[source.source_id])
                require_check(report, "explicit_policy_accepted_" + ("target" if source is target else "held"),
                              accepted["status"] == "success"
                              and accepted["retention"]["status"] == "scheduled"
                              and accepted["full_erasure_verified"] is False)
            before_expiry = await runtime.retention.inspect(fixture.owner, fixture.area, target.source_id)
            require_check(report, "scheduling_does_not_claim_or_perform_immediate_deletion",
                          before_expiry["source_state"] == "active")
            runtime.start()
            waiting = time.monotonic()
            async with asyncio.timeout(240):
                while True:
                    target_state = await runtime.retention.inspect(fixture.owner, fixture.area, target.source_id)
                    held_state = await runtime.retention.inspect(fixture.owner, fixture.area, held.source_id)
                    if (target_state["retention"]["status"] == "expired"
                            and held_state["retention"].get("reason") == "retained_dependents"):
                        break
                    if runtime.retention_task.done():
                        runtime.retention_task.result()
                        raise RuntimeError("retention_loop_stopped_before_its_pass")
                    await asyncio.sleep(0.25)
            report["automatic_expiry_wait_seconds"] = time.monotonic() - waiting
            report["retention_states"] = {"target": target_state, "held": held_state}
            require_check(report, "real_periodic_runtime_expires_target_and_holds_referenced_source",
                          target_state["source_state"] == "deleted" and held_state["source_state"] == "active")
            async with fixture.pool.acquire() as db:
                original = await db.fetchrow("select availability,content from jarvis_context_payloads where id=$1",
                                             target.payload_id)
            require_check(report, "expired_original_bytes_are_removed",
                          original["availability"] == "deleted" and original["content"] is None)
            async with asyncio.timeout(240):
                while True:
                    receipt = await runtime.deletions.inspect(fixture.owner, fixture.area, target.source_id)
                    if receipt["state"] == "scoped_cleanup_recorded":
                        break
                    if receipt["state"] == "requires_attention":
                        raise RuntimeError("native_retention_cleanup_requires_attention")
                    if runtime.task.done():
                        runtime.task.result()
                        raise RuntimeError("ingestion_loop_stopped_before_cleanup")
                    await asyncio.sleep(0.25)
            report["deletion_receipt"] = receipt
            require_check(report, "existing_runtime_outbox_records_cleanup_without_universal_erasure_claim",
                          receipt["provider"]["cleanup_recorded"] == 1 and receipt["full_erasure_verified"] is False)
            require_check(report, "expired_native_document_is_absent",
                          await client.document(target.bank_name, target.document_id) is None)
            for source in (held, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "held_and_unenrolled_native_originals_remain", True)
            with bind_tool_invocation(provider_call_id=uuid4().hex,
                    request_event_id=request.id, request_revision=request.revision):
                replay = await tools["set_memory_retention"].handler(arguments[target.source_id])
            require_check(report, "new_call_id_replays_policy_without_resurrecting_expired_source",
                          replay["status"] == "success" and replay["replayed"]
                          and replay["retention"]["policy_revision"] == 1 and replay["source_state"] == "deleted")
            await runtime.close()
            require_check(report, "runtime_shutdown_drains_both_workers",
                          runtime.task.done() and runtime.retention_task.done() and runtime.file_task.done())
            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            after = await retrieval.recall(
                fixture.owner, fixture.area, "What enclosure does temporary project " + marker + " use?")
            require_check(report, "expired_source_not_recalled_after_native_restart",
                          all(row["source_id"] != str(target.source_id) for row in after["sources"])
                          and await client.document(target.bank_name, target.document_id) is None)
            for source in (held, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "unrelated_documents_survive_native_restart", True)
            tombstone = await tools["inspect_memory_retention"].handler({
                "area_id": str(fixture.area), "source_id": str(target.source_id)})
            require_check(report, "inspectable_tombstone_survives_restart_without_returning_original_text",
                          tombstone["status"] == "success" and tombstone["source_state"] == "deleted"
                          and tombstone["tombstone_retention"]["automatic_purge"] is False
                          and marker not in json.dumps(tombstone))
            try:
                await fixture.publish("A differently worded automatic replay of " + marker,
                                      source_key="expiry-target", expected_revision=1)
            except SourceError as error:
                refused = error.code == "knowledge_source_deleted"
            else:
                refused = False
            require_check(report, "suppression_identity_prevents_reingestion_after_expiry", refused)
            for source in (held, control):
                await fixture.sources.delete_source(
                    fixture.owner, fixture.area, source.source_id, expected_revision=source.revision)
            disposal = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            await drain(fixture, disposal, report, label="retention-fixture-disposal")
            require_check(report, "final_native_fixture_disposal",
                          all([await client.document(source.bank_name, source.document_id) is None
                               for source in (held, control)]))
            report["transport"] = client.diagnostics()
        report["source_fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
            "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["limits"] = [
            "Actual periodic runtime and provider cleanup; no model selection of expiry or physical glasses.",
            "Document absence and source-fenced recall do not prove universal deletion of all semantic derivatives.",
            "Restart is not restoration of an older backup; backups and external copies remain separate.",
            "Per-pass source count is bounded; arbitrarily large dependency/version cascades are not qualified.",
        ]
        save(root / "source-retention-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    args = parser.parse_args()
    with hindsight_service(args.root, runtime=args.runtime,
                           api_key=JarvisSettings().OPENAI_API_KEY, max_starts=2) as service:
        report = asyncio.run(exercise(args.root, service))
    print(json.dumps({key: report.get(key) for key in
        ("checks", "error", "elapsed_seconds", "source_fixture_cleanup_returned")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
