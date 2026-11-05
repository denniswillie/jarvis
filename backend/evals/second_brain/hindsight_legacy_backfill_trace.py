"""Native legacy backfill, lost-retain-response replay and suppression acceptance."""
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

from app.knowledge.backfill import LegacyBackfill, NAMESPACE
from app.knowledge.hindsight import HindsightClient, HindsightError
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.memory.retrieval.query import MemoryRetrievalQuery
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
                service["source_database_url"], root / "database-manifests", test_name="native-legacy-backfill"))
            client = HindsightClient(service["url"], service["token"], timeout=10)
            stack.push_async_callback(client.close)
            tasks = TaskRepository(fixture.pool, TaskLimits())
            history = ContextRepository(tasks)
            marker = "ROBOT-" + secrets.token_hex(4)
            async with fixture.pool.acquire() as c:
                session = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                user = await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                    values($1,$2,'user','audio',$3) returning id""", fixture.owner, session,
                    "Synthetic project " + marker + " is paused until motor calibration is finished.")
                assistant = await c.fetchval("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                    values($1,$2,'assistant','audio','I cannot independently verify that calibration is finished.') returning id""",
                    fixture.owner, session)
                claim = await c.fetchval("""insert into memory_items(
                    owner_id,memory_type,source,content,title,is_explicit,status,importance,confidence,metadata)
                    values($1,'semantic_fact','inferred_observation',$2,'Legacy robot claim',true,'active',0.5,0.5,
                           '{"measurement":123.000000000000000001}'::jsonb) returning id""",
                    fixture.owner, "A legacy model incorrectly marked " + marker + " as complete.")
                await c.execute("""insert into memory_evidence(
                    owner_id,memory_item_id,evidence_type,evidence_ref_id,evidence_time,confidence)
                    values($1,$2,'message',$3,clock_timestamp(),0.7)""", fixture.owner, claim, str(user))
                forgotten = await c.fetchval("""insert into memory_items(
                    owner_id,memory_type,source,content,title,is_explicit,status,importance,confidence)
                    values($1,'semantic_fact','user_statement','Synthetic obsolete secret','Forget target',true,'active',0.5,0.5)
                    returning id""", fixture.owner)
                await c.execute("""insert into memory_forgetting_tombstones(
                    owner_id,source_memory_id,memory_type,identity_fingerprint,reason)
                    values($1,$2,'semantic_fact',$3,'Synthetic explicit forgetting')""", fixture.owner, forgotten, "b" * 64)
            tools = {tool.name: tool for tool in SourceLifecycleToolModule(
                fixture.sources, tasks, fixture.owner, fixture.area).build_tools(
                    ToolBuildContext(None, fixture.owner, session))}
            importer = LegacyBackfill(fixture.sources)
            run_id = uuid4()
            first = await importer.run(fixture.owner, fixture.area, run_id=run_id, batch_size=1)
            require_check(report, "first_batch_is_not_false_completion",
                          first["status"] == "pending" and not first["ready_for_cutover"])
            for _ in range(8):
                local = await importer.run(fixture.owner, fixture.area, run_id=run_id, batch_size=1)
                if local["status"] != "pending":
                    break
            require_check(report, "local_manifest_accounts_for_all_rows",
                          local["local_manifest_complete"] and local["recorded_records"] == local["expected_records"])
            report["local_import"] = local
            async with fixture.pool.acquire() as c:
                source_rows = [dict(r) for r in await c.fetch("""select s.*,v.document_id,b.bank_name
                    from jarvis_knowledge_sources s join jarvis_knowledge_source_versions v
                    on v.owner_id=s.owner_id and v.source_id=s.id and v.revision=s.current_revision
                    join jarvis_memory_banks b on b.owner_id=v.owner_id and b.id=v.bank_id
                    where s.owner_id=$1 and s.area_id=$2 and s.namespace=$3""", fixture.owner, fixture.area, NAMESPACE)]
                operations_before = [(r["id"], r["document_id"], r["operation"]) for r in await c.fetch(
                    "select id,document_id,operation from jarvis_memory_outbox order by id")]
            mapped = {r["source_key"]: r for r in source_rows}
            user_source = await fixture.sources.read(fixture.owner, fixture.area, mapped["jarvis_messages/" + str(user)]["id"])
            claim_source = await fixture.sources.read(fixture.owner, fixture.area, mapped["memory_items/" + str(claim)]["id"])
            control = await fixture.sources.read(fixture.owner, fixture.area, mapped["jarvis_messages/" + str(assistant)]["id"])
            suppressed = mapped["memory_items/" + str(forgotten)]
            require_check(report, "legacy_claim_trust_and_exact_decimal_preserved",
                user_source.trust.value == "user_input" and claim_source.trust.value == "derived"
                and control.trust.value == "assistant_output"
                and claim_source.content["record"]["metadata"]["measurement"] == "123.000000000000000001")
            require_check(report, "legacy_tombstone_precedes_native_admission", suppressed["status"] == "forgotten")

            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            native_retain = client.retain
            injected = 0
            async def lose_response(*args, **kwargs):
                nonlocal injected
                result = await native_retain(*args, **kwargs)
                injected += 1
                raise HindsightError("fixture_lost_retain_response", retryable=True)
            with patch.object(client, "retain", side_effect=lose_response):
                for _ in range(12):
                    await fixture.ready_jobs()
                    status = await worker.step(fixture.owner, "backfill-lost-response")
                    if injected:
                        break
            require_check(report, "native_retain_was_sent_before_injected_loss",
                          injected == 1 and status == "outcome_unknown")
            resumed = await LegacyBackfill(fixture.sources).run(fixture.owner, fixture.area, run_id=run_id)
            require_check(report, "new_importer_instance_reuses_durable_receipts",
                          resumed["local_manifest_complete"] and resumed["sources_processed_in_batch"] == 0)
            async with fixture.pool.acquire() as c:
                operations_after = [(r["id"], r["document_id"], r["operation"]) for r in await c.fetch(
                    "select id,document_id,operation from jarvis_memory_outbox order by id")]
            require_check(report, "import_replay_keeps_native_operation_identities", operations_after == operations_before)
            await drain(fixture, worker, report, label="legacy-native-replay")
            for source in (user_source, claim_source, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "all_unsuppressed_originals_reach_real_hindsight", True)
            async with fixture.pool.acquire() as c:
                dispatches = await c.fetchval("select max(dispatch_count) from jarvis_memory_outbox where operation='retain'")
            require_check(report, "lost_response_did_not_duplicate_retention", dispatches == 1)
            require_check(report, "legacy_forgotten_document_never_appears",
                          await client.document(suppressed["bank_name"], suppressed["document_id"]) is None)
            require_check(report, "imported_session_is_not_a_fabricated_context_event",
                          user_source.session_id == session and user_source.context_event_id is None
                          and local["session_association_issues"] == 0)
            reader = SourceMemoryReadBackend(fixture.sources, SourceMemoryRetrieval(fixture.sources, client),
                owner_id=fixture.owner, area_id=fixture.area)
            recalled = await reader.search(owner_id=fixture.owner, request=MemoryRetrievalQuery(
                query=marker + " paused until motor calibration is finished", session_id=session))
            require_check(report, "native_recall_keeps_imported_session_scope",
                          any(item["source_id"] == str(user_source.source_id) for item in recalled["results"])
                          and all(item["session_id"] == str(session) for item in recalled["results"]))
            excluded = await reader.search(owner_id=fixture.owner, request=MemoryRetrievalQuery(
                query=marker + " paused until motor calibration is finished", session_id=uuid4()))
            require_check(report, "other_session_does_not_receive_imported_records",
                          not excluded["results"] and excluded["coverage"]["filtered_candidates"] > 0)

            request = await history.append(fixture.owner, ContextIngress(
                session_id=session, source_namespace="voice:native-backfill",
                source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript",
                trust="user_input", source_time=datetime.now(UTC), payload={"text": "Stop recalling that original robot statement."}))
            with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=request.id,
                                      request_revision=request.revision):
                result = await tools["forget_memory_source"].handler(
                    {"source_id": str(user_source.source_id), "expected_revision": user_source.revision})
            require_check(report, "actual_lifecycle_tool_accepts_imported_original", result["status"] == "success")
            await drain(fixture, worker, report, label="legacy-bound-source-suppression")
            require_check(report, "parent_and_bound_legacy_claim_removed_from_native",
                await client.document(user_source.bank_name, user_source.document_id) is None
                and await client.document(claim_source.bank_name, claim_source.document_id) is None)
            replay = await LegacyBackfill(fixture.sources).run(fixture.owner, fixture.area, run_id=uuid4())
            require_check(report, "new_import_run_cannot_resurrect_forgotten_derivatives",
                          replay["dispositions"].get("suppressed") == 4 and replay["local_manifest_complete"])
            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "restart_preserves_control_without_restoring_suppressed_source",
                          await client.document(user_source.bank_name, user_source.document_id) is None)
            # Delete derived leaves before their retained source originals.
            for source in (claim_source, user_source, control):
                await fixture.sources.delete_source(fixture.owner, fixture.area, source.source_id,
                                                    expected_revision=source.revision)
            await fixture.sources.delete_source(fixture.owner, fixture.area, suppressed["id"],
                                                expected_revision=suppressed["current_revision"])
            await drain(fixture, worker, report, label="legacy-fixture-disposal")
            require_check(report, "fixture_native_documents_disposed",
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
            "Synthetic native service acceptance, not production history migration or live glasses.",
            "Lost response is explicit fault injection around a real native retain.",
            "No model tool-selection or answer-quality acceptance is inferred from document retention.",
            "Operational rows, credentials and external bytes were not vendor-ingested.",
            "A source-local import receipt is not full cutover, external-copy deletion or backup recovery.",
        ]
        save(root / "legacy-backfill-result.json", report)
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
