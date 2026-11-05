"""Real Hindsight ingestion through selected foreground memory tools and runtime."""
import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import json
from pathlib import Path
import secrets
import time
import traceback
from types import SimpleNamespace
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.selection import configure_selected_memory
from app.realtime.router import _create_context_capture
from app.runtime.feature_config import KnowledgeConfiguration, MemoryAuthority, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.repository import TaskRepository
from app.experience.repository import ToolExecutionRepository
from app.tools.executor import JarvisToolExecutor
from app.tools.registry import create_tool_registry
from app.tools.module import ToolBuildContext
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture


async def settled(runtime, fixture, *, timeout=150):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.task is None or runtime.task.done():
            raise RuntimeError("selected_runtime_worker_not_running")
        async with fixture.pool.acquire() as c:
            rows = await c.fetch("select status,count(*) as count from jarvis_memory_outbox group by status")
        counts = {r["status"]:r["count"] for r in rows}
        if counts.get("failed",0):
            raise RuntimeError("native_foreground_ingestion_failed")
        if all(status in {"done","cancelled","searchable"} for status in counts):
            return counts
        await asyncio.sleep(0.25)
    raise TimeoutError("native_foreground_ingestion_timeout")


async def exercise(root, service):
    report = {"checks":{},"synthetic_only":True,"production_database_used":False,
              "live_audio_model_tested":False,"full_stage_accepted":False}
    started = time.monotonic()
    try:
        setup = await asyncpg.connect(service["source_database_url"],timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(service["source_database_url"],
                root/"database-manifests",test_name="selected-foreground-memory"))
            configuration = KnowledgeConfiguration(base_url=service["url"],api_key=service["token"],
                area_id=fixture.area,
                http_timeout_seconds=10,poll_seconds=0.1)
            requests = TaskRepository(fixture.pool,TaskLimits())
            runtime = await KnowledgeRuntime.create(pool=fixture.pool,owner_id=fixture.owner,
                                                    configuration=configuration,tasks=requests)
            stack.push_async_callback(runtime.close)
            services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,knowledge_runtime=runtime,
                knowledge_issue=None,settings=SimpleNamespace(JARVIS_OWNER_ID=fixture.owner),
                tool_modules=(),assistant_context_providers=(),background_task_service=None)
            configure_selected_memory(services)
            runtime.start()
            async with fixture.pool.acquire() as c:
                sessions = [await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",fixture.owner)
                            for _ in range(2)]
            registries = {session:create_tool_registry(modules=services.tool_modules,
                context=ToolBuildContext(None,fixture.owner,session)) for session in sessions}
            captures = {session:_create_context_capture(services,owner_id=fixture.owner,session_id=session,
                provider="openai",audio_session_id=str(uuid4())) for session in sessions}
            for capture in captures.values():
                stack.push_async_callback(capture.close)
            executors = {session:JarvisToolExecutor(registries[session],
                execution_repository=ToolExecutionRepository(fixture.pool),owner_id=fixture.owner,session_id=session,
                context_capture=captures[session],require_source_tracking=True) for session in sessions}
            def catalog(session):
                return {tool.name:tool for tool in registries[session].list()}
            tools = catalog(sessions[0])
            require_check(report,"selected_catalog_has_no_legacy_memory_tools",
                {"remember","search_memory","explain_memory","forget_memory_source"} <= set(tools)
                and not {"get_memory_profile","get_semantic_fact_history","forget_memory"} & set(tools))
            history = runtime.context_for(fixture.owner)
            generations, responses = {session:0 for session in sessions}, {}
            async def request(session,text):
                capture = captures[session]
                generations[session] += 1
                generation = generations[session]
                identity = uuid4().hex
                capture.admit_openai_commit("commit-"+identity,generation)
                capture.openai_committed("item-"+identity)
                capture.openai_transcript("item-"+identity,text)
                if not await capture.flush():
                    raise RuntimeError("native_openai_capture_failed")
                response = "response-"+identity
                capture.bind_openai_response(response,generation)
                capture.bind_openai_call("probe-"+identity,response)
                record = await capture.request_for_call("probe-"+identity)
                if record is None:
                    raise RuntimeError("native_openai_request_not_bound")
                responses[record.id] = response
                return record
            marker = "ROBOT-" + secrets.token_hex(4)
            statement = "Finish calibrating the motors of synthetic project " + marker + "; it is not complete."
            original_request = await request(sessions[0],"Remember this goal: "+statement)
            parameters = {"memory_type":"goal","content":statement}
            async def invoke(tool,arguments,record,call_id):
                captures[record.session_id].bind_openai_call(call_id,responses[record.id])
                return await executors[record.session_id].execute(tool.name,arguments,provider_call_id=call_id)
            first = await invoke(tools["remember"],parameters,original_request,"first-save")
            require_check(report,"actual_foreground_tool_stores_source",first["status"]=="success")
            again = await invoke(tools["remember"],parameters,original_request,"new-provider-id")
            require_check(report,"new_provider_id_does_not_duplicate_the_statement",
                again["status"]=="success" and again["memory_id"]==first["memory_id"] and again["replayed"])
            capture = captures[sessions[0]]
            bound = first
            require_check(report,"real_realtime_factory_uses_selected_canonical_capture",
                          capture.repository is history and "source_bound_tool_result" in bound.audit_receipt())
            report["initial_queue"] = await settled(runtime,fixture)
            source = await runtime.sources.read(fixture.owner,fixture.area,UUID(first["memory_id"]))
            runtime.client.verify_document(source,await runtime.client.document(source.bank_name,source.document_id))
            require_check(report,"background_runtime_ingests_the_exact_original",
                source.trust.value=="tool_receipt" and source.content["content"]==statement)
            second_tools = catalog(sessions[1])
            query = "What remains unfinished for synthetic project "+marker+"?"
            search_request = await request(sessions[1],query)
            search = await invoke(second_tools["search_memory"],{"query":query},search_request,"cross-session-search")
            require_check(report,"selected_search_reads_evidence_across_sessions",
                search["status"]=="success" and any(
                    item.get("context_session_id")==str(sessions[0])
                    and marker in json.dumps(item.get("content"),ensure_ascii=True)
                    for item in search.get("results",[])))
            explained = await invoke(second_tools["explain_memory"],{"memory_id":first["memory_id"]},
                                     search_request,"expand-original")
            require_check(report,"selected_explanation_preserves_original_attribution",
                explained["status"]=="success" and "model_formulated_from_finalized_user_request" in
                explained["content_page"]["text"] and explained["source"]["trust"]=="tool_receipt")
            stop_request = await request(sessions[1],"Stop recalling the saved goal for "+marker+".")
            forgotten = await invoke(second_tools["forget_memory_source"],
                {"source_id":first["memory_id"],"expected_revision":source.revision},stop_request,"forget-save")
            require_check(report,"actual_selected_lifecycle_tool_accepts_forgetting",forgotten["status"]=="success")
            report["cleanup_queue"] = await settled(runtime,fixture)
            require_check(report,"forgotten_statement_is_removed_from_native",
                          await runtime.client.document(source.bank_name,source.document_id) is None)
            receipt = await history.get(fixture.owner,bound.event_id)
            require_check(report,"saved_receipt_copy_is_revoked_with_its_source",receipt.availability!="available")
            retry = await invoke(tools["remember"],parameters,original_request,"late-save-retry")
            require_check(report,"late_foreground_retry_cannot_resurrect_forgotten_statement",retry["status"]=="failed")
            gone = await invoke(second_tools["explain_memory"],{"memory_id":first["memory_id"]},
                                stop_request,"explain-forgotten")
            require_check(report,"selected_read_does_not_fall_back_to_legacy_after_forgetting",gone["status"]=="failed")
            report["transport"] = runtime.client.diagnostics()
            # Drain canonical capture before stopping its owned background consumer.
            for capture in captures.values():
                await capture.close()
            await settled(runtime,fixture)
            await runtime.close()
            require_check(report,"background_runtime_closes_its_consumer",runtime.task.done())
        report["source_fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"]={"type":type(error).__name__,"code":getattr(error,"code",None),
            "stack":[{"file":Path(frame.filename).name,"line":frame.lineno,"function":frame.name}
                     for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        report["limits"]=[
            "Actual selected tools, source capture and background Hindsight runtime with synthetic fixtures.",
            "Tool calls are explicitly driven; GPT tool choice, live audio, phone UI and glasses were not tested.",
            "Cross-session source retrieval is not a general factual-answer quality benchmark.",
            "No personal-history import, live authority switch, real email or unrelated external action.",
            "Forgetting one saved statement does not delete its separately retained user request.",
        ]
        save(root/"foreground-memory-result.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    args=parser.parse_args()
    with hindsight_service(args.root,runtime=args.runtime,api_key=JarvisSettings().OPENAI_API_KEY,max_starts=1) as service:
        report=asyncio.run(exercise(args.root,service))
    print(json.dumps({"checks":report["checks"],"error":report.get("error"),"elapsed_seconds":report["elapsed_seconds"],
                      "source_fixture_cleanup_returned":report.get("source_fixture_cleanup_returned")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__=="__main__":
    raise SystemExit(main())
