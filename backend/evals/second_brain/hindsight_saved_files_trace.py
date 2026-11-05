"""Native selected file tools, source-backed Hindsight recall and deletion acceptance."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import secrets
import time
import traceback
from types import SimpleNamespace
from uuid import UUID, uuid4

import asyncpg

from app.experience.repository import ToolExecutionRepository
from app.files.repository import JarvisFileRepository
from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.selection import configure_selected_memory
from app.realtime.router import _create_context_capture
from app.runtime.feature_config import KnowledgeConfiguration, MemoryAuthority, TaskLimits
from app.settings import JarvisSettings
from app.tasks.repository import TaskRepository
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.hindsight_foreground_memory_trace import settled
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture


async def exercise(root, service):
    report = {"checks": {}, "synthetic_only": True, "production_database_used": False,
              "storage_is_double": True, "native_hindsight": True, "model_tool_selection_tested": False,
              "full_stage_accepted": False}
    started = time.monotonic()
    try:
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(service["source_database_url"],
                root / "database-manifests", test_name="selected-file-memory"))
            runtime = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner,
                configuration=KnowledgeConfiguration(base_url=service["url"], api_key=service["token"],
                    area_id=fixture.area, http_timeout_seconds=10, poll_seconds=0.1),
                tasks=TaskRepository(fixture.pool, TaskLimits()))
            stack.push_async_callback(runtime.close)
            storage = MemoryStorage()
            settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=fixture.owner)
            services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=runtime,
                knowledge_issue=None, settings=settings,
                tool_modules=(FileToolModule(JarvisFileRepository(fixture.pool), storage),),
                assistant_context_providers=(), background_task_service=None)
            configure_selected_memory(services)
            files = next(module.repository for module in services.tool_modules if isinstance(module, FileToolModule))
            require_check(report, "selected_file_catalog_uses_canonical_publication",
                          isinstance(files, SourceFileRepository))
            runtime.start()
            async with fixture.pool.acquire() as connection:
                sessions = [await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner) for _ in range(2)]
            captures, executors = {}, {}
            for session in sessions:
                captures[session] = _create_context_capture(services, owner_id=fixture.owner, session_id=session,
                    provider="openai", audio_session_id=str(uuid4()))
                stack.push_async_callback(captures[session].close)
                registry = create_tool_registry(modules=services.tool_modules,
                    context=ToolBuildContext(settings, fixture.owner, session))
                executors[session] = JarvisToolExecutor(registry,
                    execution_repository=ToolExecutionRepository(fixture.pool),
                    owner_id=fixture.owner, session_id=session, context_capture=captures[session],
                    require_source_tracking=True)
            generations = {session: 0 for session in sessions}
            responses = {}
            async def request(session, text):
                capture = captures[session]
                generations[session] += 1
                identity = uuid4().hex
                capture.admit_openai_commit("commit-" + identity, generations[session])
                capture.openai_committed("item-" + identity)
                capture.openai_transcript("item-" + identity, text)
                if not await capture.flush():
                    raise RuntimeError("native_file_request_capture_failed")
                response = "response-" + identity
                capture.bind_openai_response(response, generations[session])
                capture.bind_openai_call("probe-" + identity, response)
                record = await capture.request_for_call("probe-" + identity)
                if record is None:
                    raise RuntimeError("native_file_request_not_bound")
                responses[record.id] = response
                return record
            async def invoke(name, arguments, record, call_id):
                captures[record.session_id].bind_openai_call(call_id, responses[record.id])
                return await executors[record.session_id].execute(name, arguments, provider_call_id=call_id)

            marker = "WORKSHOP-" + secrets.token_hex(4)
            control_marker = "GARDEN-" + secrets.token_hex(4)
            content = ("Synthetic workshop project " + marker + " is paused, not completed.\n"
                       "The next step is motor calibration before the robot arm can be tested.\n"
                       "This is a draft produced by an assistant, not a verified physical observation.\n")
            parameters = {"filename": "robot-progress.txt", "content": content,
                          "description": "Synthetic workshop draft " + marker}
            first_request = await request(sessions[0], "Save a draft of the workshop notes in robot-progress.txt.")
            first = await invoke("create_text_file", parameters, first_request, "create-first")
            require_check(report, "actual_file_tool_publishes_original", first["status"] == "success")
            repeated = await invoke("create_text_file", parameters, first_request, "create-repeated")
            require_check(report, "new_call_id_reuses_file_source_and_storage",
                          repeated["status"] == "success" and repeated["file"]["id"] == first["file"]["id"]
                          and storage.uploads == 1)
            control_request = await request(sessions[0], "Save a separate draft of the garden plan.")
            control_result = await invoke("create_text_file", {
                "filename": "garden-plan.txt", "content": "Synthetic garden " + control_marker +
                    " needs its soil moisture checked before planting basil.\n",
                "description": "Independent synthetic garden draft"}, control_request, "create-control")
            require_check(report, "unrelated_file_control_created", control_result["status"] == "success")
            source_id = UUID(first["file"]["memory_source"]["source_id"])
            control_id = UUID(control_result["file"]["memory_source"]["source_id"])
            report["initial_queue"] = await settled(runtime, fixture)
            source = await runtime.sources.read(fixture.owner, fixture.area, source_id)
            control = await runtime.sources.read(fixture.owner, fixture.area, control_id)
            for original in (source, control):
                runtime.client.verify_document(original,
                    await runtime.client.document(original.bank_name, original.document_id))
            report["file_original_check"] = {"content_equal": source.content["text"] == content,
                "trust": source.trust.value, "session_id": str(source.session_id),
                "expected_session_id": str(sessions[0])}
            require_check(report, "native_hindsight_has_exact_file_originals",
                          source.content["text"] == content and source.trust.value == "assistant_output"
                          and source.session_id == sessions[0])
            query = "What remains unfinished for the workshop project " + marker + "?"
            search_request = await request(sessions[1], query)
            found = await invoke("search_memory", {"query": query}, search_request, "search-file")
            report["search"] = found
            matches = [item for item in found.get("results", []) if item.get("source_id") == str(source_id)]
            require_check(report, "cross_session_native_recall_returns_the_file_source",
                          found["status"] == "success" and len(matches) == 1
                          and matches[0]["session_id"] == str(sessions[0])
                          and matches[0]["trust"] == "assistant_output")
            explained = await invoke("explain_memory", {"memory_id": str(source_id)},
                                     search_request, "explain-file")
            require_check(report, "original_expansion_preserves_exact_text_and_attribution",
                          explained["status"] == "success" and explained["content_page"]["complete"]
                          and json.loads(explained["content_page"]["text"])["text"] == content
                          and explained["source"]["trust"] == "assistant_output")
            file_read = await invoke("get_file_metadata", {"file_id": first["file"]["id"]},
                                     search_request, "metadata-file")
            require_check(report, "recalled_source_resolves_to_same_retained_file",
                          file_read["status"] == "success"
                          and file_read["file"]["memory_source"]["source_id"] == str(source_id))
            deletion_request = await request(sessions[1], "Delete the saved workshop draft file.")
            deleted = await invoke("delete_file", {"file_id": first["file"]["id"]},
                                   deletion_request, "delete-original")
            report["deletion"] = deleted
            require_check(report, "file_delete_acknowledges_local_and_storage_scope",
                          deleted["status"] == "success" and deleted["deleted"]
                          and deleted["storage_cleanup"] == "deleted"
                          and not deleted["full_erasure_verified"])
            denied = await invoke("explain_memory", {"memory_id": str(source_id)},
                                  deletion_request, "read-deleted")
            replay = await invoke("create_text_file", parameters, first_request, "late-create-replay")
            require_check(report, "deleted_original_and_late_save_replay_are_rejected",
                          denied["status"] == "failed" and replay["status"] == "failed"
                          and storage.uploads == 2)
            report["deletion_queue"] = await settled(runtime, fixture)
            require_check(report, "deleted_file_document_is_absent_from_real_hindsight",
                          await runtime.client.document(source.bank_name, source.document_id) is None)
            runtime.client.verify_document(control,
                await runtime.client.document(control.bank_name, control.document_id))
            require_check(report, "unrelated_native_document_survives_deletion", True)
            after = await invoke("search_memory", {"query": query}, deletion_request, "search-after-delete")
            require_check(report, "native_recall_does_not_return_deleted_file",
                          after["status"] == "success" and all(
                              item.get("source_id") != str(source_id) for item in after.get("results", [])))
            control_query = "What should happen before planting basil in " + control_marker + "?"
            control_search = await invoke("search_memory", {"query": control_query},
                                          deletion_request, "search-control")
            require_check(report, "positive_control_still_recalls_independent_file",
                          control_search["status"] == "success" and any(
                              item.get("source_id") == str(control_id) for item in control_search.get("results", [])))
            for capture in captures.values():
                await capture.close()
            cleanup = await files.delete_source_file(owner_id=fixture.owner, file_id=UUID(control_result["file"]["id"]))
            await settled(runtime, fixture)
            require_check(report, "both_file_objects_and_native_documents_disposed",
                          cleanup["deleted"] and not storage.objects
                          and await runtime.client.document(control.bank_name, control.document_id) is None)
            report["transport"] = runtime.client.diagnostics()
            await runtime.close()
            require_check(report, "ingestion_and_file_recovery_workers_stop",
                          runtime.task.done() and (runtime.file_task is None or runtime.file_task.done()))
        report["source_fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
            "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["limits"] = [
            "Actual selected tools, captured requests, background runtime and Hindsight; storage is an explicit double.",
            "Separate retained-image acceptance covers actual Supabase/Hermes; this test does not repeat it.",
            "The test drives tool calls, not GPT delegation or tool selection, TTS, phone UI or physical glasses.",
            "Deleting a file does not delete independently retained user requests or every mention of its topic.",
            "No personal-history migration, live authority switch or complete S17/M04 acceptance is claimed.",
        ]
        save(root / "saved-files-memory-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    args = parser.parse_args()
    with hindsight_service(args.root, runtime=args.runtime, api_key=JarvisSettings().OPENAI_API_KEY,
                           max_starts=1) as service:
        report = asyncio.run(exercise(args.root, service))
    print(json.dumps({key: report.get(key) for key in
        ("checks", "error", "elapsed_seconds", "source_fixture_cleanup_returned")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
