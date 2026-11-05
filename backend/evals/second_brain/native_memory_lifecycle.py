"""Spoken lifecycle requests through production OpenAI sessions and selected memory.

Only original speech and the phone transport are synthetic. Tool selection,
ASR, request authorization, canonical capture, Hindsight and source mutations
are real. Structural outcomes do not establish semantic or physical acceptance.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from hashlib import sha256
import json
import logging
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace
from uuid import UUID

import asyncpg

from app.experience.repository import ToolExecutionRepository
from app.knowledge.current_context import SourceContextProvider
from app.knowledge.models import SourcePublication
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.selection import configure_selected_memory
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.runtime.feature_config import KnowledgeConfiguration, MemoryAuthority, TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.repository import TaskRepository
from app.tools.module import ToolBuildContext
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.native_voice_tasks import (
    ObservedExecutor, ObservedPhone, completed_audio_ids, foreground_context_providers,
    safe_app_event, speak, speech_texts, wait_until,
)
from evals.second_brain.source_fixture import source_fixture


def context_reset_notice(event, *, session_id, audio_session_id, pending=None):
    """Mirror the phone's exact connection/session-scoped reset admission."""
    reset = event.get("context_reset")
    if (event.get("type") not in {"session.context_invalidated", "session.closed"}
            or event.get("audio_session_id") != audio_session_id
            or not isinstance(reset, dict) or type(reset.get("protocol_version")) is not int
            or reset["protocol_version"] != 1 or reset.get("preserve_session") is not True):
        return None
    try:
        if UUID(event.get("session_id", "")) != UUID(str(session_id)):
            return None
        identity = str(UUID(reset.get("id", "")))
    except (ValueError, TypeError, AttributeError):
        return None
    if pending is not None and pending != identity:
        return None
    return identity


class LifecyclePhone(ObservedPhone):
    """A new transport per reconnect; never synthesize a new user request."""
    def __init__(self, *, session_id, audio_session_id, started):
        super().__init__(lambda: [])
        self.started = started
        self.session_id, self.audio_session_id = session_id, audio_session_id
        self.pending_reset = None
        self.reset_acknowledgments = []
        self.expected_reset_close = False

    async def send_json(self, event):
        await super().send_json(event)
        identity = context_reset_notice(event, session_id=self.session_id,
            audio_session_id=self.audio_session_id, pending=self.pending_reset)
        if identity is None:
            return
        self.pending_reset = identity
        if event["type"] == "session.context_invalidated":
            # There is no physical output queue in this synthetic client.
            # This ACK models clearing, not hearing the interrupted response.
            if identity not in self.reset_acknowledgments:
                await self.send("session.context_cleared", protocol_version=1,
                    reset_id=identity, audio_session_id=self.audio_session_id)
                self.reset_acknowledgments.append(identity)
        else:
            self.expected_reset_close = True


async def lifecycle_answer(connections, connect, speech, name, *, timeout=45):
    """Follow real resets; silence is a failed acknowledgment, not a retry prompt."""
    phone, _, _, _ = connections[-1]
    ended = await speak(phone, speech, name)
    deadline = time.monotonic() + timeout
    reason = "response_timeout"
    while time.monotonic() < deadline:
        events = [event for current, _, _, _ in connections for event in current.events]
        if completed_audio_ids(events):
            reason = "completed"
            break
        current, _, _, job = connections[-1]
        if job.done():
            await job
            if not current.expected_reset_close:
                reason = "unexpected_session_close"
                break
            if len(connections) >= 4:
                reason = "context_reconnect_budget_exhausted"
                break
            await connect()
        else:
            await asyncio.sleep(.025)
    events = [event for current, _, _, _ in connections for event in current.events]
    ids = completed_audio_ids(events)
    observations = [row for current, _, _, _ in connections for row in current.observations]
    first = next((row for row in observations if row["event"].get("type") == "audio.output.delta"
                  and row["event"].get("response_id") in ids), None)
    return {"utterance": name, "outcome": reason, "text": speech_texts(events, ids),
        "response_ids": sorted(ids), "input_end_seconds": ended - phone.started,
        "input_end_to_forwarded_audio_ms": (
            (first["seconds"] - (ended - phone.started)) * 1000 if first else None),
        "connections": len(connections), "speech_submissions": 1,
        "user_prompt_replayed": False, "synthetic_response_request_after_reset": False}


def load_cases():
    raw = Path(__file__).with_name("memory_lifecycle_cases.json").read_bytes()
    corpus = json.loads(raw)
    if corpus.get("version") != 1 or corpus.get("synthetic") is not True:
        raise ValueError("A versioned synthetic lifecycle corpus is required")
    identities = [item["id"] for item in corpus["cases"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Lifecycle cases must have unique identities")
    for case in corpus["cases"]:
        sources = {item["id"]: item for item in case["sources"]}
        if len(sources) != len(case["sources"]) or not case["utterance"]:
            raise ValueError("Lifecycle source identities and speech must be unambiguous")
        if not set(case["expected"]["targets"] + case["expected"]["unchanged"]) <= sources.keys():
            raise ValueError("The frozen oracle must name existing sources")
        for source in sources.values():
            if ("text" in source) == ("shared_from" in source):
                raise ValueError("Sources require either original text or an exact shared original")
            if "shared_from" in source and source["shared_from"] not in sources:
                raise ValueError("Missing shared original")
    return corpus["cases"], sha256(raw).hexdigest()


def prepare_speech(directory, cases):
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    for case in cases:
        original, aiff, wav = (directory / (case["id"] + suffix) for suffix in (".txt", ".aiff", ".wav"))
        original.write_text(case["utterance"])
        subprocess.run(["/usr/bin/say", "-r", "175", "-f", str(original), "-o", str(aiff)], check=True)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1",
                        str(aiff), str(wav)], check=True)
        read_pcm(wav)
        aiff.unlink()


def structural_checks(case, before, after, tools, *, mutating_tools):
    """Frozen action/source invariants only; do not grade prose with keyword rules."""
    accepted = [row for row in tools if row.get("result", {}).get("status") == "success"]
    names = [row["tool"] for row in accepted]
    expected = case["expected"]
    checks = {"untargeted_originals_unchanged": all(after[key] == before[key] for key in expected["unchanged"])}
    if expected["action"] == "clarify":
        checks["no_accepted_mutation"] = not (set(names) & set(mutating_tools))
    elif expected["action"] == "correct":
        checks["real_correction_tool"] = names.count("correct_memory_source") == 1
        checks["one_versioned_replacement"] = all(after[key]["revision"] == before[key]["revision"] + 1
            and after[key]["state"] == "active" and after[key]["trust"] == "derived"
            and after[key]["original_content_retained"] for key in expected["targets"])
    elif expected["action"] == "remember":
        checks["new_explicit_save"] = "remember" in names and "correct_memory_source" not in names
    elif expected["action"] == "forget":
        checks["real_forgetting_tool"] = "forget_memory_source" in names
        checks["originals_preserved_under_suppression"] = all(after[key]["state"] == "forgotten"
            and after[key]["revision"] == before[key]["revision"]
            and after[key]["original_content_retained"] for key in expected["targets"])
    elif expected["action"] == "delete":
        checks["explicit_group_tool"] = "delete_memory_source_group" in names
        checks["all_exact_target_originals_deleted"] = all(after[key]["state"] == "deleted"
            and not after[key]["original_content_retained"] for key in expected["targets"])
    else:
        raise ValueError("Unknown frozen lifecycle action")
    return checks


async def snapshot(fixture, bindings):
    result = {}
    async with fixture.pool.acquire() as connection:
        for key, source in bindings.items():
            row = await connection.fetchrow("""select s.current_revision as revision,s.status as state,
                p.content,v.trust,exists(select 1 from jarvis_context_payloads original
                    where original.owner_id=s.owner_id and original.id=$3 and original.content is not null)
                    as original_content_retained
                from jarvis_knowledge_sources s
                join jarvis_knowledge_source_versions v on v.owner_id=s.owner_id and v.source_id=s.id
                    and v.revision=s.current_revision
                join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                where s.owner_id=$1 and s.id=$2""", fixture.owner, source.source_id, source.payload_id)
            item = dict(row)
            if isinstance(item["content"], str):
                item["content"] = json.loads(item["content"])
            result[key] = item
    return result


async def run_case(root, speech, case, service, settings, iteration):
    output = root / (case["id"] + "-" + str(iteration))
    output.mkdir(mode=0o700)
    report = {"case": case["id"], "iteration": iteration, "checks": {}, "tools": [],
              "worker_events": [], "synthetic_only": True, "structural_pass": False,
              "semantic_review": "pending_independent_review", "physical_playback": "not_tested",
              "frontier_used": False, "foreground_model": settings.OPENAI_REALTIME_MODEL}
    started = time.monotonic()
    connections = []
    try:
        async with source_fixture(service["source_database_url"], output / "fixtures",
                                  test_name="native-lifecycle-" + case["id"]) as fixture:
            async with AsyncExitStack() as stack:
                tasks = TaskRepository(fixture.pool, TaskLimits())
                knowledge = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
                    configuration=KnowledgeConfiguration(base_url=service["url"], api_key=service["token"], area_id=fixture.area))
                stack.push_async_callback(knowledge.close)
                repository = SessionRepository(fixture.pool)
                async with fixture.pool.acquire() as connection:
                    session_id = await connection.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                    second_area = await connection.fetchval(
                        "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Second fixture','robotics') returning id",
                        fixture.owner)
                bindings = {}
                for source in case["sources"]:
                    area = second_area if source.get("area") == "second" else fixture.area
                    if "shared_from" in source:
                        original = bindings[source["shared_from"]]
                        bindings[source["id"]] = await fixture.sources.publish(fixture.owner, SourcePublication(
                            area_id=area, namespace="fixture:lifecycle-quality", source_key=source["id"],
                            payload_id=original.payload_id, source_type=original.source_type, trust=original.trust,
                            source_time=original.source_time, time_basis=original.time_basis, expected_revision=0))
                    else:
                        bindings[source["id"]] = await fixture.publish(source["text"], area_id=area,
                            namespace="fixture:lifecycle-quality", source_key=source["id"])
                await drain(fixture, knowledge.worker, report, label="lifecycle-quality-seed")
                for source in bindings.values():
                    knowledge.client.verify_document(source, await knowledge.client.document(source.bank_name, source.document_id))
                report["positive_controls"] = True
                report["before"] = await snapshot(fixture, bindings)
                report["source_bindings"] = {key: value.original_evidence() for key, value in bindings.items()}
                configured = settings.model_copy(update={"JARVIS_OWNER_ID": fixture.owner})
                services = SimpleNamespace(settings=configured, knowledge_runtime=knowledge, knowledge_issue=None,
                    memory_authority=MemoryAuthority.HINDSIGHT, tool_modules=(),
                    assistant_context_providers=(*foreground_context_providers(),
                        SourceContextProvider(knowledge.current_context, fixture.area, selections=knowledge.area_selections)))
                configure_selected_memory(services)
                report["selected_modules"] = [module.name for module in services.tool_modules]
                report["mutating_tools"] = sorted({tool.name for module in services.tool_modules
                    for tool in module.build_tools(ToolBuildContext(configured, fixture.owner, session_id))
                    if tool.mutates_state})
                async def connect():
                    audio_id = case["id"] + "-" + str(iteration) + "-" + str(len(connections) + 1)
                    phone = LifecyclePhone(session_id=session_id, audio_session_id=audio_id, started=started)
                    capture = SessionContextCapture(knowledge.context_repository, owner_id=fixture.owner,
                        session_id=session_id, provider="openai", audio_session_id=audio_id)
                    stack.push_async_callback(capture.close)
                    session = RealtimeAudioSession(websocket=phone, context=RealtimeSessionContext(
                        audio_session_id=audio_id, owner_id=fixture.owner, jarvis_session_id=session_id,
                        client_kind="phone_audio", session=await repository.get_session(fixture.owner, session_id)),
                        settings=configured, repository=repository, tool_modules=services.tool_modules,
                        context_providers=services.assistant_context_providers, context_capture=capture,
                        tool_execution_repository=ToolExecutionRepository(fixture.pool))
                    session.tool_executor = ObservedExecutor(session.tool_executor, report["tools"], started)
                    job = asyncio.create_task(session.run())
                    connections.append((phone, session, capture, job))
                    # Register wait first so LIFO cleanup closes before joining.
                    stack.push_async_callback(asyncio.wait_for, job, 20)
                    stack.push_async_callback(session.close, "s11_native_fixture_finished")
                    await wait_until(lambda: any(event["type"] == "session.created" for event in phone.events),
                                     running=(job,), timeout=25)

                await connect()
                report["answer"] = await lifecycle_answer(connections, connect, speech, case["id"])
                for _, session, capture, job in connections:
                    await session.close("s11_native_answer_finished")
                    await asyncio.wait_for(job, 20)
                    await capture.close()
                report["capture_issues"] = [capture.issue for _, _, capture, _ in connections]
                report["after"] = await snapshot(fixture, bindings)
                report["checks"] = structural_checks(case, report["before"], report["after"], report["tools"],
                                                     mutating_tools=report["mutating_tools"])
                report["checks"]["canonical_capture_healthy"] = not any(report["capture_issues"])
                report["checks"]["no_application_error"] = not any(event["type"] == "error"
                    for phone, _, _, _ in connections for event in phone.events)
                report["checks"]["actual_forwarded_audio"] = (
                    report["answer"]["outcome"] == "completed" and bool(report["answer"]["response_ids"]))
                history = await knowledge.context_repository.voice_history(fixture.owner, session_id)
                report["canonical_history"] = [record.as_context() for record in history["records"]]
                await drain(fixture, knowledge.worker, report, label="lifecycle-quality-after")
                report["transport"] = knowledge.client.diagnostics()
                report["structural_pass"] = all(report["checks"].values())
            report["runtime_closed"] = True
        report["fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
            "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)]}
        report["structural_pass"] = False
    finally:
        events = [event for phone, _, _, _ in connections for event in phone.events]
        report["user_transcripts"] = [safe_app_event(event) for event in events if event["type"] == "user.transcript.done"]
        report["spoken_text"] = speech_texts(events)
        report["observations"] = [row for phone, _, _, _ in connections for row in phone.observations]
        report["context_reconnects"] = [{"audio_session_id": phone.audio_session_id,
            "pending_reset": phone.pending_reset, "clear_acknowledgments": phone.reset_acknowledgments,
            "expected_reset_close": phone.expected_reset_close} for phone, _, _, _ in connections]
        report["audio_files"] = []
        for connection_index, (phone, _, _, _) in enumerate(connections, 1):
            for index, (item_id, pcm) in enumerate(phone.audio.items(), 1):
                path = output / ("forwarded-" + str(connection_index) + "-" + str(index) + ".wav")
                write_wav(path, bytes(pcm))
                report["audio_files"].append({"item_id": item_id, "path": str(path),
                    "audio_session_id": phone.audio_session_id, "source": "adapter_forwarded_not_playback"})
        report["elapsed_seconds"] = time.monotonic() - started
        save(output / "result.json", report)
    print(json.dumps({key: report.get(key) for key in ("case", "iteration", "structural_pass", "checks", "error", "elapsed_seconds")}), flush=True)
    return report


async def run(root, speech, cases, repeats, service, settings, corpus_digest):
    setup = await asyncpg.connect(service["source_database_url"], timeout=10)
    try:
        await setup.execute("create extension if not exists pgcrypto with schema public; create extension if not exists vector with schema public")
    finally:
        await setup.close()
    results = []
    try:
        for iteration in range(1, repeats + 1):
            for case in cases:
                results.append(await run_case(root, speech, case, service, settings, iteration))
    finally:
        save(root / "matrix.json", {"format": "jarvis.s11.native_lifecycle.v1", "corpus_digest": corpus_digest,
            "results": results, "semantic_review": "pending_independent_review", "physical_playback": "not_tested",
            "independent_cleanup": "required", "full_stage_accepted": False})
    return 0 if len(results) == len(cases) * repeats and all(row["structural_pass"] for row in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    all_cases, corpus_digest = load_cases()
    cases = [case for case in all_cases if case["id"] in args.cases]
    if len(cases) != len(set(args.cases)) or len(args.cases) != len(set(args.cases)):
        raise ValueError("Choose distinct frozen case IDs")
    logging.disable(logging.CRITICAL)
    prepare_speech(args.speech, cases)
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY, max_starts=1) as service:
        return asyncio.run(run(args.root, args.speech, cases, args.repeats, service, settings, corpus_digest))


if __name__ == "__main__":
    raise SystemExit(main())
