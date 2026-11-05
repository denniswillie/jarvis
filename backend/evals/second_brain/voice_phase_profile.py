"""Phase/query profiling of real task handlers; no native audio/model claim.

Only SQL digests, phase names, elapsed times and synthetic receipts are retained.
The query logger never stores arguments, credentials, connection addresses or SQL.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import ExitStack
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

from app.assistant.tool_limits import ToolCallBudget
from app.experience.repository import ToolExecutionRepository
from app.realtime.gemini.operations import GeminiOperations
from app.settings import JarvisSettings
from app.runtime.feature_config import TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskCreate
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from evals.second_brain.source_fixture import source_fixture


CURRENT_CALL = ContextVar("voice_profile_call", default=None)
CURRENT_PHASE = ContextVar("voice_profile_phase", default=None)


class Profile:
    def __init__(self):
        self.spans, self.queries, self.calls = [], [], []

    async def initialize_connection(self, connection):
        connection.add_query_logger(self.query)

    def query(self, record):
        call = CURRENT_CALL.get()
        if call is None:
            return
        self.queries.append({"call": call, "phase": CURRENT_PHASE.get(),
            "sql_digest": hashlib.sha256(record.query.encode()).hexdigest(),
            "verb": record.query.lstrip().split(None, 1)[0].upper() if record.query.strip() else "",
            "duration_ms": record.elapsed * 1000,
            "error_type": type(record.exception).__name__ if record.exception else None})

    def wrap(self, original, label):
        async def measured(*args, **kwargs):
            started = time.monotonic()
            token = CURRENT_PHASE.set(label)
            error = None
            try:
                return await original(*args, **kwargs)
            except BaseException as failure:
                error = type(failure).__name__
                raise
            finally:
                CURRENT_PHASE.reset(token)
                if (call := CURRENT_CALL.get()) is not None:
                    self.spans.append({"call": call, "phase": label,
                        "duration_ms": (time.monotonic()-started)*1000, "error_type": error})
        return measured

    def install(self, stack):
        methods = {
            JarvisToolExecutor: ("execute", "_invoke"),
            ToolExecutionRepository: ("start_execution_with_status", "finalize_execution", "get_executions_by_call_ids"),
            SessionContextCapture: ("flush", "request_for_call"),
            ContextRepository: ("append", "get", "request_evidence", "snapshot", "store_manifest", "_task_sessions", "_bind_archive_sources"),
            TaskContextBuilder: ("build",),
            TaskService: ("create", "mutate"),
            TaskRepository: ("create", "replay_create", "mutate", "replay_mutation", "_voice_source", "_source_available", "lock_owner"),
            TaskWorkerRegistry: ("require",),
            GeminiOperations: ("refresh",),
        }
        for cls, names in methods.items():
            for name in names:
                stack.enter_context(patch.object(cls, name, self.wrap(getattr(cls, name), cls.__name__+"."+name)))


def distribution(values):
    ordered=sorted(values)
    return {"n":len(ordered), "p50_ms":round(ordered[math.ceil(len(ordered)*.5)-1], 3),
            "p95_ms":round(ordered[math.ceil(len(ordered)*.95)-1], 3), "max_ms":round(max(ordered), 3)}


async def run(directory: Path, *, samples=6, history_sources=0, context_workers=0):
    if not 1 <= samples <= 30:
        raise ValueError("samples must be 1..30 per provider")
    if not 0 <= history_sources <= 256 or not 0 <= context_workers <= 2:
        raise ValueError("history_sources must be 0..256 and context_workers 0..2")
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    profile = Profile()
    success = False
    try:
        with ExitStack() as stack:
            profile.install(stack)
            async with source_fixture(JarvisSettings().SUPABASE_DB_URL, directory,
                test_name="s05_voice_phase_profile", pool_init=profile.initialize_connection) as fixture:
                tasks = TaskRepository(fixture.pool, TaskLimits(owner_queue_limit=2*samples+2, lease_seconds=300))
                workers = TaskWorkerRegistry(fixture.pool)
                providers = {"frontier": ConfiguredTaskProvider("fixture-only")}
                # Explicit simulated readiness, not an actual inference worker.
                await workers.advertise(worker_id="fixture-readiness", owner_id=fixture.owner,
                    host_id="fixture-host", providers=providers, ttl_seconds=300)
                service = TaskService(tasks, providers, registry=workers)
                sources = ContextRepository(tasks)
                async with fixture.pool.acquire() as connection:
                    session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                for index in range(history_sources):
                    await sources.append(fixture.owner, ContextIngress(
                        session_id=session, source_namespace="profile-history", source_event_id=str(index),
                        receipt_id=str(uuid4()), kind="user_transcript", trust="user_input",
                        payload={"text": f"Synthetic historical source {index}: preserve this recorded fixture constraint."},
                    ))
                context_claims = []
                for index in range(context_workers):
                    task = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(),
                        title=f"Synthetic context assembly {index}", requested_outcome="Read the synthetic session.",
                        session_id=session), model="fixture-only")
                    claim = await tasks.claim_next("context-profile-"+str(index), ("frontier",),
                                                  owner_id=fixture.owner)
                    if claim is None or claim.task.id != task.id:
                        raise RuntimeError("Synthetic context load was not claimed exactly")
                    context_claims.append(claim)

                async def context_load(claim, key):
                    token = CURRENT_CALL.set(key)
                    try:
                        return await TaskContextBuilder(sources).build(claim)
                    finally:
                        CURRENT_CALL.reset(token)

                adapters = {}
                captures = []
                try:
                    for provider in ("openai", "gemini"):
                        capture = SessionContextCapture(sources, owner_id=fixture.owner, session_id=session,
                            provider=provider, audio_session_id="fixture-"+uuid4().hex)
                        captures.append(capture)
                        registry = create_tool_registry(modules=(BackgroundTaskToolModule(service),),
                            context=ToolBuildContext(None, fixture.owner, session))
                        executions = ToolExecutionRepository(fixture.pool)
                        executor = (GeminiOperations(registry=registry, repository=executions,
                            owner_id=fixture.owner, session_id=session, audio_session_id=capture.audio_session_id,
                            context_capture=capture) if provider=="gemini" else
                            JarvisToolExecutor(registry, execution_repository=executions, owner_id=fixture.owner,
                                session_id=session, context_capture=capture))
                        adapters[provider] = (capture, executor)
                    for index in range(samples):
                        for provider, (capture, executor) in adapters.items():
                            task_ref = None
                            tools = ("start_background_task", "get_background_task") if context_workers else ("start_background_task",)
                            for tool in tools:
                                if tool == "get_background_task" and task_ref is None:
                                    continue
                                key=f"{provider}:{index}:{tool}"
                                loads = [asyncio.create_task(context_load(claim, key+f":worker-{number}"))
                                         for number,claim in enumerate(context_claims)]
                                token=CURRENT_CALL.set(key)
                                started=time.monotonic()
                                result=None
                                error=None
                                try:
                                    input_id=uuid4().hex
                                    capture.input_admitted(input_id)
                                    capture.input_final(input_id, "Have the expert check this synthetic fixture."
                                                        if tool=="start_background_task" else "Read the accepted fixture task status.")
                                    arguments=({"title":"Synthetic fixture", "requested_outcome":"Check the synthetic source.", "mode":"consult"}
                                               if tool=="start_background_task" else {"task_ref":task_ref})
                                    identity=uuid4().hex
                                    if provider=="gemini":
                                        call,_=executor.receive({"id":identity,"name":tool,"args":arguments},
                                                                namespace=1,turn_number=index+1)
                                        capture.bind_call(call.execution_id,input_id)
                                        result=await executor.execute(call, still_current=lambda:True,
                                            budget=ToolCallBudget(max_total_calls=5,max_repeated_calls=2))
                                    else:
                                        capture.bind_call(identity,input_id)
                                        result=await executor.execute(tool,arguments,provider_call_id=identity)
                                    if tool=="start_background_task" and result.get("status")=="success":
                                        task_ref=result["task"]["task_ref"]
                                except Exception as failure:
                                    error=type(failure).__name__
                                finally:
                                    elapsed=(time.monotonic()-started)*1000
                                    CURRENT_CALL.reset(token)
                                    load_results = await asyncio.gather(*loads, return_exceptions=True)
                                load_errors = [type(value).__name__ for value in load_results if isinstance(value, BaseException)]
                                profile.calls.append({"call":key,"provider":provider,"ordinal":index,"tool":tool,
                                    "duration_ms":elapsed,"status":result.get("status") if result else None,
                                    "error_code":(result.get("error") or {}).get("code") if result else None,
                                    "error_type":error,"capture_issue":capture.issue,"context_load_errors":load_errors})
                                # Drain between samples; concurrent context builders
                                # are the declared load, not uncontrolled prior work.
                                await capture.flush()
                    actual=await tasks.list(fixture.owner,limit=100)
                    success=len(actual)==2*samples+context_workers and all(call["status"]=="success" and not call["context_load_errors"] for call in profile.calls)
                    fixture.fixture.manifest["profile"]={"accepted":len(actual),"expected":2*samples+context_workers,"all_accepted":success}
                    fixture.fixture.persist()
                finally:
                    for capture in captures:
                        await capture.close()
                    await workers.withdraw(fixture.owner,"fixture-readiness")
    finally:
        by_phase=defaultdict(list)
        for span in profile.spans:
            by_phase[span["phase"]].append(span["duration_ms"])
        by_provider=defaultdict(list)
        by_tool=defaultdict(list)
        for call in profile.calls:
            by_provider[call["provider"]].append(call["duration_ms"])
            by_tool[call["provider"]+":"+call["tool"]].append(call["duration_ms"])
        report={"format":"jarvis.s05.voice_phase_profile.v1","all_accepted":success,"samples_per_provider":samples,
            "simulated_worker_readiness":True,"native_audio":False,"provider_inference":False,
            "history_sources":history_sources,"context_workers":context_workers,
            "load":"Concurrent context assemblies per foreground call when enabled; no synthetic model delay.",
            "whole_tools":{key:distribution(values) for key,values in by_tool.items()},
            "whole_calls":{key:distribution(values) for key,values in by_provider.items()},
            "phases":{key:distribution(values) for key,values in by_phase.items()},
            "calls":profile.calls,"spans":profile.spans,"queries":profile.queries,
            "meaning":"Nested/concurrent phase durations must not be summed. This is not fixed-load native foreground acceptance."}
        (directory/"profile.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({key:report[key] for key in ("all_accepted","whole_calls","whole_tools","phases")}))
    if not success:
        raise RuntimeError("Some declared fixture admissions did not succeed; retained the full profile")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=6)
    parser.add_argument("--history-sources",type=int,default=0)
    parser.add_argument("--context-workers",type=int,default=0)
    args=parser.parse_args()
    asyncio.run(run(args.output,samples=args.samples,history_sources=args.history_sources,context_workers=args.context_workers))


if __name__=="__main__":
    main()
