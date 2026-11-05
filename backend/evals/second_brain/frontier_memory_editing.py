"""Read-only frontier judgment of frozen memory-editing failures.

This is not delegated mutation acceptance. Original requests and prior ASR
transcripts are synthetic inputs; native Hermes, Sol, Hindsight, source reads,
task authority, budgets and result verification remain real.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
import traceback
from typing import Literal
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, KnowledgeConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.models import TaskCreate, TaskStatus, TaskVoiceSource, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.native_memory_lifecycle import load_cases, snapshot
from evals.second_brain.source_fixture import source_fixture


ASR_REQUESTS = {
    "precise_correction": {
        "text": "Please correct the stored record for robot Cedar version one. The firmware number was wrong. "
                "It should say 2.4, not 2.3. Keep its motors, tolerance, and unfinished calibration status unchanged. "
                "Do not change the record for version two",
        "source": "/tmp/jarvis-m02-native.s98-final-1a0b007f1a8/precise_correction-1/result.json",
    },
    "ambiguous_correction": {
        "text": "I have to see the prototypes. Please correct the firmware number we stored for Cedar to 2.4.",
        "source": "/tmp/jarvis-m02-native.s98-lifecycle-1a0aff54e43/ambiguous_correction-1/result.json",
    },
}


class ProposedEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str
    revision: int = Field(ge=1)
    replacement: str = Field(min_length=1)


class EditingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["correct", "clarify"]
    edits: list[ProposedEdit] = Field(max_length=32)
    question: str | None
    explanation: str = Field(min_length=1)


# Derive the output contract from the actual oracle schema. No case-specific
# action, source identity or expected replacement is included in this instruction.
REHEARSAL_INSTRUCTIONS = (
    "For my next memory-change request, do a read-only rehearsal. Use stored originals to work out "
    "what should change, but do not perform changes or claim that they happened. "
    "If you need clarification, ask it instead of choosing for me. "
    "In your normal structured task result, make the written answer a JSON object matching "
    "the following schema, without Markdown fences. Use an empty edits list if clarification "
    "is needed. Keep the speech conversational, including any question and the fact that "
    "nothing was changed. Written-answer JSON schema: "
    + json.dumps(EditingProposal.model_json_schema(), sort_keys=True)
)


def requests(cases):
    """Same immutable population, including failures; no pass-until-lucky loop."""
    by_id = {case["id"]: case for case in cases}
    return [
        {"id": key + "-" + variant, "case": by_id[key], "variant": variant,
         "utterance": by_id[key]["utterance"] if variant == "original" else ASR_REQUESTS[key]["text"],
         "origin": "frozen_spoken_fixture" if variant == "original" else ASR_REQUESTS[key]["source"]}
        for key in ASR_REQUESTS for variant in ("original", "recorded_asr")
    ]


def proposal_checks(case, bindings, answer, status):
    """An exact synthetic oracle, never application inference or an LLM grader."""
    checks = {"proposal_schema": False, "intended_action": False,
              "exact_target_and_text": False, "appropriate_task_status": False}
    try:
        proposal = EditingProposal.model_validate(json.loads(answer))
    except (ValueError, TypeError, ValidationError):
        return checks, None
    checks["proposal_schema"] = True
    checks["intended_action"] = proposal.action == case["expected"]["action"]
    if case["expected"]["action"] == "clarify":
        checks["exact_target_and_text"] = not proposal.edits and bool((proposal.question or "").strip())
        checks["appropriate_task_status"] = status == "waiting_for_input"
    else:
        # The frozen correction changes this one literal fact only. Exact full
        # text comparison catches unrelated spelling, punctuation or fact edits.
        originals = {source["id"]: source["text"] for source in case["sources"]}
        expected = [{
            "source_id": str(bindings[key]["source_id"]), "revision": bindings[key]["revision"],
            "replacement": originals[key].replace("firmware 2.3", "firmware 2.4", 1),
        } for key in case["expected"]["targets"]]
        checks["exact_target_and_text"] = (
            [edit.model_dump() for edit in proposal.edits] == expected and proposal.question is None)
        checks["appropriate_task_status"] = status == "succeeded"
    return checks, proposal.model_dump()


def observed_originals(tools):
    sources = set()
    for tool in tools:
        result = tool["result"]
        if result.get("status") != "success":
            continue
        if tool["name"] == "search_memory":
            for row in result["results"]:
                if row.get("content_included") is True and "content" in row:
                    sources.add((row["source_id"], row["revision"]))
        elif tool["name"] == "explain_memory" and result["content_page"]["complete"]:
            row = result["source"]
            sources.add((row["source_id"], row["revision"]))
    return sources


async def run_case(root, entry, service, settings, hermes):
    output = root / entry["id"]
    output.mkdir(mode=0o700)
    case = entry["case"]
    report = {"id": entry["id"], "variant": entry["variant"], "utterance": entry["utterance"],
        "origin": entry["origin"], "model": DEFAULT_FRONTIER_MODEL, "reasoning": "medium",
        "tools": [], "worker_events": [], "checks": {}, "structural_pass": False,
        "semantic_review": "pending", "scope": "read_only_proposal_not_mutation_acceptance"}
    started = time.monotonic()
    try:
        async with source_fixture(service["source_database_url"], output / "fixtures",
                                  test_name="S11 frontier editing " + entry["id"]) as fixture:
            async with AsyncExitStack() as stack:
                tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
                knowledge = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
                    configuration=KnowledgeConfiguration(base_url=service["url"], api_key=service["token"],
                                                         area_id=fixture.area))
                stack.push_async_callback(knowledge.close)
                bindings = {source["id"]: await fixture.publish(source["text"],
                    namespace="fixture:editing-judgment", source_key=source["id"]) for source in case["sources"]}
                await drain(fixture, knowledge.worker, report, label="editing-judgment-seed")
                for source in bindings.values():
                    knowledge.client.verify_document(source, await knowledge.client.document(
                        source.bank_name, source.document_id))
                report["source_bindings"] = {key: source.original_evidence() for key, source in bindings.items()}
                report["before"] = await snapshot(fixture, bindings)
                report["positive_controls"] = True
                async with fixture.pool.acquire() as connection:
                    session = await connection.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                for identity, text in (("read-only-instructions", REHEARSAL_INSTRUCTIONS),
                                       ("current-request", entry["utterance"])):
                    request = await knowledge.context_repository.append(fixture.owner, ContextIngress(
                        session_id=session, source_namespace="voice:s11-frontier-rehearsal",
                        source_event_id=identity, receipt_id=uuid4().hex,
                        kind="user_transcript", trust="user_input",
                        source_time=datetime(2026, 9, 17, 12, tzinfo=UTC), payload={"text": text}))
                task = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(),
                    title="Read-only memory-change rehearsal",
                    requested_outcome="Assess the latest memory-change request under the preceding read-only instructions.",
                    session_id=session, knowledge_area_id=fixture.area), model=DEFAULT_FRONTIER_MODEL,
                    source=TaskVoiceSource(request.id, request.revision, session), mode="background")
                claim = await tasks.claim_next("s11-frontier-rehearsal", ("frontier",))
                if claim is None or claim.task.id != task.id:
                    raise RuntimeError("The exact synthetic task was not claimed")
                context = TaskRunner(tasks, {}).context(claim)
                factory = WorkerContextFactory(fixture.pool, knowledge=knowledge)

                async def prepare(ctx):
                    prepared = await factory(ctx)
                    report["manifest"] = prepared.manifest.content
                    report["catalog"] = prepared.tools.catalog()
                    report["authorized_operations"] = prepared.tools.authorized_operations()
                    if report["authorized_operations"]:
                        raise RuntimeError("A read-only rehearsal unexpectedly received write authority")
                    if any(prepared.tools.registry.get(tool["name"]).mutates_state
                           for tool in report["catalog"]):
                        raise RuntimeError("A mutation tool entered the read-only rehearsal")
                    native_invoke = prepared.tools.invoke

                    async def observe(name, arguments, *, provider_call_id):
                        result = await native_invoke(name, arguments, provider_call_id=provider_call_id)
                        report["tools"].append({"name": name, "arguments": arguments, "result": result})
                        return result
                    prepared.tools.invoke = observe
                    return prepared

                provider = HermesTaskProvider(FrontierConfiguration(model=DEFAULT_FRONTIER_MODEL,
                    reasoning_effort="medium", api_key=settings.OPENAI_API_KEY),
                    HermesRuntimeConfiguration(hermes, output / "hermes-state"), prepare=prepare)
                worker_started = time.monotonic()
                try:
                    outcome = await asyncio.wait_for(provider.execute(context), 240)
                    if not await tasks.finish(claim, outcome):
                        raise RuntimeError("The native outcome lost its task claim")
                except BaseException:
                    await tasks.finish(claim, WorkerOutcome(TaskStatus.FAILED,
                        {"reason": "read_only_editing_experiment_failed"}))
                    raise
                report["worker_seconds"] = time.monotonic() - worker_started
                report["outcome"] = {"status": outcome.status.value, "payload": outcome.payload}
                report["usage"] = await context.budgets.usage(fixture.owner, task.id)
                report["after"] = await snapshot(fixture, bindings)
                checks, proposal = proposal_checks(case, report["source_bindings"],
                    outcome.payload.get("answer"), outcome.status.value)
                needed = case["expected"]["unchanged"] if case["expected"]["action"] == "clarify" else case["expected"]["targets"]
                expected_originals = {(str(bindings[key].source_id), bindings[key].revision) for key in needed}
                checks.update({
                    "stored_originals_unchanged": report["before"] == report["after"],
                    "no_operation_claim": not outcome.payload.get("completed_operation_refs"),
                    "actual_originals_seen": expected_originals <= observed_originals(report["tools"]),
                    "source_evidence_cited": expected_originals <= {
                        (row.get("source_id"), row.get("revision"))
                        for row in outcome.payload.get("evidence", [])},
                    "native_read_only_authority": report["authorized_operations"] == [],
                })
                report.update(checks=checks, proposal=proposal, structural_pass=all(checks.values()))
                report["transport"] = knowledge.client.diagnostics()
            report["runtime_closed"] = True
        report["fixture_cleanup_returned"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
            "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)]}
        report["structural_pass"] = False
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(output / "result.json", report)
    print(json.dumps({key: report.get(key) for key in
        ("id", "structural_pass", "checks", "error", "worker_seconds", "elapsed_seconds")}), flush=True)
    return report


async def run(root, entries, digest, service, settings, hermes):
    connection = await asyncpg.connect(service["source_database_url"], timeout=10)
    try:
        await connection.execute("create extension if not exists pgcrypto with schema public; "
                                 "create extension if not exists vector with schema public")
    finally:
        await connection.close()
    results = []
    try:
        for entry in entries:
            results.append(await run_case(root, entry, service, settings, hermes))
    finally:
        save(root / "matrix.json", {"format": "jarvis.s11.frontier_editing_proposals.v1",
            "frozen_lifecycle_corpus_sha256": digest, "results": results,
            "semantic_review": "pending", "independent_cleanup": "required",
            "mutation_integration_tested": False, "physical_playback_tested": False,
            "full_stage_accepted": False})
    return 0 if len(results) == len(entries) and all(row["structural_pass"] for row in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.V4dY8h"))
    args = parser.parse_args()
    cases, digest = load_cases()
    logging.disable(logging.CRITICAL)
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime / "hindsight-env",
                           api_key=settings.OPENAI_API_KEY, max_starts=1) as service:
        return asyncio.run(run(args.root, requests(cases), digest, service, settings,
                               args.runtime / "hermes-env/bin/python"))


if __name__ == "__main__":
    raise SystemExit(main())
