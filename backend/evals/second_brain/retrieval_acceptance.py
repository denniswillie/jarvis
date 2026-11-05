"""Frozen native S09 retrieval acceptance, not a production inference loop.

Candidate coverage, structural citations and independent semantic grading stay
separate. Expected sources/answers never enter model context or retrieval repair.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import MemorySourceReference
from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, KnowledgeConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.models import TaskCreate, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.source_fixture import source_fixture


def acceptance_limits():
    # Production token ceilings; the isolated runner only extends its unrenewed lease.
    return TaskLimits(lease_seconds=300)


def validate_corpus(corpus):
    if corpus.get("version") != 4 or corpus.get("synthetic") is not True:
        raise ValueError("Only the frozen synthetic S09 corpus is supported")
    keys, identities = set(), set()
    for source in corpus["sources"]:
        if source["key"] in keys or not set(source.get("links", ())) <= keys:
            raise ValueError("Source identities must be unique and links point to prior originals")
        keys.add(source["key"])
        datetime.fromisoformat(source["time"])
        if not source["text"].strip():
            raise ValueError("A source requires actual original text")
    for case in corpus["cases"]:
        refs = set(case["required_candidates"] + case["required_evidence"] + case["forbidden_evidence"]
                   + case["forbidden_candidates"] + case.get("distractor_sources", []))
        if case.get("delete_before"):
            refs.add(case["delete_before"])
        if case["id"] in identities or not refs <= keys or not case["query"].strip():
            raise ValueError("Cases require unique identities, a query and existing oracle sources")
        if set(case["required_evidence"]) & set(case["forbidden_evidence"]):
            raise ValueError("An oracle cannot require and forbid the same source")
        if not set(case["expected_statuses"]) <= {status.value for status in TaskStatus}:
            raise ValueError("Expected states must use the canonical task wire values")
        identities.add(case["id"])
    if not identities:
        raise ValueError("The corpus cannot be empty")


def load_corpus():
    raw = Path(__file__).with_name("s09_extended_cases.json").read_bytes()
    corpus = json.loads(raw)
    validate_corpus(corpus)
    return corpus, hashlib.sha256(raw).hexdigest()


def structural_checks(case, bindings, recalled, payload, status, handoff, *, original_reads=()):
    candidate_ids = {row["source_id"] for row in recalled["sources"]}
    citations = {(item.get("source_id"), item.get("revision"), item.get("source_usage"))
                 for item in payload.get("evidence", ())}
    required = {(str(bindings[key].source_id), bindings[key].revision, "current")
                for key in case["required_evidence"]}
    forbidden = {str(bindings[key].source_id) for key in case["forbidden_evidence"]}
    forbidden_candidates = {str(bindings[key].source_id) for key in case["forbidden_candidates"]}
    visible_originals = {(source["source_id"], source["revision"])
                        for read in original_reads for source in read["sources"]
                        if source["content_complete"] is True}
    tools = {item["name"] for item in payload.get("tool_receipts", ())}
    return {
        "candidate_recall": recalled["status"] == "available" and
            {str(bindings[key].source_id) for key in case["required_candidates"]} <= candidate_ids,
        "no_forbidden_candidates": not (forbidden_candidates & candidate_ids),
        "expected_status": status in case["expected_statuses"],
        "required_original_citations": required <= citations,
        "required_original_contents_seen": {(identity, revision) for identity, revision, usage in required}
                                          <= visible_originals,
        "no_forbidden_originals": not any(identity in forbidden for identity, revision in visible_originals),
        "no_forbidden_citations": not any(item[0] in forbidden for item in citations),
        "required_tools": set(case["required_tools"]) <= tools,
        "uncertainty_disclosed": not case.get("require_uncertainty") or bool(payload.get("uncertainties")),
        "no_mutation_claim": not payload.get("completed_operation_refs"),
        "bounded_old_session_handoff": not case.get("require_omitted_context") or
            (handoff["omitted_count"] > 0 and not handoff["old_session_original_included"]),
    }


async def originals_searchable(sources, owner, area, originals):
    states = [(await sources.read(owner, area, source.source_id)).ingestion_status
              for source in originals]
    return all(state == "searchable" for state in states)


async def publish_corpus(fixture, knowledge, corpus):
    bindings = {}
    async with fixture.pool.acquire() as connection:
        old_session = await connection.fetchval(
            "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
        current_session = await connection.fetchval(
            "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
    old_event = None
    for item in corpus["sources"]:
        source_time = datetime.fromisoformat(item["time"])
        if item.get("origin") == "session":
            event = await knowledge.context_repository.append(fixture.owner, ContextIngress(
                session_id=old_session, source_namespace="voice:s09-old-session",
                source_event_id=item["key"], receipt_id=uuid4().hex,
                kind="user_transcript", trust="user_input", source_time=source_time,
                payload={"text": item["text"]},
            ))
            async with fixture.pool.acquire() as connection:
                identity = await connection.fetchval(
                    "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                    fixture.owner, event.id)
            source = await fixture.sources.read(fixture.owner, fixture.area, identity)
            old_event = event
        else:
            content = {"text": item["text"]}
            if item.get("links"):
                content["related_sources"] = [
                    MemorySourceReference.from_metadata(bindings[key].original_evidence()).model_dump(mode="json")
                    for key in item["links"]
                ]
            source = await fixture.publish_payload(content, source_key=item["key"],
                source_type="project_document", source_time=source_time)
        bindings[item["key"]] = source
    # Canonical current-session noise exercises S03's real character budget.
    # It is not needed in the semantic index and is not a fabricated summary.
    context = ContextRepository(knowledge.context_repository.tasks)
    for index in range(10):
        await context.append(fixture.owner, ContextIngress(
            session_id=current_session, source_namespace="fixture:s09-current-session",
            source_event_id=str(index), receipt_id=uuid4().hex,
            kind="user_transcript", trust="user_input",
            payload={"text": "Synthetic unrelated scheduling note. " * 330},
        ))
    return bindings, current_session, old_event


async def run_worker(case, fixture, knowledge, tasks, session, old_event, *, root, model, reasoning, hermes):
    task = await tasks.create(fixture.owner, TaskCreate(
        request_id=uuid4(), title="S09 synthetic retrieval acceptance",
        requested_outcome=case["query"], session_id=session, knowledge_area_id=fixture.area,
    ), model=model)
    claim = await tasks.claim_next("s09-native-acceptance", ("frontier",))
    if claim is None or claim.task.id != task.id:
        raise RuntimeError("The exact fixture task was not claimed")
    context = TaskRunner(tasks, {}).context(claim)
    handoff, trace, original_reads, verification = {}, [], [], {}
    factory = WorkerContextFactory(fixture.pool, knowledge=knowledge)

    async def prepare(ctx):
        prepared = await factory(ctx)
        content = prepared.manifest.content
        handoff.update({
            "manifest_id": str(prepared.manifest.id),
            "characters": len(json.dumps(content, ensure_ascii=True, sort_keys=True)),
            "omitted_count": content["history"]["omitted_count"],
            "included_count": content["history"]["included"],
            "old_session_original_included": any(row["id"] == str(old_event.id) for row in content["messages"]),
        })
        native_invoke = prepared.tools.invoke
        async def observe_tool(name, arguments, *, provider_call_id):
            result = await native_invoke(name, arguments, provider_call_id=provider_call_id)
            originals = []
            if result.get("status") == "success" and name == "search_memory":
                originals = [{"source_id": row["source_id"], "revision": row["revision"],
                              "content_digest": row["content_digest"],
                              "content_complete": row.get("content_included") is True and "content" in row}
                             for row in result["results"]]
            elif result.get("status") == "success" and name == "explain_memory":
                row = result["source"]
                originals = [{"source_id": row["source_id"], "revision": row["revision"],
                              "content_digest": row["content_digest"],
                              "content_complete": result["content_page"]["complete"]}]
            if originals:
                original_reads.append({"tool": name, "sources": originals})
            return result
        # Observe only after the real bridge's authorization, handler, budget
        # and final task-fence checks. Never replace its response or tools.
        prepared.tools.invoke = observe_tool
        return prepared

    native_recall = knowledge.retrieval.recall
    async def observe_recall(owner, area, query, **options):
        result = await native_recall(owner, area, query, **options)
        trace.append({"query": query, "result": result})
        return result

    provider = HermesTaskProvider(
        FrontierConfiguration(model=model, reasoning_effort=reasoning, api_key=JarvisSettings().OPENAI_API_KEY),
        HermesRuntimeConfiguration(hermes, root / "hermes-state"), prepare=prepare,
    )
    started = time.monotonic()
    from app.tasks.hermes_result import HermesTaskResult
    native_verified = HermesTaskResult.verified

    async def observe_verification(result, ctx, bridge):
        verification["unverified_candidate"] = result.model_dump(mode="json")
        return await native_verified(result, ctx, bridge)

    try:
        with patch.object(knowledge.retrieval, "recall", side_effect=observe_recall), \
                patch.object(HermesTaskResult, "verified", observe_verification):
            outcome = await asyncio.wait_for(provider.execute(context), 240)
        accepted = await tasks.finish(claim, outcome)
        if not accepted:
            raise RuntimeError("The native outcome lost its task claim")
        return {
            "task_id": str(task.id), "status": outcome.status.value, "payload": outcome.payload,
            "handoff": handoff, "worker_retrievals": trace, "original_reads": original_reads,
            "elapsed_seconds": time.monotonic() - started,
            "usage": await context.budgets.usage(fixture.owner, task.id),
        }
    except BaseException as error:
        save(root / (case["id"] + "-rejected.json"), {
            "accepted": False, "task_id": str(task.id), "handoff": handoff,
            "worker_retrievals": trace, "original_reads": original_reads,
            "error": {"type": type(error).__name__, "code": getattr(error, "code", None)},
            **verification,
        })
        await tasks.finish(claim, WorkerOutcome(TaskStatus.FAILED, {"reason": "native_acceptance_failed"}))
        raise


async def exercise(root, service, settings, corpus, corpus_digest, *, hermes, model, reasoning):
    report = {
        "format": "jarvis.s09.native_retrieval.v1", "corpus_sha256": corpus_digest,
        "corpus_version": corpus["version"],
        "model": model, "reasoning": reasoning, "cases": [], "worker_events": [],
        "checks": {}, "semantic_grade": None,
    }
    started = time.monotonic()
    try:
        async with AsyncExitStack() as stack:
            save(root / "corpus.json", corpus)
            fixture = await stack.enter_async_context(source_fixture(
                settings.SUPABASE_DB_URL, root / "fixtures", test_name="S09 extended native retrieval"))
            limits = acceptance_limits()
            report["task_limits"] = limits.model_dump(mode="json")
            tasks = TaskRepository(fixture.pool, limits)
            knowledge = await KnowledgeRuntime.create(
                pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
                configuration=KnowledgeConfiguration(base_url=service["url"], api_key=service["token"],
                                                     area_id=fixture.area, poll_seconds=.2),
            )
            stack.push_async_callback(knowledge.close)
            # Drive the maintained ingestion worker explicitly so the stale-index
            # scenario can deliberately leave a delete queued, not forge recall.
            bindings, session, old_event = await publish_corpus(fixture, knowledge, corpus)
            report["bindings"] = {key: value.original_evidence() for key, value in bindings.items()}
            await drain(fixture, knowledge.worker, report, label="s09-initial")
            report["checks"]["all_originals_indexed"] = await originals_searchable(
                fixture.sources, fixture.owner, fixture.area, bindings.values())
            for case in corpus["cases"]:
                entry = {"id": case["id"], "query": case["query"], "semantic_grade": None}
                report["cases"].append(entry)
                try:
                    if case.get("delete_before"):
                        source = bindings[case["delete_before"]]
                        positive = await knowledge.client.recall(source.bank_name, case["query"])
                        knowledge.client.verify_document(source, await knowledge.client.document(source.bank_name, source.document_id))
                        entry["stale_index_positive_control"] = source.document_id in json.dumps(positive)
                        if not entry["stale_index_positive_control"]:
                            raise RuntimeError("Deleted-source fixture lacked a real indexed positive control")
                        await fixture.sources.delete_source(fixture.owner, fixture.area, source.source_id,
                                                            expected_revision=source.revision)
                        entry["native_document_still_present"] = await knowledge.client.document(
                            source.bank_name, source.document_id) is not None
                        if not entry["native_document_still_present"]:
                            raise RuntimeError("The intended stale native document was already absent")
                    recalled = await knowledge.retrieval.recall(fixture.owner, fixture.area, case["query"])
                    entry["candidates"] = recalled
                    entry.update(await run_worker(case, fixture, knowledge, tasks, session, old_event,
                        root=root, model=model, reasoning=reasoning, hermes=hermes))
                    entry["checks"] = structural_checks(case, bindings, recalled,
                        entry["payload"], entry["status"], entry["handoff"],
                        original_reads=entry["original_reads"])
                    if case.get("delete_before"):
                        entry["checks"]["stale_candidate_rejected"] = recalled["rejected_candidates"] > 0
                except Exception as error:
                    entry["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
                save(root / "retrieval-result.json", report)
                print(json.dumps({"case": case["id"], "checks": entry.get("checks"), "error": entry.get("error")}), flush=True)
            await drain(fixture, knowledge.worker, report, label="s09-reconcile")
            deleted = bindings["deleted-pressure"]
            report["checks"]["native_deleted_document_reconciled"] = await knowledge.client.document(
                deleted.bank_name, deleted.document_id) is None
            # An actual owned-service outage must not become empty memory; local
            # exact source access must remain usable through the shared backend.
            service["stop_owned"]()
            reader = knowledge.memory_read_backend_for(fixture.owner)
            from app.memory.retrieval.query import MemoryRetrievalQuery
            from app.knowledge.models import SourceError
            try:
                await reader.search(owner_id=fixture.owner, request=MemoryRetrievalQuery(query="PrismBridge"))
            except SourceError as error:
                report["checks"]["native_outage_not_empty_memory"] = error.code == "memory_provider_unavailable"
            else:
                report["checks"]["native_outage_not_empty_memory"] = False
            original = await reader.explain(owner_id=fixture.owner, memory_id=bindings["dedup-decision"].source_id)
            report["checks"]["original_read_survives_provider_outage"] = (
                not original["provider_contacted"] and original["source"]["content_digest"] == bindings["dedup-decision"].content_digest)
            report["checks"]["all_cases_structurally_passed"] = all(
                not case.get("error") and case.get("checks") and all(case["checks"].values()) for case in report["cases"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
                           "frames": [{"file": Path(frame.filename).name, "function": frame.name,
                                       "line": frame.lineno} for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(root / "retrieval-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--hermes", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    parser.add_argument("--reasoning", default="medium")
    args = parser.parse_args()
    corpus, corpus_digest = load_corpus()
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY) as service:
        report = asyncio.run(exercise(args.root, service, settings, corpus, corpus_digest,
                                      hermes=args.hermes, model=args.model, reasoning=args.reasoning))
    print(json.dumps({"root": str(args.root), "checks": report["checks"],
                      "error": report.get("error"), "semantic_grade": report["semantic_grade"]}), flush=True)
    return 0 if report["checks"] and all(report["checks"].values()) and not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
