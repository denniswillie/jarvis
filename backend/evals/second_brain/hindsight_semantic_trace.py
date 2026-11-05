"""Native frozen-corpus, source-correction and late-retain acceptance.

Answers are recorded for independent semantic review. Neither green structural
checks nor the answer model's confidence are treated as semantic acceptance.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import AsyncExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from app.knowledge.current_context import CurrentSourceContext
from app.knowledge.hindsight import HindsightClient, original_document
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.memory.tools import create_memory_tools
from app.settings import JarvisSettings
from evals.second_brain.hindsight_privacy_trace import operation_tree_with_payload
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import NativeTraceFailure, require_check
from evals.second_brain.memory_answers import NativeAnswerControl
from evals.second_brain.source_fixture import SourceFixture, source_fixture


# Frozen test expectations, never part of model input or production inference.
EXPECTED_SOURCES = {
    "M01": {"coding-01"}, "M02": {"keys-01", "keys-02"}, "M03": {"gym-02"},
    "M04": {"business-01", "coding-01"}, "M05": {"stove-01"}, "M10": {"umbrella-01", "dose-01"},
}
CORRECTIONS = {"gym-02": "gym-01"}
SOURCE_TRUST = {
    "user_report": "user_input", "user_correction": "user_input",
    "user_commitment": "user_input", "camera_observation": "sensor",
    # A gap record or assistant guess is not a positive camera observation.
    "evidence_gap": "assistant_output",
}


def load_corpus():
    raw = Path(__file__).with_name("memory_cases.json").read_bytes()
    corpus = json.loads(raw)
    identities = [source["id"] for source in corpus["sources"]]
    questions = [question["id"] for question in corpus["questions"]]
    if (corpus.get("version") != 1 or corpus.get("synthetic") is not True
            or len(identities) != len(set(identities))
            or len(questions) != len(set(questions))
            or set(questions) != set(EXPECTED_SOURCES)):
        raise ValueError("Invalid frozen synthetic corpus")
    if not set().union(*EXPECTED_SOURCES.values(), CORRECTIONS.values()) <= set(identities):
        raise ValueError("Missing frozen source identity")
    return corpus, hashlib.sha256(raw).hexdigest()


async def publish_case(fixture, item, *, expected_revision=0):
    return await fixture.publish_payload(
        {"fixture_source_id": item["id"], "evidence_kind": item["kind"],
         "text": item["text"], "synthetic": True},
        namespace="frozen-memory-corpus", source_key=CORRECTIONS.get(item["id"], item["id"]),
        source_type=item["kind"], trust=SOURCE_TRUST[item["kind"]],
        source_time=datetime.fromisoformat(item["time"].replace("Z", "+00:00")),
        time_basis="Original event time from the frozen synthetic corpus",
        expected_revision=expected_revision,
    )


def shared_tools(fixture, client):
    retrieval = SourceMemoryRetrieval(fixture.sources, client, timeout_seconds=25)
    backend = SourceMemoryReadBackend(
        fixture.sources, retrieval, owner_id=fixture.owner, area_id=fixture.area,
    )
    return {tool.name: tool for tool in create_memory_tools(owner_id=fixture.owner, read_backend=backend)}


async def run_case(question, tools, bindings, answer_control):
    """Do not fill retrieval gaps from the fixture oracle or source bindings."""
    case = {"id": question["id"], "query": question["query"], "checks": {},
            "semantic_grade": None, "originals": []}
    started = time.monotonic()
    try:
        search = await tools["search_memory"].handler({"query": question["query"]})
        case["search_seconds"] = time.monotonic() - started
        case["search"] = search
        if search["status"] != "success":
            raise NativeTraceFailure("shared_search_failed")
        aliases = set()
        integrity = True
        for result in search["results"]:
            original = await tools["explain_memory"].handler({
                "memory_id": result["memory_id"], "revision": result["revision"],
                "expected_digest": result["content_digest"],
            })
            if original["status"] != "success" or not original["content_page"]["complete"]:
                raise NativeTraceFailure("frozen_original_expansion_incomplete")
            content = json.loads(original["content_page"]["text"])
            alias = content.get("fixture_source_id")
            expected = bindings.get(alias)
            # Bind only to returned originals. These values are for grading and
            # are never used to repair or augment the answer model's evidence.
            integrity = integrity and expected is not None
            if expected is not None:
                integrity = integrity and content == expected.content and all(
                    original["source"][key] == expected.model_dump(mode="json")[key]
                    for key in ("source_id", "revision", "content_digest", "trust", "source_time", "time_basis")
                ) and result["memory_id"] == str(expected.source_id)
            aliases.add(alias)
            case["originals"].append(original)
        case["returned_fixture_sources"] = sorted(value for value in aliases if isinstance(value, str))
        case["checks"]["original_expansion_integrity"] = integrity and bool(case["originals"])
        case["checks"]["required_source_coverage"] = EXPECTED_SOURCES[question["id"]] <= aliases
        if not case["checks"]["original_expansion_integrity"]:
            raise NativeTraceFailure("untrusted_original_expansion")
        case["answer"] = await answer_control.answer(
            question["query"], {"search_memory": search, "explain_memory": case["originals"]},
        )
        allowed = {item["memory_id"] for item in search["results"]}
        case["checks"]["answer_reference_integrity"] = bool(case["answer"]["source_ids"]) and (
            set(case["answer"]["source_ids"]) <= allowed
        )
        case["checks"]["native_answer_recorded"] = True
    except Exception as error:
        case["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    case["elapsed_seconds"] = time.monotonic() - started
    return case


class PausedNativeRetainClient(HindsightClient):
    """Pause one actual request, after Jarvis admission but before native send."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused_source = None
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.retains = Counter()

    async def retain(self, source, operation_id):
        if source.source_id == self.paused_source:
            self.entered.set()
            await self.release.wait()
        self.retains[source.document_id] += 1
        return await super().retain(source, operation_id)


async def queue_rows(fixture):
    async with fixture.pool.acquire() as connection:
        return [dict(row) for row in await connection.fetch(
            "select id,document_id,operation,status,last_error,dispatch_count "
            "from jarvis_memory_outbox where owner_id=$1 order by created_at,id", fixture.owner,
        )]


async def drain(fixture, worker, report, *, label, timeout=300, allowed_failed_ids=()):
    allowed_failed_ids = set(allowed_failed_ids)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await worker.step(fixture.owner, "native-" + label)
        rows = await queue_rows(fixture)
        report["worker_events"].append({"phase": label, "status": status})
        if any(row["status"] == "failed" and row["id"] not in allowed_failed_ids for row in rows):
            raise NativeTraceFailure("native_outbox_failed")
        if rows and all(row["status"] in {"searchable", "cancelled", "done"}
                        or (row["status"] == "failed" and row["id"] in allowed_failed_ids) for row in rows):
            return rows
        await asyncio.sleep(0.25 if status is not None else 0.5)
    raise NativeTraceFailure("native_outbox_deadline")


async def late_retain_case(fixture, client, worker, report):
    source = await fixture.publish("Synthetic late-retain marker " + uuid4().hex, source_key="late-retain")
    client.paused_source = source.source_id
    pending = asyncio.create_task(worker.step(fixture.owner, "late-native-retain"))
    try:
        await asyncio.wait_for(client.entered.wait(), timeout=10)
        deletion = await fixture.sources.delete_source(
            fixture.owner, fixture.area, source.source_id, expected_revision=source.revision,
        )
        require_check(report, "late_retain_local_barrier_precedes_native_send",
                      deletion["read_barrier"] == "deleted" and client.retains[source.document_id] == 0)
        client.release.set()
        status = await pending
        require_check(report, "late_retain_never_becomes_searchable", status in {"cancelled", "claim_lost"})
        rows = await queue_rows(fixture)
        receipt = next(row for row in rows if row["document_id"] == source.document_id and row["operation"] == "retain")
        native = await client.operation(source.bank_name, receipt["id"])
        require_check(report, "late_request_really_reached_native_service",
                      native is not None and native["operation_id"] == str(receipt["id"]))
        report["late_retain"] = {"source_id": str(source.source_id), "document_id": source.document_id,
                                "operation_id": str(receipt["id"]), "native_status": native["status"]}
        await drain(fixture, worker, report, label="late-cleanup")
        require_check(report, "late_native_document_and_operation_removed",
                      await client.document(source.bank_name, source.document_id) is None
                      and await client.operation(source.bank_name, receipt["id"]) is None
                      and client.retains[source.document_id] == 1)
    finally:
        client.release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        client.paused_source = None


async def exercise(root, configuration, settings, corpus, corpus_digest, *, answer_model="gpt-5.4-mini"):
    report = {
        "stage_accepted": False, "synthetic_only": True, "corpus_sha256": corpus_digest,
        "checks": {}, "cases": [], "worker_events": [], "semantic_grade": None,
        "answer_model": answer_model, "answer_track": "source-backed native control, not a voice-provider test",
        "grading": "Independent review against frozen oracles; no oracle is sent to the model.",
    }
    started = time.monotonic()
    try:
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                settings.SUPABASE_DB_URL, root, test_name="MA06-native-frozen-corpus",
            ))
            client = PausedNativeRetainClient(configuration["url"], configuration["token"], timeout=20)
            stack.push_async_callback(client.close)
            answers = NativeAnswerControl(settings.OPENAI_API_KEY, model=answer_model)
            stack.push_async_callback(answers.close)
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            tools = shared_tools(fixture, client)
            bindings = {}
            for item in corpus["sources"]:
                if item["id"] not in CORRECTIONS:
                    bindings[item["id"]] = await publish_case(fixture, item)
            pending = await CurrentSourceContext(fixture.sources, max_sources=16, timeout_seconds=10).describe(
                fixture.owner, fixture.area, include_content=True,
            )
            require_check(report, "pending_originals_visible_without_provider",
                          pending["not_searchable_sources"] == len(bindings)
                          and len(pending["sources"]) == len(bindings) and pending["provider_contacted"] is False)
            initial = await drain(fixture, worker, report, label="initial-ingestion")
            require_check(report, "initial_native_sources_searchable",
                          sum(row["status"] == "searchable" for row in initial) == len(bindings))
            old = bindings["gym-01"]
            old_operation = next(row["id"] for row in initial if row["document_id"] == old.document_id)
            client.verify_document(old, await client.document(old.bank_name, old.document_id))
            old_tree = await operation_tree_with_payload(client, old.bank_name, old_operation)
            require_check(report, "native_old_measurement_positive_control", bool(old_tree) and old_tree[str(old_operation)] is not None)
            correction = next(item for item in corpus["sources"] if item["id"] == "gym-02")
            bindings["gym-02"] = await publish_case(fixture, correction, expected_revision=old.revision)
            try:
                await fixture.sources.resolve_document(fixture.owner, fixture.area, old.document_id)
            except SourceError:
                local_old_denied = True
            else:
                local_old_denied = False
            require_check(report, "correction_immediately_revokes_old_current_source", local_old_denied)
            await drain(fixture, worker, report, label="correction")
            erased_tree = await operation_tree_with_payload(client, old.bank_name, old_operation, known_ids=old_tree)
            require_check(report, "superseded_native_document_and_operations_removed",
                          await client.document(old.bank_name, old.document_id) is None
                          and all(value is None for value in erased_tree.values()))
            historical = await tools["explain_memory"].handler({"memory_id": str(old.source_id), "revision": 1})
            require_check(report, "explicit_original_history_preserved",
                          historical["status"] == "success" and not historical["source"]["is_current"]
                          and json.loads(historical["content_page"]["text"]) == old.content)

            async with fixture.pool.acquire() as connection:
                other_area = await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other fixture','test') returning id",
                    fixture.other,
                )
            other = SourceFixture(fixture.fixture, fixture.pool, fixture.sources, fixture.other, fixture.owner, other_area)
            control = await other.publish(corpus["other_owner_canary"], source_key="owner-control")
            await drain(other, worker, report, label="other-owner")
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            foreign = await tools["explain_memory"].handler({"memory_id": str(control.source_id)})
            scoped = await tools["search_memory"].handler({"query": "What is the private project code?"})
            require_check(report, "native_foreign_owner_control_isolated",
                          foreign["status"] == "failed" and scoped["status"] == "success"
                          and corpus["other_owner_canary"] not in json.dumps(scoped))

            for question in corpus["questions"]:
                report["cases"].append(await run_case(question, tools, bindings, answers))
                save(root / "semantic-result.json", report)
            require_check(report, "all_frozen_cases_recorded", len(report["cases"]) == len(EXPECTED_SOURCES))
            report["checks"]["all_case_structural_checks"] = all(
                "error" not in case and case["checks"] and all(case["checks"].values()) for case in report["cases"]
            )
            await late_retain_case(fixture, client, worker, report)
            report["native_stats_before_deletion"] = await client._call("GET", client._bank(old.bank_name) + "/stats")
            current = {source.source_id: source for source in bindings.values()}
            for source in current.values():
                await fixture.sources.delete_source(
                    fixture.owner, fixture.area, source.source_id, expected_revision=source.revision,
                )
            await drain(fixture, worker, report, label="owner-a-erasure")
            after = await client.recall(old.bank_name, "Recall Alex's coding, keys, gym, stove and reminder evidence.")
            report["native_stats_after_deletion"] = await client._call("GET", client._bank(old.bank_name) + "/stats")
            require_check(report, "native_owner_a_recall_empty_after_erasure", not after["results"])
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "unrelated_native_owner_survives_erasure", True)
            await other.sources.delete_source(other.owner, other.area, control.source_id, expected_revision=1)
            await drain(other, worker, report, label="owner-b-erasure")
            report["outbox_receipts"] = [
                {**row, "id": str(row["id"])} for row in [*await queue_rows(fixture), *await queue_rows(other)]
            ]
            require_check(report, "no_duplicate_native_retains", all(count == 1 for count in client.retains.values()))
            report["provider_transport"] = client.diagnostics()
            report["scope_limits"] = [
                "Manual semantic grading required; a recorded answer is not a passed answer.",
                "No physical glasses or audio-provider memory-answer acceptance.",
                "Not a complete log, backup, restart or all-derivative privacy audit.",
                "No real-memory backfill, live routing or migration activation.",
            ]
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(root / "semantic-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.OZIsEM/hindsight-env"))
    parser.add_argument("--answer-model", default="gpt-5.4-mini")
    args = parser.parse_args()
    corpus, digest = load_corpus()
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY) as configuration:
        report = asyncio.run(exercise(args.root, configuration, settings, corpus, digest, answer_model=args.answer_model))
    print(json.dumps({"root": str(args.root), "checks": report["checks"], "error": report.get("error"),
                      "case_count": len(report["cases"]), "semantic_grade": report["semantic_grade"]}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
