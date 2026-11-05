"""Native memory-pipeline acceptance with manifested, independently cleaned owners.

Requires JARVIS_MEMORY_TEST_DB_URL and an output directory. Stop background
workers first and independently audit the manifest before restoring them.
Synthetic observation text is not evidence of camera, OCR, audio, or glasses
quality. Structural gates supplement, not replace, review of retained content.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any

import asyncpg

from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.presentation import retrieval_payload
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.model_providers.gemini_multimodal_embeddings import (
    GeminiMultimodalEmbeddingModel, gemini_multimodal_embedding_config_from_settings,
)
from app.runtime.concurrency import PriorityAsyncLimiter
from app.settings import JarvisSettings
from evals.glasses_reliability.memory_grounding_probe import (
    CASES, Case, RecordingConsolidationClient, assess_structure,
)
from evals.glasses_reliability.memory_pipeline_runtime import FixtureLedger, PipelineRuntime
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


STATE_CASES = (
    (Case("missing", "Find the blue travel pouch",
          "My blue travel pouch is missing. Please remember that current state; I'll tell you when I find it.",
          "A desk and a chair are visible. No pouch can be identified."), 0),
    (Case("uncertain_match", "Find the blue travel pouch",
          "That blue thing might be my travel pouch, but I cannot tell yet. I haven't found it; keep looking.",
          "A partly hidden blue fabric item is visible. Its identity cannot be established."), 10),
    (Case("found", "Find the blue travel pouch",
          "I've found my blue travel pouch. I've checked the label and contents: this is definitely it. It is no longer missing.",
          "The user holds a blue pouch open and examines its label and contents.", allow_completed=True), 20),
    (Case("late_missing", "Find the blue travel pouch",
          "My blue travel pouch is still missing. I haven't found it yet.",
          "The camera shows the desk. No pouch can be identified."), 5),
    (Case("repeat_found", "Find the blue travel pouch",
          "Just confirming the same update: my blue travel pouch is still found, not missing. Nothing has changed since I found it.",
          "A blue pouch rests on the desk.", allow_completed=True), 25),
)

# The standalone grounding cases have independent context. In this shared-owner
# sequence, make the intended distinct objects explicit rather than grading an
# ambiguous packing/travel-checklist reference as if its identity were known.
GROUNDING_CASES = tuple(
    replace(case, user_text=(
        "Keep helping me pack this bag for a different trip. This is a separate packing "
        "checklist, not the travel checklist we checked earlier. I still need to add the medication."
    )) if case.name == "interrupted_approval" else case
    for case in CASES
)


class PipelineRecordingClient(RecordingConsolidationClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []
        self.native_calls = []

    async def generate(self, request, *, owner_id, validation_feedback=None):
        self.requests.append({
            "event_id": str(request.event.id),
            "prior_facts": list(request.semantic_fact_context),
            "prior_facts_truncated": request.semantic_fact_context_truncated,
            "context_messages": [dict(item.payload) for item in request.context_messages],
        })
        return await super().generate(request, owner_id=owner_id, validation_feedback=validation_feedback)

    async def generate_json(self, **kwargs):
        output = await super().generate_json(**kwargs)
        self.native_calls.append({"schema_name": kwargs["schema_name"], "output": output})
        return output


def json_value(value):
    return json.loads(json.dumps(value, default=str))


async def snapshot(runtime: PipelineRuntime, *, owner: str, base: datetime, query: str, session_id=None, facts_only=False):
    runtime.ledger.require_owner(owner)
    results = {}
    for history in (False, True):
        response = await runtime.retrieval.search(
            owner_id=owner, request=MemoryRetrievalQuery(
                query=query, start_time=base - timedelta(minutes=5), end_time=datetime.now(UTC),
                session_id=session_id, memory_types=("semantic_fact",) if facts_only else (),
                include_superseded=history,
            ),
        )
        results["history" if history else "current"] = retrieval_payload(response)
    async with runtime.pool.acquire() as connection:
        rows = await connection.fetch(
            """select id, memory_type, status, title, summary, content, valid_from, valid_to,
                source_experience_event_id, superseded_by_memory_id, metadata
                from memory_items where owner_id = $1 order by created_at, id""", owner,
        )
    results["memories"] = []
    for row in rows:
        item = dict(row)
        if isinstance(item["metadata"], str):
            item["metadata"] = json.loads(item["metadata"])
        results["memories"].append(json_value(item))
    return results


def assess_state_sequence(runs: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {entry["case"]: entry for entry in runs}
    names = {case.name for case, _ in STATE_CASES}
    if not names <= by_name.keys() or any("snapshot" not in by_name[name] for name in names):
        return {"state_sequence_complete": False}
    missing = by_name["missing"]
    state_facts = [
        item for item in missing["snapshot"]["memories"]
        if item["memory_type"] == "semantic_fact" and item["metadata"].get("temporal_scope") == "state"
    ]
    # The single-state fixture gives a structural oracle without parsing prose
    # or forcing a model-specific predicate/value vocabulary into production.
    if len(state_facts) != 1:
        return {"initial_state_unambiguously_stored": False}
    old = state_facts[0]
    key = (old["metadata"].get("subject_entity_id"), old["metadata"].get("predicate"))

    def current(entry):
        ids = {item["id"] for item in entry["snapshot"]["current"]["results"]}
        return {
            item["id"] for item in entry["snapshot"]["memories"]
            if item["id"] in ids and (item["metadata"].get("subject_entity_id"), item["metadata"].get("predicate")) == key
        }

    found = by_name["found"]
    found_ids = current(found)
    historical_old = [item for item in found["snapshot"]["history"]["results"] if item["id"] == old["id"]]
    transitions = [
        fact for fact in found["consolidation"]["semantic_fact_candidates"]
        if fact["change_intent"] == "transition" and fact["previous_value"] == old["metadata"]["value"]
    ]
    return {
        "initial_state_unambiguously_stored": True,
        "missing_retrievable_as_current": current(missing) == {old["id"]},
        "possible_match_does_not_retire_missing": current(by_name["uncertain_match"]) == {old["id"]},
        "native_confirmed_transition": bool(transitions),
        "found_replaces_missing": len(found_ids) == 1 and old["id"] not in found_ids,
        "missing_retained_as_history": bool(historical_old) and all(
            item["temporal_status"] == "historical" and item["valid_to"] is not None for item in historical_old
        ),
        "late_evidence_does_not_resurrect_missing": bool(found_ids) and current(by_name["late_missing"]) == found_ids,
        "repeat_confirmation_does_not_duplicate_state": bool(found_ids) and current(by_name["repeat_found"]) == found_ids,
    }


def assess_independent_tasks(runs: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {entry["case"]: entry for entry in runs}
    first = by_name.get("supported_completion", {})
    second = by_name.get("interrupted_approval", {})
    if not first.get("snapshot") or not second.get("snapshot"):
        return {"independent_task_sequence_complete": False}
    prior = [
        item for item in first["snapshot"]["memories"]
        if item["source_experience_event_id"] == first["event_id"]
        and item["memory_type"] in {"episode", "semantic_fact"}
    ]
    later = {item["id"]: item for item in second["snapshot"]["memories"]}
    unchanged = bool(prior) and all(
        item["id"] in later and all(
            later[item["id"]].get(field) == item.get(field)
            for field in ("status", "valid_from", "valid_to", "superseded_by_memory_id", "content")
        ) for item in prior
    )
    prior_facts = [item for item in prior if item["memory_type"] == "semantic_fact"]
    conflicting = any(
        item["source_experience_event_id"] == second["event_id"]
        and item["memory_type"] == "semantic_fact"
        and item["metadata"].get("subject_entity_id") == old["metadata"].get("subject_entity_id")
        and item["metadata"].get("predicate") == old["metadata"].get("predicate")
        and item["metadata"].get("value") != old["metadata"].get("value")
        for item in later.values() for old in prior_facts
    )
    return {
        "independent_task_sequence_complete": True,
        "earlier_completion_retained": unchanged,
        "separate_task_does_not_change_prior_entity_state": not conflicting,
    }


async def run(directory: Path) -> bool:
    dsn = os.environ["JARVIS_MEMORY_TEST_DB_URL"]
    # No caller-controlled owner IDs or automatic production-URL fallback.
    ledger = FixtureLedger(directory / "manifests" / "fixture.json")
    settings = JarvisSettings()
    report = {
        "schema": "jarvis.memory_pipeline_probe.v1", "evidence_kind": "synthetic",
        "personal_images_accessed": False, "semantic_review_required": True,
        "owners": sorted(ledger.owners), "runs": [], "jobs": [], "checks": {},
        "reflection_minimum_episodes": settings.MEMORY_REFLECTION_MIN_NEW_EPISODES,
    }
    pool = client = embeddings = None

    def persist():
        (directory / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    persist()
    try:
        client = PipelineRecordingClient(
            api_key=settings.OPENAI_API_KEY or "", model=settings.OPENAI_CONSOLIDATION_MODEL or "",
            timeout_seconds=settings.OPENAI_CONSOLIDATION_TIMEOUT_SECONDS, limiter=PriorityAsyncLimiter(1),
        )
        embeddings = GeminiMultimodalEmbeddingModel(gemini_multimodal_embedding_config_from_settings(settings))
        report["models"] = {"consolidation": client.model, "embedding": embeddings.model}
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, timeout=10, command_timeout=30)
        runtime = PipelineRuntime(pool=pool, settings=settings, client=client, embedding_model=embeddings, ledger=ledger)
        base = datetime.now(UTC) - timedelta(hours=2)
        cases = (*STATE_CASES, *((case, 35 + index * 5) for index, case in enumerate(GROUNDING_CASES)))
        for case, offset in cases:
            client.generations.clear()
            client.requests.clear()
            client.native_calls.clear()
            entry = {"case": case.name, "checks": {}, "passed": False}
            report["runs"].append(entry)
            persist()
            try:
                async with asyncio.timeout(240):
                    event, evidence = await runtime.seed_event(
                        owner=ledger.owner, case=case, ended_at=base + timedelta(minutes=offset),
                    )
                    entry.update(event_id=str(event.id), session_id=str(event.session_id), fixture_evidence=evidence)
                    persist()
                    await runtime.drain(owner=ledger.owner, journal=report["jobs"], persist=persist)
                    completed = await runtime.experience.get_event(
                        owner_id=ledger.owner, session_id=event.session_id, event_id=event.id,
                    )
                    result = ConsolidationResult.model_validate(completed.consolidation_result)
                    entry["consolidation"] = result.model_dump(mode="json")
                    entry["checks"] = assess_structure(case, result, observed_until=completed.ended_at)
                    state_case = case.name in {item.name for item, _ in STATE_CASES}
                    entry["snapshot"] = await snapshot(
                        runtime, owner=ledger.owner, base=base, query="blue travel pouch" if state_case else case.task or "camera view",
                        session_id=None if state_case else event.session_id, facts_only=state_case,
                    )
                    if not state_case:
                        retrieved = entry["snapshot"]["history"]["results"]
                        entry["checks"]["event_or_episode_retrieved"] = any(
                            item["id"] == str(event.id) or item["memory_type"] == "episode" for item in retrieved
                        )
                    promoted = [
                        item for item in entry["snapshot"]["memories"]
                        if item["memory_type"] == "episode" and item["source_experience_event_id"] == str(event.id)
                    ]
                    if promoted and not state_case:
                        current_ids = {item["id"] for item in entry["snapshot"]["current"]["results"]}
                        entry["checks"]["promoted_episode_retrieved_by_default"] = all(item["id"] in current_ids for item in promoted)
                    entry["passed"] = all(entry["checks"].values())
            except Exception as error:
                entry["error_type"] = type(error).__name__
                cause = error.__cause__
                entry["cause_type"] = type(cause).__name__ if cause else None
                # Arbitrary provider exception bodies can include request URLs.
                if type(error).__module__.startswith("asyncpg"):
                    entry["database_error"] = str(error)[:2000]
                break
            finally:
                entry.update(generations=list(client.generations), requests=list(client.requests), native_calls=list(client.native_calls))
                persist()
                print(json.dumps({"case": case.name, "passed": entry["passed"], "error_type": entry.get("error_type")}), flush=True)
        report["checks"].update(assess_state_sequence(report["runs"]))
        report["checks"].update(assess_independent_tasks(report["runs"]))
        report["checks"]["all_cases_completed"] = len(report["runs"]) == len(cases) and all(item["passed"] for item in report["runs"])
        report["checks"]["native_reflection_exercised"] = any(
            job["job_type"] == "generate_reflection" and job["status"] == "completed" for job in report["jobs"]
        )
        isolated = await snapshot(runtime, owner=ledger.other_owner, base=base, query="blue travel pouch")
        report["isolation_retrieval"] = isolated
        report["checks"]["other_owner_has_no_results"] = all(not isolated[key]["results"] for key in ("current", "history"))
        report["profile"] = (await runtime.profile.get_profile(owner_id=ledger.owner)).markdown
        episode_ids = {
            item["id"] for entry in report["runs"]
            for item in entry.get("snapshot", {}).get("memories", [])
            if item["memory_type"] == "episode"
        }
        report["checks"]["profile_contains_recent_episodes"] = bool(episode_ids) and any(
            episode_id in report["profile"] for episode_id in episode_ids
        )
    except Exception as error:
        report["error_type"] = type(error).__name__
    finally:
        try:
            if client is not None:
                await client.close()
            if embeddings is not None:
                await embeddings.close()
            if pool is not None:
                await pool.close()
        finally:
            report["cleanup"] = await cleanup_fixture_owners(dsn, set(ledger.owners))
            ledger.data.update(status="cleaned", cleanup=report["cleanup"])
            ledger.persist()
            persist()
    passed = bool(report["checks"]) and all(report["checks"].values()) and "error_type" not in report
    report["structural_checks_passed"] = passed
    persist()
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output_dir)) else 1)


if __name__ == "__main__":
    main()
