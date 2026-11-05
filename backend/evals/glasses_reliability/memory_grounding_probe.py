"""Native consolidation checks against synthetic, explicitly labelled evidence.

This probe does not access the database, personal images, or audio sessions. It
uses the production consolidation generation/repair path with fixture requests.
Structured checks are not a semantic-entailment grade: inspect the retained
summaries and candidate contents against each case's evidence as well.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.experience.models import ExperienceEvent
from app.memory.consolidation.contracts import (
    ConsolidationEvidenceInput, ConsolidationRequest, ConsolidationResult,
)
from app.memory.consolidation.openai_client import OpenAIStructuredOutputClient
from app.memory.consolidation.service import ExperienceConsolidationService
from app.runtime.concurrency import PriorityAsyncLimiter
from app.settings import JarvisSettings


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    user_text: str
    observation: str
    assistant_text: str = ""
    interrupted: bool = False
    required_state: str | None = None
    allow_completed: bool = False
    scene_only: bool = False


CASES = (
    Case(
        name="paused_not_completed", task="Write a binary search function",
        user_text="I'm stopping for today. I'll finish this function tomorrow.",
        observation="The function is unfinished: neither nonmatching branch updates the search bounds.",
        assistant_text="You have completed the binary search function. Well done.",
        required_state="paused",
    ),
    Case(
        name="supported_completion", task="Finish the travel checklist",
        user_text="Please check whether every entry on my travel checklist is now checked.",
        observation="The complete travel checklist is visible. Every entry has a check mark; none remain unchecked.",
        required_state="completed", allow_completed=True,
    ),
    Case(
        name="interrupted_approval", task="Pack a travel bag",
        user_text="Keep helping me pack this bag. I still need to add the medication.",
        observation="The medication entry on the packing checklist is still unchecked.",
        assistant_text="Everything is packed and the task is complete.",
        interrupted=True,
    ),
    Case(
        name="unreadable_code", task="Check the function's correctness",
        user_text="Can you check whether my function is ready?",
        observation="The editor text is too small and soft to read. Indentation and implementation correctness cannot be assessed.",
        assistant_text="The function is perfect and complete.",
    ),
    Case(
        name="absent_from_frame", task="",
        user_text="", observation="A wall and a door are in view. No person is visible in this camera frame.",
        scene_only=True,
    ),
    Case(
        name="possible_not_confirmed", task="Find the missing merch box",
        user_text="That might be my box, but I am not sure. Please keep looking for possible matches.",
        observation="A flat rectangular item has Eras Tour branding. Its thickness is not visible; it could be a poster or a box.",
    ),
)


def build_request(case: Case, *, now: datetime | None = None) -> ConsolidationRequest:
    ended_at = now or datetime.now(UTC)
    started_at = ended_at - timedelta(minutes=1)
    owner_id = f"memory-grounding-fixture-{uuid4()}"
    event = ExperienceEvent(
        id=uuid4(), owner_id=owner_id, session_id=uuid4(), status="consolidating",
        event_kind="mixed", modalities=("text", "visual"),
        started_at=started_at, ended_at=ended_at, title="", summary="", scene="",
        activity=case.task, state={}, boundary_reason="fixture_event_closed",
        importance=0.7, confidence=0.9, keyframes_finalized_at=ended_at,
        consolidation_ready_at=ended_at, evidence_reduced_at=None,
        retention_expires_at=None, consolidation_result=None,
        consolidation_provider=None, consolidation_model=None, consolidated_at=None,
        created_at=started_at, updated_at=ended_at,
    )
    evidence = []

    def attach(evidence_type: str, payload: dict[str, Any]) -> None:
        evidence.append(ConsolidationEvidenceInput(
            evidence_id=uuid4(), evidence_type=evidence_type,
            evidence_ref_id=uuid4(), occurred_at=started_at + timedelta(seconds=10 * len(evidence)),
            source_sequence=len(evidence) + 1, payload=payload,
        ))

    if case.user_text:
        attach("message", {
            "role": "user", "modality": "text", "content": case.user_text,
            "metadata": {"experience_eligible": True, "completion_state": "finalized"},
        })
    attach("observation", {"summary": case.observation, "confidence": 0.95})
    if case.assistant_text:
        attach("message", {
            "role": "assistant", "modality": "audio", "content": case.assistant_text,
            "metadata": {
                "experience_eligible": not case.interrupted,
                "completion_state": "interrupted" if case.interrupted else "completed",
                "delivery_status": "incomplete" if case.interrupted else "forwarded",
                "playback_status": "unconfirmed",
            },
        })
    return ConsolidationRequest(
        event=event, evidence=tuple(evidence), context_messages=(), keyframes=(),
        event_end_working_state={}, owner_working_memory={}, active_reminders=(),
    )


class FixtureRequestConsolidator(ExperienceConsolidationService):
    """Replace only request loading; keep production generation and validation."""

    def __init__(self, client: OpenAIStructuredOutputClient, request: ConsolidationRequest):
        super().__init__(
            experience_repository=None, perception_repository=None, client=client,
            image_loader=None, owner_working_memory_reader=None, active_reminder_reader=None,
            max_keyframes=3, max_transcripts=24,
        )
        self.request = request

    async def build_request(self, *, owner_id, session_id, event_id):
        event = self.request.event
        if (owner_id, session_id, event_id) != (event.owner_id, event.session_id, event.id):
            raise ValueError("fixture request identity mismatch")
        return self.request


class RecordingConsolidationClient(OpenAIStructuredOutputClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generations: list[dict[str, Any]] = []

    async def generate(self, request, *, owner_id, validation_feedback=None):
        output = await super().generate(
            request, owner_id=owner_id, validation_feedback=validation_feedback,
        )
        self.generations.append({
            "raw_output": output, "validation_feedback": validation_feedback,
        })
        return output


def assess_structure(
    case: Case, result: ConsolidationResult, *, observed_until: datetime | None = None,
) -> dict[str, bool]:
    def bounded_to_observation(value: str | None) -> bool:
        try:
            end = datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
        except ValueError:
            return False
        return (
            end is not None and end.tzinfo is not None
            and end <= (observed_until or datetime.now(UTC))
        )

    states = [item.state for item in result.task_outcomes]
    checks = {"no_unsupported_completed_outcome": case.allow_completed or "completed" not in states}
    if case.required_state is not None:
        checks["required_outcome_present"] = case.required_state in states
    if case.scene_only:
        checks["no_task_inferred_from_passive_view"] = not states
        checks["no_open_ended_scene_fact"] = all(
            bounded_to_observation(item.valid_to_hint) for item in result.semantic_fact_candidates
        )
        checks["no_open_ended_scene_relation"] = all(
            bounded_to_observation(item.valid_to_hint) for item in result.relation_candidates
        )
    return checks


async def run(output: Path, *, repeat: int) -> bool:
    if repeat not in range(1, 4):
        raise ValueError("repeat must be between one and three")
    settings = JarvisSettings()
    client = RecordingConsolidationClient(
        api_key=settings.OPENAI_API_KEY or "",
        model=settings.OPENAI_CONSOLIDATION_MODEL or "",
        timeout_seconds=settings.OPENAI_CONSOLIDATION_TIMEOUT_SECONDS,
        limiter=PriorityAsyncLimiter(1),
    )
    report = {
        "schema": "jarvis.memory_grounding_probe.v1", "evidence_kind": "synthetic",
        "database_accessed": False, "personal_images_accessed": False,
        "provider": client.provider, "model": client.model,
        "semantic_review_required": True, "runs": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repetition in range(1, repeat + 1):
            for case in CASES:
                client.generations.clear()
                request = build_request(case)
                entry = {
                    "case": case.name, "repetition": repetition,
                    "started_at": request.event.started_at.isoformat(),
                    "ended_at": request.event.ended_at.isoformat(),
                    "fixture_evidence": [dict(item.payload) for item in request.evidence],
                }
                try:
                    prepared = await FixtureRequestConsolidator(client, request).consolidate(
                        owner_id=request.event.owner_id, session_id=request.event.session_id,
                        event_id=request.event.id,
                    )
                    checks = assess_structure(
                        case, prepared.result, observed_until=request.event.ended_at,
                    )
                    entry.update(
                        result=prepared.result.model_dump(mode="json"), checks=checks,
                        structural_checks_passed=all(checks.values()),
                    )
                except Exception as error:
                    # Retain the failed case without leaking arbitrary exception bodies.
                    entry.update(error_type=type(error).__name__, structural_checks_passed=False)
                entry["generations"] = list(client.generations)
                report["runs"].append(entry)
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "case": case.name, "repetition": repetition,
                    "structural_checks_passed": entry["structural_checks_passed"],
                    "error_type": entry.get("error_type"),
                }), flush=True)
    finally:
        await client.close()
    return all(item["structural_checks_passed"] for item in report["runs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1, choices=range(1, 4))
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output, repeat=args.repeat)) else 1)


if __name__ == "__main__":
    main()
