from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.experience.models import ExperienceEvent, ExperienceEventEvidence
from app.experience.repository import ExperienceEventMissingError
from app.memory.consolidation.contracts import ConsolidationValidationError
from app.memory.consolidation.openai_client import ConsolidationProviderTransientError
from app.memory.consolidation.service import (
    ExperienceConsolidationService, _build_attached_evidence,
)
from test_consolidation_contracts import _cited_output, _output, _request


class _Client:
    provider = "openai"
    model = "test-gpt"

    def __init__(self) -> None:
        self.request = None

    async def generate(
        self,
        request,
        *,
        owner_id: str,
        validation_feedback: str | None = None,
    ) -> str:  # type: ignore[no-untyped-def]
        self.request = request
        return json.dumps(_output(request))

    async def close(self) -> None:
        return None


class ConsolidationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_failure_gets_one_grounding_retry(self) -> None:
        request = _request()
        corrected = json.dumps(_output(request))
        client = AsyncMock()
        client.provider = "openai"
        client.model = "test-gpt"
        client.generate.side_effect = ["{}", corrected]
        service = object.__new__(ExperienceConsolidationService)
        service._client = client
        service.build_request = AsyncMock(return_value=request)  # type: ignore[method-assign]

        prepared = await service.consolidate(
            owner_id=request.event.owner_id,
            session_id=request.event.session_id,
            event_id=request.event.id,
        )

        self.assertEqual(prepared.result.event_title, "Picked up the keys")
        self.assertEqual(client.generate.await_count, 2)
        feedback = client.generate.await_args_list[1].kwargs["validation_feedback"]
        self.assertIn("copied verbatim", feedback)

    async def test_audio_visual_and_tool_evidence_are_bounded_and_ordered(self) -> None:
        now = datetime.now(UTC)
        event = _event(now)
        message_id = uuid4()
        tool_id = uuid4()
        evidence = [
            ExperienceEventEvidence(
                id=uuid4(),
                event_id=event.id,
                evidence_type="message",
                evidence_ref_id=message_id,
                occurred_at=now,
                source_sequence=1,
                ingested_at=now,
                metadata={},
            ),
            ExperienceEventEvidence(
                id=uuid4(),
                event_id=event.id,
                evidence_type="tool_execution",
                evidence_ref_id=tool_id,
                occurred_at=now,
                source_sequence=2,
                ingested_at=now,
                metadata={},
            ),
        ]
        repository = AsyncMock()
        repository.get_event.return_value = event
        repository.list_event_evidence.return_value = evidence
        repository.list_nonvisual_source_payloads.return_value = {
            ("message", message_id): {
                "role": "user",
                "modality": "audio",
                "content": "I keep my silver keys on the kitchen counter",
                "metadata": {},
            },
            ("tool_execution", tool_id): {
                "tool_name": "create_note",
                "arguments": {"title": "Keys"},
                "result": {"ok": True},
                "status": "completed",
                "error": None,
            },
        }
        repository.list_context_messages.return_value = []
        frame_id = uuid4()
        repository.list_event_keyframe_sources.return_value = [
            {
                "visual_frame_id": frame_id,
                "rank": 1,
                "reason": "keys visible",
                "quality_score": 0.9,
                "captured_at": now,
                "frame_description": "silver keys",
                "storage_bucket": "frames",
                "storage_path": "one.jpg",
                "content_type": "image/jpeg",
                "raw_image_status": "available",
            }
        ]
        perception = AsyncMock()
        perception.list_observations_by_ids.return_value = []
        loader = AsyncMock()
        loader.download_object.return_value = b"\xff\xd8\xff\x00\xff\xd9"
        client = _Client()
        owner_working_memory_reader = AsyncMock()
        owner_working_memory_reader.get_owner_working_memory.return_value = (
            SimpleNamespace(state={"active_visual_focus": "find the keys"})
        )
        active_reminder_reader = AsyncMock()
        active_reminder_reader.list_reminders.return_value = [
            SimpleNamespace(
                id=uuid4(),
                title="Take the keys",
                details="Bring the keys when leaving",
                remind_at=None,
                condition="when leaving home",
                updated_at=now,
            )
        ]
        service = ExperienceConsolidationService(
            experience_repository=repository,
            perception_repository=perception,
            client=client,  # type: ignore[arg-type]
            image_loader=loader,
            owner_working_memory_reader=owner_working_memory_reader,
            active_reminder_reader=active_reminder_reader,
            max_keyframes=1,
            max_transcripts=1,
        )

        prepared = await service.consolidate(
            owner_id=event.owner_id,
            session_id=event.session_id,
            event_id=event.id,
        )

        self.assertEqual(prepared.result.event_title, "Picked up the keys")
        self.assertEqual(len(client.request.evidence), 1)
        self.assertEqual(len(client.request.context_messages), 0)
        self.assertEqual(client.request.evidence[0].evidence_type, "message")
        self.assertEqual(client.request.keyframes[0].image_bytes, loader.download_object.return_value)
        self.assertEqual(
            client.request.owner_working_memory["active_visual_focus"],
            "find the keys",
        )
        self.assertEqual(client.request.active_reminders[0]["title"], "Take the keys")
        repository.list_context_messages.assert_not_awaited()


    async def test_reference_and_literal_quote_failures_can_be_corrected_once(self):
        for failure in ("invalid_reference", "missing_literal_quotation"):
            with self.subTest(failure=failure):
                request = _request()
                bad = _cited_output(
                    request,
                    reference_id=request.evidence[0].evidence_id
                    if failure == "invalid_reference" else None,
                    text="The keys are on a counter"
                    if failure == "missing_literal_quotation" else None,
                )
                good = _cited_output(request)
                service = _service_with_responses(request, json.dumps(bad), json.dumps(good))
                prepared = await service.consolidate(
                    owner_id=request.event.owner_id, session_id=request.event.session_id,
                    event_id=request.event.id,
                )
                self.assertEqual(service._client.generate.await_count, 2)
                first, second = service._client.generate.await_args_list
                self.assertIs(first.args[0], second.args[0])
                feedback = second.kwargs["validation_feedback"]
                self.assertIn(failure, feedback)
                self.assertIn("entity_candidates[0].evidence[0]", feedback)
                self.assertIn(str(request.evidence[0].evidence_ref_id), feedback)
                self.assertEqual(
                    prepared.result.entity_candidates[0].evidence[0].evidence_ref_id,
                    str(request.evidence[0].evidence_ref_id),
                )

    async def test_two_invalid_results_fail_closed_with_safe_attempt_diagnostics(self):
        request = _request()
        bad = json.dumps(_cited_output(request, text="private model hallucination"))
        service = _service_with_responses(request, bad, bad)
        with self.assertRaises(ConsolidationValidationError) as caught:
            await service.consolidate(
                owner_id=request.event.owner_id, session_id=request.event.session_id,
                event_id=request.event.id,
            )
        self.assertEqual(service._client.generate.await_count, 2)
        self.assertEqual(caught.exception.model_attempt, 2)
        diagnostic = caught.exception.safe_diagnostic()
        self.assertIn("missing_literal_quotation", diagnostic)
        self.assertIn("model_generation=2", diagnostic)
        self.assertNotIn("private model hallucination", diagnostic)

    async def test_corrected_quote_does_not_bypass_other_candidate_validation(self):
        for failure in ("confidence", "keyframe", "attribution"):
            with self.subTest(failure=failure):
                request = _request(message_role="assistant" if failure == "attribution" else "user")
                first = _cited_output(request, text="unsupported")
                second = _output(request)
                if failure == "confidence":
                    second["confidence"] = 1.1
                elif failure == "keyframe":
                    second["selected_keyframe_ids"] = [str(uuid4())]
                service = _service_with_responses(
                    request, json.dumps(first), json.dumps(second),
                )
                with self.assertRaises(ConsolidationValidationError) as caught:
                    await service.consolidate(
                        owner_id=request.event.owner_id, session_id=request.event.session_id,
                        event_id=request.event.id,
                    )
                self.assertEqual(caught.exception.model_attempt, 2)
                self.assertEqual(service._client.generate.await_count, 2)

    async def test_transient_failures_and_cancellation_are_not_grounding_retries(self):
        for failure in (ConsolidationProviderTransientError("busy"), asyncio.CancelledError()):
            for during_repair in (False, True):
                with self.subTest(failure=type(failure).__name__, during_repair=during_repair):
                    request = _request()
                    responses = (
                        (json.dumps(_cited_output(request, text="unsupported")), failure)
                        if during_repair else (failure,)
                    )
                    service = _service_with_responses(request, *responses)
                    with self.assertRaises(type(failure)):
                        await service.consolidate(
                            owner_id=request.event.owner_id, session_id=request.event.session_id,
                            event_id=request.event.id,
                        )
                    self.assertEqual(service._client.generate.await_count, len(responses))

    async def test_request_rejects_missing_foreign_owner_session_or_event(self):
        event = _request().event
        for returned in (
            None, replace(event, owner_id="foreign-owner"),
            replace(event, session_id=uuid4()), replace(event, id=uuid4()),
        ):
            with self.subTest(event=returned):
                service = object.__new__(ExperienceConsolidationService)
                service._experience_repository = AsyncMock()
                service._experience_repository.get_event.return_value = returned
                with self.assertRaises(ExperienceEventMissingError):
                    await service.build_request(
                        owner_id=event.owner_id, session_id=event.session_id, event_id=event.id,
                    )
                service._experience_repository.get_event.assert_awaited_once_with(
                    owner_id=event.owner_id, session_id=event.session_id, event_id=event.id,
                )

    def test_missing_or_foreign_attachment_sources_fail_closed(self):
        event = _request().event
        reference_id = uuid4()
        evidence = ExperienceEventEvidence(
            id=uuid4(), event_id=event.id, evidence_type="observation",
            evidence_ref_id=reference_id, occurred_at=event.started_at,
            source_sequence=1, ingested_at=event.started_at, metadata={},
        )
        for observations in (
            {},
            {reference_id: SimpleNamespace(owner_id="foreign", session_id=event.session_id)},
            {reference_id: SimpleNamespace(owner_id=event.owner_id, session_id=uuid4())},
        ):
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                _build_attached_evidence(
                    [evidence], event=event, observation_by_id=observations,
                    source_payloads={}, max_transcripts=5,
                )
        for link in (
            replace(evidence, event_id=uuid4()),
            replace(evidence, evidence_type="message"),
            replace(evidence, evidence_type="tool_execution"),
        ):
            with self.subTest(link=link), self.assertRaises(ValueError):
                _build_attached_evidence(
                    [link], event=event, observation_by_id={},
                    source_payloads={}, max_transcripts=5,
                )

    async def test_repaired_result_keeps_promotion_and_retention_policy(self):
        from app.experience.models import ExperienceEventKeyframe
        from app.memory.episodic.service import EpisodePromotionContext, EpisodePromotionService

        policy = EpisodePromotionService(
            minimum_importance=0.55, important_keyframe_minimum_importance=0.8,
        )
        for importance, expected in ((0.4, False), (0.7, True)):
            with self.subTest(importance=importance):
                request = _request()
                invalid = _cited_output(request, reference_id=request.evidence[0].evidence_id)
                corrected = _cited_output(request)
                corrected["episode_importance"] = importance
                service = _service_with_responses(
                    request, json.dumps(invalid), json.dumps(corrected),
                )
                prepared = await service.consolidate(
                    owner_id=request.event.owner_id, session_id=request.event.session_id,
                    event_id=request.event.id,
                )
                event = replace(
                    request.event, status="consolidated",
                    title=prepared.result.event_title,
                    summary=prepared.result.event_summary,
                    importance=prepared.result.episode_importance,
                    confidence=prepared.result.confidence,
                    consolidation_result=prepared.result.model_dump(mode="json"),
                    consolidation_provider=prepared.provider,
                    consolidation_model=prepared.model,
                    consolidated_at=datetime.now(UTC),
                )
                frames = tuple(ExperienceEventKeyframe(
                    event_id=event.id, visual_frame_id=frame.visual_frame_id,
                    rank=frame.rank, reason=frame.reason, quality_score=frame.quality_score,
                    selected_by="deterministic", created_at=frame.captured_at,
                ) for frame in request.keyframes)
                context = EpisodePromotionContext(
                    event=event, result=prepared.result, evidence=(), source_payloads={},
                    context_messages=(), provisional_keyframes=frames,
                    keyframe_capture_times={
                        frame.visual_frame_id: frame.captured_at for frame in request.keyframes
                    },
                    has_explicit_memory_evidence=False, reminder_deliveries=(),
                )
                plan = policy.plan(context)
                self.assertEqual(plan.promote, expected)
                self.assertFalse(plan.is_explicit)
                self.assertFalse(plan.retain_keyframes_permanently)
                self.assertEqual(
                    prepared.result.entity_candidates[0].evidence[0].evidence_ref_id,
                    str(request.evidence[0].evidence_ref_id),
                )


def _event(now: datetime) -> ExperienceEvent:
    return ExperienceEvent(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        status="consolidating",
        event_kind="mixed",
        modalities=("audio", "visual", "tool"),
        started_at=now,
        ended_at=now,
        title="",
        summary="",
        scene="kitchen",
        activity="handling keys",
        state={"scene": "kitchen"},
        boundary_reason="idle_timeout",
        importance=0.5,
        confidence=0.8,
        keyframes_finalized_at=now,
        consolidation_ready_at=now,
        evidence_reduced_at=None,
        retention_expires_at=None,
        consolidation_result=None,
        consolidation_provider=None,
        consolidation_model=None,
        consolidated_at=None,
        created_at=now,
        updated_at=now,
    )


def _service_with_responses(request, *responses):
    service = object.__new__(ExperienceConsolidationService)
    service._client = AsyncMock()
    service._client.provider = "openai"
    service._client.model = "test-gpt"
    service._client.generate.side_effect = list(responses)
    service.build_request = AsyncMock(return_value=request)
    return service
