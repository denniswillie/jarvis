from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.experience.models import ExperienceEvent
from app.memory.consolidation.contracts import (
    ConsolidationEvidenceInput,
    ConsolidationKeyframeInput,
    ConsolidationRequest,
    ConsolidationValidationError,
    parse_consolidation_result,
    consolidation_result_payload,
)


class ConsolidationContractTests(unittest.TestCase):
    def test_valid_grounded_output_is_accepted(self) -> None:
        request = _request()
        result = parse_consolidation_result(_output(request), request=request)

        self.assertEqual(result.event_title, "Picked up the keys")
        self.assertTrue(result.semantic_fact_candidates[0].explicitly_user_stated)

    def test_citation_must_belong_to_attached_event_evidence(self) -> None:
        request = _request()
        payload = _output(request)
        payload["entity_candidates"][0]["evidence"][0]["evidence_ref_id"] = str(
            uuid4()
        )

        with self.assertRaisesRegex(
            ConsolidationValidationError,
            "attached event evidence",
        ):
            parse_consolidation_result(payload, request=request)

    def test_citation_must_quote_source_text(self) -> None:
        request = _request()
        payload = _output(request)
        payload["entity_candidates"][0]["evidence"][0]["supporting_text"] = "not present"

        with self.assertRaisesRegex(ConsolidationValidationError, "not present"):
            parse_consolidation_result(payload, request=request)

    def test_explicit_fact_requires_user_message(self) -> None:
        request = _request(message_role="assistant")

        with self.assertRaisesRegex(ConsolidationValidationError, "user message"):
            parse_consolidation_result(_output(request), request=request)

    def test_selected_keyframe_must_belong_to_event(self) -> None:
        request = _request()
        payload = _output(request)
        payload["selected_keyframe_ids"] = [str(uuid4())]

        with self.assertRaisesRegex(ConsolidationValidationError, "does not belong"):
            parse_consolidation_result(payload, request=request)

    def test_unsupported_predicate_is_rejected(self) -> None:
        request = _request()
        payload = _output(request)
        payload["semantic_fact_candidates"][0]["predicate"] = "secretly_intends"

        with self.assertRaises(ConsolidationValidationError):
            parse_consolidation_result(payload, request=request)

    def test_arbitrary_mutation_fields_are_rejected(self) -> None:
        request = _request()
        payload = _output(request)
        payload["delete_previous_memory"] = True

        with self.assertRaises(ConsolidationValidationError):
            parse_consolidation_result(payload, request=request)

    def test_retraction_must_be_explicit(self) -> None:
        request = _request()
        payload = _output(request)
        fact = payload["semantic_fact_candidates"][0]
        fact["change_intent"] = "retract"
        fact["explicitly_user_stated"] = False

        with self.assertRaises(ConsolidationValidationError):
            parse_consolidation_result(payload, request=request)

    def test_multi_value_replacement_requires_previous_value(self) -> None:
        request = _request()
        payload = _output(request)
        fact = payload["semantic_fact_candidates"][0]
        fact["change_intent"] = "replace"

        with self.assertRaises(ConsolidationValidationError):
            parse_consolidation_result(payload, request=request)

    def test_retraction_rejects_previous_value(self) -> None:
        request = _request()
        payload = _output(request)
        fact = payload["semantic_fact_candidates"][0]
        fact["change_intent"] = "retract"
        fact["previous_value"] = "bronze keys"

        with self.assertRaises(ConsolidationValidationError):
            parse_consolidation_result(payload, request=request)


    def test_event_row_keyframe_and_other_event_ids_are_not_source_references(self):
        request = _request()
        for reference_id in (
            request.evidence[0].evidence_id,
            request.keyframes[0].visual_frame_id,
            _request().evidence[0].evidence_ref_id,
        ):
            with self.subTest(reference=reference_id):
                payload = _cited_output(request, reference_id=reference_id)
                with self.assertRaises(ConsolidationValidationError) as caught:
                    parse_consolidation_result(payload, request=request)
                self.assertEqual(caught.exception.code, "invalid_reference")
                self.assertEqual(caught.exception.location, "entity_candidates[0].evidence[0]")

    def test_context_and_explicitly_uncitable_sources_cannot_be_cited(self):
        request = _request()
        payload = _cited_output(request)
        source = replace(request.evidence[0], citation_allowed=False)
        for scoped in (
            replace(request, evidence=(source,)),
            replace(request, evidence=(), context_messages=(source,)),
        ):
            with self.subTest(context=bool(scoped.context_messages)):
                with self.assertRaises(ConsolidationValidationError) as caught:
                    parse_consolidation_result(payload, request=scoped)
                self.assertEqual(caught.exception.code, "uncitable_source")

    def test_unicode_multiline_and_indentation_are_literal_and_preserved(self):
        request = _request()
        text = '  Caf\u00e9 "Blue":\n\tTake the left door.\nThen wait.'
        source = replace(
            request.evidence[0],
            payload={"role": "user", "modality": "text", "content": text},
        )
        request = replace(request, evidence=(source,))
        quote = '  Caf\u00e9 "Blue":\n\tTake the left door.'
        result = parse_consolidation_result(
            _cited_output(request, text=quote), request=request,
        )
        persisted = consolidation_result_payload(result)
        self.assertEqual(
            persisted["entity_candidates"][0]["evidence"][0]["supporting_text"], quote,
        )
        self.assertEqual(
            set(persisted["entity_candidates"][0]["evidence"][0]),
            {"evidence_type", "evidence_ref_id", "supporting_text"},
        )

    def test_paraphrases_joined_passages_and_json_escapes_fail(self):
        request = _request()
        source = replace(request.evidence[0], payload={
            "role": "user", "modality": "text",
            "content": 'Caf\u00e9.\n  First door. Pause. Last door.',
            "metadata": {"private_label": "not quotation material"},
        })
        request = replace(request, evidence=(source,))
        for quote in (
            "Cafe.", "caf\u00e9.", "First entrance.", "First door. Last door.",
            "Caf\u00e9. First door.", r"Caf\u00e9.", "private_label",
            "not quotation material", '"role":"user"', "   ",
        ):
            with self.subTest(quote=quote):
                with self.assertRaises(ConsolidationValidationError) as caught:
                    parse_consolidation_result(
                        _cited_output(request, text=quote), request=request,
                    )
                self.assertEqual(caught.exception.code, "missing_literal_quotation")

    def test_observation_message_and_tool_strings_remain_citable(self):
        request = _request()
        payloads = {
            "observation": {
                "summary": "Silver keys on a counter",
                "visible_text": [{"text": "HOME"}],
                "entities": [{"name": "keys", "attributes": {"color": "silver"}}],
            },
            "message": {"role": "user", "modality": "audio", "content": "HOME"},
            "tool_execution": {
                "tool_name": "create_note", "status": "completed",
                "arguments": {"body": "HOME"},
                "result": {"note": {"body": "Silver keys", "id": "metadata-only"}},
            },
        }
        for evidence_type, payload in payloads.items():
            with self.subTest(evidence_type=evidence_type):
                source = replace(
                    request.evidence[0], evidence_type=evidence_type, payload=payload,
                )
                scoped = replace(request, evidence=(source,))
                parse_consolidation_result(
                    _cited_output(scoped, text="HOME"), request=scoped,
                )

    def test_separate_fields_and_wrong_sources_cannot_be_combined(self):
        request = _request()
        first = replace(request.evidence[0], evidence_type="tool_execution", payload={
            "arguments": {"first": "alpha", "second": "beta"},
            "result": {"id": "metadata-only", "owner_id": "private-owner"},
        })
        second = replace(
            request.evidence[0], evidence_id=uuid4(), evidence_ref_id=uuid4(),
            payload={"role": "user", "modality": "text", "content": "other source only"},
        )
        request = replace(request, evidence=(first, second))
        for quote in ("alpha beta", "other source only", "metadata-only", "private-owner"):
            with self.subTest(quote=quote), self.assertRaises(ConsolidationValidationError):
                parse_consolidation_result(_cited_output(request, text=quote), request=request)

    def test_observation_and_tool_support_cannot_be_explicit_user_attribution(self):
        request = _request()
        quote = request.evidence[0].payload["content"]
        for evidence_type, source_payload in (
            ("observation", {"summary": quote}),
            ("tool_execution", {"arguments": {"suggestion": quote}}),
        ):
            with self.subTest(evidence_type=evidence_type):
                source = replace(
                    request.evidence[0], evidence_type=evidence_type, payload=source_payload,
                )
                scoped = replace(request, evidence=(source,))
                output = _output(scoped)
                # The original fixture deliberately shares this citation across candidates.
                output["entity_candidates"][0]["evidence"][0]["evidence_type"] = evidence_type
                with self.assertRaises(ConsolidationValidationError) as caught:
                    parse_consolidation_result(output, request=scoped)
                self.assertEqual(caught.exception.code, "invalid_explicit_user_attribution")

    def test_passive_visual_user_role_is_not_citable_user_speech(self):
        request = _request()
        payload = _cited_output(request)
        source = replace(request.evidence[0], payload={
            **request.evidence[0].payload, "modality": "visual",
        })
        with self.assertRaises(ConsolidationValidationError) as caught:
            parse_consolidation_result(payload, request=replace(request, evidence=(source,)))
        self.assertEqual(caught.exception.code, "uncitable_source")


def _request(*, message_role: str = "user") -> ConsolidationRequest:
    now = datetime.now(UTC)
    event = ExperienceEvent(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        status="consolidating",
        event_kind="mixed",
        modalities=("audio", "visual"),
        started_at=now,
        ended_at=now,
        title="",
        summary="",
        scene="kitchen",
        activity="picking up keys",
        state={"scene": "kitchen"},
        boundary_reason="activity_changed",
        importance=0.4,
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
    message_id = uuid4()
    evidence = ConsolidationEvidenceInput(
        evidence_id=uuid4(),
        evidence_type="message",
        evidence_ref_id=message_id,
        occurred_at=now,
        source_sequence=1,
        payload={
            "role": message_role,
            "modality": "audio",
            "content": "I keep my silver keys on the kitchen counter",
            "metadata": {},
        },
    )
    keyframe = ConsolidationKeyframeInput(
        visual_frame_id=uuid4(),
        rank=1,
        reason="keys visible",
        quality_score=0.9,
        captured_at=now,
        frame_description="Silver keys beside a label reading HOME",
        storage_bucket="frames",
        storage_path="frame.jpg",
        content_type="image/jpeg",
        image_bytes=b"\xff\xd8\xff\x00\xff\xd9",
    )
    return ConsolidationRequest(
        event=event,
        evidence=(evidence,),
        context_messages=(),
        keyframes=(keyframe,),
        event_end_working_state={"scene": "kitchen"},
        owner_working_memory={"active_visual_focus": "find the keys"},
        active_reminders=(
            {
                "id": str(uuid4()),
                "title": "Take the keys",
                "details": "Bring the keys when leaving",
                "remind_at": None,
                "condition": "when leaving home",
                "updated_at": now.isoformat(),
            },
        ),
    )


def _output(request: ConsolidationRequest) -> dict[str, object]:
    evidence = request.evidence[0]
    citation = {
        "evidence_type": "message",
        "evidence_ref_id": str(evidence.evidence_ref_id),
        "supporting_text": "I keep my silver keys on the kitchen counter",
    }
    return {
        "event_title": "Picked up the keys",
        "event_summary": "The user handled silver keys in the kitchen.",
        "location": "kitchen",
        "actions": ["picked up keys"],
        "entity_candidates": [
            {
                "name": "silver keys",
                "category": "object",
                "confidence": 0.9,
                "evidence": [citation],
            }
        ],
        "visible_text": [],
        "episode_importance": 0.7,
        "confidence": 0.9,
        "should_promote_to_episode": True,
        "promotion_reason": "Recoverable user action",
        "selected_keyframe_ids": [str(request.keyframes[0].visual_frame_id)],
        "semantic_fact_candidates": [
            {
                "subject": "user",
                "predicate": "has",
                "value": "silver keys",
                "confidence": 0.95,
                "explicitly_user_stated": True,
                "change_intent": "assert",
                "previous_value": None,
                "valid_from_hint": None,
                "valid_to_hint": None,
                "evidence": [citation],
            }
        ],
        "relation_candidates": [],
        "reflection_candidates": [],
        "procedure_candidates": [],
    }


def _cited_output(
    request: ConsolidationRequest, *, reference_id=None, text: str | None = None,
) -> dict[str, object]:
    """Minimal candidate for testing citation policy without unrelated assertions."""
    output = _output(request)
    source = request.evidence[0]
    output["entity_candidates"][0]["evidence"] = [{
        "evidence_type": source.evidence_type,
        "evidence_ref_id": str(reference_id or source.evidence_ref_id),
        "supporting_text": text if text is not None else source.payload["content"],
    }]
    for group in (
        "visible_text", "semantic_fact_candidates", "relation_candidates",
        "reflection_candidates", "procedure_candidates",
    ):
        output[group] = []
    return output
