from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.experience.models import ExperienceEventEvidence
from app.memory.consolidation.contracts import (
    ConsolidationValidationError,
    parse_consolidation_result,
)
from app.memory.consolidation.evidence import (
    build_citation_catalog,
    event_state_for_evidence,
)
from app.memory.consolidation.prompts import build_consolidation_prompt
from app.memory.consolidation.service import ExperienceConsolidationService
from app.memory.embeddings import (
    build_event_retrieval_document,
    build_event_text_embedding_content,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval.query import MemoryRetrievalQuery, normalize_memory_query
from test_consolidation_contracts import _cited_output, _request
from test_memory_repository import _Pool


_LABEL = "Use 12 mL before 18:30."


def _visual_request(tag: str = "unsupported_retrieval_hint"):
    request = _request()
    source = replace(request.evidence[0], evidence_type="observation", payload={
        "summary": "A card displays an instruction.",
        "visible_text": [{"text": _LABEL, "confidence": 0.9}],
        "memory_tags": [tag],
        "confidence": 0.9,
        "compacted": False,
    })
    state = {
        "tags": [tag],
        "recent_summaries": [source.payload["summary"]],
        "evidence_counts": {"observation": 1},
    }
    return replace(
        request,
        event=replace(request.event, state=state),
        evidence=(source,),
        event_end_working_state=dict(state),
    )


def _prompt_payload(request):
    return json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])


class RetrievalTagEvidenceBoundaryTests(unittest.TestCase):
    def test_event_projection_preserves_typed_content_and_does_not_mutate_state(self):
        for state in (
            {},
            {"count": 0, "enabled": False, "missing": None},
            {
                "tags": ["retrieval only"],
                "recent_summaries": ["A card is visible."],
                "details": {"tags": ["literal nested content"], "enabled": False},
            },
        ):
            with self.subTest(state=state):
                original = deepcopy(state)
                projected = event_state_for_evidence(state)
                self.assertEqual(projected, {
                    key: value for key, value in state.items() if key != "tags"
                })
                self.assertIsNot(projected, state)
                self.assertEqual(state, original)

    def test_generated_tags_are_neither_prompt_content_nor_citation_authority(self):
        for tag in ("medical_dosage", "professional_mechanic", "flight_status"):
            with self.subTest(tag=tag):
                request = _visual_request(tag)
                payload = _prompt_payload(request)
                source = request.evidence[0]
                self.assertNotIn(tag, json.dumps(payload))
                self.assertNotIn("tags", payload["event"]["state"])
                self.assertNotIn("tags", payload["event_end_working_state"])
                entry = payload["attached_evidence"][0]
                self.assertFalse(any(
                    item["source_path"].startswith("memory_tags")
                    for item in entry["quotation_material"] + entry["source_context"]
                ))
                catalog = build_citation_catalog(request)
                self.assertFalse(catalog[("observation", source.evidence_ref_id)].contains_quote(tag))
                with self.assertRaises(ConsolidationValidationError) as raised:
                    parse_consolidation_result(
                        _cited_output(request, text=tag), request=request,
                    )
                self.assertEqual(raised.exception.code, "missing_literal_quotation")
                parse_consolidation_result(
                    _cited_output(request, text=_LABEL), request=request,
                )

    def test_literal_visible_text_is_citable_even_when_it_matches_a_tag(self):
        request = _visual_request("medical_dosage")
        source = replace(request.evidence[0], payload={
            **request.evidence[0].payload,
            "visible_text": [{"text": "medical_dosage", "confidence": 0.9}],
        })
        request = replace(request, evidence=(source,))
        payload = _prompt_payload(request)
        self.assertIn("medical_dosage", json.dumps(payload["attached_evidence"]))
        parse_consolidation_result(
            _cited_output(request, text="medical_dosage"), request=request,
        )

    def test_uncitable_observation_paths_also_exclude_generated_tags(self):
        request = _visual_request()
        uncitable = replace(request.evidence[0], citation_allowed=False)
        context = replace(
            uncitable, evidence_id=None, evidence_ref_id=uuid4(),
        )
        request = replace(
            request, evidence=(uncitable,), context_messages=(context,),
        )
        payload = _prompt_payload(request)
        self.assertEqual(payload["attached_evidence"], [])
        self.assertNotIn("unsupported_retrieval_hint", json.dumps(payload))
        for name in ("uncitable_evidence", "context_only_messages"):
            item = payload[name][0]
            self.assertFalse(item["citation_allowed"])
            self.assertNotIn("memory_tags", item["payload"])
            self.assertEqual(item["payload"]["visible_text"][0]["text"], _LABEL)

    def test_tool_business_tags_remain_content_in_citable_and_context_paths(self):
        request = _request()
        source = replace(request.evidence[0], evidence_type="tool_execution", payload={
            "tool_name": "create_note",
            "status": "completed",
            "arguments": {"tags": ["user-chosen label"]},
            "result": {
                "memory_tags": ["business result label"],
                "enabled": False,
                "remaining": 0,
            },
        })
        uncitable = replace(
            source, evidence_id=uuid4(), evidence_ref_id=uuid4(), citation_allowed=False,
        )
        request = replace(request, evidence=(source, uncitable))
        payload = _prompt_payload(request)
        self.assertEqual(payload["uncitable_evidence"][0]["payload"], source.payload)
        entry = payload["attached_evidence"][0]
        context = {item["source_path"]: item["value"] for item in entry["source_context"]}
        self.assertIs(context["result.enabled"], False)
        self.assertEqual(context["result.remaining"], 0)
        for text in ("user-chosen label", "business result label"):
            with self.subTest(text=text):
                parse_consolidation_result(
                    _cited_output(request, text=text), request=request,
                )

    def test_user_statement_is_not_filtered_by_tag_terminology(self):
        request = _request()
        source = replace(request.evidence[0], payload={
            "role": "user",
            "modality": "text",
            "content": 'Label this note "medical_dosage".',
        })
        request = replace(request, evidence=(source,))
        self.assertIn(source.payload["content"], [
            item["text"]
            for item in _prompt_payload(request)["attached_evidence"][0]["quotation_material"]
        ])
        parse_consolidation_result(
            _cited_output(request, text="medical_dosage"), request=request,
        )

    def test_evidence_projection_leaves_both_embedding_inputs_unchanged(self):
        request = _visual_request()
        original_state = deepcopy(request.event.state)
        before = (
            build_event_text_embedding_content(request.event),
            build_event_retrieval_document(request.event),
        )
        self.assertTrue(all("unsupported_retrieval_hint" in value for value in before))
        self.assertNotIn("unsupported_retrieval_hint", build_consolidation_prompt(request))
        self.assertEqual(request.event.state, original_state)
        self.assertEqual(before, (
            build_event_text_embedding_content(request.event),
            build_event_retrieval_document(request.event),
        ))


class ConsolidationRequestTagBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_uses_evidence_view_without_mutating_persisted_sources(self):
        fixture = _visual_request()
        event = fixture.event
        source = fixture.evidence[0]
        observation = SimpleNamespace(
            id=source.evidence_ref_id,
            owner_id=event.owner_id,
            session_id=event.session_id,
            summary=_LABEL,
            scene=None,
            activity=None,
            entities=(),
            visible_text=(),
            state_changes=(),
            memory_tags=("unsupported_retrieval_hint",),
            keyframe_candidates=(),
            confidence=0.9,
            is_compacted=False,
        )
        repository = AsyncMock()
        repository.get_event.return_value = event
        repository.list_event_evidence.return_value = [ExperienceEventEvidence(
            id=source.evidence_id,
            event_id=event.id,
            evidence_type="observation",
            evidence_ref_id=source.evidence_ref_id,
            occurred_at=source.occurred_at,
            source_sequence=source.source_sequence,
            ingested_at=source.occurred_at,
            metadata={},
        )]
        repository.list_nonvisual_source_payloads.return_value = {}
        repository.list_context_messages.return_value = []
        repository.list_event_keyframe_sources.return_value = []
        perception = AsyncMock()
        perception.list_observations_by_ids.return_value = [observation]
        client = AsyncMock()
        service = ExperienceConsolidationService(
            experience_repository=repository,
            perception_repository=perception,
            client=client,
            image_loader=None,
            owner_working_memory_reader=None,
            active_reminder_reader=None,
            max_keyframes=1,
            max_transcripts=1,
        )

        request = await service.build_request(
            owner_id=event.owner_id,
            session_id=event.session_id,
            event_id=event.id,
        )

        self.assertEqual(request.event.id, event.id)
        self.assertEqual(request.event.owner_id, event.owner_id)
        self.assertNotIn("tags", request.event.state)
        self.assertEqual(request.event_end_working_state, request.event.state)
        self.assertNotIn("memory_tags", request.evidence[0].payload)
        self.assertEqual(request.evidence[0].payload["summary"], _LABEL)
        self.assertEqual(request.evidence[0].evidence_ref_id, source.evidence_ref_id)
        self.assertEqual(event.state["tags"], ["unsupported_retrieval_hint"])
        self.assertEqual(observation.memory_tags, ("unsupported_retrieval_hint",))
        self.assertNotIn("unsupported_retrieval_hint", build_consolidation_prompt(request))
        client.generate.assert_not_awaited()


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for retrieval tag boundary integration tests",
)
class RetrievalTagBoundaryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = await asyncpg.connect(os.environ["JARVIS_TEST_DB_URL"])
        self.addAsyncCleanup(self.connection.close)
        transaction = self.connection.transaction()
        await transaction.start()

        async def rollback_transaction():
            # asyncpg decorates rollback; unittest needs an explicit async callback.
            await transaction.rollback()

        self.addAsyncCleanup(rollback_transaction)
        # Reuse the existing test shim so repository calls share this rollback-only connection.
        self.repository = MemoryRepository(_Pool(self.connection))
        self.owner_id = f"retrieval-tag-boundary-{uuid4()}"
        self.now = datetime.now(UTC)
        self.session_id = await self.connection.fetchval(
            "insert into jarvis_sessions (owner_id, title) values ($1, $2) returning id",
            self.owner_id, "Retrieval tag boundary test",
        )

    async def _insert_event(self, state, *, summary="A card displays an instruction."):
        return await self.connection.fetchval(
            """
            insert into experience_events (
                owner_id, session_id, status, event_kind, modalities,
                started_at, ended_at, keyframes_finalized_at, consolidation_ready_at,
                title, summary, scene, state
            )
            values (
                $1, $2, 'closed', 'visual', array['visual']::text[],
                $3, $3, $3, $3, 'Card observation', $4, 'table', $5::jsonb
            )
            returning id
            """,
            self.owner_id, self.session_id, self.now, summary, json.dumps(state),
        )

    def _query(self, text=None):
        return normalize_memory_query(
            MemoryRetrievalQuery(
                query=text, seconds_before_now=600,
                memory_types=("experience_event",), session_id=self.session_id,
            ),
            now=self.now + timedelta(seconds=1), default_seconds=600, max_days=3650,
        )

    async def test_tag_only_search_still_matches_without_returning_tag_as_evidence(self):
        hint = "retrievalhint" + uuid4().hex
        state = {"tags": [hint], "recent_summaries": [_LABEL], "count": 0}
        event_id = await self._insert_event(state)
        hits = await self.repository.list_lexical_retrieval_hits(
            owner_id=self.owner_id, query=self._query(hint), limit=10,
        )
        self.assertEqual([hit.document.id for hit in hits], [event_id])
        self.assertNotIn(hint, hits[0].document.content)
        self.assertIn(_LABEL, hits[0].document.content)
        stored = await self.connection.fetchval(
            "select state from experience_events where id = $1", event_id,
        )
        self.assertEqual(json.loads(stored), state)

    async def test_time_retrieval_preserves_untagged_and_literal_content(self):
        states = (
            {},
            {"count": 0, "enabled": False, "missing": None},
            {
                "tags": ["medical_dosage"],
                "recent_summaries": ["Visible text: medical_dosage"],
                "details": {"tags": ["literal nested content"]},
            },
        )
        ids = [await self._insert_event(state) for state in states]
        hits = await self.repository.list_time_retrieval_hits(
            owner_id=self.owner_id, query=self._query(), limit=10,
        )
        by_id = {hit.document.id: hit.document for hit in hits}
        self.assertEqual(set(by_id), set(ids))
        for event_id, state in zip(ids, states):
            with self.subTest(state=state):
                content = by_id[event_id].content
                self.assertTrue(content.startswith("table "))
                self.assertEqual(json.loads(content[len("table "):]), {
                    key: value for key, value in state.items() if key != "tags"
                })
