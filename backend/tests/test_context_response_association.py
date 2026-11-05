"""Offline owner/source joins. In-memory persistence is NOT SQL/native proof."""
import asyncio
from datetime import UTC, datetime
import json
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from evals.responsive_voice.response_input_association import resolve_response_input


class CanonicalMemoryStore:
    def __init__(self):
        self.rows = {}
        self.events = []
        self.request_evidence = AsyncMock(side_effect=AssertionError("No admission lookup"))

    async def append(self, owner, event):
        key = owner, event.source_namespace, event.source_event_id
        previous = self.rows.get(key)
        self.events.append(event)
        row = ContextRecord(id=uuid4(), owner_id=owner, session_id=event.session_id,
            sequence=len(self.events), origin_sequence=previous.origin_sequence if previous else len(self.events),
            source_namespace=event.source_namespace, source_event_id=event.source_event_id,
            revision=previous.revision + 1 if previous else 1,
            supersedes_event_id=previous.id if previous else None,
            kind=event.kind, trust=event.trust, finality=event.finality,
            source_time=event.source_time, recorded_at=datetime.now(UTC), payload_id=uuid4(),
            availability="available", content_digest=None, content=event.payload, is_current=True)
        self.rows[key] = row
        return row


class ResponseInputAssociationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = CanonicalMemoryStore()
        self.capture = SessionContextCapture(self.store, owner_id="fixture-owner",
            session_id=uuid4(), provider="openai", audio_session_id="fixture-audio")
        self.addAsyncCleanup(self.capture.close)

    def binding(self, response="response-1"):
        return self.capture.response_input_association(response)

    def row(self, category, identity):
        return self.store.rows[(self.capture.owner_id, self.capture.namespace,
                                self.capture._source(category, identity))]

    def resolve(self, output, request, **changes):
        options = dict(output=output, request=request, owner_id=self.capture.owner_id,
            session_id=self.capture.session_id, audio_session_id=self.capture.audio_session_id,
            response_id="response-1", item_id="output-1")
        return resolve_response_input(**(options | changes))

    def output(self, response="response-1", item="output-1"):
        self.capture.observe_output({"type": "transcript.done", "response_id": response,
            "item_id": item, "text": "PRIVATE synthetic assistant words"}, forwarded=True)
        self.capture.finish_response(response, completed=True, interrupted=False)

    async def prepare(self, *, finalized=True):
        self.capture.admit_openai_commit("request-1", 7)
        self.capture.openai_committed("provider-input-1")
        self.capture.bind_openai_response("response-1", 7)
        if finalized:
            self.capture.openai_transcript("provider-input-1", "PRIVATE synthetic request")
        self.output()
        self.assertTrue(await self.capture.flush())
        return self.row("output", "output-1"), self.row("input", "commit:request-1")

    async def test_admission_exports_only_source_identity_without_writer_or_lookup(self):
        self.capture.admit_openai_commit("request-1", 7)
        await self.capture.flush()
        count = len(self.store.events)
        writer = self.capture._writer
        self.capture.bind_openai_response("response-1", 7)
        binding = self.binding()
        self.assertEqual(binding["input_id"], "commit:request-1")
        self.assertEqual(binding["source_event_id"], self.capture._source("input", "commit:request-1"))
        self.assertEqual(binding["turn_generation"], 7)
        self.assertIs(type(binding["turn_generation"]), int)
        self.assertEqual(binding["binding_status"], "source_bound")
        self.assertIsNone(binding["canonical_event_id"])
        self.assertIsNone(binding["canonical_revision"])
        self.assertFalse(binding["full_chain_proven"])
        self.assertIs(self.capture._writer, writer)
        await asyncio.sleep(0)
        self.assertEqual(len(self.store.events), count)
        self.store.request_evidence.assert_not_awaited()
        binding["input_id"] = "tampered"
        self.assertEqual(self.binding()["input_id"], "commit:request-1")

    async def test_pending_then_final_and_revised_use_actual_canonical_revision(self):
        output, pending = await self.prepare(finalized=False)
        self.assertEqual(self.resolve(output, pending)["status"], "unavailable")
        self.capture.openai_transcript("provider-input-1", "PRIVATE finalized request")
        await self.capture.flush()
        final = self.row("input", "commit:request-1")
        resolved = self.resolve(output, final)
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["canonical_input"]["event_id"], str(final.id))
        self.assertEqual(resolved["canonical_input"]["revision"], 2)
        self.assertEqual(resolved["canonical_input"]["provider_item_id"], "provider-input-1")
        self.capture.openai_transcript("provider-input-1", "PRIVATE revised request")
        await self.capture.flush()
        revised = self.row("input", "commit:request-1")
        self.assertEqual(revised.finality, "revised")
        self.assertEqual(self.resolve(output, revised)["canonical_input"]["revision"], 3)
        self.assertEqual(self.binding(), output.content["response_input_association"])
        self.assertNotIn("PRIVATE", json.dumps(resolved))

    async def test_failed_transcript_and_absent_record_do_not_resolve(self):
        output, _ = await self.prepare(finalized=False)
        self.capture.openai_transcript("provider-input-1", None)
        await self.capture.flush()
        self.assertEqual(self.resolve(output, self.row("input", "commit:request-1"))["status"], "unavailable")
        self.assertEqual(self.resolve(output, None)["reason"], "canonical_input_missing")

    async def test_rejected_commit_is_bound_but_not_finalized_authority(self):
        self.capture.admit_openai_commit("request-1", 7)
        self.capture.bind_openai_response("response-1", 7)
        self.capture.openai_commit_rejected("request-1")
        self.output()
        await self.capture.flush()
        self.assertEqual(self.binding()["binding_status"], "source_bound")
        self.assertEqual(self.resolve(self.row("output", "output-1"),
            self.row("input", "commit:request-1"))["status"], "unavailable")

    async def test_unknown_generation_stays_unresolved_after_late_admission(self):
        self.capture.bind_openai_response("response-1", 7)
        self.capture.admit_openai_commit("request-1", 7)
        self.capture.bind_openai_response("response-1", 7)
        self.assertEqual(self.binding()["binding_status"], "unresolved")
        self.assertIsNone(self.binding()["input_id"])
        self.assertIsNone(self.capture.issue)
        self.capture.bind_openai_response("response-2", 7)
        self.assertEqual(self.binding("response-2")["input_id"], "commit:request-1")

    async def test_no_generation_or_unknown_response_never_uses_latest_input(self):
        self.capture.admit_openai_commit("latest", 99)
        self.capture.bind_openai_response("response-1", None)
        self.assertEqual(self.binding()["binding_status"], "unresolved")
        self.assertIsNone(self.binding("never-admitted"))
        self.output()
        await self.capture.flush()
        self.assertEqual(self.resolve(self.row("output", "output-1"), None)["status"], "unresolved")

    async def test_late_response_and_reversed_transcripts_keep_original_identity(self):
        self.capture.admit_openai_commit("request-1", 7)
        self.capture.admit_openai_commit("request-2", 8)
        self.capture.openai_committed("provider-input-1")
        self.capture.openai_committed("provider-input-2")
        self.capture.openai_transcript("provider-input-2", "Newer request")
        self.capture.bind_openai_response("response-1", 7)
        self.capture.openai_transcript("provider-input-1", "Older exact request")
        self.output()
        await self.capture.flush()
        output = self.row("output", "output-1")
        self.assertEqual(self.resolve(output, self.row("input", "commit:request-1"))["status"], "resolved")
        self.assertEqual(self.resolve(output, self.row("input", "commit:request-2"))["status"], "unavailable")

    async def test_identical_duplicate_is_idempotent_and_does_no_extra_persistence(self):
        await self.prepare()
        count, binding = len(self.store.events), self.binding()
        for _ in range(3):
            self.capture.bind_openai_response("response-1", 7)
        await asyncio.sleep(0)
        self.assertEqual(len(self.store.events), count)
        self.assertEqual(self.binding(), binding)
        self.assertIsNone(self.capture.issue)

    async def test_conflicting_generation_invalidates_memory_and_current_checkpoint(self):
        await self.prepare()
        self.capture.admit_openai_commit("request-2", 8)
        self.capture.bind_openai_response("response-1", 8)
        await self.capture.flush()
        self.assertEqual(self.capture.issue, "context_response_input_conflict")
        self.assertEqual(self.binding()["binding_status"], "conflict")
        self.assertIsNone(self.capture._responses["response-1"])
        output = self.row("output", "output-1")
        self.assertEqual(output.content["response_input_association"]["binding_status"], "conflict")
        self.assertEqual(self.resolve(output, self.row("input", "commit:request-1"))["status"], "unresolved")
        self.capture.bind_openai_call("call", "response-1")
        self.assertIsNone(await self.capture.request_for_call("call"))
        self.store.request_evidence.assert_not_awaited()
        count = self.capture.failed_events
        self.capture.bind_openai_response("response-1", 7)
        self.assertEqual(self.capture.failed_events, count)

    async def test_conflicting_parent_is_not_rebound(self):
        self.capture.admit_openai_commit("request-1", 7)
        first, second = uuid4(), uuid4()
        self.capture.bind_openai_response("response-1", 7, source_payload_id=first)
        self.capture.bind_openai_response("response-1", 7, source_payload_id=second)
        self.assertEqual(self.capture.issue, "context_response_source_conflict")
        self.assertEqual(self.binding()["binding_status"], "conflict")
        self.assertEqual(self.capture._response_sources["response-1"], (first,))

    async def test_reused_generation_cannot_rebind_an_existing_response(self):
        self.capture.admit_openai_commit("request-1", 7)
        self.capture.bind_openai_response("response-1", 7)
        self.capture.admit_openai_commit("different", 7)
        self.capture.bind_openai_response("response-1", 7)
        self.assertEqual(self.binding()["binding_status"], "conflict")

    async def test_generation_type_and_identity_bounds_fail_closed(self):
        for generation in (True, "7", 1.5, -1, 2**63):
            with self.subTest(generation=generation):
                self.capture.bind_openai_response("invalid", generation)
                self.assertIsNone(self.binding("invalid"))
        self.capture.bind_openai_response("x" * 513, 7)
        self.assertEqual(self.capture.issue, "context_response_identity_invalid")
        self.assertEqual(len(self.capture._responses), 0)

    async def test_response_capacity_remains_bounded(self):
        capture = SessionContextCapture(self.store, owner_id="bounded", session_id=uuid4(),
            provider="openai", audio_session_id="bounded-audio", max_sources=1)
        self.addAsyncCleanup(capture.close)
        capture.bind_openai_response("one", None)
        capture.bind_openai_response("two", None)
        self.assertEqual(len(capture._responses), 1)
        self.assertEqual(len(capture._response_generations), 1)
        self.assertEqual(capture.issue, "context_capture_source_limit")

    async def test_wrong_owner_session_audio_source_response_or_item_is_rejected(self):
        output, request = await self.prepare()
        for changes in ({"owner_id": "other"}, {"session_id": uuid4()},
                        {"audio_session_id": "other"}, {"response_id": "other"}, {"item_id": "other"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.resolve(output, request, **changes)["status"], "unavailable")
        for changes in ({"owner_id": "other"}, {"session_id": uuid4()},
                        {"source_namespace": "other"}, {"source_event_id": "other"}):
            with self.subTest(request=changes):
                self.assertEqual(self.resolve(output, request.model_copy(update=changes))["status"], "unavailable")

    async def test_canonical_availability_finality_and_current_revision_are_required(self):
        output, request = await self.prepare()
        changes = ({"is_current": False}, {"availability": "removed", "content": None},
                   {"availability": "unavailable"}, {"content": None}, {"revision": 0},
                   {"finality": "provisional"}, {"finality": "unavailable"},
                   {"kind": "assistant_output", "trust": "assistant_output"})
        for change in changes:
            with self.subTest(change=change):
                result = self.resolve(output, request.model_copy(update=change))
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["canonical_input"])
        for fields in ({"input_id": "other"}, {"audio_session_id": "other"},
                       {"audio_provider": "gemini"}, {"transcript_status": "pending"}, {"text": ""}):
            with self.subTest(content=fields):
                modified = request.model_copy(update={"content": request.content | fields})
                self.assertEqual(self.resolve(output, modified)["status"], "unavailable")

    async def test_withdrawn_or_superseded_output_cannot_be_rebuilt_from_cached_binding(self):
        output, request = await self.prepare()
        for change in ({"content": None, "availability": "removed"}, {"is_current": False},
                       {"source_event_id": "wrong-output"}):
            with self.subTest(change=change):
                self.assertEqual(self.resolve(output.model_copy(update=change), request)["status"], "unavailable")

    async def test_binding_contract_and_digest_cannot_be_replaced_by_claimed_revision(self):
        output, request = await self.prepare()
        binding = self.binding()
        for fields in ({"source_event_id": "wrong-input"}, {"owner_id": "other"},
                       {"session_id": str(uuid4())}, {"source_namespace": "other"},
                       {"canonical_event_id": str(request.id)}, {"canonical_revision": request.revision},
                       {"schema_version": True}, {"turn_generation": True},
                       {"full_chain_proven": True}, {"heard_state": "heard"}):
            with self.subTest(fields=fields):
                modified = output.model_copy(update={"content": output.content |
                    {"response_input_association": binding | fields}})
                self.assertEqual(self.resolve(modified, request)["status"], "unavailable")

    async def test_completed_output_is_not_hearing_or_full_chain_proof(self):
        output, request = await self.prepare()
        result = self.resolve(output, request)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(output.content["completion_state"], "completed")
        self.assertIsNone(output.content["heard_transcript"])
        self.assertEqual(result["heard_state"], "unknown")
        self.assertFalse(result["full_chain_proven"])
        self.assertLess(len(json.dumps(self.binding())), 4096)
        self.assertNotIn("PRIVATE", json.dumps(self.binding()))
        self.assertEqual(set(result["canonical_input"]), {
            "event_id", "payload_id", "revision", "source_namespace", "source_event_id", "provider_item_id"})

    async def test_non_openai_output_keeps_legacy_shape(self):
        capture = SessionContextCapture(self.store, owner_id="gemini", session_id=uuid4(),
            provider="gemini", audio_session_id="gemini-audio")
        self.addAsyncCleanup(capture.close)
        capture.observe_output({"type": "transcript.done", "response_id": "r", "item_id": "i",
            "text": "Synthetic legacy output"}, forwarded=True)
        await capture.flush()
        row = self.store.rows[("gemini", capture.namespace, capture._source("output", "i"))]
        self.assertNotIn("response_input_association", row.content)
        self.assertIsNone(capture.response_input_association("r"))


if __name__ == "__main__":
    unittest.main()
