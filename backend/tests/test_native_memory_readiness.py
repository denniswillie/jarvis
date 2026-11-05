"""Offline composition only: real capture/resolver, controlled canonical records.

No SQL commit, provider ingestion, manual native admission or continuation is
claimed authentic by these fixtures. Host/native attachment remains unwired.
"""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import unittest
from uuid import uuid4

from app.knowledge.models import SourceVersion
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.models import digest
from app.tools.inputs import success
from evals.responsive_voice.native_memory_readiness import (
    CanonicalMemorySnapshot, ReadinessRequest, evaluate_memory_readiness,
)
from evals.responsive_voice.native_memory_replay_control import EvidenceRef, Scope
from test_context_response_association import CanonicalMemoryStore


class MemoryReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        run = uuid4().hex
        scope = Scope(run, f"direct-image-eval-second-brain-{run}-a", f"jarvis_sb_eval_{run}_0",
                      str(uuid4()), "jarvis_" + uuid4().hex)
        self.now = datetime(2026, 10, 31, 22, tzinfo=UTC)
        self.request = ReadinessRequest(scope, "invocation-1", "a"*64, uuid4(), "audio-1", "connection-1",
            0, 1, "segment-0", "commit-client-0", "response-1", "output-1", uuid4().hex,
            self.now - timedelta(seconds=1), self.now + timedelta(seconds=10))
        self.store = CanonicalMemoryStore()
        self.capture = SessionContextCapture(self.store, owner_id=scope.owner_id,
            session_id=self.request.session_id, provider="openai", audio_session_id=self.request.audio_session_id)
        self.addAsyncCleanup(self.capture.close)
        self.capture.admit_openai_commit("request-1", 7)
        self.capture.openai_committed("provider-input-1")
        self.capture.openai_transcript("provider-input-1", "PRIVATE remember my dated report")
        self.capture.bind_openai_response("response-1", 7)
        self.capture.bind_openai_call("call-1", "response-1")
        self.capture.observe_output({"type":"transcript.done", "response_id":"response-1",
            "item_id":"output-1", "text":"PRIVATE acknowledgement"}, forwarded=True)
        self.capture.finish_response("response-1", interrupted=False, completed=True)
        self.assertTrue(await self.capture.flush())
        request_row = self.row("input", "commit:request-1")
        output = self.row("output", "output-1")
        statement = {"memory_type":"fact", "content":"PRIVATE Cedar code was seventeen",
            "occurred_at":"2026-08-01T10:00:00+00:00", "attribution":{
                "kind":"model_formulated_from_finalized_user_request",
                "request_event_id":str(request_row.id), "request_revision":request_row.revision}}
        identity = digest({"session_id":str(self.request.session_id), "statement":statement})
        memory = await self.store.append(scope.owner_id, ContextIngress(session_id=self.request.session_id,
            source_namespace="conversation-explicit-memory", source_event_id=identity, receipt_id=identity,
            kind="tool_outcome", trust="tool_receipt", payload=statement))
        # Opaque database digests are fixture values, never hashes of Python JSON.
        memory = memory.model_copy(update={"content_digest":"b"*64})
        self.source = SourceVersion(owner_id=scope.owner_id, area_id=UUID(scope.area_id), source_id=uuid4(),
            revision=1, generation=1, payload_id=memory.payload_id, document_id="source-document-1",
            bank_name=scope.memory_bank_id, content_digest="b"*64, source_type="tool_outcome", trust="tool_receipt",
            source_time=datetime(2026,8,1,10,tzinfo=UTC), time_basis="canonical event source_time",
            recorded_at=self.now, content=statement, is_current=True, ingestion_status="searchable",
            session_id=self.request.session_id, context_session_id=self.request.session_id,
            context_event_id=memory.id, context_event_revision=memory.revision, context_recorded_at=self.now)
        metadata = self.source.original_evidence()
        metadata.pop("content")
        metadata["ingestion_status"] = "pending"  # Real remember return predates indexing.
        tool_result = success({"memory_status":"stored", "memory_id":str(self.source.source_id),
                               "source":metadata, "replayed":False})
        tool = await self.store.append(scope.owner_id, self.capture._event(
            self.capture._source("bound-tool", "call-1"), kind="tool_outcome", trust="tool_receipt",
            payload={"provider_call_id":"call-1", "tool_name":"remember", "input_id":"commit:request-1",
                     "result":tool_result}))
        evidence = EvidenceRef("controlled/continuation.json", "c"*64, "fixture-only-not-native")
        admission = {"event":"input_commit_admitted", "run_id":scope.run_id, "owner_id":scope.owner_id,
            "invocation_id":self.request.invocation_id, "manifest_sha256":self.request.manifest_sha256,
            "session_id":str(self.request.session_id), "audio_session_id":"audio-1", "connection_id":"connection-1",
            "prior_input_ordinal":0, "target_input_ordinal":1, "segment_id":"segment-0",
            "client_event_id":"commit-client-0", "request_id":"request-1", "turn_generation":7, "evidence":evidence}
        continuation = {"kind":"observed_tool_result_continuation", "owner_id":scope.owner_id,
            "session_id":str(self.request.session_id), "audio_session_id":"audio-1", "connection_id":"connection-1",
            "response_id":"response-1", "request_id":"request-1", "turn_generation":7,
            "provider_call_id":"call-1", "tool_event_id":str(tool.id), "tool_revision":tool.revision,
            "tool_payload_id":str(tool.payload_id), "request_event_id":str(request_row.id),
            "request_revision":request_row.revision, "evidence":evidence}
        self.snapshot = CanonicalMemorySnapshot(scope, self.request.nonce, self.now,
            request_row, output, tool, memory, (self.source,), ({"id":uuid4(), "owner_id":scope.owner_id,
                "document_id":self.source.document_id, "generation":1, "operation":"retain",
                "status":"searchable", "dispatch_count":1},), admission, continuation)

    def row(self, category, identity):
        return self.store.rows[(self.request.scope.owner_id, self.capture.namespace,
                                self.capture._source(category, identity))]

    def evaluate(self, snapshot=None, control=None, **changes):
        return evaluate_memory_readiness(control or self.request,
            replace(snapshot or self.snapshot, **changes), now=self.now)

    def test_exact_composition_uses_resolved_revision_without_hearing_or_clock_claim(self):
        result = self.evaluate()
        self.assertEqual((result["status"], result["reason"]), ("ready", "exact_remember_searchable_and_continued"))
        self.assertEqual(result["canonical_input"]["revision"], self.snapshot.request.revision)
        self.assertIsNone(self.snapshot.output.content["response_input_association"]["canonical_revision"])
        self.assertEqual(result["retained_source"]["retain_operation_id"], str(self.snapshot.retains[0]["id"]))
        self.assertFalse(result["full_chain_proven"])
        self.assertEqual(result["heard_state"], "unknown")
        self.assertEqual(result["playback"], "not_evaluated")
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_uncommitted_missing_pointer_only_and_unrelated_tool_do_not_release(self):
        for changes in ({"tool_outcome":None}, {"memory_event":None}, {"sources":()}, {"retains":()}):
            with self.subTest(missing=next(iter(changes))):
                self.assertNotEqual(self.evaluate(**changes)["status"], "ready")
        tool = self.snapshot.tool_outcome
        for patch in ({"provider_call_id":"another-call"}, {"tool_name":"search_memory"},
                      {"input_id":"commit:other"}, {"result":{"status":"success",
                       "source_bound_tool_result":{"protocol_version":1, "context_event_id":str(tool.id)}}}):
            with self.subTest(tool_patch=next(iter(patch))):
                self.assertNotEqual(self.evaluate(tool_outcome=tool.model_copy(update={"content":tool.content | patch}))["status"], "ready")

    def test_accepted_unknown_cancelled_backlog_and_wrong_retain_never_mean_searchable(self):
        for status in ("pending", "leased", "accepted", "outcome_unknown", "failed", "cancelled", "done"):
            with self.subTest(status=status):
                self.assertNotEqual(self.evaluate(retains=(self.snapshot.retains[0] | {"status":status},))["status"], "ready")
        for patch in ({"operation":"delete"}, {"document_id":"unrelated"}, {"generation":2},
                      {"owner_id":"foreign"}, {"dispatch_count":0}, {"dispatch_count":True}):
            with self.subTest(retain=patch):
                self.assertNotEqual(self.evaluate(retains=(self.snapshot.retains[0] | patch,))["status"], "ready")
        for patch in ({"ingestion_status":"accepted"}, {"semantic_index_exclusion":"semantic_index_excluded_transport_receipt"}):
            with self.subTest(source=patch):
                self.assertNotEqual(self.evaluate(sources=(self.source.model_copy(update=patch),))["status"], "ready")

    async def test_actual_capture_correction_invalidates_saved_request_revision(self):
        self.capture.openai_transcript("provider-input-1", "PRIVATE corrected finalized request")
        self.assertTrue(await self.capture.flush())
        revised = self.row("input", "commit:request-1")
        self.assertGreater(revised.revision, self.snapshot.request.revision)
        result = self.evaluate(request=revised)
        self.assertEqual(result["reason"], "remember_request_revision_mismatch")

    def test_pending_cross_session_deleted_and_replaced_sources_are_fenced(self):
        for field in ("request", "tool_outcome", "memory_event"):
            row = getattr(self.snapshot, field)
            for patch in ({"is_current":False}, {"availability":"deleted"}, {"session_id":uuid4()}):
                with self.subTest(record=field, patch=patch):
                    self.assertNotEqual(self.evaluate(**{field:row.model_copy(update=patch)})["status"], "ready")
        for patch in ({"generation":2}, {"revision":2}, {"document_id":"replacement"},
                      {"area_id":uuid4()}, {"owner_id":"foreign"}, {"bank_name":"other-bank"},
                      {"context_event_id":uuid4()}, {"payload_id":uuid4()}, {"content_digest":"d"*64}):
            with self.subTest(source=patch):
                self.assertNotEqual(self.evaluate(sources=(self.source.model_copy(update=patch),))["status"], "ready")

    def test_same_input_preamble_and_missing_or_wrong_continuation_do_not_release(self):
        self.assertEqual(self.evaluate(continuation=None)["reason"], "explicit_tool_continuation_unavailable")
        for patch in ({"response_id":"pre-tool-commentary"}, {"provider_call_id":"other-call"},
                      {"tool_event_id":str(uuid4())}, {"tool_revision":2}, {"tool_payload_id":str(uuid4())},
                      {"request_revision":self.snapshot.request.revision + 1},
                      {"connection_id":"retired-connection"}, {"evidence":None}):
            with self.subTest(continuation=patch):
                self.assertEqual(self.evaluate(continuation=self.snapshot.continuation | patch)["reason"],
                                 "explicit_tool_continuation_unavailable")

    def test_terminal_output_is_separate_from_strict_association_resolution(self):
        output = self.snapshot.output
        for patch in ({"finality":"provisional"}, {"content":output.content | {"completion_state":"partial"}},
                      {"content":output.content | {"completion_state":"interrupted"}}):
            with self.subTest(output=patch):
                self.assertNotEqual(self.evaluate(output=output.model_copy(update=patch))["status"], "ready")

    def test_exact_manual_admission_no_nearest_latest_or_cross_connection_join(self):
        self.assertEqual(self.evaluate(admission=None)["reason"], "manual_commit_admission_unavailable")
        for key in ("run_id", "invocation_id", "manifest_sha256", "session_id", "audio_session_id", "connection_id",
                    "segment_id", "client_event_id", "request_id", "turn_generation", "target_input_ordinal"):
            old = self.snapshot.admission[key]
            with self.subTest(field=key):
                self.assertNotEqual(self.evaluate(admission=self.snapshot.admission | {
                    key:old+1 if type(old) is int else "wrong"})["status"], "ready")

    def test_fresh_nonce_deadline_and_bounded_cardinality(self):
        new_request = replace(self.request, nonce=uuid4().hex)
        self.assertEqual(self.evaluate(control=new_request)["reason"], "snapshot_scope_or_nonce")
        self.assertEqual(self.evaluate(snapshot=replace(self.snapshot, nonce=new_request.nonce), control=new_request)["status"], "ready")
        self.assertEqual(self.evaluate(observed_at=self.request.issued_at-timedelta(microseconds=1))["reason"],
                         "snapshot_not_fresh_for_request")
        self.assertEqual(evaluate_memory_readiness(self.request, self.snapshot, now=self.request.expires_at)["status"], "expired")
        self.assertEqual(self.evaluate(sources=(self.source,self.source))["reason"], "ambiguous_source_or_retain")
        self.assertEqual(self.evaluate(retains=self.snapshot.retains*2)["reason"], "ambiguous_source_or_retain")
        for _ in range(3):
            self.assertEqual(self.evaluate(), self.evaluate())  # No internal retry/release state.
        with self.assertRaises(ValueError):
            replace(self.request, expires_at=self.request.issued_at+timedelta(seconds=121))

    def test_explicit_finalized_input_predicate_does_not_claim_memory_or_playback(self):
        request = replace(self.request, canonical_requirement="finalized_input")
        result = self.evaluate(control=request, tool_outcome=None, memory_event=None, sources=(), retains=(), continuation=None)
        self.assertEqual(result["status"], "ready")
        self.assertNotIn("retained_source", result)
        self.assertEqual(result["playback"], "not_evaluated")


# UUID is kept explicit in the fixture: these are actual typed canonical models,
# not row dictionaries whose fields bypass their normal constructors.
from uuid import UUID
