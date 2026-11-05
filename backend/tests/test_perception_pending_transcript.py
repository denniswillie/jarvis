"""Real perception capture with controlled storage, not SQL or provider evidence."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputCapture, request_document
from app.live_state.models import SessionScope
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.tasks.context_events import ContextRecord
from app.tasks.context_history import canonical_history_payload
from app.tasks.models import digest
from test_perception_brief_capture import _Storage


def capture_request(owner, session, when):
    return FastPerceptionRequest(owner_id=owner, session_id=session, window_id=uuid4(),
        frames=(FastPerceptionFrame(uuid4(), 1, when, "glasses", "image/jpeg", b"offline-only"),),
        previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
        behavior_instructions="Observe only", active_reminders=(), active_focus=None,
        recent_interrupts=(), requested_at=when)


def capture_context(owner, session):
    return SimpleNamespace(owner_id=owner, session_id=session,
                           recent_visual_interrupts=(), recent_audio_transcripts=())


class _TranscriptStorage(_Storage):
    def __init__(self, scope):
        super().__init__(scope)
        self.event_dependencies = []

    async def executemany(self, query, values):
        if "insert into jarvis_context_event_dependencies" in query:
            assert self.depth
            for owner, dependent, event, revision, parent, usage in values:
                assert owner == self.scope.owner_id and self.available(parent)
                self.event_dependencies.append((dependent, event, revision, parent, usage))
            return
        await super().executemany(query, values)


class PendingTranscriptCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 31, 8, 12, 32, tzinfo=UTC)
        self.scope = SessionScope(owner_id="pending-transcript-offline", session_id=uuid4(), area_id=uuid4())
        self.storage = _TranscriptStorage(self.scope)
        self.capture = ObservationInputCapture(self.storage.context)
        self.request = capture_request(self.scope.owner_id, self.scope.session_id, self.now)
        self.context = capture_context(self.scope.owner_id, self.scope.session_id)

    def record(self, *, text=None, finality="unavailable", availability="available",
               kind="user_transcript", revision=1, recorded_at=None):
        content = {"text": text, "input_id": "private-input-" + uuid4().hex,
                   "provider_item_id": "private-item-" + uuid4().hex,
                   "transcript_status": "pending" if text is None else "finalized"}
        parent = self.storage.original(content)
        self.storage.rows[parent]["availability"] = availability
        return ContextRecord(id=uuid4(), owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            sequence=revision, origin_sequence=1, source_namespace="voice:pending-transcript-fixture",
            source_event_id="input:" + uuid4().hex, revision=revision, supersedes_event_id=None,
            kind=kind, trust="assistant_output" if kind == "assistant_output" else "user_input",
            finality=finality, source_time=self.now - timedelta(seconds=1),
            recorded_at=recorded_at or self.now - timedelta(seconds=1), payload_id=parent,
            availability=availability, content_digest=digest(content), content=content, is_current=True)

    async def prepare(self, *records):
        self.storage.context.voice_history_in_transaction.return_value = {
            "records": list(records), "total": len(records)}
        return await self.capture.prepare(self.request, self.context)

    def assert_omitted(self, prepared, snapshot, record):
        document = request_document(prepared)
        serialized = json.dumps(document)
        for identity in (str(record.id), str(record.payload_id), record.source_event_id,
                         record.content["input_id"], record.content["provider_item_id"]):
            self.assertNotIn(identity, serialized)
        self.assertNotIn(record.payload_id, self.storage.parents.get(snapshot.payload_id, set()))
        self.assertFalse(any(row[0] == snapshot.payload_id and row[1] == record.id
                             for row in self.storage.event_dependencies))
        gaps = [gap for gap in prepared.input_context_gaps if gap["reason"] == "transcript_not_finalized"]
        self.assertEqual(gaps, [{"component": "recent_conversation", "reason": "transcript_not_finalized",
            "omitted_messages": 1,
            "meaning": "Missing transcript text is unknown; do not infer words or commands."}])

    async def test_old_shared_selection_includes_pending_but_perception_omits_it(self):
        pending = self.record()
        old_voice, old_parents, _ = canonical_history_payload({"records": [pending], "total": 1})
        self.assertEqual(old_parents, (pending.payload_id,))
        self.assertEqual(old_voice["records"][0]["content"], pending.content)
        self.assertEqual(pending.dependency_usage, "current")
        prepared, snapshot = await self.prepare(pending)
        self.assertEqual(prepared.voice_context["records"], [])
        self.assertEqual(prepared.voice_context["omitted_records"], 1)
        self.assert_omitted(prepared, snapshot, pending)
        self.assertFalse(any(event[0] == "lock_roots" for event in self.storage.events))
        await self.capture.validate(prepared, snapshot)

    async def test_null_placeholder_variants_do_not_depend_on_provider_status_label(self):
        for finality, availability in (("unavailable", "available"), ("provisional", "not_finalized")):
            for status in ("pending", "unavailable", None):
                with self.subTest(finality=finality, status=status):
                    pending = self.record(finality=finality, availability=availability)
                    content = dict(pending.content)
                    if status is None:
                        content.pop("transcript_status")
                    else:
                        content["transcript_status"] = status
                    pending = pending.model_copy(update={"content": content})
                    self.storage.rows[pending.payload_id]["content"] = content
                    prepared, snapshot = await self.prepare(pending)
                    self.assert_omitted(prepared, snapshot, pending)

    async def test_final_and_revised_consumed_text_keep_exact_current_dependencies(self):
        for finality, revision in (("final", 1), ("revised", 2)):
            with self.subTest(finality=finality):
                record = self.record(text="The synthetic marker is blue.", finality=finality, revision=revision)
                prepared, snapshot = await self.prepare(record)
                self.assertEqual(prepared.voice_context["records"][0]["content"], record.content)
                self.assertEqual(self.storage.parents[snapshot.payload_id], {record.payload_id})
                self.assertIn((snapshot.payload_id, record.id, revision, record.payload_id, "current"),
                              self.storage.event_dependencies)
                self.assertEqual(prepared.input_context_gaps, ())
                await self.capture.validate(prepared, snapshot)
                # Controlled storage models removal, not execution of SQL triggers.
                self.storage.rows[record.payload_id].update(availability="deleted", content=None)
                with self.assertRaises(SourceError) as raised:
                    await self.capture.validate(prepared, snapshot)
                self.assertEqual(raised.exception.code, "observation_input_removed")

    async def test_mixed_history_keeps_real_text_and_only_generic_missing_uncertainty(self):
        pending = self.record()
        final = self.record(text="Keep observing the marker.", finality="final")
        prepared, snapshot = await self.prepare(pending, final)
        self.assert_omitted(prepared, snapshot, pending)
        self.assertEqual([item["id"] for item in prepared.voice_context["records"]], [str(final.id)])
        self.assertEqual(self.storage.parents[snapshot.payload_id], {final.payload_id})

    async def test_nonnull_and_streaming_content_are_not_blanket_historical_switched(self):
        for kind, finality, usage in (("user_transcript", "unavailable", "current"),
                                     ("assistant_output", "provisional", "historical")):
            with self.subTest(kind=kind):
                record = self.record(text="Retained partial words.", kind=kind, finality=finality)
                prepared, snapshot = await self.prepare(record)
                self.assertEqual(prepared.voice_context["records"][0]["content"]["text"], record.content["text"])
                self.assertIn((snapshot.payload_id, record.id, 1, record.payload_id, usage),
                              self.storage.event_dependencies)

    async def test_late_finalized_text_is_not_backdated_into_earlier_frame(self):
        late = self.record(text="Late synthetic words.", finality="final",
                           recorded_at=self.now + timedelta(seconds=1))
        prepared, snapshot = await self.prepare(late)
        self.assertEqual(prepared.voice_context["records"], [])
        self.assertNotIn(late.payload_id, self.storage.parents.get(snapshot.payload_id, set()))
        self.assertEqual(self.storage.event_dependencies, [])

    async def test_missing_deleted_foreign_final_originals_never_reappear_as_text(self):
        for failure in ("missing", "deleted", "foreign"):
            with self.subTest(failure=failure):
                record = self.record(text="Do not resurrect this text.", finality="final")
                if failure == "missing":
                    del self.storage.rows[record.payload_id]
                elif failure == "deleted":
                    self.storage.rows[record.payload_id].update(availability="deleted", content=None)
                else:
                    self.storage.rows[record.payload_id]["owner"] = "other-owner"
                prepared, snapshot = await self.prepare(record)
                self.assertIsNone(prepared.voice_context["records"][0]["content"])
                self.assertNotIn(record.content["text"], json.dumps(request_document(prepared)))
                self.assertNotIn(record.payload_id, self.storage.parents.get(snapshot.payload_id, set()))
                self.assertFalse(any(row[0] == snapshot.payload_id for row in self.storage.event_dependencies))

    async def test_final_source_lock_and_session_guards_still_fail_closed(self):
        record = self.record(text="Canonical consumed text.", finality="final")
        self.storage.reject_bulk_lock = True
        with self.assertRaises(SourceError) as raised:
            await self.prepare(record)
        self.assertEqual(raised.exception.code, "observation_context_changed")
        self.storage.tasks.payload.assert_not_awaited()
        foreign_context = capture_context(self.scope.owner_id, uuid4())
        with self.assertRaises(SourceError) as raised:
            await self.capture.prepare(self.request, foreign_context)
        self.assertEqual(raised.exception.code, "observation_context_scope")
        self.storage.tasks.payload.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
