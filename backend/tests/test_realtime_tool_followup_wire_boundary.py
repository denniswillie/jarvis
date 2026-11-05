"""Native followup bytes through the installed route and real sideband writer.

External socket/storage I/O and domain tool effects are controlled. Session event
handlers, capture lookup, executor, continuation, Live State publication, source
guards, WebRTC wrapper and final JSON serialization are production components.
This is not evidence of a historical provider payload, adoption or hearing.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.experience.models import ToolExecution
from app.realtime.source_context import LiveContextError
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest
from app.tools.provenance import SourceBoundToolResult
from app.tools.registry import ToolDefinition
import test_voice_context_response_pipeline as pipeline


class CanonicalIO(ContextRepository):
    """Controlled persistence; the actual request_evidence query/decoder runs."""

    def __init__(self, owner, storage):
        self.owner, self.storage = owner, storage
        self.records, self.lookups = [], []
        super().__init__(SimpleNamespace(pool=self))

    @asynccontextmanager
    async def acquire(self):
        yield self

    async def fetchrow(self, query, owner, session, namespace, source):
        assert "order by e.revision desc limit 1" in query
        self.lookups.append((owner, session, namespace, source))
        matches = [row for row in self.records if (row.owner_id, row.session_id,
                   row.source_namespace, row.source_event_id) == (owner, session, namespace, source)]
        return max(matches, key=lambda row: row.revision).model_dump() if matches else None

    async def append(self, owner, event):
        assert owner == self.owner
        prior = [row for row in self.records if (row.source_namespace, row.source_event_id)
                 == (event.source_namespace, event.source_event_id)]
        sequence = len(self.records) + 1
        record = ContextRecord(id=uuid4(), owner_id=owner, session_id=event.session_id,
            sequence=sequence, origin_sequence=prior[0].origin_sequence if prior else sequence,
            source_namespace=event.source_namespace, source_event_id=event.source_event_id,
            revision=len(prior) + 1, supersedes_event_id=prior[-1].id if prior else None,
            kind=event.kind, trust=event.trust, finality=event.finality,
            source_time=event.source_time, recorded_at=datetime.now(timezone.utc),
            payload_id=uuid4(), availability="available", content_digest=digest(event.payload),
            content=event.payload, is_current=True)
        self.records.append(record)
        self.storage.dependencies.append(tuple((owner, root, record.payload_id)
                                               for root in event.source_payload_ids))
        return record

    async def append_in_transaction(self, connection, owner, event, **kwargs):
        return await self.append(owner, event)


class ExecutionIO:
    """Return the actual durable execution type, not an ad-hoc namespace."""

    def __init__(self):
        self.records = {}

    async def start_execution_with_status(self, **fields):
        identity = fields["provider_call_id"]
        if identity in self.records:
            return self.records[identity], False
        row = ToolExecution(id=uuid4(), **fields, result=None, error=None,
                            status="started", completed_at=None, created_at=fields["started_at"])
        self.records[identity] = row
        return row, True

    async def finalize_execution(self, *, owner_id, session_id, provider_call_id, **fields):
        old = self.records[provider_call_id]
        assert (old.owner_id, old.session_id) == (owner_id, session_id)
        self.records[provider_call_id] = row = replace(old, **fields)
        return row


class ToolFollowupWireBoundaryTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only the composition helpers; do not collect inherited test methods.
    cleanup = pipeline.VoiceContextResponsePipelineTests.cleanup
    until = pipeline.VoiceContextResponsePipelineTests.until
    rows = pipeline.VoiceContextResponsePipelineTests.rows
    observation = pipeline.VoiceContextResponsePipelineTests.observation
    expire_inside_adapter_guard = pipeline.VoiceContextResponsePipelineTests.expire_inside_adapter_guard
    mutations = pipeline.VoiceContextResponsePipelineTests.mutations
    creates = pipeline.VoiceContextResponsePipelineTests.creates

    async def asyncSetUp(self):
        await pipeline.VoiceContextResponsePipelineTests.asyncSetUp(self)
        self.canonical = CanonicalIO(self.scope.owner_id, self.storage)
        self.capture = SessionContextCapture(self.canonical, owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, provider="openai",
            audio_session_id=self.session.context.audio_session_id)
        self.session._context_capture = self.capture
        self.session.tool_executor._context_capture = self.capture
        self.executions = ExecutionIO()
        self.session.tool_executor._execution_repository = self.executions
        self.addAsyncCleanup(self.capture.close)
        self.calls, self.receipts, self.wire = [], {}, []
        self.input_text = "Remember the revised setting; retain the previous record."
        trace_capture = patch.object(self.session._trace, "event", wraps=self.session._trace.event)
        self.trace_events = trace_capture.start()
        self.addCleanup(trace_capture.stop)
        original_send = self.provider.send

        async def capture_serialized(raw):
            payload = json.loads(raw)
            self.wire.append((raw, self.storage.is_in_transaction(),
                              self.session._pending_response))
            await original_send(raw)

        self.provider.send = capture_serialized

        async def remember(arguments):
            self.calls.append("remember")
            return {"status": "success", "memory_status": "stored",
                    "source": {"ingestion_status": "pending"}, "replayed": False,
                    "receipt_detail": "private-result-body"}

        async def bind_result(result, context):
            request = self.session._active_responses["tool-response"].tool_followup.request
            record = await context.capture.bound_tool_outcome(self.storage, context.call_id,
                context.tool_name, result, (request.payload_id,))
            self.receipts[context.call_id] = record
            return SourceBoundToolResult(result, record)

        async def unavailable(arguments):
            self.calls.append("unavailable")
            return {"status": "failed", "error": {"code": "fixture_unavailable",
                                                     "message": "private-failure-body"}}

        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        self.session.tool_registry.register(ToolDefinition("remember", "Store an explicit request.",
            schema, remember, mutates_state=True, result_capture=bind_result, capture_after_commit=True))
        self.session.tool_registry.register(ToolDefinition("unavailable", "Unavailable fixture action.",
            schema, unavailable))

    async def tool_turn(self, *, mixed=False):
        # Actual provider event admission/capture, not a synthetic pending state.
        await self.provider.inject("input_audio_buffer.speech_started", item_id="original-input")
        await self.provider.inject("input_audio_buffer.speech_stopped", item_id="original-input")
        await self.provider.inject("input_audio_buffer.committed", item_id="original-input")
        await self.provider.inject("conversation.item.input_audio_transcription.completed",
            item_id="original-input", transcript=self.input_text)
        await self.until(lambda: len(self.creates()) == 1)
        await self.provider.inject("response.created", response={"id": "tool-response",
            "metadata": self.creates()[0]["response"]["metadata"]})
        await self.until(lambda: "tool-response" in self.session._active_responses)
        self.tool_items = []
        for name in ("remember", "unavailable") if mixed else ("remember",):
            item = {"id": "item-" + name, "type": "function_call", "name": name,
                    "call_id": "call-" + name, "arguments": "{}"}
            self.tool_items.append(item)
            await self.provider.inject("response.output_item.done", response_id="tool-response", item=item)
        await self.until(lambda: len(self.calls) == len(self.tool_items)
            and not self.session._ordinary_tool_tasks)
        state = self.session._active_responses["tool-response"]
        self.assertIsNotNone(state.tool_followup.request)
        self.assertEqual(len(state.tool_followup.receipts), len(self.tool_items))
        self.assertEqual(self.executions.records["call-remember"].status, "completed")
        self.assertIsInstance(self.receipts["call-remember"], ContextRecord)
        self.assertEqual(len(self.canonical.lookups), len(self.tool_items))
        self.original = state.tool_followup.request
        return state

    async def finish_tool_response(self):
        await self.provider.inject("response.done", response={"id": "tool-response",
            "status": "completed", "output": self.tool_items})

    async def deferred_quiet_replacement(self, *, replace_candidate=False):
        entered, release = asyncio.Event(), asyncio.Event()
        scope_entered = asyncio.Event()
        self.addCleanup(release.set)
        old_refs = set(self.records)
        read_many = self.binder.sources._read_many
        blocked = False

        async def blocked_read(connection, owner, area, refs):
            nonlocal blocked
            if not blocked and any(ref not in old_refs for ref in refs):
                blocked = True
                entered.set()
                await release.wait()
            return await read_many(connection, owner, area, refs)

        self.binder.sources._read_many = blocked_read
        self.now += timedelta(seconds=2)  # Application selection clock only.
        await self.observation("quiet-candidate-one")
        await asyncio.wait_for(entered.wait(), 1)
        if replace_candidate:
            await self.observation("quiet-candidate-newest")
        scope = self.runtime._normal_turn_scope_guard

        @asynccontextmanager
        async def held_normal_scope():
            scope_entered.set()
            await release.wait()
            async with scope() as connection:
                yield connection

        # Hold the actual request scope, not optional refresh. The latter must
        # no longer gate an independently evidenced tool-result continuation.
        self.runtime._normal_turn_scope_guard = held_normal_scope
        await self.finish_tool_response()
        await asyncio.wait_for(scope_entered.wait(), 2)
        pending = self.session._pending_response
        self.assertIsNotNone(pending)
        self.assertIsNone(getattr(self.session, "_context_waiting_request_id", None))
        self.assertEqual(pending.tool_followup.request, self.original)
        self.assertFalse(pending.submitted)
        self.assertEqual(len(self.creates()), 1)
        return pending, release

    async def assert_followup_wire(self, pending=None):
        await self.until(lambda: len(self.creates()) == 2 or self.session._closed.is_set())
        self.assertFalse(self.session._closed.is_set(), self.session._close_reason)
        final = self.creates()[1]
        raw, guarded, reservation = next(row for row in self.wire
            if json.loads(row[0]).get("event_id") == final["event_id"])
        self.assertIsInstance(raw, str)
        self.assertTrue(guarded)
        if pending is not None:
            self.assertIs(reservation, pending)
            self.assertEqual(final["event_id"], pending.request_id)
        self.assertEqual(reservation.tool_followup.request, self.original)
        text = final["response"]["instructions"]
        marker = "Current tool-followup delivery:\n"
        self.assertEqual(text.count(marker), 1)
        directive = text.split(marker, 1)[1]
        data = json.loads(directive[directive.index('{"request":'):])
        self.assertEqual(data["request"], {"provider_item_id": "original-input",
            "canonical_event_id": str(self.original.event_id), "revision": self.original.revision,
            "quoted_user_content": {"speaker": "USER", "text": self.original.user_text,
                "status": self.original.user_content_status}})
        self.assertEqual(data["receipts"][0], {"call_id": "call-remember", "tool": "remember",
            "status": "success", "canonical_event_id": str(self.receipts["call-remember"].id),
            "local_effect_reported": False, "owner_outcome": {"memory_status": "stored",
                "source_ingestion_status": "pending", "replayed": False}})
        self.assertEqual(data["evidence_boundary"], "tool_outputs_written_not_provider_adoption_or_hearing")
        self.assertLessEqual(len((marker + directive).encode()), 4096)
        self.assertIn("Runtime context:", text)
        self.assertNotIn("private-result-body", text)
        self.assertNotIn("private-failure-body", text)
        self.assertNotIn(self.input_text, repr(reservation))
        self.assertNotIn(self.input_text, repr(reservation.tool_followup))
        self.assertNotIn(self.input_text, repr(self.executions.records))
        self.assertNotIn(self.input_text, json.dumps(self.storage.payloads, default=str))
        self.assertNotIn(self.input_text, repr(self.trace_events.call_args_list))
        self.assertNotIn("tool_choice", final["response"])
        self.assertNotIn("input", final["response"])
        snapshot = self.session._source_context.snapshot_for(final["event_id"])
        self.assertTrue({self.original.payload_id, self.receipts["call-remember"].payload_id}
                        .issubset(snapshot.sources))
        output_index = next(i for i, event in enumerate(self.provider.writes)
            if event.get("item", {}).get("call_id") == "call-remember"
            and event["item"]["type"] == "function_call_output")
        self.assertLess(output_index, self.provider.writes.index(final))
        result = json.loads(self.provider.writes[output_index]["item"]["output"])
        self.assertEqual((result["memory_status"], result["source"]["ingestion_status"]), ("stored", "pending"))
        self.assertEqual(self.calls.count("remember"), 1)
        self.assertFalse(self.session._context_revoked)
        return data

    async def test_ordinary_executor_to_native_serialized_followup(self):
        await self.tool_turn()
        await self.finish_tool_response()
        await self.assert_followup_wire()

    async def test_mixed_receipts_survive_context_wait_and_newest_candidate_replacement(self):
        await self.tool_turn(mixed=True)
        pending, release = await self.deferred_quiet_replacement(replace_candidate=True)
        release.set()
        data = await self.assert_followup_wire(pending)
        self.assertEqual(data["receipts"][1], {"call_id": "call-unavailable", "tool": "unavailable",
            "status": "failed", "canonical_event_id": None, "local_effect_reported": False,
            "owner_outcome": {}})
        quiet = [event for event in self.provider.writes
                 if event["type"] == "conversation.item.create" and
                 event.get("item", {}).get("type") == "message"]
        self.assertFalse(any("quiet-candidate-newest" in json.dumps(event) for event in quiet))
        self.assertFalse(any("quiet-candidate-one" in json.dumps(event) for event in quiet))
        self.assertIn("Current Live State briefing is temporarily unavailable", self.creates()[1]["response"]["instructions"])

    async def test_utf8_user_content_survives_actual_guarded_wire(self):
        self.input_text = "Remember: caf\u00e9, not th\u00e9; keep the previous record. \u65e7\u8a18\u9332"
        await self.tool_turn()
        pending, release = await self.deferred_quiet_replacement()
        release.set()
        data = await self.assert_followup_wire(pending)
        self.assertEqual(data["request"]["quoted_user_content"]["text"], self.input_text)

    async def test_oversized_user_content_is_wholly_omitted_not_rewritten(self):
        self.input_text = "\u65e7\u8a18\u9332" * 500
        await self.tool_turn()
        await self.finish_tool_response()
        data = await self.assert_followup_wire()
        self.assertEqual(data["request"]["quoted_user_content"],
            {"speaker": "USER", "text": None, "status": "omitted_byte_budget"})

    async def test_prewrite_source_refresh_preserves_same_followup_without_tool_replay(self):
        await self.tool_turn()
        original_guard = self.session._source_context.guard
        entered = []
        requests = []

        @asynccontextmanager
        async def source_addition(snapshot):
            pending = self.session._pending_response
            self.assertIsNotNone(pending)
            self.assertIs(self.session._source_context.snapshot_for(pending.request_id), snapshot)
            requests.append(pending)
            entered.append(snapshot)
            if len(entered) == 1:
                self.session._source_context.observe((uuid4(),))
            async with original_guard(snapshot):
                yield

        self.session._source_context.guard = source_addition
        await self.finish_tool_response()
        await self.assert_followup_wire()
        self.assertGreaterEqual(len(entered), 2)
        for pending in requests:
            self.assertIs(pending, requests[0])
        self.assertEqual(self.creates()[1]["event_id"], requests[0].request_id)
        self.assertIs(self.session._source_context.snapshot_for(requests[0].request_id), entered[-1])
        self.assertNotEqual(entered[0].payload_id, entered[-1].payload_id)
        self.assertLess(entered[0].version, entered[-1].version)
        self.assertTrue(set(entered[0].sources).issubset(entered[-1].sources))
        self.assertEqual(len(self.canonical.lookups), 1)

    async def test_withdrawn_original_after_tool_result_never_writes_followup(self):
        await self.tool_turn()
        pending, release = await self.deferred_quiet_replacement()
        self.storage.rejected.add(self.original.payload_id)
        release.set()
        await self.until(lambda: self.session._context_revoked or self.session._closed.is_set())
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(pending.submitted)
        self.assertEqual(self.calls, ["remember"])

    async def test_new_user_generation_cancels_wait_without_stale_followup(self):
        await self.tool_turn()
        pending, release = await self.deferred_quiet_replacement()
        generation = self.session._turn_generation
        await self.provider.inject("input_audio_buffer.speech_started", item_id="new-user-input")
        await self.until(lambda: self.session._turn_generation > generation)
        release.set()
        await self.until(lambda: getattr(self.session, "_context_waiting_request_id", None) is None)
        self.assertFalse(pending.submitted)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.calls, ["remember"])
