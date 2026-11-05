"""Owner/CAS/receipt behavior with controlled storage; not PostgreSQL or media."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.attention.quiet import OwnerQuietRepository, QuietCommand
from app.preferences.repository import PreferenceRepository
from app.preferences.tools import PreferenceToolModule
from app.realtime.instructions import build_tool_followup_instructions, TOOL_FOLLOWUP_MAX_BYTES
from app.realtime.models import RealtimeToolFollowup, RealtimeToolReceipt, RealtimeToolRequest
from app.realtime.telemetry import RealtimeTrace, safe_fields
from app.tasks.context_capture import SessionContextCapture
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import ToolInvocationContext
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
import test_attention_quiet as quiet_fixture
import test_realtime_tool_followup_delivery as delivery_fixture
import test_realtime_tool_followup_wire_boundary as wire_fixture


class DirectQuietResumeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = quiet_fixture.LedgerFixture()
        self.repo = OwnerQuietRepository(self.pool)
        self.owner, self.session = "offline-direct-quiet", uuid4()
        original = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.pool.authorize(self.owner, self.session, original)
        self.accepted = await self.repo.save(self.owner, self.session, original,
            QuietCommand(action="activate", expected_revision=0))
        capture = SessionContextCapture(SimpleNamespace(), owner_id=self.owner, session_id=self.session,
            provider="openai", audio_session_id="direct-quiet")
        self.record = delivery_fixture._record(capture,
            content={"provider_item_id": "resume-input", "text": "You can interrupt again now."})
        self.source = ToolInvocationContext(request_event_id=self.record.id, request_revision=self.record.revision)
        self.pool.authorize(self.owner, self.session, self.source)
        self.capture = SimpleNamespace(request_for_call=AsyncMock(return_value=self.record), tool_outcome=Mock())
        self.enforce = AsyncMock(return_value=0)
        module = PreferenceToolModule(PreferenceRepository(self.pool))
        module.bind_attention_service(SimpleNamespace(_closed=False, enforce_owner_quiet=self.enforce))
        self.registry = create_tool_registry(modules=(module,),
            context=ToolBuildContext(settings=None, owner_id=self.owner, session_id=self.session))
        self.executions = wire_fixture.ExecutionIO()
        self.executor = JarvisToolExecutor(self.registry, owner_id=self.owner, session_id=self.session,
            context_capture=self.capture, execution_repository=self.executions)

    async def resume(self, revision=1, call="resume-call"):
        return await self.executor.execute("set_attention_quiet",
            {"action": "resume", "expected_revision": revision}, provider_call_id=call)

    async def test_known_revision_direct_resume_never_reads_get_tool(self):
        with patch.object(OwnerQuietRepository, "get", new_callable=AsyncMock,
                          side_effect=AssertionError("get is not required")) as get:
            result = await self.resume()
        get.assert_not_awaited()
        self.assertEqual(result["accepted_revision"], 2)
        self.assertEqual(result["quiet"]["effective_state"], "resumed")
        self.assertEqual(result["quiet"]["episode"]["episode_id"], self.accepted["quiet"]["episode"]["episode_id"])
        self.assertFalse(result["delivery_propagation"]["playback_stop_confirmed"])
        self.assertEqual(len(self.pool.rows), 2)
        self.capture.request_for_call.assert_awaited_once_with("resume-call")
        self.enforce.assert_awaited_once_with(self.owner)
        self.assertFalse(self.pool.lock.locked())

    async def test_conflict_does_not_read_retry_substitute_revision_or_propagate(self):
        with patch.object(OwnerQuietRepository, "get", new_callable=AsyncMock) as get:
            result = await self.resume(revision=0)
        self.assertEqual(result["error"]["code"], "revision_conflict")
        self.assertEqual(len(self.pool.rows), 1)
        self.enforce.assert_not_awaited()
        get.assert_not_awaited()
        self.assertEqual(sum("attention_quiet:current" in sql for sql, _ in self.pool.calls), 2)

    async def test_completed_call_replay_has_no_new_revision_lookup_or_propagation(self):
        first = await self.resume()
        before = len(self.pool.calls)
        replay = await self.resume()
        self.assertEqual(replay, first)
        self.assertEqual(len(self.pool.calls), before)
        self.assertEqual(len(self.pool.rows), 2)
        self.assertEqual(self.capture.request_for_call.await_count, 1)
        self.assertEqual(self.enforce.await_count, 1)

    async def test_withdrawn_foreign_and_corrected_request_cannot_resume(self):
        for failure in ("withdrawn", "foreign_owner", "foreign_session", "corrected_revision"):
            with self.subTest(failure=failure):
                self.pool.sources.clear()
                owner = "other" if failure == "foreign_owner" else self.owner
                session = uuid4() if failure == "foreign_session" else self.session
                source = replace(self.source, request_revision=2) if failure == "corrected_revision" else self.source
                if failure != "withdrawn":
                    self.pool.authorize(owner, session, source)
                result = await self.resume(call=failure)
                self.assertEqual(result["error"]["code"], "request_evidence_unavailable")
                self.assertEqual(len(self.pool.rows), 1)
                self.enforce.assert_not_awaited()
        source_sql = next(sql for sql, _ in self.pool.calls if "attention_quiet:authorized_source" in sql)
        for predicate in ("s.status='active'", "p.availability='available'", "newer.supersedes_event_id=e.id", "for share of p"):
            self.assertIn(predicate, source_sql)

    async def test_propagation_failure_keeps_canonical_success_but_not_stop_proof(self):
        self.enforce.side_effect = RuntimeError("PRIVATE_TRANSPORT_MESSAGE")
        result = await self.resume()
        receipt = RealtimeToolReceipt.from_result("resume-call", "set_attention_quiet", result)
        fields = dict(receipt.owner_outcome)
        self.assertEqual(receipt.status, "success")
        self.assertEqual(fields["quiet_accepted_revision"], 2)
        self.assertEqual(fields["quiet_delivery_propagation"], "unconfirmed")
        self.assertFalse(fields["quiet_playback_stop_confirmed"])
        self.assertNotIn("PRIVATE_", repr(result))

    async def test_same_source_replay_distinguishes_old_acceptance_from_new_snapshot(self):
        await self.resume()
        newer = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.pool.authorize(self.owner, self.session, newer)
        await self.repo.save(self.owner, self.session, newer, QuietCommand(action="activate", expected_revision=2))
        replay = await self.resume(call="distinct-provider-call")
        fields = dict(RealtimeToolReceipt.from_result("distinct-provider-call", "set_attention_quiet", replay).owner_outcome)
        self.assertEqual((fields["quiet_accepted_revision"], fields["quiet_revision"]), (2, 3))
        self.assertEqual(fields["quiet_effective_state"], "quiet")
        self.assertTrue(fields["replayed"])
        self.assertEqual(len(self.pool.rows), 3)

    def test_guidance_is_conditional_without_server_current_substitution(self):
        description = self.registry.get("set_attention_quiet").description
        self.assertIn("only when no such revision is known", description)
        self.assertIn("do not automatically read and replay", description)
        self.assertIn("Briefly confirm the accepted user-facing state", description)
        self.assertIn("do not routinely recite revisions, provenance, checked propagation", description)
        self.assertIn("Explain actual failed or unconfirmed effects", description)
        self.assertIn("Never imply pending notifications were delivered or heard", description)
        schema = self.registry.get("set_attention_quiet").input_schema
        self.assertIn("expected_revision", schema["required"])


class QuietFollowupMetadataTests(unittest.TestCase):
    def setUp(self):
        capture = SessionContextCapture(SimpleNamespace(), owner_id="owner", session_id=uuid4(),
            provider="openai", audio_session_id="quiet-metadata")
        record = delivery_fixture._record(capture, content={"provider_item_id": "resume-input", "text": "PRIVATE_USER Resume."})
        self.request = RealtimeToolRequest.from_record(record, owner_id="owner", session_id=capture.session_id,
            turn_generation=1, source_namespace=capture.namespace)
        self.result = {"status": "success", "quiet": {"scope": "owner_unsolicited_attention", "revision": 2,
            "effective_state": "resumed", "suppresses_unsolicited": False, "private": "PRIVATE_STATE"},
            "accepted_revision": 2, "request_replayed": False,
            "delivery_propagation": {"state": "unconfirmed", "playback_stop_confirmed": False}}
        self.trace = RealtimeTrace(provider="openai", audio_session_id="quiet-metadata", jarvis_session_id=capture.session_id)

    def anchor(self, result=None, tool="set_attention_quiet"):
        receipt = RealtimeToolReceipt.from_result("resume-call", tool, result if result is not None else self.result)
        return build_tool_followup_instructions(RealtimeToolFollowup(self.request, (receipt,)))

    def test_quiet_fields_require_normal_tool_success_scope_and_exact_types(self):
        for tool, result in (("unrelated", self.result), ("set_attention_quiet", {**self.result, "status": "failed"}),
                             ("set_attention_quiet", {**self.result, "quiet": {**self.result["quiet"], "scope": "foreign"}})):
            with self.subTest(tool=tool, status=result["status"]):
                self.assertEqual(RealtimeToolReceipt.from_result("call", tool, result).owner_outcome, ())
        invalid = {**self.result, "accepted_revision": True, "request_replayed": "false",
            "quiet": {**self.result["quiet"], "revision": 2.5, "effective_state": "PRIVATE_VALUE", "suppresses_unsolicited": 0},
            "delivery_propagation": {"state": "PRIVATE_PROPAGATION", "playback_stop_confirmed": "false"}}
        self.assertEqual(RealtimeToolReceipt.from_result("call", "set_attention_quiet", invalid).owner_outcome, ())

    def test_metadata_digest_fields_and_privacy_match_rendered_anchor(self):
        anchor = self.anchor()
        instructions = "PRIVATE_RUNTIME" + anchor
        with patch("app.realtime.telemetry.LOGGER.info") as log:
            self.trace.tool_followup_write(request_id="request", anchor=anchor, instructions=instructions)
        row = json.loads(log.call_args.args[1])
        self.assertEqual(row["event"], "tool_followup_write")
        self.assertTrue(row["anchor_present"])
        self.assertEqual(row["anchor_status"], "available")
        self.assertEqual(row["instructions_sha256"], hashlib.sha256(instructions.encode()).hexdigest())
        self.assertEqual(row["outcome_fields"], sorted(dict(RealtimeToolReceipt.from_result("resume-call", "set_attention_quiet", self.result).owner_outcome)))
        self.assertNotIn("PRIVATE_", json.dumps(row))
        self.assertNotIn("resumed", json.dumps(row))
        self.assertIsNone(RealtimeToolReceipt.from_result("resume-call", "set_attention_quiet", self.result).event_id)

    def test_brief_success_guidance_preserves_qualified_quiet_receipt_fields(self):
        result = {**self.result, "delivery_propagation": {"state": "checked", "playback_stop_confirmed": False}}
        anchor = self.anchor(result)
        data = json.loads(anchor[anchor.index('{"request":'):])
        self.assertIn("For a successful quiet-state change, briefly confirm its accepted user-facing state", anchor)
        self.assertIn("Do not routinely recite revisions, provenance, checked propagation or missing playback receipts", anchor)
        self.assertIn("a known lack of playback evidence alone is not an operational failure", anchor)
        self.assertIn("Never imply pending notifications were delivered or heard", anchor)
        self.assertEqual(data["receipts"][0]["owner_outcome"], {
            "quiet_effective_state": "resumed", "quiet_revision": 2, "quiet_suppresses_unsolicited": False,
            "quiet_accepted_revision": 2, "replayed": False, "quiet_delivery_propagation": "checked",
            "quiet_playback_stop_confirmed": False})
        self.assertEqual(data["request"]["quoted_user_content"]["text"], self.request.user_text)
        self.assertLessEqual(len(anchor.encode()), TOOL_FOLLOWUP_MAX_BYTES)

    def test_concise_guidance_does_not_hide_unconfirmed_or_failed_effects(self):
        accepted = RealtimeToolReceipt.from_result("accepted", "set_attention_quiet", self.result)
        uncertain = RealtimeToolReceipt.from_result("uncertain", "set_attention_quiet", {
            "status": "failed", "error": {"code": "tool_receipt_unavailable_after_commit",
                "local_effect_reported": True, "message": "PRIVATE_ERROR_DETAIL"}})
        anchor = build_tool_followup_instructions(RealtimeToolFollowup(self.request, (accepted, uncertain)))
        data = json.loads(anchor[anchor.index('{"request":'):])
        self.assertIn("Keep actual failed or unconfirmed effects explicit when relevant to the requested outcome", anchor)
        self.assertIn("Do not replay a mutation", anchor)
        self.assertEqual(data["receipts"][0]["owner_outcome"]["quiet_delivery_propagation"], "unconfirmed")
        self.assertFalse(data["receipts"][0]["owner_outcome"]["quiet_playback_stop_confirmed"])
        self.assertEqual(data["receipts"][1]["status"], "failed")
        self.assertTrue(data["receipts"][1]["local_effect_reported"])
        self.assertEqual(data["receipts"][1]["owner_outcome"], {"error_code": "tool_receipt_unavailable_after_commit"})
        self.assertNotIn("PRIVATE_ERROR_DETAIL", anchor)

    def test_sanitizer_rejects_unknown_metadata_and_raw_values(self):
        self.assertEqual(safe_fields({"anchor_status": "PRIVATE", "outcome_fields": ["quiet_revision", "PRIVATE"],
            "instructions_sha256": "PRIVATE", "anchor_present": 1, "instructions": "PRIVATE"}), {})
        self.assertEqual(safe_fields({"outcome_fields": ["quiet_revision"] * 20}), {})

    def test_logging_failure_and_malformed_anchor_do_not_escape(self):
        with patch("app.realtime.telemetry.LOGGER.info", side_effect=RuntimeError("PRIVATE")):
            self.trace.tool_followup_write(request_id="request", anchor=self.anchor(), instructions="private")
        with patch("app.realtime.telemetry.LOGGER.info") as log:
            self.trace.tool_followup_write(request_id="request", anchor="malformed", instructions="private")
            self.trace.tool_followup_write(request_id="request", anchor="x" * 4097, instructions="private")
        log.assert_not_called()

    def test_bounded_diagnostics_include_one_truncation_and_absent_anchor(self):
        with patch("app.realtime.telemetry.MAX_LATENCY_RECORDS", 2), patch("app.realtime.telemetry.LOGGER.info") as log:
            for _ in range(6):
                self.trace.tool_followup_write(request_id="request", anchor="", instructions="private")
        rows = [json.loads(call.args[1]) for call in log.call_args_list]
        self.assertEqual([row["event"] for row in rows], ["tool_followup_write"] * 2 + ["latency_diagnostics_truncated"])
        self.assertFalse(rows[0]["anchor_present"])
        self.assertEqual(rows[0]["anchor_status"], "absent")

    def test_budget_drops_whole_receipts_and_diagnostic_names_match_actual_render(self):
        request = replace(self.request, user_text="x" * 1022)
        delivery = RealtimeToolFollowup(request)
        for index in range(16):
            delivery = delivery.append(RealtimeToolReceipt.from_result(f"c{index}_" + "x" * 150, "set_attention_quiet", self.result))
        anchor = build_tool_followup_instructions(delivery)
        data = json.loads(anchor[anchor.index('{"request":'):])
        self.assertLessEqual(len(anchor.encode()), TOOL_FOLLOWUP_MAX_BYTES)
        self.assertGreater(len(data["receipts"]), 0)
        self.assertEqual(len(data["receipts"]) + data["omitted_receipts"], 16)
        with patch("app.realtime.telemetry.LOGGER.info") as log:
            self.trace.tool_followup_write(request_id="request", anchor=anchor, instructions=anchor)
        fields = sorted({key for row in data["receipts"] for key in row["owner_outcome"]})
        self.assertEqual(json.loads(log.call_args.args[1])["outcome_fields"], fields)
