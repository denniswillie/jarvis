"""Frozen native lifecycle oracles never turn plausible prose into a pass."""
import unittest
import asyncio
import time
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from evals.second_brain.native_memory_lifecycle import load_cases, structural_checks
from evals.second_brain.native_memory_lifecycle import context_reset_notice, LifecyclePhone, lifecycle_answer


def grade(case, before, after, tools):
    return structural_checks(case, before, after, tools,
        mutating_tools={"correct_memory_source", "remember", "new_registered_mutation"})


class NativeLifecycleOracleTests(unittest.TestCase):
    def setUp(self):
        self.cases, self.digest = load_cases()
        self.case = next(case for case in self.cases if case["id"] == "precise_correction")
        self.original = {"revision": 1, "state": "active", "trust": "user_input",
                         "content": {"text": "Unchanged synthetic original"}, "original_content_retained": True}

    def test_frozen_case_set_and_oracle_identities(self):
        self.assertEqual(len(self.digest), 64)
        self.assertEqual({case["id"] for case in self.cases},
                         {"precise_correction", "ambiguous_correction", "new_event", "logical_forget", "shared_delete"})

    def test_correction_requires_actual_tool_and_exact_untargeted_preservation(self):
        before = {key: dict(self.original) for key in ("cedar-v1", "cedar-v2")}
        after = {key: dict(value) for key, value in before.items()}
        after["cedar-v1"] = self.original | {"revision": 2, "trust": "derived", "content": {"text": "Corrected"}}
        tools = [{"tool": "correct_memory_source", "result": {"status": "success"}}]
        self.assertTrue(all(grade(self.case, before, after, tools).values()))
        self.assertFalse(all(grade(self.case, before, after, []).values()))
        after["cedar-v2"]["content"] = {"text": "An unrelated change"}
        self.assertFalse(grade(self.case, before, after, tools)["untargeted_originals_unchanged"])

    def test_refused_or_faked_tool_does_not_prove_mutation_success(self):
        before = {key: dict(self.original) for key in ("cedar-v1", "cedar-v2")}
        checks = grade(self.case, before, before,
            [{"tool": "correct_memory_source", "result": {"status": "failed"}}])
        self.assertFalse(checks["real_correction_tool"])
        self.assertFalse(checks["one_versioned_replacement"])

    def test_ambiguity_requires_no_mutation_but_does_not_semantically_grade_answer(self):
        case = next(case for case in self.cases if case["id"] == "ambiguous_correction")
        before = {key: dict(self.original) for key in case["expected"]["unchanged"]}
        self.assertTrue(all(grade(case, before, before, []).values()))
        self.assertFalse(grade(case, before, before,
            [{"tool": "remember", "result": {"status": "success"}}])["no_accepted_mutation"])
        self.assertFalse(grade(case, before, before,
            [{"tool": "new_registered_mutation", "result": {"status": "success"}}])["no_accepted_mutation"])
        self.assertNotIn("semantic_pass", grade(case, before, before, []))

class LifecycleResetProtocolTests(unittest.IsolatedAsyncioTestCase):
    def notice(self, **changes):
        return {"type": "session.context_invalidated", "session_id": str(self.session_id),
            "audio_session_id": "fixture-audio", "context_reset": {
                "protocol_version": 1, "preserve_session": True, "id": str(self.reset_id)}, **changes}

    async def asyncSetUp(self):
        self.session_id, self.reset_id = uuid4(), uuid4()

    async def test_notice_requires_exact_identity_and_protocol(self):
        event = self.notice()
        def parse(candidate, **kwargs):
            return context_reset_notice(candidate, session_id=self.session_id,
                audio_session_id="fixture-audio", **kwargs)
        self.assertEqual(parse(event), str(self.reset_id))
        invalid = [
            event | {"audio_session_id": "other"}, event | {"session_id": str(uuid4())},
            event | {"type": "error"}, event | {"context_reset": None},
            event | {"context_reset": event["context_reset"] | {"protocol_version": True}},
            event | {"context_reset": event["context_reset"] | {"preserve_session": 1}},
            event | {"context_reset": event["context_reset"] | {"id": "not-a-uuid"}},
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                self.assertIsNone(parse(candidate))
        self.assertIsNone(parse(event, pending=str(uuid4())))

    async def test_phone_acknowledges_once_and_requires_matching_close(self):
        phone = LifecyclePhone(session_id=self.session_id, audio_session_id="fixture-audio", started=0)
        await phone.send_json(self.notice())
        await phone.send_json(self.notice())
        self.assertEqual(phone.input.qsize(), 1)
        ack = await phone.input.get()
        self.assertEqual(ack["type"], "session.context_cleared")
        self.assertEqual(ack["reset_id"], str(self.reset_id))
        await phone.send_json(self.notice(type="session.closed",
            context_reset={"protocol_version": 1, "preserve_session": True, "id": str(uuid4())}))
        self.assertFalse(phone.expected_reset_close)
        await phone.send_json(self.notice(type="session.closed"))
        self.assertTrue(phone.expected_reset_close)

    async def test_reset_reconnect_does_not_resubmit_speech_or_generate_response_request(self):
        phone = LifecyclePhone(session_id=self.session_id, audio_session_id="fixture-audio", started=time.monotonic())
        await phone.send_json(self.notice(type="session.closed"))
        ended_job = asyncio.create_task(asyncio.sleep(0))
        await ended_job
        connections = [(phone, None, None, ended_job)]
        pending = None
        async def connect():
            nonlocal pending
            replacement = LifecyclePhone(session_id=self.session_id, audio_session_id="new", started=phone.started)
            replacement.events.extend([{"type":"audio.output.delta", "response_id":"new-answer"},
                                       {"type":"response.done", "response_id":"new-answer"}])
            pending = asyncio.create_task(asyncio.sleep(60))
            connections.append((replacement, None, None, pending))
        try:
            with patch("evals.second_brain.native_memory_lifecycle.speak",
                       new=AsyncMock(return_value=time.monotonic())) as speech:
                result = await lifecycle_answer(connections, connect, None, "fixture", timeout=1)
            speech.assert_awaited_once_with(phone, None, "fixture")
            self.assertEqual(result["outcome"], "completed")
            self.assertEqual(result["connections"], 2)
            self.assertTrue(all(current.input.empty() for current, _, _, _ in connections))
        finally:
            if pending:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_reconnected_silence_is_failure_not_successful_mutation_acknowledgment(self):
        phone = LifecyclePhone(session_id=self.session_id, audio_session_id="fixture-audio", started=time.monotonic())
        job = asyncio.create_task(asyncio.sleep(60))
        try:
            with patch("evals.second_brain.native_memory_lifecycle.speak",
                       new=AsyncMock(return_value=time.monotonic())):
                result = await lifecycle_answer([(phone, None, None, job)], AsyncMock(), None, "fixture", timeout=.03)
            self.assertEqual(result["outcome"], "response_timeout")
            self.assertFalse(result["response_ids"])
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
