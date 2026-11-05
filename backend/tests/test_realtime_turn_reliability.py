"""Production-session regressions for GPT user-turn admission and settlement."""
from __future__ import annotations

import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, patch

import test_generic_visual_speech as gpt_fixture


AUDIO = base64.b64encode(b"\x00\x00" * 2400).decode()


class GPTTurnReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = gpt_fixture.GenericVisualSpeechTests()
        await self.h.asyncSetUp()
        self.addAsyncCleanup(self.h.asyncTearDown)
        self.session = self.h.session

    async def send(self, kind, **fields):
        await self.session._handle_client_event({"type": kind, **fields})

    async def append(self):
        await self.send("audio.input.append", audio_base64=AUDIO)

    async def test_create_before_commit_is_submitted_once_after_commit(self):
        await self.append()
        generation = self.session._turn_generation
        await self.send("response.create")
        self.assertFalse(self.h.creates())
        await self.send("audio.input.commit")
        self.assertEqual(len(self.h.creates()), 1)
        self.assertEqual(
            self.h.creates()[0]["response"]["metadata"]["jarvis_turn_generation"],
            str(generation),
        )
        await self.send("response.create")
        self.assertEqual(len(self.h.creates()), 1)

    async def test_repeated_early_requests_do_not_duplicate_the_committed_turn(self):
        await self.append()
        await self.send("response.create")
        await self.send("response.create")
        await self.send("audio.input.commit")
        self.assertEqual(len(self.h.creates()), 1)

    async def test_cancelled_early_request_is_not_inherited_by_new_audio(self):
        await self.append()
        await self.send("response.create")
        await self.send("response.cancel")
        await self.append()
        await self.send("audio.input.commit")
        self.assertFalse(self.h.creates())
        await self.send("response.create")
        self.assertEqual(len(self.h.creates()), 1)

    async def test_commit_alone_does_not_invent_a_response_request(self):
        await self.append()
        await self.send("audio.input.commit")
        self.assertFalse(self.h.creates())

    async def test_normal_commit_then_create_remains_single_submission(self):
        await self.append()
        await self.send("audio.input.commit")
        await self.send("response.create")
        await self.send("response.create")
        self.assertEqual(len(self.h.creates()), 1)

    async def test_old_commit_error_does_not_drop_newer_early_request(self):
        await self.append()
        await self.send("audio.input.commit")
        old_commit_id, old_generation = self.session._audio_commit_request
        await self.send("response.create")
        await self.append()
        await self.send("response.create")
        self.assertGreater(self.session._turn_generation, old_generation)
        await self.session._handle_openai_event({
            "type": "error",
            "error": {"event_id": old_commit_id, "code": "fixture_old_commit_error",
                      "message": "Old commit rejected."},
        })
        await self.send("audio.input.commit")
        self.assertEqual(len(self.h.creates()), 2)
        self.assertEqual(
            self.h.creates()[-1]["response"]["metadata"]["jarvis_turn_generation"],
            str(self.session._turn_generation),
        )

    async def test_deferred_preparation_does_not_block_the_next_user_input(self):
        entered = asyncio.Event()
        original_prepare = self.session.prepare_visual_response

        async def blocked_prepare(**_):
            entered.set()
            await asyncio.Future()

        self.session.prepare_visual_response = blocked_prepare
        self.session._client_receive_task = asyncio.current_task()
        await self.append()
        await self.send("response.create")
        await self.send("audio.input.commit")
        preparation = self.session._response_preparation_task
        await asyncio.wait_for(entered.wait(), 1)
        old_generation = self.session._turn_generation
        await self.append()
        await asyncio.gather(preparation, return_exceptions=True)
        self.assertGreater(self.session._turn_generation, old_generation)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertTrue(self.session._user_audio_pending)
        self.assertFalse(self.h.creates())
        self.session.prepare_visual_response = original_prepare
        await self.send("audio.input.commit")
        self.assertFalse(self.h.creates())
        await self.send("response.create")
        await asyncio.wait_for(self.session._response_preparation_task, 1)
        self.assertEqual(len(self.h.creates()), 1)

    async def test_preparation_failure_settles_the_turn_and_allows_new_input(self):
        original_prepare = self.session.prepare_visual_response
        self.session.prepare_visual_response = AsyncMock(
            side_effect=RuntimeError("fixture preparation failure"),
        )
        self.session._client_receive_task = asyncio.current_task()
        await self.append()
        await self.send("audio.input.commit")
        await self.send("response.create")
        await asyncio.gather(self.session._response_preparation_task, return_exceptions=True)
        self.assertIsNone(self.session._pending_response)
        self.assertFalse(self.session._normal_turn_pending)
        self.assertTrue(any(
            kind == "error" and payload.get("code") == "response_preparation_failed"
            for kind, payload in self.h.app_events
        ))
        self.session.prepare_visual_response = original_prepare
        await self.append()
        await self.send("audio.input.commit")
        await self.send("response.create")
        await asyncio.wait_for(self.session._response_preparation_task, 1)
        self.assertEqual(len(self.h.creates()), 1)

    async def test_start_timeout_is_explicit_and_does_not_replay_unknown_speech(self):
        with patch("app.realtime.session.OPENAI_RESPONSE_START_TIMEOUT_SECONDS", .001):
            await self.append()
            await self.send("audio.input.commit")
            await self.send("response.create")
            pending = self.session._pending_response
            await asyncio.wait_for(self.session._response_start_task, 1)
        self.assertIs(self.session._pending_response, pending)
        self.assertTrue(pending.submitted)
        self.assertTrue(any(
            kind == "error" and payload.get("code") == "response_start_timeout"
            for kind, payload in self.h.app_events
        ))
        await self.send("response.create")
        self.assertEqual(len(self.h.creates()), 1)
        await self.append()
        await self.send("audio.input.commit")
        await self.send("response.create")
        self.assertEqual(len(self.h.creates()), 2)

    async def test_late_cancelled_completion_cannot_settle_a_newer_turn(self):
        await self.append()
        await self.send("audio.input.commit")
        await self.send("response.create")
        await self.session._handle_openai_event({
            "type": "response.created",
            "response": {"id": "old-response",
                         "metadata": self.h.creates()[-1]["response"]["metadata"]},
        })
        await self.append()
        await self.send("audio.input.commit")
        await self.send("response.create")
        current = self.session._pending_response
        await self.session._handle_openai_event({
            "type": "response.done",
            "response": {"id": "old-response", "status": "cancelled", "output": []},
        })
        self.assertIs(self.session._pending_response, current)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertEqual(len(self.h.creates()), 2)

    async def test_deferred_request_and_release_keep_the_same_trace_generation(self):
        with patch.object(self.session._trace, "event") as trace:
            await self.append()
            generation = self.session._turn_generation
            await self.send("response.create")
            await self.send("audio.input.commit")
        relevant = [
            (call.args[0], call.kwargs.get("turn_generation"), call.kwargs.get("reason"))
            for call in trace.call_args_list
            if call.args[0] in {"response_deferred", "deferred_response_released"}
        ]
        self.assertEqual(relevant, [
            ("response_deferred", generation, "audio_commit_pending"),
            ("deferred_response_released", generation, "audio_committed"),
        ])

    async def test_cancel_before_preparation_starts_has_a_terminal_trace(self):
        self.session._client_receive_task = asyncio.current_task()
        with patch.object(self.session._trace, "event") as trace:
            await self.append()
            await self.send("audio.input.commit")
            await self.send("response.create")
            reservation = self.session._pending_response
            preparation = self.session._response_preparation_task
            await self.append()
            await asyncio.gather(preparation, return_exceptions=True)
        terminal = [
            call.kwargs for call in trace.call_args_list
            if call.args[0] == "response_not_submitted"
            and call.kwargs.get("request_id") == reservation.request_id
        ]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["status"], "cancelled")
        self.assertEqual(terminal[0]["turn_generation"], reservation.turn_generation)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertTrue(self.session._user_audio_pending)
        self.assertFalse(self.session._normal_response_requested)
