"""Production-session reset history must retain context without inventing heard speech."""
from __future__ import annotations

import json
from dataclasses import replace
import unittest
from unittest.mock import patch

import test_gemini_audio_session as fixture


class GeminiContinuityHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = fixture.GeminiAudioSessionTests()
        await self.h.asyncSetUp()
        self.addAsyncCleanup(self.h.asyncTearDown)
        await self.h.start()
        await self.h.append()
        await self.h.commit()
        self.session = self.h.session

    async def user_transcript(self, text="Which line should I type next?"):
        await self.h.wires[-1].publish({"inputTranscription": {"text": text}})
        await fixture.until(lambda: self.session._turns["a"].input_finalized)

    def incomplete_records(self, history):
        records = []
        for item in history:
            try:
                value = json.loads(item["parts"][0]["text"])
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("kind") == "jarvis_incomplete_exchange":
                records.append(value)
        return records

    async def test_reset_keeps_interrupted_step_but_not_late_unforwarded_tail(self):
        await self.user_transcript()
        await self.h.answer(text="Next type elif target greater than arr mid.", terminal=False)
        await fixture.until(lambda: bool(self.h.events("audio.output.delta")))
        await self.session._cancel_output()
        await self.h.wires[0].publish({
            "outputTranscription": {"text": "UNFORWARDED_TAIL_DO_NOT_RESTORE"},
            "interrupted": True, "turnComplete": True,
        })
        await fixture.until(lambda: self.session._turns["a"].output.completed)
        await self.session._reset_before_new_input(reason="fixture_fresh_connection")
        history = self.h.wires[-1].connect.await_args.kwargs["history"]
        records = self.incomplete_records(history)
        self.assertEqual(len(records), 1)
        output = records[0]["assistant_output"]
        self.assertEqual(output["context_transcript"], "Next type elif target greater than arr mid.")
        self.assertEqual(output["completion_state"], "interrupted")
        self.assertEqual(output["heard_text_status"], "unknown")
        self.assertNotIn("UNFORWARDED_TAIL_DO_NOT_RESTORE", json.dumps(history))
        self.assertFalse(any(item["role"] == "model" for item in history))

    async def test_committed_missing_transcript_survives_as_explicit_gap(self):
        await self.session._reset_before_new_input(reason="fixture_fresh_connection")
        history = self.h.wires[-1].connect.await_args.kwargs["history"]
        records = self.incomplete_records(history)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["input_status"], "unavailable_after_reset")
        self.assertTrue(records[0]["input_committed"])
        self.assertEqual(records[0]["turn_number"], 1)
        self.assertNotIn("user_transcript", records[0])
        self.assertNotIn(fixture.ENCODED, json.dumps(history))

    async def test_completed_but_truncated_output_is_not_restored_as_completed(self):
        await self.user_transcript()
        await self.h.answer(text="The first instruction. Then another instruction.")
        await fixture.until(lambda: bool(self.h.events("response.done")))
        self.session._record_playback({
            "item_id": self.session._turns["a"].output.item_id,
            "audio_end_ms": 5,
        })
        history = self.session._conversation_history()
        self.assertFalse(any(item["role"] == "model" for item in history))
        output = self.incomplete_records(history)[0]["assistant_output"]
        self.assertEqual(output["completion_state"], "truncated")
        self.assertEqual(output["playback_end_ms"], 5)
        self.assertEqual(output["heard_text_status"], "unknown")

    async def test_never_forwarded_generated_words_are_not_restored(self):
        await self.user_transcript()
        output = self.session._turns["a"].output
        output.consume({"serverContent": {
            "modelTurn": {"parts": [{"inlineData": {
                "mimeType": "audio/pcm;rate=24000", "data": fixture.ENCODED,
            }}]},
            "outputTranscription": {"text": "NEVER_FORWARDED_SECRET"},
            "generationComplete": True, "turnComplete": True,
        }})
        history = self.session._conversation_history()
        self.assertNotIn("NEVER_FORWARDED_SECRET", json.dumps(history))
        record = self.incomplete_records(history)[0]["assistant_output"]
        self.assertEqual(record["forwarded_audio_bytes"], 0)
        self.assertNotIn("context_transcript", record)
        self.assertFalse(any(item["role"] == "model" for item in history))

    async def test_completed_fully_forwarded_exchange_keeps_native_history_order(self):
        await self.user_transcript("A finalized question.")
        await self.h.answer(text="A finalized answer.")
        await fixture.until(lambda: bool(self.h.events("response.done")))
        history = self.session._conversation_history()
        self.assertEqual([item["role"] for item in history], ["user", "model"])
        self.assertEqual(history[0]["parts"][0]["text"], "A finalized question.")
        self.assertEqual(history[1]["parts"][0]["text"], "A finalized answer.")

    async def test_completed_answer_does_not_hide_a_missing_input_transcript(self):
        await self.h.answer(text="The answer was forwarded.")
        await fixture.until(lambda: bool(self.h.events("response.done")))
        await self.session._reset_before_new_input(reason="fixture_fresh_connection")
        history = self.h.wires[-1].connect.await_args.kwargs["history"]
        records = self.incomplete_records(history)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["input_status"], "unavailable_after_reset")
        self.assertTrue(records[0]["input_committed"])
        self.assertNotIn("assistant_output", records[0])
        self.assertEqual(history[-1]["role"], "model")

    async def test_only_partially_forwarded_completed_audio_is_incomplete_history(self):
        await self.user_transcript()
        await self.h.answer(text="Forwarded context.", terminal=False)
        await fixture.until(lambda: bool(self.h.events("audio.output.delta")))
        output = self.session._turns["a"].output
        output.consume({"serverContent": {
            "modelTurn": {"parts": [{"inlineData": {
                "mimeType": "audio/pcm;rate=24000", "data": fixture.ENCODED,
            }}]},
            "outputTranscription": {"text": " UNFORWARDED_TAIL"},
            "generationComplete": True, "turnComplete": True,
        }})
        history = self.session._conversation_history()
        self.assertFalse(any(item["role"] == "model" for item in history))
        output_record = self.incomplete_records(history)[0]["assistant_output"]
        self.assertEqual(output_record["context_transcript"], "Forwarded context.")
        self.assertNotIn("UNFORWARDED_TAIL", json.dumps(history))

    async def test_non_user_output_is_not_reported_as_missing_committed_speech(self):
        await self.h.answer(terminal=False)
        await fixture.until(lambda: bool(self.h.events("audio.output.delta")))
        turn = self.session._turns["a"]
        turn.input_bytes = 0  # Proactive turns have output but no user PCM.
        await self.session._cancel_output()
        record = self.incomplete_records(self.session._conversation_history())[0]
        self.assertFalse(record["input_committed"])
        self.assertEqual(record["input_status"], "not_applicable")

    async def test_oversized_latest_exchange_keeps_bounded_explicit_excerpts(self):
        await self.user_transcript("USER_START " + "x" * 3000 + " USER_CURRENT_STEP")
        await self.h.answer(text="ASSISTANT_START " + "y" * 3000 + " ASSISTANT_CURRENT_STEP")
        await fixture.until(lambda: bool(self.h.events("response.done")))
        with patch("app.realtime.gemini.session.MAX_CONTEXT_CHARACTERS", 2000):
            history = self.session._conversation_history()
        self.assertTrue(history)
        self.assertLessEqual(sum(len(item["parts"][0]["text"]) for item in history), 2000)
        self.assertFalse(any(item["role"] == "model" for item in history))
        record = json.loads(history[-1]["parts"][0]["text"])
        self.assertEqual(record["kind"], "jarvis_history_excerpt")
        self.assertEqual(record["turn_number"], 1)
        self.assertEqual(record["heard_text_status"], "unknown")
        self.assertTrue(record["truncated"])
        for marker in ("USER_START", "USER_CURRENT_STEP", "ASSISTANT_START", "ASSISTANT_CURRENT_STEP"):
            self.assertIn(marker, json.dumps(record))

    async def test_oversized_interrupted_history_keeps_valid_json_and_uncertainty(self):
        await self.user_transcript("What is next?")
        await self.h.answer(text="BEGIN " + '\\u2603"\\n' * 2000 + " CURRENT_STEP", terminal=False)
        await fixture.until(lambda: bool(self.h.events("audio.output.delta")))
        await self.session._cancel_output()
        with patch("app.realtime.gemini.session.MAX_CONTEXT_CHARACTERS", 2000):
            history = self.session._conversation_history()
        self.assertTrue(history)
        encoded = history[-1]["parts"][0]["text"]
        self.assertLessEqual(len(encoded), 2000)
        record = json.loads(encoded)
        self.assertEqual(record["assistant_completion_state"], "interrupted")
        self.assertEqual(record["heard_text_status"], "unknown")
        self.assertIn("CURRENT_STEP", encoded)

    async def test_empty_turns_do_not_displace_useful_history(self):
        await self.user_transcript("Useful question.")
        await self.h.answer(text="Useful answer.")
        await fixture.until(lambda: bool(self.h.events("response.done")))
        first = self.session._turns["a"]
        for number in range(2, 20):
            self.session._turns[str(number)] = replace(
                first, number=number, segment_id=str(number), input_bytes=0,
                sealed=False, input_finalized=False, input_transcript="",
                output=replace(first.output, transcript="", context_transcript="",
                               completed=False, forwarded_audio_bytes=0, provider_audio_bytes=0),
            )
        self.assertIn("Useful answer.", json.dumps(self.session._conversation_history()))

    async def test_history_retains_only_newest_sixteen_nonempty_groups_in_order(self):
        await self.user_transcript()
        await self.h.answer()
        await fixture.until(lambda: bool(self.h.events("response.done")))
        first = self.session._turns["a"]
        for number in range(2, 21):
            self.session._turns[str(number)] = replace(
                first, number=number, segment_id=str(number), input_transcript=f"Question {number}",
                output=replace(first.output, transcript=f"Answer {number}"),
            )
        history = self.session._conversation_history()
        questions = [item["parts"][0]["text"] for item in history if item["role"] == "user"]
        self.assertEqual(questions, [f"Question {number}" for number in range(5, 21)])

    async def test_completed_audio_without_output_transcript_is_an_explicit_gap(self):
        await self.user_transcript()
        await self.h.answer(text="")
        await fixture.until(lambda: bool(self.h.events("response.done")))
        history = self.session._conversation_history()
        self.assertFalse(any(item["role"] == "model" for item in history))
        output = self.incomplete_records(history)[0]["assistant_output"]
        self.assertGreater(output["forwarded_audio_bytes"], 0)
        self.assertNotIn("context_transcript", output)
        self.assertEqual(output["heard_text_status"], "unknown")
