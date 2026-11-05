from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.visual_memory.validation import jpeg_dimensions
from evals.perception_memory.direct_image_access_runner import LiveTrial, Trace, followup_recovery, load_scenarios, main, summarize, transport_gates
from evals.perception_memory.direct_image_scenes import render_scene
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


class DirectImageAccessEvaluationTests(unittest.TestCase):
    def test_matrix_has_current_historical_and_lifecycle_cases(self):
        cases = load_scenarios()
        self.assertEqual({case["mode"] for case in cases}, {"current", "historical", "lifecycle"})
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        for number in range(1, 17):
            self.assertTrue(any(case["id"].startswith(f"L{number:02d}") for case in cases))
        self.assertTrue(next(case for case in cases if case["id"] == "L15")["barge_in"])

    def test_expected_visual_labels_and_counts_do_not_appear_in_questions(self):
        for case in load_scenarios():
            question = case["question"].lower()
            expected = case["expected"]
            for key in ("count", "earlier", "current", "serial", "label"):
                if key in expected:
                    with self.subTest(case=case["id"], key=key):
                        self.assertNotIn(str(expected[key]).lower(), question)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Install the eval extra for synthetic scenes.")
    def test_all_scenes_are_bounded_jpegs_at_the_same_resolution(self):
        for case in load_scenarios():
            for key in ("scene", "initial_scene", "historical_scene"):
                if key in case:
                    with self.subTest(case=case["id"], scene=key):
                        pixels = render_scene(case[key])
                        self.assertEqual(jpeg_dimensions(pixels), (640, 480))
                        self.assertLessEqual(len(pixels), 512_000)
                        self.assertEqual(pixels, render_scene(case[key]))

    def test_live_runner_requires_both_explicit_opt_ins(self):
        for flags in ([], ["--live"], ["--allow-database-fixtures"]):
            with self.subTest(flags=flags), patch("sys.argv", ["runner", "--model", "fixture", "--output", "/tmp/unused", *flags]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    main()
                self.assertEqual(caught.exception.code, 2)

    def test_list_mode_does_not_need_credentials_or_provider_calls(self):
        with patch("sys.argv", ["runner", "--list", "--mode", "historical"]), contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn("L09", output.getvalue())
        self.assertNotIn("L01", output.getvalue())

    def test_unknown_case_cannot_silently_run_a_different_suite(self):
        with patch("sys.argv", ["runner", "--list", "--cases", "not-a-case"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_transport_requires_ack_then_output_then_pinned_continuation(self):
        trace = [
            {"direction": "send", "type": "conversation.item.create", "at": 1, "item_id": "image", "image_sha256": "hash"},
            {"direction": "receive", "type": "conversation.item.done", "at": 2, "item_id": "image"},
            {"direction": "send", "type": "conversation.item.create", "at": 3, "call_id": "call", "image_status": "attached", "provider_item_id": "image"},
            {"direction": "send", "type": "response.create", "at": 4, "requested_item_ids": ["image"]},
        ]
        self.assertTrue(all(transport_gates(trace, {"hash"}).values()))
        self.assertFalse(transport_gates([trace[0], trace[2], trace[3]], {"hash"})["ack_before_success_before_continuation"])
        self.assertFalse(transport_gates([*trace, trace[2]], {"hash"})["one_output_per_call"])
        self.assertFalse(transport_gates(trace, {"different"})["pixels_match_ingested_fixture"])

    def test_cancelled_generation_cannot_deliver_an_image_after_barge_in(self):
        trace = [
            {"direction": "driver", "type": "barge_in", "at": 1, "cancelled_generation": 2},
            {"direction": "send", "type": "conversation.item.create", "at": 2, "item_id": "image", "image_sha256": "hash", "turn_generation": 2},
        ]
        self.assertFalse(transport_gates(trace, {"hash"})["no_cancelled_generation_image_after_barge_in"])

    def test_summary_keeps_accuracy_and_physical_acceptance_unresolved(self):
        result = summarize([{
            "transport_passed": True, "cleanup": {"passed": True}, "forbidden_tool_calls": [],
            "case_id": "L04", "turns": [{"first_final_response_audio_ms": 1200}],
        }])
        self.assertEqual(result["transport_passed"], 1)
        self.assertIn("Manual review required", result["accuracy_status"])
        self.assertEqual(result["physical_glasses_testing"], "outstanding")
        self.assertEqual(result["first_final_response_audio_ms"], {"median": 1200, "min": 1200, "max": 1200})

    def test_cleanup_refuses_nonfixture_owner_without_connecting(self):
        with self.assertRaises(ValueError):
            asyncio.run(cleanup_fixture_owners("not-a-dsn", {"actual-owner"}))
        self.assertEqual(asyncio.run(cleanup_fixture_owners("not-a-dsn", set()))["remaining_rows"], 0)

    def test_code_controls_preserve_original_question_and_pixels(self):
        cases = {case["id"]: case for case in load_scenarios()}
        original = cases["L08"]
        self.assertEqual(original["question"], "What should the missing return statement be?")
        self.assertNotIn("setup_question", original)
        for identity in ("L08-explicit", "L08-context"):
            self.assertEqual(cases[identity]["scene"], original["scene"])
            self.assertEqual(cases[identity]["expected"], original["expected"])
        self.assertIn("camera", cases["L08-explicit"]["question"])
        self.assertEqual(cases["L08-context"]["question"], original["question"])
        self.assertEqual(cases["L08-context"]["initial_scene"], original["scene"])

    def test_timed_out_followup_is_still_reported_as_attempted(self):
        result = followup_recovery([
            {"type": "question.started", "question": "What is two plus two?"},
        ], [], "What is two plus two?")
        self.assertTrue(result["followup_attempted"])
        self.assertFalse(result["followup_completed"])
        self.assertFalse(followup_recovery([], [], "What is two plus two?")["followup_attempted"])


class LiveTrialFlowTests(unittest.IsolatedAsyncioTestCase):
    def trial(self):
        trial = object.__new__(LiveTrial)
        trial.trace, trial.turns = Trace(), []
        trial.manager = SimpleNamespace(session=SimpleNamespace(
            _normal_turn_pending=False, _tool_followup=None, _image_tool_tasks={},
        ))
        return trial

    def terminal_trace(self, trial, *, outcome="failed", partial=False):
        marker = trial.trace.add("driver", "question.started", question="Read the label.")
        end = trial.trace.add("driver", "utterance.ended")
        trial.trace.add("receive", "response.created", response_id="response")
        trial.trace.add("client", "user.transcript.done", text="Read the label.")
        if partial:
            trial.trace.add("client", "transcript.done", response_id="response", text="Partial answer")
            trial.trace.add("client", "audio.output.done", response_id="response")
        trial.trace.add("receive", "response.done", response_id="response", response_status=outcome)
        trial.trace.add("client", "response.done", response_id="response")
        return marker, end

    async def test_failed_response_finishes_without_spoken_answer_or_false_success(self):
        for outcome in ("failed", "cancelled", "incomplete"):
            with self.subTest(outcome=outcome):
                trial = self.trial()
                marker, end = self.terminal_trace(trial, outcome=outcome, partial=True)
                self.assertFalse(trial.completed(marker))
                result = await trial.finish_question(marker, end)
                self.assertEqual(result["response_status"], outcome)
                self.assertIsNone(result["answer"])
                self.assertIsNone(result["answer_complete_ms"])
                self.assertEqual(result["partial_transcripts"], ["Partial answer"])

    async def test_success_still_requires_completed_transcript_and_audio(self):
        trial = self.trial()
        marker, _ = self.terminal_trace(trial, outcome="completed")
        self.assertFalse(trial.completed(marker))
        trial.trace.add("client", "transcript.done", response_id="response", text="The label.")
        self.assertFalse(trial.completed(marker))
        trial.trace.add("client", "audio.output.done", response_id="response")
        self.assertTrue(trial.completed(marker))
        self.assertIsNone(trial.failed_response(marker))

    async def test_previous_turn_cancellation_is_not_new_turn_failure(self):
        trial = self.trial()
        trial.trace.add("receive", "response.created", response_id="old")
        marker = trial.trace.add("driver", "question.started", question="Replacement")
        trial.trace.add("receive", "response.done", response_id="old", response_status="cancelled")
        trial.trace.add("client", "response.done", response_id="old")
        self.assertIsNone(trial.failed_response(marker))

    async def test_failure_notice_completion_does_not_erase_provider_failure(self):
        trial = self.trial()
        marker, end = self.terminal_trace(trial)
        trial.manager.session._normal_turn_pending = True
        self.assertIsNone(trial.failed_response(marker))
        trial.trace.add("receive", "response.created", response_id="notice")
        trial.trace.add("client", "transcript.done", response_id="notice", text="I could not read that image.")
        trial.trace.add("client", "audio.output.done", response_id="notice")
        trial.trace.add("receive", "response.done", response_id="notice", response_status="completed")
        trial.trace.add("client", "response.done", response_id="notice")
        trial.manager.session._normal_turn_pending = False
        result = await trial.finish_question(marker, end)
        self.assertEqual(result["response_status"], "completed")
        self.assertEqual(result["provider_failures"], [{"response_id": "response", "status": "failed"}])
        self.assertEqual(result["answer"], "I could not read that image.")

    async def test_failed_image_turn_does_not_prevent_normal_followup(self):
        trial = self.trial()
        trial.case = {"id": "L14-rejected", "question": "Read the label.", "followup_question": "What is two plus two?"}
        trial.session_row = AsyncMock(return_value="session")
        trial.connect_audio = AsyncMock()
        marker, end = self.terminal_trace(trial)
        failed = await trial.finish_question(marker, end)
        trial.ask = AsyncMock(side_effect=[failed, {"response_status": "completed", "answer": "Four"}])
        await trial.exercise()
        self.assertEqual([call.args[0] for call in trial.ask.await_args_list],
                         [trial.case["question"], trial.case["followup_question"]])
        trial.connect_audio.assert_awaited_once_with("session")

    async def test_historical_fixture_promotes_real_event_with_frame_evidence(self):
        trial = self.trial()
        trial.pool, trial.owner, trial.manifest = object(), "direct-image-eval-unit", MagicMock()
        experience, memory = AsyncMock(), AsyncMock()
        experience.get_or_create_open_event.return_value = SimpleNamespace(id="event")
        memory.promote_event_to_episode.return_value = SimpleNamespace(id="memory")
        calls = MagicMock()
        calls.attach_mock(experience, "experience")
        calls.attach_mock(memory, "memory")
        with patch("evals.perception_memory.direct_image_access_runner.ExperienceRepository", return_value=experience), \
             patch("evals.perception_memory.direct_image_access_runner.MemoryRepository", return_value=memory):
            await trial.seed_historical_memory("session", "frame")
        self.assertEqual(experience.select_keyframe.await_args.kwargs["visual_frame_id"], "frame")
        self.assertEqual(memory.promote_event_to_episode.await_args.kwargs["event_id"], "event")
        self.assertEqual(memory.attach_evidence.await_args.kwargs["evidence_ref_id"], "frame")
        self.assertEqual([call[0] for call in calls.mock_calls], [
            "experience.get_or_create_open_event", "experience.select_keyframe", "experience.close_event",
            "memory.promote_event_to_episode", "memory.attach_evidence",
        ])
        memory.create_memory_item.assert_not_awaited()


class FixtureCleanupTests(unittest.IsolatedAsyncioTestCase):
    owners = {"direct-image-eval-cleanup-unit"}

    def connection(self, snapshots, *, referenced=True, nullable=True, foreign=False):
        tables = ["experience_events", "memory_items", "memory_jobs"]
        edges = [
            {"child_table": "experience_events", "parent_table": "memory_items",
             "child_columns": ["promoted_episode_memory_id"], "parent_columns": ["id"],
             "nullable": nullable},
            {"child_table": "memory_items", "parent_table": "experience_events",
             "child_columns": ["source_experience_event_id"], "parent_columns": ["id"],
             "nullable": nullable},
        ]
        counts = iter(snapshots)
        connection = AsyncMock()
        connection.transaction = MagicMock(return_value=MagicMock())

        async def fetch(sql, *args):
            if "from pg_class c" in sql:
                return [{"table_name": table} for table in tables]
            if "from pg_constraint fk" in sql:
                return edges
            return [{"table_name": table, "row_count": count}
                    for table, count in next(counts).items()]

        async def fetchval(sql, *args):
            if "not coalesce" in sql:
                return foreign
            return referenced

        async def execute(sql, *args):
            if sql.startswith("update"):
                return "UPDATE 1"
            return "DELETE 2" if '"memory_jobs"' in sql else "DELETE 1"

        connection.fetch.side_effect = fetch
        connection.fetchval.side_effect = fetchval
        connection.execute.side_effect = execute
        return connection

    async def cleanup(self, connection):
        with patch("evals.perception_memory.fixture_cleanup.asyncpg.connect",
                   new=AsyncMock(return_value=connection)):
            return await cleanup_fixture_owners("fixture-dsn", self.owners)

    async def test_schema_cycle_without_row_references_needs_no_update(self):
        connection = self.connection(
            [{"experience_events": 1, "memory_items": 1}, {}, {}], referenced=False,
        )
        result = await self.cleanup(connection)
        self.assertEqual(result["remaining_rows"], 0)
        self.assertFalse(any(call.args[0].startswith("update")
                             for call in connection.execute.await_args_list))
        connection.close.assert_awaited_once()

    async def test_nullable_row_cycle_is_broken_only_between_fixture_owners(self):
        connection = self.connection([{"experience_events": 1, "memory_items": 1}, {}, {}])
        result = await self.cleanup(connection)
        update, *deletes = connection.execute.await_args_list
        self.assertTrue(update.args[0].startswith('update public."experience_events"'))
        self.assertIn('"promoted_episode_memory_id" = null', update.args[0])
        self.assertIn('"promotion_decision" = \'pending\'', update.args[0])
        self.assertIn('"promotion_reason" = null', update.args[0])
        self.assertIn('"promotion_decided_at" = null', update.args[0])
        self.assertIn("child.owner_id::text = any($1::text[])", update.args[0])
        self.assertIn("parent.owner_id::text = any($1::text[])", update.args[0])
        self.assertEqual(update.args[1], sorted(self.owners))
        self.assertIn('delete from public."memory_items"', deletes[0].args[0])
        self.assertEqual(result["removed_rows"], {"memory_items": 1, "experience_events": 1})

    async def test_later_sweep_removes_trigger_created_jobs_and_reports_actual_deletes(self):
        connection = self.connection([
            {"experience_events": 1, "memory_items": 1}, {"memory_jobs": 2}, {}, {},
        ], referenced=False)
        result = await self.cleanup(connection)
        self.assertEqual(result["removed_rows"]["memory_jobs"], 2)
        self.assertEqual(result["remaining_rows"], 0)
        for call in connection.execute.await_args_list:
            self.assertIn("where owner_id::text = any($1::text[])", call.args[0])
            self.assertEqual(call.args[1], sorted(self.owners))

    async def test_foreign_owner_reference_refuses_mutation_and_rolls_back(self):
        connection = self.connection([{"memory_items": 1}], foreign=True)
        with self.assertRaisesRegex(RuntimeError, "Non-fixture rows"):
            await self.cleanup(connection)
        connection.execute.assert_not_awaited()
        self.assertIs(connection.transaction.return_value.__aexit__.await_args.args[0], RuntimeError)
        connection.close.assert_awaited_once()

    async def test_nonnullable_cycle_refuses_mutation(self):
        connection = self.connection([{"experience_events": 1, "memory_items": 1}], nullable=False)
        with self.assertRaisesRegex(RuntimeError, "Non-nullable fixture cycle"):
            await self.cleanup(connection)
        connection.execute.assert_not_awaited()
        connection.close.assert_awaited_once()

    async def test_nonconverging_trigger_rolls_back_after_bounded_sweeps(self):
        connection = self.connection([{"memory_jobs": 2}] * 5)
        with self.assertRaisesRegex(RuntimeError, "Fixture-owned rows remain"):
            await self.cleanup(connection)
        self.assertEqual(connection.execute.await_count, 4)
        self.assertIs(connection.transaction.return_value.__aexit__.await_args.args[0], RuntimeError)
        connection.close.assert_awaited_once()
