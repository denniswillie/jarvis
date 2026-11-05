"""Negative controls for native continuity checks; no provider calls."""
import unittest

from evals.glasses_reliability.gemini_continuity_probe import assess_answer, classify_checks, turn_timings


class ContinuityProbeTests(unittest.TestCase):
    def test_spelling_the_current_condition_passes(self):
        checks = assess_answer("Eight spaces. E, L, I, F, space, T A R G E T, space, greater than, space, A R R, open bracket, M I D, close bracket, colon.")
        self.assertTrue(all(checks.values()))

    def test_spelling_the_stale_return_step_cannot_pass(self):
        self.assertFalse(all(assess_answer("R E T U R N, space, M I D.").values()))

    def test_a_condition_without_character_spelling_cannot_pass(self):
        self.assertFalse(assess_answer("elif target > arr[mid]:")["explicit_character_spelling"])

    def test_spelling_only_the_keyword_cannot_pass(self):
        self.assertFalse(assess_answer("E L I F space target greater than arr mid")["explicit_character_spelling"])

    def test_extra_parenthesis_is_flagged_for_answer_review(self):
        self.assertFalse(assess_answer("E L I F, t a r g e t, greater than, a r r, bracket, m i d, bracket, parentheses, colon.")["no_added_parenthesis"])

    def test_transcript_formatting_is_not_assumed_to_be_spoken_punctuation(self):
        self.assertTrue(assess_answer("e l i f (space) t a r g e t")["no_added_parenthesis"])

    def test_written_brackets_do_not_prove_delimiters_were_dictated(self):
        self.assertFalse(assess_answer("a r r [ m i d ]")["explicit_bracket_directions"])

    def test_generic_brackets_do_not_prove_opening_and_closing_directions(self):
        self.assertFalse(assess_answer("a r r bracket m i d bracket")["explicit_bracket_directions"])

    def test_incorrect_indentation_count_is_rejected(self):
        self.assertFalse(assess_answer("Nine spaces, e l i f")["explicit_indentation_count"])

    def test_reversed_comparison_does_not_pass_on_keyword_presence(self):
        text = "Eight spaces e l i f space a r r open bracket m i d close bracket greater than t a r g e t colon. This means target is greater than arr at mid."
        self.assertFalse(assess_answer(text)["correct_comparison"])

    def test_greater_or_equal_is_not_the_requested_comparison(self):
        self.assertFalse(assess_answer("elif target >= arr[mid]:")["correct_comparison"])

    def test_missing_colon_is_not_complete_dictation(self):
        self.assertFalse(assess_answer("elif target greater than arr mid")["spoken_colon"])

    def test_closed_bracket_is_a_valid_direction_and_eight_repeated_spaces_are_valid(self):
        text = "space " * 8 + "e l i f space t a r g e t space greater than space a r r open bracket m i d closed bracket colon"
        self.assertTrue(all(assess_answer(text).values()))
        self.assertFalse(assess_answer("space " + text)["explicit_indentation_count"])

    def test_quality_failure_stays_failed_even_when_continuity_is_preserved(self):
        checks = dict.fromkeys((
            "current_elif_step", "answer_audio_complete", "input_pcm_not_replayed",
            "no_stale_end_marker", "expected_connection_count", "no_provider_error", "resets_explained",
        ), True)
        grade = classify_checks("ordinary", checks)
        self.assertTrue(grade["continuity_passed"])
        self.assertFalse(grade["dictation_quality_passed"])
        self.assertFalse(grade["passed"])

    def test_unobserved_interruption_reset_boundary_cannot_pass_continuity(self):
        checks = dict.fromkeys((
            "current_elif_step", "answer_audio_complete", "input_pcm_not_replayed",
            "no_stale_end_marker", "expected_connection_count", "no_provider_error", "resets_explained",
        ), True)
        self.assertFalse(classify_checks("interrupted_reset", checks)["continuity_passed"])
        self.assertFalse(classify_checks("ordinary", {})["passed"])

    def test_missing_audio_is_not_reported_as_zero_latency(self):
        rows = turn_timings({"turns": [{
            "segment_id": "bridge", "connection_generation": 1, "input_status": "unavailable_after_reset",
            "commit_received_at": 12, "first_provider_audio_at": None, "first_forwarded_audio_at": None,
        }]})
        self.assertIsNone(rows[0]["commit_to_first_provider_audio_ms"])
        self.assertIsNone(rows[0]["commit_to_first_forwarded_audio_ms"])
