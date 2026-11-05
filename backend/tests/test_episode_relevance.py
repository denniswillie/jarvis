"""Pure relevance policy checks; production owner/pipeline scenarios live separately."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import unittest
from uuid import uuid4

from app.live_state.episode_relevance import ActivityEvidence, cue_label, relevance_for_check
from app.live_state.models import SessionScope, SourceDependency
from app.live_state.perception import SituationalCheck


class EpisodeRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 30, 12, tzinfo=UTC)
        self.scope = SessionScope(owner_id="relevance-fixture", session_id=uuid4())
        self.check = SituationalCheck(id=uuid4(), revision=1, scope=self.scope,
            episode_or_object_ref=SourceDependency(source_namespace="episode", source_id=uuid4(), revision=1),
            origin="explicit_request", question="What is visible?", lifecycle="tracking",
            activation_cues=("When stretching",))
        self.activity = ActivityEvidence(self.scope,
            SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1),
            "stretching", 0.85, self.now - timedelta(seconds=2), self.now - timedelta(seconds=1),
            self.now + timedelta(seconds=9), ())

    def decide(self, *, check=None, activity=None, start=None):
        return relevance_for_check(check or self.check, start or self.now - timedelta(seconds=3),
            activity or self.activity, now=self.now)

    def test_generic_label_grammar_not_object_or_departure_keywords(self):
        for cue, label in (("When departing", "departing"), ("WHILE washing clothes", "washing clothes"),
                           ("During calibrating sensors.", "calibrating sensors"),
                           ("activity: preparing breakfast", "preparing breakfast"),
                           ("When stretching", "stretching")):
            with self.subTest(cue=cue):
                result = self.decide(check=self.check.model_copy(update={"activation_cues":(cue,)}),
                    activity=replace(self.activity, label=label))
                self.assertTrue(result.relevant)
                self.assertEqual(result.reason, "inferred_activity_relevance_not_completion")
                self.assertEqual(result.activity.source, self.activity.source)

    def test_ambiguous_compound_or_unknown_grammar_is_explicitly_unsupported(self):
        for cue in ("When packing and departing", "If the room is empty", "Before departing", "When not stretching",
                    "When maybe stretching", "Anything important", "After five minutes"):
            with self.subTest(cue=cue):
                self.assertIsNone(cue_label(cue))
                result = self.decide(check=self.check.model_copy(update={"activation_cues":(cue,)}))
                self.assertFalse(result.relevant)
                self.assertEqual(result.reason, "cue_grammar_unsupported")

    def test_no_substring_synonym_or_summary_guessing(self):
        for label in ("not stretching", "stretching later", "stretched", "exercise", "packing"):
            with self.subTest(label=label):
                self.assertFalse(self.decide(activity=replace(self.activity, label=label)).relevant)

    def test_uncertain_activity_never_becomes_completion_evidence(self):
        result = self.decide(activity=replace(self.activity, confidence=0.69))
        self.assertEqual((result.relevant, result.reason), (False, "activity_uncertain"))
        self.assertIsNone(self.check.resolution)
        self.assertEqual(self.check.evidence_availability, "unknown")

    def test_old_episode_future_expired_and_foreign_evidence_fail_closed(self):
        cases = (
            (replace(self.activity, started_at=self.now - timedelta(minutes=1)), None),
            (replace(self.activity, captured_at=self.now + timedelta(seconds=1)), None),
            (replace(self.activity, expires_at=self.now), None),
            (replace(self.activity, scope=self.scope.model_copy(update={"session_id":uuid4()})), None),
            (self.activity, self.now),
        )
        for activity, start in cases:
            with self.subTest(activity=activity, start=start):
                self.assertFalse(self.decide(activity=activity, start=start).relevant)

    def test_cancel_snooze_expiry_are_not_overridden_by_matching_activity(self):
        for update in ({"lifecycle":"dismissed"}, {"lifecycle":"expired"},
                       {"snoozed_until":self.now + timedelta(seconds=10)},
                       {"observation_window_expires_at":self.now}):
            with self.subTest(update=update):
                self.assertEqual(self.decide(check=self.check.model_copy(update=update)).reason, "check_inactive")

    def test_multiple_alternative_cues_are_bounded_and_independent(self):
        result = self.decide(check=self.check.model_copy(update={"activation_cues":
            ("If something happens", "When washing clothes", "When stretching")}))
        self.assertTrue(result.relevant)
        self.assertEqual(result.cue, "When stretching")


if __name__ == "__main__":
    unittest.main()
