"""Offline brief-composition regression tests; no canonical owner or model calls."""
from datetime import UTC, datetime, timedelta
import unittest
from uuid import UUID, uuid4

from app.live_state.models import LiveCapabilities, LiveSnapshot, SessionScope, SourceDependency, StateCursor
from app.live_state.perception import (
    CheckLifecycle, CheckResolution, EvidenceReference, ObservationalQuestion,
    PerceptionActivity, PriorEvidence, SituationalCheck,
)
from app.live_state.perception_context import compose_perception_brief


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def changed(model, **fields):
    return type(model)(**(model.model_dump() | fields))


class PerceptionContextRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="perception@example.test", session_id=UUID(int=1), area_id=UUID(int=2))
        self.scope_epoch = UUID(int=3)
        self.episode = SourceDependency(source_namespace="episode", source_id=UUID(int=4), revision=1)
        self.intent = SourceDependency(source_namespace="context_event", source_id=UUID(int=5), revision=1)
        self.source = SourceDependency(source_namespace="knowledge_source", source_id=UUID(int=6),
                                       revision=7, generation=3)
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=UUID(int=7), revision=1),
            as_of=NOW, readiness="ready", capabilities=LiveCapabilities(durable_responsibilities="available"))

    def check(self, **changes):
        return SituationalCheck(**(dict(id=UUID(int=8), revision=1, scope=self.scope,
            episode_or_object_ref=self.episode, origin="explicit_request", question="Are the controls visible?",
            lifecycle=CheckLifecycle.TRACKING, relevance_now=True) | changes))

    def question(self, **changes):
        return ObservationalQuestion(**(dict(id=UUID(int=9), question="What number is visible?",
                                             priority="explicit_inspection") | changes))

    def prior(self, **changes):
        return PriorEvidence(**(dict(reference=EvidenceReference(source=self.source,
            captured_at=NOW - timedelta(minutes=5), episode_or_object_ref=self.episode,
            frame_id=UUID(int=10), visibility="unclear"), finding="Control label not legible.",
            confidence=0.2, uncertainty="Shutdown is unconfirmed.") | changes))

    def compose(self, **changes):
        snapshot = changes.pop("snapshot", self.snapshot)
        return compose_perception_brief(snapshot, **(dict(now=NOW, scope_epoch=self.scope_epoch,
                                                         dependency_eligible=lambda source: True) | changes))

    def reasons(self, result):
        return {diagnostic.reason for diagnostic in result.diagnostics}

    def durable(self, **changes):
        return self.check(**(dict(continuity="durable", canonical_owner_ref=SourceDependency(
            source_namespace="responsibility", source_id=UUID(int=11), revision=2)) | changes))

    def test_minimal_startup_is_unknown_bounded_and_needs_no_dependency_predicate(self):
        result = self.compose(snapshot=changed(self.snapshot, readiness="warming"), dependency_eligible=None)
        self.assertIsNone(result.brief.activity)
        self.assertFalse(result.brief.questions)
        self.assertFalse(result.brief.prior_evidence)
        self.assertEqual(len(result.brief.scene_change_cues), 1)
        self.assertIn("warming", self.reasons(result))
        self.assertEqual(result.brief.fresh_until, NOW + timedelta(seconds=10))
        self.assertEqual(result.brief.expires_at, NOW + timedelta(seconds=30))

    def test_source_backed_inputs_need_explicit_true_eligibility(self):
        check = self.check(dependency_versions=(self.source,))
        for predicate in (None, lambda source: False, lambda source: None, lambda source: 1):
            with self.subTest(predicate=predicate):
                result = self.compose(checks=(check,), prior_evidence=(self.prior(),),
                                      dependency_eligible=predicate)
                self.assertFalse(result.brief.questions)
                self.assertFalse(result.brief.prior_evidence)
        result = self.compose(checks=(check,))
        self.assertEqual(len(result.brief.questions), 1)
        self.assertIn(self.source, result.brief.dependencies)

    def test_eligibility_failure_is_diagnosed_not_treated_as_authority(self):
        def unavailable(source):
            raise RuntimeError("canonical owner unavailable")
        result = self.compose(checks=(self.check(),), dependency_eligible=unavailable)
        self.assertFalse(result.brief.questions)
        self.assertIn("dependency_eligibility_error", self.reasons(result))

    def test_retirement_denies_source_identity_even_when_content_version_differs(self):
        for retired in (changed(self.source, generation=4), changed(self.source, revision=8)):
            with self.subTest(retired=retired):
                result = self.compose(checks=(self.check(dependency_versions=(self.source,)),),
                    prior_evidence=(self.prior(),), invalidated_sources=(retired,))
                self.assertFalse(result.brief.questions)
                self.assertFalse(result.brief.prior_evidence)
                self.assertIn("source_invalidated_or_retired", self.reasons(result))

    def test_invalidation_overflow_fails_closed_without_silently_dropping_denials(self):
        invalidations = tuple(changed(self.source, source_id=UUID(int=1000 + i)) for i in range(257))
        result = self.compose(checks=(self.check(),), prior_evidence=(self.prior(),),
                              invalidated_sources=invalidations)
        self.assertFalse(result.brief.questions)
        self.assertFalse(result.brief.prior_evidence)
        self.assertIn("invalidation_budget_fail_closed", self.reasons(result))

    def test_activity_and_intent_are_not_retained_after_their_source_becomes_ineligible(self):
        activity = PerceptionActivity(label="cooking", confidence=0.5, uncertainty="Activity uncertain.")
        result = self.compose(activity=activity, intent_ref=self.intent,
            context_dependencies=(self.source,), dependency_eligible=lambda source: source != self.source)
        self.assertIsNone(result.brief.activity)
        self.assertIsNone(result.brief.intent_ref)
        self.assertFalse(result.brief.dependencies)

    def test_cross_scope_checks_are_excluded_before_authorization(self):
        for scope in (changed(self.scope, owner_id="other"), changed(self.scope, session_id=uuid4()),
                      changed(self.scope, area_id=uuid4())):
            with self.subTest(scope=scope):
                def authorize(source):
                    self.fail("Foreign check reached scope-bound authorization.")
                result = self.compose(checks=(self.check(scope=scope),), dependency_eligible=authorize)
                self.assertFalse(result.brief.questions)
                self.assertIn("scope_mismatch", self.reasons(result))

    def test_inactive_and_irrelevant_checks_remain_unchanged_and_unassigned(self):
        resolution = CheckResolution(basis="user_report", source_refs=(EvidenceReference(
            source=self.source, captured_at=NOW, episode_or_object_ref=self.episode),),
            resolved_at=NOW, applicable_episode_or_object_ref=self.episode)
        checks = (self.check(lifecycle=CheckLifecycle.DISMISSED),
                  self.check(lifecycle=CheckLifecycle.EXPIRED),
                  self.check(lifecycle=CheckLifecycle.RESOLVED, resolution=resolution),
                  self.check(relevance_now=False))
        for check in checks:
            before = check.model_dump_json()
            with self.subTest(lifecycle=check.lifecycle, relevant=check.relevance_now):
                self.assertFalse(self.compose(checks=(check,)).brief.questions)
                self.assertEqual(check.model_dump_json(), before)

    def test_snooze_is_respected_until_exact_deadline_without_completing_check(self):
        check = self.check(snoozed_until=NOW + timedelta(seconds=5))
        self.assertFalse(self.compose(checks=(check,)).brief.questions)
        result = self.compose(checks=(check,), now=NOW + timedelta(seconds=5))
        self.assertEqual(len(result.brief.questions), 1)
        self.assertEqual(check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(check.resolution)

    def test_expired_observation_window_ends_assignment_not_responsibility(self):
        check = self.durable(observation_window_expires_at=NOW)
        result = self.compose(checks=(check,))
        self.assertFalse(result.brief.questions)
        self.assertIn("observation_window_expired", self.reasons(result))
        self.assertEqual(check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(check.resolution)

    def test_unknown_or_out_of_view_evidence_does_not_end_relevant_tracking(self):
        for availability in ("unknown", "unavailable"):
            with self.subTest(availability=availability):
                check = self.durable(evidence_availability=availability)
                result = self.compose(checks=(check,))
                self.assertEqual(len(result.brief.questions), 1)
                self.assertEqual(check.lifecycle, CheckLifecycle.TRACKING)

    def test_disabled_responsibilities_do_not_silently_replace_durable_with_ephemeral(self):
        durable = self.durable()
        ephemeral = self.check(id=UUID(int=12))
        for capability in ("disabled", "unavailable"):
            with self.subTest(capability=capability):
                snapshot = changed(self.snapshot, capabilities=LiveCapabilities(durable_responsibilities=capability))
                result = self.compose(snapshot=snapshot, checks=(durable, ephemeral))
                self.assertEqual([q.check_ref.source_id for q in result.brief.questions], [ephemeral.id])
                self.assertEqual(result.brief.durable_responsibility.status, capability)
                self.assertIn("responsibility_owner_unavailable", self.reasons(result))
                self.assertEqual(durable.lifecycle, CheckLifecycle.TRACKING)

    def test_independent_visual_focus_is_not_disabled_by_responsibility_capability(self):
        check = self.check(canonical_owner_ref=SourceDependency(source_namespace="visual_focus",
            source_id=UUID(int=8), revision=1), continuity="durable")
        snapshot = changed(self.snapshot, capabilities=LiveCapabilities(durable_responsibilities="disabled"))
        result = self.compose(snapshot=snapshot, checks=(check,))
        self.assertEqual(len(result.brief.questions), 1)
        self.assertEqual(result.brief.questions[0].focus_ref, check.canonical_owner_ref)
        self.assertEqual(result.brief.questions[0].check_ref.source_namespace, "visual_focus")

    def test_focus_owner_mismatch_and_unsupported_owners_are_rejected(self):
        for owner, reason in ((SourceDependency(source_namespace="visual_focus", source_id=uuid4(), revision=1),
                               "focus_owner_binding_mismatch"),
                              (SourceDependency(source_namespace="invented", source_id=uuid4(), revision=1),
                               "unsupported_canonical_owner")):
            with self.subTest(owner=owner):
                result = self.compose(checks=(self.check(canonical_owner_ref=owner),))
                self.assertFalse(result.brief.questions)
                self.assertIn(reason, self.reasons(result))

    def test_check_question_keeps_owner_episode_and_exact_dependencies(self):
        check = self.durable(dependency_versions=(self.source,))
        result = self.compose(checks=(check,))
        question = result.brief.questions[0]
        expected_ref = SourceDependency(source_namespace="responsibility_check", source_id=check.id, revision=1)
        self.assertEqual(question.check_ref, expected_ref)
        self.assertEqual(question.episode_or_object_ref, self.episode)
        self.assertEqual(set(result.brief.dependencies),
                         {expected_ref, self.episode, self.source, check.canonical_owner_ref})

    def test_explicit_inspections_get_slots_first_and_preserve_fifo(self):
        questions = tuple(self.question(id=UUID(int=100 + i)) for i in range(6))
        checks = tuple(self.check(id=UUID(int=200 + i)) for i in range(6))
        result = self.compose(explicit_questions=questions, checks=checks)
        self.assertEqual(result.brief.questions, questions[:4])
        self.assertEqual(len(result.brief.scene_change_cues), 1)
        self.assertIn("question_budget", self.reasons(result))
        self.assertTrue(all(c.lifecycle == CheckLifecycle.TRACKING for c in checks))

    def test_duplicate_explicit_ids_and_ambiguous_check_versions_are_not_guessed(self):
        question = self.question()
        check = self.check()
        result = self.compose(explicit_questions=(question, changed(question, question="Different question")),
                              checks=(check, changed(check, revision=2)))
        self.assertFalse(result.brief.questions)
        self.assertIn("ambiguous_question_identity", self.reasons(result))
        self.assertIn("ambiguous_check_projection", self.reasons(result))

    def test_explicit_check_questions_require_current_check_and_episode(self):
        check = self.check()
        reference = SourceDependency(source_namespace="situational_check", source_id=check.id, revision=1)
        question = self.question(check_ref=reference, episode_or_object_ref=self.episode)
        valid = self.compose(checks=(check,), explicit_questions=(question,))
        self.assertEqual(valid.brief.questions, (question,))
        for checks, wrong in (((), question), ((check,), changed(question, check_ref=changed(reference, revision=2))),
                             ((check,), changed(question, episode_or_object_ref=changed(self.episode, revision=2)))):
            with self.subTest(checks=checks, question=wrong):
                result = self.compose(checks=checks, explicit_questions=(wrong,))
                self.assertNotIn(wrong.id, {q.id for q in result.brief.questions})

    def test_generated_question_identity_stable_until_check_version_changes(self):
        check = self.check()
        first = self.compose(checks=(check,))
        renewed = self.compose(checks=(check,), now=NOW + timedelta(seconds=30), previous=first)
        replacement = self.compose(checks=(changed(check, revision=2),), previous=renewed,
                                   now=NOW + timedelta(seconds=31))
        self.assertEqual(first.brief.questions[0].id, renewed.brief.questions[0].id)
        self.assertNotEqual(first.brief.questions[0].id, replacement.brief.questions[0].id)
        self.assertEqual(replacement.brief.questions[0].check_ref.revision, 2)

    def test_unrelated_snapshot_revision_and_ordinary_reads_reuse_original_brief(self):
        first = self.compose(checks=(self.check(),))
        for seconds in (1, 11, 29):
            snapshot = changed(self.snapshot, cursor=changed(self.snapshot.cursor, revision=900),
                               as_of=NOW + timedelta(seconds=seconds))
            result = self.compose(snapshot=snapshot, checks=(self.check(),),
                                  now=NOW + timedelta(seconds=seconds), previous=first)
            self.assertTrue(result.reused)
            self.assertIs(result.brief, first.brief)
            self.assertEqual(result.selection_round, first.selection_round)
            self.assertEqual(result.brief.based_on_live_revision, 1)

    def test_expiry_renews_assignment_but_never_refreshes_prior_evidence(self):
        prior = self.prior()
        first = self.compose(prior_evidence=(prior,))
        renewed = self.compose(prior_evidence=(prior,), now=NOW + timedelta(seconds=30), previous=first)
        self.assertFalse(renewed.reused)
        self.assertEqual(renewed.brief.id, first.brief.id)
        self.assertEqual(renewed.brief.version, first.brief.version + 1)
        self.assertEqual(renewed.brief.prior_evidence, (prior,))
        self.assertEqual(renewed.brief.prior_evidence[0].reference.captured_at, NOW - timedelta(minutes=5))

    def test_oldest_selection_fairness_serves_unselected_checks_before_repeats(self):
        checks = tuple(self.check(id=UUID(int=100 + i)) for i in range(8))
        first = self.compose(checks=checks)
        next_batch = self.compose(checks=checks, previous=first, now=NOW + timedelta(seconds=30))
        first_ids = {q.check_ref.source_id for q in first.brief.questions}
        next_ids = {q.check_ref.source_id for q in next_batch.brief.questions}
        self.assertEqual(len(first_ids), 4)
        self.assertEqual(len(next_ids), 4)
        self.assertFalse(first_ids & next_ids,
                         "Never-selected equal-priority checks must precede previously selected checks.")
        third = self.compose(checks=checks, previous=next_batch, now=NOW + timedelta(seconds=60))
        self.assertEqual({q.check_ref.source_id for q in third.brief.questions}, first_ids)
        reread = self.compose(checks=checks, previous=third, now=NOW + timedelta(seconds=61))
        self.assertIs(reread.brief, third.brief)
        self.assertEqual(reread.selection_history, third.selection_history)
        self.assertEqual(len(third.selection_history), len(checks))

    def test_activity_transition_preserves_unresolved_checks_and_uncertainty(self):
        check = self.durable()
        cooking = PerceptionActivity(label="cooking", stage="finishing", confidence=0.7)
        leaving = PerceptionActivity(label="leaving", confidence=0.4, uncertainty="May only be moving rooms.")
        first = self.compose(checks=(check,), activity=cooking)
        result = self.compose(checks=(check,), activity=leaving, previous=first)
        self.assertFalse(result.reused)
        self.assertEqual(result.brief.activity, leaving)
        self.assertEqual(result.brief.questions[0].check_ref, first.brief.questions[0].check_ref)
        self.assertEqual(check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(check.resolution)

    def test_replaced_goal_removes_old_assignment_without_global_revision_fence(self):
        old_question = self.question(focus_ref=self.intent)
        first = self.compose(intent_ref=self.intent, explicit_questions=(old_question,))
        replacement = changed(self.intent, revision=2)
        new_question = self.question(id=UUID(int=12), focus_ref=replacement, question="Inspect the replacement goal.")
        result = self.compose(intent_ref=replacement, explicit_questions=(old_question, new_question),
            dependency_eligible=lambda source: source != self.intent, previous=first)
        self.assertFalse(result.reused)
        self.assertEqual(result.brief.intent_ref, replacement)
        self.assertEqual(result.brief.questions, (new_question,))
        self.assertNotIn(self.intent, result.brief.dependencies)

    def test_episode_replacement_is_semantic_even_with_unchanged_check_text(self):
        check = self.check()
        first = self.compose(checks=(check,))
        replacement = changed(check, revision=2, episode_or_object_ref=changed(self.episode, revision=2))
        result = self.compose(checks=(replacement,), previous=first)
        self.assertFalse(result.reused)
        self.assertEqual(result.brief.questions[0].episode_or_object_ref, replacement.episode_or_object_ref)
        self.assertNotIn(self.episode, result.brief.dependencies)

    def test_state_epoch_or_scope_rotation_cannot_reuse_old_assignment(self):
        first = self.compose(checks=(self.check(),))
        result = self.compose(checks=(self.check(),), previous=first,
                              snapshot=changed(self.snapshot, cursor=StateCursor(epoch=uuid4(), revision=1)))
        self.assertFalse(result.reused)
        rotated = self.compose(previous=first, scope_epoch=uuid4())
        self.assertFalse(rotated.reused)
        self.assertNotEqual(rotated.brief.id, first.brief.id)
        self.assertEqual(rotated.brief.version, 1)

    def test_prior_evidence_uncertainty_visibility_and_original_time_are_preserved(self):
        prior = self.prior()
        result = self.compose(prior_evidence=(prior,))
        self.assertEqual(result.brief.prior_evidence, (prior,))
        self.assertEqual(result.brief.prior_evidence[0].reference.visibility, "unclear")
        self.assertEqual(result.brief.prior_evidence[0].uncertainty, "Shutdown is unconfirmed.")

    def test_future_capture_is_rejected_and_prior_order_is_chronological_newest_first(self):
        old = self.prior()
        recent = changed(old, reference=changed(old.reference, captured_at=NOW - timedelta(seconds=1)))
        future = changed(old, reference=changed(old.reference, captured_at=NOW + timedelta(seconds=1)))
        result = self.compose(prior_evidence=(old, future, recent))
        self.assertEqual(result.brief.prior_evidence, (recent, old))
        self.assertIn("future_capture", self.reasons(result))

    def test_dependency_budget_omits_entire_question_instead_of_truncating_provenance(self):
        dependencies = tuple(changed(self.source, source_id=UUID(int=100 + i)) for i in range(16))
        check = self.check(dependency_versions=dependencies)
        result = self.compose(checks=(check,))
        self.assertFalse(result.brief.questions)
        self.assertFalse(result.brief.dependencies)
        self.assertIn("dependency_budget", self.reasons(result))
        self.assertEqual(check.dependency_versions, dependencies)

    def test_conflicting_source_versions_are_not_combined_into_one_brief(self):
        first = self.check(id=UUID(int=100), dependency_versions=(self.source,))
        second = self.check(id=UUID(int=101), dependency_versions=(changed(self.source, generation=4),))
        result = self.compose(checks=(first, second))
        self.assertEqual(len(result.brief.questions), 1)
        self.assertIn("conflicting_dependency_versions", self.reasons(result))
        self.assertEqual(len([d for d in result.brief.dependencies if d.source_id == self.source.source_id]), 1)

    def test_input_overflow_is_reported_for_checks_questions_and_prior_evidence(self):
        checks = tuple(self.check(id=UUID(int=1000 + i)) for i in range(129))
        questions = tuple(self.question(id=UUID(int=2000 + i)) for i in range(9))
        result = self.compose(checks=checks, explicit_questions=questions, prior_evidence=(self.prior(),) * 9)
        budget_gaps = {(d.kind, d.count) for d in result.diagnostics if d.reason == "input_budget"}
        self.assertEqual(budget_gaps, {("check", 1), ("explicit_question", 1), ("prior_evidence", 1)})
        self.assertLessEqual(len(result.brief.questions), 4)
        self.assertLessEqual(len(result.brief.prior_evidence), 8)

    def test_reordered_check_inputs_do_not_trigger_spurious_semantic_replan(self):
        checks = tuple(self.check(id=UUID(int=100 + i)) for i in range(6))
        first = self.compose(checks=checks)
        result = self.compose(checks=tuple(reversed(checks)), previous=first, now=NOW + timedelta(seconds=1))
        self.assertTrue(result.reused)
        self.assertIs(result.brief, first.brief)

    def test_time_must_be_aware_and_cannot_regress_within_composition_epoch(self):
        with self.assertRaises(ValueError):
            self.compose(now=NOW.replace(tzinfo=None))
        first = self.compose()
        with self.assertRaises(ValueError):
            self.compose(previous=first, now=NOW - timedelta(seconds=1))
