"""Local schema fixtures only: no database, services, models or lifecycle execution."""
from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import ValidationError

from app.live_state.models import SessionScope, SourceDependency
from app.live_state.perception import (
    CheckLifecycle,
    CheckResolution,
    DurableResponsibilityCapability,
    EvidenceReference,
    FutureCue,
    ObservationalQuestion,
    PerceptionBrief,
    PerceptionEvidence,
    PerceptionFinding,
    PerceptionFrameReference,
    PersistentTrackingOutcome,
    PriorEvidence,
    SituationalCheck,
)


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def identity(number):
    return UUID(int=number)


def reference(number, namespace, revision=1, generation=None):
    return SourceDependency(
        source_namespace=namespace, source_id=identity(number),
        revision=revision, generation=generation,
    )


def scope():
    return SessionScope(owner_id="fixture-owner", session_id=identity(1), area_id=None)


def episode(revision=2, number=40):
    return reference(number, "episode", revision)


def owner_reference():
    return reference(10, "responsibility", 5)


def check_reference(revision=3):
    return reference(30, "responsibility_check", revision)


def capability(status="disabled", owner_namespace="responsibility"):
    return DurableResponsibilityCapability(owner_namespace=owner_namespace, status=status)


def question(**changes):
    values = dict(
        id=identity(20), question="Are the controls visible, and what do they indicate?",
        check_ref=check_reference(), episode_or_object_ref=episode(),
    )
    values.update(changes)
    return ObservationalQuestion(**values)


def source_evidence(**changes):
    values = dict(
        source=reference(60, "knowledge_source", 7, 2),
        captured_at=NOW - timedelta(seconds=2), episode_or_object_ref=episode(),
    )
    values.update(changes)
    return EvidenceReference(**values)


def brief(**changes):
    values = dict(
        id=identity(2), version=1, scope=scope(), scope_epoch=identity(3),
        based_on_live_revision=5, durable_responsibility=capability(),
        created_at=NOW, fresh_until=NOW + timedelta(seconds=10),
        expires_at=NOW + timedelta(seconds=30),
    )
    values.update(changes)
    return PerceptionBrief(**values)


def frame(**changes):
    values = dict(
        frame_id=identity(50), frame_sequence=1, captured_at=NOW - timedelta(seconds=1),
        source=reference(50, "visual_frame"),
    )
    values.update(changes)
    return PerceptionFrameReference(**values)


def finding(**changes):
    values = dict(
        question_id=identity(20), check_ref=check_reference(), visibility="visible",
        literal_finding="The visible indicator reads OFF.", confidence=0.9,
        evidence_frame_ids=(identity(50),),
    )
    values.update(changes)
    return PerceptionFinding(**values)


def result(**changes):
    values = dict(
        brief=brief(questions=(question(),)), window_id=identity(70),
        camera_stream_epoch=identity(71), frames=(frame(),),
        requested_at=NOW + timedelta(seconds=1), analyzed_at=NOW + timedelta(seconds=2),
        findings=(finding(),),
    )
    values.update(changes)
    return PerceptionEvidence(**values)


def resolution(**changes):
    values = dict(
        basis="user_report", source_refs=(source_evidence(),),
        resolved_at=NOW + timedelta(seconds=2), applicable_episode_or_object_ref=episode(),
    )
    values.update(changes)
    return CheckResolution(**values)


def check(**changes):
    values = dict(
        id=identity(30), revision=3, scope=scope(), episode_or_object_ref=episode(),
        origin="explicit_request", continuity="durable", canonical_owner_ref=owner_reference(),
        question="Is shutdown established for this cooking episode?",
        lifecycle=CheckLifecycle.TRACKING,
    )
    values.update(changes)
    return SituationalCheck(**values)


def outcome(**changes):
    values = dict(scope=scope(), capability=capability(), status="unavailable")
    values.update(changes)
    return PersistentTrackingOutcome(**values)


def cue(**changes):
    values = dict(
        id=identity(80), revision=1, scope=scope(), episode_or_object_ref=episode(),
        question="Is the shutdown check still unresolved?", activation_cues=("departure",),
        created_at=NOW, expires_at=NOW + timedelta(minutes=5),
    )
    values.update(changes)
    return FutureCue(**values)


class LivePerceptionContractTests(unittest.TestCase):
    def assert_round_trip(self, value):
        restored = type(value).model_validate_json(value.model_dump_json())
        self.assertEqual(restored, value)
        return restored

    def test_minimal_brief_with_disabled_owner(self):
        value = self.assert_round_trip(brief())
        self.assertEqual(value.durable_responsibility.status, "disabled")
        self.assertEqual(value.questions, ())
        self.assertEqual(value.prior_evidence, ())
        self.assertIsNone(value.activity)

    def test_persistent_acceptance_requires_available_matching_owner(self):
        accepted = outcome(
            status="accepted", capability=capability("available"),
            canonical_owner_ref=owner_reference(), check_ref=check_reference(),
        )
        self.assertEqual(self.assert_round_trip(accepted).status, "accepted")
        for status in ("disabled", "unavailable"):
            with self.subTest(status=status), self.assertRaises(ValidationError):
                outcome(
                    status="accepted", capability=capability(status),
                    canonical_owner_ref=owner_reference(), check_ref=check_reference(),
                )
        for missing in ("canonical_owner_ref", "check_ref"):
            values = dict(
                status="accepted", capability=capability("available"),
                canonical_owner_ref=owner_reference(), check_ref=check_reference(),
            )
            values[missing] = None
            with self.subTest(missing=missing), self.assertRaises(ValidationError):
                outcome(**values)
        with self.assertRaises(ValidationError):
            outcome(canonical_owner_ref=reference(10, "visual_focus"))

    def test_unavailable_owner_preserves_existing_durable_projection(self):
        stored = check()
        for status in ("disabled", "unavailable"):
            with self.subTest(status=status):
                receipt = outcome(
                    capability=capability(status), canonical_owner_ref=owner_reference(),
                    check_ref=check_reference(),
                )
                self.assertEqual(self.assert_round_trip(receipt).status, "unavailable")
                self.assertEqual(stored.lifecycle, CheckLifecycle.TRACKING)
                self.assertIsNone(stored.resolution)

    def test_durable_continuity_requires_owner_but_ephemeral_does_not(self):
        with self.assertRaises(ValidationError):
            check(canonical_owner_ref=None)
        ephemeral = check(continuity="ephemeral", canonical_owner_ref=None)
        self.assertIsNone(self.assert_round_trip(ephemeral).canonical_owner_ref)

    def test_expiry_dismissal_and_snooze_do_not_resolve_tracking(self):
        tracking = check(
            last_observed_at=NOW - timedelta(seconds=10),
            observation_window_expires_at=NOW - timedelta(seconds=1),
            snoozed_until=NOW + timedelta(minutes=1),
        )
        self.assertEqual(tracking.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(tracking.resolution)
        for lifecycle in CheckLifecycle:
            if lifecycle == CheckLifecycle.RESOLVED:
                continue
            with self.subTest(lifecycle=lifecycle):
                self.assertIsNone(check(lifecycle=lifecycle).resolution)
                with self.assertRaises(ValidationError):
                    check(lifecycle=lifecycle, resolution=resolution())
        with self.assertRaises(ValidationError):
            check(lifecycle=CheckLifecycle.RESOLVED)

    def test_explicit_resolution_bases_round_trip(self):
        visual = source_evidence(frame_id=identity(50), visibility="visible")
        for basis in ("user_report", "trusted_source", "visual_observation"):
            with self.subTest(basis=basis):
                evidence = (visual,) if basis == "visual_observation" else (source_evidence(),)
                value = check(
                    lifecycle=CheckLifecycle.RESOLVED,
                    resolution=resolution(basis=basis, source_refs=evidence),
                )
                self.assertEqual(self.assert_round_trip(value).resolution.basis, basis)
        with self.assertRaises(ValidationError):
            resolution(source_refs=())
        with self.assertRaises(ValidationError):
            resolution(basis="expired")

    def test_all_visibility_findings_are_observations_not_resolutions(self):
        for visibility in ("visible", "unclear", "not_visible"):
            with self.subTest(visibility=visibility):
                observation = result(findings=(finding(visibility=visibility),))
                restored = self.assert_round_trip(observation)
                self.assertEqual(restored.findings[0].visibility, visibility)
                self.assertNotIn("resolution", restored.model_dump())
        with self.assertRaises(ValidationError):
            finding(visibility="absent")

    def test_visual_resolution_requires_visible_frame_evidence(self):
        for visibility in ("unclear", "not_visible"):
            with self.subTest(visibility=visibility), self.assertRaises(ValidationError):
                resolution(
                    basis="visual_observation",
                    source_refs=(source_evidence(frame_id=identity(50), visibility=visibility),),
                )
        with self.assertRaises(ValidationError):
            resolution(basis="visual_observation")
        for values in ({"frame_id": identity(50)}, {"visibility": "visible"}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                source_evidence(**values)

    def test_resolution_rejects_different_episode_identity_or_revision(self):
        for other in (episode(number=41), episode(revision=3)):
            with self.subTest(other=other):
                with self.assertRaises(ValidationError):
                    resolution(applicable_episode_or_object_ref=other)
                with self.assertRaises(ValidationError):
                    check(
                        episode_or_object_ref=other, lifecycle=CheckLifecycle.RESOLVED,
                        resolution=resolution(),
                    )

    def test_historical_generation_is_preserved_not_certified_current(self):
        old = reference(60, "knowledge_source", 7, 2)
        withdrawn = reference(60, "knowledge_source", 7, 3)
        prior = PriorEvidence(reference=source_evidence(source=old), finding="Earlier indicator was off.", confidence=0.7)
        observation = result(
            brief=brief(prior_evidence=(prior,), dependencies=(old,), based_on_live_revision=900),
            findings=(), camera_stream_epoch=identity(900),
        )
        restored = self.assert_round_trip(observation)
        self.assertEqual(restored.brief.dependencies[0].revision, withdrawn.revision)
        self.assertNotEqual(restored.brief.dependencies[0].generation, withdrawn.generation)
        self.assertEqual(restored.brief.prior_evidence[0].reference.captured_at, NOW - timedelta(seconds=2))
        payload = json.loads(observation.model_dump_json())
        self.assertEqual(payload["brief"]["dependencies"][0]["generation"], 2)
        # No current authority is consulted: runtime rejection of old generation is P1/P2 work.

    def test_fixed_brief_window_and_epochs_survive_serialization(self):
        observation = result(
            brief=brief(questions=(question(),), version=4, scope_epoch=identity(300)),
            analyzed_at=NOW + timedelta(minutes=1),
        )
        restored = self.assert_round_trip(observation)
        self.assertEqual(restored.brief.version, 4)
        self.assertEqual(restored.brief.scope_epoch, identity(300))
        self.assertEqual(restored.window_id, identity(70))
        self.assertEqual(restored.camera_stream_epoch, identity(71))
        self.assertGreater(restored.analyzed_at, restored.brief.expires_at)
        # A late historical envelope is representable, not automatically a fresh answer.

    def test_question_requires_episode_for_a_check(self):
        with self.assertRaises(ValidationError):
            question(episode_or_object_ref=None)
        inspection = question(check_ref=None, episode_or_object_ref=None)
        observation = result(brief=brief(questions=(inspection,)), findings=(finding(check_ref=None),))
        self.assertIsNone(self.assert_round_trip(observation).findings[0].check_ref)

    def test_findings_reject_unknown_questions_or_changed_check_references(self):
        invalid = (
            finding(question_id=identity(999)),
            finding(check_ref=check_reference(revision=4)),
            finding(check_ref=reference(31, "responsibility_check", 3)),
            finding(check_ref=None),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValidationError):
                result(findings=(candidate,))
        with self.assertRaises(ValidationError):
            brief(questions=(question(), question()))
        with self.assertRaises(ValidationError):
            result(findings=(finding(), finding()))

    def test_finding_frames_must_be_unique_and_in_actual_window(self):
        for identifiers in ((identity(999),), (identity(50), identity(50))):
            with self.subTest(identifiers=identifiers), self.assertRaises(ValidationError):
                result(findings=(finding(evidence_frame_ids=identifiers),))
        with self.assertRaises(ValidationError):
            finding(evidence_frame_ids=())
        with self.assertRaises(ValidationError):
            result(frames=(frame(), frame()))

    def test_model_handles_cannot_leak_into_internal_uuid_fields(self):
        for factory, values in (
            (frame, {"frame_id": "frame:0"}),
            (question, {"id": "question:0"}),
            (finding, {"evidence_frame_ids": ("frame:0",)}),
        ):
            with self.subTest(factory=factory.__name__), self.assertRaises(ValidationError):
                factory(**values)

    def test_four_question_and_frame_window_bounds(self):
        questions = tuple(question(id=identity(100 + index)) for index in range(4))
        frames = tuple(frame(
            frame_id=identity(200 + index), frame_sequence=index,
            captured_at=NOW - timedelta(seconds=4 - index),
        ) for index in range(4))
        value = result(brief=brief(questions=questions), frames=frames, findings=())
        self.assertEqual(len(value.brief.questions), 4)
        self.assertEqual(len(value.frames), 4)
        with self.assertRaises(ValidationError):
            brief(questions=questions + (question(id=identity(104)),))
        with self.assertRaises(ValidationError):
            result(frames=frames + (frame(frame_id=identity(204)),))
        with self.assertRaises(ValidationError):
            result(frames=())

    def test_auxiliary_collection_bounds(self):
        prior = PriorEvidence(reference=source_evidence(), finding="Earlier observation.", confidence=0.5)
        self.assertEqual(len(brief(scene_change_cues=("one", "two", "three")).scene_change_cues), 3)
        cases = (
            (brief, {"scene_change_cues": ("cue",) * 4}),
            (brief, {"prior_evidence": (prior,) * 9}),
            (brief, {"dependencies": (owner_reference(),) * 17}),
            (result, {"input_gaps": ("missing",) * 9}),
            (result, {"meaningful_scene_or_activity_changes": ("change",) * 4}),
            (check, {"activation_cues": ("cue",) * 4}),
            (check, {"evidence_refs": (source_evidence(),) * 9}),
            (check, {"dependency_versions": (owner_reference(),) * 17}),
            (resolution, {"source_refs": (source_evidence(),) * 9}),
            (cue, {"activation_cues": ()}),
            (cue, {"activation_cues": ("cue",) * 4}),
        )
        for factory, values in cases:
            with self.subTest(factory=factory.__name__, field=next(iter(values))), self.assertRaises(ValidationError):
                factory(**values)

    def test_scalar_bounds_and_unknown_fields(self):
        cases = (
            (brief, {"version": 0}), (brief, {"based_on_live_revision": -1}),
            (check, {"revision": 0}), (question, {"question": ""}),
            (question, {"question": "x" * 1001}), (finding, {"confidence": 1.1}),
            (finding, {"confidence": -0.1}), (brief, {"unrecognized": True}),
        )
        for factory, values in cases:
            with self.subTest(factory=factory.__name__, values=values), self.assertRaises(ValidationError):
                factory(**values)

    def test_aware_times_and_temporal_order_constraints(self):
        cases = (
            (brief, {"created_at": NOW.replace(tzinfo=None)}),
            (source_evidence, {"captured_at": NOW.replace(tzinfo=None)}),
            (brief, {"fresh_until": NOW - timedelta(seconds=1)}),
            (brief, {"expires_at": NOW + timedelta(seconds=1)}),
            (result, {"analyzed_at": NOW}),
            (result, {"requested_at": NOW - timedelta(seconds=1)}),
            (result, {"frames": (frame(captured_at=NOW + timedelta(seconds=2)),)}),
            (result, {"frames": (frame(), frame(frame_id=identity(51), captured_at=NOW - timedelta(seconds=2)))}),
            (check, {"last_observed_at": NOW, "observation_window_expires_at": NOW - timedelta(seconds=1)}),
            (resolution, {"resolved_at": NOW - timedelta(seconds=3)}),
            (cue, {"expires_at": NOW}),
        )
        for factory, values in cases:
            with self.subTest(factory=factory.__name__, field=next(iter(values))), self.assertRaises(ValidationError):
                factory(**values)
        future_prior = PriorEvidence(
            reference=source_evidence(captured_at=NOW + timedelta(seconds=1)),
            finding="Not prior evidence.", confidence=0.5,
        )
        with self.assertRaises(ValidationError):
            brief(prior_evidence=(future_prior,))

    def test_contracts_and_nested_collections_are_immutable(self):
        value = brief(questions=(question(),))
        with self.assertRaises(ValidationError):
            value.version = 2
        with self.assertRaises(ValidationError):
            value.questions[0].question = "Changed question"
        with self.assertRaises(TypeError):
            value.questions[0] = question()
        with self.assertRaises(AttributeError):
            value.questions.append(question())
        restored = self.assert_round_trip(result())
        self.assertIsInstance(restored.frames, tuple)
        self.assertIsInstance(restored.findings, tuple)
        self.assertIsInstance(restored.findings[0].evidence_frame_ids, tuple)

    def test_future_cue_is_not_a_completion_or_persistence_receipt(self):
        value = self.assert_round_trip(cue())
        self.assertIsNone(value.check_ref)
        self.assertNotIn("resolution", value.model_dump())
        self.assertNotIn("canonical_owner_ref", value.model_dump())
        with self.assertRaises(ValidationError):
            cue(resolution=resolution())


if __name__ == "__main__":
    unittest.main(verbosity=2)
