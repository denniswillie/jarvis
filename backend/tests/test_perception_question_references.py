"""One real analyzer's fixed-assignment handles and typed observation mapping.

Storage/model fixtures are offline. No check owner is invoked or settled here.
"""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.live_state.models import GuardedPayloadBinding, SessionScope, SourceDependency
from app.live_state.perception import (
    DurableResponsibilityCapability, ObservationalQuestion, PerceptionBrief,
    PerceptionEvidence, PerceptionFrameReference,
)
from app.perception.contracts import (
    FastPerceptionFrame, FastPerceptionOutputError, FastPerceptionRequest, parse_fast_perception_output,
)
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.live_brief import seal_live_brief_input
from app.perception.prompts import build_fast_perception_prompt
from app.perception.references import PerceptionReferences
from app.perception.result_validation import parse_referenced_perception_output
from app.tasks.models import digest


def make_request(*, brief=True, repeated_check=False):
    now = datetime.now(UTC) - timedelta(seconds=1)
    scope = SessionScope(owner_id="question-fixture", session_id=uuid4(), area_id=uuid4())
    episode = SourceDependency(source_namespace="episode", source_id=uuid4(), revision=3, generation=2)
    check = SourceDependency(source_namespace="situational_check", source_id=uuid4(), revision=7, generation=5)
    another = check if repeated_check else SourceDependency(source_namespace="situational_check", source_id=uuid4(), revision=4)
    questions = (
        ObservationalQuestion(id=uuid4(), question="Are the stove controls visible?", check_ref=check, episode_or_object_ref=episode),
        ObservationalQuestion(id=uuid4(), question="What position is the visible switch in?", check_ref=another, episode_or_object_ref=episode),
        ObservationalQuestion(id=uuid4(), question="Is there a meaningful activity change?"),
    )
    dependencies = (episode, check) if repeated_check else (episode, check, another)
    assignment = PerceptionBrief(id=uuid4(), version=2, scope=scope, scope_epoch=uuid4(),
        based_on_live_revision=8, questions=questions, dependencies=dependencies,
        durable_responsibility=DurableResponsibilityCapability(owner_namespace="fixture", status="unavailable"),
        created_at=now, fresh_until=now + timedelta(minutes=1), expires_at=now + timedelta(minutes=2))
    binding = GuardedPayloadBinding(scope=scope, payload_root_ids=(uuid4(),), dependencies=dependencies)
    request = FastPerceptionRequest(owner_id=scope.owner_id, session_id=scope.session_id, window_id=uuid4(),
        frames=(FastPerceptionFrame(uuid4(), 1, now, "glasses", "image/jpeg", b"fixture"),),
        previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
        behavior_instructions="Observe, do not act.", active_reminders=(), active_focus=None,
        recent_interrupts=(), requested_at=now,
        perception_brief=assignment if brief else None, perception_brief_binding=binding if brief else None)
    if brief:
        # A local receipt fixture for the real deterministic capture seal. Other
        # suites exercise canonical publication/transaction authority separately.
        snapshot = ObservationInputSnapshot(scope.owner_id, scope.session_id, uuid4(), digest(request_document(request)))
        request = seal_live_brief_input(request, snapshot)
    return request


def finding(**changes):
    return dict(question_id="question:0", check_id="check:0", visibility="unclear",
        literal_finding="The controls are too blurry to read.", confidence=.7,
        evidence_frame_ids=["frame:0"]) | changes


def payload(*findings):
    return dict(summary="A kitchen is visible.", starts_new_event=False, confidence=.8,
                perception_findings=list(findings))


def parse(request, *findings, refs=None):
    return parse_referenced_perception_output(json.dumps(payload(*findings)), request=request,
        references=refs or PerceptionReferences(request))


class QuestionRegistryTests(unittest.TestCase):
    def test_exact_request_local_question_and_check_registries_are_immutable(self):
        request = make_request()
        refs = PerceptionReferences(request)
        first = request.perception_brief.questions[0]
        self.assertEqual(refs.questions["question:0"], str(first.id))
        self.assertEqual(refs.checks["check:0"], str(first.check_ref.source_id))
        with self.assertRaises(TypeError):
            refs.questions["question:0"] = str(uuid4())
        with self.assertRaises(TypeError):
            refs.checks["check:0"] = str(uuid4())

    def test_shared_check_has_one_handle_and_preserves_question_pairs(self):
        request = make_request(repeated_check=True)
        refs = PerceptionReferences(request)
        self.assertEqual(len(refs.checks), 1)
        result = parse(request, finding(), finding(question_id="question:1"))
        self.assertEqual(len(result.output.perception_findings), 2)
        self.assertEqual(result.output.perception_findings[0].check_ref, result.output.perception_findings[1].check_ref)

    def test_registry_cannot_be_reused_for_changed_brief_binding_or_capture(self):
        request = make_request()
        refs = PerceptionReferences(request)
        brief = request.perception_brief
        changes = (
            {"version": brief.version + 1}, {"id": uuid4()}, {"scope_epoch": uuid4()},
            {"questions": tuple(reversed(brief.questions))}, {"based_on_live_revision": 9},
        )
        for change in changes:
            with self.subTest(change=tuple(change)), self.assertRaisesRegex(FastPerceptionOutputError, "scope"):
                parse(replace(request, perception_brief=brief.model_copy(update=change)), finding(), refs=refs)
        for changed in (
            replace(request, perception_brief_input_digest="0" * 64),
            replace(request, perception_brief_binding=request.perception_brief_binding.model_copy(update={"payload_root_ids": (uuid4(),)})),
            replace(request, window_id=uuid4()),
        ):
            with self.assertRaisesRegex(FastPerceptionOutputError, "scope"):
                parse(changed, finding(), refs=refs)

    def test_question_dependency_must_be_bound_with_exact_version(self):
        request = make_request()
        brief = request.perception_brief
        question = brief.questions[0]
        altered = question.model_copy(update={"check_ref": question.check_ref.model_copy(update={"generation": 6})})
        malformed = replace(request, perception_brief=brief.model_copy(update={"questions": (altered,)}))
        with self.assertRaisesRegex(FastPerceptionOutputError, "unbound_question"):
            PerceptionReferences(malformed)

    def test_same_check_uuid_cannot_select_between_conflicting_versions(self):
        request = make_request()
        brief = request.perception_brief
        question = brief.questions[0]
        conflicting = question.check_ref.model_copy(update={"revision": 99})
        second = brief.questions[1].model_copy(update={"check_ref": conflicting})
        # This deliberately uses model_copy to exercise registry protection even
        # when a caller bypasses the canonical binding constructor's validator.
        deps = brief.dependencies + (conflicting,)
        malformed = replace(request, perception_brief=brief.model_copy(update={"questions": (question, second), "dependencies": deps}),
            perception_brief_binding=request.perception_brief_binding.model_copy(update={"dependencies": deps}))
        with self.assertRaisesRegex(FastPerceptionOutputError, "ambiguous_check"):
            PerceptionReferences(malformed)

    def test_prompt_copies_handles_but_keeps_original_read_only_versions(self):
        request = make_request()
        prompt = build_fast_perception_prompt(request, references=PerceptionReferences(request))
        context = json.loads(prompt.split("Context JSON:\n```json\n")[1].split("\n```")[0])
        assignment = context["perception_assignment"]
        self.assertEqual(assignment["assignment_reference_version"], 1)
        self.assertEqual(assignment["questions"][0]["id"], "question:0")
        self.assertEqual(assignment["questions"][0]["check_id"], "check:0")
        self.assertEqual(assignment["questions"][0]["check_ref"], request.perception_brief.questions[0].check_ref.model_dump(mode="json"))
        self.assertEqual(assignment["questions"][2]["check_id"], None)
        self.assertIn("Visibility reports the quality of the view, not completion", prompt)
        self.assertIn("perception_findings", prompt)


class QuestionMappingTests(unittest.TestCase):
    def test_handles_map_to_frozen_typed_evidence_with_versions(self):
        request = make_request()
        result = parse(request, finding())
        self.assertEqual(result.rejected, ())
        mapped, = result.output.perception_findings
        self.assertEqual(mapped.question_id, request.perception_brief.questions[0].id)
        self.assertEqual(mapped.check_ref, request.perception_brief.questions[0].check_ref)
        self.assertEqual((mapped.check_ref.revision, mapped.check_ref.generation), (7, 5))
        self.assertEqual(mapped.evidence_frame_ids, (request.primary_frame.frame_id,))
        self.assertNotIn("question:0", result.output.model_dump_json())
        self.assertNotIn("check:0", result.output.model_dump_json())
        self.assertEqual(parse_fast_perception_output(result.output.model_dump_json(), request=request), result.output)
        evidence = PerceptionEvidence(brief=request.perception_brief, window_id=request.window_id,
            camera_stream_epoch=uuid4(), frames=tuple(PerceptionFrameReference(frame_id=f.frame_id,
                frame_sequence=f.frame_sequence, captured_at=f.captured_at) for f in request.frames),
            requested_at=request.requested_at, analyzed_at=datetime.now(UTC), findings=tuple(result.output.perception_findings))
        self.assertEqual(evidence.findings, (mapped,))

    def test_all_visibility_values_remain_observations_not_resolution(self):
        request = make_request()
        for visibility in ("visible", "not_visible", "unclear"):
            with self.subTest(visibility=visibility):
                output = parse(request, finding(visibility=visibility)).output
                mapped, = output.perception_findings
                self.assertEqual(mapped.visibility, visibility)
                self.assertNotIn("resolved", mapped.model_dump())
                self.assertNotIn("completed", mapped.model_dump())
                self.assertEqual(output.prospective_matches, [])
                self.assertIsNone(output.interrupt_candidate)

    def test_bad_optional_reference_does_not_discard_scene_or_other_question(self):
        request = make_request()
        invalid = (
            {"question_id": "question:99"}, {"question_id": "frame:0"},
            {"check_id": "question:0"}, {"check_id": "check:1"}, {"check_id": None},
            {"question_id": str(uuid4())}, {"check_id": str(uuid4())},
            {"evidence_frame_ids": ["frame:99"]}, {"evidence_frame_ids": ["frame:0", "frame:0"]},
            {"check_ref": request.perception_brief.questions[0].check_ref.model_dump(mode="json")},
            {"resolved": True}, {"visibility": "completed"}, {"literal_finding": ""},
        )
        for changes in invalid:
            with self.subTest(changes=tuple(changes)):
                result = parse(request, finding(**changes), finding(question_id="question:1", check_id="check:1"))
                self.assertEqual(result.output.summary, "A kitchen is visible.")
                self.assertEqual(len(result.output.perception_findings), 1)
                self.assertEqual(result.output.perception_findings[0].question_id, request.perception_brief.questions[1].id)
                self.assertEqual(len(result.rejected), 1)

    def test_unbound_question_can_only_report_null_check(self):
        request = make_request()
        result = parse(request, finding(question_id="question:2", check_id=None))
        self.assertIsNone(result.output.perception_findings[0].check_ref)
        rejected = parse(request, finding(question_id="question:2"))
        self.assertEqual(rejected.output.perception_findings, [])

    def test_duplicate_conflicting_findings_are_both_rejected_not_first_wins(self):
        request = make_request()
        result = parse(request, finding(visibility="visible"), finding(visibility="not_visible"),
                       finding(question_id="question:1", check_id="check:1"))
        self.assertEqual(len(result.output.perception_findings), 1)
        self.assertEqual([r.reason for r in result.rejected], ["duplicate_question_finding"] * 2)

    def test_oversized_collection_is_rejected_without_discarding_scene(self):
        result = parse(make_request(), *(finding() for _ in range(5)))
        self.assertEqual(result.output.perception_findings, [])
        self.assertEqual(result.rejected[0].reason, "invalid_candidate_collection")
        self.assertEqual(result.output.summary, "A kitchen is visible.")

    def test_canonical_parser_rejects_forged_check_versions_and_foreign_frames(self):
        request = make_request()
        good = parse(request, finding()).output.model_dump(mode="json")
        for changes in (
            {"check_ref": request.perception_brief.questions[0].check_ref.model_copy(update={"revision": 8}).model_dump(mode="json")},
            {"evidence_frame_ids": [str(uuid4())]},
            {"question_id": str(uuid4())},
        ):
            raw = {**good, "perception_findings": [{**good["perception_findings"][0], **changes}]}
            with self.subTest(changes=tuple(changes)), self.assertRaises(FastPerceptionOutputError):
                parse_fast_perception_output(json.dumps(raw), request=request)

    def test_legacy_requests_remain_valid_and_cannot_acquire_check_authority(self):
        request = make_request(brief=False)
        refs = PerceptionReferences(request)
        self.assertEqual(dict(refs.questions), {})
        self.assertEqual(dict(refs.checks), {})
        result = parse(request, finding())
        self.assertEqual(result.output.summary, "A kitchen is visible.")
        self.assertEqual(result.output.perception_findings, [])
        self.assertEqual(len(result.rejected), 1)
        output = parse_fast_perception_output(json.dumps({"summary": "Legacy scene.", "starts_new_event": False, "confidence": .7}), request=request)
        self.assertEqual(output.perception_findings, [])
        self.assertNotIn("perception_findings", build_fast_perception_prompt(request))

    def test_uuid_reference_mode_and_check_pairing(self):
        request = make_request()
        refs = PerceptionReferences(request, short_handles=False)
        question = request.perception_brief.questions[0]
        result = parse(request, finding(question_id=str(question.id), check_id=str(question.check_ref.source_id),
            evidence_frame_ids=[str(request.primary_frame.frame_id)]), refs=refs)
        self.assertEqual(result.rejected, ())
        prompt = build_fast_perception_prompt(request)
        self.assertIn('"check_id": "' + str(question.check_ref.source_id) + '"', prompt)


class RealQuestionAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_single_analyzer_publishes_typed_findings(self):
        request = make_request()
        model = SimpleNamespace(provider="fixture", model="fixture",
            generate_temporal_json=AsyncMock(return_value=json.dumps(payload(finding()))))
        output = await FastPerceptionAnalyzer(model).analyze(request)
        model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(output.perception_findings[0].check_ref, request.perception_brief.questions[0].check_ref)
        kwargs = model.generate_temporal_json.await_args.kwargs
        self.assertIn('"id": "question:0"', kwargs["prompt"])
        self.assertIn('"check_id": "check:0"', kwargs["prompt"])
        self.assertIn("frame:0", kwargs["images"][0].label)
