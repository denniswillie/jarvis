"""Authoritative speech/result metadata, without model retelling or playback claims."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.attention.tts import OpenAISpeechProvider, SpeechConfiguration
from app.tasks.hermes import HermesTaskProvider
from app.tasks.hermes_result import (
    EXPERT_RESULT_FORMAT, ExpertSpeechChunk, HermesTaskResult, inspect_expert_output,
)
from app.tasks.models import TaskError


def context():
    return SimpleNamespace(tasks=SimpleNamespace(pool=None),
        task=SimpleNamespace(id=uuid4(), owner_id="fixture-owner", revision=3, model="fixture-model"),
        claim=SimpleNamespace(task_revision=3, attempt_id=uuid4()))


def result(**changes):
    return HermesTaskResult.model_validate({
        "status":"succeeded", "answer":"mid = (low + high) // 2",
        "speech":[{"text":"Add low and high in parentheses, then integer-divide by two."}],
        "evidence":[], "completed_operation_refs":[], "uncertainties":[], **changes,
    })


class ExpertOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifier_stamps_exact_content_and_trusted_task_attempt(self):
        ctx = context()
        expert = result()
        payload = (await expert.verified(ctx, SimpleNamespace())).payload
        self.assertEqual(payload["format"], EXPERT_RESULT_FORMAT)
        self.assertEqual(payload["authorship"]["task_id"], str(ctx.task.id))
        self.assertEqual(payload["authorship"]["attempt_id"], str(ctx.claim.attempt_id))
        self.assertEqual(payload["authorship"]["task_revision"], 3)
        self.assertEqual(payload["authorship"]["model"], ctx.task.model)
        original = deepcopy(payload)
        output = inspect_expert_output(payload, task=ctx.task)
        self.assertTrue(output.expert_available)
        self.assertTrue(output.speech_ready)
        self.assertEqual(output.content, expert)
        self.assertEqual(payload, original)

    async def test_tampered_written_or_spoken_content_fails_closed(self):
        ctx = context()
        payload = (await result().verified(ctx, SimpleNamespace())).payload
        changes = ({"answer":"Unsupported rewritten answer."},
            {"speech":[{"text":"A different speaker invented this."}]},
            {"uncertainties":["Changed uncertainty."]}, {"status":"failed"},
            {"authorship":None}, {"speech":[]}, {"completed_operation_refs":["invented"]})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(TaskError) as error:
                inspect_expert_output(payload | change, task=ctx.task)
            self.assertEqual(error.exception.code, "task_result_unavailable")

    async def test_foreign_owner_task_revision_and_model_cannot_reuse_output(self):
        ctx = context()
        payload = (await result().verified(ctx, SimpleNamespace())).payload
        for changed in ({"owner_id":"foreign"}, {"id":uuid4()}, {"revision":4}, {"model":"other-model"}):
            task = SimpleNamespace(**(vars(ctx.task) | changed))
            with self.subTest(changed=changed), self.assertRaises(TaskError):
                inspect_expert_output(payload, task=task)

    async def test_clarification_uncertainty_and_multichunk_speech_are_preserved(self):
        ctx = context()
        expert = result(status="waiting_for_input", answer="Please show the switch. I cannot verify its state.",
            speech=[{"text":"Please show the switch."}, {"text":"I cannot verify its state."}],
            uncertainties=["No current original image."])
        payload = (await expert.verified(ctx, SimpleNamespace())).payload
        output = inspect_expert_output(payload, task=ctx.task)
        self.assertTrue(output.speech_ready)
        self.assertEqual(output.content.model_dump(mode="json"), expert.model_dump(mode="json"))

    async def test_runtime_failure_is_neither_expert_answer_nor_speech(self):
        payload = HermesTaskProvider.incomplete("hermes_native_incomplete").payload
        before = deepcopy(payload)
        output = inspect_expert_output(payload, task=context().task)
        self.assertEqual(output.kind, "runtime")
        self.assertFalse(output.expert_available)
        self.assertFalse(output.speech_ready)
        self.assertIsNone(output.content)
        self.assertEqual(payload, before)

    def test_legacy_runtime_and_minimal_answers_do_not_invent_authorship(self):
        for payload in ({"answer":"An ambiguous older result."},
            {"format":"jarvis.hermes.task_result.v1", "answer":"I could not finish this task.",
             "uncertainties":["hermes_native_incomplete"]},
            {"format":"unknown.future.format", **result().model_dump(mode="json")}):
            with self.subTest(payload=payload):
                output = inspect_expert_output(payload, task=context().task)
                self.assertEqual(output.kind, "unverified")
                self.assertFalse(output.expert_available)
                self.assertFalse(output.speech_ready)

    def test_complete_legacy_written_result_needs_owning_expert_speech_preparation(self):
        payload = result().model_dump(mode="json", exclude={"speech"})
        payload["format"] = "jarvis.hermes.task_result.v1"
        output = inspect_expert_output(payload, task=context().task)
        self.assertEqual(output.kind, "legacy_expert")
        self.assertTrue(output.expert_available)
        self.assertFalse(output.speech_ready)
        self.assertEqual(output.unavailable_reason, "expert_speech_not_prepared")
        self.assertEqual(output.content.answer, payload["answer"])

    def test_provider_schema_cannot_supply_server_authorship(self):
        schema = HermesTaskResult.model_json_schema()
        self.assertNotIn("authorship", schema["properties"])
        self.assertNotIn("format", schema["properties"])
        self.assertEqual(set(schema["properties"]), set(schema["required"]))
        for change in ({"authorship":{"kind":"expert"}}, {"format":EXPERT_RESULT_FORMAT}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                result(**change)
        raw = result().model_dump(mode="json", exclude={"speech"})
        with self.assertRaises(ValidationError):
            HermesTaskResult.model_validate(raw)

    def test_speech_utf8_nul_empty_and_chunk_bounds(self):
        for value in ("", " \n ", "x\x00y", "a" * 1601, "\u03bb" * 801):
            with self.subTest(length=len(value)), self.assertRaises(ValidationError):
                ExpertSpeechChunk(text=value)
        self.assertEqual(ExpertSpeechChunk(text="\u03bb" * 800).text, "\u03bb" * 800)
        with self.assertRaises(ValidationError):
            result(speech=[{"text":"x"}] * 65)

    def test_absent_result_has_no_provenance_or_delivery_claim(self):
        output = inspect_expert_output(None, task=context().task)
        self.assertEqual(output.kind, "absent")
        self.assertFalse(output.expert_available)
        self.assertFalse(output.speech_ready)

    async def test_verified_speech_fits_default_tts_without_rewriting(self):
        client = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace())
        tts = OpenAISpeechProvider(client, SpeechConfiguration())
        ctx = context()
        text = "\u03bb" * 800
        payload = (await result(speech=[{"text":text}]).verified(ctx, SimpleNamespace())).payload
        output = inspect_expert_output(payload, task=ctx.task)
        request = tts.prepare(output.content.speech[0].text)
        self.assertEqual(request.text, text)
        self.assertEqual(payload["speech"], [{"text":text}])
