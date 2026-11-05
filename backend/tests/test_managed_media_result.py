import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.sandbox.media import ManagedMediaTools
from app.tasks.hermes_result import HermesTaskResult, inspect_expert_output
from app.tasks.media_result import verified_media_result
from app.tasks.models import ClaimLost, TaskError
from app.workspace.media_recipe import MediaRecipeError, RetainedRef
from app.workspace.media_render import AcceptedMediaOutput
from test_second_brain_hermes import context


class ManagedMediaResultTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = context()
        self.context.tasks.pool = None
        self.plan = "a" * 64
        self.reference = "op_" + uuid4().hex
        self.output = AcceptedMediaOutput(owner_id=self.context.task.owner_id,
            task_id=self.context.task.id, action_id=uuid4(), render_plan_sha256=self.plan,
            file=RetainedRef(file_id=uuid4(), version_id="source_saved", content_sha256="b" * 64),
            content_type="video/mp4", size_bytes=1000)
        self.receipt = {"status": "success", "task_operation": {
            "operation_ref": self.reference, "action_id": str(self.output.action_id), "state": "accepted"},
            "file": {"id": str(self.output.file.file_id), "content_type": "video/mp4", "size_bytes": 1000,
                     "memory_source": {"document_id": self.output.file.version_id}}}
        self.action = SimpleNamespace(parameters={"request": {
            "format": "mp4", "operation_key": "media-output:" + self.plan}})
        self.base = SimpleNamespace(_check=AsyncMock(), actions={self.reference: self.action})
        # Isolate the verifier from constructor dependencies; it still uses the
        # real domain type and real accepted-output validation, not a fake API.
        self.media = object.__new__(ManagedMediaTools)
        self.media.context = self.context
        self.media.bridge = self.base
        self.media.store = SimpleNamespace(load_accepted_output=AsyncMock(return_value=self.output))
        self.bridge = SimpleNamespace(media=self.media,
            verified_operation=AsyncMock(return_value=self.receipt))

    def result(self, **changes):
        return HermesTaskResult.model_validate({"status": "succeeded", "answer": "The edited video is saved.",
            "evidence": [], "completed_operation_refs": [self.reference], "uncertainties": [],
            "speech": [{"text": "The edited video is saved."}], **changes})

    async def verify(self, *, status="succeeded", receipts=None):
        return await verified_media_result(self.context, self.bridge, status,
                                          [self.receipt] if receipts is None else receipts)

    async def test_ordinary_tasks_do_not_gain_a_media_requirement(self):
        self.assertIsNone(await verified_media_result(self.context, SimpleNamespace(), "succeeded", []))
        self.media.store.load_accepted_output.assert_not_awaited()

    async def test_no_output_cannot_finish_a_media_task(self):
        with self.assertRaises(TaskError) as error:
            await self.result(completed_operation_refs=[]).verified(self.context, self.bridge)
        self.assertEqual(error.exception.code, "task_media_result_unconfirmed")
        self.bridge.verified_operation.assert_not_awaited()

    async def test_a_saved_recipe_is_not_an_edited_video(self):
        self.action.parameters["request"] = {"format": "json", "operation_key": "media-recipe:" + self.plan}
        with self.assertRaises(TaskError):
            await self.verify()
        self.media.store.load_accepted_output.assert_not_awaited()

    async def test_exact_accepted_output_is_attached_without_rewriting_authorship(self):
        result = self.result()
        outcome = await result.verified(self.context, self.bridge)
        self.assertEqual(outcome.payload["verified_media"]["outputs"], [self.output.model_dump(mode="json")])
        self.assertEqual(outcome.payload["answer"], result.answer)
        self.assertEqual(outcome.payload["speech"], [{"text": "The edited video is saved."}])
        self.assertEqual(inspect_expert_output(outcome.payload, task=self.context.task).kind, "expert")
        self.bridge.verified_operation.assert_awaited_once_with(self.reference)
        self.media.store.load_accepted_output.assert_awaited_once_with(
            self.context.task.owner_id, self.context.task.id, self.plan)
        self.assertEqual(self.base._check.await_count, 2)

    async def test_missing_or_withdrawn_output_is_not_success(self):
        for output in (None, SimpleNamespace(**self.output.model_dump())):
            with self.subTest(output=type(output).__name__):
                self.media.store.load_accepted_output.return_value = output
                with self.assertRaises(TaskError):
                    await self.verify()

    async def test_source_failure_is_preserved_without_resubmitting(self):
        self.media.store.load_accepted_output.side_effect = MediaRecipeError("media_publication_unavailable", "Gone")
        with self.assertRaises(TaskError) as error:
            await self.verify()
        self.assertEqual(error.exception.code, "media_publication_unavailable")
        self.media.store.load_accepted_output.assert_awaited_once()

    async def test_changed_accepted_identity_is_rejected(self):
        original = self.output
        changes = ({"owner_id": "different-owner"}, {"task_id": uuid4()}, {"action_id": uuid4()},
                   {"render_plan_sha256": "c" * 64}, {"size_bytes": 1001},
                   {"file": RetainedRef(file_id=uuid4(), version_id="source_saved", content_sha256="b" * 64)},
                   {"file": original.file.model_copy(update={"version_id": "source_replacement"})})
        for change in changes:
            with self.subTest(change=tuple(change)):
                self.media.store.load_accepted_output.return_value = original.model_copy(update=change)
                with self.assertRaises(TaskError):
                    await self.verify()

    async def test_unknown_publication_or_missing_retained_metadata_is_rejected(self):
        for change in ({"status": "outcome_unknown"}, {"file": None}, {"file": {}},
                       {"file": dict(self.receipt["file"], memory_source=None)}):
            with self.subTest(change=tuple(change)):
                with self.assertRaises(TaskError):
                    await self.verify(receipts=[dict(self.receipt, **change)])

    async def test_unrelated_mp4_and_wrong_result_owner_are_not_media_completion(self):
        self.action.parameters["request"]["operation_key"] = "some-mp4"
        with self.assertRaises(TaskError):
            await self.verify()
        self.media.store.load_accepted_output.assert_not_awaited()
        self.media.context = SimpleNamespace()
        with self.assertRaises(TaskError) as error:
            await self.verify()
        self.assertEqual(error.exception.code, "task_media_result_scope")

    async def test_unattached_action_is_rejected(self):
        self.base.actions.clear()
        with self.assertRaises(TaskError):
            await self.verify()

    async def test_real_incomplete_status_does_not_invent_a_saved_file(self):
        for status in ("waiting_for_input", "failed", "paused"):
            with self.subTest(status=status):
                result = await self.result(status=status, answer="The requested edit is unfinished.",
                    completed_operation_refs=[], speech=[{"text": "The requested edit is unfinished."}]
                ).verified(self.context, self.bridge)
                self.assertEqual(result.status.value, status)
                self.assertEqual(result.payload["verified_media"]["outputs"], [])
        self.media.store.load_accepted_output.assert_not_awaited()

    async def test_claim_loss_during_final_revalidation_cannot_deliver_output(self):
        self.base._check.side_effect = [None, ClaimLost()]
        with self.assertRaises(ClaimLost):
            await self.verify()

    async def test_generic_receipt_failure_precedes_media_verification(self):
        self.bridge.verified_operation.side_effect = TaskError("task_result_operation_unconfirmed", "Missing")
        with self.assertRaises(TaskError):
            await self.result().verified(self.context, self.bridge)
        self.media.store.load_accepted_output.assert_not_awaited()

    async def test_changed_receipt_version_cannot_borrow_another_outputs_proof(self):
        receipt = deepcopy(self.receipt)
        receipt["file"]["memory_source"]["document_id"] = "source_other"
        with self.assertRaises(TaskError):
            await self.verify(receipts=[receipt])


if __name__ == "__main__":
    unittest.main()
