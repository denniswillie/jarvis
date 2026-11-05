"""Renderer wiring to durable media authority; no cloud or SQL effects.

Detailed pin/grant validation is covered by test_lean_media_admission. These
tests prove fresh and recovered renderer paths cannot bypass that same owner.
"""
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.sandbox.media import ManagedMediaTools
from app.tasks.models import ClaimLost, TaskError


class MediaInputBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner = "fixture-owner"
        self.task = SimpleNamespace(owner_id=self.owner)
        self.locked_task = SimpleNamespace(owner_id=self.owner)
        self.connection = object()
        self.in_transaction = False

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.owner)
            self.in_transaction = True
            try:
                yield self.connection
            finally:
                self.in_transaction = False

        self.tasks = SimpleNamespace(transaction=transaction,
            lock_claim=AsyncMock(return_value=(self.locked_task, {})))
        self.media = object.__new__(ManagedMediaTools)
        self.media.context = SimpleNamespace(tasks=self.tasks, task=self.task, claim=object())
        self.recipe = SimpleNamespace(template=object(), previous_recipe=None,
            sources=[SimpleNamespace(file=object())])
        self.media.job = SimpleNamespace(bundle=SimpleNamespace(
            renderer=SimpleNamespace(template=self.recipe.template)))
        self.media.files = SimpleNamespace(media_inspector=SimpleNamespace(inspect_retained_input=AsyncMock()))
        self.media.inputs = SimpleNamespace(stage=AsyncMock(), revalidate_staged=AsyncMock())
        self.media.store = SimpleNamespace(_verify_reference=AsyncMock(), previous_recipe=AsyncMock())
        self.media.bridge = SimpleNamespace(_check=AsyncMock())
        self.media._prepared_allocation = AsyncMock()
        self.authority = AsyncMock(return_value=object())
        self.patcher = patch("app.tasks.media_editing.require_media_recipe_input", self.authority)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    async def test_exact_locked_task_and_recipe_reach_authority_in_owner_transaction(self):
        async def checked(*args):
            self.assertTrue(self.in_transaction)
            return "resolved-pin"

        self.authority.side_effect = checked
        self.assertEqual(await self.media._require_task_source(self.recipe), "resolved-pin")
        self.authority.assert_awaited_once_with(self.tasks, self.connection, self.locked_task, self.recipe)
        self.tasks.lock_claim.assert_awaited_once_with(self.connection, self.media.context.claim)
        self.assertFalse(self.in_transaction)

    async def test_authority_failure_is_not_downgraded_or_retried(self):
        error = TaskError("media_command_scope_unavailable", "Source authority withdrawn.", 403)
        self.authority.side_effect = error
        with self.assertRaises(TaskError) as raised:
            await self.media._require_task_source(self.recipe)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.authority.await_count, 1)
        self.assertFalse(self.in_transaction)

    async def test_lost_claim_never_checks_or_stages_source(self):
        self.tasks.lock_claim.side_effect = ClaimLost()
        with self.assertRaises(ClaimLost):
            await self.media._require_task_source(self.recipe)
        self.authority.assert_not_awaited()
        self.media.inputs.stage.assert_not_awaited()

    async def test_finalized_user_input_route_still_goes_through_authority(self):
        self.authority.return_value = None
        self.assertIsNone(await self.media._require_task_source(self.recipe))
        self.authority.assert_awaited_once()

    async def test_fresh_render_denies_before_inspection_or_staging(self):
        self.authority.side_effect = TaskError("media_command_recipe_mismatch", "Wrong admitted video.", 403)
        with self.assertRaises(TaskError):
            await self.media._fresh_plan(SimpleNamespace(recipe=self.recipe))
        self.authority.assert_awaited_once()
        self.media.files.media_inspector.inspect_retained_input.assert_not_awaited()
        self.media.inputs.stage.assert_not_awaited()

    async def test_reconstructed_render_denies_before_reusing_allocation(self):
        self.authority.side_effect = TaskError("media_command_predecessor_not_admitted", "Wrong predecessor.", 403)
        with self.assertRaises(TaskError):
            await self.media._fresh_plan(SimpleNamespace(recipe=self.recipe), original={"id": "existing"})
        self.authority.assert_awaited_once()
        self.media._prepared_allocation.assert_not_awaited()
        self.media.inputs.stage.assert_not_awaited()

    async def test_recovered_sources_deny_before_trusting_retained_or_staged_bytes(self):
        self.authority.side_effect = TaskError("media_command_scope_unavailable", "Source withdrawn.", 403)
        with self.assertRaises(TaskError):
            await self.media._sources(SimpleNamespace(recipe=SimpleNamespace(recipe=self.recipe)))
        self.authority.assert_awaited_once()
        self.media.store._verify_reference.assert_not_awaited()
        self.media.inputs.revalidate_staged.assert_not_awaited()
        self.media.bridge._check.assert_not_awaited()

    async def test_recovery_preserves_reference_staging_and_claim_rechecks_in_order(self):
        events = []
        async def authority(*args):
            events.append("authority")
        async def reference(*args):
            events.append("reference")
        async def staged():
            events.append("staged")
        async def claim():
            events.append("claim")
        self.authority.side_effect = authority
        self.media.store._verify_reference.side_effect = reference
        self.media.inputs.revalidate_staged.side_effect = staged
        self.media.bridge._check.side_effect = claim
        await self.media._sources(SimpleNamespace(recipe=SimpleNamespace(recipe=self.recipe)))
        self.assertEqual(events, ["authority", "reference", "staged", "claim"])
        self.media.store._verify_reference.assert_awaited_once_with(self.owner, self.recipe.sources[0].file)


if __name__ == "__main__":
    unittest.main()
