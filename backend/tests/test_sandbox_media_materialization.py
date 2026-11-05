"""Real provider/SDK mapping with doubles; no cloud, SQL, render or publication."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxInfo, SandboxPolicy, SandboxScope
from app.workspace.media_recipe import MediaRecipeError, TemplateInventoryEntry, validate_recipe
from app.workspace.media_render import prepare_render
from app.workspace.media_staging import INSTALL_RENDER_INPUTS, expected_materialization_receipt, materialization_request
from tests.test_workspace_media_render import inputs


class MediaMaterializationProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plan = prepare_render(**inputs())
        self.policy = SandboxPolicy()
        self.scope = SandboxScope(owner_id=self.plan.owner_id, task_id=self.plan.task_id,
            task_revision=1, attempt_id=uuid4(), allocation_id=uuid4())
        self.handle = SandboxInfo(sandbox_id=self.plan.sandbox_id, scope=self.scope, policy=self.policy,
            state="running", expires_at=datetime.now(UTC) + timedelta(minutes=5),
            template_id=self.plan.binding.physical_template_id)
        self.raw = SimpleNamespace(sandbox_id=self.handle.sandbox_id, template_id=self.handle.template_id,
            metadata=self.scope.metadata(self.policy), state="running", end_at=self.handle.expires_at,
            allow_internet_access=False,
            network={"allow_out": [], "deny_out": ["0.0.0.0/0"], "allow_public_traffic": False},
            lifecycle={"on_timeout": "kill", "auto_resume": False}, volume_mounts=[])
        self.result = SimpleNamespace(exit_code=0, stdout=json.dumps(expected_materialization_receipt(self.plan)))
        self.run = AsyncMock(return_value=self.result)
        self.box = SimpleNamespace(commands=SimpleNamespace(run=self.run))
        self.api = SimpleNamespace(get_info=AsyncMock(return_value=self.raw),
            connect=AsyncMock(return_value=self.box), create=AsyncMock(), kill=AsyncMock())
        self.provider = E2BProvider(E2BConfiguration(api_key="synthetic-media-key"),
            sdk=SimpleNamespace(AsyncSandbox=self.api))

    async def test_fixed_isolated_installer_and_exact_receipt_not_render_or_publication(self):
        receipt = await self.provider.materialize_media_render(self.handle, self.plan)
        self.assertEqual(receipt, expected_materialization_receipt(self.plan))
        self.run.assert_awaited_once()
        outer = shlex.split(self.run.call_args.args[0])
        self.assertEqual(outer[:3], ["exec", "/usr/bin/env", "-i"])
        self.assertEqual(outer[-4:-1], ["/bin/sh", "-eu", "-c"])
        inner = shlex.split(outer[-1])
        self.assertEqual(inner[:5], ["exec", "/usr/bin/python3", "-I", "-S", "-c"])
        self.assertEqual(len(inner), 7)
        self.assertEqual(inner[5], INSTALL_RENDER_INPUTS)
        self.assertEqual(json.loads(inner[6]), materialization_request(self.plan))
        self.assertNotIn("synthetic-media-key", outer[-1])
        self.assertEqual(self.run.call_args.kwargs["user"], "root")
        self.assertEqual(self.run.call_args.kwargs["cwd"], "/")
        self.assertEqual(self.run.call_args.kwargs["timeout"], 20)
        self.assertFalse(self.run.call_args.kwargs["background"])
        self.api.connect.assert_awaited_once()
        self.assertEqual(self.api.connect.call_args.kwargs["retries"], 0)
        self.assertEqual(self.api.get_info.await_count, 3)
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()
        self.assertNotIn("published", receipt)
        self.assertNotIn("rendered", receipt)

    async def test_caption_cannot_escape_the_single_json_argument(self):
        args = inputs()
        value = args["recipe"].recipe.model_dump(mode="json")
        value["captions"][0]["text"] = "'; $(touch /tmp/not-a-command) `whoami`\n\"text\""
        args["recipe"] = validate_recipe(value, inspections=[args["source"].inspection],
            policy=args["binding"].policy, inventory=[TemplateInventoryEntry(
                template=value["template"], output_profile=value["output_profile"], ready=True)])
        plan = prepare_render(**args)
        self.result.stdout = json.dumps(expected_materialization_receipt(plan))
        await self.provider.materialize_media_render(self.handle, plan)
        inner = shlex.split(shlex.split(self.run.call_args.args[0])[-1])
        self.assertEqual(len(inner), 7)
        envelope = json.loads(json.loads(inner[6])["envelope"])
        self.assertEqual(envelope["recipe"]["captions"][0]["text"], value["captions"][0]["text"])

    async def test_untyped_context_rejected_before_sdk_effects(self):
        for handle, plan in ((None, self.plan), (self.handle, {})):
            with self.subTest(handle=handle), self.assertRaises(ValueError):
                await self.provider.materialize_media_render(handle, plan)
        self.api.get_info.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_foreign_owner_or_task_rejected_before_sdk_effects(self):
        for change in ({"owner_id": "foreign"}, {"task_id": uuid4()}):
            with self.subTest(change=change), self.assertRaises(SandboxError) as caught:
                await self.provider.materialize_media_render(self.handle, replace(self.plan, **change))
            self.assertEqual(caught.exception.code, "sandbox_media_scope_mismatch")
        self.api.get_info.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_wrong_supplied_allocation_or_image_rejected_before_sdk_effects(self):
        for change in ({"sandbox_id": "foreign"}, {"template_id": "foreign"}):
            with self.subTest(change=change), self.assertRaises(MediaRecipeError):
                await self.provider.materialize_media_render(self.handle.model_copy(update=change), self.plan)
        self.api.get_info.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_actual_provider_image_is_checked_before_connect_or_preparation(self):
        self.raw.template_id = "wrong-real-image"
        with self.assertRaises(MediaRecipeError):
            await self.provider.materialize_media_render(self.handle, self.plan)
        self.api.get_info.assert_awaited_once()
        self.api.connect.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_actual_owner_and_network_checks_cannot_be_bypassed(self):
        for change in ({"metadata": {}}, {"allow_internet_access": True}):
            before = {key: getattr(self.raw, key) for key in change}
            for key, value in change.items():
                setattr(self.raw, key, value)
            with self.subTest(change=change), self.assertRaises(SandboxError):
                await self.provider.materialize_media_render(self.handle, self.plan)
            for key, value in before.items():
                setattr(self.raw, key, value)
        self.api.connect.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_bad_receipts_never_confirm_or_retry_preparation(self):
        expected = expected_materialization_receipt(self.plan)
        invalid = [None, "", "null", "[]", "{}", "x" * 8193,
            json.dumps(expected | {"extra": True}),
            json.dumps(expected | {"plan_sha256": "0" * 64}),
            json.dumps(expected | {"source_sha256": "0" * 64}),
            json.dumps(expected | {"size_bytes": float(expected["size_bytes"])}),
            json.dumps(expected | {"output_path": "/tmp/not-authorized.mp4"}),
            json.dumps(expected)[:-1] + ',"format":"jarvis.media-inputs.v1"}',
            json.dumps(expected) + "\nnot-a-receipt"]
        for raw in invalid:
            self.run.reset_mock()
            self.result.stdout = raw
            with self.subTest(raw=raw), self.assertRaises(SandboxError) as caught:
                await self.provider.materialize_media_render(self.handle, self.plan)
            self.assertEqual(caught.exception.code, "sandbox_media_receipt_invalid")
            self.run.assert_awaited_once()
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_nonzero_installer_exit_is_failure_not_a_receipt(self):
        self.result.exit_code = 1
        with self.assertRaises(SandboxError) as caught:
            await self.provider.materialize_media_render(self.handle, self.plan)
        self.assertEqual(caught.exception.code, "sandbox_privileged_command_failed")
        self.run.assert_awaited_once()

    async def test_lost_installer_ack_retains_original_allocation_without_retry(self):
        self.run.side_effect = TimeoutError("synthetic lost installer acknowledgement")
        with self.assertRaises(SandboxError) as caught:
            await self.provider.materialize_media_render(self.handle, self.plan)
        self.assertEqual(caught.exception.code, "sandbox_provider_outcome_uncertain")
        self.run.assert_awaited_once()
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_cancellation_is_not_cleanup_or_permission_to_repeat(self):
        self.run.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.provider.materialize_media_render(self.handle, self.plan)
        self.run.assert_awaited_once()
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_lost_connection_never_submits_an_installer(self):
        self.api.connect.side_effect = TimeoutError("synthetic lost connection")
        with self.assertRaises(SandboxError):
            await self.provider.materialize_media_render(self.handle, self.plan)
        self.run.assert_not_awaited()
        self.api.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
