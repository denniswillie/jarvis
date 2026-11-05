"""Physical image selection using the real provider/broker with SDK doubles.

No cloud allocation, installed-image qualification, SQL or model calls occur.
The existing SQL journeys separately exercise durable submission and publication.
"""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.plugins.authority import ManagedPluginBinding
from app.plugins.managed import FilesDataBundle, ManagedFilesDataJob
from app.sandbox.broker import SandboxCommandBroker
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxInfo, SandboxPolicy, SandboxScope
from app.tasks.models import TaskError


class BoundImageProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.policy = SandboxPolicy()
        self.scope = SandboxScope(owner_id="synthetic-image-owner", task_id=uuid4(),
            task_revision=1, attempt_id=uuid4(), allocation_id=uuid4())
        self.raw = SimpleNamespace(sandbox_id="synthetic-resource", template_id="qualified-image-id",
            metadata=self.scope.metadata(self.policy), state="running",
            end_at=datetime.now(UTC) + timedelta(minutes=5), allow_internet_access=False,
            network={"allow_out": [], "deny_out": ["0.0.0.0/0"], "allow_public_traffic": False},
            lifecycle={"on_timeout": "kill", "auto_resume": False}, volume_mounts=[])
        self.api = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(sandbox_id=self.raw.sandbox_id)),
            get_info=AsyncMock(return_value=self.raw))
        self.provider = E2BProvider(E2BConfiguration(api_key="synthetic-image-binding-key"),
            sdk=SimpleNamespace(AsyncSandbox=self.api))

    async def test_bound_create_selects_only_exact_image_and_preserves_granted_policy(self):
        handle = await self.provider.create_bound(self.scope, self.policy, template_id="qualified-image-id")
        self.assertEqual(handle.template_id, "qualified-image-id")
        self.assertEqual(handle.policy, self.policy)
        self.api.create.assert_awaited_once()
        args = self.api.create.call_args.kwargs
        self.assertEqual(args["template"], "qualified-image-id")
        self.assertEqual(args["metadata"], self.scope.metadata(self.policy))
        self.assertEqual(args["timeout"], self.policy.lifetime_seconds)
        self.assertEqual(args["envs"], {})
        self.assertEqual(args["retries"], 0)
        self.assertFalse(args["allow_internet_access"])
        self.assertEqual(args["network"], self.raw.network)
        self.assertEqual(args["lifecycle"], self.raw.lifecycle)
        self.assertNotIn("synthetic-image-binding-key", str(args["metadata"]))

    async def test_ordinary_creation_keeps_generic_template(self):
        await self.provider.create(self.scope, self.policy)
        self.assertEqual(self.api.create.call_args.kwargs["template"], "base")
        self.assertEqual(self.api.create.call_args.kwargs["metadata"], self.scope.metadata(self.policy))

    async def test_invalid_physical_identifiers_fail_before_any_provider_effect(self):
        for value in (None, "", " padded ", "a/b", "name:latest", "name\n", "x" * 201, 7):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await self.provider.create_bound(self.scope, self.policy, template_id=value)
        self.api.create.assert_not_awaited()
        self.api.get_info.assert_not_awaited()

    async def test_returned_wrong_image_fails_without_retry_or_fallback(self):
        self.raw.template_id = "unexpected-image"
        with self.assertRaises(SandboxError) as caught:
            await self.provider.create_bound(self.scope, self.policy, template_id="qualified-image-id")
        self.assertEqual(caught.exception.code, "sandbox_template_mismatch")
        self.api.create.assert_awaited_once()
        self.api.get_info.assert_awaited_once()

    async def test_lost_creation_acknowledgment_is_not_retried(self):
        self.api.create.side_effect = TimeoutError("synthetic lost acknowledgment")
        with self.assertRaises(SandboxError):
            await self.provider.create_bound(self.scope, self.policy, template_id="qualified-image-id")
        self.api.create.assert_awaited_once()
        self.api.get_info.assert_not_awaited()

    async def test_bound_image_cannot_bypass_network_or_owner_checks(self):
        self.raw.allow_internet_access = True
        with self.assertRaises(SandboxError) as caught:
            await self.provider.create_bound(self.scope, self.policy, template_id="qualified-image-id")
        self.assertEqual(caught.exception.code, "sandbox_policy_mismatch")
        self.raw.allow_internet_access = False
        self.raw.metadata = {}
        with self.assertRaises(SandboxError) as caught:
            await self.provider.create_bound(self.scope, self.policy, template_id="qualified-image-id")
        self.assertEqual(caught.exception.code, "sandbox_scope_mismatch")


class BoundImageBrokerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.task_id = "synthetic-image-owner", uuid4()
        configuration = uuid4()
        self.job = ManagedFilesDataJob(
            ManagedPluginBinding(self.owner, "files_data", configuration, 1),
            FilesDataBundle(configuration_id=configuration, bundle_version="synthetic-v1",
                bundle_sha256="a" * 64, sandbox_template_id="qualified-image-id", executable="/fixture/python3"))
        self.scope = SandboxScope(owner_id=self.owner, task_id=self.task_id, task_revision=1,
            attempt_id=uuid4(), allocation_id=uuid4())
        self.policy = SandboxPolicy()
        self.handle = SandboxInfo(sandbox_id="synthetic-resource", scope=self.scope, policy=self.policy,
            state="running", expires_at=datetime.now(UTC) + timedelta(minutes=5), template_id="qualified-image-id")
        self.reserved = {"id": self.scope.allocation_id, "phase": "reserved"}
        self.preparing = self.reserved | {"phase": "preparing"}
        self.ready = self.reserved | {"phase": "ready"}
        self.token = uuid4()
        self.broker = object.__new__(SandboxCommandBroker)
        self.broker.context = SimpleNamespace(claim=object(), task=SimpleNamespace(owner_id=self.owner, id=self.task_id))
        self.broker.grant_id = uuid4()
        self.broker._managed = AsyncMock(return_value=self.job)
        self.broker.managed_plugins = SimpleNamespace(ensure_submitted=AsyncMock(return_value=self.job))
        self.broker.provider = SimpleNamespace(create=AsyncMock(return_value=self.handle),
            create_bound=AsyncMock(return_value=self.handle), prepare=AsyncMock(),
            discover=AsyncMock(return_value=[self.handle.sandbox_id]), info=AsyncMock(return_value=self.handle))
        self.broker.allocations = SimpleNamespace(
            scope=Mock(return_value=self.scope), policy=Mock(return_value=self.policy),
            begin_creation=AsyncMock(return_value=self.token),
            record_created=AsyncMock(return_value=self.preparing), creation_uncertain=AsyncMock(),
            handle=Mock(return_value=self.handle), ready=AsyncMock(return_value=self.ready),
            current=AsyncMock(return_value=self.ready))

    async def test_managed_creation_uses_persisted_binding_after_durable_submission(self):
        async def create(scope, policy, *, template_id):
            self.broker.managed_plugins.ensure_submitted.assert_awaited_once()
            self.assertEqual((scope, policy, template_id),
                             (self.scope, self.policy, self.job.bundle.sandbox_template_id))
            return self.handle
        self.broker.provider.create_bound.side_effect = create
        self.assertEqual(await self.broker._ready(self.reserved), self.ready)
        self.broker.provider.create_bound.assert_awaited_once()
        self.broker.provider.create.assert_not_awaited()
        self.broker.allocations.record_created.assert_awaited_once_with(
            self.owner, self.scope.allocation_id, self.token, self.handle)
        self.broker.provider.prepare.assert_awaited_once_with(self.handle)

    async def test_missing_bound_creation_never_falls_back_or_claims_submission(self):
        self.broker.provider.create_bound = None
        with self.assertRaises(SandboxError) as caught:
            await self.broker._ready(self.reserved)
        self.assertEqual(caught.exception.code, "managed_template_selection_unavailable")
        self.broker.managed_plugins.ensure_submitted.assert_not_awaited()
        self.broker.allocations.begin_creation.assert_not_awaited()
        self.broker.provider.create.assert_not_awaited()

    async def test_submission_denial_precedes_provider_effect(self):
        self.broker.managed_plugins.ensure_submitted.side_effect = TaskError("plugin_disabled", "Disabled")
        with self.assertRaises(TaskError):
            await self.broker._ready(self.reserved)
        self.broker.allocations.begin_creation.assert_not_awaited()
        self.broker.provider.create_bound.assert_not_awaited()

    async def test_unknown_bound_creation_retains_original_allocation_for_recovery(self):
        self.broker.provider.create_bound.side_effect = TimeoutError("synthetic unknown create")
        with self.assertRaises(TimeoutError):
            await self.broker._ready(self.reserved)
        self.broker.allocations.creation_uncertain.assert_awaited_once_with(
            self.owner, self.scope.allocation_id, self.token)
        self.broker.provider.create.assert_not_awaited()
        self.broker.provider.prepare.assert_not_awaited()

    async def test_uncertain_allocation_is_discovered_not_created_again(self):
        self.assertEqual(await self.broker._ready(self.reserved | {"phase": "uncertain", "submission_token": self.token}), self.ready)
        self.broker.provider.discover.assert_awaited_once_with(self.scope, self.policy)
        self.broker.provider.info.assert_awaited_once_with(self.handle.sandbox_id, self.scope, self.policy)
        self.broker.provider.create_bound.assert_not_awaited()
        self.broker.provider.create.assert_not_awaited()

    async def test_ready_allocation_cannot_bypass_image_check(self):
        self.broker.allocations.handle.return_value = self.handle.model_copy(update={"template_id": "wrong"})
        with self.assertRaises(TaskError) as caught:
            await self.broker._ready(self.ready)
        self.assertEqual(caught.exception.code, "managed_bundle_image_mismatch")
        self.broker.provider.prepare.assert_not_awaited()
        self.broker.provider.create_bound.assert_not_awaited()

    async def test_ordinary_job_keeps_original_provider_method(self):
        self.broker._managed.return_value = None
        self.assertEqual(await self.broker._ready(self.reserved), self.ready)
        self.broker.provider.create.assert_awaited_once_with(self.scope, self.policy)
        self.broker.provider.create_bound.assert_not_awaited()
        self.broker.managed_plugins.ensure_submitted.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
