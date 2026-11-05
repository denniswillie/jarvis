"""Retained lifecycle qualification against SDK doubles only; no cloud effects."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from pydantic import ValidationError

from app.hermes_sessions.sandbox_provider import RetainedE2BProvider, RetainedSandboxInfo, RetainedSandboxScope
from app.sandbox.e2b_provider import E2BProvider, SandboxCommand, WORKSPACE
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxInfo, SandboxPolicy, SandboxScope


class ExitError(Exception):
    pass


class Pages:
    def __init__(self, rows):
        self.rows, self.has_next = rows, True

    async def next_items(self):
        self.has_next = False
        return self.rows


class RetainedSandboxProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = RetainedSandboxScope(owner_id="fixture-owner", profile_id="fixture-profile",
            logical_session_id="fixture-master", allocation_id=uuid4())
        self.task_scope = SandboxScope(owner_id="fixture-owner", task_id=uuid4(), task_revision=1,
            attempt_id=uuid4(), allocation_id=uuid4())
        self.policy = SandboxPolicy()
        self.deadline = datetime.now(UTC) + timedelta(seconds=300)
        self.raw = SimpleNamespace(sandbox_id="same-sandbox", template_id="base", state="running",
            metadata=self.scope.metadata(self.policy), end_at=self.deadline, allow_internet_access=False,
            network={"allow_out": [], "deny_out": ["0.0.0.0/0"], "allow_public_traffic": False},
            lifecycle={"on_timeout": "pause", "auto_resume": False}, volume_mounts=[])
        self.native_command = SimpleNamespace(pid=31, exit_code=0, stdout="", stderr="",
            wait=AsyncMock(return_value=SimpleNamespace(exit_code=0)), disconnect=AsyncMock())
        self.commands = SimpleNamespace(run=AsyncMock(return_value=self.native_command),
            connect=AsyncMock(return_value=self.native_command), list=AsyncMock(return_value=[]),
            kill=AsyncMock(return_value=True))
        self.box = SimpleNamespace(sandbox_id=self.raw.sandbox_id, commands=self.commands, set_timeout=AsyncMock())
        self.api = SimpleNamespace(create=AsyncMock(return_value=self.box), get_info=AsyncMock(return_value=self.raw),
            connect=AsyncMock(side_effect=self.connect), pause=AsyncMock(side_effect=self.pause),
            kill=AsyncMock(return_value=True), list=Mock(return_value=Pages([self.raw])))
        self.sdk = SimpleNamespace(AsyncSandbox=self.api, CommandExitException=ExitError,
            SandboxState=SimpleNamespace(RUNNING="running", PAUSED="paused"), SandboxQuery=SimpleNamespace)
        self.configuration = E2BConfiguration(api_key="fixture-private-key")
        self.provider = RetainedE2BProvider(self.configuration, sdk=self.sdk)
        self.legacy = E2BProvider(self.configuration, sdk=self.sdk)

    async def connect(self, sandbox_id, **kwargs):
        self.assertEqual(sandbox_id, self.raw.sandbox_id)
        self.raw.state, self.raw.end_at = "running", self.deadline
        return self.box

    async def pause(self, sandbox_id, **kwargs):
        self.assertEqual(sandbox_id, self.raw.sandbox_id)
        self.raw.state, self.raw.end_at = "paused", None

    async def handle(self):
        return await self.provider.info(self.raw.sandbox_id, self.scope, self.policy)

    def test_scope_has_only_real_session_identity_and_stable_private_metadata(self):
        self.assertEqual(set(self.scope.model_dump()), {"owner_id", "profile_id", "logical_session_id", "allocation_id"})
        metadata = self.scope.metadata(self.policy)
        self.assertEqual(metadata["jarvis_kind"], "hermes-session-sandbox-v1")
        self.assertEqual(metadata["jarvis_allocation"], str(self.scope.allocation_id))
        self.assertEqual(metadata["jarvis_policy"], self.policy.fingerprint)
        self.assertFalse({"jarvis_task", "jarvis_revision", "jarvis_attempt"} & set(metadata))
        for field in ("owner_id", "profile_id", "logical_session_id"):
            self.assertNotIn(getattr(self.scope, field), str(metadata))
            changed = self.scope.model_copy(update={field: "different"})
            self.assertNotEqual(changed.metadata(self.policy), metadata)
        self.assertEqual(RetainedSandboxScope.model_validate_json(self.scope.model_dump_json()), self.scope)

    def test_invalid_or_task_shaped_scope_is_not_admitted(self):
        for field in ("owner_id", "profile_id", "logical_session_id"):
            for value in ("", " padded ", "bad\x00identity"):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    RetainedSandboxScope.model_validate(self.scope.model_dump() | {field: value})
        with self.assertRaises(ValidationError):
            RetainedSandboxScope.model_validate(self.scope.model_dump() | {"task_id": uuid4()})

    async def test_create_uses_pause_policy_no_auto_resume_and_returns_retained_handle(self):
        self.api.create.assert_not_awaited()
        handle = await self.provider.create(self.scope, self.policy)
        self.assertIsInstance(handle, RetainedSandboxInfo)
        self.assertEqual(handle.provider_scope, self.scope)
        self.assertEqual(set(handle.model_dump()), {"sandbox_id", "scope", "policy", "state", "expires_at", "template_id"})
        self.assertIsNone(handle.native_scope)
        options = self.api.create.call_args.kwargs
        self.assertEqual(options["lifecycle"], {"on_timeout": "pause", "auto_resume": False})
        self.assertEqual(options["metadata"], self.scope.metadata(self.policy))
        self.assertEqual(options["envs"], {})
        self.assertTrue(options["secure"])
        self.assertEqual(options["timeout"], 300)
        self.assertEqual(options["retries"], 0)
        self.assertNotIn("fixture-private-key", str(options["metadata"]))
        with self.assertRaises(ValidationError):
            RetainedSandboxInfo.model_validate(handle.model_dump() | {"native_scope": self.scope})

    async def test_lifetime_ceiling_is_checked_before_creation(self):
        for seconds in (59, 3601, True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                await self.provider.create(self.scope, self.policy.model_copy(update={"lifetime_seconds": seconds}))
        self.api.create.assert_not_awaited()
        self.policy = SandboxPolicy(lifetime_seconds=3600)
        self.raw.metadata = self.scope.metadata(self.policy)
        await self.provider.create(self.scope, self.policy)
        self.assertEqual(self.api.create.call_args.kwargs["timeout"], 3600)

    async def test_provider_scope_types_fail_both_ways_before_sdk_calls(self):
        for provider, wrong_scope in ((self.provider, self.task_scope), (self.legacy, self.scope)):
            for invoke in (
                lambda: provider.create(wrong_scope, self.policy),
                lambda: provider.info(self.raw.sandbox_id, wrong_scope, self.policy),
                lambda: provider.discover(wrong_scope, self.policy),
            ):
                with self.assertRaises(SandboxError) as caught:
                    await invoke()
                self.assertEqual(caught.exception.code, "sandbox_scope_mismatch")
        self.api.create.assert_not_awaited()
        self.api.get_info.assert_not_awaited()
        self.api.list.assert_not_called()

    async def test_native_metadata_kind_is_rejected_both_ways(self):
        self.raw.metadata = self.task_scope.metadata(self.policy)
        with self.assertRaises(SandboxError) as caught:
            await self.handle()
        self.assertEqual(caught.exception.code, "sandbox_scope_mismatch")
        self.raw.metadata = self.scope.metadata(self.policy)
        with self.assertRaises(SandboxError) as caught:
            await self.legacy.info(self.raw.sandbox_id, self.task_scope, self.policy)
        self.assertEqual(caught.exception.code, "sandbox_scope_mismatch")

    async def test_each_resource_identity_component_is_revalidated(self):
        for key in self.scope.metadata(self.policy):
            self.raw.metadata = self.scope.metadata(self.policy) | {key: "foreign"}
            with self.subTest(key=key), self.assertRaises(SandboxError):
                await self.handle()
        self.api.connect.assert_not_awaited()

    async def test_wrong_lifecycle_or_network_is_not_a_retained_handle(self):
        for lifecycle in ({"on_timeout": "kill", "auto_resume": False},
                          {"on_timeout": "pause", "auto_resume": True}, {}):
            self.raw.lifecycle = lifecycle
            with self.assertRaises(SandboxError) as caught:
                await self.handle()
            self.assertEqual(caught.exception.code, "sandbox_policy_mismatch")
        self.raw.lifecycle = {"on_timeout": "pause", "auto_resume": False}
        self.raw.network["allow_public_traffic"] = True
        with self.assertRaises(SandboxError):
            await self.handle()

    async def test_explicit_pause_keeps_sdk_default_mode_and_same_identity(self):
        paused = await self.provider.pause(await self.handle())
        self.assertEqual((paused.sandbox_id, paused.scope, paused.state), (self.raw.sandbox_id, self.scope, "paused"))
        self.assertIsNone(paused.expires_at)
        self.api.pause.assert_awaited_once_with(self.raw.sandbox_id, **self.provider._options())
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_explicit_resume_uses_same_id_and_bounded_runtime(self):
        self.raw.state, self.raw.end_at = "paused", None
        handle = await self.handle()
        for seconds in (0, 301, True):
            with self.assertRaises(ValueError):
                await self.provider.resume(handle, timeout_seconds=seconds)
        resumed = await self.provider.resume(handle, timeout_seconds=90)
        self.assertIsInstance(resumed, RetainedSandboxInfo)
        self.assertEqual((resumed.sandbox_id, resumed.scope, resumed.state), (handle.sandbox_id, self.scope, "running"))
        self.assertEqual(self.api.connect.call_args.args, (handle.sandbox_id,))
        self.assertEqual(self.api.connect.call_args.kwargs["timeout"], 90)
        self.box.set_timeout.assert_awaited_once_with(90, **self.provider._options())
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_commands_do_not_implicitly_resume_a_paused_sandbox(self):
        self.raw.state, self.raw.end_at = "paused", None
        with self.assertRaises(SandboxError) as caught:
            await self.provider.start(await self.handle(), "printf fixture", operation_id=uuid4())
        self.assertEqual(caught.exception.code, "sandbox_paused")
        self.api.connect.assert_not_awaited()
        self.commands.run.assert_not_awaited()

    async def test_inherited_prepare_run_and_attach_keep_retained_handle(self):
        prepared = await self.provider.prepare(await self.handle())
        self.assertIsInstance(prepared, RetainedSandboxInfo)
        self.assertEqual(prepared.scope, self.scope)
        self.assertEqual(self.commands.run.call_args.kwargs["user"], "root")
        command = await self.provider.start(prepared, "printf fixture", operation_id=uuid4())
        self.assertIsInstance(command, SandboxCommand)
        self.assertEqual(self.commands.run.call_args.kwargs["user"], "jarvis")
        self.assertEqual(self.commands.run.call_args.kwargs["cwd"], WORKSPACE)
        self.assertEqual((await command.wait(timeout_seconds=1)).pid, 31)
        attached = await self.provider.reconnect_command(prepared, 31)
        self.assertEqual(attached.pid, 31)
        await attached.disconnect()
        self.commands.connect.assert_awaited_once()
        self.api.create.assert_not_awaited()

    async def test_discovery_uses_exact_session_metadata(self):
        self.assertEqual(await self.provider.discover(self.scope, self.policy), [self.raw.sandbox_id])
        query = self.api.list.call_args.kwargs["query"]
        self.assertEqual(query.metadata, self.scope.metadata(self.policy))
        self.assertEqual(query.state, ["running", "paused"])

    async def test_retained_kill_and_delete_are_always_refused_without_sdk_calls(self):
        for method in (self.provider.kill, self.provider.delete):
            with self.assertRaises(SandboxError) as caught:
                await method(self.raw.sandbox_id, self.scope, self.policy)
            self.assertEqual(caught.exception.code, "retained_sandbox_delete_forbidden")
        with self.assertRaises(SandboxError):
            await self.legacy.kill(self.raw.sandbox_id, self.scope, self.policy)
        self.api.get_info.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_legacy_creation_and_owned_kill_behavior_stay_unchanged(self):
        self.raw.metadata = self.task_scope.metadata(self.policy)
        self.raw.lifecycle = {"on_timeout": "kill", "auto_resume": False}
        handle = await self.legacy.create(self.task_scope, self.policy)
        self.assertIsInstance(handle, SandboxInfo)
        self.assertEqual(self.api.create.call_args.kwargs["lifecycle"], self.raw.lifecycle)
        # Legacy cleanup still works when the owned VM's network policy is wrong.
        self.raw.network["allow_public_traffic"] = True
        self.assertTrue(await self.legacy.kill(handle.sandbox_id, self.task_scope, self.policy))
        self.api.kill.assert_awaited_once()
