import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.sandbox_provider import RetainedSandboxInfo
from app.hermes_sessions.sandboxes import RetainedSandboxService
from app.sandbox.models import SandboxError, SandboxPolicy


class Provider:
    def __init__(self):
        self.items = {}
        self.creates = self.prepares = self.pauses = self.resumes = 0
        self.create_error = self.prepare_error = self.resume_error = None
        self.lost_create_ack = False
        self.ambiguous = False

    async def create(self, scope, policy):
        self.creates += 1
        creation_number = self.creates
        await asyncio.sleep(0)
        if self.create_error:
            raise self.create_error
        info = RetainedSandboxInfo(sandbox_id=f"retained-{creation_number}", scope=scope, policy=policy,
            state="running", expires_at=datetime.now(timezone.utc) + timedelta(seconds=policy.lifetime_seconds),
            template_id=policy.template)
        self.items[info.sandbox_id] = info
        if self.lost_create_ack:
            raise SandboxError("sandbox_create_unknown")
        return info

    async def discover(self, scope, policy):
        matches = [key for key, item in self.items.items() if item.scope == scope and item.policy == policy]
        return matches * 2 if self.ambiguous else matches

    async def info(self, sandbox_id, scope, policy):
        if sandbox_id not in self.items:
            raise SandboxError("sandbox_not_found")
        return self.items[sandbox_id]

    async def prepare(self, info):
        self.prepares += 1
        if self.prepare_error:
            raise self.prepare_error

    async def pause(self, info):
        self.pauses += 1
        self.items[info.sandbox_id] = info.model_copy(update={"state": "paused"})
        return self.items[info.sandbox_id]

    async def resume(self, info, *, timeout_seconds=None):
        self.resumes += 1
        if self.resume_error:
            raise self.resume_error
        self.items[info.sandbox_id] = info.model_copy(update={"state": "running"})
        return self.items[info.sandbox_id]


class SandboxServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "resources.sqlite"
        self.store = SessionSandboxBindings(self.path, owner_id="owner", profile_id="profile")
        self.provider = Provider()
        self.policy = SandboxPolicy(lifetime_seconds=3600)
        self.service = RetainedSandboxService(self.store, self.provider, policy=self.policy)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def reopen(self):
        self.store.close()
        self.store = SessionSandboxBindings(self.path, owner_id="owner", profile_id="profile")
        self.service = RetainedSandboxService(self.store, self.provider, policy=self.policy)

    async def test_idle_and_inspection_do_not_allocate(self):
        self.assertEqual(self.provider.creates, 0)
        with self.assertRaisesRegex(SandboxError, "not_allocated"):
            await self.service.inspect("master")
        self.assertEqual(self.provider.creates, 0)

    async def test_new_retained_sandbox_defaults_to_desktop(self):
        service = RetainedSandboxService(self.store, self.provider)
        info = await service.ensure("default-desktop")
        self.assertEqual(info.policy.template, "desktop")
        self.assertEqual(info.template_id, "desktop")
        self.assertEqual(SandboxPolicy().template, "base")

    async def test_config_without_template_keeps_limits_and_defaults_to_desktop(self):
        info = await self.service.ensure("child")
        self.assertEqual(info.policy.template, "desktop")
        self.assertEqual(info.policy.lifetime_seconds, 3600)

    async def test_explicit_terminal_template_remains_supported(self):
        service = RetainedSandboxService(self.store, self.provider,
            policy=SandboxPolicy(template="base", lifetime_seconds=3600))
        info = await service.ensure("legacy")
        self.assertEqual(info.policy.template, "base")

    async def test_desktop_default_preserves_existing_terminal_across_restart(self):
        old = RetainedSandboxService(self.store, self.provider,
            policy=SandboxPolicy(template="base", lifetime_seconds=3600))
        original = await old.ensure("legacy")
        await old.pause("legacy")
        self.reopen()
        paused = await self.service.inspect("legacy")
        self.assertEqual(paused.state, "paused")
        self.assertEqual(self.provider.resumes, 0)
        resumed = await self.service.resume("legacy")
        reused = await self.service.ensure("legacy")
        self.assertEqual(resumed.sandbox_id, original.sandbox_id)
        self.assertEqual(reused.policy, original.policy)
        self.assertEqual(reused.template_id, "base")
        self.assertEqual(self.store.get("legacy").scope.allocation_id, original.scope.allocation_id)
        desktop = await self.service.ensure("new-child")
        self.assertEqual(desktop.policy.template, "desktop")
        self.assertEqual(self.provider.creates, 2)

    async def test_default_change_does_not_repeat_uncertain_terminal_creation(self):
        old = RetainedSandboxService(self.store, self.provider,
            policy=SandboxPolicy(template="base", lifetime_seconds=3600))
        self.provider.lost_create_ack = True
        with self.assertRaisesRegex(SandboxError, "create_unknown"):
            await old.ensure("legacy")
        self.reopen()
        recovered = await self.service.ensure("legacy")
        self.assertEqual(recovered.policy.template, "base")
        self.assertEqual(self.provider.creates, 1)

    async def test_lazy_concurrent_use_allocates_once(self):
        handles = await asyncio.gather(*(self.service.ensure("master") for _ in range(5)))
        self.assertEqual(len({handle.sandbox_id for handle in handles}), 1)
        self.assertEqual((self.provider.creates, self.provider.prepares), (1, 1))

    async def test_children_do_not_share_the_master_sandbox(self):
        master, child = await asyncio.gather(self.service.ensure("master"), self.service.ensure("child"))
        self.assertNotEqual(master.sandbox_id, child.sandbox_id)
        self.assertNotEqual(master.scope.allocation_id, child.scope.allocation_id)

    async def test_restart_reuses_exact_sandbox(self):
        first = await self.service.ensure("child")
        self.reopen()
        second = await self.service.ensure("child")
        self.assertEqual(first.sandbox_id, second.sandbox_id)
        self.assertEqual((self.provider.creates, self.provider.prepares), (1, 1))

    async def test_lost_create_ack_recovers_by_metadata_without_recreating(self):
        self.provider.lost_create_ack = True
        with self.assertRaisesRegex(SandboxError, "create_unknown"):
            await self.service.ensure("child")
        self.reopen()
        handle = await self.service.ensure("child")
        self.assertEqual(handle.sandbox_id, "retained-1")
        self.assertEqual(self.provider.creates, 1)

    async def test_empty_discovery_does_not_authorize_another_create(self):
        self.provider.create_error = SandboxError("sandbox_create_unknown")
        with self.assertRaises(SandboxError):
            await self.service.ensure("child")
        self.provider.create_error = None
        with self.assertRaisesRegex(SandboxError, "creation_unknown"):
            await self.service.ensure("child")
        self.assertEqual(self.provider.creates, 1)

    async def test_ambiguous_discovery_is_not_adopted(self):
        self.provider.lost_create_ack = True
        with self.assertRaises(SandboxError):
            await self.service.ensure("child")
        self.provider.ambiguous = True
        with self.assertRaisesRegex(SandboxError, "creation_ambiguous"):
            await self.service.ensure("child")
        self.assertIsNone(self.store.get("child").sandbox_id)

    async def test_missing_bound_sandbox_is_not_replaced(self):
        handle = await self.service.ensure("child")
        del self.provider.items[handle.sandbox_id]
        with self.assertRaisesRegex(SandboxError, "not_found"):
            await self.service.ensure("child")
        self.assertEqual(self.provider.creates, 1)

    async def test_pause_requires_explicit_resume(self):
        first = await self.service.ensure("child")
        paused = await self.service.pause("child")
        self.assertEqual(paused.state, "paused")
        with self.assertRaisesRegex(SandboxError, "session_sandbox_paused"):
            await self.service.ensure("child")
        self.assertEqual(self.provider.resumes, 0)
        resumed = await self.service.resume("child")
        self.assertEqual(resumed.sandbox_id, first.sandbox_id)
        self.assertEqual(resumed.state, "running")
        self.assertEqual(self.provider.creates, 1)

    async def test_provider_auto_pause_is_observed_without_resuming(self):
        first = await self.service.ensure("child")
        await self.provider.pause(first)
        self.assertEqual((await self.service.inspect("child")).state, "paused")
        self.assertEqual(self.provider.resumes, 0)

    async def test_prepare_failure_retains_allocation(self):
        self.provider.prepare_error = SandboxError("bootstrap_unknown")
        with self.assertRaises(SandboxError):
            await self.service.ensure("child")
        self.assertIsNotNone(self.store.get("child").sandbox_id)
        self.provider.prepare_error = None
        await self.service.ensure("child")
        self.assertEqual(self.provider.creates, 1)

    async def test_failed_resume_survives_restart_and_recovers_only_lifecycle(self):
        first = await self.service.ensure("child")
        await self.service.pause("child")
        self.provider.resume_error = SandboxError("sandbox_resume_unknown")
        with self.assertRaises(SandboxError):
            await self.service.resume("child")
        self.reopen()
        self.assertEqual(self.store.get("child").pending_action, "resume")
        with self.assertRaisesRegex(SandboxError, "lifecycle_pending"):
            await self.service.ensure("child")
        self.provider.resume_error = None
        recovered = await self.service.recover("child")
        self.assertEqual(recovered.sandbox_id, first.sandbox_id)
        self.assertIsNone(self.store.get("child").pending_action)
        self.assertEqual(self.provider.creates, 1)

    async def test_recover_does_not_resume_without_recorded_request(self):
        await self.service.ensure("child")
        await self.service.pause("child")
        self.reopen()
        self.assertEqual((await self.service.recover("child")).state, "paused")
        self.assertEqual(self.provider.resumes, 0)

    async def test_changed_policy_does_not_mutate_existing_sandbox(self):
        await self.service.ensure("child")
        other = RetainedSandboxService(self.store, self.provider, policy=SandboxPolicy())
        with self.assertRaisesRegex(SandboxError, "policy_changed"):
            await other.ensure("child")
        self.assertEqual(self.provider.creates, 1)

    async def test_wrong_resource_return_is_refused(self):
        first = await self.service.ensure("master")
        child = await self.service.ensure("child")
        self.provider.items[first.sandbox_id] = child
        with self.assertRaisesRegex(SandboxError, "identity_mismatch"):
            await self.service.inspect("master")

    async def test_launch_fences_lifecycle_until_released(self):
        async with self.service.use("child"):
            pause = asyncio.create_task(self.service.pause("child"))
            await asyncio.sleep(0)
            self.assertEqual(self.provider.pauses, 0)
        self.assertEqual((await pause).state, "paused")

    def test_profile_has_one_resource_owner(self):
        with self.assertRaisesRegex(SandboxError, "owner_busy"):
            SessionSandboxBindings(self.path, owner_id="owner", profile_id="profile")

    def test_wrong_profile_cannot_adopt_resource_store(self):
        self.store.close()
        with self.assertRaisesRegex(SandboxError, "identity_mismatch"):
            SessionSandboxBindings(self.path, owner_id="owner", profile_id="other-profile")

    async def test_detaching_store_does_not_pause_or_delete(self):
        await self.service.ensure("child")
        self.store.close()
        self.assertEqual(self.provider.pauses, 0)
        self.assertEqual(len(self.provider.items), 1)

    async def test_micro_pause_resumes_same_sandbox_without_repreparing(self):
        original = await self.service.ensure("child")
        resumed = await self.service.micro_pause("child")
        self.assertEqual(resumed.sandbox_id, original.sandbox_id)
        self.assertEqual(resumed.state, "running")
        self.assertEqual((self.provider.creates, self.provider.prepares,
                          self.provider.pauses, self.provider.resumes), (1, 1, 1, 1))
        self.assertIsNone(self.store.get("child").pending_action)

    async def test_micro_pause_does_not_wake_an_already_paused_sandbox(self):
        await self.service.ensure("child")
        await self.service.pause("child")
        with self.assertRaisesRegex(SandboxError, "session_sandbox_paused"):
            await self.service.micro_pause("child")
        self.assertEqual((self.provider.pauses, self.provider.resumes), (1, 0))

    async def test_micro_pause_failed_resume_recovers_without_another_pause(self):
        original = await self.service.ensure("child")
        self.provider.resume_error = SandboxError("sandbox_resume_unknown")
        with self.assertRaises(SandboxError):
            await self.service.micro_pause("child")
        self.reopen()
        self.assertEqual(self.store.get("child").pending_action, "resume")
        self.provider.resume_error = None
        resumed = await self.service.recover("child")
        self.assertEqual(resumed.sandbox_id, original.sandbox_id)
        self.assertEqual(self.provider.pauses, 1)
        self.assertEqual(self.provider.creates, 1)

    async def test_micro_pause_lost_pause_ack_can_resume_after_restart(self):
        original = await self.service.ensure("child")
        real_pause = self.provider.pause

        async def lost_ack(info):
            await real_pause(info)
            raise SandboxError("sandbox_pause_unknown")

        self.provider.pause = lost_ack
        with self.assertRaises(SandboxError):
            await self.service.micro_pause("child")
        self.reopen()
        self.provider.pause = real_pause
        resumed = await self.service.recover("child")
        self.assertEqual(resumed.sandbox_id, original.sandbox_id)
        self.assertEqual(self.provider.pauses, 1)

    async def test_micro_pause_keeps_execution_fenced_through_resume(self):
        await self.service.ensure("child")
        resume_started, release_resume = asyncio.Event(), asyncio.Event()
        real_resume = self.provider.resume

        async def delayed_resume(info, **kwargs):
            resume_started.set()
            await release_resume.wait()
            return await real_resume(info, **kwargs)

        self.provider.resume = delayed_resume
        cycling = asyncio.create_task(self.service.micro_pause("child"))
        await asyncio.wait_for(resume_started.wait(), 1)
        command_ready = asyncio.create_task(self.service.ensure("child"))
        await asyncio.sleep(0)
        self.assertFalse(command_ready.done())
        release_resume.set()
        await cycling
        self.assertEqual((await command_ready).state, "running")

    async def test_micro_pause_cancellation_leaves_recoverable_resume_intent(self):
        await self.service.ensure("child")
        self.provider.resume_error = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.service.micro_pause("child")
        self.reopen()
        self.provider.resume_error = None
        self.assertEqual((await self.service.recover("child")).state, "running")
        self.assertEqual(self.provider.pauses, 1)

    async def test_micro_pause_refuses_another_pending_lifecycle_request(self):
        await self.service.ensure("child")
        self.store.intend(self.store.get("child"), "pause")
        with self.assertRaisesRegex(SandboxError, "lifecycle_pending"):
            await self.service.micro_pause("child")
        self.assertEqual((self.provider.pauses, self.provider.resumes), (0, 0))
