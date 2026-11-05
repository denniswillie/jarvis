import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.sandbox_keepalive import KeepaliveJournal, SandboxKeepalive
from app.hermes_sessions.sandboxes import RetainedSandboxService
from app.sandbox.models import SandboxError, SandboxPolicy
from test_sandboxes import Provider


class ClockProvider(Provider):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def deadline(self, info):
        updated = info.model_copy(update={"expires_at": datetime.fromtimestamp(
            self.clock() + info.policy.lifetime_seconds, timezone.utc)})
        self.items[updated.sandbox_id] = updated
        return updated

    async def create(self, scope, policy):
        return self.deadline(await super().create(scope, policy))

    async def resume(self, info, **kwargs):
        return self.deadline(await super().resume(info, **kwargs))


class KeepaliveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = 1_000_000.0
        self.clock = lambda: self.now
        self.bindings = SessionSandboxBindings(self.root / "resources.sqlite", owner_id="owner", profile_id="profile")
        self.provider = ClockProvider(self.clock)
        self.service = RetainedSandboxService(self.bindings, self.provider, policy=SandboxPolicy(lifetime_seconds=3600))
        self.notices = []
        self.accept_notices = True
        self.journal = KeepaliveJournal(self.root / "maintenance.sqlite", bindings=self.bindings)
        self.manager = self.compose()

    def compose(self):
        async def deliver(notice):
            self.notices.append(notice)
            return self.accept_notices
        return SandboxKeepalive(master_session_id="master", sandboxes=self.service, journal=self.journal,
                                deliver=deliver, clock=self.clock)

    def tearDown(self):
        self.journal.close()
        self.bindings.close()
        self.temp.cleanup()

    async def enable(self):
        await self.service.ensure("child")
        return await self.manager.enable(actor_session_id="master", target_session_id="child")

    async def test_only_master_can_enable(self):
        with self.assertRaisesRegex(SandboxError, "master_required"):
            await self.manager.enable(actor_session_id="child", target_session_id="child")
        self.assertEqual(self.provider.creates, 0)

    async def test_enable_is_allocation_free_and_uses_real_deadline(self):
        with self.assertRaisesRegex(SandboxError, "not_allocated"):
            await self.manager.enable(actor_session_id="master", target_session_id="child")
        policy = await self.enable()
        self.assertEqual(policy["due_at"], self.now + 3300)
        self.assertEqual((self.provider.creates, self.provider.pauses, self.provider.resumes), (1, 0, 0))

    async def test_no_cycle_or_model_notice_before_due(self):
        await self.enable()
        await self.manager.tick()
        self.assertEqual(self.notices, [])
        self.assertEqual(self.provider.pauses, 0)

    async def test_due_cycle_records_exact_resource_without_waking_master(self):
        policy = await self.enable()
        self.now += 3300
        with self.assertLogs("app.hermes_sessions.sandbox_keepalive", level="INFO") as logs:
            await self.manager.tick()
        self.assertEqual((self.provider.pauses, self.provider.resumes), (1, 1))
        self.assertEqual(self.notices, [])
        row = self.journal._db.execute("SELECT result, delivered FROM attempts").fetchone()
        notice = json.loads(row["result"])
        self.assertTrue(row["delivered"])
        self.assertEqual((notice["master"], notice["target"], notice["sandbox_id"]),
                         ("master", "child", policy["sandbox_id"]))
        self.assertEqual(notice["outcome"], "micro_pause_resumed")
        self.assertIn(policy["sandbox_id"], logs.output[0])
        self.assertEqual(self.journal.pending_notices(), [])
        self.assertEqual(self.journal.policy("child")["due_at"], self.now + 3300)

    async def test_failed_delivery_retry_reuses_event_without_repeating_cycle(self):
        await self.enable()
        self.now += 3300
        self.provider.resume_error = SandboxError("sandbox_resume_unknown")
        self.accept_notices = False
        await self.manager.tick()
        self.accept_notices = True
        await self.manager.tick()
        await self.manager.tick()
        self.assertEqual(len(self.notices), 2)
        self.assertEqual(self.notices[0]["event_id"], self.notices[1]["event_id"])
        self.assertEqual(self.provider.pauses, 1)

    async def test_old_queued_success_is_acknowledged_without_waking_master(self):
        policy = await self.enable()
        event = self.journal.begin(policy, self.now)
        self.journal.finish(event, outcome="micro_pause_resumed", observed_state="running")
        # Receipt persisted by the previous version, before silent successes.
        with self.journal._db:
            self.journal._db.execute("UPDATE attempts SET delivered=0")
        self.journal.close()
        self.journal = KeepaliveJournal(self.root / "maintenance.sqlite", bindings=self.bindings)
        self.manager = self.compose()
        await self.manager.tick()
        self.assertEqual(self.notices, [])
        self.assertEqual(self.journal.pending_notices(), [])
        self.assertEqual(self.provider.pauses, 0)

    async def test_disabled_policy_does_not_stop_programs(self):
        await self.enable()
        await self.manager.disable(actor_session_id="master", target_session_id="child")
        self.now += 4000
        await self.manager.tick()
        self.assertEqual(self.provider.pauses, 0)
        self.assertEqual(self.notices, [])

    async def test_paused_resource_requires_master_instead_of_automatic_resume(self):
        await self.enable()
        await self.service.pause("child")
        self.now += 3300
        await self.manager.tick()
        self.assertEqual(self.provider.resumes, 0)
        self.assertEqual(self.notices[0]["outcome"], "paused_requires_master")
        self.assertFalse(self.journal.policy("child")["enabled"])

    async def test_provider_failure_is_not_reported_as_success(self):
        await self.enable()
        self.provider.resume_error = SandboxError("sandbox_resume_unknown")
        self.now += 3300
        await self.manager.tick()
        self.assertEqual(self.notices[0]["outcome"], "unconfirmed")
        self.assertFalse(self.journal.policy("child")["enabled"])
        self.assertEqual(self.bindings.get("child").pending_action, "resume")

    async def test_cancelled_cycle_recovers_and_keeps_same_notification(self):
        await self.enable()
        self.provider.resume_error = asyncio.CancelledError()
        self.now += 3300
        with self.assertRaises(asyncio.CancelledError):
            await self.manager.tick()
        event_id = self.journal.unfinished()[0]["event_id"]
        self.journal.close()
        self.journal = KeepaliveJournal(self.root / "maintenance.sqlite", bindings=self.bindings)
        self.manager = self.compose()
        self.provider.resume_error = None
        await self.manager.tick()
        self.assertEqual(self.provider.pauses, 1)
        self.assertEqual(self.notices[0]["event_id"], event_id)
        self.assertEqual(self.notices[0]["outcome"], "recovered_running_cycle_unconfirmed")
        self.assertEqual(self.journal.unfinished(), [])

    async def test_ordinary_use_extending_deadline_does_not_trigger_early_cycle(self):
        await self.enable()
        self.now += 3300
        info = await self.service.inspect("child")
        self.provider.deadline(info)
        await self.manager.tick()
        self.assertEqual(self.provider.pauses, 0)
        self.assertEqual(self.notices, [])
        self.assertEqual(self.journal.policy("child")["due_at"], self.now + 3300)

    async def test_stopped_timer_performs_no_work(self):
        await self.enable()
        self.now += 3300
        stop = asyncio.Event()
        stop.set()
        await self.manager.run(stop)
        self.assertEqual(self.provider.pauses, 0)
