"""Focused duty foundations with real isolated persistence, not native/glasses acceptance."""
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.reminders.duties import ChangeDuty, DutySpec, ResponsibilityRepository, SaveDuty
from app.reminders.duty_runtime import initialize_responsibilities
from app.reminders.duty_preferences import NotificationPolicy, ResponsibilityPreferences, SetNotificationPreferences
from app.reminders.duty_tools import ResponsibilityToolModule
from app.reminders.repository import ReminderRepository
from app.runtime.feature_config import SecondBrainSettings
from app.tools.inputs import ToolInputError
from app.tools.invocation import ToolInvocationContext, bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class DutyModelTests(unittest.IsolatedAsyncioTestCase):
    def test_local_daily_schedule_survives_dst_and_skips_missed_periods(self):
        spec = DutySpec(title="Laundry", first_at="2026-10-24T09:00:00+01:00",
                        timezone="Europe/Dublin", every_days=1)
        selected, following = spec.occurrence(spec.first_at, datetime(2026,10,26,12,tzinfo=UTC))
        self.assertEqual(selected, datetime(2026,10,26,9,tzinfo=UTC))
        self.assertEqual(following, datetime(2026,10,27,9,tzinfo=UTC))

    def test_future_once_remains_one_occurrence(self):
        start = datetime.now(UTC) + timedelta(days=1)
        spec = DutySpec(title="Buy paper", first_at=start)
        self.assertEqual(spec.occurrence(start, datetime.now(UTC)), (start, None))

    def test_missing_timezone_and_invalid_cadence_are_rejected(self):
        for extra in ({"first_at":"2026-09-18T12:00:00"}, {"every_days":True},
                      {"every_days":0}, {"timezone":"Not/AZone"}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                DutySpec.model_validate({"title":"Keys", "first_at":datetime.now(UTC), **extra})

    async def test_disabled_capability_is_inert(self):
        services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL=None))
        self.assertIsNone(await initialize_responsibilities(services, SecondBrainSettings(_env_file=None)))
        self.assertFalse(hasattr(services, "tool_modules"))

    def test_only_session_bound_tools_are_registered(self):
        module = ResponsibilityToolModule(ResponsibilityRepository(None))
        self.assertEqual(module.build_tools(
            ToolBuildContext(settings=None, owner_id="fixture", session_id=None)), [])
        names = {tool.name for tool in module.build_tools(
            ToolBuildContext(settings=None, owner_id="fixture", session_id=uuid4()))}
        self.assertEqual(names, {"save_responsibility", "change_responsibility",
                                "list_responsibilities", "set_responsibility_preferences"})


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class DutyDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.repo = ResponsibilityRepository(self.f.pool)
        self.reminders = ReminderRepository(self.f.pool)
        async with self.f.pool.acquire() as c:
            self.session = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.other_session = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)

    async def source(self, *, session=None, owner=None):
        owner = owner or self.f.owner
        async with self.f.pool.acquire() as c:
            await c.execute("insert into jarvis_context_streams(owner_id) values($1) on conflict do nothing", owner)
            payload = await c.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", owner, json.dumps({"text":"Explicit fixture user request."}))
            event = await c.fetchval("""insert into jarvis_context_events
                (owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,$2,'focused-duty-fixture',$3,'user_transcript','final','user_input',$4) returning id""",
                owner, session or self.session, str(uuid4()), payload)
        return ToolInvocationContext(request_event_id=event, request_revision=1)

    async def create(self, *, condition=None, every_days=None):
        source = await self.source()
        request = SaveDuty(spec=DutySpec(title="Check belongings", details="Take the umbrella.",
            condition=condition, first_at=datetime.now(UTC)-timedelta(seconds=1),
            timezone="Europe/Dublin", every_days=every_days))
        return await self.repo.save(self.f.owner, self.session, source, request), source, request

    def change_request(self, result, operation, **kwargs):
        return ChangeDuty(responsibility_id=result["responsibility"]["id"],
            expected_revision=result["responsibility"]["revision"], operation=operation,
            occurrence_id=result["occurrences"][0]["id"] if operation!="cancel" else None, **kwargs)

    async def test_restart_and_provider_retry_do_not_duplicate_rule_or_occurrence(self):
        first, source, request = await self.create(every_days=1)
        restarted = ResponsibilityRepository(self.f.pool)
        replay = await restarted.save(self.f.owner, self.session, source, request)
        self.assertTrue(replay["request_replayed"])
        self.assertEqual(first["responsibility"]["id"], replay["responsibility"]["id"])
        await restarted.materialize(self.f.owner)
        later = await restarted.list(self.f.owner)
        self.assertEqual(len(later["duties"]), 1)
        self.assertEqual(len(later["duties"][0]["occurrences"]), 1)
        with self.assertRaises(ToolInputError):
            await restarted.save(self.f.other, self.other_session, source, request)

    async def test_recurring_materialization_collapses_missed_periods_and_never_reports_completion(self):
        first, _, _ = await self.create(every_days=1)
        await self.repo.materialize(self.f.owner, now=datetime.now(UTC)+timedelta(days=3))
        later = (await self.repo.list(self.f.owner))["duties"][0]
        self.assertEqual(len(later["occurrences"]), 2)
        self.assertEqual(later["occurrences"][0]["status"], "open")
        self.assertEqual(later["occurrences"][1]["status"], "superseded")
        self.assertTrue(all(item["report_state"]=="unknown" for item in later["occurrences"]))
        self.assertNotEqual(later["occurrences"][0]["reminder_id"], first["occurrences"][0]["reminder_id"])

    async def test_situational_snooze_hides_cue_and_blocks_timer_then_cancellation_survives_restart(self):
        first, _, _ = await self.create(condition="When the office doorway is visible", every_days=1)
        self.assertEqual(len(await self.reminders.list_active_reminders(owner_id=self.f.owner)), 1)
        snoozed = await self.repo.change(self.f.owner, self.session, await self.source(),
            self.change_request(first, "snooze", until=datetime.now(UTC)+timedelta(hours=2)))
        self.assertEqual(await self.reminders.list_active_reminders(owner_id=self.f.owner), [])
        self.assertEqual(await self.reminders.claim_due_reminders(owner_id=self.f.owner,
            claim_timeout_seconds=120, max_attempts=3), [])
        self.assertNotEqual(snoozed["occurrences"][0]["reminder_id"], first["occurrences"][0]["reminder_id"])
        cancelled = await self.repo.change(self.f.owner, self.session, await self.source(),
                                           self.change_request(snoozed, "cancel"))
        self.assertEqual(cancelled["responsibility"]["status"], "cancelled")
        restarted = ResponsibilityRepository(self.f.pool)
        await restarted.materialize(self.f.owner, now=datetime.now(UTC)+timedelta(days=4))
        self.assertEqual(await self.reminders.list_active_reminders(owner_id=self.f.owner), [])

    async def test_notification_completion_is_not_action_completion_and_reports_keep_provenance(self):
        first, _, _ = await self.create()
        reminder_id = UUID(first["occurrences"][0]["reminder_id"])
        await self.reminders.complete_reminder(owner_id=self.f.owner, reminder_id=reminder_id)
        view = (await self.repo.list(self.f.owner))["duties"][0]
        self.assertEqual(view["occurrences"][0]["report_state"], "unknown")
        reported = await self.repo.change(self.f.owner, self.session, await self.source(),
                                          self.change_request(view, "report_done"))
        self.assertEqual(reported["occurrences"][0]["report_state"], "user_reported_done")
        conflicting = await self.repo.change(self.f.owner, self.session, await self.source(),
                                              self.change_request(reported, "report_not_done"))
        self.assertEqual(conflicting["occurrences"][0]["report_state"], "user_reported_not_done")
        self.assertEqual(conflicting["occurrences"][0]["status"], "open")
        self.assertEqual(conflicting["responsibility"]["status"], "active")
        self.assertEqual(len((await self.repo.list(self.f.owner))["duties"]), 1)
        self.assertEqual(len(conflicting["report_history"]), 2)
        self.assertEqual(conflicting["report_history"][0]["report_state"], "user_reported_not_done")
        self.assertIn("not that Jarvis observed", conflicting["meaning"])

    async def test_edit_invalidates_old_occurrence_and_preserves_unrelated_duty(self):
        first, _, _ = await self.create(condition="When leaving the desk")
        unrelated, _, _ = await self.create()
        request = SaveDuty(responsibility_id=first["responsibility"]["id"], expected_revision=1,
            spec=DutySpec(title="Keys instead", condition="When entering the hallway",
                          first_at=datetime.now(UTC), every_days=7))
        changed = await self.repo.save(self.f.owner, self.session, await self.source(), request)
        self.assertEqual(changed["responsibility"]["revision"], 2)
        self.assertEqual(changed["occurrences"][1]["status"], "superseded")
        control = (await self.repo.list(self.f.owner, responsibility_id=UUID(unrelated["responsibility"]["id"])))["duties"][0]
        self.assertEqual(control["responsibility"]["revision"], 1)
        with self.assertRaises(ToolInputError):
            await self.repo.change(self.f.owner, self.session, await self.source(),
                                   self.change_request(first, "cancel"))

    async def test_real_tool_module_requires_canonical_request_and_can_read_in_a_later_session(self):
        tools = {t.name:t for t in ResponsibilityToolModule(self.repo).build_tools(
            ToolBuildContext(settings=None, owner_id=self.f.owner, session_id=self.session))}
        args = SaveDuty(spec=DutySpec(title="Laundry", first_at=datetime.now(UTC), every_days=7)).model_dump(mode="json")
        refused = await tools["save_responsibility"].handler(args)
        self.assertEqual(refused["error"]["code"], "request_evidence_required")
        source = await self.source()
        with bind_tool_invocation(provider_call_id="first", request_event_id=source.request_event_id, request_revision=1):
            first = await tools["save_responsibility"].handler(args)
        with bind_tool_invocation(provider_call_id="new-provider-id", request_event_id=source.request_event_id, request_revision=1):
            replay = await tools["save_responsibility"].handler(args)
        self.assertEqual(first["status"], "success")
        self.assertTrue(replay["request_replayed"])
        async with self.f.pool.acquire() as c:
            later_session = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
        later_tools = {t.name:t for t in ResponsibilityToolModule(self.repo).build_tools(
            ToolBuildContext(settings=None, owner_id=self.f.owner, session_id=later_session))}
        later = await later_tools["list_responsibilities"].handler({})
        self.assertEqual(later["duties"][0]["responsibility"]["id"], first["responsibility"]["id"])


    async def test_report_corrections_do_not_revive_cancelled_or_superseded_work(self):
        for replacement in (False, True):
            with self.subTest(replacement=replacement):
                first, _, _ = await self.create()
                done = await self.repo.change(self.f.owner, self.session, await self.source(),
                                              self.change_request(first, "report_done"))
                if replacement:
                    current = await self.repo.save(self.f.owner, self.session, await self.source(),
                        SaveDuty(responsibility_id=done["responsibility"]["id"],
                            expected_revision=done["responsibility"]["revision"],
                            spec=DutySpec(title="A later request", first_at=datetime.now(UTC)+timedelta(days=2))))
                else:
                    current = await self.repo.change(self.f.owner, self.session, await self.source(),
                                                      self.change_request(done, "cancel"))
                changed = await self.repo.change(self.f.owner, self.session, await self.source(),
                    ChangeDuty(responsibility_id=current["responsibility"]["id"],
                        expected_revision=current["responsibility"]["revision"],
                        occurrence_id=first["occurrences"][0]["id"], operation="report_not_done"))
                old = next(o for o in changed["occurrences"] if o["id"]==first["occurrences"][0]["id"])
                self.assertNotEqual(old["status"], "open")
                self.assertEqual(changed["responsibility"]["status"], "active" if replacement else "cancelled")

    async def test_preference_tool_is_registered_and_persists_through_the_real_handler(self):
        tools = {tool.name:tool for tool in ResponsibilityToolModule(self.repo).build_tools(
            ToolBuildContext(settings=None, owner_id=self.f.owner, session_id=self.session))}
        arguments = SetNotificationPreferences(expected_revision=0, policy=NotificationPolicy(
            timezone="Europe/Dublin", quiet_start="22:00", quiet_end="08:00")).model_dump(mode="json")
        refused = await tools["set_responsibility_preferences"].handler(arguments)
        self.assertEqual(refused["error"]["code"], "request_evidence_required")
        source = await self.source()
        with bind_tool_invocation(provider_call_id="preferences",
                                  request_event_id=source.request_event_id, request_revision=1):
            saved = await tools["set_responsibility_preferences"].handler(arguments)
        self.assertEqual(saved["status"], "success")
        read = await tools["list_responsibilities"].handler({})
        self.assertEqual(read["notification_preferences"]["revision"], 1)
        self.assertEqual(read["notification_preferences"]["policy"]["quiet_start"], "22:00:00")

    async def set_policy(self, policy, expected=0, source=None):
        return await ResponsibilityPreferences(self.repo).save(self.f.owner, self.session,
            source or await self.source(), SetNotificationPreferences(expected_revision=expected, policy=policy))

    async def test_quiet_hours_are_conversation_owned_replay_safe_and_gate_current_cues(self):
        first, _, _ = await self.create(condition="When the doorway is visible")
        source = await self.source()
        policy = NotificationPolicy(timezone="Europe/Dublin", quiet_start="22:00", quiet_end="08:00")
        await self.set_policy(policy, source=source)
        replay = await self.set_policy(policy, source=source)
        self.assertTrue(replay["request_replayed"])
        self.assertEqual(replay["notification_preferences"]["revision"], 1)
        async with self.f.pool.acquire() as c:
            boundary = await c.fetchval("select jarvis_responsibility_notice_time($1,$2,$3)",
                self.f.owner, UUID(first["occurrences"][0]["reminder_id"]),
                datetime(2026,9,18,22,tzinfo=UTC))
        self.assertEqual(boundary, datetime(2026,9,19,7,tzinfo=UTC))
        await self.set_policy(NotificationPolicy(paused_until=datetime.now(UTC)+timedelta(hours=1)), expected=1)
        self.assertEqual(await self.reminders.list_active_reminders(owner_id=self.f.owner), [])
        from app.memory.prospective.repository import ProspectiveRepository
        self.assertIsNone(await ProspectiveRepository(self.f.pool).try_claim_subject(
            owner_id=self.f.owner, subject_type="conditional_reminder",
            subject_id=UUID(first["occurrences"][0]["reminder_id"]), claim_timeout_seconds=120))

    async def test_queued_notice_resumes_when_pause_clears_without_a_second_occurrence(self):
        from app.attention.repository import AttentionRepository
        from app.tasks.repository import TaskRepository
        from app.runtime.feature_config import TaskLimits
        await self.create()
        due = (await self.reminders.claim_due_reminders(owner_id=self.f.owner,
            claim_timeout_seconds=120, max_attempts=3))[0]
        inbox = AttentionRepository(TaskRepository(self.f.pool, TaskLimits()))
        await self.reminders.finalize_trigger(owner_id=self.f.owner, reminder_id=due.id,
            claim_token=due.trigger_claim_token, session_id=self.session, message="fixture", attention=inbox)
        await self.set_policy(NotificationPolicy(paused_until=datetime.now(UTC)+timedelta(hours=1)))
        self.assertIsNone(await inbox.claim_next(self.f.owner, self.session, "quiet-fixture"))
        await self.set_policy(NotificationPolicy(), expected=1)
        self.assertIsNotNone(await inbox.claim_next(self.f.owner, self.session, "awake-fixture"))
        self.assertEqual(len((await self.repo.list(self.f.owner))["duties"][0]["occurrences"]), 1)

    async def test_preference_change_preserves_an_independent_inbox_deferral(self):
        from app.attention.repository import AttentionRepository
        from app.tasks.repository import TaskRepository
        from app.runtime.feature_config import TaskLimits
        await self.create()
        due = (await self.reminders.claim_due_reminders(owner_id=self.f.owner,
            claim_timeout_seconds=120, max_attempts=3))[0]
        inbox = AttentionRepository(TaskRepository(self.f.pool, TaskLimits()))
        await self.reminders.finalize_trigger(owner_id=self.f.owner, reminder_id=due.id,
            claim_token=due.trigger_claim_token, session_id=self.session, message="fixture", attention=inbox)
        await self.set_policy(NotificationPolicy(paused_until=datetime.now(UTC)+timedelta(hours=1)))
        manual = datetime.now(UTC)+timedelta(hours=6)
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_attention_intents set not_before=$2 where owner_id=$1", self.f.owner, manual)
        await self.set_policy(NotificationPolicy(), expected=1)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select not_before from jarvis_attention_intents where owner_id=$1",
                                             self.f.owner), manual)

    async def test_inference_requires_confirmation_unless_conversation_policy_allows_it(self):
        request = SaveDuty(spec=DutySpec(title="A suggested routine", first_at=datetime.now(UTC),
            every_days=1, basis="inferred_under_policy"))
        with self.assertRaises(ToolInputError):
            await self.repo.save(self.f.owner, self.session, await self.source(), request)
        await self.set_policy(NotificationPolicy(confirm_inferred_obligations=False))
        saved = await self.repo.save(self.f.owner, self.session, await self.source(), request)
        self.assertEqual(saved["responsibility"]["spec"]["basis"], "inferred_under_policy")
