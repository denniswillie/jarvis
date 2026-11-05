"""Offline runtime contracts; no provider, device or PostgreSQL acceptance claim."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.assistant.models import BehaviorProfileUpdate
from app.assistant.profile_runtime import BehaviorProfileUpdates
from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.realtime.lean_session import LEAN_INSTRUCTIONS, LeanRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session_manager import RealtimeSessionManager
from app.settings import JarvisSettings
from test_behavior_profile import ProfilePool
from test_lean_voice_session import FakeProvider, FakeSocket


OWNER = "synthetic-runtime-profile-owner"


def profile_event(revision, content="Short answers", owner=OWNER):
    payload = dict(type="behavior_profile.committed", owner_id=owner,
                   event_id=f"behavior-profile:{owner}:{revision}", revision=revision,
                   behavior_instructions=content, committed_at="2026-11-02T03:00:00Z")
    return SimpleNamespace(owner_id=owner, revision=revision,
                           as_payload=lambda: dict(payload))


class ProfileNotifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_is_nonblocking_and_coalesces_newest_per_owner(self):
        receiver = SimpleNamespace(handle_behavior_profile_committed=AsyncMock())
        notifier = BehaviorProfileUpdates(receiver)
        self.addAsyncCleanup(notifier.close)
        first, latest = profile_event(1), profile_event(3)
        self.assertTrue(notifier.submit(first))
        self.assertTrue(notifier.submit(latest))
        self.assertFalse(notifier.submit(profile_event(2)))
        self.assertFalse(notifier.submit(profile_event(3)))
        receiver.handle_behavior_profile_committed.assert_not_awaited()
        await asyncio.wait_for(notifier.task, 1)
        receiver.handle_behavior_profile_committed.assert_awaited_once_with(latest)

    async def test_capacity_rejects_new_owner_without_dropping_pending_owner(self):
        receiver = SimpleNamespace(handle_behavior_profile_committed=AsyncMock())
        notifier = BehaviorProfileUpdates(receiver, max_owners=1)
        self.addAsyncCleanup(notifier.close)
        self.assertTrue(notifier.submit(profile_event(1)))
        with self.assertLogs("app.assistant.profile_runtime", level="WARNING"):
            self.assertFalse(notifier.submit(profile_event(1, owner="another-owner")))
        latest = profile_event(2)
        self.assertTrue(notifier.submit(latest))
        await asyncio.wait_for(notifier.task, 1)
        receiver.handle_behavior_profile_committed.assert_awaited_once_with(latest)

    async def test_failed_delivery_does_not_block_next_owner(self):
        receiver = SimpleNamespace(handle_behavior_profile_committed=AsyncMock(
            side_effect=[RuntimeError("fixture unavailable"), None]))
        notifier = BehaviorProfileUpdates(receiver)
        self.addAsyncCleanup(notifier.close)
        notifier.submit(profile_event(1))
        notifier.submit(profile_event(1, owner="another-owner"))
        with self.assertLogs("app.assistant.profile_runtime", level="WARNING"):
            await asyncio.wait_for(notifier.task, 1)
        self.assertEqual(receiver.handle_behavior_profile_committed.await_count, 2)

    async def test_timeout_does_not_block_next_owner(self):
        delivered = []

        async def receive(event):
            if event.owner_id == OWNER:
                await asyncio.Event().wait()
            delivered.append(event.owner_id)

        notifier = BehaviorProfileUpdates(SimpleNamespace(
            handle_behavior_profile_committed=receive), timeout_seconds=0.01)
        self.addAsyncCleanup(notifier.close)
        notifier.submit(profile_event(1))
        notifier.submit(profile_event(1, owner="another-owner"))
        with self.assertLogs("app.assistant.profile_runtime", level="WARNING"):
            await asyncio.wait_for(notifier.task, 1)
        self.assertEqual(delivered, ["another-owner"])

    async def test_close_cancels_owned_delivery_and_rejects_future_submissions(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def receive(event):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        notifier = BehaviorProfileUpdates(SimpleNamespace(handle_behavior_profile_committed=receive))
        self.addAsyncCleanup(notifier.close)
        notifier.submit(profile_event(1))
        await asyncio.wait_for(entered.wait(), 1)
        notifier.submit(profile_event(2))
        await notifier.close()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(notifier.task.done())
        self.assertEqual(notifier.pending, {})
        self.assertFalse(notifier.submit(profile_event(3)))


class ProfileFanoutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = RealtimeSessionManager()

    async def register(self, owner=OWNER):
        context = SimpleNamespace(owner_id=owner, jarvis_session_id=uuid4(),
                                  audio_session_id=str(uuid4()))
        session = SimpleNamespace(context=context, _closed=asyncio.Event(),
                                  _offer_app_event=Mock(),
                                  notify_behavior_profile_committed=Mock())
        await self.manager.register(owner_id=owner,
            jarvis_session_id=context.jarvis_session_id,
            audio_session_id=context.audio_session_id, session=session)
        return session

    async def test_exact_owner_and_current_delivery_session_are_used(self):
        first, second = await self.register(), await self.register()
        foreign = await self.register("another-owner")
        event = profile_event(4)
        await self.manager.handle_behavior_profile_committed(event)
        for session in (first, second):
            kind = session._offer_app_event.call_args.args[0]
            payload = session._offer_app_event.call_args.kwargs
            self.assertEqual(kind, "behavior_profile.committed")
            self.assertEqual(payload["owner_id"], OWNER)
            self.assertEqual(payload["session_id"], str(session.context.jarvis_session_id))
            self.assertEqual(payload["audio_session_id"], session.context.audio_session_id)
            session.notify_behavior_profile_committed.assert_called_once_with(event)
        foreign._offer_app_event.assert_not_called()
        foreign.notify_behavior_profile_committed.assert_not_called()
        self.assertNotIn("session_id", event.as_payload())

    async def test_closed_sessions_and_resource_shutdown_are_not_notified(self):
        closed, active = await self.register(), await self.register()
        closed._closed.set()
        await self.manager.handle_behavior_profile_committed(profile_event(1))
        closed._offer_app_event.assert_not_called()
        active._offer_app_event.assert_called_once()
        self.manager._resources_closing = True
        await self.manager.handle_behavior_profile_committed(profile_event(2))
        self.assertEqual(active._offer_app_event.call_count, 1)

    async def test_one_bad_session_does_not_suppress_another(self):
        broken, healthy = await self.register(), await self.register()
        broken._offer_app_event.side_effect = RuntimeError("fixture closed")
        with self.assertLogs("app.realtime.session_manager", level="WARNING"):
            await self.manager.handle_behavior_profile_committed(profile_event(1))
        healthy.notify_behavior_profile_committed.assert_called_once()


class LeanProfileRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = ProfilePool()
        self.repository = BehaviorProfileRepository(self.pool)
        self.service = BehaviorProfileService(self.repository)
        self.provider = FakeProvider()
        self.session = LeanRealtimeAudioSession(websocket=FakeSocket(),
            context=RealtimeSessionContext(audio_session_id=str(uuid4()), owner_id=OWNER,
                jarvis_session_id=uuid4(), client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(),
                                       get_recent_messages=AsyncMock()),
            offer="v=0", provider_connection=self.provider,
            runtime_services=SimpleNamespace(behavior_profile_service=self.service))
        self.session._lean_provider_ready = True

    async def asyncTearDown(self):
        await self.session.close("test_done")

    async def drain(self):
        tasks = tuple(task for task in self.session._lean_work
                      if task is not self.session._lean_app_writer)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 1)

    def tracked(self):
        return SimpleNamespace(generation=self.session._turn_generation,
                               budget=SimpleNamespace(record=Mock()))

    async def test_real_shared_tools_are_advertised_without_preloading(self):
        await self.session._send_session_update()
        config = self.provider.sent[-1]["session"]
        self.assertEqual(config["instructions"], LEAN_INSTRUCTIONS)
        self.assertEqual({tool["name"] for tool in config["tools"]},
                         {"live_briefing", "get_behavior_instructions", "set_behavior_instructions"})
        self.assertEqual(self.pool.writes, 0)

    async def test_startup_does_not_wait_for_profile_read(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.service.get

        async def delayed(**kwargs):
            entered.set()
            await release.wait()
            return await original(**kwargs)

        self.service.get = delayed
        self.session._load_live_state = AsyncMock()
        self.session._lean_provider_ready = False
        try:
            await asyncio.wait_for(self.session._initialize_openai_session(), 0.5)
            await asyncio.wait_for(entered.wait(), 0.5)
            self.assertTrue(self.session._realtime_context_ready.is_set())
            self.assertIsNone(self.session._lean_profile)
        finally:
            release.set()
        await self.drain()

    async def test_late_bootstrap_cannot_overwrite_newer_commit(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def stale_read(**kwargs):
            entered.set()
            await release.wait()
            return profile_event(1, "Old").as_payload()

        self.service.get = stale_read
        load = self.session._spawn(self.session._load_behavior_profile())
        await asyncio.wait_for(entered.wait(), 1)
        self.session.notify_behavior_profile_committed(profile_event(2, "Current"))
        release.set()
        await self.drain()
        self.assertTrue(load.done())
        self.assertEqual(self.session._lean_profile["revision"], 2)
        self.assertEqual(self.session._lean_profile_sent_revision, 2)
        self.assertTrue(self.provider.sent)
        for event in self.provider.sent:
            self.assertEqual(event["type"], "session.update")
            self.assertTrue(event["session"]["instructions"].endswith("Current"))

    async def test_provider_writer_resolves_latest_profile_not_queued_stale_text(self):
        gate = asyncio.Event()
        entered = asyncio.Event()
        original = self.provider.send_event

        async def delayed(payload, **kwargs):
            entered.set()
            await gate.wait()
            return await original(payload, **kwargs)

        self.provider.send_event = delayed
        self.session.notify_behavior_profile_committed(profile_event(1, "Old"))
        await asyncio.wait_for(entered.wait(), 1)
        self.session.notify_behavior_profile_committed(profile_event(2, "Newest"))
        gate.set()
        await self.drain()
        self.assertEqual(len(self.provider.sent), 1)
        self.assertTrue(self.provider.sent[0]["session"]["instructions"].endswith("Newest"))

    async def test_invalid_foreign_or_stale_profiles_are_not_accepted(self):
        current = profile_event(3).as_payload()
        self.assertTrue(self.session._accept_behavior_profile(current))
        invalid = [dict(current, owner_id="foreign"), dict(current, revision=True),
                   dict(current, revision=-1), dict(current, revision="4"),
                   dict(current, revision=2), current,
                   dict(current, revision=4, behavior_instructions="x" * 12001),
                   dict(current, revision=4, behavior_instructions=None)]
        for candidate in invalid:
            with self.subTest(candidate=candidate.keys()):
                self.assertFalse(self.session._accept_behavior_profile(candidate))
        self.assertEqual(self.session._lean_profile["revision"], 3)

    async def test_clear_removes_preference_without_forcing_speech_or_cancellation(self):
        self.session.notify_behavior_profile_committed(profile_event(1))
        await self.drain()
        self.session._current_response_id = "ongoing-response"
        self.session.notify_behavior_profile_committed(profile_event(2, ""))
        await self.drain()
        self.assertEqual(self.provider.sent[-1]["session"]["instructions"], LEAN_INSTRUCTIONS)
        self.assertEqual([event["type"] for event in self.provider.sent],
                         ["session.update", "session.update"])
        self.assertEqual(self.session._current_response_id, "ongoing-response")

    async def test_phone_edit_and_voice_edit_share_the_same_repository_and_cas(self):
        await self.service.update(owner_id=OWNER,
            update=BehaviorProfileUpdate(behavior_instructions="Phone edit", expected_revision=0))
        stale = await self.session._execute_behavior_tool("set_behavior_instructions",
            dict(behavior_instructions="Stale voice edit", expected_revision=0), self.tracked())
        self.assertEqual(stale["error"]["code"], "behavior_profile_revision_conflict")
        result = await self.session._execute_behavior_tool("set_behavior_instructions",
            dict(behavior_instructions="Voice edit", expected_revision=1), self.tracked())
        self.assertEqual(result["outcome"], "completed")
        self.assertTrue(result["effect_committed"])
        self.assertEqual((await self.service.get(owner_id=OWNER))["behavior_instructions"], "Voice edit")
        self.assertEqual(self.pool.writes, 2)
        self.assertEqual(self.session._lean_profile_sent_revision, 2)

    async def test_interrupted_unadmitted_mutation_never_writes(self):
        tracked = self.tracked()
        self.session._turn_generation += 1
        result = await self.session._execute_behavior_tool("set_behavior_instructions",
            dict(behavior_instructions="Obsolete", expected_revision=0), tracked)
        self.assertEqual(result["outcome"], "rejected")
        self.assertFalse(result["effect_committed"])
        self.assertEqual(self.pool.writes, 0)
        tracked.budget.record.assert_not_called()

    async def test_interruption_does_not_undo_an_already_admitted_write(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.session._lean_behavior_tools["set_behavior_instructions"].handler

        async def delayed(arguments):
            entered.set()
            await release.wait()
            return await original(arguments)

        self.session._lean_behavior_tools["set_behavior_instructions"] = SimpleNamespace(handler=delayed)
        task = asyncio.create_task(self.session._execute_behavior_tool("set_behavior_instructions",
            dict(behavior_instructions="Accepted", expected_revision=0), self.tracked()))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.session._turn_generation += 1
        finally:
            release.set()
        result = await asyncio.wait_for(task, 1)
        self.assertTrue(result["effect_committed"])
        self.assertEqual(self.pool.writes, 1)
        self.assertEqual([event["type"] for event in self.provider.sent], ["session.update"])

    async def test_ambiguous_write_failure_does_not_claim_rollback_or_repeat(self):
        handler = AsyncMock(side_effect=RuntimeError("fixture uncertain commit"))
        self.session._lean_behavior_tools["set_behavior_instructions"] = SimpleNamespace(handler=handler)
        result = await self.session._execute_behavior_tool("set_behavior_instructions",
            dict(behavior_instructions="Maybe saved", expected_revision=0), self.tracked())
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertIsNone(result["effect_committed"])
        handler.assert_awaited_once()
        self.assertEqual(self.provider.sent, [])

    async def test_unconfirmed_provider_write_is_not_recorded_as_applied(self):
        self.session._accept_behavior_profile(profile_event(1).as_payload())
        self.provider.send_event = AsyncMock(return_value=False)
        self.assertFalse(await self.session._push_behavior_profile())
        self.assertEqual(self.session._lean_profile_sent_revision, -1)
        self.assertFalse(self.session._closed.is_set())

    async def test_tool_has_one_correlated_output_then_one_protocol_continuation(self):
        await self.session._handle_openai_event({"type": "response.created", "response": {"id": "profile-response"}})
        await self.session._handle_openai_event({"type": "response.function_call_arguments.done",
            "response_id": "profile-response", "call_id": "profile-call", "name": "set_behavior_instructions",
            "arguments": json.dumps(dict(behavior_instructions="Concise", expected_revision=0))})
        await self.session._handle_openai_event({"type": "response.done",
            "response": {"id": "profile-response", "status": "completed", "output": []}})
        await self.drain()
        self.assertEqual([event["type"] for event in self.provider.sent],
                         ["session.update", "conversation.item.create", "response.create"])
        output = self.provider.sent[1]["item"]
        self.assertEqual(output["call_id"], "profile-call")
        self.assertEqual(output["type"], "function_call_output")
        self.assertTrue(json.loads(output["output"])["effect_committed"])
        self.assertEqual(self.pool.writes, 1)


if __name__ == "__main__":
    unittest.main()
