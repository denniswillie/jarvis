"""Offline retained-result/reconnect regressions; no provider or live profile."""
from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.hermes_sessions.client import HermesGatewayError
from app.live_state.companion import CompanionState
from app.realtime.live_retained import RetainedLiveWork
from app.realtime.live_session import LiveAudioSession
from app.realtime.live_state_reply import compact_live_state_reply
from app.tasks.models import TaskError


class RetainedStateReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.master = NS(logical_id="retained-master", native_id="native-master")
        self.native = {"session_id": self.master.native_id, "running": False,
            "execution_generation": 6, "pending": [], "messages": [
                {"role": "user", "row_id": 101, "content": "Search the visible browser for movies."},
                {"role": "assistant", "row_id": 112, "timestamp": 1793800555.0,
                 "content": "The release calendar lists the films; Dublin showtimes are not confirmed."}]}
        self.service = NS(bindings=NS(owner_id="fixture"), master=AsyncMock(), send=AsyncMock(),
            client=NS(snapshot=AsyncMock(return_value=self.native)))
        self.runtime = NS(master=self.master, bridge=NS(service=self.service), wait_ready=AsyncMock())
        self.session = self.voice_session()
        self.work = self.new_work(self.session)

    def voice_session(self):
        return NS(request=NS(route=NS(scope=NS(owner_id="fixture", session_id=uuid4()))))

    def new_work(self, session):
        work = RetainedLiveWork(session, self.runtime)
        self.addAsyncCleanup(work.close)
        return work

    def live_session(self, *, companion):
        session = object.__new__(LiveAudioSession)
        session.context = NS(owner_id="fixture")
        session.request = self.voice_session().request
        current = NS(owner_id="fixture", read_current=AsyncMock(
            side_effect=lambda: {"human": {}, "system": {}}))
        session.services = NS(companion_state=current if companion else None)
        session.live = NS(requires_reconstruction=False, service=NS(snapshot=lambda: {}))
        session.work = self.new_work(session)
        return session

    async def test_live_session_awaits_retained_read_on_both_paths_after_reconnect(self):
        selection = NS(text="legacy state", expires_at=None, parent_payload_root_ids=())
        with patch("app.realtime.live_session.select_voice_context", return_value=selection):
            for companion in (True, False):
                for reconnect in range(2):
                    with self.subTest(companion=companion, reconnect=reconnect):
                        session = self.live_session(companion=companion)
                        value, roots = await session.read_live_state()
                        work = value["system"]["work"] if companion else value["work"]
                        self.assertEqual(work["master_session_id"], self.master.logical_id)
                        self.assertEqual(work["recent_results"][0]["row_id"], 112)
                        self.assertEqual(work["recent_results"][0]["text"],
                                         self.native["messages"][-1]["content"])
                        self.assertFalse(work["running"])
                        self.assertEqual(roots, ())
                        session.work.detach()
        self.assertEqual(self.service.client.snapshot.await_count, 4)
        self.service.master.assert_not_awaited()
        self.service.send.assert_not_awaited()

    async def test_live_session_preserves_sync_snapshot_fallback_on_both_paths(self):
        selection = NS(text="legacy state", expires_at=None, parent_payload_root_ids=())
        expected = {"last_admission": "existing-sync-work"}
        with patch("app.realtime.live_session.select_voice_context", return_value=selection):
            for companion in (True, False):
                with self.subTest(companion=companion):
                    session = self.live_session(companion=companion)
                    session.work = NS(snapshot=Mock(return_value=expected))
                    value, _ = await session.read_live_state()
                    work = value["system"]["work"] if companion else value["work"]
                    self.assertEqual(work, expected)
                    session.work.snapshot.assert_called_once_with()
        self.service.client.snapshot.assert_not_awaited()

    async def test_reply_skips_remote_inventory_and_keeps_cached_identity_and_result(self):
        refresh = AsyncMock(side_effect=AssertionError("Reply must not refresh remote inventory"))
        inventory = {"ready": False, "items": [{"allocation_id": "cached-exact-id",
            "state": "unknown", "stale": True, "can_view": False}]}
        state = CompanionState(owner_id="fixture", working_memory=None, reminders=None,
            sandbox_refresh=refresh, sandbox_projection=lambda owner: deepcopy(inventory))
        session = self.live_session(companion=True)
        session.services.companion_state = state
        async with asyncio.timeout(1):
            value, _ = await session.read_live_state(purpose="reply")
        self.assertEqual(value["system"]["sandboxes"], inventory)
        self.assertEqual(value["system"]["work"]["recent_results"][0]["row_id"], 112)
        refresh.assert_not_called()
        self.service.client.snapshot.assert_awaited_once_with("native-master")

    async def test_action_and_default_reads_wait_for_fresh_inventory(self):
        for purpose in ("act", None):
            with self.subTest(purpose=purpose):
                entered, release = asyncio.Event(), asyncio.Event()
                inventory = {"ready": False, "items": []}
                async def refresh(owner):
                    self.assertEqual(owner, "fixture")
                    entered.set()
                    await release.wait()
                    inventory.update(ready=True, items=[{"allocation_id": "fresh-exact-id",
                        "generation": "fresh-generation", "state": "running", "can_view": True}])
                state = CompanionState(owner_id="fixture", working_memory=None, reminders=None,
                    sandbox_refresh=refresh, sandbox_projection=lambda owner: deepcopy(inventory))
                session = self.live_session(companion=True)
                session.services.companion_state = state
                args = {} if purpose is None else {"purpose": purpose}
                task = asyncio.create_task(session.read_live_state(**args))
                try:
                    async with asyncio.timeout(1):
                        await entered.wait()
                        self.assertFalse(task.done())
                        release.set()
                        value, _ = await task
                    self.assertTrue(value["system"]["sandboxes"]["ready"])
                    self.assertEqual(value["system"]["sandboxes"]["items"][0]["allocation_id"],
                                     "fresh-exact-id")
                finally:
                    release.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_reply_supports_legacy_state_double_and_sync_work(self):
        session = self.live_session(companion=True)
        session.work = NS(snapshot=Mock(return_value={"last_admission": "sync-work"}))
        value, _ = await session.read_live_state(purpose="reply")
        self.assertEqual(value["system"]["work"], {"last_admission": "sync-work"})
        session.services.companion_state.read_current.assert_awaited_once_with()
        session.work.snapshot.assert_called_once_with()
        self.service.client.snapshot.assert_not_awaited()

    async def test_fresh_voice_restores_result_before_duplicate_connection_metadata(self):
        self.assertIsNone(self.work.snapshot()["master_session_id"])
        original = deepcopy(self.native)
        work = await self.work.read_snapshot()
        sandboxes = {"ready": True, "items": [{"sandbox_allocation_id": "exact-allocation",
            "hermes_session_id": "retained-master", "generation": "g" * 64,
            "state": "running", "can_view": True}]}
        reply = compact_live_state_reply({"human": {}, "system": {
            "sandboxes": sandboxes, "app": {"sandboxes": deepcopy(sandboxes),
                "recording": {"device_capabilities_received": False}}, "work": work}}, "system")
        self.assertEqual(next(iter(reply["system"])), "work")
        self.assertTrue(reply["over_budget"])
        result = reply["system"]["work"]["recent_results"][0]
        self.assertEqual(result, {"row_id": 112, "timestamp": 1793800555.0,
            "text": self.native["messages"][-1]["content"]})
        self.assertFalse(reply["system"]["work"]["running"])
        self.assertEqual(reply["system"]["work"]["execution_generation"], 6)
        self.assertNotIn("sandboxes", reply["system"]["app"])
        self.assertEqual(reply["system"]["sandboxes"], sandboxes)
        self.assertEqual(self.native, original)
        self.service.master.assert_not_awaited()
        self.service.send.assert_not_awaited()
        self.service.client.snapshot.assert_awaited_once_with("native-master")
        self.assertIsNone(self.work.master)  # Do not bypass dialogue dedup restore.

    async def test_reconnect_and_subsequent_read_use_current_native_records(self):
        first = await self.work.read_snapshot()
        self.work.detach()
        reconnect = self.new_work(self.voice_session())
        second = await reconnect.read_snapshot()
        self.assertEqual(first["recent_results"], second["recent_results"])
        self.assertEqual(first["master_session_id"], second["master_session_id"])
        self.native["execution_generation"] = 7
        self.native["messages"].append({"role": "assistant", "row_id": 120,
            "timestamp": 1793800666.0, "content": "The later search has a different result."})
        refreshed = await reconnect.read_snapshot()
        self.assertEqual(refreshed["recent_results"][0]["row_id"], 120)
        self.assertEqual(refreshed["execution_generation"], 7)
        self.assertEqual(self.service.client.snapshot.await_count, 3)
        self.service.send.assert_not_awaited()

    async def test_running_work_does_not_relabel_an_older_answer_as_current_completion(self):
        self.native.update(running=True, execution_generation=7, pending=[{"status": "queued"}])
        value = await self.work.read_snapshot()
        self.assertTrue(value["running"])
        self.assertEqual(value["pending_count"], 1)
        self.assertEqual(value["recent_results"][0]["row_id"], 112)
        self.assertNotIn("outcome", value)

    async def test_only_public_answer_content_is_projected(self):
        self.native["messages"].extend([
            {"role": "tool", "content": "private tool output"},
            {"role": "assistant", "content": "tool preamble", "tool_calls": [{"id": "call"}]},
            {"role": "assistant", "reasoning": "private thought", "content": None},
            {"role": "assistant", "content": "model merge", "display_metadata": {"model_only": True}},
            {"role": "assistant", "row_id": 119, "archived": True, "content": [
                {"type": "reasoning", "text": "private block"},
                {"type": "text", "text": "Public answer."}]}])
        results = (await self.work.read_snapshot())["recent_results"]
        self.assertEqual([row["row_id"] for row in results], [119, 112])
        self.assertEqual(results[0]["text"], "Public answer.")
        self.assertTrue(results[0]["archived"])
        self.assertNotIn("private", json.dumps(results))

    async def test_result_budget_is_explicit_and_safe_for_escaped_unicode(self):
        self.native["messages"] = [{"role": "assistant", "row_id": i,
            "content": "\U0001f30d" * 7000} for i in range(5)]
        value = await self.work.read_snapshot()
        self.assertEqual(value["recent_results"][0]["row_id"], 4)
        self.assertTrue(value["recent_results"][0]["text_truncated"])
        self.assertEqual(value["recent_results_omitted"], 4)
        self.assertLessEqual(sum(len(json.dumps(row["text"], ensure_ascii=True))
            for row in value["recent_results"]), 6000)
        self.assertLess(len(json.dumps(value, ensure_ascii=True)), 7000)

    async def test_recent_result_count_is_bounded_with_explicit_omissions(self):
        self.native["messages"] = [{"role": "assistant", "row_id": i,
            "content": "Result " + str(i)} for i in range(5)]
        value = await self.work.read_snapshot()
        self.assertEqual([row["row_id"] for row in value["recent_results"]], [4, 3, 2])
        self.assertEqual(value["recent_results_omitted"], 2)

    async def test_no_recorded_answer_is_explicit(self):
        self.native["messages"] = []
        value = await self.work.read_snapshot()
        self.assertEqual(value["recent_result_status"], "no_recorded_result")
        self.assertEqual(value["recent_results"], [])

    async def test_repeated_restart_notices_do_not_evict_saved_work(self):
        for i in range(4):
            self.native["messages"].extend([
                {"role": "user", "content":
                 "[SYSTEM EVENT: backend_restarted; not a new user request]\nInspect existing work."},
                {"role": "assistant", "row_id": 120 + i, "content": "No action needed."}])
        value = await self.work.read_snapshot()
        self.assertEqual([r["row_id"] for r in value["recent_results"]], [112])
        self.assertEqual(len(self.native["messages"]), 10)

    async def test_background_completion_after_restart_is_still_a_result(self):
        self.native["messages"].extend([
            {"role": "user", "content":
             "[SYSTEM EVENT: backend_restarted; not a new user request]\nInspect existing work."},
            {"role": "assistant", "row_id": 120, "content": "No action needed."},
            {"role": "user", "content": "[SYSTEM EVENT: background_wake]\nA download finished."},
            {"role": "assistant", "row_id": 122, "content": "Your download is ready."}])
        value = await self.work.read_snapshot()
        self.assertEqual([r["row_id"] for r in value["recent_results"]], [122, 112])

    async def test_wrong_owner_and_missing_master_do_not_read_or_create_work(self):
        self.session.request.route.scope.owner_id = "another-owner"
        with self.assertRaises(TaskError):
            await self.work.read_snapshot()
        self.session.request.route.scope.owner_id = "fixture"
        self.runtime.master = None
        with self.assertRaises(TaskError):
            await self.work.read_snapshot()
        self.service.client.snapshot.assert_not_awaited()
        self.service.master.assert_not_awaited()

    async def test_failed_or_wrong_native_read_never_returns_cached_success(self):
        await self.work.read_snapshot()
        self.service.client.snapshot.side_effect = HermesGatewayError("offline")
        with self.assertRaises(TaskError):
            await self.work.read_snapshot()
        self.service.client.snapshot.side_effect = None
        self.native["session_id"] = "another-master"
        with self.assertRaises(TaskError):
            await self.work.read_snapshot()

    async def test_detach_during_read_discards_the_result(self):
        async def detach(_):
            self.work.detach()
            return self.native
        self.service.client.snapshot.side_effect = detach
        with self.assertRaises(TaskError):
            await self.work.read_snapshot()
        with self.assertRaises(TaskError):
            await self.work.read_snapshot()
        self.service.client.snapshot.assert_awaited_once()


class SandboxProjectionTests(unittest.TestCase):
    def test_nonidentical_app_view_and_all_exact_action_ids_survive(self):
        snapshot = {"human": {}, "system": {
            "sandboxes": {"items": [{"sandbox_allocation_id": "root-exact", "can_view": False}]},
            "app": {"sandboxes": {"items": [{"sandbox_allocation_id": "app-exact", "can_view": True}]}},
            "reminders": [{"reminder_id": "reminder-exact"}],
            "work": {"master_session_id": "master-exact"}}}
        original = deepcopy(snapshot)
        for section in ("all", "system"):
            reply = compact_live_state_reply(snapshot, section, max_framed_bytes=115)
            self.assertEqual(reply["system"], snapshot["system"])
            self.assertEqual(next(iter(reply["system"])), "work")
            self.assertTrue(reply["over_budget"])
        self.assertEqual(snapshot, original)
