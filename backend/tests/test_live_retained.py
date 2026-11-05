"""Retained native admission, original dialogue and independent work lifetime."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.realtime.live_retained import PREFIX, RetainedLiveWork
from app.tasks.models import TaskError


class RetainedLiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.route = NS(scope=NS(owner_id="fixture", session_id=uuid4()), route_id=uuid4())
        self.master = NS(logical_id="master", native_id="native-master")
        self.receipt = NS(session_id="master", admission_id="admission-1", status="queued")
        self.service = NS(bindings=NS(owner_id="fixture"), master=AsyncMock(return_value=self.master),
            client=NS(snapshot=AsyncMock(return_value={"messages": []}),
                      wait_for_change=AsyncMock(), request=AsyncMock()),
            send=AsyncMock(return_value=self.receipt),
            receipt=AsyncMock(return_value={"status": "terminal", "outcome": "completed",
                                           "result": {"final_response": "The file is ready."}}))
        self.runtime = NS(bridge=NS(service=self.service), wait_ready=AsyncMock())
        self.session = NS(services=NS(live_target_directory=None), ready=True,
                          request=NS(route=self.route))
        self.bridge = RetainedLiveWork(self.session, self.runtime)
        self.addAsyncCleanup(self.bridge.close)
        self.publish = AsyncMock()

    def packet(self, identity="delegation-1", parts=None):
        if parts is None:
            parts = [{"context_event_id": "turn-1", "sequence": 1, "speaker": "user",
                      "delta": "Edit the ORIGINAL video.\nKeep its sound.", "start_ms": 100, "end_ms": 800}]
        return NS(route=self.route, window=NS(request=NS(delegation_id=identity)),
                  as_context=lambda: {"dialogue": parts, "window": {"speech_finality": "unavailable"}})

    async def test_exact_original_handoff_and_reply_not_task_creation(self):
        packet = self.packet()
        await self.bridge(packet, publish=self.publish, is_current=lambda: True)
        sent = self.service.send.await_args.kwargs["text"]
        body = json.loads(sent[len(PREFIX):])
        self.assertEqual(body["dialogue"], packet.as_context()["dialogue"])
        self.assertNotIn("live_state", body)
        await asyncio.sleep(0)
        self.assertEqual(self.publish.await_args.args[0].text, "The file is ready.")
        self.assertIsNone(self.publish.await_args.args[0].task_id)

    async def test_dynamic_app_state_is_not_attached_to_original_dialogue(self):
        packet = self.packet()
        original = packet.as_context()
        original["application_context"] = {
            "recording": {"recording_id": "dynamic-recording-canary"},
            "resource_context": {"sandbox_id": "dynamic-sandbox-canary"}}
        packet.as_context = lambda: original
        await self.bridge(packet, publish=self.publish, is_current=lambda: True)
        sent = self.service.send.await_args.kwargs["text"]
        body = json.loads(sent[len(PREFIX):])
        self.assertEqual(body["dialogue"], original["dialogue"])
        self.assertNotIn("application_context", body)
        self.assertNotIn("dynamic-recording-canary", sent)
        self.assertNotIn("dynamic-sandbox-canary", sent)

    async def test_real_result_guard_ignores_legacy_quiet_ledger(self):
        from test_attention_quiet import LedgerFixture
        from test_attention_quiet_delivery import activate
        pool = LedgerFixture()
        @asynccontextmanager
        async def scope_guard():
            yield pool
        self.session.live = NS(scope_guard=scope_guard)
        result = NS(route=self.route)
        async with self.bridge._guard(result):
            pass
        await activate(pool, "fixture", self.route.scope.session_id)
        async with self.bridge._guard(result) as connection:
            self.assertIs(connection, pool)

    async def test_result_guard_still_rejects_changed_route_unready_and_detached_sessions(self):
        for field, value, result in (
            ("ready", False, NS(route=self.route)),
            ("ready", True, NS(route=NS(scope=self.route.scope, route_id=uuid4()))),
            ("_detached", True, NS(route=self.route)),
        ):
            self.session.ready = True
            target = self.bridge if field == "_detached" else self.session
            setattr(target, field, value)
            with self.subTest(field=field, result=result), self.assertRaises(TaskError):
                async with self.bridge._guard(result):
                    self.fail("unavailable or changed voice route admitted delivery")

    async def test_followup_forwards_only_unseen_parts_to_same_master(self):
        first = self.packet()
        await self.bridge(first, publish=self.publish, is_current=lambda: True)
        second = dict(first.as_context()["dialogue"][0], context_event_id="turn-2",
                      sequence=2, delta="Actually keep only ten seconds.")
        await self.bridge(self.packet("delegation-2", first.as_context()["dialogue"] + [second]),
                          publish=self.publish, is_current=lambda: True)
        body = json.loads(self.service.send.await_args.kwargs["text"][len(PREFIX):])
        self.assertEqual(body["dialogue"], [second])
        self.assertEqual(self.service.send.await_args.args[0], "master")
        self.service.master.assert_awaited_once()

    async def test_existing_native_history_prevents_repeated_dialogue(self):
        packet = self.packet()
        self.service.client.snapshot.return_value = {"messages": [{"role": "user", "content": PREFIX +
            json.dumps({"format": "jarvis.original_dialogue.v1", "dialogue": packet.as_context()["dialogue"]})}]}
        await self.bridge(packet, publish=self.publish, is_current=lambda: True)
        self.service.send.assert_not_awaited()

    async def test_unknown_admission_reuses_same_identity_and_text(self):
        self.service.send.side_effect = [TimeoutError(), self.receipt]
        with self.assertRaises(TimeoutError):
            await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.assertEqual(self.bridge.seen, set())
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        first, second = self.service.send.await_args_list
        self.assertEqual(first, second)
        self.assertEqual(self.bridge.seen, {"turn-1"})

    async def test_wrong_owner_cannot_send(self):
        self.route.scope.owner_id = "another-owner"
        with self.assertRaises(TaskError):
            await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.service.send.assert_not_awaited()

    async def test_admitted_followup_uses_same_native_admission_for_steering(self):
        self.service.steer_live_input = AsyncMock()
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.service.steer_live_input.assert_awaited_once_with(self.receipt,
            jarvis_session_id=self.route.scope.session_id)
        self.service.send.assert_awaited_once()

    async def test_steer_is_admitted_before_transfer_even_with_live_controls(self):
        calls = []
        async def send(*args, **kwargs):
            calls.append("admit")
            return self.receipt
        async def steer(receipt, **kwargs):
            calls.append("steer")
            self.assertIs(receipt, self.receipt)
            return {"status": "started"}
        self.service.send.side_effect = send
        self.service.steer_live_input = AsyncMock(side_effect=steer)
        self.service.control_live_input = AsyncMock()
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.assertEqual(calls, ["admit", "steer"])
        self.service.control_live_input.assert_not_awaited()

    async def test_unread_steer_returned_to_native_queue_keeps_its_result_watcher(self):
        self.service.client.change_sequence = 0
        self.service.control_live_input = AsyncMock()
        self.service.steer_live_input = AsyncMock(return_value={"status": "started"})
        self.service.receipt.side_effect = [
            {"status": "started"},
            {"status": "queued"},  # The finishing turn did not consume the steer.
            {"status": "terminal", "outcome": "completed",
             "result": {"final_response": "Opened the first link."}},
        ]
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        await asyncio.wait_for(asyncio.gather(*tuple(self.bridge._jobs)), 5)
        self.service.send.assert_awaited_once()
        self.service.control_live_input.assert_not_awaited()
        self.assertEqual(self.publish.await_count, 1)
        self.assertEqual(self.publish.await_args.args[0].text, "Opened the first link.")

    async def test_already_finished_turn_still_delivers_queued_followup_once(self):
        self.service.control_live_input = AsyncMock()
        self.service.steer_live_input = AsyncMock(return_value={"status": "queued"})
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        await asyncio.gather(*tuple(self.bridge._jobs))
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.service.send.assert_awaited_once()
        self.service.control_live_input.assert_not_awaited()
        self.publish.assert_awaited_once()

    async def test_steered_guidance_has_no_independent_empty_reply(self):
        self.service.receipt.return_value = {"status": "terminal", "outcome": "steered"}
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        await asyncio.gather(*tuple(self.bridge._jobs))
        self.publish.assert_not_awaited()

    async def test_queue_never_invokes_live_controls(self):
        self.service.steer_live_input = AsyncMock()
        self.service.control_live_input = AsyncMock()
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True, delivery="queue")
        self.service.send.assert_awaited_once()
        self.service.steer_live_input.assert_not_awaited()
        self.service.control_live_input.assert_not_awaited()

    async def test_redirect_joins_existing_result_without_resubmission(self):
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.service.control_live_input = AsyncMock(return_value=(self.receipt, "redirected"))
        parts = [{"context_event_id": "new-guidance", "sequence": 2, "speaker": "user",
                  "delta": "No, open the first result instead."}]
        await self.bridge(self.packet("followup", parts), publish=self.publish,
                          is_current=lambda: True, delivery="redirect")
        await asyncio.gather(*tuple(self.bridge._jobs))
        self.service.send.assert_awaited_once()
        self.service.control_live_input.assert_awaited_once()
        self.assertIn(parts[0]["delta"], self.service.control_live_input.await_args.kwargs["text"])
        self.publish.assert_awaited_once()

    async def test_idle_redirect_queues_original_once(self):
        self.service.control_live_input = AsyncMock(return_value=None)
        self.service.steer_live_input = AsyncMock()
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True, delivery="redirect")
        self.service.send.assert_awaited_once()
        self.service.steer_live_input.assert_not_awaited()

    async def test_uncertain_redirect_does_not_retry_or_submit(self):
        self.service.control_live_input = AsyncMock(return_value=(self.receipt, "unconfirmed"))
        for _ in range(2):
            await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True, delivery="redirect")
        self.service.control_live_input.assert_awaited_once()
        self.service.send.assert_not_awaited()
        self.assertEqual(self.bridge.last_issue, "hermes_guidance_delivery_unconfirmed")

    async def test_cancelled_redirect_cannot_be_replayed_as_a_new_input(self):
        self.service.control_live_input = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True, delivery="redirect")
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True, delivery="redirect")
        self.service.control_live_input.assert_awaited_once()
        self.service.send.assert_not_awaited()
        self.assertIsNone(self.bridge.pending)

    async def test_uncertain_steering_keeps_admission_and_result_watcher(self):
        self.service.steer_live_input = AsyncMock(side_effect=TimeoutError())
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        await asyncio.gather(*tuple(self.bridge._jobs))
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.service.send.assert_awaited_once()
        self.publish.assert_awaited_once()
        self.assertEqual(self.bridge.last_issue, "hermes_guidance_delivery_unconfirmed")

    async def test_failed_initial_history_fetch_can_be_retried(self):
        self.service.client.snapshot.side_effect = [TimeoutError(), {"messages": []}]
        with self.assertRaises(TimeoutError):
            await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.assertIsNone(self.bridge.master)
        self.service.send.assert_not_awaited()
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        self.assertEqual(self.service.client.snapshot.await_count, 2)
        self.service.send.assert_awaited_once()

    async def test_scoped_input_is_bound_before_send_without_fake_foreground(self):
        self.runtime.prepare_voice_input = AsyncMock()
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        args = self.runtime.prepare_voice_input.await_args.kwargs
        self.assertEqual(args["session_id"], "master")
        self.assertEqual(args["jarvis_session_id"], self.route.scope.session_id)
        self.assertNotIn("request_event_id", args)

    async def test_detach_never_cancels_native_work(self):
        self.service.receipt.return_value = {"status": "running"}
        await self.bridge(self.packet(), publish=self.publish, is_current=lambda: True)
        await self.bridge.close()
        self.service.client.request.assert_not_awaited()
        self.publish.assert_not_awaited()

    def test_no_automatic_caption_forwarding_or_directory_io(self):
        self.bridge.offer_captions(("an unrelated conversation fragment",))
        self.assertIsNone(self.bridge.router_directory())
        self.service.send.assert_not_awaited()
        self.runtime.wait_ready.assert_not_awaited()
