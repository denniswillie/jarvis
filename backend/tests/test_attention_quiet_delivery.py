"""Offline actual writer/control paths; no providers, DB, acoustics or UI proof."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.attention.quiet import OwnerQuietRepository, QuietCommand
from app.attention.quiet_delivery import (
    QuietDeliverySuppressed, assert_unsolicited_allowed, quiet_response_guard, unsolicited_response_origin,
)
from app.realtime.live_delegation import current_result, result_guard
from app.realtime.live_session import LiveAudioSession
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveResponseSources
from app.tasks.models import TaskError, TaskStatus
from app.tools.invocation import ToolInvocationContext
from app.voice_routes.live_transport import LiveTransportError
from test_attention_quiet import LedgerFixture
from test_live_voice_transport import TransportFixture
import test_generic_visual_speech as visual_fixture
import test_live_readonly_flow as flow_fixture


async def activate(pool, owner, session):
    source = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
    pool.authorize(owner, session, source)
    repo = OwnerQuietRepository(pool)
    return await repo.save(owner, session, source,
        QuietCommand(action="activate", expected_revision=(await repo.get(owner)).revision))


class QuietDeliveryGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = LedgerFixture()
        self.pool.is_in_transaction = lambda: self.pool.lock.locked()
        self.owner, self.session = "quiet-writer-fixture", uuid4()
        self.sources = LiveResponseSources(self.pool, owner_id=self.owner,
            session_id=self.session, audio_session_id="fixture")

    async def test_all_known_unsolicited_origins_use_same_policy(self):
        await activate(self.pool, self.owner, self.session)
        for origin in ("visual_interrupt", "reminder", "prospective", "live_readonly_result", "live_hermes_notice"):
            with self.subTest(origin=origin), self.assertRaises(QuietDeliverySuppressed):
                async with quiet_response_guard(self.sources, origin=origin):
                    self.fail("No writer may execute while quiet")
        self.assertEqual(len(self.pool.rows), 1)

    async def test_no_storage_or_failed_policy_read_fails_closed(self):
        self.sources.pool = None
        with self.assertRaises(QuietDeliverySuppressed):
            async with quiet_response_guard(self.sources, origin="visual_interrupt"):
                self.fail("Missing authority is not consent")
        bad = SimpleNamespace(fetchrow=AsyncMock(side_effect=RuntimeError("private database data")))
        with self.assertRaises(QuietDeliverySuppressed) as error:
            await assert_unsolicited_allowed(bad, self.owner, origin="live_readonly_result")
        self.assertEqual(error.exception.code, "owner_quiet_unavailable")
        self.assertNotIn("private database data", str(error.exception))

    async def test_owner_lock_linearizes_commit_after_started_write(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def writer():
            async with quiet_response_guard(self.sources, origin="visual_interrupt"):
                self.assertIs(self.sources._current_connection(), self.pool)
                entered.set()
                await release.wait()
        writing = asyncio.create_task(writer())
        await entered.wait()
        command = asyncio.create_task(activate(self.pool, self.owner, self.session))
        try:
            await asyncio.sleep(0)
            self.assertFalse(command.done())
        finally:
            release.set()
            await writing
            await command
        with self.assertRaises(QuietDeliverySuppressed):
            async with quiet_response_guard(self.sources, origin="visual_interrupt"):
                pass

    async def test_source_guard_borrows_owner_transaction(self):
        visited = []
        @asynccontextmanager
        async def source_guard():
            visited.append(self.sources._current_connection())
            async with self.sources.context_guard():
                yield
        async with quiet_response_guard(self.sources, origin="visual_interrupt", source_guard=source_guard):
            self.assertEqual(visited, [self.pool])
        self.assertIsNone(self.sources._current_connection())

    def test_provenance_follows_unsolicited_continuations_not_prompt_flags(self):
        self.assertEqual(unsolicited_response_origin("visual_interrupt"), "visual_interrupt")
        self.assertEqual(unsolicited_response_origin("tool_continuation", {"jarvis_delivery_kind": "visual_interrupt"}),
                         "visual_interrupt")
        self.assertIsNone(unsolicited_response_origin("user", {"quiet": True}))
        self.assertIsNone(unsolicited_response_origin("user", {"jarvis_initiation": "unsolicited"}))


class QuietVisualRuntimeTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = visual_fixture.GenericVisualSpeechTests.asyncSetUp
    asyncTearDown = visual_fixture.GenericVisualSpeechTests.asyncTearDown
    _record_event = visual_fixture.GenericVisualSpeechTests._record_event
    candidate = visual_fixture.GenericVisualSpeechTests.candidate
    creates = visual_fixture.GenericVisualSpeechTests.creates

    async def quiet(self):
        await activate(self.repository._pool, "owner", self.session.context.jarvis_session_id)

    async def test_quiet_rejects_direct_visual_before_reserving_response(self):
        await self.quiet()
        self.assertFalse(await self.candidate())
        self.assertEqual(self.creates(), [])
        self.assertIsNone(self.session._pending_response)

    async def test_quiet_race_rechecked_inside_actual_provider_writer(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.provider.send_event
        async def delayed(event, **kwargs):
            entered.set()
            await release.wait()
            return await original(event, **kwargs)
        self.provider.send_event = delayed
        pending = asyncio.create_task(self.candidate())
        try:
            async with asyncio.timeout(1):
                await entered.wait()
            await self.quiet()
            release.set()
            self.assertFalse(await pending)
            self.assertEqual(self.creates(), [])
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)

    async def test_new_visual_pcm_and_hint_blocked_but_direct_reply_is_not(self):
        await self.quiet()
        self.session._send_app_event = RealtimeAudioSession._send_app_event.__get__(self.session)
        self.session._active_responses["visual"] = SimpleNamespace(delivery_kind="visual_interrupt", metadata={})
        self.session._active_responses["user"] = SimpleNamespace(delivery_kind="user", metadata={})
        self.assertFalse(await self.session._send_app_event("audio.output.delta",
            response_id="visual", item_id="visual-item", audio_base64="AA=="))
        self.assertFalse(await self.session._send_app_event("visual.interrupt.started", interrupt_id=str(uuid4())))
        self.session.websocket.send_text.assert_not_awaited()
        self.assertTrue(await self.session._send_app_event("audio.output.delta",
            response_id="user", item_id="user-item", audio_base64="AA=="))
        self.session.websocket.send_text.assert_awaited_once()
        self.assertTrue((await OwnerQuietRepository(self.repository._pool).get("owner")).suppresses_unsolicited)


class QuietLiveCallbackTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = flow_fixture.ReadonlyFlowTests.asyncSetUp
    asyncTearDown = flow_fixture.ReadonlyFlowTests.asyncTearDown
    wire_start = flow_fixture.ReadonlyFlowTests.wire_start
    finish_model_boundary = flow_fixture.ReadonlyFlowTests.finish_model_boundary
    real_flow = flow_fixture.ReadonlyFlowTests.real_flow
    admit = flow_fixture.ReadonlyFlowTests.admit
    prepare = flow_fixture.ReadonlyFlowTests.prepare

    def suppression(self):
        return QuietDeliverySuppressed("owner_quiet_quiet", "live_readonly_result")

    async def test_actual_live_writer_suppression_is_not_source_revocation(self):
        receipt, _, _, result, host = await self.real_flow()
        with patch("app.realtime.live_delegation.assert_unsolicited_allowed", AsyncMock(side_effect=self.suppression())):
            response = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertFalse(response.provider_received)
        self.assertEqual(response.status, "rejected")
        self.assertEqual(response.reason, "owner_quiet_quiet")
        self.assertFalse(self.wire.route.requires_fenced_replacement)
        self.assertFalse(any(r["type"] == "session.commentary.append" for r in self.wire.wire.socket.sent))
        self.assertEqual((await self.tasks.get(self.scope.owner_id, receipt.task.id)).status, TaskStatus.WAITING)
        before = len(self.wire.wire.socket.sent)
        replay = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertEqual(replay, response)
        self.assertEqual(len(self.wire.wire.socket.sent), before)

    async def test_partial_context_then_quiet_does_not_emit_final_commentary_or_cancel_route(self):
        receipt, _, _, result, host = await self.real_flow(answer="Evidence is limited. " * 60)
        self.assertGreater(len(result.chunks(self.wire.route._b.count_tokens)), 1)
        thinking_before = sum(r["type"] == "session.thinking.append" for r in self.wire.wire.socket.sent)
        calls = 0
        async def gate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise self.suppression()
        with patch("app.realtime.live_delegation.assert_unsolicited_allowed", gate):
            response = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertGreater(calls, 1)
        self.assertEqual(response.status, "unknown")  # Earlier context was handed off, not a complete reply.
        self.assertEqual(response.reason, "owner_quiet_quiet")
        self.assertFalse(response.provider_received)
        self.assertFalse(self.wire.route.requires_fenced_replacement)
        self.assertFalse(any(r["type"] == "session.commentary.append" for r in self.wire.wire.socket.sent))
        self.assertEqual(sum(r["type"] == "session.thinking.append" for r in self.wire.wire.socket.sent),
                         thinking_before + 1)
        self.assertEqual((await self.tasks.get(self.scope.owner_id, receipt.task.id)).status, TaskStatus.WAITING)

    async def test_callback_suppression_never_cancels_work_or_emits_notice(self):
        receipt, _, _, _, host = await self.real_flow()
        host._closed = asyncio.Event()
        host._mark_closing = Mock()
        host._readonly_notice = AsyncMock()
        with patch("app.realtime.live_session.assert_unsolicited_allowed", AsyncMock(side_effect=self.suppression())), \
             patch.object(host.readonly_admission, "cancel", AsyncMock()) as cancel, \
             patch.object(host.adapter, "submit_delegation_result", AsyncMock()) as submit:
            await LiveAudioSession._watch_readonly_result(host, receipt)
        cancel.assert_not_awaited()
        submit.assert_not_awaited()
        host._readonly_notice.assert_not_awaited()
        host._mark_closing.assert_not_called()
        self.assertEqual(host.notification_omission, "owner_quiet_quiet")
        self.assertEqual((await self.tasks.get(self.scope.owner_id, receipt.task.id)).status, TaskStatus.WAITING)

    async def test_callback_writer_race_suppression_does_not_cancel_work(self):
        receipt, _, _, result, host = await self.real_flow(status="succeeded")
        # Finish through the real verifier/task owner so status, result event,
        # attempt and active manifest agree. A status-only rewrite is not a
        # canonical completed result. Cancellation below remains the real hook.
        running = await self.admit(delegation=replace(self.request, delegation_id="quiet-running"))
        self.db.running(running.task)
        host._readonly_receipts[running.delegation_id] = running
        result_payload = (await self.tasks.get(self.scope.owner_id, receipt.task.id)).result_payload_id
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            complete, detail = await self.tasks.live_context_in_transaction(
                connection, self.scope.owner_id, receipt.task.id)
            self.assertEqual(complete.status, TaskStatus.SUCCEEDED)
            self.assertIsNotNone(detail["result"])
            self.assertEqual(detail["manifest_status"], "active")
            self.assertIsNotNone(detail["manifest_id"])
            self.assertIsNotNone(detail["manifest_payload_id"])
            self.assertEqual(await current_result(self.tasks, connection, receipt, self.route), result)
        self.assertTrue(any(event[1] == receipt.task.id and event[3] == receipt.task.revision
                            and event[4] is not None and event[5] == "succeeded" and event[6] == result_payload
                            for event in self.db.task_events))
        self.assertEqual((await self.tasks.get(self.scope.owner_id, running.task.id)).status, TaskStatus.RUNNING)
        real_cancel = host.readonly_admission.cancel
        host._closed = asyncio.Event()
        host._mark_closing = Mock()
        host._readonly_notice = AsyncMock()
        with patch("app.realtime.live_session.assert_unsolicited_allowed", AsyncMock()) as preflight, \
             patch("app.realtime.live_delegation.assert_unsolicited_allowed", AsyncMock(side_effect=self.suppression())) as writer_gate, \
             patch.object(host.readonly_admission, "cancel", AsyncMock(wraps=real_cancel)) as cancel:
            await LiveAudioSession._watch_readonly_result(host, receipt)
        preflight.assert_awaited_once()
        writer_gate.assert_awaited_once()
        self.assertEqual(writer_gate.await_args.kwargs["origin"], "live_readonly_result")
        cancel.assert_not_awaited()
        host._readonly_notice.assert_not_awaited()
        host._mark_closing.assert_not_called()
        self.assertEqual(host.notification_omission, "owner_quiet_quiet")
        self.assertFalse(self.wire.route.requires_fenced_replacement)
        complete = await self.tasks.get(self.scope.owner_id, receipt.task.id)
        active = await self.tasks.get(self.scope.owner_id, running.task.id)
        self.assertEqual(complete.status, TaskStatus.SUCCEEDED)
        self.assertEqual(complete.result_payload_id, result_payload)
        self.assertEqual(complete.revision, receipt.task.revision)
        self.assertEqual(active.status, TaskStatus.RUNNING)
        self.assertEqual(active.revision, running.task.revision)
        self.assertIsNotNone(active.current_attempt_id)
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            # Withholding delivery did not revoke the source-backed result.
            self.assertIsNotNone(await current_result(self.tasks, connection, receipt, self.route))

    async def test_completed_status_without_matching_result_event_is_not_readable(self):
        receipt, _, _, _, _ = await self.real_flow()
        # Deliberately reproduce the prior invalid fixture, separate from the
        # positive writer-race test. The canonical reader must not accept it.
        self.db.tasks[receipt.task.id]["status"] = "succeeded"
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            _, detail = await self.tasks.live_context_in_transaction(connection, self.scope.owner_id, receipt.task.id)
            self.assertIsNone(detail["result"])
            self.assertIsNone(detail["manifest_id"])
            with self.assertRaises(TaskError) as error:
                await current_result(self.tasks, connection, receipt, self.route)
            self.assertEqual(error.exception.code, "live_readonly_result_unavailable")

    async def test_handed_off_commentary_without_ack_is_unknown_not_quiet_or_success(self):
        receipt, _, _, result, host = await self.real_flow()
        self.wire.wire.socket.auto_ack = False
        response = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertEqual(response.status, "unknown")
        self.assertEqual(response.reason, "delegation_result_submission_unconfirmed")
        self.assertFalse(response.provider_received)
        self.assertTrue(any(r["type"] == "session.commentary.append" for r in self.wire.wire.socket.sent))
        self.assertTrue(self.wire.route.requires_fenced_replacement)
        self.assertFalse(hasattr(response, "played"))
        self.assertFalse(hasattr(response, "heard"))
        before = len([r for r in self.wire.wire.socket.sent if r["type"] == "session.commentary.append"])
        retry = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertFalse(retry.provider_received)
        self.assertEqual(len([r for r in self.wire.wire.socket.sent if r["type"] == "session.commentary.append"]), before)


class QuietTransportBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = TransportFixture()
        await self.fixture.start()

    async def asyncTearDown(self):
        await self.fixture.close()

    async def test_prewrite_denial_preserves_typed_reason_and_never_builds_payload(self):
        error = QuietDeliverySuppressed("owner_quiet_quiet", "live_readonly_result")
        @asynccontextmanager
        async def deny():
            raise error
            yield
        payload = Mock(return_value={"type": "fixture.notice"})
        with self.assertRaises(QuietDeliverySuppressed) as caught:
            await self.fixture.transport.send(payload, guard=deny)
        self.assertIs(caught.exception, error)
        payload.assert_not_called()
        self.assertEqual(self.fixture.socket.sent, [])

    async def test_guard_exit_after_socket_handoff_is_not_claimed_prewrite_suppression(self):
        @asynccontextmanager
        async def late_denial():
            yield
            raise QuietDeliverySuppressed("owner_quiet_quiet", "live_readonly_result")
        with self.assertRaisesRegex(LiveTransportError, "sideband_write_unconfirmed"):
            await self.fixture.transport.send(lambda: {"type": "fixture.notice"}, guard=late_denial)
        self.assertEqual(self.fixture.socket.sent, [{"type": "fixture.notice"}])

    async def test_socket_failure_after_handoff_cannot_masquerade_as_quiet_denial(self):
        async def uncertain_send(data):
            raise QuietDeliverySuppressed("owner_quiet_quiet", "live_readonly_result")
        self.fixture.socket.send = AsyncMock(side_effect=uncertain_send)
        with self.assertRaisesRegex(LiveTransportError, "sideband_write_unconfirmed"):
            await self.fixture.transport.send(lambda: {"type": "fixture.notice"})
        self.fixture.socket.send.assert_awaited_once()

