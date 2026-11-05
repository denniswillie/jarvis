"""Task completion is stale context, not permission to reuse it or lose a turn.

Real task repository/projector/binder, route bridge/adapter, and a real session
publisher/writer. Storage and external transports are controlled, not SQL/native
qualification. The returned task result is not a claim of delivery or hearing.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import LiveSnapshot, StateCursor
from app.live_state.projection import select_voice_context
from app.live_state.task_context import TaskContextProjector
from app.realtime.live_state_bridge import _ContextValidityCheck, prepare_voice_context
from app.realtime.models import RealtimeToolFollowup, RealtimeToolReceipt, RealtimeToolRequest
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository
from app.voice_routes.contracts import RouteIdentity, RouteStart
from app.voice_routes.coordinator import CoordinationPhase, VoiceRouteCoordinator
import test_live_state_task_context as task_fixture
import test_realtime_voice_route as route_fixture
import test_voice_context_response_pipeline as pipeline


def finish_task(db, identity, now):
    """Controlled canonical storage transition, not a genuine worker/SQL claim."""
    row = db.tasks[identity]
    attempt = row["current_attempt_id"]
    result = db.payload({"answer": "Three alternatives; choose B; five steps."})
    manifest = db.payload({"request": "Original authorized task context"})
    row.update(status="succeeded", current_attempt_id=None, result_payload_id=result,
               updated_at=now, last_event_sequence=row["last_event_sequence"] + 1)
    db.receipts[identity] = dict(event_id=uuid4(), result_recorded_at=now,
        result_attempt_id=attempt, manifest_id=uuid4(), manifest_status="active",
        manifest_payload_id=manifest, task_id=identity, revision=row["revision"],
        event_type="succeeded", payload_id=result)


class TaskProgressGuardTests(unittest.IsolatedAsyncioTestCase):
    setUp = task_fixture.TaskContextTests.setUp
    project = task_fixture.TaskContextTests.project

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0)

    async def setup_route(self):
        self.identity = self.db.task()
        self.db.tasks[self.identity]["current_attempt_id"] = uuid4()
        self.route = RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1)
        self.epoch = uuid4()
        self.control = route_fixture.FakeControl(self.route)
        self.adapter = route_fixture.make_adapter(self.control, clock=lambda: self.now)

        @asynccontextmanager
        async def guard(update):
            if update.route != self.route:
                raise SourceError("voice_context_not_admitted", "Wrong route")
            if update.expires_at <= self.now:
                raise SourceError("voice_context_expired", "Expired")
            async with self.binder.guard(update.guard_binding) as connection:
                yield connection

        self.guard = guard
        self.adapter._bindings = replace(self.adapter._bindings, source_guard=guard)
        self.installed = await self.update(91)
        validity = _ContextValidityCheck(guard, SimpleNamespace(healthy=True,
            requires_reconstruction=False), lambda: self.now)
        self.owner = VoiceRouteCoordinator(self.scope, validity=validity,
            clock=lambda: self.now, preflight_expiry_refresh=True)
        await self.owner.start(self.adapter, RouteStart(route=self.route, command_id=uuid4(),
            authorization_ref=uuid4(), requested_at=self.now, initial_context=self.installed))
        self.addAsyncCleanup(self.owner.close)
        self.bridge = self.owner._bridge
        self.db.tasks[self.identity]["last_event_sequence"] += 2
        self.db.tasks[self.identity]["updated_at"] = self.now - timedelta(seconds=1)
        self.pending = await self.update(113)

    async def update(self, revision):
        selection = select_voice_context(LiveSnapshot(scope=self.scope,
            cursor=StateCursor(epoch=self.epoch, revision=revision), as_of=self.now,
            readiness="ready", entries=await self.project([self.identity])), now=self.now)
        return await prepare_voice_context(selection, self.route, bind=self.binder,
                                           clock=lambda: self.now)

    async def test_installed91_and_pending113_refresh_to_current_without_reusing_either(self):
        await self.setup_route()
        finish_task(self.db, self.identity, self.now)
        for old in (self.installed, self.pending):
            self.assertGreater(old.expires_at, self.now)
            with self.assertRaises(SourceError):
                async with self.guard(old):
                    self.fail("Stale task state was authorized")
        self.owner.offer_context(self.pending)
        await self.until(lambda: self.bridge.status.fenced or self.bridge.awaiting_fresh_context)
        self.assertFalse(self.bridge.status.fenced, self.bridge.status.reason)
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.assertEqual(len(self.control.writes), 1)
        self.assertFalse(await self.bridge.wait_for_fresh_context(.01))
        current = await self.update(114)
        self.owner.offer_context(current)
        self.assertTrue(await self.bridge.wait_for_fresh_context(1))
        self.assertEqual(self.adapter._context, current)
        self.assertNotEqual(current.guard_binding, self.pending.guard_binding)
        self.assertIn("succeeded", current.text)
        self.assertEqual([x["type"] for x in self.control.writes],
            ["conversation.item.create", "conversation.item.delete", "conversation.item.create"])
        self.assertEqual(self.owner.offer_context(current).disposition.value, "duplicate")
        self.assertEqual(len(self.control.writes), 3)

    async def test_stale_installed_does_not_block_already_current_candidate(self):
        await self.setup_route()
        finish_task(self.db, self.identity, self.now)
        current = await self.update(114)
        self.owner.offer_context(current)
        await self.until(lambda: self.bridge.status.last_receipt is not None)
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual(self.adapter._context, current)

    async def test_correction_cancel_ownership_withdrawal_and_replaced_attempt_stay_hard(self):
        for fault in ("revision", "cancelled", "cancel_requested", "request", "withdrawal",
                      "owner", "attempt", "manifest"):
            with self.subTest(fault=fault):
                self.setUp()
                await self.setup_route()
                finish_task(self.db, self.identity, self.now)
                row = self.db.tasks[self.identity]
                if fault == "revision": row["revision"] += 1
                elif fault in ("cancelled", "cancel_requested"): row["status"] = fault
                elif fault == "request": self.db.events[row["request_event_id"]]["is_current"] = False
                elif fault == "withdrawal": self.db.remove(row["description_payload_id"])
                elif fault == "owner": row["owner_id"] = "foreign"
                elif fault == "attempt": self.db.receipts[self.identity]["result_attempt_id"] = uuid4()
                elif fault == "manifest": self.db.receipts[self.identity]["manifest_status"] = "revoked"
                self.owner.offer_context(self.pending)
                await self.until(lambda: self.bridge.status.fenced)
                self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
                self.assertFalse(self.bridge.awaiting_fresh_context)
                self.assertEqual(len(self.control.writes), 1)
                await self.owner.close()

    async def test_completion_after_unacknowledged_delete_is_not_unwritten_recovery(self):
        await self.setup_route()
        # A local delete send is not an ACK; this remains an unknown mutation.
        self.control.item_event = None
        write = self.control._write

        async def change_after_delete(event):
            result = await write(event)
            if self.control.writes[-1]["type"] == "conversation.item.delete":
                finish_task(self.db, self.identity, self.now)
            return result

        self.control._write = change_after_delete
        self.owner.offer_context(self.pending)
        await self.until(lambda: self.bridge.status.fenced)
        self.assertEqual([x["type"] for x in self.control.writes],
                         ["conversation.item.create", "conversation.item.delete"])
        self.assertFalse(self.bridge.awaiting_fresh_context)


class TaskStorage(pipeline.Storage):
    """Share controlled canonical payload rows with the existing session fixture."""
    task_db = None

    async def fetchval(self, query, owner, value=None):
        result = await super().fetchval(query, owner, value)
        if self.task_db is not None and query.startswith("insert into jarvis_context_payloads"):
            self.task_db.payloads[result] = dict(owner_id=owner, content=json.loads(value),
                availability="available", created_at=task_fixture.NOW)
        return result

    async def executemany(self, query, rows):
        rows = tuple(rows)
        await super().executemany(query, rows)
        if self.task_db is not None:
            for owner, parent, child in rows:
                self.task_db.parents.setdefault(child, set()).add(parent)


class TaskProgressSessionTests(unittest.IsolatedAsyncioTestCase):
    cleanup = pipeline.VoiceContextResponsePipelineTests.cleanup
    until = pipeline.VoiceContextResponsePipelineTests.until
    rows = pipeline.VoiceContextResponsePipelineTests.rows
    mutations = pipeline.VoiceContextResponsePipelineTests.mutations
    creates = pipeline.VoiceContextResponsePipelineTests.creates

    async def asyncSetUp(self):
        with patch.object(pipeline, "Storage", TaskStorage):
            await pipeline.VoiceContextResponsePipelineTests.asyncSetUp(self)

    async def observation(self, summary, **_):
        if not hasattr(self, "task_db"):
            self.task_db = task_fixture.Database(self.scope)
            self.storage.task_db = self.task_db
            self.task_id = self.task_db.task(area=self.scope.area_id)
            self.task_db.tasks[self.task_id]["current_attempt_id"] = uuid4()
            binder = CanonicalContextBinder(SourceRepository(self.task_db), self.scope,
                                            clock=lambda: self.now)
            self.task_projector = TaskContextProjector(binder, TaskRepository(self.task_db, TaskLimits()))
            self.live.binder, self.live.bind, self.live.guard = binder, binder, binder.guard
            self.live.scope_guard = binder._transaction
        entries = await self.task_projector.project([self.task_id], received_at=self.now,
                                                     expires_at=self.now + timedelta(seconds=60))
        self.assertEqual(len(entries), 1)
        self.assertTrue(self.producer.publish(entries[0]).queued)
        await self.service.wait_idle()
        return entries[0]

    async def expire_inside_adapter_guard(self, roots):
        for root in roots:
            if root in self.task_db.payloads and not self.task_db.available(self.scope.owner_id, root):
                self.storage.rejected.add(root)

    async def invalidate_pending(self):
        bridge = self.runtime.coordinator._bridge
        validity = bridge._validity
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(update):
            entered.set()
            await release.wait()
            return await validity(update)

        bridge._validity = delayed
        self.task_db.tasks[self.task_id]["last_event_sequence"] += 1
        await self.observation("pending running projection")
        await asyncio.wait_for(entered.wait(), 2)
        finish_task(self.task_db, self.task_id, self.now)
        release.set()
        await self.until(lambda: bridge.status.fenced or bridge.awaiting_fresh_context)
        bridge._validity = validity
        self.assertFalse(bridge.status.fenced, bridge.status.reason)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.session._closed.is_set())

    def followup(self):
        row = self.task_db.tasks[self.task_id]
        request = self.task_db.events[row["request_event_id"]]
        reservation = self.session._reserve_response_locked("normal_user_turn", {})
        reservation.tool_followup = RealtimeToolFollowup(
            RealtimeToolRequest(self.scope.owner_id, self.scope.session_id,
                reservation.turn_generation, request["id"], request["payload_id"],
                request["revision"], "original-input"),
            (RealtimeToolReceipt("original-status-call", "get_background_task", "success",
                event_id=self.task_db.receipts[self.task_id]["event_id"],
                payload_id=row["result_payload_id"]),))
        self.session._source_context.observe((request["payload_id"], row["result_payload_id"]))
        return reservation

    def hold_normal_scope(self):
        from contextlib import asynccontextmanager
        entered, release = asyncio.Event(), asyncio.Event()
        self.addCleanup(release.set)
        original = self.runtime._normal_turn_scope_guard

        @asynccontextmanager
        async def held():
            entered.set()
            await release.wait()
            async with original() as connection:
                yield connection

        self.runtime._normal_turn_scope_guard = held
        return entered, release

    async def test_owned_followup_writes_once_without_optional_projection(self):
        await self.invalidate_pending()
        reservation = self.followup()
        preparing = asyncio.create_task(self.session._prepare_and_send_response(reservation, None, None))
        self.addAsyncCleanup(self.join, preparing)
        self.assertTrue(await asyncio.wait_for(preparing, 2))
        self.assertFalse(self.runtime.ready)
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.creates()[0]["event_id"], reservation.request_id)
        self.assertEqual(self.creates()[0]["response"]["instructions"].count("original-status-call"), 1)
        self.assertTrue(reservation.submitted)
        await asyncio.sleep(.12)
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(self.session._closed.is_set())

    async def test_missing_optional_projection_does_not_block_owned_followup(self):
        await self.invalidate_pending()
        self.session.settings.OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS = .04
        reservation = self.followup()
        self.assertTrue(await asyncio.wait_for(
            self.session._prepare_and_send_response(reservation, None, None), .5))
        self.assertEqual(len(self.creates()), 1)
        self.assertTrue(reservation.submitted)
        self.assertIn("Current Live State briefing is temporarily unavailable", self.creates()[0]["response"]["instructions"])
        self.assertFalse(self.runtime.coordinator._bridge.status.fenced)
        self.assertFalse(self.session._closed.is_set())

    async def test_cancelled_waiter_cannot_deliver_after_fresh_projection(self):
        await self.invalidate_pending()
        reservation = self.followup()
        entered, release = self.hold_normal_scope()
        preparing = asyncio.create_task(self.session._prepare_and_send_response(reservation, None, None))
        self.addAsyncCleanup(self.join, preparing)
        await asyncio.wait_for(entered.wait(), 2)
        preparing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await preparing
        release.set()
        await self.observation("current projection after cancellation")
        await self.until(lambda: self.runtime.ready)
        self.assertEqual(self.creates(), [])
        self.assertIsNone(self.session._pending_response)

    async def join(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
