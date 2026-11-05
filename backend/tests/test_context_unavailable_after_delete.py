"""9f6948 local-unavailability race through the actual session/publication path.

External provider and canonical storage are controlled fixtures; no SQL, RTP,
audio, model quality or native acceptance is implied by this composition.
"""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.runtime import LiveStateRuntime
from app.voice_routes.contracts import ContextWriteEvidence
from app.voice_routes.coordinator import CoordinationPhase
import test_context_confirmed_delete_expiry as delete_fixtures
import test_voice_context_response_pipeline as pipeline


class ContextUnavailablePipelineTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = pipeline.VoiceContextResponsePipelineTests.asyncSetUp
    cleanup = pipeline.VoiceContextResponsePipelineTests.cleanup
    until = pipeline.VoiceContextResponsePipelineTests.until
    rows = pipeline.VoiceContextResponsePipelineTests.rows
    observation = pipeline.VoiceContextResponsePipelineTests.observation
    expire_inside_adapter_guard = pipeline.VoiceContextResponsePipelineTests.expire_inside_adapter_guard
    mutations = pipeline.VoiceContextResponsePipelineTests.mutations
    creates = pipeline.VoiceContextResponsePipelineTests.creates
    question_and_accept = pipeline.VoiceContextResponsePipelineTests.question_and_accept

    async def lose_live_state_after_exact_delete(self):
        self.storage.on_lock = None
        # Actual health and canonical guard implementation, without its worker
        # processes. The health transition is a controlled refresh failure.
        health = object.__new__(LiveStateRuntime)
        health._started, health._closing, health._healthy = True, False, True
        # Production uses _SessionBinder, whose lifecycle field is separate
        # from the reused generic canonical guard's scope/source validation.
        self.binder.scope_changed = False
        health.scope = self.scope
        health.service, health.binder, health.failure = self.service, self.binder, None
        self.live.guard = health.guard
        adapter = self.runtime.adapter
        old_id = adapter._context_item
        self.assertIsNotNone(old_id)
        old_item = adapter._items[old_id]
        before = len(self.mutations())
        original = self.provider.inject

        async def inject(kind, **fields):
            if kind == "conversation.item.deleted" and fields.get("item_id") == old_id:
                self.assertEqual(old_item.state, "delete_pending")
                health._healthy = False
            await original(kind, **fields)

        self.provider.inject = inject
        self.addCleanup(setattr, self.provider, "inject", original)
        await self.observation("candidate-not-authorized-for-create")
        await self.until(lambda: bool(self.rows(self.readiness, "source_guard_rejected")))
        rejected = self.rows(self.readiness, "source_guard_rejected")[-1]
        await self.until(lambda: any(row.get("update_id") == rejected["update_id"]
            and row.get("owned_work_joined") is True for row in self.rows(self.publication, "adapter_return"))
            and (self.runtime.coordinator._bridge.status.fenced or any(
                row.get("update_id") == rejected["update_id"]
                for row in self.rows(self.publication, "context_refresh_wait"))))
        returned = next(row for row in self.rows(self.publication, "adapter_return")
                        if row.get("update_id") == rejected["update_id"])
        self.assertEqual(rejected["error_code"], "live_state_runtime_unavailable")
        self.assertFalse(rejected["guard_yielded"])
        self.assertEqual(old_item.state, "deleted")
        self.assertIsNone(adapter._context_item)
        self.assertNotIn(old_id, adapter._items)
        self.assertEqual([e["type"] for e in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertEqual(returned["write_state"], "write_possible")
        self.assertTrue(returned["deleted_context_acknowledged"])
        self.assertFalse(returned["replacement_create_possible"])
        self.assertTrue(returned["owned_work_joined"])
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertIsNone(self.runtime.coordinator._bridge.status.retry_update_id)
        self.assertEqual(self.rows(self.publication, "context_refresh_wait")[-1]["outcome"],
                         "unavailable_after_acknowledged_delete")
        self.assertFalse(await self.runtime.wait_for_context_ready(.01))
        self.assertFalse(self.creates())
        return health, before, old_id

    async def recover_and_answer(self):
        health, before, old_id = await self.lose_live_state_after_exact_delete()
        ask = asyncio.create_task(self.question_and_accept())
        try:
            pin = await asyncio.wait_for(ask, 3)
            self.assertFalse(self.runtime.ready)
            self.assertIsNone(self.runtime.adapter._context_item)
            self.assertEqual([e["type"] for e in self.mutations()[before:]], ["conversation.item.delete"])
            self.assertIn("Current Live State briefing is temporarily unavailable",
                          self.creates()[0]["response"]["instructions"])
            await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
                                       item_id="answer-item", transcript="Controlled recovered answer.")
            await self.provider.inject("response.done", response={"id": "answer", "status": "completed",
                "output": [{"id": "answer-item", "type": "message", "phase": "final_answer",
                            "content": [{"type": "output_audio"}]}]})
            await self.until(lambda: any(e["type"] == "response.done" for e in self.app.events))
            await self.provider.inject("output_audio_buffer.stopped", response_id="answer")
            await self.until(lambda: "answer" not in self.session._direct_output_pending)
            health._healthy = True
            await self.observation("fresh-independently-guarded-context")
            await self.until(lambda: self.runtime.ready and
                "fresh-independently-guarded-context" in self.runtime.adapter._context.text)
            self.assertEqual([e["type"] for e in self.mutations()[before:]],
                             ["conversation.item.delete", "conversation.item.create"])
            self.assertNotEqual(self.runtime.adapter._context_item, old_id)
            self.assertNotIn("candidate-not-authorized-for-create", self.runtime.adapter._context.text)
            self.assertEqual(len(self.creates()), 1)
            self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
            self.assertFalse(self.session._context_revoked)
            assistant = [c.kwargs for c in self.repository.create_message.await_args_list
                         if c.kwargs["role"] == "assistant"]
            self.assertEqual(len(assistant), 1)
            self.assertEqual(assistant[0]["source_payload_id"], pin.payload_id)
        finally:
            ask.cancel()
            await asyncio.gather(ask, return_exceptions=True)

    async def test_temporary_unavailability_recovers_waiting_same_turn_without_redelete(self):
        await self.recover_and_answer()

    async def test_same_recovery_does_not_depend_on_latency_logging(self):
        with patch("app.realtime.telemetry.RealtimeTrace.latency_event", return_value=None):
            await self.recover_and_answer()

    async def test_closing_live_state_cannot_resume_even_when_health_bit_recovers(self):
        health, before, _ = await self.lose_live_state_after_exact_delete()
        health._healthy, health._closing = True, True
        await self.observation("must-not-publish-through-closing-runtime")
        await self.until(lambda: self.runtime.coordinator.status.phase == CoordinationPhase.DEGRADED
                         or self.session._closed.is_set())
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.creates())
        self.assertEqual([e["type"] for e in self.mutations()[before:]], ["conversation.item.delete"])

    async def test_original_source_withdrawal_during_hold_still_retires_connection(self):
        await self.lose_live_state_after_exact_delete()
        root = next(iter(self.session._source_context._sources))
        self.storage.rejected.add(root)
        await self.until(lambda: self.session._context_revoked)
        await self.until(lambda: self.session._closed.is_set())
        self.assertEqual(self.session._close_reason, "live_context_rebuild_required")
        self.assertFalse(self.creates())


class ContextUnavailableProofTests(unittest.IsolatedAsyncioTestCase):
    voice = delete_fixtures.ConfirmedDeleteExpiryTests.voice
    route = delete_fixtures.ConfirmedDeleteExpiryTests.route
    until = delete_fixtures.ConfirmedDeleteExpiryTests.until
    setup_route = delete_fixtures.ConfirmedDeleteExpiryTests.setup_route
    rows = delete_fixtures.ConfirmedDeleteExpiryTests.rows
    writes = delete_fixtures.ConfirmedDeleteExpiryTests.writes
    settle = delete_fixtures.ConfirmedDeleteExpiryTests.settle

    async def test_possible_create_with_local_unavailability_remains_unknown(self):
        await self.setup_route(failure="after_create")
        original = self.control._write

        async def write(event):
            result = await original(event)
            if self.control.writes[-1]["type"] == "conversation.item.create":
                raise SourceError("live_state_runtime_unavailable", "private after-write error")
            return result

        self.control._write = write
        await self.settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        self.assertTrue(self.rows("adapter_return")[-1]["replacement_create_possible"])
        self.assertEqual(self.rows("context_refresh_wait"), [])

    async def test_no_join_nonoptin_or_actual_fence_cannot_recover(self):
        for failure in ("unjoined", "nonoptin", "route_fenced", "withdrawal"):
            with self.subTest(failure=failure):
                await self.setup_route(failure="after_create", enabled=failure != "nonoptin")
                original = self.runtime._atomic_guard

                @asynccontextmanager
                async def guard(update):
                    async with original(update) as connection:
                        if update.cursor.revision == 1 and self.adapter._context_item is None:
                            if failure == "route_fenced":
                                self.adapter._source_epoch += 1
                            code = "source_withdrawn" if failure == "withdrawal" else "live_state_runtime_unavailable"
                            raise SourceError(code, "private source failure")
                        yield connection

                self.runtime._atomic_guard = guard
                if failure == "unjoined":
                    with patch.object(ContextWriteEvidence, "mark_joined", lambda _: None):
                        await self.settle()
                else:
                    await self.settle()
                self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
                self.assertEqual(self.rows("context_refresh_wait"), [])
                self.assertEqual(len(self.writes("conversation.item.create")), 1)
                await self.owner.close()


class LiveStateAvailabilityDiagnosticTests(unittest.TestCase):
    def runtime(self):
        runtime = object.__new__(LiveStateRuntime)
        runtime.scope = SimpleNamespace(session_id=uuid4(), owner_id="PRIVATE-owner")
        runtime._started, runtime._closing, runtime._healthy = True, False, True
        runtime.service = SimpleNamespace(needs_resync=False)
        runtime.binder = SimpleNamespace(scope_changed=False)
        runtime.failure = None
        return runtime

    def test_static_causes_preserve_rejection_without_content(self):
        for reason in ("not_started", "closing", "scope_changed", "resync_required", "worker_failed", "refresh_not_ready"):
            with self.subTest(reason=reason):
                runtime = self.runtime()
                if reason == "not_started": runtime._started = False
                if reason == "closing": runtime._closing = True
                if reason == "scope_changed": runtime.binder.scope_changed = True
                if reason == "resync_required": runtime.service.needs_resync = True
                if reason == "worker_failed": runtime.failure = RuntimeError("PRIVATE-worker-details")
                if reason == "refresh_not_ready": runtime._healthy = False
                with patch("app.live_state.runtime._AVAILABILITY_LOGGER.warning") as log:
                    with self.assertRaises(SourceError) as caught:
                        runtime._require_usable()
                self.assertEqual(caught.exception.code, "live_state_runtime_unavailable")
                row = json.loads(log.call_args.args[1])
                self.assertEqual(row["reason"], reason)
                self.assertEqual(row["session_id"], str(runtime.scope.session_id))
                self.assertTrue(set(row) <= {"stage", "reason", "session_id", "started", "closing", "healthy",
                                            "needs_resync", "scope_changed", "worker_failed"})
                self.assertNotIn("PRIVATE", str(log.call_args_list))

    def test_healthy_runtime_does_not_emit_rejection(self):
        with patch("app.live_state.runtime._AVAILABILITY_LOGGER.warning") as log:
            self.assertIsNone(self.runtime()._require_usable())
        log.assert_not_called()

    def test_logging_failure_cannot_change_source_rejection(self):
        runtime = self.runtime()
        runtime._healthy = False
        with patch("app.live_state.runtime._AVAILABILITY_LOGGER.warning", side_effect=RuntimeError("PRIVATE-log")):
            with self.assertRaises(SourceError) as caught:
                runtime._require_usable()
        self.assertEqual(caught.exception.code, "live_state_runtime_unavailable")

    def test_eight_records_then_single_truncation_marker(self):
        runtime = self.runtime()
        runtime._healthy = False
        with patch("app.live_state.runtime._AVAILABILITY_LOGGER.warning") as log:
            for _ in range(25):
                with self.assertRaises(SourceError):
                    runtime._require_usable()
        rows = [json.loads(call.args[1]) for call in log.call_args_list]
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[-1]["stage"], "truncated")
        self.assertTrue(rows[-1]["absence_after_limit_is_inconclusive"])


if __name__ == "__main__":
    unittest.main()
