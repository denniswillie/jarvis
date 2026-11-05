"""Pre-publication diagnostics over the real attachment, no external calls."""
import asyncio
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.models import GuardedPayloadBinding
from app.realtime.live_state_bridge import SessionLiveStateBridge, _PREPARATION_LOG_LIMIT
from app.voice_routes.bridge import ContextOfferResult, OfferDisposition
import test_live_state_voice_attachment as fixture


class PreparationDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = fixture.NOW
        self.current = fixture.snapshot(text="PRIVATE_FACT_NOT_FOR_LOGS")
        self.live = SimpleNamespace(requires_reconstruction=False,
            service=SimpleNamespace(snapshot=lambda: self.current),
            bind=AsyncMock(return_value=fixture.BINDING))
        self.session = SimpleNamespace(_closed=asyncio.Event(), close=AsyncMock())
        self.route = SimpleNamespace(ready=True, request=fixture.request(),
            offer_context=Mock(return_value=ContextOfferResult(OfferDisposition.QUEUED)))
        self.bridge = SessionLiveStateBridge(self.session, self.live, self.route, clock=lambda: self.now)
        logging = patch("app.realtime.live_state_bridge._PREPARATION_LOGGER.warning")
        self.log = logging.start()
        self.addCleanup(logging.stop)

    def rows(self):
        return [json.loads(call.args[1]) for call in self.log.call_args_list]

    def assert_private(self):
        serialized = json.dumps(self.rows())
        for private in ("PRIVATE_FACT_NOT_FOR_LOGS", "PRIVATE_ERROR", "PRIVATE_REASON", fixture.SCOPE.owner_id):
            self.assertNotIn(private, serialized)

    async def test_success_correlates_exact_binding_and_offer_without_claiming_write_ack(self):
        await self.bridge.publish_latest()
        update = self.route.offer_context.call_args.args[0]
        entry, offered = self.rows()
        self.assertEqual((entry["stage"], offered["stage"]), ("binding_entry", "offer_return"))
        self.assertEqual(offered["outcome"], "queued")
        self.assertEqual(offered["update_id"], str(update.update_id))
        self.assertEqual(offered["bound_payload_root_ids"], [str(fixture.ROOT)])
        self.assertEqual(offered["parent_payload_root_ids"], [str(fixture.ROOT)])
        self.assertTrue(offered["binding_returned"])
        self.assertTrue(all(not changed for changed in offered["key_changes"].values()))
        self.assertGreaterEqual(offered["elapsed_ms"], 0)
        self.assertNotIn("provider_acknowledged", json.dumps(self.rows()))
        self.assertIs(self.bridge._last_update, update)
        self.assert_private()

    async def test_changed_text_is_omitted_with_exact_safe_difference(self):
        async def bind(selection):
            self.current = fixture.snapshot(2, text="PRIVATE_FACT_NOT_FOR_LOGS changed")
            return fixture.BINDING
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        last = self.rows()[-1]
        self.assertEqual(last["code"], "selection_changed_during_binding")
        self.assertEqual(last["latest_cursor_revision"], 2)
        self.assertEqual(last["key_changes"], dict(text=True, dependencies=False,
            parent_roots=False, entry_ids=False, observation_history=False))
        self.assertTrue(self.bridge._wake.is_set())
        self.route.offer_context.assert_not_called()
        self.assert_private()

    async def test_cursor_only_update_keeps_existing_admission_behavior(self):
        async def bind(selection):
            self.current = fixture.snapshot(2, text="PRIVATE_FACT_NOT_FOR_LOGS")
            return fixture.BINDING
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        self.route.offer_context.assert_called_once()
        self.assertEqual(self.rows()[-1]["latest_cursor_revision"], 2)
        self.assertTrue(all(not changed for changed in self.rows()[-1]["key_changes"].values()))

    async def test_changed_roots_are_not_confused_with_changed_prose(self):
        async def bind(selection):
            binding = GuardedPayloadBinding(scope=fixture.SCOPE,
                payload_root_ids=(uuid4(),), dependencies=(fixture.SOURCE,))
            self.current = self.current.model_copy(update={
                "entries": (self.current.entries[0].model_copy(update={"guard_binding": binding}),),
                "cursor": self.current.cursor.model_copy(update={"revision": 2})})
            return fixture.BINDING
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        self.assertEqual(self.rows()[-1]["key_changes"], dict(text=False, dependencies=False,
            parent_roots=True, entry_ids=False, observation_history=False))
        self.route.offer_context.assert_not_called()

    async def test_expiry_after_binding_records_no_offer_or_freshness_extension(self):
        async def bind(selection):
            self.now += timedelta(seconds=11)
            return fixture.BINDING
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        last = self.rows()[-1]
        self.assertEqual((last["stage"], last["code"]), ("preparation_exit", "preparation_expired"))
        self.assertTrue(last["binding_returned"])
        self.assertEqual(last["bound_payload_root_ids"], [str(fixture.ROOT)])
        self.route.offer_context.assert_not_called()
        self.assertIsNone(self.bridge._last_update)

    async def test_known_source_error_preserves_rejection_and_redacts_message(self):
        self.live.bind.side_effect = SourceError("live_state_runtime_unavailable", "PRIVATE_ERROR")
        await self.bridge.publish_latest()
        self.assertEqual(self.rows()[-1]["code"], "live_state_runtime_unavailable")
        self.assertEqual(self.bridge.last_omission, "live_state_runtime_unavailable")
        self.route.offer_context.assert_not_called()
        self.assert_private()

    async def test_unknown_source_code_is_redacted_not_reinterpreted(self):
        self.live.bind.side_effect = SourceError("PRIVATE_REASON", "PRIVATE_ERROR")
        await self.bridge.publish_latest()
        self.assertEqual(self.rows()[-1]["code"], "unrecognized_reason")
        self.assertEqual(self.bridge.last_omission, "PRIVATE_REASON")
        self.assert_private()

    async def test_timeout_is_distinguished_from_source_rejection(self):
        self.live.bind.side_effect = TimeoutError("PRIVATE_ERROR")
        await self.bridge.publish_latest()
        self.assertEqual(self.rows()[-1]["code"], "binding_timeout")
        self.route.offer_context.assert_not_called()
        self.assert_private()

    async def test_cancellation_still_propagates_without_offer(self):
        self.live.bind.side_effect = asyncio.CancelledError("PRIVATE_ERROR")
        with self.assertRaises(asyncio.CancelledError):
            await self.bridge.publish_latest()
        self.assertEqual(self.rows()[-1]["code"], "binding_cancelled")
        self.route.offer_context.assert_not_called()
        self.assertIsNone(self.bridge.last_omission)
        self.assert_private()

    async def test_unexpected_error_still_propagates_without_private_message(self):
        self.live.bind.side_effect = RuntimeError("PRIVATE_ERROR")
        with self.assertRaisesRegex(RuntimeError, "PRIVATE_ERROR"):
            await self.bridge.publish_latest()
        self.assertEqual(self.rows()[-1]["code"], "binding_unexpected_error")
        self.route.offer_context.assert_not_called()
        self.assert_private()

    async def test_logging_failure_does_not_change_success_or_rejection(self):
        self.log.side_effect = RuntimeError("logger failed")
        await self.bridge.publish_latest()
        self.route.offer_context.assert_called_once()
        self.bridge._last_update = self.bridge._last_key = None
        self.live.bind.side_effect = SourceError("live_state_context_revoked", "PRIVATE_ERROR")
        await self.bridge.publish_latest()
        self.assertEqual(self.route.offer_context.call_count, 1)
        self.assertEqual(self.bridge.last_omission, "live_state_context_revoked")

    async def test_record_cap_has_one_marker_and_never_blocks_offering(self):
        attempts = _PREPARATION_LOG_LIMIT + 3
        for _ in range(attempts):
            self.bridge._last_update = self.bridge._last_key = None
            await self.bridge.publish_latest()
        self.assertEqual(self.route.offer_context.call_count, attempts)
        self.assertEqual(len(self.rows()), _PREPARATION_LOG_LIMIT + 1)
        self.assertEqual(self.rows()[-1], {"stage": "truncated", "record_limit": _PREPARATION_LOG_LIMIT,
                                        "absence_after_limit_is_inconclusive": True})
        self.assert_private()


if __name__ == "__main__":
    unittest.main()
