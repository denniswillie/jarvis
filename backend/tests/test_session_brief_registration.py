"""Real pipeline/attachment lifecycle; no provider, database or device access."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.models import SessionScope
from app.perception.live_brief import LivePerceptionBriefBridge
from app.realtime.live_state_bridge import SessionLiveStateBridge
from test_fast_perception_pipeline import _Harness


class SessionBriefRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.harness = _Harness(static=True)
        self.pipeline = self.harness.pipeline
        self.scope = SessionScope(owner_id=self.harness.owner_id, session_id=self.harness.session_id)
        self.bridge = self.make_bridge()
        self.publisher = object()
        self.pipeline.set_observation_publication(self.publisher)
        self.attachments = []

    def make_bridge(self, scope=None):
        return LivePerceptionBriefBridge(scope=scope or self.scope, scope_epoch=uuid4(), enabled=True)

    def lookup(self, scope=None):
        scope = scope or self.scope
        return self.pipeline.live_brief_for(scope.owner_id, scope.session_id)

    def attachment(self, bridge=None):
        bridge = bridge or self.bridge
        session = SimpleNamespace(close=AsyncMock())

        async def close_live():
            bridge.close()

        live = SimpleNamespace(perception_brief_bridge=bridge, close=AsyncMock(side_effect=close_live))
        route = SimpleNamespace(request=SimpleNamespace(route=SimpleNamespace(scope=self.scope)))
        attachment = SessionLiveStateBridge(session, live, route, visual_pipeline=self.pipeline)
        self.attachments.append(attachment)
        return attachment

    async def asyncTearDown(self):
        for attachment in self.attachments:
            await attachment.close()
        await self.pipeline.close()

    async def test_disabled_capture_rejects_registration_without_poisoning_pipeline(self):
        self.pipeline.set_observation_publication(None)
        self.assertFalse(self.pipeline.supports_perception_briefs)
        with self.assertRaises(RuntimeError):
            self.pipeline.register_live_brief(self.bridge)
        self.assertIsNone(self.lookup())

    async def test_scope_exact_lookup_and_identity_safe_unregister(self):
        self.pipeline.register_live_brief(self.bridge)
        self.pipeline.register_live_brief(self.bridge)
        self.assertIs(self.lookup(), self.bridge)
        foreign = SessionScope(owner_id="other", session_id=self.scope.session_id)
        self.assertIsNone(self.lookup(foreign))
        self.assertIsNone(self.lookup(SessionScope(owner_id=self.scope.owner_id, session_id=uuid4())))
        replacement = self.make_bridge()
        with self.assertRaises(RuntimeError):
            self.pipeline.register_live_brief(replacement)
        self.pipeline.unregister_live_brief(replacement)
        self.assertIs(self.lookup(), self.bridge)
        self.pipeline.unregister_live_brief(self.bridge)
        self.pipeline.register_live_brief(replacement)
        self.pipeline.unregister_live_brief(self.bridge)
        self.assertIs(self.lookup(), replacement)

    async def test_capture_owner_cannot_be_changed_beneath_registered_session(self):
        self.pipeline.register_live_brief(self.bridge)
        self.pipeline.set_observation_publication(self.publisher)
        with self.assertRaises(RuntimeError):
            self.pipeline.set_observation_publication(None)
        self.assertTrue(self.pipeline.supports_perception_briefs)
        self.assertIs(self.lookup(), self.bridge)

    async def test_pipeline_shutdown_forgets_but_does_not_close_borrowed_bridge(self):
        self.pipeline.register_live_brief(self.bridge)
        await self.pipeline.close()
        self.assertIsNone(self.lookup())
        self.assertFalse(self.bridge._closed)
        with self.assertRaises(RuntimeError):
            self.pipeline.register_live_brief(self.bridge)

    async def test_attachment_activates_only_on_registered_and_joins_owner_on_close(self):
        attachment = self.attachment()
        self.assertIsNone(self.lookup())
        attachment.registered()
        attachment.registered()
        self.assertIs(self.lookup(), self.bridge)
        await attachment.close("finished")
        self.assertIsNone(self.lookup())
        self.assertTrue(self.bridge._closed)
        attachment.live.close.assert_awaited_once()
        attachment.session.close.assert_awaited_once_with("finished")
        with self.assertRaises(RuntimeError):
            attachment.registered()

    async def test_rejected_attachment_cleanup_cannot_remove_existing_bridge(self):
        self.pipeline.register_live_brief(self.bridge)
        newcomer = self.attachment(self.make_bridge())
        with self.assertRaises(RuntimeError):
            newcomer.registered()
        await newcomer.close("rejected")
        self.assertIs(self.lookup(), self.bridge)
        self.assertFalse(self.bridge._closed)

    async def test_scope_mismatch_does_not_register(self):
        wrong = self.make_bridge(SessionScope(owner_id="other", session_id=self.scope.session_id))
        attachment = self.attachment(wrong)
        with self.assertRaises(RuntimeError):
            attachment.registered()
        self.assertIsNone(self.lookup())

    async def test_unregister_happens_before_slow_session_shutdown(self):
        attachment = self.attachment()
        attachment.registered()
        entered, release = asyncio.Event(), asyncio.Event()

        async def close_session(reason):
            entered.set()
            await release.wait()

        attachment.session.close.side_effect = close_session
        closing = asyncio.create_task(attachment.close())
        try:
            async with asyncio.timeout(1):
                await entered.wait()
            self.assertIsNone(self.lookup())
        finally:
            release.set()
            await closing
