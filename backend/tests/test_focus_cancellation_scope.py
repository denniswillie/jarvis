"""A stale stop cannot erase a replacement or revoke its pending announcement."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.memory.prospective.delivery import ProspectiveDeliveryAttempt
from app.memory.prospective.service import ProspectiveMemoryService
from app.memory.prospective.tools import create_prospective_tools


class FocusCancellationScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_stop_requires_an_explicit_valid_identity(self):
        service = SimpleNamespace(clear_visual_focus=AsyncMock(return_value=None))
        tool = next(tool for tool in create_prospective_tools(service=service, owner_id="owner")
                    if tool.name == "clear_visual_focus")
        self.assertEqual(tool.input_schema["required"], ["focus_id"])
        for arguments in ({}, {"focus_id": None}, {"focus_id": ""}, {"focus_id": "the old box"}, {"focus_id": True}):
            with self.subTest(arguments=arguments):
                result = await tool.handler(arguments)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        service.clear_visual_focus.assert_not_awaited()

    async def test_tool_uses_trusted_owner_and_reports_the_requested_id(self):
        service = SimpleNamespace(clear_visual_focus=AsyncMock(return_value=None))
        tool = next(tool for tool in create_prospective_tools(service=service, owner_id="owner")
                    if tool.name == "clear_visual_focus")
        focus_id = uuid4()
        result = await tool.handler({"focus_id": str(focus_id), "owner_id": "foreign"})
        service.clear_visual_focus.assert_awaited_once_with(owner_id="owner", focus_id=focus_id)
        self.assertEqual(result, {"status": "success", "cleared": False,
                                  "focus_id": str(focus_id), "visual_focus": None})

    async def test_inactive_or_expired_id_revokes_only_that_focus_pending_audio(self):
        repository = SimpleNamespace(clear_active_visual_focus=AsyncMock(return_value=None))
        service = ProspectiveMemoryService(repository=repository,
            realtime_session_manager=object(), settings=object())

        def delivery(*, owner="owner", subject_type="visual_focus", subject_id=None):
            result = ProspectiveDeliveryAttempt(owner_id=owner, event_id=uuid4(),
                subject_type=subject_type, subject_id=subject_id or uuid4(),
                preflight=AsyncMock(return_value=True), persist=AsyncMock())
            result.on_revoke = AsyncMock()
            return result

        old = delivery()
        replacement = delivery()
        foreign = delivery(owner="foreign", subject_id=old.subject_id)
        reminder = delivery(subject_type="conditional_reminder", subject_id=old.subject_id)
        service._pending_deliveries = {item.event_id: item for item in (old, replacement, foreign, reminder)}
        self.assertIsNone(await service.clear_visual_focus(owner_id="owner", focus_id=old.subject_id))
        repository.clear_active_visual_focus.assert_awaited_once_with(
            owner_id="owner", expected_focus_id=old.subject_id)
        self.assertTrue(old.revoked)
        old.on_revoke.assert_awaited_once()
        for item in (replacement, foreign, reminder):
            self.assertFalse(item.revoked)
            item.on_revoke.assert_not_awaited()

    async def test_repository_failure_does_not_revoke_pending_work(self):
        repository = SimpleNamespace(clear_active_visual_focus=AsyncMock(side_effect=RuntimeError("fixture failure")))
        service = ProspectiveMemoryService(repository=repository,
            realtime_session_manager=object(), settings=object())
        delivery = ProspectiveDeliveryAttempt(owner_id="owner", event_id=uuid4(),
            subject_type="visual_focus", subject_id=uuid4(),
            preflight=AsyncMock(return_value=True), persist=AsyncMock())
        service._pending_deliveries[delivery.event_id] = delivery
        with self.assertRaises(RuntimeError):
            await service.clear_visual_focus(owner_id="owner", focus_id=delivery.subject_id)
        self.assertFalse(delivery.revoked)
