from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.memory.prospective.models import VisualFocus
from app.memory.prospective.tools import create_prospective_tools


class _Service:
    def __init__(self) -> None:
        self.focus = _focus()
        self.set_arguments = None
        self.clear_arguments = None

    async def set_visual_focus(self, **kwargs):
        self.set_arguments = kwargs
        return self.focus

    async def get_visual_focus(self, **kwargs):
        return self.focus

    async def clear_visual_focus(self, **kwargs):
        self.clear_arguments = kwargs
        if self.focus is None or self.focus.id != kwargs["focus_id"]:
            return None
        focus, self.focus = self.focus, None
        return focus


class VisualFocusToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_set_uses_trusted_owner_and_normalized_contract(self) -> None:
        service = _Service()
        tools = {tool.name: tool for tool in create_prospective_tools(service=service, owner_id="owner")}
        result = await tools["set_visual_focus"].handler(
            {
                "target": " keys ",
                "description": " silver ring ",
                "purpose": " find them ",
                "mode": "find_once",
            }
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(service.set_arguments["owner_id"], "owner")
        self.assertEqual(service.set_arguments["target"], "keys")

    async def test_clear_is_idempotent(self) -> None:
        service = _Service()
        tools = {tool.name: tool for tool in create_prospective_tools(service=service, owner_id="owner")}
        focus_id = service.focus.id
        arguments = {"focus_id": str(focus_id)}
        self.assertTrue((await tools["clear_visual_focus"].handler(arguments))["cleared"])
        self.assertFalse((await tools["clear_visual_focus"].handler(arguments))["cleared"])
        self.assertEqual(service.clear_arguments, {"owner_id": "owner", "focus_id": focus_id})


def _focus() -> VisualFocus:
    now = datetime.now(UTC)
    return VisualFocus(
        id=uuid4(), owner_id="owner", target="keys", description="silver",
        purpose="find", mode="find_once", created_from_message_id=None,
        created_at=now, expires_at=now + timedelta(minutes=15), completed_at=None,
        cleared_at=None, completion_evidence=None, trigger_claim_token=None,
        trigger_claimed_at=None,
    )
