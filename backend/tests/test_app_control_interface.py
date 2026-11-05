"""Thin integration surface for callers sharing the app-control dispatcher."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from app.app_controls import AppControlService


def controls():
    return AppControlService(
        owner_id="owner",
        pool=Mock(),
        sandbox_views=Mock(),
        presentations=Mock(),
    )


@pytest.mark.asyncio
async def test_invoke_preserves_callers_operation_and_uses_existing_dispatcher():
    service = controls()
    result = {"outcome": "admitted", "effect_committed": True}
    service.dispatch = AsyncMock(return_value=result)
    session_id, operation_id = uuid4(), uuid4()

    assert await service.invoke(
        "owner", session_id, "recording.start",
        operation_id=operation_id, include_jarvis_audio=False,
    ) is result
    service.dispatch.assert_awaited_once_with(
        owner_id="owner", session_id=session_id, operation_id=operation_id,
        command={"action": "recording.start", "include_jarvis_audio": False},
    )


@pytest.mark.asyncio
async def test_invoke_allocates_command_identity_when_caller_has_none():
    service = controls()
    service.dispatch = AsyncMock(return_value={"outcome": "view_not_active"})
    session_id, view_id = uuid4(), uuid4()

    await service.invoke("owner", session_id, "sandbox.stop", view_id=str(view_id))

    arguments = service.dispatch.await_args.kwargs
    assert isinstance(arguments["operation_id"], UUID)
    assert arguments["command"] == {"action": "sandbox.stop", "view_id": str(view_id)}


def test_projection_uses_current_phone_session_and_no_external_io():
    service = controls()
    session_id = uuid4()
    service.bind(SimpleNamespace(
        context=SimpleNamespace(owner_id="owner", jarvis_session_id=session_id),
        is_current=lambda: True, is_ready=lambda: True,
        snapshot=lambda: {"recording": {"current_id": "recording-1"}},
    ))
    service.views.cached_inventory.return_value = {"items": []}
    service.views.current.return_value = None

    result = service.projection("owner")
    assert result["phone_connected"] is True
    assert result["recording"] == {"current_id": "recording-1"}
    assert result["sandboxes"] == {"items": []}
    assert not service.pool.mock_calls
    service.views.choices.assert_not_called()


def test_inventory_warms_without_gallery_and_stops_with_phone_binding():
    async def scenario():
        service = controls()
        session_id = uuid4()
        refreshed = asyncio.Event()
        async def refresh(*args, **kwargs):
            service.views.cached_inventory.return_value = {"items": [{"id": "sandbox-1"}]}
            refreshed.set()
            return {"items": []}
        service.views.choices = AsyncMock(side_effect=refresh)
        application = SimpleNamespace(
            context=SimpleNamespace(owner_id="owner", jarvis_session_id=session_id),
            is_current=lambda: True, is_ready=lambda: True, snapshot=lambda: {},
        )

        service.bind(application)
        task = service._inventory_task
        service.views.choices.assert_not_awaited()
        await asyncio.wait_for(refreshed.wait(), timeout=1)
        service.views.choices.assert_awaited_once_with("owner", session_id, limit=32)
        assert service.projection("owner")["sandboxes"]["items"][0]["id"] == "sandbox-1"
        service.views.choices.assert_awaited_once()
        assert not service.pool.mock_calls

        service.unbind(application)
        await asyncio.gather(task, return_exceptions=True)
        assert task.done()
        assert service._inventory_task is None
        assert service.projection("owner")["phone_connected"] is False
    asyncio.run(scenario())


def test_projection_without_phone_is_still_available():
    service = controls()
    service.snapshot = Mock(return_value={"phone_connected": False})

    assert service.projection("owner") == {"phone_connected": False}
    service.snapshot.assert_called_once_with(owner_id="owner", session_id=None)
    assert not service.pool.mock_calls


def test_definition_exposes_control_app_without_owner_or_session_arguments():
    descriptor = AppControlService.definition()
    assert descriptor["name"] == "control_app"
    properties = descriptor["parameters"]["properties"]
    assert "action" in properties
    assert "owner_id" not in properties
    assert "session_id" not in properties
