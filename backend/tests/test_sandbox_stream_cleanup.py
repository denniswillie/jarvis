"""A disconnected viewer must not raise again while its stream is closing."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from starlette.websockets import WebSocketDisconnect

from app.sandbox.view_router import stream


def test_already_disconnected_socket_closes_without_escaping():
    websocket = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(runtime_services=SimpleNamespace(
            sandbox_views=SimpleNamespace(leases={}),
        ))),
        scope={"subprotocols": []},
        close=AsyncMock(side_effect=WebSocketDisconnect(code=1006)),
    )
    asyncio.run(stream(websocket, uuid4()))
    assert websocket.close.await_count == 2
