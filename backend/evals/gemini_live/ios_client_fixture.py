"""Loopback-only iOS wire fixtures. No models, database, credentials, or domain jobs.

Run in a disposable container published only on 127.0.0.1. Supply its WebSocket
URL to the native test scheme as JARVIS_CLIENT_FIXTURE_URL.
"""
from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


async def handle(socket):
    request = socket.request
    target = urlsplit(request.path)
    query = parse_qs(target.query)

    async def send(kind, **values):
        await socket.send(json.dumps({"type": kind, **values}))

    if request.headers.get("Authorization") != "Bearer fixture-token":
        await send("error", code="fixture_auth_failed", message="Fixture bearer not supplied.", recoverable=False)
        return
    providers = query.get("audio_provider", [])
    if len(providers) != 1 or providers[0] not in {"openai", "gemini"}:
        await send("error", code="fixture_selection_failed", message="Explicit provider required.", recoverable=False)
        return
    provider = providers[0]
    if target.path == "/disabled":
        await send("error", code="audio_provider_unavailable", message="Gemini Live audio is disabled.", recoverable=False)
        return
    audio_id = query.get("audio_session_id", ["fixture-audio"])[0]
    if target.path != "/legacy":
        await send("audio.provider.selected", provider=provider, protocol_version=1,
                   model="fixture-model", audio_session_id=audio_id)
    await send("session.created", session_id=query.get("session_id", ["fixture-session"])[0],
               audio_session_id=audio_id,
               **({"playback_receipts":{"protocol_version":1,"sample_rate":24000}} if target.path == "/playback" else {}))
    if target.path == "/playback":
        await send("audio.output.delta", response_id="fixture-response", item_id="fixture-item",
                   audio_base64=base64.b64encode(bytes(4800)).decode("ascii"))
        await send("audio.output.done", response_id="fixture-response", item_id="fixture-item")
    events = []
    try:
        async for raw in socket:
            event = json.loads(raw)
            events.append(event)
            if len(events) > 16:
                return
            if event.get("type") == "session.close":
                await send("fixture.received", events=events, authorization_verified=True)
                return
    except ConnectionClosed:
        pass


async def run():
    async with serve(handle, "0.0.0.0", 8765, max_size=65536):
        print("Isolated iOS WebSocket fixture ready on port 8765", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(run())
