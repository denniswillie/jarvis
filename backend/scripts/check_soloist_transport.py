"""Bounded local real-source check. No public server, phone, or raw music file.

PYTHONPATH=backend <test-python> backend/scripts/check_soloist_transport.py
Uses the retained private Soloist pairing; pauses and closes owned processes.
HTTP authentication is replaced only inside this isolated TestClient app.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.media_audio import SoloistMediaService, router, validate_packet


root = Path(__file__).resolve().parents[2]
home = root / ".jarvis-dev/soloist"
session_id, media_id = uuid4(), uuid4()
receipt = {"test": "local_real_soloist_through_media_websocket", "passed": False,
           "auth": "isolated_test_override", "phone_or_glasses_tested": False}


@asynccontextmanager
async def lifespan(app):
    service = SoloistMediaService(owner_id="probe-owner", home=home)
    app.state.media_audio = service
    app.state.runtime_services = SimpleNamespace(app_controls=SimpleNamespace(
        _active=lambda value: object() if value == session_id else None))
    try:
        async with asyncio.timeout(20):
            lease = await service.start(session_id=session_id, media_id=media_id, uri=None,
                song_title=os.environ.get("JARVIS_TEST_SONG_TITLE"),
                artist=os.environ.get("JARVIS_TEST_SONG_ARTIST"))
            if lease.song_title:
                (home / "direct-music-selection.json").write_text(json.dumps({
                    "uri": lease.uri, "song_title": lease.song_title, "artist": lease.artist,
                    "source": "Spotify Web API", "hermes_used": False}))
        yield
    finally:
        await service.close()
        receipt["owned_player_stopped"] = service.process is None


app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(owner_id="probe-owner")

if __name__ == "__main__":
    import sys
    if "--serve" in sys.argv:
        import uvicorn
        (home / "probe-endpoint.json").write_text(json.dumps({"url": f"ws://127.0.0.1:18765/media-audio/{media_id}/stream?session_id={session_id}"}))
        uvicorn.run(app, host="127.0.0.1", port=18765, log_level="warning")
        raise SystemExit(0)
    try:
        with TestClient(app) as client:
            frames = packets = nonzero = 0
            squared = peak = 0
            began = time.monotonic()
            previous = None
            gaps = 0
            with client.websocket_connect(f"/media-audio/{media_id}/stream?session_id={session_id}") as websocket:
                while frames < 480_000:
                    payload = websocket.receive_bytes()
                    validate_packet(payload)
                    _, _, sequence, timestamp, source = struct.unpack("!BBHII", payload[:12])
                    values = struct.unpack("!" + "h" * ((len(payload) - 12) // 2), payload[12:])
                    count = len(values) // 2
                    if previous is not None and (source != previous[2] or sequence != (previous[0] + 1) % 65536
                                                 or timestamp != (previous[1] + previous[3]) % (2**32)):
                        gaps += 1
                    previous = (sequence, timestamp, source, count)
                    frames += count; packets += 1
                    nonzero += sum(value != 0 for value in values)
                    squared += sum(value * value for value in values)
                    peak = max(peak, max(abs(value) for value in values))
            receipt.update(frames=frames, packets=packets, source_seconds=frames / 48000,
                           wall_seconds=time.monotonic() - began, nonzero_samples=nonzero,
                           peak=peak / 32768, rms=math.sqrt(squared / (frames * 2)) / 32768,
                           sequence_gaps=gaps, passed=nonzero > 1000 and gaps == 0)
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        if type(error).__name__ == "WebSocketDisconnect":
            receipt["stream_failure"] = error.reason
        # Only MediaError contains intentionally public codes; never dump command arguments.
        if type(error).__name__ == "MediaError":
            receipt["error_code"] = str(error)
    finally:
        (home / "transport-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    
    raise SystemExit(0 if receipt["passed"] else 1)
