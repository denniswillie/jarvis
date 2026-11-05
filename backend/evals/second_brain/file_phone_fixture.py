"""Interactive phone file fixture: real file routes/sources, isolated database and memory storage."""
import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
from io import BytesIO
import os
from pathlib import Path
import secrets
import socket
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import uvicorn

from app.auth.static_bearer import require_static_bearer
from app.errors import JarvisHTTPError
from app.files.router import router
from app.files.tools import FileToolModule, _file_payload
from app.knowledge.files import SourceFileRepository
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.runtime.feature_config import MemoryAuthority
from app.settings import JarvisSettings, get_settings
from app.visual_memory.models import ValidatedVisualFrame, VisualFrameStorageObject
from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.storage import VisualFrameStorageClient
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.hindsight_service import save
from evals.second_brain.postgres_recovery import postgres_recovery_fixture
from evals.second_brain.source_fixture import source_fixture


class FixtureServer(uvicorn.Server):
    def handle_exit(self, sig, frame):
        # This process owns database/storage cleanup outside the HTTP server.
        # Uvicorn's default handler re-raises the signal before those scopes exit.
        self.should_exit = True


class PausableStorage(MemoryStorage):
    def __init__(self):
        super().__init__()
        self.pause_path = None
        self.read_entered = asyncio.Event()
        self.read_release = asyncio.Event()

    async def read_object(self, **kwargs):
        value = await super().read_object(**kwargs)
        if self.pause_path == kwargs["path"]:
            self.pause_path = None
            self.read_entered.set()
            await self.read_release.wait()
        return value


async def serve(root, database):
    state = {"format": "jarvis.file_phone_fixture.v1", "pid": os.getpid(), "status": "starting",
             "synthetic_only": True, "native_models_contacted": False, "requests": [],
             "cleanup_errors": [], "files": []}
    path = root / "phone-fixture.json"
    token = "phone-fixture-" + secrets.token_hex(24)
    save(root / "client.json", {"token": token})
    try:
        async with source_fixture(database.dsn("source"), root / "database-manifests",
                                  test_name="interactive-file-phone") as fixture:
            storage = PausableStorage()
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            files = SourceFileRepository(fixture.sources, storage, owner_id=fixture.owner, area_id=fixture.area)
            text = "\n    mid = (low + high) // 2\n    return mid\n\n"
            seeded = [
                await files.create_text_file(owner_id=fixture.owner, session_id=session,
                    filename="workshop-draft.txt", content=text,
                    description="Synthetic code draft. Leading spaces and final newlines are intentional.",
                    operation_key="phone-text"),
                await files.create_text_file(owner_id=fixture.owner, session_id=session,
                    filename="pending-read.txt", content="Synthetic pending download control.\n",
                    description="Used for interruption and revocation testing.", operation_key="phone-pending"),
            ]
            pixels = Image.new("RGB", (640, 320), "#23464b")
            draw = ImageDraw.Draw(pixels)
            draw.rectangle((20, 20, 620, 300), outline="#f0c47a", width=6)
            draw.text((48, 92), "SYNTHETIC TEST IMAGE", font=ImageFont.load_default(size=34), fill="white")
            draw.text((48, 160), "Retained original / no real camera", font=ImageFont.load_default(size=25), fill="#f0c47a")
            buffer = BytesIO()
            pixels.save(buffer, format="JPEG", quality=95)
            raw = buffer.getvalue()
            now, visual = datetime.now(UTC), uuid4()
            raw_path = VisualFrameStorageClient.build_storage_path(storage,
                owner_id=fixture.owner, session_id=session, visual_session_id=visual, frame_sequence=0)
            frames = VisualFrameRepository(fixture.pool)
            pending = await frames.create_pending_frame(owner_id=fixture.owner, session_id=session,
                audio_session_id=None, visual_session_id=visual,
                frame=ValidatedVisualFrame(0, now, "glasses", "image/jpeg", 640, 320, raw),
                server_received_at=now, storage_bucket="synthetic-raw", storage_path=raw_path,
                raw_image_expires_at=now + timedelta(minutes=5))
            frame = await frames.mark_frame_available(owner_id=fixture.owner, frame_id=pending.id,
                stored_object=VisualFrameStorageObject("synthetic-raw", raw_path, len(raw)))
            window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=120)
            lease = window.begin_stream(fixture.owner, session, visual)
            window.put(SensoryVisualFrame(frame_id=frame.id, owner_id=fixture.owner, session_id=session,
                audio_session_id=None, visual_session_id=visual, frame_sequence=0, source="glasses",
                captured_at=now, server_received_at=now, content_type="image/jpeg", width=640, height=320,
                storage_bucket="synthetic-raw", storage_path=raw_path,
                raw_image_expires_at=frame.raw_image_expires_at, image_bytes=raw), stream_lease=lease)
            files.current_images = CurrentImageService(frame_window=window, max_age_seconds=60,
                max_image_bytes=512000, max_width=1024, max_height=1024)
            seeded.append(await files.capture_current_frame(owner_id=fixture.owner, session_id=session,
                filename="synthetic-view.jpg", description="A generated fixture, not a real camera observation.",
                operation_key="phone-image"))
            window.end_stream(lease)
            for file in seeded:
                content = await files.read_content(owner_id=fixture.owner, file_id=file.id)
                state["files"].append({"id": str(file.id), "filename": file.filename,
                    "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
            settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=fixture.owner,
                                     JARVIS_STATIC_BEARER_TOKEN=token)
            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[get_settings] = lambda: settings
            app.state.runtime_services = SimpleNamespace(settings=settings, memory_authority=MemoryAuthority.HINDSIGHT,
                tool_modules=(FileToolModule(files, storage),))
            @app.exception_handler(JarvisHTTPError)
            async def errors(request: Request, error: JarvisHTTPError):
                return JSONResponse({"error": {"code": error.code, "message": error.message}},
                                    status_code=error.status_code)
            @app.middleware("http")
            async def record(request, call_next):
                response = await call_next(request)
                state["requests"].append({"method": request.method, "path": request.url.path,
                    "status": response.status_code, "time": datetime.now(UTC).isoformat(),
                    "authorized": request.headers.get("authorization") == "Bearer " + token})
                state["requests"] = state["requests"][-500:]
                save(path, state)
                return response
            protected = [Depends(require_static_bearer)]
            @app.get("/sessions/active", dependencies=protected)
            async def active(): return {"session": None, "realtime": None}
            @app.get("/reminders/active", dependencies=protected)
            async def reminders(): return {"reminders": []}
            @app.get("/reminders/events", dependencies=protected)
            async def events(): return {"events": []}
            @app.get("/oauth/google/status", dependencies=protected)
            async def calendar(): return {"connected": False, "expires_at": None, "scopes": []}
            @app.get("/memory/recent", dependencies=protected)
            async def memory(): return {"items": [], "next_cursor": None, "view": "original_sources"}
            @app.get("/memory/working-state", dependencies=protected)
            async def working(): return {"session": None, "owner": None}
            @app.get("/health")
            async def health(): return {"status": "ok", "synthetic_only": True}
            @app.get("/__fixture/status", dependencies=protected)
            async def status():
                return {"files": [_file_payload(file) for file in await files.list_files(
                    owner_id=fixture.owner, query=None, limit=8)],
                    "read_entered": storage.read_entered.is_set(), "objects": len(storage.objects)}
            @app.post("/__fixture/pause/{file_id}", dependencies=protected)
            async def pause(file_id: UUID):
                file = await files.get_file(owner_id=fixture.owner, file_id=file_id)
                storage.read_entered.clear()
                storage.read_release.clear()
                storage.pause_path = file.storage_path
                return {"armed": True}
            @app.post("/__fixture/release", dependencies=protected)
            async def release():
                storage.read_release.set()
                return {"released": True}
            @app.post("/__fixture/revoke/{file_id}", dependencies=protected)
            async def revoke(file_id: UUID):
                return await files.delete_source_file(owner_id=fixture.owner, file_id=file_id)
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            state["url"] = "http://localhost:" + str(port)
            server = FixtureServer(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=5))
            save(root / "client.json", {"token": token, "url": state["url"]})
            state["status"] = "ready"
            save(path, state)
            print("PHONE_FIXTURE_READY " + state["url"], flush=True)
            try:
                await server.serve(sockets=[listener])
            finally:
                storage.read_release.set()
                listener.close()
                for file in seeded:
                    try:
                        await files.delete_source_file(owner_id=fixture.owner, file_id=file.id)
                    except Exception as error:
                        state["cleanup_errors"].append({"type": type(error).__name__,
                                                       "code": getattr(error, "code", None)})
                state["objects_after_cleanup"] = len(storage.objects)
        state["status"] = "cleaned" if not state["cleanup_errors"] else "cleanup_failed"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
        raise
    finally:
        save(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    with postgres_recovery_fixture(args.root, data_mebibytes=1024) as database:
        asyncio.run(serve(args.root, database))


if __name__ == "__main__":
    main()
