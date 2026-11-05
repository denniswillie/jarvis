"""Authenticated view-only standard noVNC, with provider secrets held server-side."""
import asyncio
import json
import logging
from pathlib import Path
import secrets
import ssl
import time
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, WebSocket
import certifi
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import connect

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.runtime.http_privacy import private_http_logs
from app.sandbox.view_access import unavailable


router = APIRouter(prefix="/sandbox-views", tags=["sandbox-view"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]
PRIVATE = {"Cache-Control": "private, no-store", "Vary": "Authorization", "Referrer-Policy": "no-referrer",
           "X-Content-Type-Options": "nosniff"}
ASSETS = Path(__file__).with_name("view_assets")
# WebSocket debug logs include handshake credentials. Keep this isolated from
# application logging configuration, including provider failure diagnostics.
TRANSPORT_LOGGER = logging.Logger("jarvis.sandbox.private_websocket")
TRANSPORT_LOGGER.disabled = True


class ViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    allocation_id: UUID
    generation: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume: bool = Field(default=False, strict=True)


def service(request):
    value = getattr(getattr(request.app.state, "runtime_services", None), "sandbox_views", None)
    if value is None:
        raise unavailable("sandbox_view_unavailable", "Sandbox viewing isn't available yet.", 503)
    return value


@router.get("")
async def choices(request: Request, principal: Principal, session_id: UUID,
                  after: str = "", limit: int = Query(default=8, ge=1, le=32)):
    return JSONResponse(await service(request).choices(principal.owner_id, session_id, after=after, limit=limit), headers=PRIVATE)


@router.get("/inventory")
async def cached_inventory(request: Request, principal: Principal,
                           after: str = "", limit: int = Query(default=8, ge=1, le=32)):
    return JSONResponse(service(request).cached_inventory(principal.owner_id, after=after, limit=limit), headers=PRIVATE)


@router.get("/current")
async def current_view(request: Request, principal: Principal):
    return JSONResponse({"view": service(request).current(principal.owner_id)}, headers=PRIVATE)


@router.get("/thumbnail")
async def thumbnail(request: Request, principal: Principal, session_id: UUID, allocation_id: UUID,
                    generation: str = Query(pattern=r"^[0-9a-f]{64}$")):
    result = await service(request).thumbnail(principal.owner_id, session_id, allocation_id, generation)
    return JSONResponse(result, headers=PRIVATE)


@router.post("")
async def open_view(body: ViewRequest, request: Request, principal: Principal):
    lease = await service(request).open(principal.owner_id, body.session_id, body.allocation_id,
                                       body.generation, resume=body.resume)
    return JSONResponse(lease.wire(), headers=PRIVATE)


@router.get("/assets/{path:path}")
async def assets(path: str):
    selected = (ASSETS / path).resolve()
    if (not selected.is_relative_to(ASSETS.resolve()) or not selected.is_file()
            or selected.suffix not in {".js", ".css"}):
        raise unavailable("sandbox_view_asset_missing", "Viewer content is unavailable.", 404)
    return FileResponse(selected, headers={"Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff"})


@router.get("/{identity}")
async def status(identity: UUID, request: Request, principal: Principal, session_id: UUID):
    views = service(request)
    lease = views.get(principal.owner_id, session_id, identity)
    await views.validate(lease)
    return JSONResponse(lease.wire(), headers=PRIVATE)


@router.delete("/{identity}")
async def close_view(identity: UUID, request: Request, principal: Principal, session_id: UUID):
    service(request).revoke(principal.owner_id, session_id, identity)
    return JSONResponse({"closed": True}, headers=PRIVATE)


@router.get("/{identity}/page")
async def page(identity: UUID, request: Request, principal: Principal, session_id: UUID):
    views = service(request)
    lease = views.get(principal.owner_id, session_id, identity)
    await views.validate(lease, native=False)
    config = json.dumps({"ticket": lease.ticket, "socket": f"/sandbox-views/{identity}/stream"})
    html = '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>html,body,#screen{margin:0;width:100%;height:100%;overflow:hidden;background:#161d1c}canvas{outline:none}</style>
</head><body><div id="screen" aria-label="Sandbox screen, view only"></div>
<script id="view-config" type="application/json">''' + config + '''</script>
<script type="module" src="/sandbox-views/assets/viewer.js"></script></body></html>'''
    return HTMLResponse(html, headers=PRIVATE | {"Content-Security-Policy":
        "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; connect-src 'self'; img-src data: blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})


class NoRedirectConnect(connect):
    def process_redirect(self, exception):
        return exception


@router.websocket("/{identity}/stream")
async def stream(websocket: WebSocket, identity: UUID):
    reserved = False
    jobs = []
    lease = None
    try:
        views = service(websocket)
        lease = views.leases.get(identity)
        protocols = websocket.scope.get("subprotocols", [])
        if (lease is None or len(protocols) != 2 or protocols[0] != "jarvis-view-v1"
                or not secrets.compare_digest(protocols[1], lease.ticket) or lease.connected):
            await websocket.close(code=1008)
            return
        connection = await views.validate(lease)
        if lease.connected or lease.closed.is_set():
            await websocket.close(code=1008)
            return
        lease.connected = True
        reserved = True
        await websocket.accept(subprotocol="jarvis-view-v1")
        # Standard binary RFB forwarding only. No cookies, provider identifiers,
        # redirect targets or credentials are returned to the browser.
        with private_http_logs():
            async with NoRedirectConnect(connection.url,
                    additional_headers={"e2b-traffic-access-token": connection.token}, proxy=None,
                    ssl=ssl.create_default_context(cafile=certifi.where()) if connection.url.startswith("wss:") else None,
                    logger=TRANSPORT_LOGGER,
                    open_timeout=10, close_timeout=2, max_size=8 * 1024 * 1024, max_queue=4,
                    compression=None) as upstream:
                async def receive():
                    window, count = time.monotonic(), 0
                    while True:
                        data = await websocket.receive_bytes()
                        if time.monotonic() - window >= 1:
                            window, count = time.monotonic(), 0
                        count += len(data)
                        if len(data) > 65536 or count > 262144:
                            raise ValueError("viewer_input_limit")
                        await upstream.send(data)

                async def send():
                    count = 0
                    async for data in upstream:
                        if not isinstance(data, bytes):
                            raise ValueError("viewer_binary_required")
                        count += len(data)
                        if count > 256 * 1024 * 1024:
                            raise ValueError("viewer_transfer_limit")
                        await websocket.send_bytes(data)

                async def guard():
                    iterations = 0
                    while True:
                        try:
                            await asyncio.wait_for(lease.closed.wait(), timeout=5)
                            return
                        except TimeoutError:
                            iterations += 1
                            await views.validate(lease, native=iterations % 3 == 0)

                jobs = [asyncio.create_task(operation()) for operation in (receive, send, guard)]
                done, _ = await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED)
                for job in done:
                    job.result()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        # Never log upstream exception/headers/URLs. Native status inspection
        # distinguishes known shutdown/access loss from a transport failure.
        pass
    finally:
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        if lease is not None and reserved:
            lease.connected = False
        try:
            await websocket.close(code=1000 if lease is not None and lease.closed.is_set() else 1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
