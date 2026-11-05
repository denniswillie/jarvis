"""One bounded existing-desktop viewer proof; never starts a sandbox or agent."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from dotenv import dotenv_values
from playwright.async_api import async_playwright


INIT = """() => {
    window.__jarvisViewEvents = [];
    window.__jarvisSocketStates = [];
    window.webkit = {messageHandlers: {sandboxView: {postMessage: data => {
        window.__jarvisViewEvents.push({state: data.state, at: performance.now()});
    }}}};
    const NativeSocket = window.WebSocket;
    window.WebSocket = new Proxy(NativeSocket, {construct(target, args) {
        const socket = Reflect.construct(target, args);
        const state = {readyState: socket.readyState, opened: false, closed: false};
        window.__jarvisSocketStates.push(state);
        socket.addEventListener('open', () => {
            state.opened = true; state.readyState = socket.readyState;
        });
        socket.addEventListener('error', () => {state.error = true;});
        socket.addEventListener('close', event => {
            state.closed = true; state.readyState = socket.readyState;
            state.closeCode = event.code; state.wasClean = event.wasClean;
        });
        return socket;
    }});
} """

CANVASES = """() => [...document.querySelectorAll('canvas')].map(canvas => {
    const rect = canvas.getBoundingClientRect();
    const result = {width: canvas.width, height: canvas.height,
        cssWidth: rect.width, cssHeight: rect.height};
    if (!canvas.width || !canvas.height) return result;
    try {
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let paintedPixels = 0;
        for (let alpha = 3; alpha < pixels.length; alpha += 4) {
            if (pixels[alpha] !== 0) paintedPixels++;
        }
        const step = Math.max(4, Math.ceil(pixels.length / 65536 / 4) * 4);
        let sum = 0, squares = 0, count = 0, opaque = 0, min = 255, max = 0;
        const colors = new Set();
        for (let i = 0; i < pixels.length; i += step) {
            const brightness = (pixels[i] + pixels[i+1] + pixels[i+2]) / 3;
            sum += brightness; squares += brightness * brightness; count++;
            if (pixels[i+3]) opaque++;
            min = Math.min(min, brightness); max = Math.max(max, brightness);
            if (colors.size < 1024) colors.add([pixels[i],pixels[i+1],pixels[i+2],pixels[i+3]].join(','));
        }
        Object.assign(result, {paintedPixels, samples: count, nontransparent: opaque, uniqueColors: colors.size,
            meanBrightness: sum/count, variance: Math.max(0, squares/count - (sum/count)**2), min, max});
    } catch (error) {result.error = error.name;}
    return result;
})"""


async def run(args):
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    token = dotenv_values(args.env_file).get("JARVIS_STATIC_BEARER_TOKEN")
    if not token:
        raise RuntimeError("Configured bearer token unavailable")
    base = args.base_url.rstrip("/")
    evidence = {"kind": "existing-retained-desktop-browser-proof", "success": False,
                "session_id": args.session_id, "allocation_id": args.allocation_id,
                "new_sandbox": False, "new_task": False, "checks": {},
                "websockets": [], "viewer_events": [], "canvases": [],
                "console": [], "request_failures": [], "http_failures": [], "diagnostic_errors": []}
    def safe(value):
        text = str(value).replace(token, "[redacted]")
        return re.sub(r"(?:wss?|https?)://[^\s\"']+", "[endpoint]", text)[:1200]
    def observe_socket(socket):
        state = {"path": urlsplit(socket.url).path, "received": 0, "sent": 0,
                 "received_bytes": 0, "sent_bytes": 0, "initial_messages": [], "closed": False}
        evidence["websockets"].append(state)
        def frame(direction, payload):
            payload = payload.encode() if isinstance(payload, str) else payload
            state[direction] += 1
            state[direction + "_bytes"] += len(payload)
            if len(state["initial_messages"]) < 12:
                record = {"direction": direction, "bytes": len(payload)}
                if payload.startswith(b"RFB "):
                    record["rfb_version"] = payload[:12].decode("ascii", errors="replace")
                elif len(payload) <= 4:
                    record["protocol_bytes_hex"] = payload.hex()
                state["initial_messages"].append(record)
        socket.on("framereceived", lambda payload: frame("received", payload))
        socket.on("framesent", lambda payload: frame("sent", payload))
        socket.on("close", lambda: state.update(closed=True))
        socket.on("socketerror", lambda error: state.update(error=safe(error)))

    view_id = None
    async with httpx.AsyncClient(base_url=base, headers={"Authorization": "Bearer " + token}, timeout=20) as client:
        try:
            for name, authorization, expected in (("anonymous", "", 401), ("invalid_bearer", "Bearer invalid-view-proof", 403)):
                response = await client.get("/sandbox-views/current", headers={"Authorization": authorization})
                evidence["checks"][name] = response.status_code
                assert response.status_code == expected
            response = await client.get("/sandbox-views/current")
            response.raise_for_status()
            assert response.json()["view"] is None, "Another view is active; refusing to replace it"
            response = await client.get("/sandbox-views", params={"session_id": args.session_id})
            response.raise_for_status()
            target = next(item for item in response.json()["items"] if item["allocation_id"] == args.allocation_id)
            assert target["state"] == "running" and target["capability"] == "desktop" and target["can_view"]
            evidence["generation"] = target["generation"]
            response = await client.post("/sandbox-views", json={"session_id": args.session_id,
                "allocation_id": args.allocation_id, "generation": target["generation"], "resume": False})
            evidence["checks"]["open_status"] = response.status_code
            response.raise_for_status()
            lease = response.json()
            view_id = str(UUID(lease.get("view_id") or lease.get("id")))
            evidence["view_id"] = view_id
            page_path = f"/sandbox-views/{view_id}/page"
            response = await client.get(page_path, params={"session_id": str(uuid4())})
            evidence["checks"]["wrong_session"] = response.status_code
            assert 400 <= response.status_code < 500

            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
                page = None
                try:
                    context = await browser.new_context(extra_http_headers={"Authorization": "Bearer " + token},
                                                        viewport={"width": 1100, "height": 800})
                    await context.add_init_script("(" + INIT + ")();")
                    async def local_only(route):
                        if route.request.url.startswith(base + "/"):
                            await route.continue_()
                        else:
                            await route.abort()
                    await context.route("**/*", local_only)
                    page = await context.new_page()
                    page.on("websocket", observe_socket)
                    page.on("pageerror", lambda error: evidence["console"].append({"type": "pageerror", "text": safe(error)}))
                    page.on("console", lambda message: evidence["console"].append({"type": message.type, "text": safe(message.text)})
                            if message.type in ("warning", "error") and len(evidence["console"]) < 50 else None)
                    page.on("requestfailed", lambda request: evidence["request_failures"].append(
                        {"path": urlsplit(request.url).path, "failure": safe(request.failure)}))
                    page.on("response", lambda response: evidence["http_failures"].append(
                        {"path": urlsplit(response.url).path, "status": response.status}) if response.status >= 400 else None)
                    response = await page.goto(base + page_path + "?session_id=" + args.session_id,
                                               wait_until="domcontentloaded", timeout=20000)
                    evidence["checks"]["page_status"] = response.status
                    await page.wait_for_function(
                        "() => (" + CANVASES + ")().some(c => c.paintedPixels > 0) && "
                        "window.__jarvisViewEvents.some(e => e.state === 'frame')",
                        polling=250, timeout=30000)
                except Exception as exc:
                    evidence["render_error"] = {"type": type(exc).__name__, "message": safe(exc)}
                finally:
                    if page is not None:
                        for name, expression in (("viewer_events", "window.__jarvisViewEvents"),
                                                 ("socket_states", "window.__jarvisSocketStates"), ("canvases", CANVASES)):
                            try:
                                evidence[name] = await asyncio.wait_for(page.evaluate(expression), timeout=5)
                            except Exception as exc:
                                evidence["diagnostic_errors"].append({"step": name, "error": safe(exc)})
                        try:
                            await page.screenshot(path=str(output / "desktop.png"), full_page=True, timeout=10000)
                            evidence["screenshot"] = str(output / "desktop.png")
                        except Exception as exc:
                            evidence["diagnostic_errors"].append({"step": "screenshot", "error": safe(exc)})
                    await browser.close()
            frame = any(item["state"] == "frame" for item in evidence["viewer_events"] or [])
            painted = any(item.get("paintedPixels", 0) > 0 for item in evidence["canvases"] or [])
            evidence["checks"]["painted_pixels_present"] = painted
            evidence["checks"]["frame_reported"] = frame
            evidence["rendering_outcome"] = ("painted_frame" if frame else "painted_without_frame_report") if painted else "no_painted_pixels"
            evidence["success"] = frame and painted
        except Exception as exc:
            evidence["error"] = {"type": type(exc).__name__, "message": safe(exc)}
        finally:
            if view_id is not None:
                try:
                    response = await client.delete(f"/sandbox-views/{view_id}", params={"session_id": args.session_id})
                    evidence["checks"]["close_status"] = response.status_code
                    assert response.status_code == 200
                    response = await client.get(f"/sandbox-views/{view_id}/page", params={"session_id": args.session_id})
                    evidence["checks"]["revoked_page_status"] = response.status_code
                    assert 400 <= response.status_code < 500
                    response = await client.get("/sandbox-views", params={"session_id": args.session_id})
                    response.raise_for_status()
                    target = next(item for item in response.json()["items"] if item["allocation_id"] == args.allocation_id)
                    evidence["checks"]["preserved_state"] = target["state"]
                    evidence["checks"]["preserved_generation"] = target["generation"] == evidence["generation"]
                    assert target["state"] == "running" and evidence["checks"]["preserved_generation"]
                except Exception as exc:
                    evidence["success"] = False
                    evidence["cleanup_error"] = {"type": type(exc).__name__, "message": safe(exc)}
            evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
            (output / "report.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps(evidence, indent=2))
    return 0 if evidence["success"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--env-file", default="backend/.env")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
