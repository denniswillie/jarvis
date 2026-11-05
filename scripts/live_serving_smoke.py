"""One bounded voice trial against the serving app, reusing the WebRTC fixture.

This replaces physical microphone/speaker hardware only. HTTP authentication,
the app WebSocket, Live, Luna and retained Hermes are the serving paths. Output
PCM is decoded browser audio, not proof of phone/glasses playback or usefulness.
Preparation is offline; execution requires --execute-paid. No service is started.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from live_hermes_voice_acceptance import (
    BROWSER_PROGRAM, BrowserPeer, CAPABILITIES, RATE, collect_audio, prepare,
    scrub, validate_bundle,
)
from evals.responsive_voice.synthetic_webrtc import Browser, annotate_connection_error
from app.settings import JarvisSettings
from app.visual_memory.validation import validate_visual_frame_event


class Timeline:
    def __init__(self, root):
        self.output = (root / "serving-timeline.jsonl").open("x")

    def record(self, kind, **values):
        self.output.write(json.dumps({"kind": kind,
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_s": time.monotonic(), **scrub(values)}, default=str) + "\n")
        self.output.flush()

    def close(self):
        self.output.close()


class ServingPeer(BrowserPeer):
    def __init__(self, browser, trace, socket):
        super().__init__(browser, trace)
        self.socket = socket
        self.activated = asyncio.Event()
        self.failure = None

    async def respond(self, event):
        self.trace.record("app_client_send", payload=event)
        try:
            await self.socket.send(json.dumps(event))
        except Exception as error:
            annotate_connection_error(error, connection="backend_audio_ws", operation="send")
            raise

    async def receive(self):
        while True:
            try:
                raw = await self.socket.recv()
            except ConnectionClosedOK:
                return
            except Exception as error:
                detail = annotate_connection_error(error, connection="backend_audio_ws", operation="receive")
                self.trace.record("connection_error", payload=detail)
                raise
            event = json.loads(raw)
            await self.send_json(event)
            if event.get("type") == "live.media.activate":
                self.activated.set()
            if event.get("type") in {"audio.error", "error"}:
                self.failure = event
                self.activated.set()
                return


def visual_event(image_bytes, *, settings, source="glasses", frame_sequence=1):
    event = {"type": "visual.frame.capture", "source": source,
             "frame_sequence": frame_sequence, "captured_at": datetime.now(timezone.utc).isoformat(),
             "content_type": "image/jpeg",
             "image_base64": base64.b64encode(image_bytes).decode()}
    validate_visual_frame_event(event, settings=settings, allowed_sources={source})
    return event


def followup_inputs(seconds, followup, followup_after, second_followup, second_followup_after):
    if second_followup is not None and followup is None:
        raise ValueError("A second follow-up requires the first")
    result = []
    for name, path, after in (("followup", followup, followup_after),
                              ("second_followup", second_followup, second_followup_after)):
        if path is None:
            continue
        if not 1 <= after <= seconds - 10 or (result and after <= result[-1]["after"]):
            raise ValueError("Follow-ups must be ordered with ten seconds left to observe")
        path = path.resolve()
        _, _, clips = validate_bundle(path)
        if len(clips) != 1:
            raise ValueError("Each follow-up bundle must contain one clip")
        result.append({"name": name, "path": path, "after": after, "clip": clips[0]})
    return result


async def run(root, *, seconds, execute_paid, followup=None, followup_after=12, image_path=None,
              second_followup=None, second_followup_after=24, image_interval=None):
    if not execute_paid:
        raise ValueError("Execution requires --execute-paid")
    if not 15 <= seconds <= 120:
        raise ValueError("Use a 15-120 second observation window")
    if image_interval is not None and (image_path is None or not .2 <= image_interval <= 5):
        raise ValueError("Repeated images require --image and a 0.2-5 second interval")
    root = root.resolve()
    _, limits, clips = validate_bundle(root)
    if len(clips) != 1:
        raise ValueError("This serving smoke accepts one speech clip")
    followups = followup_inputs(seconds, followup, followup_after,
                                second_followup, second_followup_after)
    settings = JarvisSettings()
    image_bytes = image_path.read_bytes() if image_path is not None else None
    if image_bytes is not None:
        visual_event(image_bytes, settings=settings)  # Reject invalid input before paid/session work.
    # Never repeat a partially completed trial under the same evidence identity.
    with (root / "serving-attempt.json").open("x") as output:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat()}, output)
    headers = {"Authorization": "Bearer " + settings.JARVIS_STATIC_BEARER_TOKEN}
    trace = Timeline(root)
    report = {"scope": "serving_http_websocket_and_real_provider_audio",
        "physical_playback": False, "useful_answer_latency_ms": None,
        "semantic_acceptance": "not_automatically_judged", "cleanup": {}}
    browser = peer = receiver = visual_socket = None
    session_id = None
    started = time.monotonic()
    async with httpx.AsyncClient(base_url="http://127.0.0.1:9000", headers=headers,
                                 timeout=20, follow_redirects=False) as http:
        try:
            response = await http.post("/sessions/start", json={
                "title": "Synthetic retained voice smoke", "resume_active": False})
            response.raise_for_status()
            session_id = response.json()["session"]["id"]
            report["session_id"] = session_id
            directory = root / "serving-browser"
            directory.mkdir(mode=0o700)
            browser = Browser(directory, root / "speech" / clips[0]["relative_path"])
            await browser.start()
            await browser.cdp.evaluate(BROWSER_PROGRAM)
            url = ("ws://127.0.0.1:9000/realtime/audio/ws?session_id=" + session_id
                   + "&client_kind=phone_audio&audio_provider=openai&audio_transport=webrtc&voice_route=live")
            async with connect(url, additional_headers=headers, max_size=2**20,
                               close_timeout=3, proxy=None) as socket:
                peer = ServingPeer(browser, trace, socket)
                offer = await peer.js("acceptance.prepare()")
                encoded = base64.b64encode((root / "speech" / clips[0]["relative_path"]).read_bytes()).decode()
                await peer.js("acceptance.load('work'," + json.dumps(encoded) + ")")
                for item in followups:
                    data = (item["path"] / "speech" / item["clip"]["relative_path"]).read_bytes()
                    await peer.js("acceptance.load(" + json.dumps(item["name"]) + ","
                                  + json.dumps(base64.b64encode(data).decode()) + ")")
                receiver = asyncio.create_task(peer.receive())
                await peer.respond({"type": "audio.webrtc.offer", "sdp": offer["sdp"],
                    "capture_mode": "auto", "voice_route": CAPABILITIES})
                await asyncio.wait_for(peer.activated.wait(), 35)
                if peer.failure:
                    raise RuntimeError("serving_audio_start_failed")
                report["ready_elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
                if image_bytes is not None:
                    visual_socket = await connect(
                        "ws://127.0.0.1:9000/realtime/visual/ws?session_id=" + session_id
                        + "&visual_session_id=" + str(uuid4()),
                        additional_headers=headers, max_size=2**20, close_timeout=3, proxy=None)
                    created = json.loads(await asyncio.wait_for(visual_socket.recv(), 10))
                    trace.record("visual_server_event", payload=created)
                    if created.get("type") != "visual.session.created":
                        raise RuntimeError("visual_session_start_failed")
                    source = "iphone_camera" if "iphone_camera" in created.get("allowed_sources", []) else "glasses"
                    frame = visual_event(image_bytes, settings=settings, source=source)
                    trace.record("synthetic_visual_input", source=source, captured_at=frame["captured_at"],
                                 bytes=len(image_bytes), sha256=sha256(image_bytes).hexdigest())
                    await visual_socket.send(json.dumps(frame))
                    accepted = json.loads(await asyncio.wait_for(visual_socket.recv(), 20))
                    trace.record("visual_server_event", payload=accepted)
                    if accepted.get("type") != "visual.frame.accepted":
                        raise RuntimeError("visual_frame_not_accepted")
                    report["visual_input"] = {"synthetic": True, "source": source,
                        "frame_id": accepted["frame_id"], "captured_at": frame["captured_at"],
                        "sha256": sha256(image_bytes).hexdigest(), "analysis_verified": False}
                speech = await peer.js("acceptance.play('work')")
                speech["last_active_sample"] = speech["start_sample"] + clips[0]["last_active_sample_exclusive"]
                report["speech"] = speech
                trace.record("speech_scheduled", payload=speech)
                deadline = time.monotonic() + seconds
                followup_origin = time.monotonic()
                next_frame_at = followup_origin + image_interval if image_interval is not None else None
                frame_sequence = 1
                while time.monotonic() < deadline:
                    if receiver.done():
                        await receiver
                        raise RuntimeError("serving_audio_closed_before_observation_end")
                    for event in await peer.js("acceptance.drain()"):
                        trace.record("client_data_channel", payload=event)
                    state = await peer.js("acceptance.snapshot()")
                    if state.get("fatal"):
                        raise RuntimeError("browser_media_failed")
                    if next_frame_at is not None and time.monotonic() >= next_frame_at:
                        frame_sequence += 1
                        frame = visual_event(image_bytes, settings=settings, source=source,
                                             frame_sequence=frame_sequence)
                        trace.record("synthetic_visual_input", source=source,
                                     frame_sequence=frame_sequence, captured_at=frame["captured_at"],
                                     bytes=len(image_bytes), sha256=sha256(image_bytes).hexdigest())
                        await visual_socket.send(json.dumps(frame))
                        accepted = json.loads(await asyncio.wait_for(visual_socket.recv(), 20))
                        trace.record("visual_server_event", payload=accepted)
                        if accepted.get("type") != "visual.frame.accepted":
                            raise RuntimeError("visual_frame_not_accepted")
                        next_frame_at = time.monotonic() + image_interval
                    if followups and time.monotonic() >= followup_origin + followups[0]["after"]:
                        item = followups.pop(0)
                        extra = await peer.js("acceptance.play(" + json.dumps(item["name"]) + ")")
                        extra["last_active_sample"] = extra["start_sample"] + item["clip"]["last_active_sample_exclusive"]
                        report[item["name"] + "_speech"] = extra
                        trace.record(item["name"] + "_speech_scheduled", payload=extra)
                    await asyncio.sleep(.2)
                report["audio"] = await collect_audio(peer, root, limits)
                signal = report["audio"].get("firstSignal")
                if signal is not None:
                    report["first_signal_relative_to_last_input_ms"] = round(
                        (signal - speech["last_active_sample"]) * 1000 / RATE, 2)
                    report["observation_completed"] = True
                else:
                    report["failure"] = {"type": "RuntimeError", "code": "serving_output_silent"}
        except Exception as error:
            report["failure"] = {"type": type(error).__name__,
                                 "code": str(error) if type(error) is RuntimeError else "see_timeline",
                                 "connection": getattr(error, "_jarvis_connection_diagnostic", None),
                                 "browser_before_cleanup": browser.process_state() if browser is not None else None}
            trace.record("trial_failure", payload=report["failure"])
            if peer is not None:
                try:
                    report["audio"] = await collect_audio(peer, root, limits)
                except Exception as capture_error:
                    report["capture_error"] = type(capture_error).__name__
                    report["capture_connection_error"] = getattr(capture_error, "_jarvis_connection_diagnostic", None)
                    trace.record("capture_failure", payload=report["capture_connection_error"])
        finally:
            if receiver is not None:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            if browser is not None:
                report["cleanup"]["browser"] = await browser.close()
            if visual_socket is not None:
                await visual_socket.close()
                report["cleanup"]["visual_socket_closed"] = True
            if session_id is not None:
                response = await http.post("/sessions/" + session_id + "/close")
                report["cleanup"]["session_closed"] = response.is_success
            trace.close()
            (root / "serving-result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({key: report[key] for key in (
        "observation_completed", "ready_elapsed_ms", "first_signal_relative_to_last_input_ms", "failure"
    ) if key in report}))
    return 0 if report.get("observation_completed") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--question", default="Create a text file named voice-smoke.txt containing the result of seventeen times twenty-three.")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--followup", type=Path, help="Optional separately prepared second speech bundle")
    parser.add_argument("--followup-after", type=float, default=12)
    parser.add_argument("--second-followup", type=Path, help="Optional third speech bundle for repeated interruption")
    parser.add_argument("--second-followup-after", type=float, default=24)
    parser.add_argument("--image", type=Path, help="Optional synthetic JPEG sent through the normal visual socket")
    parser.add_argument("--image-interval", type=float,
                        help="Repeat the synthetic image with fresh capture times, every 0.2-5 seconds")
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "prepare":
        prepare(args.root, question=args.question)
    else:
        project = Path(__file__).resolve().parents[1]
        load_dotenv(project / "backend/.env")
        load_dotenv(project / ".jarvis-dev/phone-viewer-integration.gu1kvk/serving.env", override=True)
        raise SystemExit(asyncio.run(run(args.root, seconds=args.seconds, execute_paid=args.execute_paid,
                                        followup=args.followup, followup_after=args.followup_after,
                                        second_followup=args.second_followup,
                                        second_followup_after=args.second_followup_after,
                                        image_path=args.image, image_interval=args.image_interval)))
