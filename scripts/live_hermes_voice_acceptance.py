#!/usr/bin/env python3
"""Bounded Live -> real Jarvis session -> Hermes acceptance, never a new agent.

Prepare is local macOS speech generation. Run requires --execute-paid, an
explicit application factory and database credentials. No default application
factory, server restart, provider retry, E2B, or serving owner is installed.

Parent integration seam (factory is an async context manager):

    @asynccontextmanager
    async def factory(*, fixture, session_id, websocket, offer,
                      client_capabilities, limits, artifact_dir, settings):
        # Compose NORMAL production services on fixture.pool/owner/area.
        # Register the unstarted LiveAudioSession with its session manager.
        # Supply the real native client-delegation handler and domain tools.
        # Do not start a scheduler, session.run(), or runner.run() here.
        yield HarnessApplication(session=session, runner=owner_scoped_runner)
        # Close remaining composition resources, never fixture.pool.

The runner starts/stops only those two objects. Limits must also be applied to
the task repository by the factory. The isolated fixture reuses DatabaseFixture
for manifested, independently audited SQL teardown. Actual Gemini/vision/iOS/Bluetooth/acoustics are
NOT exercised. Browser PCM is decoded remote audio, not proof of human hearing.
Raw synthetic dialogue/context is retained privately, line by line. Credentials,
SDP and private reasoning output are excluded. No classifier injects a command,
note, transcript, delegation or result on behalf of a model.

Example (the sibling runtime module supplies the production composition):
  PYTHONPATH=backend python scripts/live_hermes_voice_acceptance.py prepare /ABS/run
  PYTHONPATH=backend python scripts/live_hermes_voice_acceptance.py run /ABS/run \
    --factory live_hermes_voice_runtime:application --env-file backend/.env --execute-paid
  # Listen to output.wav and inspect timeline.txt/SQL receipts before reviewing:
  python scripts/live_hermes_voice_acceptance.py review /ABS/run \
    --useful-audio-sample 123456 --task-correct --spoken-result-correct
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from unittest.mock import patch
from uuid import uuid4
import wave


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

FORMAT = "jarvis.live_hermes_voice_acceptance.v1"
RATE = 48000
DEFAULT_QUESTION = "Please work out a three-step packing checklist for a rainy day walk and explain your choices."
CAPABILITIES = dict(supported_versions=[1], supported_transports=["native_voice"],
    supported_media_topologies=["phone_to_provider"], input_codecs=["webrtc_negotiated"],
    output_codecs=["webrtc_negotiated"], local_output_fencing=True, client_playback_boundaries=[])
TABLES = ("jarvis_background_tasks", "jarvis_task_attempts", "jarvis_task_actions",
          "jarvis_task_events")


class HarnessError(RuntimeError):
    """Only stable codes escape into console reports; no credential-bearing errors."""


def require(value, code):
    if not value:
        raise HarnessError(code)


@dataclass(frozen=True)
class Limits:
    scenario_seconds: int = 180
    bootstrap_seconds: int = 360
    cleanup_seconds: int = 45
    active_seconds: int = 120
    tool_steps: int = 20
    request_output_tokens: int = 4096
    task_output_tokens: int = 8192
    task_input_tokens: int = 64000
    max_attempts: int = 2
    max_model_requests: int = 24
    max_trace_bytes: int = 32 * 1024 * 1024
    max_rows: int = 512

    def validate(self):
        ceilings = Limits()
        for name, value in asdict(self).items():
            require(type(value) is int and 0 < value <= getattr(ceilings, name), "invalid_run_limit")
        require(self.scenario_seconds >= 10, "scenario_too_short")
        return self


@dataclass
class HarnessApplication:
    session: Any
    runner: Any


@asynccontextmanager
async def isolated_fixture(dsn, directory, *, test_name, limits, cleanup_gate):
    """Same manifested schema/bootstrap as source_fixture, with a live-work hold.

    Never drop a worker's ledger while native shutdown is unconfirmed. The
    standard DatabaseFixture recovery command can clean a retained manifest
    after its exact resources have been reconciled by the owner.
    """
    import asyncpg
    from app.knowledge.sources import SourceRepository
    from app.migrations.runner import apply_migrations, quote_schema
    from evals.second_brain.fixtures import DatabaseFixture
    from evals.second_brain.source_fixture import SourceFixture, close_fixture_pool
    fixture = DatabaseFixture(dsn, directory, test_name=test_name)
    pool = None
    try:
        async with asyncio.timeout(limits.bootstrap_seconds):
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=300)
            try:
                schema = await fixture.create_schema(connection)
                await apply_migrations(connection, schema=schema, fresh=True)
            finally:
                await connection.close(timeout=5)
            async def setup(connection):
                await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,public,extensions")
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, timeout=10, command_timeout=10, setup=setup)
            async with pool.acquire() as connection:
                area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                    "values($1,'Voice acceptance fixture','test') returning id", fixture.owners[0])
        yield SourceFixture(fixture, pool, SourceRepository(pool), *fixture.owners, area)
    finally:
        try:
            if pool is not None:
                await close_fixture_pool(pool)
        finally:
            if cleanup_gate["safe"]:
                async with asyncio.timeout(180):
                    await fixture.cleanup()
            else:
                fixture.manifest.update(status="cleanup_deferred", cleanup_error_type="NativeShutdownUnconfirmed")
                fixture.persist()


async def stop_application(app, provider, jobs, limits):
    """Both owners get a shutdown request even when either close raises."""
    receipt = {"session_and_worker_joined": False, "errors": []}
    try:
        async with asyncio.timeout(limits.cleanup_seconds):
            stopped = await asyncio.gather(app.session.close("test_cleanup"), app.runner.stop(), return_exceptions=True)
            receipt["errors"] = [type(error).__name__ for error in stopped if isinstance(error, BaseException)]
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            # Existing native stop owns process/gateway/private-copy cleanup.
            # Do not launch a new recovery worker or replay any model request.
            for run in tuple(provider._runs.values()):
                await run.stop()
            receipt["session_and_worker_joined"] = (not provider._runs and not app.runner._active
                and all(job.done() for job in jobs))
            receipt["provider_shutdown_confirmed"] = app.session.shutdown_confirmed
    except Exception as error:
        receipt["errors"].append(type(error).__name__)
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
    return receipt


def private_write(path, value, *, exclusive=True):
    raw = (json.dumps(value, ensure_ascii=True, indent=2, default=str, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC) | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "wb") as output:
        output.write(raw)


def read_json(path, maximum=8 * 1024 * 1024):
    require(not path.is_symlink() and path.stat().st_size <= maximum, "unsafe_or_oversized_input")
    return json.loads(path.read_bytes())


def private_root(path):
    path = Path(path).absolute()
    require(path.is_dir() and not path.is_symlink() and not path.stat().st_mode & 0o077,
            "private_run_directory_required")
    return path


def prepare(root, *, question=DEFAULT_QUESTION, busy_followup=None, continuation=None, speech_builder=None):
    """No credentials, network, microphone or speakers. Optional hooks are speech."""
    from evals.responsive_voice.speech_assets import prepare_speech_bundle
    limits = Limits().validate()
    script = [("work", question)]
    if busy_followup:
        script.append(("busy_followup", busy_followup))
    if continuation:
        script.append(("continuation", continuation))
    require(all(isinstance(text, str) and 0 < len(text) <= 500 for _, text in script), "invalid_script")
    root = Path(root).absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    mask = os.umask(0o077)
    try:
        speech = (speech_builder or prepare_speech_bundle)(root / "speech", script)
        manifest = dict(format=FORMAT, limits=asdict(limits), script=script, speech=speech,
            hooks={"busy_followup": "first_running_native_attempt", "continuation": "completed_result_and_idle_work"},
            first_scenario="generic_background_work", paid_execution=False,
            oracle="Review completed task result and actual native spoken result; never supplied as model context.",
            physical_hearing=False)
        private_write(root / "manifest.json", manifest)
        return manifest
    finally:
        os.umask(mask)


def validate_bundle(root):
    from evals.responsive_voice.speech_assets import inspect_pcm
    root = private_root(root)
    manifest = read_json(root / "manifest.json", 65536)
    require(manifest.get("format") == FORMAT, "wrong_manifest_format")
    limits = Limits(**manifest["limits"]).validate()
    script = manifest["script"]
    require(1 <= len(script) <= 3 and script[0][0] == "work", "invalid_script")
    identities = [row[0] for row in script]
    require(len(set(identities)) == len(identities) and set(identities) <= {"work", "busy_followup", "continuation"},
            "invalid_script_ids")
    clips = manifest["speech"]["clips"]
    require(len(clips) == len(script), "clip_count_mismatch")
    for clip, (identity, text) in zip(clips, script):
        require(clip["id"] == identity and clip["text"] == text and clip["relative_path"] == identity + ".wav",
                "clip_script_mismatch")
        path = root / "speech" / clip["relative_path"]
        require(not path.is_symlink() and path.stat().st_size <= RATE * 30 + 4096, "unsafe_speech_asset")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"], "speech_hash_mismatch")
        actual = inspect_pcm(path)
        require(all(clip.get(key) == actual[key] for key in actual), "speech_metadata_mismatch")
    return manifest, limits, clips


def scrub(value):
    """Keep all context text; omit credentials and SDP, not arbitrary prompt fields."""
    from evals.responsive_voice.model_wire_trace import sanitized_payload
    safe, _ = sanitized_payload(value)
    def walk(item):
        if isinstance(item, dict):
            return {key: "[transport or credential omitted]" if key.lower() in {
                "sdp", "answer_sdp", "offer_sdp", "token", "bearer_token", "secret", "password", "dsn"
            } else walk(child) for key, child in item.items()}
        if isinstance(item, list):
            return [walk(child) for child in item]
        return item
    return walk(safe)


class Trace:
    def __init__(self, root, limits):
        self.root, self.limits = root, limits
        self.stream = os.fdopen(os.open(root / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
        self.text = os.fdopen(os.open(root / "timeline.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
        self.bytes = self.sequence = self.live_creations = self.worker_requests = 0
        self.last_commentary = self.last_caption = None
        self.model_ids = set()
        self.closed = False

    def record(self, stage, *, raw=None, **fields):
        from evals.responsive_voice.model_wire_trace import readable
        payload = json.loads(raw) if raw is not None else fields.pop("payload", None)
        self.sequence += 1
        row = scrub(dict(sequence=self.sequence, utc=datetime.now(UTC).isoformat(),
                         monotonic_s=time.monotonic(), stage=stage, **fields))
        if payload is not None:
            row["payload"] = scrub(payload)
        encoded = json.dumps(row, ensure_ascii=True, default=str, allow_nan=False) + "\n"
        readable_row = "\n".join(readable(row)) + "\n\n"
        self.bytes += len(encoded.encode()) + len(readable_row.encode())
        require(self.bytes <= self.limits.max_trace_bytes, "trace_budget_exceeded")
        self.stream.write(encoded)
        self.stream.flush()
        self.text.write(readable_row)
        self.text.flush()
        if isinstance(payload, dict):
            if stage == "send_returned" and fields.get("kind") == "session.commentary.append":
                self.last_commentary = row["monotonic_s"]
            if stage == "receive" and payload.get("type") == "session.output_transcript.delta":
                self.last_caption = row["monotonic_s"]
        return self.sequence

    def close(self):
        if not self.closed:
            self.closed = True
            self.stream.close()
            self.text.close()


class ObservedSocket:
    def __init__(self, socket, trace):
        self.socket, self.trace = socket, trace

    def __getattr__(self, name):
        return getattr(self.socket, name)

    async def send(self, raw):
        event = json.loads(raw)
        sequence = self.trace.record("send_attempt", raw=raw, source="live")
        result = await self.socket.send(raw)
        self.trace.record("send_returned", payload=event, source="live", attempt_sequence=sequence,
                          kind=event.get("type"))
        return result

    async def recv(self):
        raw = await self.socket.recv()
        self.trace.record("receive", raw=raw, source="live")
        return raw

    async def close(self):
        return await self.socket.close()


@contextmanager
def observe_live(trace):
    """Observe normal production HTTP/socket calls; never replace their results."""
    import httpx
    from app.voice_routes import live_transport
    original_send, original_connect = httpx.AsyncClient.send, live_transport.connect

    async def send(client, request, *args, **kwargs):
        matched = request.url.host == "api.openai.com" and request.url.path == "/v1/live/sessions" and request.method == "POST"
        if matched:
            trace.live_creations += 1
            require(trace.live_creations == 1, "live_creation_retry_forbidden")
            body = json.loads(await request.aread())
            require(body.get("session", {}).get("model") == "gpt-live-1", "wrong_live_model")
            require(body["session"].get("delegation") == {"type": "client"}, "client_delegation_required")
            trace.record("creation_send_attempt", payload=body, source="live")
        response = await original_send(client, request, *args, **kwargs)
        if matched:
            trace.record("creation_returned", http_status=response.status_code, source="live")
        return response

    async def connect(*args, **kwargs):
        return ObservedSocket(await original_connect(*args, **kwargs), trace)

    with patch.object(httpx.AsyncClient, "send", send), patch.object(live_transport, "connect", connect):
        yield


class ObservedUpstream:
    def __init__(self, upstream, trace):
        self.upstream, self.trace = upstream, trace

    async def count(self, payload):
        self.trace.record("worker_count_send", payload=payload, source="hermes")
        count = await self.upstream.count(payload)
        self.trace.record("worker_count_returned", count=count, source="hermes")
        return count

    async def events(self, payload):
        self.trace.worker_requests += 1
        require(self.trace.worker_requests <= self.trace.limits.max_model_requests, "worker_request_budget_exceeded")
        self.trace.model_ids.add(payload.get("model"))
        self.trace.record("worker_request", payload=payload, source="hermes")
        async for event in self.upstream.events(payload):
            # Retain ordinary output/tool events, not private reasoning output.
            if "reasoning" not in str(event.data.get("type", "")):
                value = dict(event.data)
                if isinstance(value.get("response"), dict):
                    value["response"] = {**value["response"], "output": [item for item in value["response"].get("output", [])
                        if item.get("type") != "reasoning"]}
                self.trace.record("worker_response", payload=value, source="hermes")
            yield event

    async def aclose(self):
        await self.upstream.aclose()


# The same Browser/CDP process helper and PCM48k speech as the existing replay.
# Only hardware input/output is substituted. Control receipts follow actual
# browser operations, never a synthetic claim that a media operation succeeded.
BROWSER_PROGRAM = r"""(() => {
 const rate=48000, maxSamples=240*rate, state={active:false, closed:false, fatal:null,
   inputs:[], packets:[], firstSignal:null,lastSignal:null, mediaEpoch:null};
 let ac,pc,node,input,remote,playback,gate,clips={},pcm=[],received=0;
 const module=`class Capture extends AudioWorkletProcessor {
   process(inputs,outputs) {const a=inputs[0]?.[0];
     if(a) this.port.postMessage({frame:currentFrame,samples:a});
     return true;
   }} registerProcessor('capture',Capture);`;
 async function prepare() {
   ac=new AudioContext({sampleRate:rate}); await ac.resume();
   if(ac.sampleRate!==rate) throw Error('sample_rate_mismatch');
   const url=URL.createObjectURL(new Blob([module],{type:'application/javascript'}));
   await ac.audioWorklet.addModule(url); URL.revokeObjectURL(url);
   node=new AudioWorkletNode(ac,'capture'); gate=ac.createGain();gate.gain.value=0;
   node.connect(gate);gate.connect(ac.destination);
   node.port.onmessage=e=>{if(!state.active)return;const {frame,samples}=e.data;
     if(frame+samples.length>maxSamples){state.fatal='audio_capture_budget';pc.close();return;}
     const out=new Int16Array(samples.length);let first=null,last=null;
     for(let i=0;i<samples.length;i++){out[i]=Math.round(Math.max(-1,Math.min(1,samples[i]))*32767);
       if(Math.abs(out[i])>=128){if(first===null)first=frame+i;last=frame+i;}}
     if(first!==null){if(state.firstSignal===null)state.firstSignal=first;state.lastSignal=last;}
     pcm.push({frame,data:out});received+=out.length;
   };
   input=ac.createMediaStreamDestination();pc=new RTCPeerConnection();
   input.stream.getTracks().forEach(t=>{t.enabled=false;pc.addTrack(t,input.stream);});
   pc.ontrack=e=>{const stream=e.streams[0]||new MediaStream([e.track]);
     // Chromium needs a playback sink to pull remote WebRTC audio. Keep it
     // muted: the worklet captures before the separate zero-gain output sink.
     playback=document.createElement('audio');playback.autoplay=true;playback.muted=true;
     playback.srcObject=stream;document.body.append(playback);
     playback.play().catch(()=>{state.fatal='remote_playback_start_failed';});
     remote=ac.createMediaStreamSource(stream);remote.connect(node);};
   pc.onconnectionstatechange=()=>{if(pc.connectionState==='failed')state.fatal='peer_failed';};
   const dc=pc.createDataChannel('oai-events');dc.onmessage=e=>{
     try{if(state.packets.length>=2048)throw Error();state.packets.push(JSON.parse(e.data));}
     catch{state.fatal='data_channel_capture_failed';}};
   await pc.setLocalDescription(await pc.createOffer());
   if(pc.iceGatheringState!=='complete')await new Promise((resolve,reject)=>{
     const timer=setTimeout(()=>reject(Error('offer_ice_timeout')),10000);
     pc.addEventListener('icegatheringstatechange',()=>{if(pc.iceGatheringState==='complete'){clearTimeout(timer);resolve();}});});
   state.mediaEpoch={performance_ms:performance.now(),audio_seconds:ac.currentTime};
   return {sdp:pc.localDescription.sdp,epoch:state.mediaEpoch};
 }
 async function answer(sdp){await pc.setRemoteDescription({type:'answer',sdp});
   if(pc.connectionState!=='connected')await new Promise((resolve,reject)=>{
     const timer=setTimeout(()=>reject(Error('media_connection_timeout')),10000);
     pc.addEventListener('connectionstatechange',()=>{if(pc.connectionState==='connected'){clearTimeout(timer);resolve();}});});
   return {provider_started:pc.connectionState==='connected',output_gated:true};}
 function activate(){if(pc.connectionState!=='connected'||state.closed)throw Error('not_connected');
   state.active=true;input.stream.getTracks().forEach(t=>t.enabled=true);
   return {provider_started:true,output_gated:false};}
 async function load(id,encoded){const raw=Uint8Array.from(atob(encoded),x=>x.charCodeAt(0));
   clips[id]=await ac.decodeAudioData(raw.buffer);return true;}
 function play(id){if(!state.active||state.closed||!clips[id])throw Error('input_not_ready');
   const start=ac.currentTime+.05,s=ac.createBufferSource();s.buffer=clips[id];s.connect(input);s.start(start);
   const row={id,start_sample:Math.round(start*rate),end_sample:Math.round((start+s.buffer.duration)*rate),
     performance_ms:performance.now(),audio_seconds:ac.currentTime};state.inputs.push(row);return row;}
 function snapshot(){return {...state,packets:undefined,audio_sample:Math.floor(ac.currentTime*rate),received_samples:received};}
 function drain(){const rows=state.packets;state.packets=[];return rows;}
 function chunk(offset,count){const out=new Int16Array(count);for(const block of pcm){
   const begin=Math.max(offset,block.frame),end=Math.min(offset+count,block.frame+block.data.length);
   if(end>begin)out.set(block.data.subarray(begin-block.frame,end-block.frame),begin-offset);}
   const bytes=new Uint8Array(out.buffer);let binary='';for(let i=0;i<bytes.length;i+=8192)
     binary+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(binary);}
 async function fence(){state.active=false;state.closed=true;input?.stream.getTracks().forEach(t=>t.stop());
   pc?.getReceivers().forEach(r=>r.track?.stop());pc?.close();remote?.disconnect();
   if(playback){playback.pause();playback.srcObject=null;playback.remove();}
   return {output_fenced:pc?.connectionState==='closed',observed_at:new Date().toISOString()};}
 window.acceptance={prepare,answer,activate,load,play,snapshot,drain,chunk,fence};
 window.cleanup=async()=>{await fence();await ac?.close();return {closed:true};};return true;
})()"""


class BrowserPeer:
    """In-process app WebSocket endpoint; all media still crosses real WebRTC."""
    def __init__(self, browser, trace):
        self.browser, self.trace = browser, trace
        self.queue, self.lock = asyncio.Queue(maxsize=64), asyncio.Lock()
        self.closed = False

    async def js(self, expression):
        async with self.lock:
            return await asyncio.wait_for(self.browser.cdp.evaluate(expression), 16)

    async def receive_text(self):
        event = await self.queue.get()
        if event is None:
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect(1000)
        return json.dumps(event)

    async def respond(self, event):
        self.trace.record("app_client_send", payload=event)
        self.queue.put_nowait(event)

    async def send_json(self, event):
        self.trace.record("app_server_send", payload=event)
        kind = event.get("type")
        if kind == "voice_route.selected":
            await self.respond({key: value for key, value in event.items() if key != "capabilities"} | {"type": "voice_route.ready"})
        elif kind in {"live.media.prepare", "live.media.activate"}:
            result = await self.js("acceptance.answer(" + json.dumps(event["answer_sdp"]) + ")" if kind.endswith("prepare")
                                   else "acceptance.activate()")
            require(result.get("provider_started") is True and result.get("output_gated") is kind.endswith("prepare"),
                    "actual_media_operation_failed")
            await self.respond({key: value for key, value in event.items() if key != "answer_sdp"} |
                dict(type="live.media.receipt", phase=kind.rsplit(".", 1)[1], receipt_id=str(uuid4()), **result))
        elif kind in {"voice_route.close", "voice_route.interrupt"}:
            result = await self.js("acceptance.fence()")
            require(result.get("output_fenced") is True, "actual_output_fence_failed")
            command = event["payload"]
            await self.respond(dict(type="voice_route.event", protocol_version=1, audio_session_id=event["audio_session_id"],
                payload=dict(route=command["route"], command_id=command["command_id"],
                    operation="close" if kind.endswith("close") else "interrupt_speech", status="completed", **result)))

    async def send_text(self, text):
        await self.send_json(json.loads(text))

    async def close(self, **kwargs):
        self.closed = True
        self.queue.put_nowait(None)


async def canonical_snapshot(fixture, limits):
    result = {}
    async with fixture.pool.acquire() as connection:
        for table in TABLES:  # Identifiers are fixed, never supplied by a model.
            rows = await connection.fetch(f"select to_jsonb(t) as row from {table} t where owner_id=$1 limit $2",
                                          fixture.owner, limits.max_rows + 1)
            require(len(rows) <= limits.max_rows, "canonical_export_budget")
            result[table] = [json.loads(row["row"]) if isinstance(row["row"], str) else row["row"] for row in rows]
    return result


def state_facts(snapshot):
    tasks = snapshot.get("jarvis_background_tasks", [])
    attempts = snapshot.get("jarvis_task_attempts", [])
    committed = [row for row in tasks if row.get("status") == "succeeded" and row.get("result_payload_id")]
    running = [row for row in attempts if row.get("status") == "running" and row.get("provider_state") == "live"]
    idle = bool(tasks) and all(row.get("status") in {"succeeded", "waiting_for_input"} for row in tasks)
    return dict(completed_results=[str(row["result_payload_id"]) for row in committed],
                task_ids=[row["id"] for row in tasks], running=bool(running), idle=idle,
                task_states=[row.get("status") for row in tasks])


def check_application(app, fixture, session_id, limits):
    from app.realtime.live_session import LiveAudioSession
    from app.tasks.hermes import HermesTaskProvider
    from app.tasks.runner import TaskRunner
    require(isinstance(app.session, LiveAudioSession) and isinstance(app.runner, TaskRunner), "real_application_required")
    require(app.session.context.owner_id == fixture.owner and app.session.context.jarvis_session_id == session_id,
            "session_fixture_scope_mismatch")
    require(app.runner.owner_id == fixture.owner and app.runner.repository.pool is fixture.pool, "worker_fixture_scope_mismatch")
    require(not app.session.ready and not app.runner._active and not app.session._tasks, "construction_must_be_inert")
    require(callable(app.session.delegation_handler), "native_live_handler_required")
    require(set(app.runner.providers) == {"frontier"}, "only_frontier_worker_allowed")
    provider = app.runner.providers["frontier"]
    require(isinstance(provider, HermesTaskProvider), "real_hermes_required")
    for field in ("active_seconds", "tool_steps", "request_output_tokens", "task_output_tokens", "task_input_tokens", "max_attempts"):
        require(0 < getattr(app.runner.repository.limits, field) <= getattr(limits, field), "worker_budget_not_applied")
    require(app.runner.repository.limits.global_concurrency == 1 and app.runner.repository.limits.owner_concurrency == 1,
            "single_scoped_worker_required")
    return provider


async def collect_audio(peer, root, limits):
    state = await peer.js("acceptance.snapshot()")
    require(state.get("fatal") is None, "browser_capture_failed")
    count = state["audio_sample"]
    require(count <= 240 * RATE, "audio_export_budget")
    with wave.open(str(root / "output.wav"), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for offset in range(0, count, RATE):
            length = min(RATE, count - offset)
            encoded = await peer.js(f"acceptance.chunk({offset},{length})")
            raw = base64.b64decode(encoded, validate=True)
            require(len(raw) == length * 2, "audio_export_size_mismatch")
            output.writeframesraw(raw)
    (root / "output.wav").chmod(0o600)
    return state


async def drive(app, fixture, peer, trace, root, clips, limits, report):
    started = time.monotonic()
    deadline = started + limits.scenario_seconds
    report["scenario_started_monotonic_s"] = started
    played = set()
    baseline_operations = set()
    initial_ids = None
    previous = None
    while time.monotonic() < deadline:
        require(app.session.failure is None, "live_session_failed")
        state = await peer.js("acceptance.snapshot()")
        require(not state.get("fatal"), "browser_failed")
        for event in await peer.js("acceptance.drain()"):
            trace.record("client_data_channel", payload=event)
        snapshot = await canonical_snapshot(fixture, limits)
        facts = state_facts(snapshot)
        encoded = json.dumps(snapshot, sort_keys=True, default=str)
        if encoded != previous:
            trace.record("canonical_observation", payload=snapshot)
            previous = encoded
        if app.session.ready:
            for clip in clips:
                identity = clip["id"]
                if identity in played:
                    continue
                eligible = identity == "work" or (
                    identity == "busy_followup" and "work" in played and facts["running"]
                    and state["audio_sample"] > report["inputs"][0]["end_sample"]) or (
                    identity == "continuation" and facts["idle"] and facts["completed_results"]
                    and trace.last_commentary is not None and trace.last_caption is not None
                    and trace.last_caption > trace.last_commentary
                    and (state["lastSignal"] is not None and state["audio_sample"] - state["lastSignal"] > RATE))
                if not eligible:
                    continue
                before = time.monotonic()
                row = await peer.js("acceptance.play(" + json.dumps(identity) + ")")
                after = time.monotonic()
                row.update(last_active_sample=row["start_sample"] + clip["last_active_sample_exclusive"],
                           host_before_monotonic_s=before, host_after_monotonic_s=after,
                           observed_task_ids=facts["task_ids"], observed_running=facts["running"])
                report["inputs"].append(row)
                trace.record("speech_scheduled", payload=row, source_sha256=clip["sha256"])
                if identity == "continuation":
                    baseline_operations = set(facts["completed_results"])
                    initial_ids = set(facts["task_ids"])
                played.add(identity)
                break
        post_result = (trace.last_commentary is not None and trace.last_caption is not None
                       and trace.last_caption > trace.last_commentary and state["lastSignal"] is not None)
        if (facts["idle"] and set(facts["completed_results"]) - baseline_operations
                and len(played) == len(clips) and post_result and state["audio_sample"] - state["lastSignal"] > RATE):
            report["evidence_collected"] = True
            report["continuation_same_work"] = None if initial_ids is None else set(facts["task_ids"]) == initial_ids
            break
        await asyncio.sleep(.2)
    report["hooks"] = {clip["id"]: "injected" if clip["id"] in played else "not_observed_within_budget" for clip in clips}
    report["canonical"] = await canonical_snapshot(fixture, limits)
    report["audio"] = await collect_audio(peer, root, limits)
    report["useful_audio_latency_ms"] = None
    report["semantic_acceptance"] = "requires_audio_and_task_review"
    report["physical_hearing"] = False


async def run(root, *, factory, settings, execute_paid=False, fixture_factory=None):
    require(execute_paid is True, "explicit_paid_execution_required")
    root = private_root(root)
    manifest, limits, clips = validate_bundle(root)
    require(isinstance(factory, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*:[A-Za-z_][A-Za-z_0-9]*", factory),
            "explicit_factory_required")
    require(settings.OPENAI_LIVE_MODEL == "gpt-live-1", "wrong_live_model")
    require(bool(settings.SUPABASE_DB_URL) and bool(settings.OPENAI_API_KEY), "explicit_credentials_required")
    module, name = factory.split(":")
    construct = getattr(importlib.import_module(module), name)
    # Exclusive marker prevents replay of a partially completed paid run.
    private_write(root / "attempt.spent.json", dict(factory=factory, utc=datetime.now(UTC).isoformat()))
    from evals.responsive_voice.synthetic_webrtc import Browser
    from app.tasks.model_gateway import OpenAIResponsesHTTP
    trace = Trace(root, limits)
    report = dict(format=FORMAT, evidence_collected=False, inputs=[], cleanup={}, paid_execution=True,
                  boundaries=["desktop WebRTC, not iOS", "in-process app WebSocket, not HTTP authentication",
                              "no glasses, echo, Bluetooth, vision, or E2B", "decoded PCM, not heard audio"])
    browser = peer = None
    cleanup_gate = {"safe": True}
    try:
        async with (fixture_factory or isolated_fixture)(settings.SUPABASE_DB_URL, root / "fixtures",
                test_name="live_hermes_voice_acceptance", limits=limits, cleanup_gate=cleanup_gate) as fixture:
            report["fixture_manifest"] = str(fixture.fixture.path)
            report["fixture_owner"] = fixture.owner
            async with fixture.pool.acquire() as connection:
                session_id = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            report["session_id"] = str(session_id)
            (root / "browser").mkdir(mode=0o700)
            browser = Browser(root / "browser", root / "speech" / clips[0]["relative_path"])
            async with asyncio.timeout(limits.bootstrap_seconds):
                await browser.start()
                await browser.cdp.evaluate(BROWSER_PROGRAM)
                peer = BrowserPeer(browser, trace)
                offer = await peer.js("acceptance.prepare()")
                for clip in clips:
                    encoded = base64.b64encode((root / "speech" / clip["relative_path"]).read_bytes()).decode()
                    await peer.js("acceptance.load(" + json.dumps(clip["id"]) + "," + json.dumps(encoded) + ")")
                with observe_live(trace):
                    async with construct(fixture=fixture, session_id=session_id, websocket=peer,
                            offer=offer["sdp"], client_capabilities=dict(CAPABILITIES), limits=limits,
                            artifact_dir=root, settings=settings) as app:
                        provider = check_application(app, fixture, session_id, limits)
                        original_factory = provider.upstream_factory
                        def upstream():
                            real = original_factory()
                            require(isinstance(real, OpenAIResponsesHTTP), "real_worker_upstream_required")
                            return ObservedUpstream(real, trace)
                        provider.upstream_factory = upstream
                        jobs = []
                        try:
                            cleanup_gate["safe"] = False
                            jobs = [asyncio.create_task(app.session.run()), asyncio.create_task(app.runner.run())]
                            async with asyncio.timeout(limits.scenario_seconds + 25):
                                await drive(app, fixture, peer, trace, root, clips, limits, report)
                        finally:
                            try:
                                cleanup = await stop_application(app, provider, jobs, limits)
                                report["cleanup"].update(cleanup)
                                cleanup_gate["safe"] = cleanup["session_and_worker_joined"]
                                require(cleanup_gate["safe"], "worker_cleanup_unconfirmed")
                            finally:
                                provider.upstream_factory = original_factory
            report["cleanup"]["application_context_closed"] = True
        report["cleanup"]["fixture_cleaned"] = cleanup_gate["safe"]
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "code": str(error) if isinstance(error, HarnessError) else "see_private_timeline"}
        if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
            raise
    finally:
        if browser is not None:
            try:
                async with asyncio.timeout(limits.cleanup_seconds):
                    report["cleanup"]["browser"] = await browser.close()
            except Exception as error:
                report["cleanup"]["browser_error_type"] = type(error).__name__
        report["live_creations"], report["worker_model_requests"] = trace.live_creations, trace.worker_requests
        report["worker_models"] = sorted(trace.model_ids)
        trace.close()
        private_write(root / "result.json", report)
    return report


def review(root, *, useful_audio_sample, task_correct, spoken_result_correct):
    root = private_root(root)
    report = read_json(root / "result.json")
    require(report.get("evidence_collected") is True and not report.get("failure"), "run_incomplete")
    require(report.get("cleanup", {}).get("fixture_cleaned") is True
            and report["cleanup"].get("session_and_worker_joined") is True, "cleanup_incomplete")
    require(task_correct is True and spoken_result_correct is True, "explicit_semantic_review_required")
    require(type(useful_audio_sample) is int, "invalid_useful_audio_sample")
    last_input = report["inputs"][0]["last_active_sample"]
    require(useful_audio_sample > last_input, "useful_audio_precedes_request_end")
    with wave.open(str(root / "output.wav"), "rb") as audio:
        require(audio.getframerate() == RATE and useful_audio_sample < audio.getnframes(), "useful_audio_outside_capture")
        audio.setpos(useful_audio_sample)
        from array import array
        samples = array("h", audio.readframes(RATE // 10))
        if sys.byteorder != "little":
            samples.byteswap()
        require(any(abs(sample) >= 128 for sample in samples), "useful_audio_window_has_no_signal")
    result = dict(format=FORMAT, semantic_acceptance="reviewer_confirmed", task_correct=True,
        spoken_result_correct=True, useful_audio_sample=useful_audio_sample,
        useful_audio_latency_ms=(useful_audio_sample-last_input)*1000/RATE,
        boundary="same real-time browser audio clock; PCM energy end, not last-word alignment; no hearing claim")
    private_write(root / "review.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--busy-followup")
    p.add_argument("--continuation")
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    p.add_argument("--factory", required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--execute-paid", action="store_true")
    p = sub.add_parser("review")
    p.add_argument("root", type=Path)
    p.add_argument("--useful-audio-sample", type=int, required=True)
    p.add_argument("--task-correct", action="store_true")
    p.add_argument("--spoken-result-correct", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.root, question=args.question, busy_followup=args.busy_followup, continuation=args.continuation)
        elif args.command == "review":
            result = review(args.root, useful_audio_sample=args.useful_audio_sample,
                            task_correct=args.task_correct, spoken_result_correct=args.spoken_result_correct)
        else:
            require(args.execute_paid, "explicit_paid_execution_required")
            from app.settings import JarvisSettings
            settings = JarvisSettings(_env_file=args.env_file)
            result = asyncio.run(run(args.root, factory=args.factory, settings=settings, execute_paid=args.execute_paid))
        print(json.dumps({"format": FORMAT, "command": args.command, "evidence_collected": result.get("evidence_collected"),
                          "semantic_acceptance": result.get("semantic_acceptance", "not_evaluated"),
                          "failure": result.get("failure")}, default=str))
        return 1 if result.get("failure") or args.command == "run" and not result.get("evidence_collected") else 0
    except Exception as error:
        print(json.dumps({"failure_type": type(error).__name__,
                          "code": str(error) if isinstance(error, HarnessError) else "harness_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
