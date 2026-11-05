"""One owned Soloist receiver; bounded, authenticated PCM transport to the phone.

RTP L16/48000/2 packets use RFC4571 framing on the local helper pipe. Each
WebSocket binary message carries one complete RTP packet, without that prefix.
This service does not own microphone capture, Voice, recording, or a sandbox.
"""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import fcntl
import logging
import os
from pathlib import Path
import re
import struct
import sys
from time import monotonic
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, ConfigDict
import httpx

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer


class MediaError(RuntimeError):
    """Public error codes only: never propagate subprocess arguments or output."""


def validate_packet(packet: bytes) -> None:
    if not 16 <= len(packet) <= 3852 or (len(packet) - 12) % 4:
        raise MediaError("invalid_audio_packet")
    if packet[0] != 0x80 or packet[1] & 0x7f != 96:
        raise MediaError("invalid_audio_format")


@dataclass
class MediaLease:
    id: UUID
    session_id: UUID
    uri: str | None
    volume: float = 0.65
    state: str = "awaiting_phone"
    claimed: bool = False
    capture: asyncio.subprocess.Process | None = None
    requested_uri: str | None = None
    song_title: str | None = None
    artist: str | None = None
    playback_ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


async def terminate(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 2)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()


class SoloistMediaService:
    playback_wait_seconds = 2.0

    def __init__(self, *, owner_id: str, home: Path):
        self.owner_id = str(owner_id)
        self.home = home.resolve()
        self.binary = self.home / "bin/soloist"
        self.helper = self.home / "JarvisSoloistCapture.app/Contents/MacOS/JarvisSoloistCapture"
        self.process: asyncio.subprocess.Process | None = None
        self.lease: MediaLease | None = None
        self.lock = asyncio.Lock()
        self._ownership = None
        self._search_token = None
        self._search_token_until = 0.0

    @classmethod
    def configured(cls, owner_id: str):
        home = os.environ.get("JARVIS_SOLOIST_HOME")
        if not home or sys.platform != "darwin":
            return None
        return cls(owner_id=owner_id, home=Path(home))

    def snapshot(self, owner_id: str, session_id: UUID) -> dict | None:
        if str(owner_id) != self.owner_id:
            raise MediaError("media_owner_mismatch")
        lease = self.lease
        if lease is None or lease.session_id != session_id:
            return None
        return {"media_id": str(lease.id), "source_kind": "spotify",
                "state": lease.state, "base_volume": lease.volume,
                "state_basis": "phone_receipt_not_audibility",
                "capabilities": {"volume": True, "stop": True, "record_pcm": True}}

    async def _ctl(self, *arguments: str) -> bytes:
        process = await asyncio.create_subprocess_exec(
            str(self.binary), "ctl", *arguments, "--data-dir", str(self.home / "state"),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 5)
            if process.returncode != 0 or len(output) > 65536:
                raise MediaError("player_control_failed")
            return output
        finally:
            await terminate(process)

    async def _ensure_player(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        if not self.binary.is_file() or not self.helper.is_file():
            raise MediaError("media_runtime_not_installed")
        if self._ownership is None:
            ownership = open(self.home / "jarvis-media.lock", "a")
            try:
                fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                ownership.close()
                raise MediaError("media_runtime_in_use") from None
            self._ownership = ownership
        pid_file = self.home / "state/soloist.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if pid > 1:
                    os.kill(pid, 0)
                    raise MediaError("existing_receiver_requires_handoff")
            except (ValueError, ProcessLookupError):
                pass
        try:
            key = (self.home / "api-key").read_text().strip()
        except OSError:
            raise MediaError("soloist_key_unavailable") from None
        if not key or "placeholder" in key.lower():
            raise MediaError("soloist_key_unavailable")
        # Retain native login; never adopt/terminate an independently started receiver.
        self.process = await asyncio.create_subprocess_exec(
            str(self.binary), "-n", "Jarvis", "-k", key,
            "-D", str(self.home / "state"), "-C", str(self.home / "cache"),
            "-z", "100", "-i", "35", "-w", "127.0.0.1:0",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            for _ in range(30):
                if self.process.returncode is not None:
                    break
                try:
                    await self._ctl("status")
                    await self._ctl("pause")
                    return
                except MediaError:
                    await asyncio.sleep(0.2)
            raise MediaError("soloist_start_failed")
        except BaseException:
            await terminate(self.process)
            self.process = None
            raise

    async def search_track(self, song_title: str, artist: str | None) -> str:
        """Resolve a named track through Spotify, without an agent or browser."""
        try:
            async with asyncio.timeout(6), httpx.AsyncClient(timeout=3, follow_redirects=False) as client:
                if not self._search_token or monotonic() >= self._search_token_until:
                    try:
                        client_id = (self.home / "spotify-client-id").read_text().strip()
                        secret = (self.home / "spotify-client-secret").read_text().strip()
                    except OSError:
                        raise MediaError("Spotify song lookup credentials are not configured.") from None
                    if not client_id or not secret or secret.startswith("REPLACE_"):
                        raise MediaError("Spotify song lookup credentials are not configured.")
                    response = await client.post("https://accounts.spotify.com/api/token",
                        auth=(client_id, secret), data={"grant_type": "client_credentials"})
                    response.raise_for_status()
                    token = response.json()
                    if not isinstance(token.get("access_token"), str) or not token["access_token"]:
                        raise ValueError("invalid token response")
                    self._search_token = token["access_token"]
                    self._search_token_until = monotonic() + max(0, int(token["expires_in"]) - 60)
                title_filter = song_title.replace('"', ' ')
                query = f'track:"{title_filter}"'
                if artist:
                    artist_filter = artist.replace('"', ' ')
                    query += f' artist:"{artist_filter}"'
                response = await client.get("https://api.spotify.com/v1/search",
                    headers={"Authorization": "Bearer " + self._search_token},
                    params={"q": query, "type": "track", "limit": 10, "market": "IE"})
                if response.status_code == 401:
                    self._search_token = None
                response.raise_for_status()
                items = response.json()["tracks"]["items"]
                normalize = lambda value: "".join(c for c in value.casefold() if c.isalnum())
                for item in items:
                    if normalize(item["name"]) != normalize(song_title):
                        continue
                    if artist and not any(normalize(a["name"]) == normalize(artist) for a in item["artists"]):
                        continue
                    uri = item["uri"]
                    if re.fullmatch(r"spotify:track:[A-Za-z0-9]{22}", uri):
                        return uri
                raise MediaError("No matching Spotify track was found; please clarify the title or artist.")
        except (httpx.HTTPError, TimeoutError, KeyError, TypeError, ValueError):
            raise MediaError("Spotify song lookup failed. No new music was started.") from None

    async def start(self, *, session_id: UUID, media_id: UUID, uri: str | None,
                    song_title: str | None = None, artist: str | None = None) -> MediaLease:
        if uri is not None and re.fullmatch(r"spotify:(track|album|playlist):[A-Za-z0-9]{22}", uri) is None:
            raise MediaError("invalid_spotify_uri")
        if (uri and song_title) or (artist and not song_title):
            raise MediaError("Supply a song title and optional artist, or a Spotify URI, not both.")
        async with self.lock:
            if self.lease is not None and self.lease.id == media_id:
                if (self.lease.session_id != session_id or
                        (self.lease.requested_uri, self.lease.song_title, self.lease.artist) != (uri, song_title, artist)):
                    raise MediaError("media_command_conflict")
                return self.lease
            selected_uri = await self.search_track(song_title, artist) if song_title else uri
            await self._ensure_player()
            if self.lease is not None:
                self.lease.state = "replaced"
                self.lease.playback_ready.set()
                await self._ctl("pause")
                await terminate(self.lease.capture)
            self.lease = MediaLease(media_id, session_id, selected_uri,
                requested_uri=uri, song_title=song_title, artist=artist)
            return self.lease

    def current(self, session_id: UUID, media_id: UUID) -> MediaLease:
        lease = self.lease
        if lease is None or lease.id != media_id or lease.session_id != session_id:
            raise MediaError("media_not_current")
        return lease

    def phone_state(self, *, session_id: UUID, media_id: UUID, state: str) -> None:
        lease = self.current(session_id, media_id)
        # Separate HTTP receipts may arrive out of order. Queue admission cannot
        # undo an actual render receipt for this exact source.
        if state == "receiving" and lease.state == "playing":
            return
        lease.state = state
        if state in ("playing", "failed", "stopped"):
            lease.playback_ready.set()

    async def wait_for_playback(self, lease: MediaLease) -> str:
        # No playback/service lock is held here. The independent phone receipt
        # wakes this call; a slow phone does not hold conversation indefinitely.
        if self.lease is lease and not lease.playback_ready.is_set():
            try:
                await asyncio.wait_for(lease.playback_ready.wait(), self.playback_wait_seconds)
            except TimeoutError:
                pass
        if self.lease is not lease:
            return lease.state if lease.state in ("failed", "stopped", "replaced") else "replaced"
        return lease.state

    async def stop(self, *, session_id: UUID, media_id: UUID) -> bool:
        async with self.lock:
            if self.lease is None or self.lease.id != media_id or self.lease.session_id != session_id:
                return False
            lease, self.lease = self.lease, None
            if lease.state != "failed":
                lease.state = "stopped"
            lease.playback_ready.set()
            try:
                with contextlib.suppress(MediaError):
                    await self._ctl("pause")
            finally:
                await terminate(lease.capture)
            return True

    async def packets(self, *, session_id: UUID, media_id: UUID):
        async with self.lock:
            lease = self.current(session_id, media_id)
            if lease.claimed:
                raise MediaError("media_already_attached")
            lease.claimed = True
            try:
                assert self.process is not None
                # The native helper logs only capture setup/status, never music
                # or account credentials. Keep one bounded-lifetime diagnostic.
                with open(self.home / "capture.log", "wb") as diagnostic:
                    lease.capture = await asyncio.create_subprocess_exec(
                        str(self.helper), "--pid", str(self.process.pid), "--mute-local",
                        stdout=asyncio.subprocess.PIPE, stderr=diagnostic, limit=8192)
                await self._ctl("play", *([lease.uri] if lease.uri else []))
                lease.state = "sending"
            except BaseException:
                await terminate(lease.capture)
                lease.state = "failed"
                lease.playback_ready.set()
                raise
        try:
            assert lease.capture is not None and lease.capture.stdout is not None
            while self.lease is lease:
                async with asyncio.timeout(5):
                    prefix = await lease.capture.stdout.readexactly(2)
                    length, = struct.unpack("!H", prefix)
                    if not 16 <= length <= 3852:
                        raise MediaError("invalid_audio_packet")
                    packet = await lease.capture.stdout.readexactly(length)
                validate_packet(packet)
                yield packet
        finally:
            try:
                async with self.lock:
                    if self.lease is lease:
                        lease.state = "disconnected"
                        lease.playback_ready.set()
                        with contextlib.suppress(MediaError):
                            await self._ctl("pause")
            finally:
                await terminate(lease.capture)

    async def close(self) -> None:
        if self.lease is not None:
            await self.stop(session_id=self.lease.session_id, media_id=self.lease.id)
        await terminate(self.process)
        self.process = None
        if self._ownership is not None:
            self._ownership.close()
            self._ownership = None


router = APIRouter(prefix="/media-audio", tags=["media-audio"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]


def selected_service(connection, principal, session_id: UUID, media_id: UUID):
    service = getattr(connection.app.state, "media_audio", None)
    controls = getattr(getattr(connection.app.state, "runtime_services", None), "app_controls", None)
    if service is None or str(principal.owner_id) != service.owner_id:
        raise HTTPException(404, "Media unavailable")
    if controls is None or controls._active(session_id) is None:
        raise HTTPException(409, "Voice session is not active")
    try:
        service.current(session_id, media_id)
    except MediaError:
        raise HTTPException(409, "Media is no longer selected") from None
    return service


class MediaReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    state: Literal["receiving", "playing", "failed", "stopped"]


@router.post("/{media_id}/state")
async def media_state(media_id: UUID, body: MediaReceipt, request: Request, principal: Principal):
    service = selected_service(request, principal, body.session_id, media_id)
    service.phone_state(session_id=body.session_id, media_id=media_id, state=body.state)
    if body.state in ("failed", "stopped"):
        await service.stop(session_id=body.session_id, media_id=media_id)
    return {"accepted": True}


@router.websocket("/{media_id}/stream")
async def media_stream(websocket: WebSocket, media_id: UUID, session_id: UUID, principal: Principal):
    service = selected_service(websocket, principal, session_id, media_id)
    await websocket.accept()

    async def produce():
        async with contextlib.aclosing(service.packets(session_id=session_id, media_id=media_id)) as packets:
            async for packet in packets:
                selected_service(websocket, principal, session_id, media_id)
                await asyncio.wait_for(websocket.send_bytes(packet), 0.5)

    producer = asyncio.create_task(produce())
    disconnected = asyncio.create_task(websocket.receive())
    try:
        done, _ = await asyncio.wait((producer, disconnected), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (MediaError, OSError, asyncio.IncompleteReadError, TimeoutError, HTTPException) as error:
        logging.getLogger(__name__).warning("Media stream stopped: %s", type(error).__name__)
        with contextlib.suppress(RuntimeError, OSError):
            await websocket.close(code=1011, reason=str(error) if isinstance(error, MediaError) else type(error).__name__)
    finally:
        for task in (producer, disconnected):
            task.cancel()
        await asyncio.gather(producer, disconnected, return_exceptions=True)
        with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
            await websocket.close()
