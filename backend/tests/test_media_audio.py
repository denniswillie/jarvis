import asyncio
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.app_controls import AppCommand, AppControlService
from app.media_audio import (MediaError, MediaLease, MediaReceipt, SoloistMediaService,
                            media_state, selected_service, validate_packet)


def packet():
    return struct.pack("!BBHII", 0x80, 0xe0, 0, 0, 0x534f4c4f) + struct.pack("!hh", 8192, -8192) * 48


def test_packet_and_action_boundaries():
    validate_packet(packet())
    for invalid in (b"", packet()[:-1], bytes([0x90]) + packet()[1:], packet()[:1] + b"\x61" + packet()[2:]):
        with pytest.raises(MediaError):
            validate_packet(invalid)
    AppCommand(action="recording.start", include_jarvis_audio=False)
    AppCommand(action="media.start")
    AppCommand(action="media.start", spotify_uri="spotify:track:" + "a" * 22)
    AppCommand(action="media.volume", media_id=uuid4(), volume=0.2)
    for bad in (
        {"action": "media.start", "include_jarvis_audio": True},
        {"action": "media.start", "spotify_uri": "https://untrusted.example/audio"},
        {"action": "media.stop"}, {"action": "media.volume", "media_id": uuid4(), "volume": 2},
        {"action": "recording.start", "include_jarvis_audio": False, "media_id": uuid4()},
    ):
        with pytest.raises(ValidationError):
            AppCommand(**bad)


@pytest.mark.asyncio
async def test_start_is_idempotent_and_old_stop_cannot_close_replacement(tmp_path):
    service = SoloistMediaService(owner_id="owner", home=tmp_path)
    service._ensure_player = AsyncMock()
    service._ctl = AsyncMock(return_value=b"")
    session = uuid4()
    first = await service.start(session_id=session, media_id=uuid4(), uri=None)
    assert await service.start(session_id=session, media_id=first.id, uri=None) is first
    with pytest.raises(MediaError):
        await service.start(session_id=uuid4(), media_id=first.id, uri=None)
    second = await service.start(session_id=session, media_id=uuid4(), uri=None)
    assert not await service.stop(session_id=session, media_id=first.id)
    assert service.current(session, second.id) is second
    with pytest.raises(MediaError):
        service.current(uuid4(), second.id)
    assert await service.stop(session_id=session, media_id=second.id)


@pytest.mark.asyncio
async def test_rfc4571_pipe_decode_and_disconnect_pause(tmp_path):
    service = SoloistMediaService(owner_id="owner", home=tmp_path)
    service.process = SimpleNamespace(pid=123)
    service._ctl = AsyncMock(return_value=b"")
    service.lease = lease = MediaLease(uuid4(), uuid4(), None)
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!H", len(packet())) + packet())
    process = SimpleNamespace(stdout=reader, returncode=0)
    with patch("app.media_audio.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn:
        stream = service.packets(session_id=lease.session_id, media_id=lease.id)
        assert await anext(stream) == packet()
        assert "--mute-local" in spawn.call_args.args
        with pytest.raises(MediaError):
            await anext(service.packets(session_id=lease.session_id, media_id=lease.id))
        await stream.aclose()
    assert lease.state == "disconnected"
    assert service._ctl.call_args.args == ("pause",)


@pytest.mark.asyncio
async def test_rejected_phone_start_releases_selected_media(tmp_path):
    media = SoloistMediaService(owner_id="owner", home=tmp_path)
    media._ensure_player = AsyncMock()
    media._ctl = AsyncMock(return_value=b"")
    control = AppControlService(owner_id="owner", pool=None, sandbox_views=None, presentations=None, media_audio=media)
    session = uuid4()
    control._application = SimpleNamespace(context=SimpleNamespace(jarvis_session_id=session),
        is_current=lambda: True, is_ready=lambda: True)
    control._notify = lambda *_: False
    result = await control._dispatch("owner", session, uuid4(), AppCommand(action="media.start"))
    assert result["outcome"] == "unavailable"
    assert media.lease is None


@pytest.mark.asyncio
async def test_controls_use_same_lease_and_only_project_ram(tmp_path):
    media = SoloistMediaService(owner_id="owner", home=tmp_path)
    media.playback_wait_seconds = 0
    media._ensure_player = AsyncMock()
    media._ctl = AsyncMock(return_value=b"")
    control = AppControlService(owner_id="owner", pool=None, sandbox_views=None, presentations=None, media_audio=media)
    session, operation = uuid4(), uuid4()
    events = []
    control._application = SimpleNamespace(context=SimpleNamespace(jarvis_session_id=session, audio_session_id="audio"),
        is_current=lambda: True, is_ready=lambda: True, snapshot=lambda: {},
        offer_event=lambda event: events.append(event) or True)
    result = await control._dispatch("owner", session, operation, AppCommand(action="media.start"))
    assert result["outcome"] == "media_requested"
    assert events[0]["session_id"] == str(session)
    assert control.snapshot("owner", session)["media"]["state"] == "awaiting_phone"
    await control._dispatch("owner", session, uuid4(), AppCommand(action="media.volume", media_id=operation, volume=0.1))
    assert events[-1]["action"] == "volume"
    assert media.lease.volume == 0.1
    await control._dispatch("owner", session, uuid4(), AppCommand(action="media.stop", media_id=operation))
    assert events[-1]["action"] == "stop"
    assert media.lease is None


def playback_control(tmp_path):
    media = SoloistMediaService(owner_id="owner", home=tmp_path)
    media._ensure_player = AsyncMock()
    media._ctl = AsyncMock(return_value=b"")
    control = AppControlService(owner_id="owner", pool=None, sandbox_views=None,
                                presentations=None, media_audio=media)
    session = uuid4()
    notified = asyncio.Event()
    control._application = SimpleNamespace(
        context=SimpleNamespace(jarvis_session_id=session, audio_session_id="audio"),
        is_current=lambda: True, is_ready=lambda: True,
        offer_event=lambda event: notified.set() or True)
    return media, control, session, notified


@pytest.mark.asyncio
async def test_start_waits_for_render_not_queue_and_returns_confirmation(tmp_path):
    media, control, session, notified = playback_control(tmp_path)
    operation = uuid4()
    pending = asyncio.create_task(control._dispatch("owner", session, operation, AppCommand(action="media.start")))
    await asyncio.wait_for(notified.wait(), 1)
    media.phone_state(session_id=session, media_id=operation, state="receiving")
    assert not pending.done()
    assert not media.lease.playback_ready.is_set()
    media.phone_state(session_id=session, media_id=operation, state="playing")
    result = await asyncio.wait_for(pending, 1)
    assert result["outcome"] == "media_playing"
    assert result["playback_confirmed"] is True
    assert result["summary"] == "Playing music."
    media.phone_state(session_id=session, media_id=operation, state="receiving")
    assert media.lease.state == "playing"


@pytest.mark.asyncio
async def test_early_confirmation_is_not_lost_and_repeat_does_not_restart(tmp_path):
    media, control, session, _ = playback_control(tmp_path)
    operation = uuid4()
    lease = await media.start(session_id=session, media_id=operation, uri=None)
    media.phone_state(session_id=session, media_id=operation, state="playing")
    with patch("app.media_audio.asyncio.wait_for", side_effect=AssertionError("Already confirmed")):
        result = await control._dispatch("owner", session, operation, AppCommand(action="media.start"))
    assert result["outcome"] == "media_playing"
    assert media.lease is lease
    media._ensure_player.assert_awaited_once()


@pytest.mark.asyncio
async def test_unconfirmed_start_is_bounded_and_does_not_claim_playback(tmp_path):
    media, control, session, _ = playback_control(tmp_path)
    media.playback_wait_seconds = 0.01
    result = await asyncio.wait_for(control._dispatch("owner", session, uuid4(), AppCommand(action="media.start")), 1)
    assert result["outcome"] == "media_requested"
    assert result["playback_confirmed"] is False
    assert result["summary"] == "Starting music."
    assert media.lease.state == "awaiting_phone"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["stop", "replace", "fail"])
async def test_stopped_failed_or_replaced_start_never_confirms(tmp_path, ending):
    media, control, session, notified = playback_control(tmp_path)
    operation = uuid4()
    pending = asyncio.create_task(control._dispatch("owner", session, operation, AppCommand(action="media.start")))
    await asyncio.wait_for(notified.wait(), 1)
    if ending == "replace":
        replacement = await media.start(session_id=session, media_id=uuid4(), uri=None)
    else:
        if ending == "fail":
            media.phone_state(session_id=session, media_id=operation, state="failed")
        await media.stop(session_id=session, media_id=operation)
    result = await asyncio.wait_for(pending, 1)
    assert result["outcome"] == "unavailable"
    if ending == "replace":
        assert media.lease is replacement
        with pytest.raises(MediaError):
            media.phone_state(session_id=session, media_id=operation, state="playing")
        assert not replacement.playback_ready.is_set()


@pytest.mark.asyncio
async def test_phone_endpoint_accepts_render_receipt_for_exact_current_lease(tmp_path):
    media, controls, session, _ = playback_control(tmp_path)
    lease = await media.start(session_id=session, media_id=uuid4(), uri=None)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        media_audio=media, runtime_services=SimpleNamespace(app_controls=controls))))
    receipt = MediaReceipt(session_id=session, state="playing")
    assert await media_state(lease.id, receipt, request, SimpleNamespace(owner_id="owner")) == {"accepted": True}
    assert await media.wait_for_playback(lease) == "playing"
    with pytest.raises(HTTPException):
        await media_state(uuid4(), receipt, request, SimpleNamespace(owner_id="owner"))


def test_stream_access_requires_exact_owner_active_session_and_selection(tmp_path):
    media = SoloistMediaService(owner_id="owner", home=tmp_path)
    media.lease = MediaLease(uuid4(), uuid4(), None)
    controls = SimpleNamespace(_active=lambda sid: object() if sid == media.lease.session_id else None)
    connection = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(media_audio=media,
        runtime_services=SimpleNamespace(app_controls=controls))))
    owner = SimpleNamespace(owner_id="owner")
    assert selected_service(connection, owner, media.lease.session_id, media.lease.id) is media
    for principal, session, selection in ((SimpleNamespace(owner_id="other"), media.lease.session_id, media.lease.id),
                                        (owner, uuid4(), media.lease.id), (owner, media.lease.session_id, uuid4())):
        with pytest.raises(HTTPException):
            selected_service(connection, principal, session, selection)


@pytest.mark.asyncio
async def test_pause_precedes_removing_process_mute(tmp_path):
    service = SoloistMediaService(owner_id="owner", home=tmp_path)
    service.lease = lease = MediaLease(uuid4(), uuid4(), None)
    events = []
    async def control(*arguments):
        events.append(arguments[0])
    async def remove_tap(process):
        events.append("remove_tap")
    service._ctl = control
    with patch("app.media_audio.terminate", remove_tap):
        await service.stop(session_id=lease.session_id, media_id=lease.id)
    assert events == ["pause", "remove_tap"]


@pytest.mark.asyncio
async def test_native_helper_uses_same_packet_format():
    helper = Path(__file__).resolve().parents[2] / ".jarvis-dev/soloist/JarvisSoloistCapture.app/Contents/MacOS/JarvisSoloistCapture"
    if not helper.is_file():
        pytest.skip("Native macOS helper not built")
    process = await asyncio.create_subprocess_exec(str(helper), "--self-test", stdout=asyncio.subprocess.PIPE)
    result, _ = await asyncio.wait_for(process.communicate(), 3)
    assert process.returncode == 0
    assert result == struct.pack("!H", len(packet())) + packet()
