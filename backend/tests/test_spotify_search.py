from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from app.app_controls import AppCommand, AppControlService
from app.media_audio import MediaError, SoloistMediaService


URI = "spotify:track:" + "a" * 22


def mock_search(tmp_path, items):
    (tmp_path / "spotify-client-id").write_text("test-client")
    (tmp_path / "spotify-client-secret").write_text("test-secret")
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.host == "accounts.spotify.com":
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"tracks": {"items": items}})

    client = httpx.AsyncClient
    return calls, patch("app.media_audio.httpx.AsyncClient", side_effect=lambda **kw:
        client(transport=httpx.MockTransport(respond), **kw))


@pytest.mark.asyncio
async def test_named_playback_calls_search_and_player_without_a_worker(tmp_path):
    service = SoloistMediaService(owner_id="owner", home=tmp_path)
    service.playback_wait_seconds = 0
    service._ensure_player = AsyncMock()
    service._ctl = AsyncMock()
    calls, mocked = mock_search(tmp_path, [
        {"name": "Lost In Japan - Remix", "artists": [{"name": "Shawn Mendes"}], "uri": URI},
        {"name": "Lost In Japan", "artists": [{"name": "Shawn Mendes"}], "uri": URI},
    ])
    control = AppControlService(owner_id="owner", pool=None, sandbox_views=None,
        presentations=None, media_audio=service)
    session, operation = uuid4(), uuid4()
    events = []
    control._application = SimpleNamespace(context=SimpleNamespace(jarvis_session_id=session, audio_session_id="test-audio"),
        is_current=lambda: True, is_ready=lambda: True,
        offer_event=lambda event: events.append(event) or True)
    command = AppCommand(action="media.start", song_title="Lost in Japan", artist="Shawn Mendes")
    with mocked:
        result = await control._dispatch("owner", session, operation, command)
        repeated = await control._dispatch("owner", session, operation, command)
        await service.search_track("Lost in Japan", "Shawn Mendes")
    assert result["outcome"] == repeated["outcome"] == "media_requested"
    assert service.lease.uri == URI
    assert service._ensure_player.await_count == 1
    assert len(calls) == 3  # one token request; duplicate start did not search again
    assert calls[1].url.params["q"] == 'track:"Lost in Japan" artist:"Shawn Mendes"'
    assert events[0]["type"] == "media.command"
    with pytest.raises(MediaError, match="media_command_conflict"):
        await service.start(session_id=session, media_id=operation, uri=None, song_title="Different song")


@pytest.mark.asyncio
async def test_missing_or_wrong_artist_does_not_start_a_player(tmp_path):
    service = SoloistMediaService(owner_id="owner", home=tmp_path)
    service._ensure_player = AsyncMock()
    _, mocked = mock_search(tmp_path, [
        {"name": "Lost In Japan", "artists": [{"name": "Someone else"}], "uri": URI}])
    with mocked, pytest.raises(MediaError, match="No matching"):
        await service.start(session_id=uuid4(), media_id=uuid4(), uri=None,
            song_title="Lost In Japan", artist="Shawn Mendes")
    service._ensure_player.assert_not_awaited()
    assert service.lease is None


@pytest.mark.asyncio
async def test_missing_credentials_is_a_plain_error(tmp_path):
    service = SoloistMediaService(owner_id="owner", home=tmp_path)
    with pytest.raises(MediaError, match="credentials are not configured"):
        await service.search_track("A song", None)


@pytest.mark.parametrize("arguments", [
    {"action": "media.start", "song_title": " "},
    {"action": "media.start", "artist": "Shawn Mendes"},
    {"action": "media.start", "song_title": "Lost in Japan", "spotify_uri": URI},
    {"action": "recording.start", "include_jarvis_audio": False, "song_title": "Lost in Japan"},
])
def test_named_song_fields_are_unambiguous(arguments):
    with pytest.raises(ValidationError):
        AppCommand(**arguments)
