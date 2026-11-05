from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.models import RealtimeSessionContext
from app.realtime.openai_client import build_session_update_event
from app.realtime.session import RealtimeAudioSession
from app.settings import JarvisSettings


def _settings(**overrides: object) -> JarvisSettings:
    with patch.dict(os.environ, {}, clear=True):
        return JarvisSettings(
            _env_file=None,
            OPENAI_API_KEY="test-openai-key",
            OPENAI_REALTIME_MODEL="gpt-realtime-2.1",
            **overrides,
        )


class GPTSetupBaselineTests(unittest.TestCase):
    def test_unused_gemini_configuration_cannot_change_gpt_setup(self) -> None:
        tool = {
            "type": "function",
            "name": "fixture_lookup",
            "description": "Read an isolated fixture.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        }
        expected = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime-2.1",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                        "transcription": {"model": "gpt-realtime-whisper"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "cedar",
                    },
                },
                "instructions": "Unmodified GPT instructions.",
                "tools": [tool],
                "tool_choice": "auto",
            },
        }
        for overrides in (
            {},
            {"GEMINI_LIVE_ENABLED": False, "GEMINI_LIVE_MODEL": None},
            {
                "GEMINI_LIVE_ENABLED": "invalid-unused-flag",
                "GEMINI_LIVE_MODEL": "",
                "GEMINI_API_KEY": None,
                "AUDIO_PROVIDER_DEFAULT": "invalid-unused-default",
            },
        ):
            with self.subTest(overrides=overrides):
                event = build_session_update_event(
                    settings=_settings(**overrides),
                    instructions="Unmodified GPT instructions.",
                    tools=[tool],
                )
                self.assertEqual(event, expected)

    def test_empty_gpt_registry_stays_empty(self) -> None:
        event = build_session_update_event(
            settings=_settings(GEMINI_LIVE_ENABLED=True),
            instructions="GPT without tools.",
            tools=[],
        )
        self.assertEqual(event["session"]["tools"], [])
        self.assertEqual(event["session"]["tool_choice"], "none")


class GPTUnstartedCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_session_closes_once_before_provider_setup(self) -> None:
        events: list[dict[str, object]] = []

        async def send_text(raw: str) -> None:
            events.append(json.loads(raw))

        websocket = SimpleNamespace(
            send_text=AsyncMock(side_effect=send_text),
            send_json=AsyncMock(side_effect=events.append),
            close=AsyncMock(),
        )
        repository = SimpleNamespace(
            create_message=AsyncMock(),
            get_recent_messages=AsyncMock(return_value=[]),
        )
        session = RealtimeAudioSession(
            websocket=websocket,
            context=RealtimeSessionContext(
                audio_session_id="fixture-audio-run",
                owner_id="fixture-owner",
                jarvis_session_id=uuid4(),
                client_kind="rayban_audio",
                session=SimpleNamespace(),
            ),
            settings=_settings(GEMINI_LIVE_ENABLED="invalid-unused-flag"),
            repository=repository,
            tool_modules=(),
            context_providers=(),
        )
        with patch(
            "app.realtime.openai_client.websockets.connect",
            side_effect=AssertionError("Cleanup must not connect to a provider"),
        ):
            await asyncio.wait_for(
                asyncio.gather(
                    session.close("provider_selection_failed"),
                    session.close("provider_selection_failed"),
                ),
                timeout=1,
            )

        self.assertTrue(session._closed.is_set())
        self.assertEqual([event["type"] for event in events], ["session.closed"])
        websocket.close.assert_awaited_once()
        repository.create_message.assert_not_awaited()
        repository.get_recent_messages.assert_not_awaited()
