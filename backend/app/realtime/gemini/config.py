from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, is_configured_secret


@dataclass(frozen=True, slots=True)
class GeminiLiveConfig:
    """Run-scoped configuration. Construction never opens a provider connection."""

    api_key: str = field(repr=False)
    model: str = "gemini-3.1-flash-live-preview"
    voice: str = "Puck"
    input_sample_rate: int = 24000
    setup_timeout_seconds: float = 20.0
    write_timeout_seconds: float = 5.0
    close_timeout_seconds: float = 3.0
    preparation_timeout_seconds: float = 0.75
    response_timeout_seconds: float = 90.0
    max_input_chunk_bytes: int = 262144
    max_provider_message_bytes: int = 8 * 1024 * 1024
    receive_queue_size: int = 16

    def __post_init__(self) -> None:
        for value in (self.api_key, self.model, self.voice):
            if not is_configured_secret(value):
                raise ValueError("Gemini Live requires configured credentials, model, and voice")
        for value in (
            self.setup_timeout_seconds, self.write_timeout_seconds,
            self.close_timeout_seconds, self.preparation_timeout_seconds,
            self.response_timeout_seconds,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Gemini time budgets must be finite and positive")
        for value in (
            self.input_sample_rate, self.max_input_chunk_bytes,
            self.max_provider_message_bytes, self.receive_queue_size,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Gemini media limits must be positive integers")

    @classmethod
    def from_settings(cls, settings: JarvisSettings) -> GeminiLiveConfig:
        try:
            if settings.JARVIS_AUDIO_FORMAT != "pcm16":
                raise ValueError("Gemini requires PCM16 input")
            return cls(
                api_key=settings.GEMINI_API_KEY or "",
                model=settings.GEMINI_LIVE_MODEL or "",
                voice=settings.GEMINI_LIVE_VOICE,
                input_sample_rate=settings.JARVIS_AUDIO_SAMPLE_RATE,
                max_input_chunk_bytes=settings.JARVIS_REALTIME_MAX_AUDIO_CHUNK_BYTES,
                preparation_timeout_seconds=float(
                    settings.GEMINI_LIVE_PREPARATION_TIMEOUT_SECONDS
                ),
            )
        except (ValueError, TypeError) as error:
            raise JarvisHTTPError(
                503, "gemini_live_not_configured",
                "Gemini Live runtime configuration is invalid.",
                details={"provider": "gemini"},
            ) from None
