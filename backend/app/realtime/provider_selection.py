from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, is_configured_secret

if TYPE_CHECKING:
    from app.realtime.gemini.config import GeminiLiveConfig


class AudioProvider(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True, slots=True)
class AudioProviderSelection:
    provider: AudioProvider
    model: str | None
    explicit: bool
    gemini_config: GeminiLiveConfig | None = field(default=None, repr=False, compare=False)

    def as_event(self, *, audio_session_id: str) -> dict[str, object]:
        """Routing acknowledgement only; session.created still establishes readiness."""
        return {
            "type": "audio.provider.selected",
            "protocol_version": 1,
            "audio_session_id": audio_session_id,
            "provider": self.provider.value,
            "model": self.model,
        }


def select_audio_provider(
    *, settings: JarvisSettings, requested_values: Sequence[str],
) -> AudioProviderSelection:
    """Select an available provider without initializing either transport.

    Gemini configuration is validated and frozen only on its selected branch.
    OpenAI retains its existing configuration validation in the audio router.
    """
    if len(requested_values) > 1:
        raise JarvisHTTPError(
            400,
            "invalid_audio_provider",
            "audio_provider must be supplied at most once.",
        )

    explicit = bool(requested_values)
    value = requested_values[0] if explicit else settings.AUDIO_PROVIDER_DEFAULT
    try:
        provider = AudioProvider(value)
    except (ValueError, TypeError):
        if explicit:
            raise JarvisHTTPError(
                400,
                "invalid_audio_provider",
                "audio_provider must be openai or gemini.",
            ) from None
        raise JarvisHTTPError(
            503,
            "audio_provider_default_invalid",
            "The default audio provider is not configured correctly.",
        ) from None

    if provider is AudioProvider.GEMINI:
        require_gemini_live_enabled(settings)
        if not is_configured_secret(settings.GEMINI_API_KEY) or not is_configured_secret(
            settings.GEMINI_LIVE_MODEL
        ):
            raise JarvisHTTPError(
                503, "gemini_live_not_configured", "Gemini Live audio is not configured.",
                details={"provider": "gemini", "reason": "missing_configuration"},
            )
        # A GPT request never imports or constructs Live configuration.
        from app.realtime.gemini.config import GeminiLiveConfig

        config = GeminiLiveConfig.from_settings(settings)
        return AudioProviderSelection(
            provider=provider, model=config.model, explicit=explicit, gemini_config=config,
        )

    return AudioProviderSelection(
        provider=provider,
        model=settings.OPENAI_REALTIME_MODEL,
        explicit=explicit,
    )


def require_gemini_live_enabled(settings: JarvisSettings) -> None:
    try:
        enabled = TypeAdapter(bool).validate_python(settings.GEMINI_LIVE_ENABLED)
    except ValidationError:
        raise JarvisHTTPError(
            503,
            "gemini_live_not_configured",
            "Gemini Live enablement is not configured correctly.",
            details={"provider": "gemini", "reason": "invalid_enablement"},
        ) from None

    if not enabled:
        raise JarvisHTTPError(
            503,
            "audio_provider_unavailable",
            "Gemini Live audio is disabled.",
            details={"provider": "gemini", "reason": "disabled"},
        )
