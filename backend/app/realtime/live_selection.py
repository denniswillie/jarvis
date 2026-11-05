"""Explicit Live admission without importing or creating a provider transport."""
from dataclasses import dataclass
from collections.abc import Sequence

from pydantic import TypeAdapter, ValidationError

from app.errors import JarvisHTTPError
from app.realtime.provider_selection import AudioProvider
from app.settings import is_configured_secret


@dataclass(frozen=True, slots=True)
class LiveVoiceConfiguration:
    model: str
    voice: str


def _configured_identifier(value):
    return (isinstance(value, str) and 0 < len(value) <= 256
            and is_configured_secret(value)
            and not any(char.isspace() or ord(char) < 32 for char in value))


def select_live_voice_route(*, settings, requested_values: Sequence[str],
                            provider: AudioProvider, transport: str) -> LiveVoiceConfiguration | None:
    """None preserves existing provider routing; Live never silently falls back."""
    if len(requested_values) > 1 or (requested_values and requested_values[0] not in {"realtime", "live"}):
        raise JarvisHTTPError(400, "invalid_voice_route", "voice_route must be supplied once as realtime or live.")
    if not requested_values:
        return None
    if provider is not AudioProvider.OPENAI:
        raise JarvisHTTPError(400, "unsupported_voice_route_provider", "This voice route requires OpenAI audio.")
    if requested_values[0] == "realtime":
        return None
    if transport != "webrtc":
        raise JarvisHTTPError(400, "unsupported_live_transport", "Live requires direct WebRTC audio.")
    try:
        responsive = TypeAdapter(bool).validate_python(getattr(settings, "RESPONSIVE_VOICE_ENABLED", False))
        enabled = TypeAdapter(bool).validate_python(getattr(settings, "OPENAI_LIVE_ENABLED", False))
    except ValidationError:
        raise JarvisHTTPError(503, "live_voice_not_configured", "Live voice enablement is not configured correctly.") from None
    if not responsive or not enabled:
        raise JarvisHTTPError(503, "live_voice_disabled", "The requested Live voice route is not enabled.")
    key = getattr(settings, "OPENAI_API_KEY", None)
    model = getattr(settings, "OPENAI_LIVE_MODEL", None)
    voice = getattr(settings, "OPENAI_LIVE_VOICE", None)
    if (not isinstance(key, str) or not is_configured_secret(key)
            or not _configured_identifier(model) or not _configured_identifier(voice)):
        raise JarvisHTTPError(503, "live_voice_not_configured", "Live voice configuration is incomplete.")
    return LiveVoiceConfiguration(model=model, voice=voice)


def require_live_client_offer(*, capture_mode, client_capabilities):
    if capture_mode != "auto":
        raise JarvisHTTPError(400, "unsupported_live_capture_mode", "Live requires automatic conversational capture.")
    if not isinstance(client_capabilities, dict) or not client_capabilities:
        raise JarvisHTTPError(400, "live_voice_capabilities_required", "Live requires a compatible native route offer.")
    # Actual capability matching, gated-ready and media receipts are enforced by
    # LiveClientChannel before provider creation. An object is not that proof.
