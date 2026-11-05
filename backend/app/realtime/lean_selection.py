"""Explicit session-start selection. Never fall back after a partial call."""
from app.errors import JarvisHTTPError


LEAN_MODE = "lean_v1"


def select_lean_voice(*, settings, requested_values, provider, transport, live_selected=False):
    if not requested_values:
        return False
    if len(requested_values) != 1 or requested_values[0] not in {"legacy", LEAN_MODE}:
        raise JarvisHTTPError(400, "invalid_conversation_mode", "Select one supported conversation mode.")
    if requested_values[0] == "legacy":
        return False
    if provider != "openai" or transport != "webrtc" or live_selected:
        raise JarvisHTTPError(400, "lean_transport_required", "Lean voice requires direct OpenAI Realtime audio.")
    if getattr(settings, "LEAN_VOICE_ENABLED", False) is not True:
        raise JarvisHTTPError(503, "lean_voice_disabled", "Lean voice is not enabled on this server.")
    return True


def require_lean_offer(payload, *, selected, capture_mode):
    mode = payload.get("conversation_mode", "legacy")
    if mode != (LEAN_MODE if selected else "legacy"):
        raise JarvisHTTPError(400, "conversation_mode_mismatch", "The offer and selected conversation mode disagree.")
    if selected and (capture_mode != "auto" or payload.get("voice_route") is not None):
        raise JarvisHTTPError(400, "lean_offer_conflict", "Lean v1 requires automatic speech and exclusive native turn ownership.")
