from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.settings import JarvisSettings
from app.visual_memory.models import ValidatedVisualFrame, VisualFrameSource


SUPPORTED_VISUAL_FRAME_TYPE = "visual.frame.capture"
JPEG_CONTENT_TYPE = "image/jpeg"
_JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass(frozen=True)
class VisualFrameValidationError(Exception):
    code: str
    message: str


def validate_visual_frame_event(
    values: dict[str, Any],
    *,
    settings: JarvisSettings,
    allowed_sources: set[str],
) -> ValidatedVisualFrame:
    if values.get("type") != SUPPORTED_VISUAL_FRAME_TYPE:
        raise VisualFrameValidationError(
            "unsupported_visual_event",
            "Only visual.frame.capture events are supported on this websocket.",
        )

    source = _required_string(values, "source")
    if source not in allowed_sources:
        raise VisualFrameValidationError(
            "source_not_allowed",
            "This visual source is not allowed for the current backend configuration.",
        )

    content_type = _required_string(values, "content_type").lower()
    if content_type != JPEG_CONTENT_TYPE:
        raise VisualFrameValidationError(
            "invalid_content_type",
            "Visual frames must be JPEG images.",
        )

    frame_sequence = _required_positive_int(values, "frame_sequence")
    captured_at = _required_datetime(values, "captured_at")
    image_base64 = _required_string(values, "image_base64")
    image_bytes = _decode_image(image_base64)
    if len(image_bytes) > settings.VISUAL_FRAME_MAX_BYTES:
        raise VisualFrameValidationError(
            "frame_too_large",
            f"Visual frame exceeds {settings.VISUAL_FRAME_MAX_BYTES} bytes.",
        )

    width, height = _jpeg_dimensions(image_bytes)
    if width > settings.VISUAL_FRAME_MAX_WIDTH or height > settings.VISUAL_FRAME_MAX_HEIGHT:
        raise VisualFrameValidationError(
            "frame_too_large",
            (
                "Visual frame dimensions exceed "
                f"{settings.VISUAL_FRAME_MAX_WIDTH}x{settings.VISUAL_FRAME_MAX_HEIGHT}."
            ),
        )

    declared_width = _optional_positive_int(values, "width")
    declared_height = _optional_positive_int(values, "height")
    if declared_width is not None and declared_width != width:
        raise VisualFrameValidationError("invalid_jpeg", "Declared frame width does not match JPEG bytes.")
    if declared_height is not None and declared_height != height:
        raise VisualFrameValidationError("invalid_jpeg", "Declared frame height does not match JPEG bytes.")

    return ValidatedVisualFrame(
        frame_sequence=frame_sequence,
        captured_at=captured_at,
        source=source,  # type: ignore[arg-type]
        content_type=content_type,
        width=width,
        height=height,
        image_bytes=image_bytes,
    )


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VisualFrameValidationError("invalid_visual_frame", f"{key} is required.")
    return value.strip()


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool):
        value = None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdigit():
        parsed = int(value)
    else:
        raise VisualFrameValidationError("invalid_visual_frame", f"{key} must be a positive integer.")
    if parsed < 0:
        raise VisualFrameValidationError("invalid_visual_frame", f"{key} must be non-negative.")
    return parsed


def _optional_positive_int(values: dict[str, Any], key: str) -> int | None:
    if key not in values or values[key] is None:
        return None
    return _required_positive_int(values, key)


def _required_datetime(values: dict[str, Any], key: str) -> datetime:
    raw = _required_string(values, key)
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise VisualFrameValidationError("invalid_visual_frame", f"{key} must be ISO-8601.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _decode_image(image_base64: str) -> bytes:
    try:
        decoded = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise VisualFrameValidationError("invalid_jpeg", "Frame image is not valid base64.") from error
    if not decoded:
        raise VisualFrameValidationError("invalid_jpeg", "Frame image is empty.")
    return decoded


def jpeg_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Reuse ingest's JPEG parser when resolving an existing image for delivery."""
    return _jpeg_dimensions(image_bytes)


def _jpeg_dimensions(image_bytes: bytes) -> tuple[int, int]:
    if len(image_bytes) < 4 or not image_bytes.startswith(b"\xff\xd8"):
        raise VisualFrameValidationError("invalid_jpeg", "Frame image is not a JPEG.")

    index = 2
    length = len(image_bytes)
    while index < length - 1:
        if image_bytes[index] != 0xFF:
            index += 1
            continue
        while index < length and image_bytes[index] == 0xFF:
            index += 1
        if index >= length:
            break

        marker = image_bytes[index]
        index += 1
        if marker == 0xD9:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > length:
            break

        segment_length = int.from_bytes(image_bytes[index : index + 2], "big")
        if segment_length < 2:
            raise VisualFrameValidationError("invalid_jpeg", "JPEG segment length is invalid.")
        segment_start = index + 2
        segment_end = index + segment_length
        if segment_end > length:
            raise VisualFrameValidationError("invalid_jpeg", "JPEG segment is truncated.")

        if marker in _JPEG_SOF_MARKERS:
            if segment_start + 5 > segment_end:
                raise VisualFrameValidationError("invalid_jpeg", "JPEG size segment is invalid.")
            height = int.from_bytes(image_bytes[segment_start + 1 : segment_start + 3], "big")
            width = int.from_bytes(image_bytes[segment_start + 3 : segment_start + 5], "big")
            if width <= 0 or height <= 0:
                raise VisualFrameValidationError("invalid_jpeg", "JPEG dimensions are invalid.")
            return width, height

        index = segment_end

    raise VisualFrameValidationError("invalid_jpeg", "JPEG dimensions could not be read.")
