"""Pure validation for the first bounded media-edit recipe.

This module performs no I/O. Its caller supplies already resolved source
inspections, template inventory and policy. Matching those inputs is NOT proof
of current source permission, installed software, successful rendering or a
publication grant. Those checks stay with the existing source/task owners.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Annotated, Any, Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, field_validator, model_validator


FPS = 24
MAX_RECIPE_BYTES = 64 * 1024
LocalId = Annotated[str, Field(strict=True, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
VersionId = Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class MediaRecipeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class FrozenRecord(BaseModel):
    # Tuples below also prevent mutations through a nested collection.
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


def _stored_text(value: str) -> str:
    if "\x00" in value:
        raise ValueError("Text cannot contain NUL")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ValueError("Text must be valid UTF-8") from None
    return value


class RetainedRef(FrozenRecord):
    """An exact retained file reference, never a mutable generation or URL."""
    file_id: UUID
    version_id: VersionId
    content_sha256: Digest

    @field_validator("version_id")
    @classmethod
    def opaque_version(cls, value):
        if not value.strip():
            raise ValueError("A nonblank exact version is required")
        return _stored_text(value)  # Do not trim or interpret opaque identity.


class TemplatePin(FrozenRecord):
    id: Literal["jarvis.clip.v1"]
    version: Annotated[str, Field(strict=True, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    bundle_sha256: Digest


class RecipeSource(FrozenRecord):
    source_id: LocalId
    file: RetainedRef


class EditSegment(FrozenRecord):
    segment_id: LocalId
    source_id: LocalId
    in_ms: NonnegativeInt
    out_ms: NonnegativeInt


class EditCaption(FrozenRecord):
    caption_id: LocalId
    start_frame: NonnegativeInt
    end_frame: NonnegativeInt
    text: Annotated[StrictStr, Field(max_length=200)]

    @field_validator("text")
    @classmethod
    def literal_text(cls, value):
        # HTML-looking strings are ordinary literal captions. A template must
        # use a text node, never innerHTML or an executable interpolation.
        return _stored_text(value)


class MediaEditRecipeV1(FrozenRecord):
    schema_version: Literal[1]
    edit_id: UUID
    previous_recipe: RetainedRef | None
    template: TemplatePin
    sources: tuple[RecipeSource, ...] = Field(min_length=1, max_length=1)
    segments: tuple[EditSegment, ...] = Field(min_length=1, max_length=8)
    captions: tuple[EditCaption, ...] = Field(max_length=16)
    output_profile: Literal["portrait_720p24_v1"]
    fit: Literal["contain"]
    audio: Literal["preserve", "mute"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @model_validator(mode="after")
    def identities(self):
        for values, field in ((self.sources, "source_id"), (self.segments, "segment_id"), (self.captions, "caption_id")):
            ids = [getattr(value, field) for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate " + field)
        source_ids = {source.source_id for source in self.sources}
        if any(segment.source_id not in source_ids for segment in self.segments):
            raise ValueError("A segment must name a declared source")
        return self


class SourceInspection(FrozenRecord):
    """Caller-supplied inspection of exact bytes; not an authorization token."""
    file: RetainedRef
    duration_ms: PositiveInt
    size_bytes: PositiveInt
    video_decodable: StrictBool
    audio_streams: NonnegativeInt
    audio_decodable: StrictBool


class TemplateInventoryEntry(FrozenRecord):
    """A caller-supplied inventory observation; this code never probes a host."""
    template: TemplatePin
    output_profile: Literal["portrait_720p24_v1"]
    ready: StrictBool


class PreviousRecipe(FrozenRecord):
    """Exact predecessor identity and edit lineage resolved by the caller."""
    file: RetainedRef
    edit_id: UUID


class MediaRecipePolicy(FrozenRecord):
    # Required explicit caller limits. No new runtime default or grant is issued.
    max_source_bytes: PositiveInt
    max_source_duration_ms: PositiveInt
    max_output_frames: PositiveInt
    max_recipe_bytes: Annotated[int, Field(strict=True, ge=1, le=MAX_RECIPE_BYTES)] = MAX_RECIPE_BYTES


def _canonical_bytes(recipe: MediaEditRecipeV1) -> bytes:
    return json.dumps(recipe.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8", errors="strict")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number")


def parse_recipe(value: Any) -> MediaEditRecipeV1:
    """Validate the wire shape without changing text, order or cut boundaries.

    Canonicalization normalizes UUID serialization and JSON object key order;
    it never trims text, rounds times or replaces an exact source reference.
    JSON text additionally rejects duplicate object keys.
    """
    try:
        if isinstance(value, (str, bytes, bytearray)):
            value = json.loads(value, object_pairs_hook=_unique_json_object, parse_constant=_reject_constant)
        recipe = MediaEditRecipeV1.model_validate(value)
        if len(_canonical_bytes(recipe)) > MAX_RECIPE_BYTES:
            raise MediaRecipeError("media_recipe_too_large", "Canonical recipe exceeds 64 KiB.")
        return recipe
    except MediaRecipeError:
        raise
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise MediaRecipeError("media_recipe_invalid", "Recipe does not match the bounded media-edit schema.") from None


def canonical_recipe(value: Any) -> bytes:
    """Canonical UTF-8 data for persistence/digest, not evidence of authority."""
    return _canonical_bytes(parse_recipe(value))


def recipe_digest(value: Any) -> str:
    return hashlib.sha256(canonical_recipe(value)).hexdigest()


def _input(model, value):
    try:
        return model.model_validate(value)
    except (ValidationError, ValueError, TypeError):
        raise MediaRecipeError("media_validation_context_invalid", "Invalid supplied inspection, inventory or policy.") from None


@dataclass(frozen=True)
class ValidatedMediaRecipe:
    """Pure validation result; deliberately contains no completed/rendered flag."""
    recipe: MediaEditRecipeV1
    canonical_bytes: bytes
    digest: str
    duration_in_frames: int

    def template_props(self) -> dict[str, Any]:
        """Fresh data-only props. Trusted staging binds source IDs separately.

        No path, external URL, source credentials, JSX, filtergraph or renderer
        executable is accepted here. Captions remain literal strings, including
        markup-looking text; the renderer must honor that text-node contract.
        """
        output_from = 0
        segments = []
        for segment in self.recipe.segments:
            from_frame = segment.in_ms * FPS // 1000
            to_frame = segment.out_ms * FPS // 1000
            duration = to_frame - from_frame
            segments.append({"segmentId": segment.segment_id, "sourceId": segment.source_id,
                "fromFrame": from_frame, "toFrame": to_frame, "outputFromFrame": output_from,
                "durationInFrames": duration})
            output_from += duration
        return {"schemaVersion": 1, "width": 720, "height": 1280, "fps": FPS,
            "durationInFrames": self.duration_in_frames, "fit": self.recipe.fit, "audio": self.recipe.audio,
            "segments": segments,
            "captions": [{"captionId": caption.caption_id, "fromFrame": caption.start_frame,
                "toFrame": caption.end_frame, "text": caption.text} for caption in self.recipe.captions]}


def validate_recipe(value: Any, *, inspections: Sequence[SourceInspection],
                    inventory: Sequence[TemplateInventoryEntry], policy: MediaRecipePolicy,
                    previous: PreviousRecipe | None = None) -> ValidatedMediaRecipe:
    """Check recipe data against explicit trusted inputs; never perform effects.

    Integer milliseconds at 24 fps can express exact boundaries only every
    125 ms (three frames). Unsupported boundaries are rejected, never rounded.
    This is intentionally not arbitrary-frame-rate normalization.
    """
    recipe = parse_recipe(value)
    policy = _input(MediaRecipePolicy, policy)
    canonical = _canonical_bytes(recipe)
    if len(canonical) > policy.max_recipe_bytes:
        raise MediaRecipeError("media_recipe_too_large", "Canonical recipe exceeds its explicit byte allowance.")

    supplied = [_input(SourceInspection, item) for item in inspections]
    if len(supplied) != 1 or supplied[0].file != recipe.sources[0].file:
        raise MediaRecipeError("media_source_reference_mismatch", "One inspection of the exact selected file/version/hash is required.")
    inspection = supplied[0]
    if inspection.size_bytes > policy.max_source_bytes or inspection.duration_ms > policy.max_source_duration_ms:
        raise MediaRecipeError("media_source_limit", "The inspected source exceeds this operation's byte or duration allowance.")
    if not inspection.video_decodable:
        raise MediaRecipeError("media_video_unavailable", "The supplied inspection does not establish decodable video.")
    if recipe.audio == "preserve" and (inspection.audio_streams == 0 or not inspection.audio_decodable):
        raise MediaRecipeError("media_audio_unavailable", "Preserving audio requires a decodable source audio stream; mute must be explicit.")

    candidates = [_input(TemplateInventoryEntry, item) for item in inventory]
    matching_version = [item for item in candidates if item.template.id == recipe.template.id
                        and item.template.version == recipe.template.version]
    if len(matching_version) > 1:
        raise MediaRecipeError("media_template_ambiguous", "Template inventory has more than one entry for the selected version.")
    if (not matching_version or matching_version[0].template != recipe.template
            or matching_version[0].output_profile != recipe.output_profile or not matching_version[0].ready):
        raise MediaRecipeError("media_template_unavailable", "The exact template bundle/profile is not ready in the supplied inventory.")

    if recipe.previous_recipe is None:
        if previous is not None:
            raise MediaRecipeError("media_predecessor_mismatch", "A first recipe cannot claim an unreferenced predecessor.")
    else:
        if previous is None:
            raise MediaRecipeError("media_predecessor_mismatch", "An exact predecessor inspection is required for a revision.")
        predecessor = _input(PreviousRecipe, previous)
        if predecessor.file != recipe.previous_recipe or predecessor.edit_id != recipe.edit_id:
            raise MediaRecipeError("media_predecessor_mismatch", "The predecessor reference or edit lineage does not match.")

    total_frames = 0
    for segment in recipe.segments:
        if not segment.in_ms < segment.out_ms <= inspection.duration_ms:
            raise MediaRecipeError("media_interval_invalid", "Each half-open cut must lie within the inspected source duration.")
        if segment.in_ms * FPS % 1000 or segment.out_ms * FPS % 1000:
            raise MediaRecipeError("media_interval_off_grid", "24 fps integer-millisecond cuts must use exact 125 ms boundaries; no rounding is applied.")
        total_frames += (segment.out_ms - segment.in_ms) * FPS // 1000
    if total_frames > policy.max_output_frames:
        raise MediaRecipeError("media_output_limit", "The cut duration exceeds this operation's output-frame allowance.")
    last_end = 0
    for caption in sorted(recipe.captions, key=lambda item: (item.start_frame, item.end_frame)):
        if not caption.start_frame < caption.end_frame <= total_frames:
            raise MediaRecipeError("media_caption_interval_invalid", "Caption frames must lie inside the half-open output timeline.")
        if caption.start_frame < last_end:
            raise MediaRecipeError("media_caption_overlap", "This template supports only nonoverlapping literal captions.")
        last_end = caption.end_frame
    return ValidatedMediaRecipe(recipe=recipe, canonical_bytes=canonical,
        digest=hashlib.sha256(canonical).hexdigest(), duration_in_frames=total_frames)
