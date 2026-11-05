"""C0 structured document wire models. Contents are data, never authority."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator


def _stored_text(value: str) -> str:
    # PostgreSQL jsonb/text cannot retain NUL or an unpaired Unicode surrogate.
    # Reject them at the data boundary, before entering a write transaction.
    if "\x00" in value:
        raise ValueError("Text cannot contain NUL")
    value.encode("utf-8")
    return value


StoredText = Annotated[StrictStr, AfterValidator(_stored_text)]
LocalId = Annotated[str, Field(strict=True, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")]
Title = Annotated[StoredText, Field(min_length=1, max_length=200)]
Scalar = Union[StoredText, StrictInt, Annotated[StrictFloat, Field(allow_inf_nan=False)], StrictBool, None]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TextBlock(StrictRecord):
    type: Literal["text"]
    id: LocalId
    text: StoredText


class TableColumn(StrictRecord):
    id: LocalId
    label: StoredText


class TableRow(StrictRecord):
    id: LocalId
    cells: dict[LocalId, Scalar]


class TableBlock(StrictRecord):
    type: Literal["table"]
    id: LocalId
    columns: list[TableColumn] = Field(max_length=16)
    rows: list[TableRow]

    @model_validator(mode="after")
    def valid_table(self):
        columns = [column.id for column in self.columns]
        rows = [row.id for row in self.rows]
        if len(set(columns)) != len(columns) or len(set(rows)) != len(rows):
            raise ValueError("Column and row identities must be unique within their table")
        if any(set(row.cells) - set(columns) for row in self.rows):
            raise ValueError("Every cell must name a column in its table")
        return self


Block = Annotated[Union[TextBlock, TableBlock], Field(discriminator="type")]


class DocumentContent(StrictRecord):
    schema_version: Literal[1]
    blocks: list[Block] = Field(max_length=32)

    @model_validator(mode="before")
    @classmethod
    def exact_schema(cls, value):
        if isinstance(value, dict) and type(value.get("schema_version")) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @model_validator(mode="after")
    def valid_document(self):
        ids = [block.id for block in self.blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("Block identities must be unique within their document")
        if sum(len(block.rows) for block in self.blocks if isinstance(block, TableBlock)) > 200:
            raise ValueError("document_too_large: at most 200 table rows")
        # Import locally so the plain text projection and this wire schema share
        # exactly one byte/body bound instead of drifting independently.
        from app.notes.workspace_content import canonical_content, render_body
        if len(canonical_content(self).encode("utf-8")) > 65536 or len(render_body(self)) > 20000:
            raise ValueError("document_too_large: content/body bound exceeded")
        return self


class SetTitle(StrictRecord):
    op: Literal["set_title"]
    title: Title


class SetText(StrictRecord):
    op: Literal["set_text"]
    block_id: LocalId
    text: StoredText


class SetCell(StrictRecord):
    op: Literal["set_cell"]
    block_id: LocalId
    row_id: LocalId
    column_id: LocalId
    value: Scalar


class AppendRows(StrictRecord):
    op: Literal["append_rows"]
    block_id: LocalId
    rows: list[TableRow]


class RemoveRows(StrictRecord):
    op: Literal["remove_rows"]
    block_id: LocalId
    row_ids: list[LocalId]


class AppendBlock(StrictRecord):
    op: Literal["append_block"]
    block: Block


Patch = Annotated[Union[SetTitle, SetText, SetCell, AppendRows, RemoveRows, AppendBlock], Field(discriminator="op")]


class CreateNote(StrictRecord):
    action: Literal["create"]
    title: Title
    content: DocumentContent


class PatchNote(StrictRecord):
    action: Literal["patch"]
    artifact_id: UUID
    base_version_id: Annotated[str, Field(strict=True, min_length=1, max_length=200)]
    changes: list[Patch] = Field(min_length=1, max_length=32)


class RestoreNote(StrictRecord):
    action: Literal["restore"]
    artifact_id: UUID
    base_version_id: Annotated[str, Field(strict=True, min_length=1, max_length=200)]
    restore_version_id: Annotated[str, Field(strict=True, min_length=1, max_length=200)]


class DeleteNote(StrictRecord):
    action: Literal["delete"]
    artifact_id: UUID
    base_version_id: Annotated[str, Field(strict=True, min_length=1, max_length=200)]


NoteCommand = Annotated[Union[CreateNote, PatchNote, RestoreNote, DeleteNote], Field(discriminator="action")]


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """Constructed by authenticated HTTP/Core admission, never model JSON.

    Operation identity alone is not an admission mechanism. The caller must
    validate the active foreground capability before constructing write context;
    the domain separately revalidates owner/session/source and rejects workers.
    """
    owner_id: str
    session_id: UUID
    operation_id: UUID | None = None

    def __post_init__(self):
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("An exact trusted owner is required")
        if not isinstance(self.session_id, UUID):
            raise ValueError("A trusted session UUID is required")
        if self.operation_id is not None and not isinstance(self.operation_id, UUID):
            raise ValueError("An admitted operation UUID is required")


class ArtifactSummary(StrictRecord):
    artifact_id: UUID
    version_id: str
    revision: int
    current_version_id: str
    current_revision: int
    is_current: bool
    kind: Literal["note"] = "note"
    title: str
    updated_at: datetime


class ArtifactView(StrictRecord):
    artifact: ArtifactSummary
    content: DocumentContent


class ArtifactPage(StrictRecord):
    items: list[ArtifactSummary]
    next_cursor: str | None = None


class VersionPage(ArtifactPage):
    unavailable_count: int = 0
