"""Bounded read operations shared by local inspection and outbound transport."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConnectorError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class RootRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    root_id: UUID
    label: str = Field(min_length=1, max_length=100)
    path: str
    device: int
    inode: int


class ReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    operation: Literal["status", "files", "read", "search", "symbols", "definition", "references", "changes"]
    path: str | None = Field(default=None, max_length=2048)
    query: str | None = Field(default=None, min_length=1, max_length=200)
    language: Literal["python", "typescript"] | None = None
    line: int | None = Field(default=None, ge=1, le=1000000)
    column: int | None = Field(default=None, ge=1, le=1000000)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1000000)
    max_chars: int = Field(default=16000, ge=1, le=64000)
    expected_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_snapshot: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def required_inputs(self):
        if self.path is not None and self.operation not in {"read", "search", "definition", "references"}:
            raise ValueError("A path is not applicable to this operation")
        if self.query is not None and self.operation not in {"files", "search", "symbols"}:
            raise ValueError("A query is not applicable to this operation")
        if self.language is not None and self.operation not in {"symbols", "definition", "references"}:
            raise ValueError("A language is not applicable to this operation")
        if (self.line is not None or self.column is not None) and self.operation not in {"definition", "references"}:
            raise ValueError("Positions are only supported for definition and reference lookup")
        if self.expected_snapshot is not None and self.operation == "read":
            raise ValueError("Exact reads use a file digest, not a project snapshot cursor")
        if self.expected_digest is not None and self.operation not in {"read", "search", "definition", "references"}:
            raise ValueError("A file digest is not applicable to this operation")
        if (self.offset or self.max_chars != 16000) and self.operation != "read":
            raise ValueError("Character paging is only supported for reads")
        if self.operation == "symbols" and self.language is None:
            raise ValueError("Project symbol discovery requires a supported language")
        if self.operation in {"read", "definition", "references"} and not self.path:
            raise ValueError("This operation requires a relative file path")
        if self.operation in {"definition", "references"} and (self.line is None or self.column is None):
            raise ValueError("Source positions use one-based Unicode code-point columns")
        if self.operation in {"search", "symbols"} and self.query is None:
            raise ValueError("This operation requires a query")
        if self.operation == "read" and self.offset and self.expected_digest is None:
            raise ValueError("Continuation requires the original file digest")
        return self


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    file_bytes: int = Field(default=262144, ge=1024, le=1048576)
    snapshot_bytes: int = Field(default=4194304, ge=1024, le=16777216)
    files: int = Field(default=1000, ge=1, le=10000)
    entries: int = Field(default=20000, ge=1, le=100000)
    depth: int = Field(default=24, ge=1, le=64)
