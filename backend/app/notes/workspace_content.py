"""Pure canonical document validation, text projection and atomic patches."""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache

from pydantic import TypeAdapter, ValidationError

from app.knowledge.models import SourceError
from app.workspace.models import DocumentContent, NoteCommand


MAX_VALIDATION_ISSUES = 8
MAX_VALIDATION_PATH_DEPTH = 12
MAX_VALIDATION_SEGMENT_LENGTH = 32
MAX_VALIDATION_INDEX = 1_000_000
_ISSUE_CODES = frozenset({
    "missing", "extra_forbidden", "invalid_key", "string_type", "string_unicode",
    "string_too_short", "string_too_long", "string_pattern_mismatch", "int_type",
    "float_type", "bool_type", "finite_number", "uuid_type", "uuid_parsing",
    "uuid_version", "model_type", "model_attributes_type", "dict_type", "list_type",
    "literal_error", "union_tag_invalid", "union_tag_not_found", "too_long",
    "too_short", "value_error", "invalid_value",
})


@lru_cache(maxsize=2)
def _validation_schema(command):
    # Used only to recognize fixed structural path segments, not as a new
    # validator or provider schema. Dynamic dictionary keys never pass through.
    return TypeAdapter(NoteCommand).json_schema() if command else DocumentContent.model_json_schema()


def _validation_path(location, code, schema):
    definitions = schema.get("$defs", {})
    node, path = schema, []
    for offset, segment in enumerate(location):
        if len(path) >= MAX_VALIDATION_PATH_DEPTH:
            break
        if "$ref" in node:
            node = definitions.get(node["$ref"].removeprefix("#/$defs/"), {})
        mapping = node.get("discriminator", {}).get("mapping", {})
        if type(segment) is str and segment in mapping:
            # Pydantic includes known union tags in loc; they are not fields.
            node = {"$ref": mapping[segment]}
            continue
        if node.get("type") == "array":
            if type(segment) is int and 0 <= segment <= MAX_VALIDATION_INDEX:
                path.append(segment)
                node = node.get("items", {})
                continue
            path.append("*")
            break
        if node.get("type") == "object":
            properties = node.get("properties", {})
            if (type(segment) is str and segment in properties
                    and len(segment) <= MAX_VALIDATION_SEGMENT_LENGTH
                    and not (offset == len(location) - 1 and code in {"extra_forbidden", "invalid_key"})):
                path.append(segment)
                node = properties[segment]
                continue
            # Extra fields and cells' user-chosen column IDs are opaque, even
            # when their spelling happens to equal a known schema field.
            path.append("*")
        # Scalar-union implementation labels are not document path segments.
        break
    return path


class DocumentValidationError(SourceError):
    """Existing rejection plus bounded, value-free structural repair details.

    Only validation_issues is safe for a tool response; never serialize the
    chained exception or raw Pydantic errors. '*' denotes an opaque location.
    """
    def __init__(self, error: ValidationError, *, command=False):
        errors = error.errors(include_input=False, include_context=False, include_url=False)
        oversized = any("document_too_large" in e["msg"] or e["type"] == "too_long" for e in errors)
        super().__init__("document_too_large" if oversized else "invalid_document",
            "The document exceeds a supported size bound." if oversized else "The document or patch is invalid.", 422)
        self.validation_issues: list[dict] = []
        schema = _validation_schema(command)
        for issue in errors:
            code = issue["type"] if issue["type"] in _ISSUE_CODES else "invalid_value"
            safe = {"path": _validation_path(issue["loc"], code, schema), "code": code}
            if safe not in self.validation_issues:
                self.validation_issues.append(safe)
                if len(self.validation_issues) == MAX_VALIDATION_ISSUES:
                    break


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_content(content: DocumentContent) -> str:
    return canonical_json(content.model_dump(mode="json"))


def parameters_digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scalar(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_body(content: DocumentContent) -> str:
    rendered = []
    for block in content.blocks:
        if block.type == "text":
            rendered.append(block.text)
            continue
        def cell(value):
            return _scalar(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
        lines = ["\t".join(cell(column.label) for column in block.columns)]
        lines.extend("\t".join(cell(row.cells.get(column.id)) for column in block.columns) for row in block.rows)
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def validate_content(value) -> DocumentContent:
    try:
        return DocumentContent.model_validate(value)
    except ValidationError as error:
        raise DocumentValidationError(error) from error


def parse_command(value):
    try:
        return TypeAdapter(NoteCommand).validate_python(value)
    except ValidationError as error:
        raise DocumentValidationError(error, command=True) from error


def content_for_note(note) -> DocumentContent:
    """Legacy notes are projected literally, without mutation or interpretation."""
    if "workspace_v1" not in note.metadata:
        return validate_content({"schema_version": 1, "blocks": [
            {"type": "text", "id": "legacy_body", "text": note.body}]})
    content = validate_content(note.metadata["workspace_v1"])
    if render_body(content) != note.body:
        raise SourceError("note_content_mismatch", "The note body and structured content disagree.")
    return content


def apply_patches(title: str, content: DocumentContent, changes) -> tuple[str, DocumentContent]:
    """Prepare all changes on a copy. Persist nothing until every target validates."""
    value = copy.deepcopy(content.model_dump(mode="json"))
    for change in changes:
        patch = change.model_dump(mode="json")
        op = patch["op"]
        if op == "set_title":
            title = patch["title"]
        elif op == "append_block":
            value["blocks"].append(patch["block"])
        else:
            block = next((b for b in value["blocks"] if b["id"] == patch["block_id"]), None)
            if block is None:
                raise SourceError("patch_target_unavailable", "The referenced block is not in this document.")
            if op == "set_text":
                if block["type"] != "text":
                    raise SourceError("patch_target_type", "Text replacement requires a text block.")
                block["text"] = patch["text"]
            else:
                if block["type"] != "table":
                    raise SourceError("patch_target_type", "This edit requires a table block.")
                if op == "set_cell":
                    row = next((r for r in block["rows"] if r["id"] == patch["row_id"]), None)
                    if row is None or not any(c["id"] == patch["column_id"] for c in block["columns"]):
                        raise SourceError("patch_target_unavailable", "The exact row or column is not present.")
                    row["cells"][patch["column_id"]] = patch["value"]
                elif op == "append_rows":
                    block["rows"].extend(patch["rows"])
                elif op == "remove_rows":
                    ids = set(patch["row_ids"])
                    if len(ids) != len(patch["row_ids"]) or ids - {r["id"] for r in block["rows"]}:
                        raise SourceError("patch_target_unavailable", "Every removal must identify a distinct existing row.")
                    block["rows"] = [row for row in block["rows"] if row["id"] not in ids]
    return title, validate_content(value)
