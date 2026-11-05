"""Isolated C0 content/patch contract tests; no providers or database."""
import copy
import json
import unittest
from types import SimpleNamespace

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from app.knowledge.models import SourceError
from app.notes.workspace_content import (
    DocumentValidationError, MAX_VALIDATION_INDEX, MAX_VALIDATION_ISSUES,
    MAX_VALIDATION_PATH_DEPTH, MAX_VALIDATION_SEGMENT_LENGTH,
    apply_patches, canonical_content, content_for_note, parse_command, render_body, validate_content,
)


def document():
    return {"schema_version": 1, "blocks": [
        {"type": "text", "id": "intro", "text": "Back day"},
        {"type": "table", "id": "sets", "columns": [
            {"id": "exercise", "label": "Exercise"}, {"id": "reps", "label": "Reps"},
            {"id": "observed", "label": "Observed"}], "rows": [
                {"id": "set_1", "cells": {"exercise": "Row", "reps": None, "observed": False}},
                {"id": "set_2", "cells": {"exercise": "Row", "reps": 10, "observed": True}}]}]}


class WorkspaceContentTests(unittest.TestCase):
    def test_rejected_column_names_identify_missing_label_without_aliasing(self):
        for wrong_field in ("name", "title"):
            value = document()
            column = value["blocks"][1]["columns"][0]
            column[wrong_field] = column.pop("label")
            with self.subTest(field=wrong_field), self.assertRaises(DocumentValidationError) as caught:
                parse_command({"action": "create", "title": "Private title", "content": value})
            error = caught.exception
            self.assertIsInstance(error, SourceError)
            self.assertEqual((error.code, error.status_code), ("invalid_document", 422))
            self.assertEqual(str(error), "The document or patch is invalid.")
            self.assertIn({"path": ["content", "blocks", 1, "columns", 0, "label"], "code": "missing"},
                          error.validation_issues)
            self.assertIn({"path": ["content", "blocks", 1, "columns", 0, "*"], "code": "extra_forbidden"},
                          error.validation_issues)
            self.assertNotIn("Private title", json.dumps(error.validation_issues))
            self.assertNotIn("label", column)  # No repair/alias mutation.

    def test_generic_table_still_accepts_exact_labels_and_unknown_values(self):
        value = {"schema_version": 1, "blocks": [{"type": "table", "id": "inventory",
            "columns": [{"id": "item", "label": "Item"}, {"id": "count", "label": "Count"}],
            "rows": [{"id": "first", "cells": {"item": "Paper", "count": None}}]}]}
        parsed = parse_command({"action": "create", "title": "Inventory", "content": value})
        self.assertEqual(parsed.content.model_dump(mode="json"), value)

    def test_patch_rejection_reports_its_own_missing_field(self):
        with self.assertRaises(DocumentValidationError) as caught:
            parse_command({"action": "patch", "artifact_id": "00000000-0000-4000-8000-000000000001",
                "base_version_id": "private_version", "changes": [{"op": "set_cell",
                    "block_id": "private_block", "row_id": "private_row", "column_id": "private_column"}]})
        self.assertEqual(caught.exception.validation_issues,
                         [{"path": ["changes", 0, "value"], "code": "missing"}])

    def test_dynamic_keys_values_and_known_name_lookalikes_are_never_returned(self):
        secret = "PRIVATE_DYNAMIC_KEY_" + "x" * 300
        value = document()
        value["blocks"][1]["rows"][0]["cells"] = {"label": {"private_value": secret}, secret: []}
        value[secret] = "private_extra_value"
        with self.assertRaises(DocumentValidationError) as caught:
            validate_content(value)
        issues = caught.exception.validation_issues
        encoded = json.dumps(issues)
        for forbidden in (secret, "private_value", "private_extra_value", '"label"', '"input"', '"ctx"', '"msg"'):
            self.assertNotIn(forbidden, encoded)
        self.assertTrue(any(issue["path"] == ["blocks", 1, "rows", 0, "cells", "*"] for issue in issues))
        self.assertTrue(all(set(issue) == {"path", "code"} for issue in issues))

    def test_validation_issues_have_finite_count_paths_indices_and_length(self):
        with self.assertRaises(DocumentValidationError) as caught:
            validate_content({"schema_version": 1,
                "blocks": [{"type": "text", "id": f"b{i}"} for i in range(32)]})
        issues = caught.exception.validation_issues
        self.assertEqual(len(issues), MAX_VALIDATION_ISSUES)
        for issue in issues:
            self.assertLessEqual(len(issue["path"]), MAX_VALIDATION_PATH_DEPTH)
            self.assertLessEqual(len(issue["code"]), MAX_VALIDATION_SEGMENT_LENGTH)
            for segment in issue["path"]:
                if type(segment) is int:
                    self.assertTrue(0 <= segment <= MAX_VALIDATION_INDEX)
                else:
                    self.assertLessEqual(len(segment), MAX_VALIDATION_SEGMENT_LENGTH)
        self.assertLess(len(json.dumps(issues)), 4096)
        raw = ValidationError.from_exception_data("DocumentContent", [{"type": "missing",
            "loc": ("blocks", 10**100, *("private_deep_key" for _ in range(100))), "input": "private_value"}])
        self.assertEqual(DocumentValidationError(raw).validation_issues,
                         [{"path": ["blocks", "*"], "code": "missing"}])

    def test_custom_error_codes_and_messages_cannot_supply_output_text(self):
        raw = ValidationError.from_exception_data("DocumentContent", [{
            "type": PydanticCustomError("private_error_code", "private_message {secret}", {"secret": "private_context"}),
            "loc": ("schema_version",), "input": "private_input"}])
        error = DocumentValidationError(raw)
        self.assertEqual(error.validation_issues, [{"path": ["schema_version"], "code": "invalid_value"}])
        self.assertNotIn("private_", json.dumps(error.validation_issues))

    def test_unknown_and_bool_projection_preserves_literal_values(self):
        content = validate_content(document())
        self.assertEqual(render_body(content), "Back day\n\nExercise\tReps\tObserved\nRow\tunknown\tfalse\nRow\t10\ttrue")
        self.assertIsNone(content.blocks[1].rows[0].cells["reps"])

    def test_patch_addresses_identity_and_preserves_original(self):
        content = validate_content(document())
        changes = parse_command({"action": "patch", "artifact_id": "00000000-0000-4000-8000-000000000001",
            "base_version_id": "source_base", "changes": [
                {"op": "set_cell", "block_id": "sets", "row_id": "set_1", "column_id": "reps", "value": 8},
                {"op": "set_title", "title": "Corrected back day"}]}).changes
        title, edited = apply_patches("Back day", content, changes)
        self.assertEqual(title, "Corrected back day")
        self.assertEqual(edited.blocks[1].rows[0].cells["reps"], 8)
        self.assertIsNone(content.blocks[1].rows[0].cells["reps"])

    def test_invalid_later_patch_does_not_mutate_base(self):
        content = validate_content(document())
        before = canonical_content(content)
        command = parse_command({"action": "patch", "artifact_id": "00000000-0000-4000-8000-000000000001",
            "base_version_id": "source_base", "changes": [
                {"op": "set_text", "block_id": "intro", "text": "Changed"},
                {"op": "set_cell", "block_id": "sets", "row_id": "missing", "column_id": "reps", "value": 8}]})
        with self.assertRaises(SourceError) as caught:
            apply_patches("Back day", content, command.changes)
        self.assertEqual(caught.exception.code, "patch_target_unavailable")
        self.assertEqual(canonical_content(content), before)

    def test_legacy_projection_does_not_write_or_infer_rows(self):
        note = SimpleNamespace(metadata={"custom": "kept"}, body="3 sets of rows, maybe eight reps.")
        result = content_for_note(note)
        self.assertEqual(result.blocks[0].id, "legacy_body")
        self.assertEqual(result.blocks[0].text, note.body)
        self.assertEqual(note.metadata, {"custom": "kept"})

    def test_body_envelope_mismatch_is_not_silently_repaired(self):
        with self.assertRaises(SourceError) as caught:
            content_for_note(SimpleNamespace(metadata={"workspace_v1": document()}, body="Wrong body"))
        self.assertEqual(caught.exception.code, "note_content_mismatch")

    def test_validation_rejects_duplicate_and_unknown_identities(self):
        cases = []
        value = document(); value["blocks"].append(copy.deepcopy(value["blocks"][0])); cases.append(value)
        value = document(); value["blocks"][1]["columns"].append({"id": "reps", "label": "Duplicate"}); cases.append(value)
        value = document(); value["blocks"][1]["rows"].append(copy.deepcopy(value["blocks"][1]["rows"][0])); cases.append(value)
        value = document(); value["blocks"][1]["rows"][0]["cells"]["missing"] = 8; cases.append(value)
        value = document(); value["blocks"][0]["id"] = "not an id"; cases.append(value)
        value = document(); value["blocks"][0]["extra"] = "forbidden"; cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(SourceError):
                validate_content(value)

    def test_scalars_are_not_coerced_and_nonfinite_rejected(self):
        for scalar in ([], {}, float("nan"), float("inf"), float("-inf")):
            value = document(); value["blocks"][1]["rows"][0]["cells"]["reps"] = scalar
            with self.subTest(scalar=scalar), self.assertRaises(SourceError):
                validate_content(value)
        value = document(); value["blocks"][1]["rows"][0]["cells"]["reps"] = "08"
        self.assertEqual(validate_content(value).blocks[1].rows[0].cells["reps"], "08")

    def test_schema_bool_is_not_version_integer(self):
        value = document(); value["schema_version"] = True
        with self.assertRaises(SourceError):
            validate_content(value)

    def test_unstorable_text_rejected_before_persistence(self):
        for text in ("bad\x00text", "bad\ud800text"):
            value = document(); value["blocks"][0]["text"] = text
            with self.subTest(text=repr(text)), self.assertRaises(SourceError):
                validate_content(value)

    def test_body_and_normalized_byte_bounds(self):
        for text in ("x" * 20001, "😀" * 17000):
            with self.subTest(chars=len(text)), self.assertRaises(SourceError) as caught:
                validate_content({"schema_version": 1, "blocks": [{"type": "text", "id": "text", "text": text}]})
            self.assertEqual(caught.exception.code, "document_too_large")

    def test_row_limit_is_total_across_tables(self):
        value = document()
        table = value["blocks"][1]
        table["rows"] = [{"id": f"row_{i}", "cells": {}} for i in range(101)]
        second = copy.deepcopy(table); second["id"] = "second"; value["blocks"].append(second)
        with self.assertRaises(SourceError) as caught:
            validate_content(value)
        self.assertEqual(caught.exception.code, "document_too_large")

    def test_mutations_require_exact_base_and_reject_extra_authority(self):
        for command in (
            {"action": "patch", "artifact_id": "00000000-0000-4000-8000-000000000001", "changes": []},
            {"action": "create", "title": "Back day", "content": document(), "owner_id": "someone"},
        ):
            with self.assertRaises(SourceError):
                parse_command(command)

    def test_append_and_remove_rows_keep_stable_ids(self):
        command = parse_command({"action": "patch", "artifact_id": "00000000-0000-4000-8000-000000000001",
            "base_version_id": "source_base", "changes": [
                {"op": "append_rows", "block_id": "sets", "rows": [{"id": "set_3", "cells": {"reps": 6}}]},
                {"op": "remove_rows", "block_id": "sets", "row_ids": ["set_1"]}]})
        _, content = apply_patches("Back day", validate_content(document()), command.changes)
        self.assertEqual([r.id for r in content.blocks[1].rows], ["set_2", "set_3"])


if __name__ == "__main__":
    unittest.main()
