"""Exact export binding/adoption contracts without SQL or storage I/O."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.workspace_exports import validate_note_export_original
from legacy_note_export_fixture import note_export_spec
from app.knowledge.models import SourceError
from app.notes.workspace_content import canonical_content, render_body
from legacy_note_export_fixture import NoteExport
from app.workspace.models import DocumentContent, WorkspaceContext
from app.workspace.presentation import PresentationRequest


class ExactNoteExportBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ctx = WorkspaceContext("fixture-owner", uuid4(), uuid4())
        self.document = DocumentContent.model_validate({"schema_version": 1, "blocks": []})
        self.export = NoteExport(self.ctx.owner_id, self.ctx.session_id, self.ctx.operation_id,
            uuid4(), "source_exact", uuid4(), uuid4(), 2, uuid4(), "a" * 64, "text", b"")
        self.source = SimpleNamespace(source_id=self.export.source_id, revision=2,
            payload_id=self.export.source_payload_id, content_digest="a" * 64, generation=99)
        self.connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetchrow=AsyncMock(), fetchval=AsyncMock(return_value=True))
        self.sources = SimpleNamespace(pool=object())
        self.note = SimpleNamespace()

    async def validate(self, export=None, *, row_changes=None, spec_changes=None, dependency=None):
        export = export or self.export
        spec = note_export_spec(self.ctx, export)
        if spec_changes:
            spec = spec.model_copy(update=spec_changes)
        now = datetime.now(UTC)
        request = PresentationRequest(request_id=export.request_id, action="download",
            artifact_kind="note", artifact_id=export.artifact_id, version_id=export.version_id,
            format=export.format, created_at=now, expires_at=now + timedelta(minutes=5))
        self.connection.fetchrow.return_value = {
            "session_id": self.ctx.session_id, "payload": request.model_dump(mode="json")}
        if row_changes:
            self.connection.fetchrow.return_value.update(row_changes)
        snapshot = SimpleNamespace(note=self.note, source=self.source)
        with patch("app.files.workspace_exports.WorkspaceNoteRepository") as repository, \
             patch("app.files.workspace_exports.content_for_note", return_value=self.document):
            repository.return_value.read_note_in_transaction = AsyncMock(return_value=snapshot)
            text = await validate_note_export_original(self.connection, self.sources,
                owner_id=self.ctx.owner_id, area_id=export.area_id, spec=spec,
                content=export.content, metadata_payload_id=dependency)
            call = repository.return_value.read_note_in_transaction.call_args
            self.assertIs(call.args[0], self.connection)
            self.assertEqual(call.args[2:], (export.artifact_id, export.version_id))
            return text

    def test_binding_has_no_mutable_generation_or_body(self):
        spec = note_export_spec(self.ctx, self.export)
        binding = spec.metadata["workspace_note_export"]
        self.assertNotIn("generation", binding)
        self.assertNotIn("content", binding)
        self.assertEqual(spec.filename, self.export.filename)
        self.assertEqual(spec.content_type, self.export.content_type)

    async def test_exact_empty_text_is_valid_without_relaxing_ordinary_text(self):
        self.assertEqual(await self.validate(), "")
        from app.knowledge.files import SourceFileRepository
        with self.assertRaises(ValueError):
            SourceFileRepository._validate_text("")

    async def test_json_is_exact_canonical_content(self):
        raw = canonical_content(self.document).encode()
        export = replace(self.export, format="json", content=raw)
        self.assertEqual((await self.validate(export)).encode(), raw)
        self.assertNotEqual(export.content_type, "text/plain; charset=utf-8")

    def test_owner_session_and_operation_cannot_be_rebound(self):
        for change in ({"owner_id": "foreign"}, {"session_id": uuid4()}, {"request_id": uuid4()}):
            with self.subTest(change=change), self.assertRaises(SourceError):
                note_export_spec(self.ctx, replace(self.export, **change))

    async def test_wrong_session_receipt_is_rejected(self):
        with self.assertRaises(SourceError) as error:
            await self.validate(row_changes={"session_id": uuid4()})
        self.assertEqual(error.exception.code, "workspace_export_not_admitted")

    async def test_byte_change_is_not_exported(self):
        with self.assertRaises(SourceError) as error:
            await self.validate(replace(self.export, content=b"invented"))
        self.assertEqual(error.exception.code, "workspace_export_bytes_changed")

    async def test_source_revision_or_payload_change_is_rejected(self):
        self.source.revision = 3
        with self.assertRaises(SourceError) as error:
            await self.validate()
        self.assertEqual(error.exception.code, "workspace_export_source_changed")

    async def test_generation_change_does_not_change_exact_historical_bytes(self):
        self.source.generation = 100
        self.assertEqual(await self.validate(), "")

    async def test_adoption_requires_source_copy_dependency(self):
        self.connection.fetchval.return_value = False
        with self.assertRaises(SourceError) as error:
            await self.validate(dependency=uuid4())
        self.assertEqual(error.exception.code, "workspace_export_dependency_missing")

    async def test_manifest_cannot_override_filename_or_content_type(self):
        for change in ({"filename": "different.txt"}, {"content_type": "text/html"},
                       {"source": "upload"}):
            with self.subTest(change=change), self.assertRaises(SourceError):
                await self.validate(spec_changes=change)

    async def test_no_nested_or_absent_owner_transaction(self):
        self.connection.is_in_transaction = lambda: False
        with self.assertRaises(RuntimeError):
            await self.validate()
