"""Deterministic producer seam checks; no storage, SQL or device operations."""
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.notes.models import JarvisNote
from legacy_note_export_fixture import ExactNoteExporter
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import PresentationRequest


class ExactNoteExportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 1, tzinfo=UTC)
        self.ctx = WorkspaceContext("owner", uuid4(), uuid4())
        self.note = JarvisNote(id=uuid4(), session_id=self.ctx.session_id, title="Saved note", body="Café <script>",
            source="assistant", created_at=self.now, updated_at=self.now)
        self.source = SimpleNamespace(owner_id=self.ctx.owner_id, document_id="opaque-v1",
            area_id=uuid4(), source_id=uuid4(), revision=2, generation=3, payload_id=uuid4(),
            content_digest="b" * 64, is_current=True)
        self.reads, self.exports = [], []
        self.file_id = uuid4()
        self.request = PresentationRequest(request_id=self.ctx.operation_id, action="download", artifact_kind="note",
            artifact_id=self.note.id, version_id=self.source.document_id, format="text", created_at=self.now,
            expires_at=self.now + timedelta(minutes=5))
        self.notes = SimpleNamespace(read_snapshot=self.read)

    async def read(self, ctx, artifact_id, version_id):
        self.reads.append((ctx, artifact_id, version_id))
        return SimpleNamespace(note=self.note, source=self.source)

    async def publish(self, ctx, export):
        self.exports.append(export)
        return {"file_id": self.file_id, "byte_count": len(export.content),
                "content_sha256": sha256(export.content).hexdigest()}

    async def test_text_and_json_preserve_exact_literal_content_and_source(self):
        exporter = ExactNoteExporter(self.notes, self.publish)
        text_receipt = await exporter(self.ctx, self.request)
        text = self.exports[-1]
        self.assertEqual(text.content, self.note.body.encode("utf-8"))
        self.assertEqual(text.content_type, "text/plain; charset=utf-8")
        self.assertEqual(text.source_payload_id, self.source.payload_id)
        self.assertEqual((text.source_id, text.source_revision), (self.source.source_id, 2))
        self.assertEqual(text_receipt.file_id, self.file_id)
        json_ctx = WorkspaceContext(self.ctx.owner_id, self.ctx.session_id, uuid4())
        json_request = self.request.model_copy(update={"format": "json", "request_id": json_ctx.operation_id})
        await exporter(json_ctx, json_request)
        result = self.exports[-1]
        self.assertEqual(result.content_type, "application/json")
        self.assertEqual(json.loads(result.content), {"schema_version": 1,
            "blocks": [{"type": "text", "id": "legacy_body", "text": self.note.body}]})
        self.assertEqual(self.reads, [(self.ctx, self.note.id, "opaque-v1"), (json_ctx, self.note.id, "opaque-v1")])

    async def test_same_request_remains_identical_when_version_becomes_history(self):
        exporter = ExactNoteExporter(self.notes, self.publish)
        await exporter(self.ctx, self.request)
        self.source.is_current = False
        self.source.generation += 1  # SourceVersion exposes CURRENT generation even for history.
        await exporter(self.ctx, self.request)
        first, retry = self.exports
        self.assertEqual(first, retry)
        self.assertEqual(first.operation_key, "workspace-note-export:" + str(self.ctx.operation_id))
        self.assertEqual(first.version_id, "opaque-v1")

    async def test_wrong_owner_or_version_cannot_reach_publication(self):
        exporter = ExactNoteExporter(self.notes, self.publish)
        for field, value in (("owner_id", "other"), ("document_id", "opaque-v2")):
            original = getattr(self.source, field)
            setattr(self.source, field, value)
            with self.subTest(field=field), self.assertRaises(JarvisHTTPError) as error:
                await exporter(self.ctx, self.request)
            self.assertEqual(error.exception.code, "workspace_export_version")
            self.assertFalse(error.exception.details["effect_committed"])
            setattr(self.source, field, original)
        self.assertEqual(self.exports, [])

    async def test_wrong_admitted_operation_and_unsupported_kind_do_not_read(self):
        exporter = ExactNoteExporter(self.notes, self.publish)
        with self.assertRaises(JarvisHTTPError) as error:
            await exporter(WorkspaceContext(self.ctx.owner_id, self.ctx.session_id, uuid4()), self.request)
        self.assertEqual(error.exception.code, "workspace_export_scope")
        with self.assertRaises(JarvisHTTPError):
            await exporter(self.ctx, self.request.model_copy(update={"artifact_kind": "file", "format": None}))
        self.assertEqual(self.reads, [])
        self.assertEqual(self.exports, [])

    async def test_bad_receipt_is_not_misreported_as_no_effect(self):
        async def wrong_receipt(ctx, export):
            return {"file_id": self.file_id, "byte_count": len(export.content), "content_sha256": "c" * 64}
        with self.assertRaises(JarvisHTTPError) as error:
            await ExactNoteExporter(self.notes, wrong_receipt)(self.ctx, self.request)
        self.assertEqual(error.exception.code, "workspace_export_receipt_mismatch")
        self.assertIsNot(error.exception.details.get("effect_committed"), False)

    async def test_unknown_publisher_failure_keeps_same_identity_for_retry(self):
        attempts = []
        async def lost_response(ctx, export):
            attempts.append(export)
            if len(attempts) == 1:
                raise TimeoutError("The upload outcome is not known")
            return await self.publish(ctx, export)
        exporter = ExactNoteExporter(self.notes, lost_response)
        with self.assertRaises(TimeoutError):
            await exporter(self.ctx, self.request)
        await exporter(self.ctx, self.request)
        self.assertEqual(attempts[0], attempts[1])


if __name__ == "__main__":
    unittest.main()
