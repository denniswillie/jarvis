"""File/presentation commands remain independent of retired note services."""
from datetime import UTC, datetime
from types import SimpleNamespace
import json
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.files.models import JarvisFile
from app.workspace.commands import WorkspaceCommands
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import PresentationService
from test_workspace_presentation import MemoryRequests

OWNER, SESSION = "test-owner", uuid4()

def retained_file():
    return JarvisFile(id=uuid4(), session_id=SESSION, filename="answer.txt", content_type="text/plain",
        storage_bucket="private", storage_path="private/object", size_bytes=4, description="Saved answer",
        source="assistant", source_visual_frame_id=None, created_at=datetime(2026, 11, 1, tzinfo=UTC),
        metadata={}, memory_source={"document_id": "opaque_file_v1", "revision": 1})

class ExactRequests(MemoryRequests):
    async def get(self, ctx, request_id, *, digest=None):
        if self.value is not None and request_id != self.value.request_id:
            return None
        return await super().get(ctx, request_id, digest=digest)

class WorkspaceCommandSchemaTests(unittest.TestCase):
    def test_notes_are_not_advertised_or_accepted(self):
        commands = WorkspaceCommands(files=object())
        self.assertNotIn("live_note", commands.schemas())
        for name, args in (("live_note", {"instruction": "Edit"}),
                           ("live_read", {"domain": "artifact", "artifact_kind": "note"}),
                           ("live_read", {"domain": "artifact"})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                commands.normalize(name, args)

    def test_commands_are_only_advertised_with_backing_services(self):
        self.assertEqual(set(WorkspaceCommands().schemas()), {"live_read"})
        self.assertNotIn("live_present", WorkspaceCommands(files=object()).schemas())
        commands = WorkspaceCommands(SimpleNamespace(exporter=None), files=object())
        self.assertEqual(commands.schemas()["live_present"]["properties"]["action"]["enum"], ["open"])
        commands = WorkspaceCommands(SimpleNamespace(exporter=object()), files=object())
        self.assertEqual(set(commands.schemas()["live_present"]["properties"]["action"]["enum"]), {"open", "download"})
        self.assertNotIn("DocumentContent", json.dumps(commands.schemas()))

    def test_file_search_and_exact_presentation(self):
        commands = WorkspaceCommands(SimpleNamespace(exporter=object()), files=object())
        self.assertEqual(commands.normalize("live_read", {"domain": "artifact", "artifact_kind": "file"})["limit"], 8)
        args = {"action": "download", "artifact_kind": "file", "artifact_id": str(uuid4())}
        self.assertEqual(commands.normalize("live_present", args), args)
        for values in ({"domain": "artifact", "artifact_kind": "file", "cursor": "page"},
                       {"domain": "artifact", "artifact_kind": "file", "artifact_id": str(uuid4()), "query": "x"}):
            with self.assertRaises(ValueError):
                commands.normalize("live_read", values)
        with self.assertRaises(ValueError):
            commands.normalize("live_present", args | {"format": "text"})

    def test_json_authority_and_bounds_rejected(self):
        commands = WorkspaceCommands(files=object())
        for values in (None, [], {"domain": "artifact", "owner_id": OWNER},
                       {"domain": "artifact", 1: "coerced key"},
                       {"domain": "operation", "operation_id": str(uuid4()), "session_id": str(SESSION)},
                       {"domain": "artifact", "artifact_kind": "file", "limit": True},
                       {"domain": "artifact", "artifact_kind": "file", "limit": 9},
                       {"domain": "artifact", "artifact_kind": "file", "query": float("nan")},
                       {"domain": "artifact", "artifact_kind": "file", "query": "x" * 501}):
            with self.subTest(values=repr(values)[:70]), self.assertRaises(ValueError):
                commands.normalize("live_read", values)

class WorkspaceCommandCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_file_and_bounded_search_without_notes(self):
        ctx, file = WorkspaceContext(OWNER, SESSION), retained_file()
        repository = SimpleNamespace(list_files=AsyncMock(return_value=[file]))
        files = SimpleNamespace(get=AsyncMock(return_value=file), scoped=AsyncMock(return_value=repository))
        commands = WorkspaceCommands(files=files)
        args = {"domain": "artifact", "artifact_kind": "file", "artifact_id": str(file.id), "version_id": "opaque_file_v1"}
        result = await commands.execute(ctx, "live_read", args)
        files.get.assert_awaited_once_with(OWNER, SESSION, file.id, "opaque_file_v1")
        self.assertNotIn("storage_path", result["file"])
        self.assertNotIn("storage_bucket", result["file"])
        listing = await commands.execute(ctx, "live_read", {"domain": "artifact", "artifact_kind": "file", "limit": 1})
        self.assertTrue(listing["page_full"])
        self.assertNotIn("next_cursor", listing)
        files.get.side_effect = JarvisHTTPError(404, "file_version_unavailable", "Unavailable")
        with self.assertRaises(JarvisHTTPError):
            await commands.execute(ctx, "live_read", args)
        with self.assertRaises(TypeError):
            await commands.execute({}, "live_read", args)

    async def test_presentation_receipt_does_not_resolve_or_repeat(self):
        requests = ExactRequests(datetime(2026, 11, 1, tzinfo=UTC))
        resolve = AsyncMock(return_value="file:v1")
        service = PresentationService(requests, resolver=resolve, exporter=object())
        commands = WorkspaceCommands(service, files=object())
        ctx = WorkspaceContext(OWNER, SESSION, uuid4())
        args = {"action": "open", "artifact_kind": "file", "artifact_id": str(uuid4())}
        await commands.execute(ctx, "live_present", args)
        resolve.reset_mock()
        receipt = await commands.execute(ctx, "live_read", {"domain": "operation", "operation_id": str(ctx.operation_id)})
        self.assertEqual(receipt["operation"]["outcome"], "accepted")
        resolve.assert_not_awaited()
        self.assertEqual(requests.reservations, 1)
        unknown = await commands.execute(ctx, "live_read", {"domain": "operation", "operation_id": str(uuid4())})
        self.assertEqual(unknown["outcome"], "outcome_unknown")
        self.assertIsNone(await commands.receipt(ctx, "live_note", uuid4()))
