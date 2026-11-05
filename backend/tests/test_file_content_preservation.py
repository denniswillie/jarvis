"""File creation preserves literal code/text while ordinary tool strings stay trimmed."""
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.files.models import JarvisFile
from app.files.tools import create_file_tools
from app.tools.inputs import ToolInputError, required_string
from evals.second_brain.file_storage_fixture import MemoryStorage


class RequiredLiteralStringTests(unittest.TestCase):
    def test_default_behavior_is_unchanged_and_literal_bounds_count_whitespace(self):
        self.assertEqual(required_string({"text": " \nx\t "}, "text", max_length=1), "x")
        value = " \nx\t "
        self.assertEqual(required_string({"text": value}, "text", preserve_whitespace=True), value)
        with self.assertRaises(ToolInputError):
            required_string({"text": value}, "text", max_length=1, preserve_whitespace=True)
        for value in (None, "", " \n\t", 42):
            with self.subTest(value=value), self.assertRaises(ToolInputError):
                required_string({"text": value}, "text", preserve_whitespace=True)


class FileLiteralContentTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_and_legacy_tools_preserve_exact_content(self):
        content = "\n    return middle\n\n"
        for source_backed in (False, True):
            with self.subTest(source_backed=source_backed):
                storage = MemoryStorage()
                file = JarvisFile(id=uuid4(), session_id=uuid4(), filename="code.txt",
                    content_type="text/plain; charset=utf-8", storage_bucket=storage.bucket,
                    storage_path="fixture", size_bytes=len(content.encode()), description="Draft",
                    source="assistant", source_visual_frame_id=None, created_at=datetime.now(UTC))
                create = AsyncMock(return_value=file)
                repository = SimpleNamespace(**{"create_text_file" if source_backed else "create_file": create})
                tool = next(t for t in create_file_tools(repository=repository, storage=storage,
                    owner_id="literal-content-fixture", session_id=file.session_id) if t.name=="create_text_file")
                result = await tool.handler({"filename":"code.txt", "content":content})
                self.assertEqual(result["status"], "success", result)
                if source_backed:
                    self.assertEqual(create.await_args.kwargs["content"], content)
                else:
                    self.assertEqual(list(storage.objects.values()), [content.encode()])

    async def test_overlong_literal_content_fails_before_storage_or_source_write(self):
        create = AsyncMock()
        storage = MemoryStorage()
        tool = next(t for t in create_file_tools(repository=SimpleNamespace(create_text_file=create),
            storage=storage, owner_id="literal-content-fixture", session_id=uuid4())
            if t.name=="create_text_file")
        result = await tool.handler({"filename":"code.txt", "content":"x" + " " * 200000})
        self.assertEqual(result["status"], "failed")
        create.assert_not_awaited()
        self.assertFalse(storage.objects)

