"""Small deterministic storage double shared by file publication boundary tests."""
import asyncio

from app.files.models import StoredFileObject
from app.files.storage import JarvisFileStorageClient
from app.knowledge.models import SourceError


class MemoryStorage:
    bucket = "synthetic-files"
    build_file_path = JarvisFileStorageClient.build_file_path

    def __init__(self):
        self.objects, self.uploads, self.deletes = {}, 0, 0
        self.mode = None
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def upload_bytes(self, *, path, content, content_type):
        self.uploads += 1
        if self.mode == "pause_before":
            self.entered.set()
            await self.release.wait()
        if self.mode == "fail_before":
            raise RuntimeError("Synthetic upload uncertainty")
        self.objects[(self.bucket, path)] = content
        if self.mode == "pause_after":
            self.entered.set()
            await self.release.wait()
        if self.mode == "fail_after":
            raise RuntimeError("Synthetic lost upload acknowledgement")
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=len(content))

    async def read_object(self, *, bucket, path, max_bytes):
        content = self.objects.get((bucket, path))
        if content is not None and len(content) > max_bytes:
            raise SourceError("fixture_object_oversized", "Synthetic object too large")
        return content

    async def delete_object(self, *, bucket, path):
        self.deletes += 1
        self.objects.pop((bucket, path), None)
        if self.mode == "delete_ack_lost":
            raise RuntimeError("Synthetic lost delete acknowledgement")
