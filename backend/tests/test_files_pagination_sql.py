"""Actual source-backed library browsing; in-memory object storage is explicit.

Run only the named new case in the manifested SQL lane. The inherited fixture
uses a synthetic JPEG and real capture/publication/source/HTTP access boundaries.
"""
import asyncio
from uuid import UUID

import httpx

from app.files.pagination import encode_cursor
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from test_second_brain_saved_files import SavedFileDatabaseTests


class FilePaginationSQLTests(SavedFileDatabaseTests):
    async def test_media_filter_before_limit_pages_and_access(self):
        async with asyncio.timeout(180):
            media = [await self.save(filename=f"library-{i}.jpg", operation_key=f"library-{i}") for i in range(3)]
            documents = []
            for i in range(9):
                documents.append(await self.files.create_text_file(owner_id=self.f.owner, session_id=self.session,
                    filename=f"newer-{i}.txt", content="Synthetic newer document.",
                    description="Pagination fixture", operation_key=f"newer-{i}"))
            # Nine newer documents previously hid every image behind the eight-row cap.
            old = await self.files.list_files(owner_id=self.f.owner, query=None, limit=8)
            self.assertEqual(len(old), 8)
            self.assertEqual([f.id for f in old], [f.id for f in reversed(documents[-8:])])
            first = await self.files.list_file_page(owner_id=self.f.owner, query=None, limit=2, media_only=True)
            self.assertEqual([f.id for f in first.files], [media[2].id, media[1].id])
            self.assertIsNotNone(first.next_cursor)
            second = await self.files.list_file_page(owner_id=self.f.owner, query=None, limit=2,
                media_only=True, cursor=first.next_cursor)
            self.assertEqual([f.id for f in second.files], [media[0].id])
            self.assertIsNone(second.next_cursor)
            self.assertTrue(all(f.memory_source for f in first.files + second.files))
            exact = encode_cursor(media[1].created_at, media[1].id)
            self.assertEqual(exact, first.next_cursor)
            query = await self.files.list_file_page(owner_id=self.f.owner, query="library-1", media_only=True)
            self.assertEqual([f.id for f in query.files], [media[1].id])
            with self.assertRaises(SourceError):
                await self.files.list_file_page(owner_id=self.f.other, query=None, media_only=True, cursor=exact)
            async with self.f.pool.acquire() as connection:
                other_area = await connection.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain)
                    values($1,'Other page area','test') returning id""", self.f.owner)
            other = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=other_area)
            self.assertEqual((await other.list_file_page(owner_id=self.f.owner, query=None,
                media_only=True, cursor=exact)).files, [])
            # The actual HTTP endpoint forwards the opaque cursor and retains source identities.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.application()),
                    base_url="http://fixture.invalid", headers={"Authorization":
                    "Bearer synthetic-file-test-only-secret"}) as client:
                response = await client.get("/files", params={"session_id": str(self.session),
                    "media_only": "true", "limit": 2})
                self.assertEqual(response.status_code, 200, response.text)
                page = response.json()
                self.assertEqual([f["id"] for f in page["files"]], [str(f.id) for f in first.files])
                self.assertEqual(page["next_cursor"], exact)
                self.assertTrue(page["page_full"])
                response = await client.get("/files", params={"session_id": str(self.session),
                    "media_only": "true", "cursor": exact, "limit": 2})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIsNone(response.json()["next_cursor"])
                invalid = await client.get("/files", params={"session_id": str(self.session),
                    "media_only": "true", "cursor": "?"})
                self.assertEqual(invalid.status_code, 422)
            await self.f.sources.delete_source(self.f.owner, self.f.area,
                UUID(media[0].memory_source["source_id"]), expected_revision=1)
            self.assertEqual((await self.files.list_file_page(owner_id=self.f.owner, query=None,
                media_only=True, cursor=exact)).files, [])
