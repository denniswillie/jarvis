"""Isolated SQL upgrade and policy bounds, not production media admission proof."""
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.migrations.runner import apply_migrations, load_migrations
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested Core database lane required",
)
class FilePublicationPolicyMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_document_upgrade_and_immutable_policy_bounds(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        migrations = tuple(item for item in load_migrations() if item.version <= 47)
        self.assertEqual(migrations[-1].version, 47)
        fixture = await self.enterAsyncContext(source_fixture(
            dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
            test_name=self.id(), migrations=migrations[:-1],
        ))

        async def insert(connection, *, size, policy=None, limit=None):
            payload = await connection.fetchval(
                "insert into jarvis_context_payloads(owner_id,availability,content) "
                "values($1,'available','{}'::jsonb) returning id", fixture.owner,
            )
            values = [fixture.owner, fixture.area, str(uuid4()), uuid4(), payload, size]
            columns = ""
            extra_values = ""
            if policy is not None:
                columns = ",transfer_policy,transfer_limit_bytes"
                extra_values = ",$7,$8"
                values.extend((policy, limit))
            return await connection.fetchrow(
                "insert into jarvis_file_publications(owner_id,area_id,operation_key,file_id,"
                "metadata_payload_id,size_bytes,parameters_digest,content_sha256,storage_bucket,storage_path"
                + columns + ") values($1,$2,$3,$4,$5,$6,repeat('a',64),repeat('b',64),"
                "'synthetic-media','fixture/' || $3" + extra_values + ") returning *", *values,
            )

        async with fixture.pool.acquire() as connection:
            legacy = await insert(connection, size=8388608)

        # No pool borrowers or active transactions cross the administrative DDL.
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
        try:
            self.assertEqual(await apply_migrations(
                connection, schema=fixture.fixture.manifest["schemas"][0], migrations=migrations,
            ), (47,))
        finally:
            await connection.close(timeout=5)
        await fixture.pool.expire_connections()

        async with fixture.pool.acquire() as connection:
            upgraded = await connection.fetchrow(
                "select * from jarvis_file_publications where id=$1", legacy["id"],
            )
            self.assertEqual(upgraded["transfer_policy"], "document_v1")
            self.assertEqual(upgraded["transfer_limit_bytes"], 8388608)
            self.assertEqual(upgraded["size_bytes"], legacy["size_bytes"])
            for policy, maximum in (
                ("document_v1", 8388608),
                ("mp4_capture_v1", 67108864),
                ("mp4_output_v1", 33554432),
            ):
                with self.subTest(policy=policy):
                    row = await insert(connection, size=maximum, policy=policy, limit=maximum)
                    self.assertEqual(row["size_bytes"], maximum)
                    for size, limit in ((maximum + 1, maximum), (1, maximum + 1),
                                        (0, 0), (-1, maximum), (1001, 1000)):
                        with self.subTest(size=size, limit=limit):
                            with self.assertRaises(asyncpg.CheckViolationError):
                                await insert(connection, size=size, policy=policy, limit=limit)
                    with self.assertRaises(asyncpg.CheckViolationError):
                        await connection.execute(
                            "update jarvis_file_publications set transfer_limit_bytes=$2 where id=$1",
                            row["id"], maximum - 1,
                        )
            with self.assertRaises(asyncpg.CheckViolationError):
                await insert(connection, size=1, policy="unknown_policy", limit=1000)
            with self.assertRaises(asyncpg.CheckViolationError):
                await insert(connection, size=8388609)
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    "update jarvis_file_publications set transfer_policy='mp4_capture_v1' where id=$1",
                    legacy["id"],
                )
