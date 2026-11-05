"""Reusable manifested source fixture, never a production owner."""
from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json

import asyncpg

from app.knowledge.models import SourcePublication
from app.knowledge.sources import SourceRepository
from app.migrations.runner import apply_migrations, quote_schema
from evals.second_brain.fixtures import DatabaseFixture


@dataclass
class SourceFixture:
    fixture: DatabaseFixture
    pool: object
    sources: SourceRepository
    owner: str
    other: str
    area: object

    async def publish(self, text="Synthetic user report", **changes):
        return await self.publish_payload({"text": text}, **changes)

    async def publish_payload(self, content, **changes):
        """Retain the supplied synthetic original, not a generated replacement."""
        async with self.pool.acquire() as connection:
            payload = await connection.fetchval(
                "insert into jarvis_context_payloads(owner_id,availability,content) values($1,'available',$2::jsonb) returning id",
                self.owner, json.dumps(content, allow_nan=False),
            )
        request = SourcePublication(**(dict(
            area_id=self.area, namespace="fixture", source_key="source-one", payload_id=payload,
            source_type="experience", trust="user_input", source_time=datetime(2026, 9, 14, 12, tzinfo=UTC),
            time_basis="original synthetic report", expected_revision=0,
        ) | changes))
        return await self.sources.publish(self.owner, request)

    async def ready_jobs(self):
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_memory_outbox set next_attempt_at=clock_timestamp()")


async def close_fixture_pool(pool, *, timeout=10):
    try:
        await asyncio.wait_for(pool.close(), timeout=timeout)
    except BaseException:
        pool.terminate()
        raise


@asynccontextmanager
async def source_fixture(dsn, directory, *, test_name, pool_init=None, migrations=None):
    fixture = DatabaseFixture(dsn, directory, test_name=test_name)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(fixture.cleanup)
        # Full legacy vector-index DDL can exceed 150 seconds on the personal
        # Supabase instance. This bounds bootstrap only, not application queries.
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=300)
        try:
            schema = await fixture.create_schema(connection)
            await apply_migrations(connection, schema=schema, fresh=True, migrations=migrations)
        finally:
            await connection.close(timeout=5)

        async def setup(connection):
            await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,public,extensions")
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, timeout=10,
                                         command_timeout=30, setup=setup, init=pool_init)
        stack.push_async_callback(close_fixture_pool, pool)
        async with pool.acquire() as connection:
            area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Fixture','test') returning id",
                fixture.owners[0],
            )
        yield SourceFixture(fixture, pool, SourceRepository(pool), *fixture.owners, area)
