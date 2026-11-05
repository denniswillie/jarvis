"""Core's manifested SQL lane only; never a provider/device qualification.

Register PRESENTATION_REQUESTS_MIGRATION.sql before running. The isolated
service tests do not establish PostgreSQL capacity, locking or persistence.
"""
import asyncio
from datetime import timedelta
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import PresentationAck, PresentationRepository, PresentationRequest
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core's explicit manifested database lane is required")
class WorkspacePresentationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_replay_expiry_ordering_and_nonblocking_reads(self):
        fixture = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        async with fixture.pool.acquire() as conn:
            registered = await conn.fetchval("select to_regclass('jarvis_workspace_requests')")
            self.assertIsNotNone(registered, "Core must register the presentation migration; a missing schema is not a pass")
            session = await conn.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
            other_session = await conn.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
        repo = PresentationRepository(fixture.pool)
        contexts = [WorkspaceContext(fixture.owner, session, uuid4()) for _ in range(9)]
        target = {"action": "open", "artifact_kind": "note", "artifact_id": uuid4(),
                  "version_id": "fixture-exact-v1", "content_ready": True}
        async def reserve(ctx):
            return await repo.reserve(ctx, digest=ctx.operation_id.hex * 2, target=target)

        # Nine real transactions racing must reserve exactly eight slots.
        outcomes = await asyncio.wait_for(asyncio.gather(*(reserve(ctx) for ctx in contexts),
                                                       return_exceptions=True), timeout=25)
        successes = [item for item in outcomes if isinstance(item, PresentationRequest)]
        failures = [item for item in outcomes if isinstance(item, BaseException)]
        self.assertEqual(len(successes), 8)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], JarvisHTTPError)
        self.assertEqual(failures[0].code, "presentation_capacity")
        first = successes[0]
        ctx = next(value for value in contexts if value.operation_id == first.request_id)
        self.assertEqual(await reserve(ctx), first)  # Replay precedes capacity.
        with self.assertRaises(JarvisHTTPError) as conflict:
            await repo.reserve(ctx, digest="f" * 64, target=target)
        self.assertEqual(conflict.exception.code, "operation_conflict")
        with self.assertRaises(JarvisHTTPError) as wrong_session:
            await repo.get(WorkspaceContext(fixture.owner, other_session), first.request_id)
        self.assertEqual(wrong_session.exception.code, "workspace_operation_scope")

        claimed = await repo.ack(ctx, first.request_id, PresentationAck(session_id=session, state="claimed"))
        self.assertEqual(await repo.ack(ctx, first.request_id,
            PresentationAck(session_id=session, state="claimed")), claimed)

        # Fixture-only clock aging, not a production method for expiring claims.
        async with repo._transaction(ctx) as conn:
            now = await conn.fetchval("select clock_timestamp()")
            for request in [claimed] + [item for item in successes if item.request_id != first.request_id]:
                aged = PresentationRequest.model_validate(request.model_dump() | {
                    "created_at": now - timedelta(minutes=6), "expires_at": now - timedelta(minutes=1)})
                await repo._save(conn, ctx, aged)
        self.assertEqual((await repo.get(ctx, first.request_id)).state, "claimed")
        expired_id = next(item.request_id for item in successes if item.request_id != first.request_id)
        self.assertEqual((await repo.get(ctx, expired_id)).state, "expired")
        self.assertEqual((await repo.ack(ctx, expired_id,
            PresentationAck(session_id=session, state="claimed"))).state, "expired")

        # A new reservation reuses expired capacity without changing the claim.
        replacement = WorkspaceContext(fixture.owner, session, uuid4())
        fresh = await reserve(replacement)
        self.assertEqual(fresh.state, "pending")
        await repo.ack(replacement, fresh.request_id, PresentationAck(session_id=session, state="dismissed"))
        # Recent terminal rows and expired pending rows cannot evict the old claim.
        for _ in range(10):
            terminal_ctx = WorkspaceContext(fixture.owner, session, uuid4())
            terminal = await reserve(terminal_ctx)
            await repo.ack(terminal_ctx, terminal.request_id,
                          PresentationAck(session_id=session, state="dismissed"))
        listed = await repo.list(ctx)
        self.assertEqual(len(listed), 8)
        self.assertEqual(listed[0].request_id, first.request_id)
        self.assertEqual(listed[0].state, "claimed")

        # Hold BOTH journal mutation lock and one row lock in another connection.
        # Pure polling must finish before those locks are released.
        async with repo._transaction(ctx) as writer:
            await repo._row(writer, ctx, first.request_id)
            read, items = await asyncio.wait_for(asyncio.gather(
                repo.get(ctx, first.request_id), repo.list(ctx)), timeout=5)
            self.assertEqual(read.state, "claimed")
            self.assertEqual(items[0].request_id, first.request_id)

        shown_ack = PresentationAck(session_id=session, state="displayed")
        shown = await repo.ack(ctx, first.request_id, shown_ack)
        self.assertEqual(shown.state, "displayed")  # Claimed work can finish after offer expiry.
        self.assertEqual(await repo.ack(ctx, first.request_id, shown_ack), shown)
        self.assertEqual((await repo.ack(ctx, first.request_id,
            PresentationAck(session_id=session, state="claimed"))).state, "displayed")
        with self.assertRaises(JarvisHTTPError) as terminal_conflict:
            await repo.ack(ctx, first.request_id, PresentationAck(session_id=session, state="failed"))
        self.assertEqual(terminal_conflict.exception.code, "presentation_terminal_conflict")


if __name__ == "__main__":
    unittest.main()
