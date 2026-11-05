"""Scope read shape, using the real selection service and controlled SQL results.

These tests count calls, not latency. Manifested SQL tests qualify query behavior.
"""
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.knowledge.sources import SourceRepository
from app.live_state.models import SessionScope
from app.live_state.runtime import _SessionBinder


class ScopeConnection:
    def __init__(self, area):
        self.calls = []
        self.session_exists = True
        self.row = dict(area_id=area, area_status="active", revision=None, receipt_event_id=None)

    async def fetchval(self, query, *args):
        self.calls.append((query, args))
        if "from jarvis_sessions" not in query:
            raise AssertionError("Unexpected duplicate scope query")
        return self.session_exists

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        if "for share of a" not in query:
            raise AssertionError("Selected area must remain locked")
        return self.row


class ScopeRoundtripTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="scope-roundtrip-fixture", session_id=uuid4(), area_id=uuid4())
        self.sources = SourceRepository(object())
        self.selections = SessionKnowledgeAreas(self.sources,
            SimpleNamespace(pool=self.sources.pool), self.scope.area_id)
        self.access = Mock()
        self.binder = _SessionBinder(self.sources, self.scope,
            clock=lambda: datetime.now(UTC), area_selections=self.selections, access_check=self.access)
        self.connection = ScopeConnection(self.scope.area_id)

    async def test_default_scope_uses_one_session_lock_and_one_selected_area_read(self):
        await self.binder._session(self.connection)
        self.access.assert_called_once_with(self.scope.owner_id)
        self.assertEqual(len(self.connection.calls), 2)
        self.assertIn("for share", self.connection.calls[0][0])
        self.assertEqual(self.binder.selection, dict(area_id=str(self.scope.area_id),
            area_status="active", selection_revision=0, selection_event_id=None, origin="configured_default"))

    async def test_revision_or_receipt_change_fences_even_when_area_stays_the_same(self):
        await self.binder._session(self.connection)
        self.connection.row.update(revision=1, receipt_event_id=uuid4())
        with self.assertRaises(SourceError) as raised:
            await self.binder._session(self.connection)
        self.assertEqual(raised.exception.code, "live_state_scope_changed")
        self.assertTrue(self.binder.scope_changed)
        self.connection.row.update(revision=None, receipt_event_id=None)
        with self.assertRaises(SourceError):
            await self.binder._session(self.connection)

    async def test_foreign_or_missing_session_rejects_before_area_read(self):
        self.connection.session_exists = False
        with self.assertRaises(SourceError) as raised:
            await self.binder._session(self.connection)
        self.assertEqual(raised.exception.code, "knowledge_session_unavailable")
        self.assertEqual(len(self.connection.calls), 1)
        self.assertIsNone(self.binder.selection)

    async def test_missing_selected_area_never_falls_back(self):
        self.connection.row = None
        with self.assertRaises(SourceError) as raised:
            await self.binder._session(self.connection)
        self.assertEqual(raised.exception.code, "knowledge_area_unavailable")
        self.assertIsNone(self.binder.selection)

    async def test_archived_deleted_and_replaced_area_cannot_authorize_context(self):
        for change in (dict(area_status="archived"), dict(area_status="deleted"), dict(area_id=uuid4())):
            with self.subTest(change=change):
                self.binder.scope_changed = False
                self.connection.row = dict(area_id=self.scope.area_id, area_status="active",
                    revision=None, receipt_event_id=None) | change
                with self.assertRaises(SourceError) as raised:
                    await self.binder._session(self.connection)
                self.assertEqual(raised.exception.code, "live_state_scope_changed")

    async def test_access_rejection_never_touches_database(self):
        self.access.side_effect = PermissionError("fixture")
        with self.assertRaises(PermissionError):
            await self.binder._session(self.connection)
        self.assertEqual(self.connection.calls, [])

    async def test_unscoped_binder_keeps_base_session_validation(self):
        self.binder.scope = self.scope.model_copy(update={"area_id": None})
        self.binder.area_selections = None
        await self.binder._session(self.connection)
        self.assertEqual(len(self.connection.calls), 1)
        self.assertIn("for key share", self.connection.calls[0][0])

    async def test_history_cutoff_and_sessionless_default_are_passed_to_query(self):
        await self.selections.read_in_transaction(self.connection, self.scope.owner_id,
            self.scope.session_id, through_sequence=123)
        self.assertEqual(self.connection.calls[-1][1],
            (self.scope.owner_id, self.scope.session_id, 123, self.scope.area_id))
        self.connection.calls.clear()
        await self.selections.read_in_transaction(self.connection, self.scope.owner_id, None)
        self.assertEqual(len(self.connection.calls), 1)


if __name__ == "__main__":
    unittest.main()
