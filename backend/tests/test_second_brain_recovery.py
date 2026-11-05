"""Fail-closed operator checkpoint checks, without a model or database."""
import unittest
from unittest.mock import AsyncMock, patch
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from uuid import uuid4

from app.knowledge.lifecycle_journal import JournalCheckpoint
from app.knowledge.recovery import (
    RecoveryCheckpoint, RecoveryNotReady, _check_journals, _check_state,
    _ledger_digest, verify_paused_recovery,
)


def checkpoint(**changes):
    return RecoveryCheckpoint(**(dict(
        schema_name="fixture", system_identifier="123456", timeline=1,
        target_lsn="0/6000028", migration_digest="a" * 64,
        journals=(JournalCheckpoint(owner_id="fixture", epoch=uuid4(), through_sequence=2),),
    ) | changes))


class RecoveryContractTests(unittest.TestCase):
    def test_refused_existing_output_directory_is_not_overwritten(self):
        from evals.second_brain.rehearse_recovery import main
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            path.write_text("previous report")
            with patch("sys.argv", ["rehearse_recovery", "--root", root]), \
                 patch("evals.second_brain.rehearse_recovery.postgres_recovery_fixture",
                       side_effect=FileExistsError), redirect_stdout(StringIO()), \
                 self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(path.read_text(), "previous report")

    def test_checkpoint_rejects_unbounded_ambiguous_or_injected_inputs(self):
        pin = checkpoint()
        for changes in (
            {"schema_name": 'public";drop schema public;--'}, {"target_lsn": "0/0;shutdown"},
            {"target_lsn": "FFFFFFFFF/1"}, {"timeline": True}, {"timeline": 0},
            {"system_identifier": 123}, {"migration_digest": "bad"}, {"journals": ()},
            {"journals": pin.journals * 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                checkpoint(**changes)

    def test_epoch_exact_cursor_and_contiguity_are_required(self):
        pin = checkpoint()
        row = dict(owner_id="fixture", epoch=pin.journals[0].epoch, last_sequence=2,
                   event_count=2, first=1, last=2)
        _check_journals(pin, [row])
        for change in (
            {"owner_id": "other"}, {"epoch": uuid4()}, {"last_sequence": 1},
            {"last_sequence": 3}, {"event_count": 1}, {"first": 2}, {"last": 3},
        ):
            with self.subTest(change=change), self.assertRaises(RecoveryNotReady):
                _check_journals(pin, [row | change])
        with self.assertRaises(RecoveryNotReady):
            _check_journals(pin, [])

    def test_empty_initialized_journal_is_not_a_missing_journal(self):
        pin = checkpoint(journals=(JournalCheckpoint(owner_id="fixture", epoch=uuid4(), through_sequence=0),))
        _check_journals(pin, [dict(owner_id="fixture", epoch=pin.journals[0].epoch,
                                  last_sequence=0, event_count=0, first=None, last=None)])

    def test_primary_pause_request_wrong_cluster_timeline_and_wal_fail(self):
        pin = checkpoint()
        state = dict(in_recovery=True, pause_state="paused", system_identifier=pin.system_identifier,
                     timeline=1, target_matches=True, target_reached=True)
        _check_state(pin, state)
        for change in (
            {"in_recovery": False}, {"pause_state": "pause requested"}, {"pause_state": "not paused"},
            {"system_identifier": "other"}, {"timeline": 2}, {"target_matches": False},
            {"target_reached": False},
        ):
            with self.subTest(change=change), self.assertRaises(RecoveryNotReady):
                _check_state(pin, state | change)

    def test_ledger_digest_changes_with_any_migration_identity(self):
        ledger = [dict(version=1, name="foundations", checksum="a" * 64)]
        self.assertEqual(_ledger_digest(ledger), _ledger_digest(ledger))
        for change in ({"version": 2}, {"name": "other"}, {"checksum": "b" * 64}):
            self.assertNotEqual(_ledger_digest(ledger), _ledger_digest([ledger[0] | change]))
        with self.assertRaises(RecoveryNotReady):
            _ledger_digest([])


class RecoveryVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_primary_is_rejected_in_read_only_transaction_before_pause_function(self):
        class Connection:
            def __init__(self):
                self.context = AsyncMock()
                self.fetchval = AsyncMock(return_value=False)
            def is_in_transaction(self):
                return False
            def transaction(self, **kwargs):
                self.options = kwargs
                return self.context
        connection = Connection()
        with self.assertRaisesRegex(RecoveryNotReady, "recovery_not_paused"):
            await verify_paused_recovery(connection, checkpoint())
        self.assertEqual(connection.options, {"isolation": "repeatable_read", "readonly": True})
        connection.fetchval.assert_awaited_once_with("select pg_is_in_recovery()")
