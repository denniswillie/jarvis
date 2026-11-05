"""Manual recovery records its actual outcome and cannot wait forever on a pool."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from evals.second_brain.fixtures import DatabaseFixture
from evals.second_brain.source_fixture import close_fixture_pool


class FixtureRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovery_preserves_identity_and_persists_verified_cleanup(self):
        with TemporaryDirectory() as directory:
            original = DatabaseFixture("fixture-only", Path(directory), test_name=self.id())
            original.manifest.update(status="cleanup_failed", cleanup_error_type="TimeoutError")
            original.persist()
            restored = DatabaseFixture.load("fixture-only", original.path)
            evidence = {"remaining_schemas": 0, "remaining_public_owner_rows": 0,
                        "owners": list(original.owners)}
            async def clean(dsn, manifest):
                self.assertEqual(dsn, "fixture-only")
                self.assertEqual(manifest["run_id"], original.run_id)
                self.assertEqual(json.loads(original.path.read_text())["status"], "cleaning")
                return evidence
            with patch("evals.second_brain.fixtures.cleanup_manifest", side_effect=clean):
                self.assertEqual(await restored.cleanup(), evidence)
            saved = json.loads(original.path.read_text())
            self.assertEqual(saved["status"], "cleaned")
            self.assertEqual(saved["cleanup"], evidence)
            self.assertEqual(restored.run_id, original.run_id)
            self.assertEqual(restored.owners, original.owners)
            self.assertEqual(list(Path(directory).glob("*.json")), [original.path])

    async def test_failed_recovery_is_not_reported_as_cleaned(self):
        with TemporaryDirectory() as directory:
            fixture = DatabaseFixture("fixture-only", Path(directory), test_name=self.id())
            with patch("evals.second_brain.fixtures.cleanup_manifest", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    await DatabaseFixture.load("fixture-only", fixture.path).cleanup()
            saved = json.loads(fixture.path.read_text())
            self.assertEqual(saved["status"], "cleanup_failed")
            self.assertEqual(saved["cleanup_error_type"], "TimeoutError")

    def test_loading_foreign_owner_manifest_is_refused_without_rewriting_it(self):
        with TemporaryDirectory() as directory:
            fixture = DatabaseFixture("fixture-only", Path(directory), test_name=self.id())
            fixture.manifest["owners"] = ["not-a-fixture-owner"]
            fixture.persist()
            before = fixture.path.read_bytes()
            with self.assertRaises(ValueError):
                DatabaseFixture.load("fixture-only", fixture.path)
            self.assertEqual(fixture.path.read_bytes(), before)

    async def test_stalled_pool_is_terminated_and_timeout_remains_visible(self):
        async def stalled():
            await asyncio.Future()
        pool = SimpleNamespace(close=AsyncMock(side_effect=stalled), terminate=Mock())
        with self.assertRaises(TimeoutError):
            await close_fixture_pool(pool, timeout=0.01)
        pool.terminate.assert_called_once_with()
        pool.close.assert_awaited_once_with()

    async def test_clean_pool_shutdown_is_not_forced(self):
        pool = SimpleNamespace(close=AsyncMock(), terminate=Mock())
        await close_fixture_pool(pool)
        pool.close.assert_awaited_once_with()
        pool.terminate.assert_not_called()
