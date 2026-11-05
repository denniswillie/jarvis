"""Offline retirement diagnostics and a pure stale-source-snapshot reproduction.

The actual rebuild method runs with minimal session fields and a controlled
retirement coroutine. No provider retirement, socket, DB or native run occurs.
The version-mismatch reproduction establishes possible behavior, not the cause
of any captured native failure.
"""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from app.knowledge.models import SourceError
from app.realtime import session as session_module
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError, LiveResponseSources, ResponseSourceSnapshot


FORMAT = "source_context_retirement_diagnostic cause_type=%s cause_code=%s caller_line=%s"
CODES = (
    "live_context_revoked", "live_context_changed", "live_context_transaction_required",
    "live_context_source_invalid", "live_context_source_limit",
    "live_context_storage_unavailable", "live_context_response_limit",
)


class SourceContextRetirementDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def owner(self, *, close_reason="unknown", closed=False):
        owner = SimpleNamespace(
            _closed=asyncio.Event(), _context_revoked=False, _context_reset_id=None,
            _close_reason=close_reason, _context_retirement_task=None,
            _retire_source_context=AsyncMock(return_value=None),
        )
        if closed:
            owner._closed.set()
        self.addAsyncCleanup(self.join_retirement, owner)
        return owner

    async def join_retirement(self, owner):
        if owner._context_retirement_task is not None:
            await asyncio.wait_for(owner._context_retirement_task, 1)

    async def assert_retirement(self, owner, *, close_reason="live_context_rebuild_required"):
        self.assertTrue(owner._context_revoked)
        self.assertEqual(UUID(owner._context_reset_id).version, 4)
        self.assertEqual(owner._close_reason, close_reason)
        self.assertIsInstance(owner._context_retirement_task, asyncio.Task)
        await self.join_retirement(owner)
        owner._retire_source_context.assert_awaited_once_with()

    def diagnostic(self, warning, cause_type, cause_code, *, caller_line=None):
        warning.assert_called_once_with(FORMAT, cause_type, cause_code, caller_line)
        self.assertEqual(warning.call_args.kwargs, {})  # No exc_info/stack traceback.
        rendered = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertNotIn(session_module.__file__, rendered)
        return rendered

    async def test_active_except_retains_original_live_context_error_codes(self):
        for code in CODES:
            with self.subTest(code=code):
                owner = self.owner()
                original = LiveContextError(code)
                with patch.object(session_module.LOGGER, "warning") as warning:
                    try:
                        raise original
                    except LiveContextError as caught:
                        self.assertIs(sys.exception(), original)
                        RealtimeAudioSession._begin_context_rebuild(owner)
                        self.assertIs(sys.exception(), original)
                        self.assertIs(caught, original)
                        self.assertEqual(caught.code, code)
                rendered = self.diagnostic(warning, "LiveContextError", code)
                self.assertNotIn(str(original), rendered)
                await self.assert_retirement(owner)

    async def test_source_error_type_and_code_are_filtered_independently(self):
        for code, expected in (("live_context_revoked", "live_context_revoked"),
                               ("PRIVATE_SOURCE_CODE", "unrecognized_code")):
            with self.subTest(code=code):
                owner = self.owner()
                with patch.object(session_module.LOGGER, "warning") as warning:
                    try:
                        raise SourceError(code, "PRIVATE_SOURCE_TEXT api_key=PRIVATE_KEY")
                    except SourceError:
                        RealtimeAudioSession._begin_context_rebuild(owner)
                rendered = self.diagnostic(warning, "SourceError", expected)
                self.assertNotIn("PRIVATE", rendered)
                await self.assert_retirement(owner)

    async def test_timeout_is_distinguished_from_source_revocation(self):
        owner = self.owner()
        with patch.object(session_module.LOGGER, "warning") as warning:
            try:
                async with asyncio.timeout(0):
                    await asyncio.sleep(0)
            except TimeoutError as error:
                self.assertIs(sys.exception(), error)
                RealtimeAudioSession._begin_context_rebuild(owner)
        rendered = self.diagnostic(warning, "TimeoutError", "unrecognized_code")
        self.assertNotIn("cause_code=live_context_revoked", rendered)
        await self.assert_retirement(owner)

    async def test_unknown_secret_type_code_and_text_are_redacted(self):
        secret_type = type("PRIVATE_CUSTOMER_EXCEPTION_TYPE", (RuntimeError,), {})
        for code in ("PRIVATE_BEARER_CODE", None, 123, {"token": "PRIVATE_TOKEN"}):
            with self.subTest(code_type=type(code).__name__):
                owner = self.owner()
                error = secret_type("PRIVATE_TRANSCRIPT /PRIVATE/path api_key=PRIVATE_KEY")
                error.code = code
                with patch.object(session_module.LOGGER, "warning") as warning:
                    try:
                        raise error
                    except RuntimeError:
                        RealtimeAudioSession._begin_context_rebuild(owner)
                rendered = self.diagnostic(warning, "unrecognized_error", "unrecognized_code")
                self.assertNotIn("PRIVATE", rendered)
                self.assertNotIn(owner._context_reset_id, rendered)
                await self.assert_retirement(owner)

    async def test_no_exception_logs_after_fence_flags_but_before_scheduling(self):
        owner = self.owner()
        self.assertIsNone(sys.exception())
        observations = []

        def observe_order(*args, **kwargs):
            # Capture only: production deliberately swallows logger errors, so
            # assertions here would not reliably fail the test.
            observations.append((
                owner._context_revoked, owner._context_reset_id,
                owner._close_reason, owner._context_retirement_task,
                owner._retire_source_context.call_count,
            ))

        with patch.object(session_module.LOGGER, "warning", side_effect=observe_order) as warning:
            RealtimeAudioSession._begin_context_rebuild(owner)
        self.assertEqual(len(observations), 1)
        revoked, reset_id, close_reason, retirement_task, retirement_calls = observations[0]
        self.assertTrue(revoked)
        self.assertEqual(UUID(reset_id).version, 4)
        self.assertEqual(reset_id, owner._context_reset_id)
        self.assertEqual(close_reason, "live_context_rebuild_required")
        self.assertIsNone(retirement_task)
        self.assertEqual(retirement_calls, 0)
        self.diagnostic(warning, "none", "none")
        await self.assert_retirement(owner)

    async def test_repeated_retirement_does_not_duplicate_log_reset_or_task(self):
        owner = self.owner()
        with patch.object(session_module.LOGGER, "warning") as warning:
            RealtimeAudioSession._begin_context_rebuild(owner)
            reset_id, retirement = owner._context_reset_id, owner._context_retirement_task
            for _ in range(3):
                RealtimeAudioSession._begin_context_rebuild(owner)
            await retirement
            RealtimeAudioSession._begin_context_rebuild(owner)
        self.diagnostic(warning, "none", "none")
        self.assertEqual(owner._context_reset_id, reset_id)
        self.assertIs(owner._context_retirement_task, retirement)
        await self.assert_retirement(owner)

    async def test_closed_session_does_not_log_change_flags_or_schedule(self):
        owner = self.owner(closed=True)
        with patch.object(session_module.LOGGER, "warning") as warning:
            for _ in range(3):
                RealtimeAudioSession._begin_context_rebuild(owner)
        warning.assert_not_called()
        self.assertFalse(owner._context_revoked)
        self.assertIsNone(owner._context_reset_id)
        self.assertEqual(owner._close_reason, "unknown")
        self.assertIsNone(owner._context_retirement_task)
        owner._retire_source_context.assert_not_called()

    async def test_logging_failure_still_fences_and_schedules_retirement(self):
        for original_reason in ("unknown", "PRIVATE_EXISTING_CLOSE_REASON"):
            with self.subTest(existing_reason=original_reason != "unknown"):
                owner = self.owner(close_reason=original_reason)
                with patch.object(session_module.LOGGER, "warning",
                                  side_effect=RuntimeError("PRIVATE_LOG_SINK_FAILURE")) as warning:
                    try:
                        raise LiveContextError("live_context_changed")
                    except LiveContextError as original:
                        RealtimeAudioSession._begin_context_rebuild(owner)
                        self.assertIs(sys.exception(), original)
                rendered = self.diagnostic(warning, "LiveContextError", "live_context_changed")
                self.assertNotIn("PRIVATE", rendered)
                await self.assert_retirement(owner, close_reason=(
                    "live_context_rebuild_required" if original_reason == "unknown" else original_reason))

    async def test_exception_metadata_failure_cannot_prevent_retirement(self):
        class BrokenMetadataError(Exception):
            @property
            def code(self):
                raise RuntimeError("PRIVATE_METADATA_FAILURE")

        owner = self.owner()
        with patch.object(session_module.LOGGER, "warning") as warning:
            try:
                raise BrokenMetadataError("PRIVATE_ERROR_TEXT")
            except BrokenMetadataError as original:
                RealtimeAudioSession._begin_context_rebuild(owner)
                self.assertIs(sys.exception(), original)
        warning.assert_not_called()
        await self.assert_retirement(owner)

    async def test_caller_line_requires_exact_session_filename_not_basename(self):
        # Controlled code-object filenames exercise the branch without editing
        # production source or claiming these are actual production call sites.
        external = "/PRIVATE_CALLER_DIRECTORY/" + Path(session_module.__file__).name
        for filename, expected_line in ((session_module.__file__, 2), (external, None)):
            with self.subTest(internal=expected_line is not None):
                namespace = {"Session": RealtimeAudioSession}
                exec(compile("def invoke(owner):\n    Session._begin_context_rebuild(owner)\n",
                             filename, "exec"), namespace)
                owner = self.owner()
                with patch.object(session_module.LOGGER, "warning") as warning:
                    namespace["invoke"](owner)
                rendered = self.diagnostic(warning, "none", "none", caller_line=expected_line)
                self.assertNotIn(filename, rendered)
                self.assertNotIn("PRIVATE", rendered)
                await self.assert_retirement(owner)

    async def test_added_source_rejects_old_snapshot_before_any_pool_access(self):
        pool = Mock(spec=["acquire"])
        pool.acquire.side_effect = AssertionError("No database boundary may be reached")
        sources = LiveResponseSources(pool, owner_id="offline-source-fixture",
            session_id=uuid4(), audio_session_id="offline-source-retirement")
        original_source, additional_source = uuid4(), uuid4()
        sources.observe((original_source,))
        # prepare() needs SQL. Seed only its immutable snapshot value here;
        # actual guard() and observe() perform the version-mismatch decision.
        snapshot = ResponseSourceSnapshot(uuid4(), (original_source,), sources._version)
        sources.observe((additional_source,))
        owner = self.owner()
        with patch.object(session_module.LOGGER, "warning") as warning:
            try:
                async with sources.guard(snapshot):
                    self.fail("Old source snapshot must not enter the writer")
            except LiveContextError as error:
                self.assertEqual(error.code, "live_context_changed")
                RealtimeAudioSession._begin_context_rebuild(owner)
        pool.acquire.assert_not_called()
        self.assertEqual(snapshot.sources, (original_source,))
        self.assertNotEqual(snapshot.version, sources._version)
        self.diagnostic(warning, "LiveContextError", "live_context_changed")
        await self.assert_retirement(owner)


if __name__ == "__main__":
    unittest.main()
