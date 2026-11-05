"""Offline release-policy/burst checks, not SQL or physical playback evidence."""
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.attention.notifications import QuietSummaryMember, speech_attribution
from app.attention.quiet import OwnerQuietRepository, QuietCommand, QuietDecision, quiet_delivery_decision
from app.attention.quiet_resume import coalesce_ended_episode, summary_message
from app.tools.invocation import ToolInvocationContext
from test_attention_quiet import LedgerFixture
from test_attention_quiet_resume_sql import record_qualification_check


class QualificationReportingTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_lifecycle_leaves_dependents_unreached_not_failed(self):
        report, failures, save = {"checks": []}, [], Mock()
        prerequisite = AsyncMock(side_effect=AssertionError("fixture lifecycle failed"))
        ready = await record_qualification_check(report, failures, save, "lifecycle", prerequisite)
        self.assertFalse(ready)
        dependent = AsyncMock()
        self.assertFalse(await record_qualification_check(report, failures, save, "dependent", dependent,
                                                         prerequisite_ready=ready))
        prerequisite.assert_awaited_once()
        dependent.assert_not_awaited()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["assertion"], "fixture lifecycle failed")
        self.assertEqual(report["checks"][1], {"name": "dependent", "status": "unreached",
                                               "reason": "migration_lifecycle_prerequisite_failed"})
        self.assertEqual(save.call_count, 2)

    async def test_successful_lifecycle_allows_dependent_assertions(self):
        report, failures, save = {"checks": []}, [], Mock()
        ready = await record_qualification_check(report, failures, save, "lifecycle", AsyncMock())
        dependent = AsyncMock()
        self.assertTrue(await record_qualification_check(report, failures, save, "dependent", dependent,
                                                        prerequisite_ready=ready))
        dependent.assert_awaited_once()
        self.assertEqual([item["status"] for item in report["checks"]], ["passed", "passed"])
        self.assertEqual(failures, [])


class QuietResumeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = LedgerFixture()
        self.repo = OwnerQuietRepository(self.db)
        self.owner, self.session = "quiet-resume-offline", uuid4()

    async def command(self, action, **kwargs):
        source = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.db.authorize(self.owner, self.session, source)
        return await self.repo.save(self.owner, self.session, source, QuietCommand(
            action=action, expected_revision=(await self.repo.get(self.owner)).revision, **kwargs))

    async def test_repeated_resume_keeps_first_release_boundary(self):
        await self.command("activate")
        first = await self.command("resume")
        for count in (1, 2, 3):
            with self.subTest(repeats=count):
                self.db.now += timedelta(seconds=10)
                for _ in range(count):
                    latest = await self.command("resume")
                self.assertEqual(latest["quiet"]["episode"]["resumed_at"],
                                 first["quiet"]["episode"]["resumed_at"])

    async def test_direct_questions_and_reconstruction_never_resume_repeated_quiet(self):
        first_id = None
        for count in (1, 2, 3):
            with self.subTest(repeats=count):
                for _ in range(count):
                    result = await self.command("activate")
                    first_id = first_id or result["quiet"]["episode"]["episode_id"]
                    self.assertEqual(result["quiet"]["episode"]["episode_id"], first_id)
                    decision = await quiet_delivery_decision(None, self.owner, kind="solicited_reply", channel="voice")
                    self.assertTrue(decision.allowed)
                    reconstructed = await OwnerQuietRepository(self.db).get(self.owner)
                    self.assertTrue(reconstructed.suppresses_unsolicited)
                    self.assertEqual(reconstructed.revision, result["accepted_revision"])

    async def test_expiry_requires_local_schema_not_a_prompt_or_public_fallback(self):
        await self.command("activate", expires_at=self.db.now + timedelta(seconds=1))
        self.db.now += timedelta(seconds=1)
        decision = await quiet_delivery_decision(self.db, self.owner, kind="unsolicited", channel="voice")
        self.assertEqual(decision.reason, "owner_quiet_expired")
        with patch("app.attention.quiet_resume.attention_quiet_schema_ready", AsyncMock(return_value=False)) as readiness:
            self.assertFalse(await coalesce_ended_episode(SimpleNamespace(), self.db, self.owner, decision))
        readiness.assert_awaited_once_with(self.db)
        self.assertFalse(any("insert into jarvis_attention" in sql for sql, _ in self.db.calls))

    async def test_inactive_and_quiet_do_not_require_schema_42(self):
        with patch("app.attention.quiet_resume.attention_quiet_schema_ready", AsyncMock()) as readiness:
            self.assertTrue(await coalesce_ended_episode(None, None, self.owner,
                QuietDecision(True, "owner_quiet_inactive", "voice")))
            self.assertFalse(await coalesce_ended_episode(None, None, self.owner,
                QuietDecision(False, "owner_quiet_quiet", "voice")))
        readiness.assert_not_awaited()

    def test_summary_is_bounded_metadata_not_a_claim_of_hearing_or_task_success(self):
        members = tuple(QuietSummaryMember(intent_id=uuid4(), generation=1,
            source_namespace=namespace, source_id=uuid4(), source_revision=1,
            source_payload_id=uuid4(), task_revision=1 if namespace == "task_result" else None)
            for namespace in ("task_result", "task_result", "reminder_event"))
        text = summary_message(members)
        self.assertEqual(text, "Quiet-mode summary: 2 task updates, 1 reminder available in your inbox.")
        self.assertLess(len(text.encode("utf-8")), 400)
        self.assertEqual(summary_message(()), "")
        self.assertEqual(speech_attribution("quiet_resume_summary")[1], "attention_ledger")
