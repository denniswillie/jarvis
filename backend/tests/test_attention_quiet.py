"""Offline policy/owner/tool/dispatch tests; not PostgreSQL or media acceptance."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from pydantic import ValidationError

from app.attention.models import AttentionError
from app.attention.push_repository import PushRepository
from app.attention.quiet import (
    CAPABILITY, OWNER_LOCK, OWNER_SHARED_LOCK, OwnerQuietRepository, QuietCommand, quiet_delivery_decision,
)
from app.attention.quiet_tools import create_quiet_tools
from app.attention.repository import AttentionRepository
from app.preferences.repository import PreferenceRepository
from app.preferences.tools import PreferenceToolModule
from app.tools.inputs import ToolInputError
from app.tools.invocation import TaskToolInvocation, ToolInvocationContext, bind_tool_invocation
from app.tools.module import ToolBuildContext


class LedgerFixture:
    """Shared persisted fixture across repository instances; never opens a socket.

    SQL tags implement a narrow fake, not a SQL engine or constraint proof. Tests
    assert the authorization predicates separately. Production SQL is unexecuted.
    """

    def __init__(self):
        self.now = datetime(2026, 11, 1, 9, tzinfo=UTC)
        self.rows = []
        self.payloads = {}
        self.sources = {}
        self.dependencies = []
        self.calls = []
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self):
        async with self.lock:
            yield self

    def authorize(self, owner, session, source):
        self.sources[owner, source.request_event_id, source.request_revision, session] = uuid4()

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        if "attention_quiet:current" in sql:
            owner, capability = args
            rows = [r for r in self.rows if r["owner"] == owner and r["capability"] == capability]
            latest = max(rows, key=lambda r: r["revision"]) if rows else None
            payload = self.payloads[latest["payload"]] if latest else {}
            return {"evaluated_at": self.now, "revision": latest["revision"] if latest else None,
                    "content": payload.get("content"), "availability": payload.get("availability")}
        if "attention_quiet:request" in sql:
            owner, capability, event, revision = args
            matches = [r for r in self.rows if (r["owner"], r["capability"], r["event"], r["request_revision"])
                       == (owner, capability, event, revision)]
            if not matches:
                return None
            row = matches[-1]
            return {"revision": row["revision"], **self.payloads[row["payload"]]}
        raise AssertionError("Unexpected fixture fetchrow: " + sql)

    async def fetchval(self, sql, *args):
        self.calls.append((sql, args))
        if "attention_quiet:authorized_source" in sql:
            return self.sources.get(tuple(args))
        if "attention_quiet:payload" in sql:
            identity = uuid4()
            self.payloads[identity] = {"content": json.loads(args[1]), "availability": "available"}
            return identity
        if "select exists(select 1 from jarvis_sessions" in sql:
            return True
        if "select exists(select 1 from jarvis_attention_attempts" in sql:
            return False
        raise AssertionError("Unexpected fixture fetchval: " + sql)

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if "select id from jarvis_attention_intents" in sql:
            return []
        raise AssertionError("Unexpected fixture fetch: " + sql)

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        if sql in (OWNER_LOCK, OWNER_SHARED_LOCK) or "refresh_jarvis_responsibility_quiet" in sql:
            return "SELECT 1"
        if "attention_quiet:dependency" in sql:
            self.dependencies.append(args)
            return "INSERT 0 1"
        if "attention_quiet:append" in sql:
            owner, capability, revision, event, request_revision, payload = args
            self.rows.append(dict(owner=owner, capability=capability, revision=revision,
                                  event=event, request_revision=request_revision, payload=payload))
            return "INSERT 0 1"
        raise AssertionError("Unexpected fixture execute: " + sql)


class QuietTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pool = LedgerFixture()
        self.repo = OwnerQuietRepository(self.pool)
        self.owner = "offline-quiet-owner"
        self.session = uuid4()

    def source(self, owner=None, session=None):
        source = ToolInvocationContext(provider_call_id="call", request_event_id=uuid4(), request_revision=1)
        self.pool.authorize(owner or self.owner, session or self.session, source)
        return source

    async def save(self, action="activate", revision=0, **kwargs):
        return await self.repo.save(self.owner, self.session, self.source(),
                                    QuietCommand(action=action, expected_revision=revision, **kwargs))

    async def test_default_inactive_and_owner_isolation(self):
        self.assertEqual((await self.repo.get(self.owner)).effective_state, "inactive")
        await self.save()
        self.assertTrue((await self.repo.get(self.owner)).suppresses_unsolicited)
        self.assertFalse((await self.repo.get("other-eval-owner")).suppresses_unsolicited)

    async def test_indefinite_reconstruction_does_not_resume_or_reset(self):
        first = await self.save()
        self.pool.now += timedelta(days=90)
        reconstructed = await OwnerQuietRepository(self.pool).get(self.owner)
        self.assertEqual(reconstructed.revision, 1)
        self.assertEqual(reconstructed.effective_state, "quiet")
        self.assertEqual(str(reconstructed.policy.episode.episode_id), first["quiet"]["episode"]["episode_id"])
        self.assertIsNone(reconstructed.policy.episode.expires_at)
        self.assertEqual(len(self.pool.rows), 1)

    async def test_exact_expiry_at_boundary_without_revision_or_deadline_reset(self):
        expiry = self.pool.now + timedelta(minutes=2)
        await self.save(expires_at=expiry, timezone="Europe/Dublin")
        self.pool.now = expiry - timedelta(microseconds=1)
        self.assertEqual((await self.repo.get(self.owner)).effective_state, "quiet")
        self.pool.now = expiry
        reconstructed = await OwnerQuietRepository(self.pool).get(self.owner)
        self.assertEqual(reconstructed.effective_state, "expired")
        self.assertEqual(reconstructed.policy.episode.expires_at, expiry)
        self.assertEqual(reconstructed.policy.episode.timezone, "Europe/Dublin")
        self.assertEqual(reconstructed.revision, 1)

    async def test_nonfuture_expiry_is_rejected_without_writes(self):
        for expiry in (self.pool.now, self.pool.now - timedelta(seconds=1)):
            with self.assertRaises(ToolInputError) as error:
                await self.save(expires_at=expiry)
            self.assertEqual(error.exception.code, "quiet_expiry_not_future")
        self.assertEqual(self.pool.rows, [])

    async def test_resume_preserves_episode_and_does_not_request_replay(self):
        first = await self.save()
        result = await self.save("resume", 1)
        self.assertEqual(result["quiet"]["revision"], 2)
        self.assertEqual(result["quiet"]["effective_state"], "resumed")
        self.assertEqual(result["quiet"]["episode"]["episode_id"], first["quiet"]["episode"]["episode_id"])
        self.assertFalse(result["quiet"]["automatic_replay_requested"])

    async def test_explicit_revision_updates_same_active_episode(self):
        first = await self.save()
        second = await self.save(revision=1, expires_at=self.pool.now + timedelta(minutes=10))
        self.assertEqual(first["quiet"]["episode"]["episode_id"], second["quiet"]["episode"]["episode_id"])
        self.assertEqual(second["accepted_revision"], 2)

    async def test_new_activation_after_resume_has_new_episode(self):
        first = await self.save()
        await self.save("resume", 1)
        later = await self.save(revision=2)
        self.assertNotEqual(first["quiet"]["episode"]["episode_id"], later["quiet"]["episode"]["episode_id"])

    async def test_duplicate_old_activation_does_not_revive_after_resume(self):
        source = self.source()
        command = QuietCommand(action="activate", expected_revision=0)
        await self.repo.save(self.owner, self.session, source, command)
        await self.save("resume", 1)
        replay = await self.repo.save(self.owner, self.session, source, command)
        self.assertTrue(replay["request_replayed"])
        self.assertEqual(replay["accepted_revision"], 1)
        self.assertEqual(replay["quiet"]["effective_state"], "resumed")
        self.assertEqual(len(self.pool.rows), 2)

    async def test_duplicate_expired_activation_does_not_extend_expiry(self):
        source = self.source()
        expiry = self.pool.now + timedelta(seconds=1)
        command = QuietCommand(action="activate", expected_revision=0, expires_at=expiry)
        await self.repo.save(self.owner, self.session, source, command)
        self.pool.now = expiry
        replay = await self.repo.save(self.owner, self.session, source, command)
        self.assertEqual(replay["quiet"]["effective_state"], "expired")
        self.assertEqual(len(self.pool.rows), 1)

    async def test_same_request_conflict_and_stale_revision_rejected(self):
        source = self.source()
        await self.repo.save(self.owner, self.session, source, QuietCommand(action="activate", expected_revision=0))
        with self.assertRaises(ToolInputError) as error:
            await self.repo.save(self.owner, self.session, source, QuietCommand(action="resume", expected_revision=1))
        self.assertEqual(error.exception.code, "request_conflict")
        with self.assertRaises(ToolInputError) as error:
            await self.save("resume", 0)
        self.assertEqual(error.exception.code, "revision_conflict")
        self.assertEqual(len(self.pool.rows), 1)

    async def test_concurrent_commands_have_one_revision_winner(self):
        outcomes = await asyncio.gather(self.save(), self.save(), return_exceptions=True)
        self.assertEqual(sum(isinstance(value, dict) for value in outcomes), 1)
        self.assertEqual(sum(isinstance(value, ToolInputError) for value in outcomes), 1)
        self.assertEqual(len(self.pool.rows), 1)

    async def test_missing_or_worker_conversation_authority_rejected(self):
        event, attempt = uuid4(), uuid4()
        sources = [ToolInvocationContext(), ToolInvocationContext(
            request_event_id=event, request_revision=1,
            task=TaskToolInvocation(task_id=uuid4(), task_revision=1, attempt_id=attempt))]
        for source in sources:
            with self.assertRaises(ToolInputError) as error:
                await self.repo.save(self.owner, self.session, source, QuietCommand(action="activate", expected_revision=0))
            self.assertEqual(error.exception.code, "request_evidence_required")
        self.assertEqual(self.pool.calls, [])

    async def test_wrong_owner_session_or_withdrawn_request_rejected(self):
        source = self.source()
        for owner, session in (("other", self.session), (self.owner, uuid4())):
            with self.assertRaises(ToolInputError) as error:
                await self.repo.save(owner, session, source, QuietCommand(action="activate", expected_revision=0))
            self.assertEqual(error.exception.code, "request_evidence_unavailable")
        self.pool.sources.clear()
        with self.assertRaises(ToolInputError):
            await self.repo.save(self.owner, self.session, source, QuietCommand(action="activate", expected_revision=0))
        self.assertEqual(self.pool.rows, [])

    async def test_authorization_sql_and_canonical_owner_lock_are_explicit(self):
        await self.save()
        query = next(sql for sql, _ in self.pool.calls if "attention_quiet:authorized_source" in sql)
        for predicate in ("e.owner_id=$1", "e.revision=$3", "e.session_id=$4", "s.status='active'",
                          "e.trust='user_input'", "e.finality in ('final','revised')", "supersedes_event_id=e.id",
                          "p.availability='available'", "for share of p"):
            self.assertIn(predicate, query)
        self.assertIn((OWNER_LOCK, (self.owner,)), self.pool.calls)
        self.assertEqual(len(self.pool.dependencies), 1)

    async def test_policy_withdrawal_or_corruption_fails_closed_not_old_revision(self):
        await self.save()
        await self.save("resume", 1)
        latest = self.pool.payloads[self.pool.rows[-1]["payload"]]
        latest["availability"] = "removed"
        self.assertEqual((await self.repo.get(self.owner)).effective_state, "unavailable")
        latest["availability"] = "available"
        latest["content"] = "invalid json"
        snapshot = await self.repo.get(self.owner)
        self.assertTrue(snapshot.suppresses_unsolicited)
        self.assertEqual(snapshot.revision, 2)
        result = await self.save("resume", 2)
        self.assertEqual(result["quiet"]["effective_state"], "resumed")
        self.assertIsNone(result["quiet"]["episode"]["episode_id"])

    async def test_unavailable_receipt_cannot_be_replayed(self):
        source = self.source()
        request = QuietCommand(action="activate", expected_revision=0)
        await self.repo.save(self.owner, self.session, source, request)
        self.pool.payloads[self.pool.rows[-1]["payload"]]["availability"] = "removed"
        with self.assertRaises(ToolInputError) as error:
            await self.repo.save(self.owner, self.session, source, request)
        self.assertEqual(error.exception.code, "quiet_receipt_unavailable")
        self.assertEqual(len(self.pool.rows), 1)

    async def test_all_unsolicited_channels_block_without_urgent_exception(self):
        await self.save()
        for channel in ("voice", "push", "ui"):
            decision = await quiet_delivery_decision(self.pool, self.owner, kind="unsolicited", channel=channel)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "owner_quiet_quiet")
            self.assertEqual(decision.revision, 1)
        with self.assertRaises(ValueError):
            await quiet_delivery_decision(self.pool, self.owner, kind="emergency", channel="voice")

    async def test_direct_reply_does_not_read_or_mutate_quiet(self):
        await self.save()
        broken_database = SimpleNamespace(fetchrow=AsyncMock(side_effect=RuntimeError("unavailable")))
        decision = await quiet_delivery_decision(broken_database, self.owner, kind="solicited_reply", channel="voice")
        self.assertTrue(decision.allowed)
        broken_database.fetchrow.assert_not_awaited()
        self.assertEqual((await self.repo.get(self.owner)).effective_state, "quiet")
        self.assertEqual(len(self.pool.rows), 1)
        for channel in ("push", "ui"):
            with self.assertRaises(ValueError):
                await quiet_delivery_decision(self.pool, self.owner, kind="solicited_reply", channel=channel)

    async def test_duty_policy_is_not_promoted_and_work_tables_are_not_mutated(self):
        payload = uuid4()
        self.pool.payloads[payload] = {"availability": "available", "content": {"paused_until": "2099-01-01T00:00:00Z"}}
        self.pool.rows.append(dict(owner=self.owner, capability="responsibility_notifications", revision=7,
                                  event=uuid4(), request_revision=1, payload=payload))
        self.assertFalse((await self.repo.get(self.owner)).suppresses_unsolicited)
        await self.save()
        await self.save("resume", 1)
        self.assertEqual(self.pool.rows[0]["revision"], 7)
        self.assertEqual(self.pool.payloads[payload]["content"]["paused_until"], "2099-01-01T00:00:00Z")
        for sql, _ in self.pool.calls:
            self.assertNotIn("refresh_jarvis_responsibility_quiet", sql)
            self.assertNotIn("update jarvis_background_tasks", sql)
            self.assertNotIn("jarvis_behavior_profiles", sql)
        self.assertEqual([r["capability"] for r in self.pool.rows[1:]], [CAPABILITY, CAPABILITY])

    async def test_tools_require_trusted_context_and_commit_before_success(self):
        tools = {tool.name: tool for tool in create_quiet_tools(repository=self.repo, owner_id=self.owner, session_id=self.session)}
        setter = tools["set_attention_quiet"]
        args = {"action": "activate", "expected_revision": 0}
        rejected = await setter.handler(args)
        self.assertEqual(rejected["error"]["code"], "request_evidence_required")
        source = self.source()
        with bind_tool_invocation(provider_call_id="call", request_event_id=source.request_event_id, request_revision=1):
            result = await setter.handler(args)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(self.pool.rows), 1)
        self.assertTrue(setter.mutates_state)
        self.assertNotIn("owner_id", setter.input_schema["properties"])
        self.assertNotIn("request_event_id", setter.input_schema["properties"])
        get_result = await tools["get_attention_quiet"].handler({})
        self.assertEqual(get_result["quiet"]["effective_state"], "quiet")

    async def test_tools_fail_honestly_on_bad_arguments_or_database_error(self):
        repo = SimpleNamespace(save=AsyncMock(side_effect=RuntimeError("private connection data")),
                               get=AsyncMock(side_effect=RuntimeError("private connection data")))
        getter, setter = create_quiet_tools(repository=repo, owner_id=self.owner, session_id=self.session)
        self.assertEqual((await getter.handler({"extra": True}))["error"]["code"], "invalid_tool_input")
        self.assertEqual((await setter.handler({"action": "activate", "expected_revision": True}))["error"]["code"], "invalid_tool_input")
        for result in (await getter.handler({}), await setter.handler({"action": "activate", "expected_revision": 0})):
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["code"], "attention_quiet_unavailable")
            self.assertNotIn("private connection data", str(result))

    async def test_existing_preference_module_registers_structured_tools(self):
        module = PreferenceToolModule(PreferenceRepository(self.pool))
        names = {tool.name for tool in module.build_tools(ToolBuildContext(None, self.owner, self.session))}
        self.assertEqual(names, {"set_preference", "get_preferences", "get_attention_quiet", "set_attention_quiet"})

    def test_schema_rejects_ambiguous_expiry_scope_and_authority_arguments(self):
        base = {"action": "activate", "expected_revision": 0}
        invalid = [
            {"expires_at": "2026-11-01T10:00:00"}, {"expires_at": 12345},
            {"timezone": "not/a/timezone"}, {"owner_id": "other"},
            {"request_event_id": str(uuid4())}, {"expected_revision": True},
            {"action": "resume", "timezone": "UTC"},
            {"expires_at": "2026-07-01T12:00:00+00:00", "timezone": "Europe/Dublin"},
        ]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                QuietCommand.model_validate(base | fields)
        command = QuietCommand.model_validate(base | {
            "expires_at": "2026-07-01T12:00:00+01:00", "timezone": "Europe/Dublin"})
        self.assertEqual(command.expires_at.utcoffset(), timedelta(hours=1))


class QuietDispatchTests(QuietTests):
    """Use real attention/push hook methods with a narrow offline transaction seam."""

    def setUp(self):
        super().setUp()

        @asynccontextmanager
        async def transaction(owner):
            async with self.pool.transaction():
                await self.pool.execute(OWNER_LOCK, owner)
                yield self.pool

        self.tasks = SimpleNamespace(pool=self.pool, transaction=transaction)
        self.attention = AttentionRepository(self.tasks)
        self.attention._recover = AsyncMock()
        self.attention._entry = AsyncMock(side_effect=AssertionError("Should not claim/read source while quiet"))
        self.push = PushRepository(self.attention, SimpleNamespace())
        self.push._recover = AsyncMock()

    async def test_voice_claim_suppressed_after_maintenance_without_claim_mutation(self):
        await self.save()
        result = await self.attention.claim_next(self.owner, self.session, "worker")
        self.assertIsNone(result)
        self.attention._recover.assert_awaited_once_with(self.pool, self.owner)
        self.attention._entry.assert_not_awaited()
        self.assertFalse(any("insert into jarvis_attention_attempts" in sql for sql, _ in self.pool.calls))

    async def test_pending_voice_write_is_fenced_after_activation(self):
        claim = SimpleNamespace(owner_id=self.owner)
        await self.save()
        with self.assertRaises(AttentionError) as error:
            async with self.attention.guard(claim):
                self.fail("A quiet write must not reach the transport")
        self.assertEqual(error.exception.code, "owner_quiet_quiet")
        self.attention._entry.assert_not_awaited()

    async def test_push_claim_preserves_duty_maintenance_and_stops_before_claim(self):
        await self.save()
        self.assertIsNone(await self.push.claim_next(self.owner, "worker"))
        self.attention._recover.assert_awaited_once()
        self.push._recover.assert_awaited_once()
        self.assertTrue(any("refresh_jarvis_responsibility_quiet" in sql for sql, _ in self.pool.calls))
        self.attention._entry.assert_not_awaited()

    async def test_push_preparation_rechecks_quiet_without_submitting(self):
        await self.save()
        self.assertIsNone(await self.push.prepare_submission(SimpleNamespace(owner_id=self.owner)))
        self.attention._entry.assert_not_awaited()
        self.assertFalse(any("phase='submitted'" in sql for sql, _ in self.pool.calls))

    async def test_inactive_owner_still_reaches_normal_voice_admission(self):
        self.assertIsNone(await self.attention.claim_next(self.owner, self.session, "worker"))
        self.assertTrue(any("select id from jarvis_attention_intents" in sql for sql, _ in self.pool.calls))


if __name__ == "__main__":
    unittest.main()
