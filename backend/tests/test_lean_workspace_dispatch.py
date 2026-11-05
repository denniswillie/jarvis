import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.experience.repository import ToolExecutionConflictError
from app.live_state.command_identity import command_operation_id
from app.live_state.workspace_dispatch import LeanWorkspaceDispatcher
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation


class Journal:
    def __init__(self):
        self.rows, self.starts, self.finishes = {}, [], []
        self.fail_finalize = False
        self.after_start = None

    async def start_execution_with_status(self, **kwargs):
        self.starts.append(deepcopy(kwargs))
        key = kwargs["provider_call_id"]
        old = self.rows.get(key)
        if old is not None:
            if old.arguments != kwargs["arguments"] or old.tool_name != kwargs["tool_name"]:
                raise ToolExecutionConflictError()
            return old, False
        row = SimpleNamespace(arguments=kwargs["arguments"], tool_name=kwargs["tool_name"], status="started", result=None)
        self.rows[key] = row
        if self.after_start:
            self.after_start()
        return row, True

    async def finalize_execution(self, **kwargs):
        self.finishes.append(deepcopy(kwargs))
        if self.fail_finalize:
            raise OSError("fixture finalization loss")
        row = self.rows[kwargs["provider_call_id"]]
        row.status, row.result = kwargs["status"], kwargs["result"]
        return row


class Commands:
    def __init__(self):
        self.calls, self.receipts = [], {}
        self.action = None

    def schemas(self):
        return {name: {"type": "object"} for name in ("live_read", "live_present")}

    def normalize(self, name, arguments):
        if "owner_id" in arguments:
            raise ValueError("untrusted authority")
        return deepcopy(arguments)

    async def receipt(self, ctx, name, operation_id):
        return self.receipts.get(operation_id)

    async def execute(self, ctx, name, arguments):
        self.calls.append((ctx, name, deepcopy(arguments)))
        if self.action:
            return await self.action(ctx, name, arguments)
        if name == "live_read":
            return {"status": "success", "outcome": "available", "effect_committed": False, "summary": "Current saved file."}
        result = {"status": "success", "outcome": "completed", "effect_committed": True,
            "operation_id": str(ctx.operation_id), "summary": "Saved PRIVATE TITLE",
            "artifact": {"artifact_id": str(uuid4()), "version_id": "source_exact", "kind": "file", "title": "PRIVATE TITLE"}}
        self.receipts[ctx.operation_id] = result
        return result


class LeanWorkspaceDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session, self.provider = "owner", uuid4(), "rtc_exact"
        self.active = True
        self.commands, self.journal = Commands(), Journal()
        self.dispatch = LeanWorkspaceDispatcher(owner_id=self.owner, session_id=self.session,
            provider_session_key=lambda: self.provider, is_active=lambda: self.active,
            commands=self.commands, journal=self.journal, read_timeout=.02, command_timeout=.03)
        self.args = {"action": "download", "title": "PRIVATE TITLE", "content": {"secret": "PRIVATE BODY"}}

    async def run_command(self, **kwargs):
        return await self.dispatch.execute("live_present", self.args, provider_call_id=" call_1 ", **kwargs)

    async def test_exact_identity_and_ref_only_audit(self):
        result = await self.run_command()
        operation = command_operation_id(self.owner, self.session, self.provider, " call_1 ")
        self.assertEqual(result["operation_id"], str(operation))
        self.assertEqual(self.commands.calls[0][0].operation_id, operation)
        self.assertEqual(self.journal.starts[0]["provider_call_id"], "lean:" + str(operation))
        audit = self.journal.starts[0]["arguments"]
        self.assertEqual((audit["source_namespace"], audit["kind"], audit["trust"]),
                         ("voice_command", "command_request", "derived"))
        self.assertNotIn("PRIVATE", repr(self.journal.starts))
        self.assertNotIn("PRIVATE", repr(self.journal.finishes))

    async def test_read_never_journals_or_requires_transcript(self):
        result = await self.dispatch.execute("live_read", {"domain": "artifact", "query": "test"}, provider_call_id="read")
        self.assertEqual(result["outcome"], "available")
        self.assertIsNone(self.commands.calls[0][0].operation_id)
        self.assertFalse(self.journal.starts)

    async def test_same_command_uses_current_domain_receipt(self):
        first = await self.run_command()
        operation = self.commands.calls[0][0].operation_id
        self.commands.receipts[operation] = dict(first, summary="Saved result is now unavailable.")
        second = await self.run_command()
        self.assertEqual(second["summary"], "Saved result is now unavailable.")
        self.assertEqual(len(self.commands.calls), 1)

    async def test_reused_call_with_different_arguments_conflicts(self):
        await self.run_command()
        self.args["title"] = "another"
        result = await self.run_command()
        self.assertEqual(result["outcome"], "conflict")
        self.assertFalse(result["effect_committed"])
        self.assertEqual(len(self.commands.calls), 1)

    async def test_reused_call_with_different_tool_conflicts(self):
        await self.run_command()
        result = await self.dispatch.execute("live_present", {"action": "open"}, provider_call_id=" call_1 ")
        self.assertEqual(result["outcome"], "conflict")
        self.assertEqual(len(self.commands.calls), 1)

    async def test_response_loss_reconciles_domain_not_executor(self):
        self.journal.fail_finalize = True
        first = await self.run_command()
        self.assertTrue(first["effect_committed"])
        self.assertEqual(next(iter(self.journal.rows.values())).status, "started")
        self.assertEqual(await self.run_command(), first)
        self.assertEqual(len(self.commands.calls), 1)

    async def test_missing_receipt_never_reexecutes_started_command(self):
        async def fail(ctx, name, args):
            raise OSError("unknown remote outcome")
        self.commands.action = fail
        result = await self.run_command()
        self.assertIsNone(result["effect_committed"])
        self.assertEqual((await self.run_command())["outcome"], "outcome_unknown")
        self.assertEqual(len(self.commands.calls), 1)

    async def test_known_rejection_replay_does_not_apply_again(self):
        async def rejected(ctx, name, args):
            raise JarvisHTTPError(409, "file_version_conflict", "Current version changed.")
        self.commands.action = rejected
        self.assertFalse((await self.run_command())["effect_committed"])
        self.assertFalse((await self.run_command())["effect_committed"])
        self.assertEqual(len(self.commands.calls), 1)

    async def test_error_after_commit_returns_domain_effect(self):
        async def committed(ctx, name, args):
            receipt = {"status": "success", "outcome": "accepted", "effect_committed": True,
                       "operation_id": str(ctx.operation_id), "summary": "Phone offer saved."}
            self.commands.receipts[ctx.operation_id] = receipt
            raise JarvisHTTPError(404, "file_source_unavailable", "Source withdrawn after reservation.")
        self.commands.action = committed
        result = await self.run_command()
        self.assertTrue(result["effect_committed"])
        self.assertEqual(result["outcome"], "accepted")

    async def test_timeout_after_submission_is_unknown(self):
        async def slow(ctx, name, args):
            await asyncio.sleep(1)
        self.commands.action = slow
        result = await self.run_command()
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertIsNone(result["effect_committed"])
        self.assertFalse(self.journal.finishes)

    async def test_read_timeout_is_no_effect(self):
        async def slow(ctx, name, args):
            await asyncio.sleep(1)
        self.commands.action = slow
        result = await self.dispatch.execute("live_read", {"domain": "artifact"}, provider_call_id="read")
        self.assertFalse(result["effect_committed"])
        self.assertFalse(self.journal.starts)

    async def test_closed_session_not_admitted(self):
        self.active = False
        self.assertFalse((await self.run_command())["effect_committed"])
        self.assertFalse(self.journal.starts)

    async def test_missing_provider_identity_not_guessed(self):
        self.provider = None
        self.assertFalse((await self.run_command())["effect_committed"])
        self.assertFalse(self.journal.starts)

    async def test_interruption_before_domain_admission_changes_no_domain(self):
        self.assertFalse((await self.run_command(admit_if=lambda: False))["effect_committed"])
        self.assertFalse(self.commands.calls)
        self.assertEqual(len(self.journal.starts), 1)

    async def test_interrupted_replay_reports_existing_commit(self):
        original = await self.run_command()
        self.assertEqual(await self.run_command(admit_if=lambda: False), original)
        self.assertEqual(len(self.commands.calls), 1)

    async def test_journal_failure_on_replay_does_not_claim_no_effect(self):
        await self.run_command()
        async def failed(**kwargs):
            raise OSError("journal unavailable")
        self.journal.start_execution_with_status = failed
        result = await self.run_command()
        self.assertIsNone(result["effect_committed"])
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertEqual(len(self.commands.calls), 1)

    async def test_receipt_timeout_on_replay_does_not_claim_no_effect(self):
        await self.run_command()
        async def slow(*args):
            await asyncio.sleep(1)
        self.commands.receipt = slow
        result = await self.run_command()
        self.assertIsNone(result["effect_committed"])
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertEqual(len(self.commands.calls), 1)

    async def test_interruption_during_journal_prevents_new_effect(self):
        allowed = [True]
        self.journal.after_start = lambda: allowed.__setitem__(0, False)
        self.assertFalse((await self.run_command(admit_if=lambda: allowed[0]))["effect_committed"])
        self.assertFalse(self.commands.calls)
        self.assertFalse(next(iter(self.journal.rows.values())).result["effect_committed"])

    async def test_worker_cannot_borrow_foreground_scope(self):
        with bind_tool_invocation(provider_call_id="worker", task=TaskToolInvocation(uuid4(), 1, uuid4())):
            result = await self.run_command()
        self.assertFalse(result["effect_committed"])
        self.assertFalse(self.journal.starts)

    async def test_json_cannot_supply_owner(self):
        self.args["owner_id"] = "other"
        self.assertEqual((await self.run_command())["outcome"], "rejected")
        self.assertFalse(self.journal.starts)

    async def test_tool_handler_without_native_boundary_fails_closed(self):
        definition = next(item for item in self.dispatch.definitions() if item.name == "live_present")
        self.assertEqual((await definition.handler(self.args))["error"]["code"], "command_admission_required")
        self.assertFalse(self.journal.starts)

    def test_without_journal_only_reads_are_advertised(self):
        dispatch = LeanWorkspaceDispatcher(owner_id=self.owner, session_id=self.session,
            provider_session_key=lambda: self.provider, is_active=lambda: True, commands=self.commands)
        self.assertEqual([item.name for item in dispatch.definitions()], ["live_read"])

    async def test_new_speech_retires_navigation_hint_not_durable_offer(self):
        intents, current = [], [True]
        self.dispatch.on_presentation_intent = lambda *args: intents.append(args)
        async def opened(ctx, name, args):
            current[0] = False
            return {"status": "success", "outcome": "accepted", "effect_committed": True,
                "operation_id": str(ctx.operation_id), "summary": "Phone offer saved.",
                "presentation": {"request_id": str(ctx.operation_id), "state": "pending"}}
        self.commands.action = opened
        result = await self.dispatch.execute("live_present", {"action": "open"}, provider_call_id="open",
                                             admit_if=lambda: current[0])
        self.assertTrue(result["effect_committed"])
        self.assertFalse(intents)

    async def test_file_open_does_not_gain_note_auto_navigation(self):
        intents = []
        self.dispatch.on_presentation_intent = lambda *args: intents.append(args)
        async def opened(ctx, name, args):
            return {"status": "success", "outcome": "accepted", "effect_committed": True,
                "operation_id": str(ctx.operation_id), "summary": "Phone offer saved.",
                "presentation": {"request_id": str(ctx.operation_id), "state": "pending"}}
        self.commands.action = opened
        await self.dispatch.execute("live_present", {"action": "open", "artifact_kind": "file"}, provider_call_id="open-file")
        self.assertFalse(intents)

    async def test_generic_error_details_are_not_validation_feedback(self):
        async def rejected(ctx, name, arguments):
            raise JarvisHTTPError(422, "invalid_document", "Invalid document.", details={
                "validation_issues": [{"path": ["PRIVATE KEY"], "code": "PRIVATE VALUE"}]})
        self.commands.action = rejected
        for name, args in (("live_present", self.args), ("live_read", {"domain": "artifact"})):
            with self.subTest(tool=name):
                result = await self.dispatch.execute(name, args, provider_call_id=name)
                self.assertNotIn("validation_issues", result["error"])
                self.assertNotIn("PRIVATE", repr(result))

