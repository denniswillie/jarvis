"""Durable note acceptance is distinct from a final note publication."""
from copy import deepcopy
import unittest
from uuid import uuid4

from app.live_state.workspace_dispatch import LeanWorkspaceDispatcher
from test_lean_workspace_dispatch import Commands, Journal


class EnglishCommands(Commands):
    def __init__(self):
        super().__init__()
        self.admissions = []

    async def execute(self, ctx, name, arguments, *, command_admission=None):
        self.calls.append((ctx, name, deepcopy(arguments)))
        self.admissions.append(command_admission)
        receipt = {"status": "accepted", "outcome": "accepted", "effect_committed": False,
                   "operation_id": str(ctx.operation_id), "task_ref": "task_" + uuid4().hex,
                   "task_revision": 1, "summary": "The edit is queued, not saved."}
        self.receipts[ctx.operation_id] = receipt
        return receipt


class EnglishNoteDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.commands, self.journal = EnglishCommands(), Journal()
        self.dispatch = LeanWorkspaceDispatcher(owner_id="owner", session_id=uuid4(),
            provider_session_key=lambda: "rtc_note", is_active=lambda: True,
            commands=self.commands, journal=self.journal)
        self.arguments = {"instruction": "Append PRIVATE CONTENT to my note."}

    async def edit(self):
        return await self.dispatch.execute("live_note", self.arguments, provider_call_id="note_call")

    async def test_english_edit_passes_exact_trusted_task_admission(self):
        result = await self.edit()
        admitted = self.commands.admissions[0]
        self.assertEqual(admitted.capability, "tasks.create")
        self.assertEqual(admitted.tool_name, "live_note")
        self.assertEqual(admitted.normalized_arguments, self.arguments)
        self.assertEqual(str(admitted.operation_id), result["operation_id"])
        self.assertEqual(result["outcome"], "accepted")
        self.assertFalse(result["effect_committed"])
        self.assertNotIn("PRIVATE", repr(self.journal.starts + self.journal.finishes))
        self.assertEqual(self.journal.finishes[0]["result"]["task_ref"], result["task_ref"])

    async def test_repeated_call_returns_current_durable_task_receipt(self):
        first = await self.edit()
        self.assertEqual(await self.edit(), first)
        self.assertEqual(len(self.commands.calls), 1)

    async def test_missing_accepted_receipt_is_unknown_not_rejected_or_repeated(self):
        await self.edit()
        self.commands.receipts.clear()
        result = await self.edit()
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertIsNone(result["effect_committed"])
        self.assertEqual(len(self.commands.calls), 1)

    async def test_later_terminal_receipt_can_report_exact_saved_version(self):
        await self.edit()
        operation = self.commands.calls[0][0].operation_id
        final = {"status": "success", "outcome": "completed", "effect_committed": True,
                 "operation_id": str(operation),
                 "artifact": {"kind": "note", "artifact_id": str(uuid4()), "version_id": "saved-v2"}}
        self.commands.receipts[operation] = final
        self.assertEqual(await self.edit(), final)
        self.assertEqual(len(self.commands.calls), 1)
