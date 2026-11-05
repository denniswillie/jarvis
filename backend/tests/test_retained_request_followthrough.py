"""Pending-request policy at the actual response writer, not model compliance."""
import asyncio

import pytest

from test_realtime_unwritten_request_sources import composed, cancelled_waiter_then_current


@pytest.mark.parametrize("current_text", [
    "What is WorkbenchPlan's status? Separately, what is two plus three? Do not restart it.",
    "Correction: do not cancel WorkbenchPlan. Keep working. What is two plus three?",
])
def test_pending_request_actions_and_withdrawal_rules_reach_the_response(current_text):
    async def run():
        async with composed() as fx:
            write, _ = await cancelled_waiter_then_current(fx, current_text=current_text)
            instructions = write["response"]["instructions"]
            assert "does not erase unfinished earlier requests" in instructions
            assert "Perform the requested tool actions for each still-applicable retained request" in instructions
            assert "A later status check or unrelated question does not withdraw it" in instructions
            assert "perform the authorized change first, then check the resulting state" in instructions
            assert "explain what remains unresolved or ask for the missing clarification" in instructions
            assert "Newer explicit corrections or withdrawals take precedence" in instructions
            assert "do not execute a superseded request" in instructions
            assert "Unavailable retained requests cannot authorize tools" in instructions
            assert "never redo an operation merely to acknowledge it" in instructions
            assert "Select jarvis_request_source separately for EACH tool call" in instructions
            assert current_text in instructions
            # The policy does not deterministically execute a cancellation,
            # fabricate a tool receipt or assume a semantic interpretation.
            assert not fx.tasks.calls
    asyncio.run(run())
