from __future__ import annotations

import pytest

from app.assistant.tool_limits import ToolCallBudget, ToolCallLimitError, tool_call_fingerprint


def test_tool_fingerprint_canonicalizes_argument_order() -> None:
    assert tool_call_fingerprint(
        tool_name="search",
        arguments={"b": 2, "a": 1},
    ) == tool_call_fingerprint(
        tool_name="search",
        arguments={"a": 1, "b": 2},
    )


def test_tool_budget_rejects_identical_call_after_repeat_limit() -> None:
    budget = ToolCallBudget(max_total_calls=12, max_repeated_calls=2)
    budget.record(tool_name="search", arguments={"query": "keys"})
    budget.record(tool_name="search", arguments={"query": "keys"})

    with pytest.raises(ToolCallLimitError) as error:
        budget.record(tool_name="search", arguments={"query": "keys"})

    assert error.value.code == "repeated_tool_call_limit_reached"


def test_tool_budget_rejects_calls_after_total_limit() -> None:
    budget = ToolCallBudget(max_total_calls=2, max_repeated_calls=2)
    budget.record(tool_name="search", arguments={"query": "one"})
    budget.record(tool_name="search", arguments={"query": "two"})

    with pytest.raises(ToolCallLimitError) as error:
        budget.record(tool_name="search", arguments={"query": "three"})

    assert error.value.code == "tool_call_limit_reached"
