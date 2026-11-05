from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


class ToolCallLimitError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ToolCallBudget:
    max_total_calls: int
    max_repeated_calls: int
    total_calls: int = 0
    _calls_by_fingerprint: Counter[str] = field(default_factory=Counter)

    def record(self, *, tool_name: str, arguments: dict[str, Any]) -> None:
        if self.total_calls >= self.max_total_calls:
            raise ToolCallLimitError(
                "tool_call_limit_reached",
                "The tool-call limit for this assistant response has been reached. Respond using the information already available.",
            )
        fingerprint = tool_call_fingerprint(tool_name=tool_name, arguments=arguments)
        if self._calls_by_fingerprint[fingerprint] >= self.max_repeated_calls:
            raise ToolCallLimitError(
                "repeated_tool_call_limit_reached",
                "This identical tool call has already been attempted too many times. Change the arguments or respond using the information already available.",
            )
        self.total_calls += 1
        self._calls_by_fingerprint[fingerprint] += 1


def tool_call_fingerprint(*, tool_name: str, arguments: dict[str, Any]) -> str:
    canonical_arguments = json.dumps(
        arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return f"{tool_name}:{canonical_arguments}"
