from __future__ import annotations

from app.assistant.context import DEFAULT_BEHAVIOR_INSTRUCTIONS, merge_behavior_instructions


def test_behavior_instructions_merge_persistent_and_session_scopes() -> None:
    behavior = merge_behavior_instructions(
        persistent="Always tell me when I leave my keys behind.",
        session_override="During this walk, keep responses especially short.",
    )

    assert "Persistent owner instructions" in behavior.combined
    assert "leave my keys behind" in behavior.combined
    assert "Current session override" in behavior.combined
    assert "especially short" in behavior.combined


def test_behavior_instructions_use_default_when_both_scopes_are_empty() -> None:
    behavior = merge_behavior_instructions(persistent="", session_override="")

    assert behavior.combined == DEFAULT_BEHAVIOR_INSTRUCTIONS
