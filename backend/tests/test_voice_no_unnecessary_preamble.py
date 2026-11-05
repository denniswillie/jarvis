from __future__ import annotations

import json

import pytest

from app.assistant.base_prompt import build_base_prompt
from app.realtime.instructions import build_realtime_instructions


@pytest.mark.parametrize("tasks_enabled", [False, True])
def test_shared_policy_does_not_treat_commentary_as_inaudible(tasks_enabled: bool) -> None:
    prompt = build_base_prompt(tasks_enabled=tasks_enabled)

    assert "When the answer is already available, start with the useful answer" in prompt
    assert "evidence is missing or unclear, state that limitation directly." in prompt
    assert "Commentary may be audible on the active voice route" in prompt
    assert "suppresses commentary-phase speech by default" not in prompt
    assert "Do not rely on commentary to convey essential content." in prompt
    assert "one brief natural intention acknowledgment before tool acceptance" in prompt
    assert "not evidence that work has started" in prompt
    assert "until a tool receipt confirms that state" in prompt
    assert "client, not confirmed heard." in prompt


@pytest.mark.parametrize("tasks_enabled", [False, True])
def test_response_instructions_preserve_policy_and_runtime_evidence(tasks_enabled: bool) -> None:
    live_view = {"status": "unavailable", "reason": "no_current_pixels"}
    instructions = build_realtime_instructions(
        context=None, live_view=live_view, tasks_enabled=tasks_enabled,
    )
    policy, runtime = instructions.split("\n\nRuntime context:\n```json\n", 1)

    assert policy == build_base_prompt(tasks_enabled=tasks_enabled)
    assert json.loads(runtime.removesuffix("\n```")) == {"live_view": live_view}


def test_confirmed_task_acknowledgments_and_ownership_are_preserved() -> None:
    prompt = build_base_prompt(tasks_enabled=True)

    assert "After the tools confirm durable acceptance, give one brief joint acknowledgment when a wait is warranted" in prompt
    assert "and you have not already acknowledged your intention" in prompt
    assert "Do not repeat an acknowledgment for a known task after an interruption, retry, or reset." in prompt
    assert "Interrupting playback, a provider reset, or disconnecting does not cancel accepted work." in prompt


def test_unavailable_workers_do_not_become_a_background_promise() -> None:
    prompt = build_base_prompt(tasks_enabled=False)

    assert "Background expert delegation is not available in this session." in prompt
    assert "or can continue background reasoning after this reply." in prompt


@pytest.mark.parametrize("tasks_enabled", [False, True])
def test_intention_does_not_become_filler_or_unconfirmed_effects(tasks_enabled: bool) -> None:
    prompt = build_base_prompt(tasks_enabled=tasks_enabled)
    assert "an acknowledgment never replaces the requested work" in prompt
    assert "queued, started, saved, cancelled, or completed until a tool receipt confirms that state" in prompt
    assert "immediate answer, greeting, thanks, or quiet dismissal" in prompt
    assert "Never use it to override do-not-disturb instructions" in prompt
    assert "not potentially suppressed commentary" in prompt
    assert "Do not repeat it just because a tool returned" in prompt
    assert "only after tools confirm accepted background work" not in prompt
    assert "call the necessary tools before speaking" not in prompt
    assert "do not give a spoken startup preamble" not in prompt


def test_multiple_tasks_and_failed_acceptance_preserve_truthful_follow_through() -> None:
    prompt = build_base_prompt(tasks_enabled=True)
    assert "submit all of them and collect their admission results before claiming which tasks were accepted" in prompt
    assert "Report partial acceptance accurately" in prompt
    assert "Report a failure or partial acceptance even if you already acknowledged the request" in prompt
    assert "do not leave an unsupported promise standing" in prompt
