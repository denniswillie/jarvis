from __future__ import annotations

import json
from collections.abc import Sequence

from app.memory.reflection.repository import ReflectionEpisode


REFLECTION_INSTRUCTIONS = """You are Jarvis's slow cross-episode reflection model.

Derive grounded patterns only from the supplied durable episodes.

Rules:
- Cite only supplied episode IDs.
- Copy episode identifiers exactly; never abbreviate, combine, or reconstruct them.
- Do not turn one weak observation into a stable preference or procedure.
- Preserve uncertainty and contradictions instead of resolving them silently.
- Treat task status and camera observations as time-scoped episode evidence, not timeless personal traits. Stopping or pausing work is not completing it.
- Respect later state changes without erasing the earlier history. User reports and assistant approvals are not automatically verified outcomes or evidence that a requested action executed.
- Do not merge different entities or tasks merely because they have similar descriptions or nearby timestamps. Without evidence of a shared identity, a later unfinished task does not reverse another task's completion; describe them separately.
- Preference candidates must be stable personal preferences supported by repeated episodes.
- Procedure candidates must describe repeated ordered behavior, not executable code or tool calls.
- Mark a procedure explicit only when an episode marked is_explicit contains a direct user instruction.
- Keep procedure_key short, stable, and conceptual; do not include versions, timestamps, or UUIDs.
- Do not create reminders, speak to the user, or propose proactive interruptions.
- Return only the structured response required by the schema.
"""


def build_reflection_prompt(episodes: Sequence[ReflectionEpisode]) -> str:
    payload = {
        "episodes": [
            {
                "id": str(episode.id),
                "session_id": str(episode.session_id) if episode.session_id else None,
                "title": episode.title,
                "summary": episode.summary,
                "content": episode.content,
                "occurred_at": (
                    episode.occurred_at.isoformat()
                    if episode.occurred_at is not None
                    else None
                ),
                "created_at": episode.created_at.isoformat(),
                "is_explicit": episode.is_explicit,
            }
            for episode in episodes
        ]
    }
    return (
        "Reflect across these ordered durable episodes. Return only the required "
        "structured response.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
