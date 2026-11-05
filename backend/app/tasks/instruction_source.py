"""Read effective task input under the existing source/transaction authority."""
from uuid import UUID
from app.tasks.models import TaskError, TaskVoiceSource, json_object


def _scope_error():
    return TaskError("task_instruction_unavailable", "The effective task instruction is unavailable or changed.", 403)


async def task_instruction_source(connection, task, *, lock=True) -> TaskVoiceSource:
    """Read effective instructions, never substitute a later control event."""
    row = await connection.fetchrow(
        "select content,availability from jarvis_context_payloads "
        "where owner_id=$1 and id=$2" + (" for share" if lock else ""), task.owner_id, task.description_payload_id)
    if row is None or row["availability"] != "available":
        raise _scope_error()
    try:
        raw = json_object(row["content"])["instruction_request"]
        if not isinstance(raw, dict) or set(raw) != {"event_id", "revision", "session_id"}:
            raise ValueError()
        source = TaskVoiceSource(UUID(raw["event_id"]), raw["revision"], UUID(raw["session_id"]))
        if source.session_id != task.session_id:
            raise ValueError()
        return source
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _scope_error() from None

