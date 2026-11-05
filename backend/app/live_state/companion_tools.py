"""Small owner-bound operational tools shared by Voice and retained Hermes."""
from dataclasses import asdict
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from app.tools.registry import ToolDefinition
from app.app_controls import APP_CONTROL_DESCRIPTION, AppCommand


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReminderArguments(Arguments):
    action: Literal["create", "complete", "delete"]
    reminder_id: StrictStr | None = Field(description=
        "Create: null. Complete/delete: exact existing ID. If unknown, first select "
        "read_live_state with section=system and purpose=act; never guess an ID.")
    title: StrictStr | None = Field(max_length=200,
        description="Create: reminder title. Complete/delete: null, not a search query.")
    condition: StrictStr | None = Field(max_length=1000,
        description="Create: situational trigger. Complete/delete: null, not a search query.")
    repeat_until_confirmed: StrictBool = Field(description=
        "Create: whether to repeat until confirmed. Complete/delete: false.")


class ForgetArguments(Arguments):
    document_ids: list[StrictStr] = Field(min_length=1, max_length=20)
    reminder_ids: list[StrictStr] = Field(max_length=20)


SCHEMAS = {
    "control_app": (AppCommand, APP_CONTROL_DESCRIPTION),
    "manage_reminder": (ReminderArguments,
        "Create a situational reminder, complete this occurrence, or delete a reminder. "
        "Timed, recurring and periodic checks belong to Hermes sandbox scripts, not this tool."),
    "forget_memory": (ForgetArguments,
        "On an explicit user forgetting request, delete exact Hindsight document IDs found by recall "
        "and invalidate matching current state. Supply any matching active reminder IDs from read_live_state. "
        "Never delete recordings, sandbox files or Hermes conversation history. Clarify ambiguous requests."),
}


class CompanionTools:
    name = "companion_operations"

    def __init__(self, state, *, memory_intake=None, app_controls=None):
        self.state, self.memory_intake = state, memory_intake
        self.app_controls = app_controls
        self.owner_id = state.owner_id

    def names(self):
        return frozenset(name for name in SCHEMAS
                         if (name != "forget_memory" or self.memory_intake is not None)
                         and (name != "control_app" or self.app_controls is not None))

    def voice_tools(self):
        # Forgetting can require recall/clarification first: let retained Hermes
        # resolve the exact documents, rather than making a one-shot router guess.
        result = []
        for name, (model, description) in SCHEMAS.items():
            if name == "forget_memory" or name not in self.names():
                continue
            schema = model.model_json_schema()
            schema["required"] = list(schema["properties"])
            for prop in schema["properties"].values():
                prop.pop("default", None)
            result.append({"type": "function", "name": name, "description": description,
                           "strict": True, "parameters": schema})
        return result

    async def invoke(self, name, arguments, *, owner_id, session_id=None, operation_id=None,
                     notify_voice=False):
        if owner_id != self.owner_id or name not in self.names():
            raise ValueError("Operational tool owner or capability mismatch")
        args = SCHEMAS[name][0].model_validate(arguments).model_dump(mode="json")
        if name == "control_app":
            return await self.app_controls.invoke(owner_id, session_id, operation_id=operation_id,
                                                  notify_voice=notify_voice, **args)
        if name == "manage_reminder":
            if args["action"] == "create":
                if args["reminder_id"] is not None or not args["title"] or not args["condition"]:
                    raise ValueError("Creating a reminder needs title/condition and no target ID")
            else:
                if not args["reminder_id"] or args["title"] is not None or args["condition"] is not None:
                    raise ValueError("Complete/delete requires only a known reminder ID")
                UUID(args["reminder_id"])
            result = await self.state.manage_reminder(**args, session_id=session_id)
            return {"reminder": result, "status": "applied" if result else "not_found"}
        ids = tuple(dict.fromkeys(args["document_ids"]))
        if any(not value.strip() or len(value) > 512 for value in ids):
            raise ValueError("Exact nonempty document IDs are required")
        reminder_ids = tuple(UUID(value) for value in args["reminder_ids"])
        # Invalidate local matching reminders first. Remote failure must not
        # allow a known-forgotten intention to announce itself while retrying.
        for reminder_id in reminder_ids:
            await self.state.manage_reminder(action="delete", reminder_id=reminder_id)
        result = await self.memory_intake.forget(ids)
        return {"status": "applied" if result.requested_deletions_applied else "incomplete",
                **asdict(result),
                "note": "Source deletion and local invalidation are separate operations; "
                        "derived Hindsight summary cleanup may still be pending."}

    def build_tools(self, context):
        if context.owner_id != self.owner_id:
            raise ValueError("Operational tool owner mismatch")
        result = []
        for name in sorted(self.names()):
            model, description = SCHEMAS[name]
            async def invoke(arguments, name=name):
                return await self.invoke(name, arguments, owner_id=context.owner_id,
                                         session_id=context.session_id, notify_voice=True)
            result.append(ToolDefinition(name, description, model.model_json_schema(),
                                         invoke, mutates_state=True))
        return result
