"""General duty tools. Interpretation belongs to the model, not item-specific rules."""
from dataclasses import dataclass
from typing import ClassVar
from uuid import UUID

from pydantic import ValidationError

from app.reminders.duties import ChangeDuty, ResponsibilityRepository, SaveDuty
from app.reminders.duty_preferences import ResponsibilityPreferences, SetNotificationPreferences
from app.tools.inputs import ToolInputError, failed, input_error_payload, success
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class ResponsibilityToolModule:
    repository: ResponsibilityRepository
    name: ClassVar[str] = "responsibilities"

    def build_tools(self, context):
        if context.session_id is None:
            return []

        def mutation(model, method):
            async def handler(arguments):
                try:
                    request = model.model_validate(arguments)
                    return success(await method(context.owner_id, context.session_id,
                                                current_tool_invocation(), request))
                except ValidationError as error:
                    return failed("invalid_tool_input", "; ".join(
                        ".".join(map(str, item["loc"])) + ": " + item["msg"]
                        for item in error.errors(include_input=False, include_url=False)))
                except ToolInputError as error:
                    return input_error_payload(error)
            return handler

        async def list_duties(arguments):
            try:
                include = arguments.get("include_inactive", False)
                if type(include) is not bool:
                    raise ValueError("include_inactive must be boolean.")
                identity = UUID(arguments["responsibility_id"]) if arguments.get("responsibility_id") else None
                return success(await self.repository.list(context.owner_id, include_inactive=include,
                                                          responsibility_id=identity))
            except (ValueError, TypeError):
                return failed("invalid_tool_input", "Use a valid responsibility id and boolean include_inactive.")

        return [
            ToolDefinition(
                name="set_responsibility_preferences", mutates_state=True,
                description="Set duty-notification quiet hours, temporary pause, and whether inferred obligations "
                    "need confirmation. List responsibilities first for the current preference revision, then send "
                    "the full replacement policy. Quiet times use the specified IANA timezone; null times disable "
                    "quiet hours and null paused_until removes a temporary pause. Preferences apply to pending "
                    "duty speech and phone fallback, not ordinary replies or unrelated expert results. They cannot "
                    "undo speech or a push already submitted. Allowing inference still requires a captured user "
                    "request; it does not create an autonomous routine-discovery system.",
                input_schema=SetNotificationPreferences.model_json_schema(),
                handler=mutation(SetNotificationPreferences, ResponsibilityPreferences(self.repository).save)),
            ToolDefinition(
                name="save_responsibility", mutates_state=True,
                description="Teach or replace a reusable duty from the current user request. Supports a one-off "
                    "time or situational reminder, and local-time repetition every N days (7 for weekly). "
                    "first_at is the first time or situational activation time; include its UTC offset and an IANA timezone. "
                    "condition is a general visual/session cue, not a claim it occurred. "
                    "Confirm inferred recurring obligations unless the current stored policy explicitly allows inference; use inferred_under_policy for that case. "
                    "For an update, list duties and supply its exact id/revision plus the full replacement spec. "
                    "Use different request_slot values for independent duties within one request and preserve "
                    "the same slot/arguments on retry. Do not recreate a duty after interrupted speech.",
                input_schema=SaveDuty.model_json_schema(),
                handler=mutation(SaveDuty, self.repository.save)),
            ToolDefinition(
                name="change_responsibility", mutates_state=True,
                description="Cancel a duty, snooze its selected occurrence, or record an explicit user completion "
                    "or noncompletion report. List duties first for exact rule revision and occurrence id. "
                    "Use report_done/report_not_done only for what the user actually reported, never from "
                    "object visibility, silence, disconnection, or notification playback. Snooze postpones this "
                    "duty until the supplied timestamp; missed recurring periods collapse to one current occurrence. "
                    "Reporting noncompletion does not itself schedule another alert; snooze explicitly if requested. "
                    "Cancel stops future occurrences and pending notifications. Physical actions are never verified "
                    "by these tools. Preserve request_slot and arguments when retrying.",
                input_schema=ChangeDuty.model_json_schema(),
                handler=mutation(ChangeDuty, self.repository.change)),
            ToolDefinition(
                name="list_responsibilities",
                description="Inspect duties, due/pending occurrences, snooze times and source-linked user reports "
                    "across sessions. Include inactive duties for past-action questions. Distinguish user reports "
                    "from observations; if evidence is missing, say you do not know. Notification completion does "
                    "not prove the user acted, and a visible object does not prove its use.",
                input_schema={"type":"object", "additionalProperties":False, "properties":{
                    "include_inactive":{"type":"boolean"},
                    "responsibility_id":{"type":["string","null"]}}},
                handler=list_duties),
        ]
