"""Conversation tools for canonical quiet state, never profile-string enforcement."""
from pydantic import ValidationError

from app.attention.quiet import QuietCommand
from app.tools.inputs import ToolInputError, failed, input_error_payload, success
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition


def create_quiet_tools(*, repository, owner_id, session_id, enforce_owner_quiet=None):
    async def get_attention_quiet(arguments):
        if arguments:
            return failed("invalid_tool_input", "get_attention_quiet takes no arguments.")
        try:
            return success({"quiet": (await repository.get(owner_id)).payload()})
        except Exception:
            return failed("attention_quiet_unavailable", "Canonical quiet state could not be read; do not assume interruptions are enabled.")

    async def set_attention_quiet(arguments):
        try:
            request = QuietCommand.model_validate(arguments)
            result = await repository.save(owner_id, session_id, current_tool_invocation(), request)
        except ValidationError:
            return failed("invalid_tool_input", "Use an action, current integer revision, and optional exact offset-aware expiry/IANA timezone.")
        except ToolInputError as error:
            return input_error_payload(error)
        except Exception:
            return failed("attention_quiet_unavailable", "Quiet-state commit could not be confirmed; read current state before retrying or claiming success.")

        # save() has exited its transaction. A delivery failure cannot undo or
        # misreport the canonical commit. Replayed requests may safely recheck
        # active output, but never manufacture a new episode or resume replay.
        propagation = {"state": "unconfirmed", "stop_requests": None,
                       "reason": "attention_delivery_unavailable",
                       "playback_stop_confirmed": False}
        if enforce_owner_quiet is not None:
            try:
                stopped = await enforce_owner_quiet(owner_id)
                propagation.update(state="checked", stop_requests=stopped, reason=None)
            except Exception:
                propagation["reason"] = "attention_delivery_propagation_unconfirmed"
        return success({**result, "delivery_propagation": propagation})

    return [
        ToolDefinition(
            name="get_attention_quiet",
            description="Read canonical owner-wide do-not-disturb episode, effective expiry and revision. Reading never resumes interruptions.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            handler=get_attention_quiet,
        ),
        ToolDefinition(
            name="set_attention_quiet", mutates_state=True,
            description=(
                "Honor the current user's explicit focus/do-not-disturb or resume request. Reuse the latest authoritative quiet revision "
                "already returned for this owner as expected_revision; call get_attention_quiet only when no such revision is known. "
                "A known revision permits a direct resume call, not a read-first round trip. The repository still checks it atomically. "
                "On revision_conflict, do not automatically read and replay the old command against a newer episode; "
                "reconcile the current state and the user's still-current intent before any new command. "
                "Use activate with no expiry for 'please don't disturb' until explicit resume; set an exact offset-aware expires_at only if "
                "the user requested an expiry, preserving their IANA timezone and matching offset when supplied. "
                "Use resume only on an explicit request, never because the user asks a question, changes goals or reconnects. "
                "Owner quiet applies to unsolicited voice/push/UI attention; direct replies and authorized background work continue. "
                "This does not change duty-specific quiet hours, cancel tasks, stop perception/memory, replace behavior instructions or replay results. "
                "No emergency exceptions. A captured current user conversation request is required; no worker or scene-text authority. "
                "Success confirms canonical state. Check delivery_propagation separately; a stop request does not prove playback stopped. "
                "Briefly confirm the accepted user-facing state; do not routinely recite revisions, provenance, checked propagation "
                "or missing playback receipts. Explain actual failed or unconfirmed effects when relevant to the requested outcome, "
                "but a known lack of playback evidence alone is not an operational failure. "
                "Never imply pending notifications were delivered or heard. "
                "A retried provider call may return its original receipt; distinguish accepted_revision from the returned quiet.revision. "
                "Read current quiet state only when a fresh snapshot is actually needed. "
                "Not every legacy/native delivery path is integrated, and buffered/played output cannot be assumed recalled."
            ),
            input_schema=QuietCommand.model_json_schema(), handler=set_attention_quiet,
        ),
    ]
