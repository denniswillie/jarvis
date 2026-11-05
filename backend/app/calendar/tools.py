from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from app.calendar.google_oauth import GoogleCalendarError, GoogleCredentialService
from app.tools.inputs import (
    ToolInputError,
    failed,
    input_error_payload,
    optional_positive_int,
    optional_string,
    required_string,
    success,
)
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


class CalendarInputError(Exception):
    pass


class GoogleCalendarClient:
    def __init__(self, *, credentials: GoogleCredentialService, default_timezone: str) -> None:
        self._credentials = credentials
        self.default_timezone = default_timezone

    async def execute(
        self,
        *,
        owner_id: str,
        operation: Callable[[Resource], Any],
    ) -> Any:
        credentials = await self._credentials.credentials_for_owner(owner_id=owner_id)

        def run() -> Any:
            service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
            return operation(service).execute()

        try:
            return await asyncio.to_thread(run)
        except HttpError as error:
            status = getattr(error.resp, "status", None)
            if status == 404:
                raise GoogleCalendarError(
                    "calendar_event_not_found",
                    "The Google Calendar event was not found.",
                ) from error
            if status in {401, 403}:
                raise GoogleCalendarError(
                    "google_calendar_reauthorization_required",
                    "Google Calendar must be reconnected from Jarvis Settings.",
                ) from error
            raise GoogleCalendarError(
                "google_calendar_api_failed",
                "Google Calendar could not complete that request.",
            ) from error


@dataclass(frozen=True)
class CalendarToolModule:
    client: GoogleCalendarClient
    name: ClassVar[str] = "calendar"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        return create_calendar_tools(
            client=self.client,
            owner_id=context.owner_id,
        )


def create_calendar_tools(*, client: GoogleCalendarClient, owner_id: str) -> list[ToolDefinition]:
    async def list_calendar_events(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            time_min = _required_datetime(arguments, "time_min")
            time_max = _required_datetime(arguments, "time_max")
            if time_max <= time_min:
                raise CalendarInputError("time_max must be after time_min.")
            calendar_id = optional_string(arguments, "calendar_id", max_length=500) or "primary"
            limit = optional_positive_int(arguments, "limit", default=20, max_value=50)
        except ToolInputError as error:
            return input_error_payload(error)
        except CalendarInputError as error:
            return failed("invalid_tool_input", str(error))

        try:
            result = await client.execute(
                owner_id=owner_id,
                operation=lambda service: service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    timeZone=client.default_timezone,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=limit,
                ),
            )
        except GoogleCalendarError as error:
            return failed(error.code, error.message)
        events = result.get("items", []) if isinstance(result, dict) else []
        return success({"events": [_event_payload(event) for event in events[:limit]]})

    async def create_calendar_event(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            summary = required_string(arguments, "summary", max_length=500)
            description = _text_value(arguments, "description", max_length=10_000, default="")
            location = _text_value(arguments, "location", max_length=1_000, default="")
            calendar_id = optional_string(arguments, "calendar_id", max_length=500) or "primary"
            start, end = _event_range(arguments, client.default_timezone, required=True)
        except ToolInputError as error:
            return input_error_payload(error)
        except CalendarInputError as error:
            return failed("invalid_tool_input", str(error))

        body = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": start,
            "end": end,
        }
        try:
            event = await client.execute(
                owner_id=owner_id,
                operation=lambda service: service.events().insert(
                    calendarId=calendar_id,
                    body=body,
                ),
            )
        except GoogleCalendarError as error:
            return failed(error.code, error.message)
        return success({"event": _event_payload(event)})

    async def update_calendar_event(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            event_id = required_string(arguments, "event_id", max_length=1_024)
            calendar_id = optional_string(arguments, "calendar_id", max_length=500) or "primary"
            patch: dict[str, Any] = {}
            for key, max_length in (("summary", 500), ("description", 10_000), ("location", 1_000)):
                if key in arguments and arguments[key] is not None:
                    patch[key] = _text_value(arguments, key, max_length=max_length, default="")
            range_value = _event_range(arguments, client.default_timezone, required=False)
        except ToolInputError as error:
            return input_error_payload(error)
        except CalendarInputError as error:
            return failed("invalid_tool_input", str(error))

        try:
            existing = await client.execute(
                owner_id=owner_id,
                operation=lambda service: service.events().get(
                    calendarId=calendar_id,
                    eventId=event_id,
                ),
            )
            if range_value is not None:
                start, end = range_value
                existing_all_day = isinstance(existing, dict) and "date" in existing.get("start", {})
                requested_all_day = "date" in start
                if existing_all_day != requested_all_day:
                    return failed(
                        "calendar_event_type_change_not_supported",
                        "Changing between timed and all-day events is not supported yet.",
                    )
                patch["start"] = start
                patch["end"] = end
            if not patch:
                return failed("invalid_tool_input", "At least one event field must be updated.")
            event = await client.execute(
                owner_id=owner_id,
                operation=lambda service: service.events().patch(
                    calendarId=calendar_id,
                    eventId=event_id,
                    body=patch,
                ),
            )
        except GoogleCalendarError as error:
            return failed(error.code, error.message)
        return success({"event": _event_payload(event)})

    async def delete_calendar_event(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            event_id = required_string(arguments, "event_id", max_length=1_024)
            calendar_id = optional_string(arguments, "calendar_id", max_length=500) or "primary"
        except ToolInputError as error:
            return input_error_payload(error)
        try:
            await client.execute(
                owner_id=owner_id,
                operation=lambda service: service.events().delete(
                    calendarId=calendar_id,
                    eventId=event_id,
                ),
            )
        except GoogleCalendarError as error:
            return failed(error.code, error.message)
        return success({"deleted": True, "calendar_id": calendar_id, "event_id": event_id})

    range_properties = {
        "start_time": {"type": ["string", "null"], "description": "RFC 3339 timestamp for a timed event."},
        "end_time": {"type": ["string", "null"], "description": "RFC 3339 timestamp for a timed event."},
        "start_date": {"type": ["string", "null"], "description": "YYYY-MM-DD for an all-day event."},
        "end_date": {"type": ["string", "null"], "description": "Exclusive YYYY-MM-DD end date for an all-day event."},
        "time_zone": {"type": ["string", "null"], "description": "IANA timezone; defaults to Europe/Dublin."},
    }
    return [
        ToolDefinition(
            name="list_calendar_events",
            description="List Google Calendar events in a required time window. Use primary unless another calendar is explicitly needed.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["time_min", "time_max"],
                "properties": {
                    "time_min": {"type": "string", "description": "Inclusive RFC 3339 window start."},
                    "time_max": {"type": "string", "description": "Exclusive RFC 3339 window end."},
                    "calendar_id": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
                },
            },
            handler=list_calendar_events,
        ),
        ToolDefinition(
            name="create_calendar_event",
            mutates_state=True,
            description="Create a timed or all-day Google Calendar event without confirmation. Supply exactly one complete time or date range.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["summary"],
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "calendar_id": {"type": ["string", "null"]},
                    **range_properties,
                },
            },
            handler=create_calendar_event,
        ),
        ToolDefinition(
            name="update_calendar_event",
            mutates_state=True,
            description="Update an existing Google Calendar event without confirmation. Timed/all-day type conversion is not supported.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["event_id"],
                "properties": {
                    "event_id": {"type": "string"},
                    "calendar_id": {"type": ["string", "null"]},
                    "summary": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    **range_properties,
                },
            },
            handler=update_calendar_event,
        ),
        ToolDefinition(
            name="delete_calendar_event",
            mutates_state=True,
            description="Delete a Google Calendar event without confirmation when the user asks or Jarvis determines it is appropriate.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["event_id"],
                "properties": {
                    "event_id": {"type": "string"},
                    "calendar_id": {"type": ["string", "null"]},
                },
            },
            handler=delete_calendar_event,
        ),
    ]


def _required_datetime(arguments: dict[str, Any], name: str) -> datetime:
    raw = required_string(arguments, name, max_length=100)
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise CalendarInputError(f"{name} must be an RFC 3339 timestamp.") from error
    if value.tzinfo is None:
        raise CalendarInputError(f"{name} must include a UTC offset or Z.")
    return value


def _event_range(
    arguments: dict[str, Any],
    default_timezone: str,
    *,
    required: bool,
) -> tuple[dict[str, str], dict[str, str]] | None:
    time_values = (arguments.get("start_time"), arguments.get("end_time"))
    date_values = (arguments.get("start_date"), arguments.get("end_date"))
    has_time = any(value is not None for value in time_values)
    has_date = any(value is not None for value in date_values)
    if has_time and has_date:
        raise CalendarInputError("Use either start_time/end_time or start_date/end_date, not both.")
    if not has_time and not has_date:
        if required:
            raise CalendarInputError("A complete timed or all-day event range is required.")
        return None
    if has_time:
        start = _required_datetime(arguments, "start_time")
        end = _required_datetime(arguments, "end_time")
        if end <= start:
            raise CalendarInputError("end_time must be after start_time.")
        timezone_name = _timezone(arguments.get("time_zone"), default_timezone)
        return (
            {"dateTime": start.isoformat(), "timeZone": timezone_name},
            {"dateTime": end.isoformat(), "timeZone": timezone_name},
        )
    start_date = _required_date(arguments, "start_date")
    end_date = _required_date(arguments, "end_date")
    if end_date <= start_date:
        raise CalendarInputError("end_date must be after start_date and is exclusive.")
    return ({"date": start_date.isoformat()}, {"date": end_date.isoformat()})


def _required_date(arguments: dict[str, Any], name: str) -> date:
    raw = required_string(arguments, name, max_length=10)
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise CalendarInputError(f"{name} must use YYYY-MM-DD format.") from error


def _timezone(raw: Any, default: str) -> str:
    value = raw.strip() if isinstance(raw, str) and raw.strip() else default
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise CalendarInputError("time_zone must be a valid IANA timezone.") from error
    return value


def _text_value(arguments: dict[str, Any], name: str, *, max_length: int, default: str) -> str:
    value = arguments.get(name, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise CalendarInputError(f"{name} must be a string.")
    if len(value) > max_length:
        raise CalendarInputError(f"{name} must be at most {max_length} characters.")
    return value.strip()


def _event_payload(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    return {
        "id": event.get("id"),
        "status": event.get("status"),
        "summary": event.get("summary", ""),
        "description": event.get("description", ""),
        "location": event.get("location", ""),
        "start": event.get("start", {}),
        "end": event.get("end", {}),
        "html_link": event.get("htmlLink"),
        "updated": event.get("updated"),
    }
