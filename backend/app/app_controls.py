"""Shared Phone/Voice intents over existing recording and presentation owners."""
import asyncio
import logging
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError


APP_CONTROL_DESCRIPTION = (
    "Control Spotify music, Jarvis recording, its sandbox desktop, or show a link on the phone. "
    "show_link displays url and description on the phone; returning a URL alone does not display it. "
    "media.start takes song_title and optional artist, or an exact spotify_uri; omit all to resume. "
    "Basic music playback uses this API directly, never Hermes or a browser. "
    "recording.start needs only confirmed include_jarvis_audio, no ID or state lookup; ask only if unanswered. "
    "Ordinary recording includes playing music. Only explicit voiceover mode silences and excludes all playback. "
    "Stop/volume actions need exact IDs from app state. A desktop view does not show terminal/headless work; "
    "do not open it as progress for that work. For browser/GUI tasks, open the app before showing its screen. "
    "An explicit desktop-inspection request may show it as-is. Resume only when requested. "
    "media.stop and media.volume "
    "need media_id from app state; volume is 0 to 1. Acceptance is not proof of playback, capture or display."
)


class AppCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str | None = Field(default=None, min_length=8, max_length=4096,
        description="Only show_link: exact HTTP(S) URL to offer on the phone, not an invented address.")
    description: str | None = Field(default=None, min_length=1, max_length=1000, pattern=r"\S",
        description="Only show_link: short user-facing explanation of the link.")
    recording_mode: Literal["ordinary", "voiceover"] | None = Field(default=None,
        description="Only recording.start. Omit for ordinary recording; voiceover means microphone only and requires include_jarvis_audio=false.")
    media_id: UUID | None = Field(default=None, description="Required only for media.stop or media.volume; use app.media.media_id.")
    spotify_uri: str | None = Field(default=None, pattern=r"^spotify:(track|album|playlist):[A-Za-z0-9]{22}$",
        description="Optional exact URI for media.start, instead of song_title/artist. Never guess; use the title when unknown.")
    song_title: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"\S",
        description="Only media.start: requested song title. Backend resolves it; no ID lookup or Hermes needed.")
    artist: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"\S",
        description="Only media.start with song_title: artist name if supplied by the user.")
    volume: float | None = Field(default=None, ge=0, le=1, strict=True,
        description="Required only for media.volume, from 0 (muted) to 1.")
    action: Literal["show_link", "recording.start", "recording.stop", "sandbox.start", "sandbox.stop", "media.start", "media.stop", "media.volume"]
    include_jarvis_audio: bool | None = Field(default=None, strict=True,
        description="Required for recording.start; null for every other action.")
    recording_id: UUID | None = Field(default=None,
        description="Required for recording.stop; null for every other action.")
    allocation_id: UUID | None = Field(default=None,
        description="Required for sandbox.start; null for every other action, including sandbox.stop.")
    generation: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$",
        description="Required for sandbox.start; null for every other action, including sandbox.stop.")
    resume: bool = Field(default=False, strict=True,
        description="True only for an explicitly requested sandbox.start resume; otherwise false.")
    view_id: UUID | None = Field(default=None,
        description="Required for sandbox.stop: use current app.view.id, not the allocation or presentation ID. Otherwise null.")

    @model_validator(mode="after")
    def exact_action(self):
        if self.action == "show_link":
            supplied = {key for key, value in self.model_dump().items()
                        if key != "action" and value is not None
                        and not (key == "resume" and value is False)}
            if supplied != {"url", "description"}:
                raise ValueError("show_link requires only url and description")
            if any(ord(char) <= 32 or ord(char) == 127 for char in self.url) or "\\" in self.url:
                raise ValueError("Link URL cannot contain whitespace, control characters or backslashes")
            parsed = urlsplit(self.url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError("Link must be an HTTP(S) URL without embedded credentials")
            _ = parsed.port
            return self
        if self.url is not None or self.description is not None:
            raise ValueError("Link fields belong only to show_link")
        if self.recording_mode is not None and (self.action != "recording.start"
                or (self.recording_mode == "voiceover" and self.include_jarvis_audio is not False)):
            raise ValueError("Recording mode belongs only to start; voiceover excludes Jarvis audio")
        if self.action.startswith("media."):
            allowed = {"media.start": {"spotify_uri", "song_title", "artist"}, "media.stop": {"media_id"},
                       "media.volume": {"media_id", "volume"}}[self.action]
            supplied = {key for key, value in self.model_dump().items()
                        if key != "action" and value is not None and not (key == "resume" and value is False)}
            if supplied - allowed or (self.action != "media.start" and supplied != allowed):
                raise ValueError("Use only the fields belonging to this media action")
            if (self.spotify_uri is not None and self.song_title is not None) or (self.artist is not None and self.song_title is None):
                raise ValueError("Use song_title and optional artist, or spotify_uri, not both")
            return self
        if any(value is not None for value in (self.media_id, self.spotify_uri, self.volume, self.song_title, self.artist)):
            raise ValueError("Media fields are only valid for media actions")
        required = {"recording.start": {"include_jarvis_audio"}, "recording.stop": {"recording_id"},
                    "sandbox.start": {"allocation_id", "generation"}, "sandbox.stop": {"view_id"}}[self.action]
        supplied = {key for key in ("include_jarvis_audio", "recording_id", "allocation_id", "generation", "view_id")
                    if getattr(self, key) is not None}
        if supplied != required or (self.resume and self.action != "sandbox.start"):
            raise ValueError("Supply only the exact action's parameters; recording start needs an explicit audio choice")
        return self


def unavailable(summary):
    return {"status": "success", "outcome": "unavailable", "effect_committed": False, "summary": summary}


class AppControlService:
    async def invoke(
        self,
        owner_id: str,
        session_id: UUID,
        action: str,
        *,
        operation_id: UUID | None = None,
        notify_voice: bool = False,
        **arguments,
    ) -> dict:
        """Shared Phone/Voice entrypoint; callers may supply a stable command ID."""
        from uuid import uuid4

        return await self.dispatch(
            owner_id=owner_id,
            session_id=session_id,
            operation_id=operation_id if operation_id is not None else uuid4(),
            command={"action": action, **arguments},
            notify_voice=notify_voice,
        )

    def projection(self, owner_id: str) -> dict:
        """Read the current phone/view state without database or provider work."""
        self._owner(owner_id)
        application = self._application
        session_id = application.context.jarvis_session_id if application is not None else None
        return self.snapshot(owner_id=owner_id, session_id=session_id)

    """One active phone binding, not a second recorder, job or sandbox registry.

    Main passes trusted owner/session/call identities to dispatch. The HTTP
    adapter derives owner from authentication. Domain services retain authority.
    """
    def __init__(self, *, owner_id, pool, sandbox_views, presentations, media_audio=None):
        self.owner_id, self.pool = owner_id, pool
        self.views, self.presentations = sandbox_views, presentations
        self.media_audio = media_audio
        self._application = None
        self._last_command = None
        self._link_requests = OrderedDict()
        self.send_link_notice = None
        self._inventory_task = None

    @staticmethod
    def definition():
        return {"name": "control_app", "parameters": AppCommand.model_json_schema(),
                "description": APP_CONTROL_DESCRIPTION}

    def _owner(self, owner_id):
        if owner_id != self.owner_id:
            raise JarvisHTTPError(403, "app_control_owner_mismatch", "App controls belong to another account.")

    def bind(self, application):
        self._owner(application.context.owner_id)
        if self._application is not application:
            self._disconnect_links(self._application)
        if self._inventory_task is not None:
            self._inventory_task.cancel()
            self._inventory_task = None
        self._application = application
        if self.views is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._inventory_task = loop.create_task(self._refresh_inventory(application))

    def unbind(self, application):
        self._disconnect_links(application)
        if self._application is application:
            self._application = None
            if self._inventory_task is not None:
                self._inventory_task.cancel()
                self._inventory_task = None

    async def _refresh_inventory(self, application):
        # Reuse the viewer's owner-checked cache. Never create or resume a VM,
        # block a state read, or keep polling after this phone binding closes.
        while self._application is application:
            try:
                await asyncio.wait_for(self.views.choices(
                    self.owner_id, application.context.jarvis_session_id, limit=32,
                ), timeout=10)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Background sandbox inventory refresh unavailable (%s)", type(exc).__name__,
                )
            await asyncio.sleep(15)

    def _active(self, session_id):
        application = self._application
        if (application is None or application.context.jarvis_session_id != session_id
                or not application.is_current() or not application.is_ready()):
            return None
        return application

    def snapshot(self, owner_id, session_id):
        """Detached RAM state; no DB, provider, model or device request."""
        self._owner(owner_id)
        application = self._active(session_id)
        recording = application.snapshot().get("recording") if application is not None else None
        return {"phone_connected": application is not None, "recording": deepcopy(recording),
                "media": self.media_audio.snapshot(owner_id, session_id) if self.media_audio else None,
                "sandboxes": self.views.cached_inventory(owner_id) if self.views is not None else None,
                "view": self.views.current(owner_id) if self.views is not None else None,
                "view_basis": "server_lease_not_display_ack",
                "last_command": deepcopy(self._last_command) if self._last_command is not None
                    and self._last_command["session_id"] == str(session_id) else None}

    def _notify(self, session_id, payload):
        application = self._active(session_id)
        if application is None:
            return False
        return application.offer_event({**payload, "session_id": str(session_id),
                                        "audio_session_id": application.context.audio_session_id}) is True

    async def dispatch(self, *, owner_id, session_id, operation_id, command, notify_voice=False):
        self._owner(owner_id)
        if not isinstance(session_id, UUID) or not isinstance(operation_id, UUID):
            raise JarvisHTTPError(422, "app_control_identity_invalid", "Use exact session and operation UUIDs.")
        parsed = AppCommand.model_validate(command)
        async with self.pool.acquire() as connection:
            owned = await connection.fetchval("select true from jarvis_sessions where owner_id=$1 and id=$2",
                                              owner_id, session_id)
        if not owned:
            raise JarvisHTTPError(404, "app_control_session_unavailable", "This session is unavailable.")
        record = {"operation_id": str(operation_id), "session_id": str(session_id),
                  "action": parsed.action, "outcome": "pending"}
        self._last_command = record
        try:
            result = await self._dispatch(owner_id, session_id, operation_id, parsed,
                                          notify_voice=notify_voice)
        except BaseException:
            record["outcome"] = "unconfirmed"
            raise
        record.update(outcome=result["outcome"], summary=result["summary"])
        return result

    def _disconnect_links(self, application):
        for entry in self._link_requests.values():
            if entry.application is application:
                if not entry.reply.done():
                    entry.reply.set_result("unavailable")
                if entry.notice_task is not None and not entry.notice_task.done():
                    entry.notice_task.cancel()

    def notice_current(self, notice):
        """Only the acknowledged link's original phone/voice binding may receive it."""
        try:
            entry = self._link_requests.get(UUID(notice["id"]))
            return (entry is not None and entry.notice == notice
                    and self._active(entry.session_id) is entry.application
                    and datetime.now(UTC) < datetime.fromisoformat(notice["expires_at"]))
        except (KeyError, TypeError, ValueError):
            return False

    async def _send_link_notice(self, entry):
        if self.send_link_notice is None or not self.notice_current(entry.notice):
            entry.notice_status = "unavailable"
            return
        entry.notice_status = "sending"
        try:
            result = await asyncio.wait_for(self.send_link_notice(self.owner_id, entry.notice), timeout=5)
            entry.notice_status = result.get("status", "unknown")
        except asyncio.CancelledError:
            entry.notice_status = "unknown"
            raise
        except Exception:
            # The phone effect already happened. Never retry that effect or
            # replay an uncertain announcement into another voice connection.
            entry.notice_status = "unknown"
            logging.getLogger(__name__).warning("Link voice confirmation was not confirmed")

    @staticmethod
    def _link_result(operation_id, outcome):
        return {"status": "success", "operation_id": str(operation_id),
                "outcome": {"displayed": "link_displayed", "unavailable": "unavailable"}.get(outcome, "link_sent"),
                "effect_committed": True if outcome == "displayed" else False if outcome == "unavailable" else None,
                "summary": {"displayed": "The link is displayed on your phone. Tap Open link to continue.",
                            "unavailable": "Link display could not be confirmed; the phone may be unavailable."}.get(
                                outcome, "The link was sent to your phone; display is not yet confirmed.")}

    async def _show_link(self, session_id, operation_id, command, *, notify_voice=False):
        entry = self._link_requests.get(operation_id)
        if entry is not None:
            if (entry.session_id, entry.url, entry.description) != (session_id, command.url, command.description):
                raise JarvisHTTPError(409, "app_link_operation_conflict", "This operation already identifies another link.")
        else:
            application = self._active(session_id)
            if application is None:
                return unavailable("The phone is not connected; the link was not sent.")
            # Bounded display receipts, not a persistent notification queue.
            if len(self._link_requests) >= 128:
                oldest = next((key for key, value in self._link_requests.items()
                               if value.reply.done() and
                               (value.notice_task is None or value.notice_task.done())), None)
                if oldest is None:
                    return unavailable("Too many links are awaiting phone confirmation.")
                del self._link_requests[oldest]
            entry = SimpleNamespace(session_id=session_id, application=application,
                                    url=command.url, description=command.description,
                                    notify_voice=notify_voice and self.send_link_notice is not None,
                                    notice=None, notice_task=None, notice_status="pending",
                                    reply=asyncio.get_running_loop().create_future())
            self._link_requests[operation_id] = entry
            if not self._notify(session_id, {"type": "app.link", "operation_id": str(operation_id),
                                            "url": command.url, "description": command.description}):
                entry.reply.set_result("unavailable")
        try:
            outcome = await asyncio.wait_for(asyncio.shield(entry.reply), timeout=2)
        except asyncio.TimeoutError:
            outcome = None
        result = self._link_result(operation_id, outcome)
        if entry.notify_voice:
            result["voice_confirmation"] = {"owner": "backend", "status": entry.notice_status,
                "note": "The backend reports the phone acknowledgement to Voice; do not repeat that confirmation."}
        return result

    def acknowledge_link(self, owner_id, session_id, operation_id, outcome):
        self._owner(owner_id)
        entry = self._link_requests.get(operation_id)
        if entry is None or entry.session_id != session_id:
            raise JarvisHTTPError(404, "app_link_not_found", "No matching link request exists.")
        if outcome not in {"displayed", "unavailable"}:
            raise JarvisHTTPError(422, "app_link_outcome_invalid", "Unsupported link display outcome.")
        if not entry.reply.done():
            if self._active(session_id) is not entry.application:
                raise JarvisHTTPError(409, "app_link_session_closed", "The link's phone session is no longer active.")
            entry.reply.set_result(outcome)
            if entry.notify_voice:
                entry.notice = {"id": str(operation_id), "kind": "app_action_result",
                    "audio_session_id": str(entry.application.context.audio_session_id),
                    "text": ("The requested link is now displayed on the phone." if outcome == "displayed"
                             else "The phone could not display the requested link."),
                    "expires_at": (datetime.now(UTC) + timedelta(seconds=30)).isoformat()}
                entry.notice_task = asyncio.create_task(self._send_link_notice(entry),
                                                       name="app-link-voice-confirmation")
        result = self._link_result(operation_id, entry.reply.result())
        if self._last_command is not None and self._last_command.get("operation_id") == str(operation_id):
            self._last_command.update(outcome=result["outcome"], summary=result["summary"])
        return result

    async def _dispatch(self, owner_id, session_id, operation_id, command, *, notify_voice=False):
        if command.action == "show_link":
            return await self._show_link(session_id, operation_id, command, notify_voice=notify_voice)
        if command.action.startswith("media."):
            from app.media_audio import MediaError
            if self.media_audio is None or self._active(session_id) is None:
                return unavailable("Music playback is not available in this connected session.")
            try:
                if command.action == "media.start":
                    lease = await self.media_audio.start(session_id=session_id, media_id=operation_id,
                        uri=command.spotify_uri, song_title=command.song_title, artist=command.artist)
                    sent = self._notify(session_id, {"type": "media.command", "action": "start",
                                                    "media_id": str(lease.id), "volume": lease.volume})
                    if not sent:
                        await self.media_audio.stop(session_id=session_id, media_id=lease.id)
                        return unavailable("The phone did not accept the music connection.")
                    playback = await self.media_audio.wait_for_playback(lease)
                    title = lease.song_title or "music"
                    if lease.artist:
                        title += " by " + lease.artist
                    if playback == "playing":
                        return {"status": "success", "outcome": "media_playing", "media_id": str(lease.id),
                                "playback_confirmed": True, "summary": "Playing " + title + "."}
                    if playback in ("failed", "stopped", "replaced", "disconnected"):
                        return unavailable("Music did not start; the connection failed or the request was stopped or replaced.")
                    return {"status": "success", "outcome": "media_requested", "media_id": str(lease.id),
                            "playback_confirmed": False, "summary": "Starting " + title + "."}
                lease = self.media_audio.current(session_id, command.media_id)
                if command.action == "media.stop":
                    await self.media_audio.stop(session_id=session_id, media_id=command.media_id)
                    self._notify(session_id, {"type": "media.command", "action": "stop", "media_id": str(lease.id)})
                    return {"status": "success", "outcome": "media_stopped", "summary": "Music stream stopped."}
                sent = self._notify(session_id, {"type": "media.command", "action": "volume",
                                                "media_id": str(lease.id), "volume": command.volume})
                if not sent:
                    return unavailable("The phone is unavailable for the volume change.")
                lease.volume = command.volume
                return {"status": "success", "outcome": "media_volume_requested", "summary": "Volume change sent to phone."}
            except (MediaError, OSError) as error:
                return unavailable(str(error) if isinstance(error, MediaError) else "Music connection unavailable.")
        if command.action.startswith("recording."):
            application = self._active(session_id)
            if application is None:
                return unavailable("No connected phone recording dispatcher is ready.")
            arguments = ({"action": "start", "include_jarvis_audio": command.include_jarvis_audio}
                         if command.action == "recording.start" else
                         {"action": "stop", "recording_id": str(command.recording_id)})
            if command.action == "recording.start" and command.recording_mode is not None:
                arguments["recording_mode"] = command.recording_mode
            # Stable command namespace, not a fabricated speech/request record.
            return await application.execute_recording(arguments, provider_call_id=str(operation_id),
                backend_session_key="app-control:" + str(session_id),
                admit_if=lambda: self._active(session_id) is application)
        if self.views is None:
            return unavailable("Sandbox viewing is unavailable.")
        if command.action == "sandbox.start":
            if self.presentations is None:
                return unavailable("Phone presentation delivery is unavailable.")
            ctx = SimpleNamespace(owner_id=owner_id, session_id=session_id, operation_id=operation_id)
            result = await self.views.present(ctx, self.presentations, allocation_id=command.allocation_id,
                                             generation=command.generation, resume=command.resume)
            if result.get("presentation", {}).get("state") == "pending":
                self._notify(session_id, {"type": "workspace.presentation.intent",
                                          "request_id": result["presentation"]["request_id"]})
            return result
        closed = self.views.stop(owner_id, command.view_id)
        notified = self._notify(session_id, {"type": "workspace.sandbox.closed", "view_id": str(command.view_id)})
        return {"status": "success", "outcome": "server_view_closed" if closed else "view_not_active",
                "effect_committed": closed, "view_id": str(command.view_id), "device_command_enqueued": notified,
                "summary": "The selected server view is closed. Phone dismissal is not confirmed; sandbox work continues."}


class AppControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    operation_id: UUID
    command: AppCommand


class AppLinkAcknowledgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    operation_id: UUID
    outcome: Literal["displayed", "unavailable"]


router = APIRouter(prefix="/app-controls", tags=["app-controls"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]


def service(request):
    value = getattr(getattr(request.app.state, "runtime_services", None), "app_controls", None)
    if value is None:
        raise JarvisHTTPError(503, "app_controls_unavailable", "App controls are not configured.")
    return value


@router.post("")
async def dispatch(body: AppControlRequest, request: Request, principal: Principal):
    return await service(request).dispatch(owner_id=principal.owner_id, session_id=body.session_id,
        operation_id=body.operation_id, command=body.command)


@router.get("")
async def snapshot(request: Request, principal: Principal, session_id: UUID):
    return service(request).snapshot(principal.owner_id, session_id)


@router.post("/link-ack")
async def acknowledge_link(body: AppLinkAcknowledgment, request: Request, principal: Principal):
    return service(request).acknowledge_link(
        principal.owner_id, body.session_id, body.operation_id, body.outcome)
