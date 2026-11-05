"""Session-owned evidence capture bound to the actual camera socket lease."""
from dataclasses import replace
from datetime import UTC, datetime

from app.knowledge.models import SourceError
from app.knowledge.observations import ObservationPublication
from app.perception.evidence_consumer import PerceptionEvidenceConsumer, canonical_episode_snapshots
from app.perception.frame_window import RecentVisualFrameWindow


class CameraAssignmentBinding:
    """Synchronous capture before model work; no storage, model or per-frame task.

    Membership is checked once at request capture. Later frames and retention
    eviction do not revoke the original analysis; a stopped/replaced connection
    does, including reconnects that reuse the same visual-session ID.
    """

    def __init__(self, window: RecentVisualFrameWindow, scope, *, is_live=lambda: True):
        self.window, self.scope, self.is_live = window, scope, is_live

    def capture(self, request):
        if (not self.is_live() or (request.owner_id, request.session_id) != (
                self.scope.owner_id, self.scope.session_id)):
            return request
        if request.camera_connection_id is not None:
            # Never relabel an already captured request as the new connection.
            return request
        if request.perception_brief_input_digest is not None:
            raise ValueError("Camera ownership must be captured before input sealing")
        lease = self.window.stream_lease(request.owner_id, request.session_id)
        if lease is None:
            return request
        available = {frame.frame_id: frame for frame in self.window.window(request.owner_id, request.session_id)}
        for frame in request.frames:
            original = available.get(frame.frame_id)
            if original is None or (
                    original.owner_id, original.session_id, original.visual_session_id,
                    original.frame_sequence, original.captured_at, original.source,
                    original.content_type, original.image_bytes) != (
                    request.owner_id, request.session_id, lease.visual_session_id,
                    frame.frame_sequence, frame.captured_at, frame.source,
                    frame.content_type, frame.image_bytes):
                return request
        return replace(request, camera_visual_session_id=lease.visual_session_id,
                       camera_connection_id=lease.connection_id)

    def is_current(self, request, visual_session_id):
        if (not self.is_live() or (request.owner_id, request.session_id) != (
                self.scope.owner_id, self.scope.session_id)
                or request.camera_visual_session_id != visual_session_id
                or request.camera_connection_id is None):
            return False
        lease = self.window.stream_lease(request.owner_id, request.session_id)
        return (lease is not None and lease.visual_session_id == visual_session_id
                and lease.connection_id == request.camera_connection_id)


class SessionPerceptionEvidence:
    """Borrow existing owners; registration/close fence all pending evidence."""

    def __init__(self, *, canonical, publication, window, clock=lambda: datetime.now(UTC),
                 resolver=None, focus_resolver=None):
        self.scope = canonical.scope
        if publication.owner_id != self.scope.owner_id:
            raise ValueError("Evidence publication belongs to another owner")
        self._active = False
        self._closed = False
        self.resolver = resolver
        self.focus_resolver = focus_resolver
        self.focus_resolution_issue = None
        self.last_focus_resolution_scheduled = False
        self.resolution_issue = None
        self.last_resolution_scheduled = False
        self.camera = CameraAssignmentBinding(window, self.scope,
            is_live=lambda: self._active and not self._closed)
        self.consumer = _SessionEvidenceConsumer(self, publication, enabled=True,
            camera_is_current=self.camera.is_current,
            episode_snapshots=canonical_episode_snapshots(canonical), clock=clock)

    def activate(self):
        if self._closed:
            raise RuntimeError("A retired evidence session cannot reactivate")
        self._active = True

    def capture(self, request):
        return self.camera.capture(request)

    def close(self):
        self._active = False
        self._closed = True
        if self.resolver is not None:
            self.resolver.retire()
        if self.focus_resolver is not None:
            self.focus_resolver.retire()

    async def aclose(self):
        if not self._closed:
            self.close()
        if self.resolver is not None:
            await self.resolver.close()
        if self.focus_resolver is not None:
            await self.focus_resolver.close()


class _SessionEvidenceConsumer(PerceptionEvidenceConsumer):
    """Only committed, original-session evidence may request owner evaluation."""

    def __init__(self, session, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session

    async def consume(self, *, request, output, observation, analysis_input, effect_guard):
        result = await super().consume(request=request, output=output, observation=observation,
            analysis_input=analysis_input, effect_guard=effect_guard)
        session = self.session
        session.last_resolution_scheduled = False
        session.last_focus_resolution_scheduled = False
        if (result.status == "recorded" and session.resolver is not None
                and session._active and not session._closed):
            try:
                # consume returned after its evidence transaction committed.
                # offer only schedules bounded work; never await the model here.
                session.last_resolution_scheduled = session.resolver.offer(request=request,
                    evidence=result, effect_guard=effect_guard, camera_is_current=session.camera.is_current)
                session.resolution_issue = None
            except Exception:
                # Scene recording remains useful even if optional settlement
                # admission is unavailable. No success or notification is implied.
                session.resolution_issue = "visual_resolution_admission_failed"
        if (result.status == "recorded" and session.focus_resolver is not None
                and session._active and not session._closed):
            try:
                session.last_focus_resolution_scheduled = session.focus_resolver.offer(request=request,
                    evidence=result, effect_guard=effect_guard, camera_is_current=session.camera.is_current)
                session.focus_resolution_issue = None
            except Exception:
                session.focus_resolution_issue = "focus_resolution_admission_failed"
        return result


def create_session_perception_evidence(live, pipeline, *, clock):
    """Unavailable/default-off perception never gains a fabricated camera owner."""
    if pipeline is None or not getattr(pipeline, "supports_perception_briefs", False):
        return None
    publication = getattr(pipeline, "perception_publication", None)
    window = getattr(pipeline, "perception_frame_window", None)
    if (not isinstance(publication, ObservationPublication)
            or not isinstance(window, RecentVisualFrameWindow)
            or not callable(getattr(pipeline, "register_perception_evidence_session", None))):
        return None
    resolver = None
    episodes = getattr(live, "episode_context", None)
    resources = getattr(pipeline, "perception_resolution_resources", None)
    if resources is not None and callable(getattr(episodes, "require_owner", None)):
        try:
            owner = episodes.require_owner()
        except SourceError:
            owner = None
        if owner is not None:
            from app.reminders.visual_resolution import EpisodeVisualResolver
            model, limiter, bus = resources
            resolver = EpisodeVisualResolver(owner=owner, canonical=live.binder,
                model=model, limiter=limiter, bus=bus, enabled=True)
    focus_resolver = None
    prospective = getattr(pipeline, "perception_focus_service", None)
    if resources is not None and callable(getattr(prospective, "process_matches", None)):
        from app.perception.focus_resolution import FocusVisualResolver
        model, limiter, _ = resources
        focus_resolver = FocusVisualResolver(canonical=live.binder, prospective=prospective,
            model=model, limiter=limiter, enabled=True)
    return SessionPerceptionEvidence(canonical=live.binder, publication=publication,
        window=window, clock=clock, resolver=resolver, focus_resolver=focus_resolver)
