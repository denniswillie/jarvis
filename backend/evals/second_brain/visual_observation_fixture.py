"""Owned visual producer fixture; synthetic by default, optional explicit native analyzer."""
from types import SimpleNamespace

from app.assistant.repository import BehaviorProfileRepository
from app.memory.prospective.context import ProspectiveContextBuilder
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.reminders.repository import ReminderRepository
from app.visual_memory.interrupts import VisualInterruptService
from app.perception.contracts import FastPerceptionOutput
from app.perception.frame_window import RecentVisualFrameWindow
from app.perception.repository import PerceptionRepository
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.visual_memory.pipeline import VisualAnalysisPipeline
from app.visual_memory.repository import VisualFrameRepository, _record_from_row


async def run_visual_fixture(fixture, publisher, frame_id, summary, *, on_analyze=None,
                             analyzer_output=None, notification_observer=None, attention_repository=None,
                             native_analyzer=None, image_bytes=None, settings=None):
    if native_analyzer is not None and (not image_bytes or analyzer_output is not None):
        raise ValueError("Native fixture needs actual pixels and cannot override model output")
    requests, diagnostics, notifications, outputs = [], [], [], []
    async def analyze(request):
        requests.append(request)
        if on_analyze is not None:
            await on_analyze(request)
        output = (await native_analyzer.analyze(request) if native_analyzer is not None else
                  analyzer_output or FastPerceptionOutput(summary=summary, confidence=0.8, starts_new_event=False))
        outputs.append(output)
        return output
    async def record_visual(**kwargs):
        notifications.append({"kind": "visual", **kwargs})
        if notification_observer is not None:
            await notification_observer("visual", kwargs)
        return False

    async def record_prospective(**kwargs):
        notifications.append({"kind": "prospective", **kwargs})
        if notification_observer is not None:
            await notification_observer("prospective", kwargs)
        return None
    async def diagnostic(event):
        diagnostics.append(event)
    async with fixture.pool.acquire() as connection:
        row = await connection.fetchrow(
            "select * from visual_frames where owner_id=$1 and id=$2", fixture.owner, frame_id)
    if row is None:
        raise ValueError("Fixture frame is outside its owned scope")
    frame = _record_from_row(row)
    repository = PerceptionRepository(fixture.pool)
    policy = WorkingMemoryPolicy(ttl_seconds=120, entity_ttl_seconds=120,
        visible_text_ttl_seconds=120, max_entities=40, max_recent_changes=30)
    settings = settings or JarvisSettings(_env_file=None)
    visual_repository = VisualFrameRepository(fixture.pool)
    prospective = ProspectiveRepository(fixture.pool)
    manager = SimpleNamespace(speak_visual_interrupt=record_visual,
                              speak_prospective_interrupt=record_prospective,
                              attention_notifications_enabled=attention_repository is not None,
                              attention_repository=attention_repository)
    working = WorkingMemoryService(repository=repository, policy=policy)
    pipeline = VisualAnalysisPipeline(
        settings=settings, repository=visual_repository,
        session_repository=SessionRepository(fixture.pool),
        behavior_repository=BehaviorProfileRepository(fixture.pool),
        interrupt_service=VisualInterruptService(repository=visual_repository,
            realtime_session_manager=manager, dedupe_seconds=settings.VISUAL_INTERRUPT_DEDUPE_SECONDS,
            max_age_seconds=settings.PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS),
        model_limiter=PriorityAsyncLimiter(1),
        fast_analyzer=SimpleNamespace(provider=native_analyzer.provider if native_analyzer else "fixture",
            model=native_analyzer.model if native_analyzer else "scripted-not-a-model-call", analyze=analyze),
        perception_repository=repository, sensory_frame_window=RecentVisualFrameWindow(
            max_frames_per_session=4, ttl_seconds=10),
        domain_event_bus=DomainEventBus(),
        working_memory_service=working,
        prospective_context_builder=ProspectiveContextBuilder(focus_reader=prospective,
            reminder_reader=ReminderRepository(fixture.pool), goal_reader=None,
            owner_working_memory_reader=working, max_conditional_reminders=4, max_goals=4, max_commitments=4),
        prospective_service=ProspectiveMemoryService(repository=prospective,
            realtime_session_manager=manager, settings=settings))
    pipeline.set_observation_publication(publisher)
    try:
        await pipeline.enqueue(frame=frame, image_bytes=image_bytes if image_bytes is not None else b"synthetic-not-real-pixels",
                               send_diagnostic=diagnostic)
        await pipeline.drain()
    finally:
        await pipeline.close()
    if notifications and notification_observer is None:
        raise AssertionError("Synthetic fixture produced an unexpected notification candidate")
    return {"requests": requests, "diagnostics": diagnostics, "notifications": notifications, "outputs": outputs}
