"""Isolated SQL composition for the existing replay port, not a second pipeline.

The caller owns a manifested disposable schema and its independent cleanup audit.
Only controlled model inputs and ambient context are fixture boundaries. Camera
capture, composition/binding, publication, evidence, focus and attention execute
their existing production implementations. This is not visual-quality scoring.
"""
import asyncio
from datetime import UTC, datetime
import hashlib
import json
import traceback
from types import SimpleNamespace
from uuid import UUID, uuid4

from app.attention.repository import AttentionRepository
from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import request_document
from app.knowledge.observations import ObservationPublication
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import LiveSnapshot, SessionScope, StateCursor
from app.live_state.perception_runtime import SessionPerceptionRuntime
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.evidence_consumer import EVIDENCE_NAMESPACE
from app.perception.evidence_session import create_session_perception_evidence
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.repository import PerceptionRepository
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.visual_memory.context import VisualAnalyzerContext
from app.visual_memory.models import VisualFrameRecord
from app.visual_memory.pipeline import VisualAnalysisPipeline
from app.visual_memory.repository import VisualFrameRepository
from evals.responsive_voice.video_replay import BudgetedTemporalModel, PipelineReplayPort


def now():
    return datetime.now(UTC)


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


class CapturingAnalyzer(FastPerceptionAnalyzer):
    def __init__(self, model):
        super().__init__(model)
        self.requests, self.outputs = [], []

    async def analyze(self, request):
        self.requests.append(request)
        output = await super().analyze(request)
        self.outputs.append(output)
        return output


class FocusPipelineReplay:
    """Compose only against an already isolated, caller-owned database scope."""

    def __init__(self, *, pool, connection, owner, area, model, schema,
                 execution_kind='synthetic', evaluation_provenance=None):
        if not schema.startswith('jarvis_sb_eval_') or not owner.startswith('direct-image-eval-second-brain-'):
            raise ValueError('replay_requires_disposable_evaluation_scope')
        if execution_kind not in {'synthetic', 'hosted_model'}:
            raise ValueError('unsupported_canonical_replay_execution_kind')
        self.execution_kind = execution_kind
        self.evaluation_provenance = dict(evaluation_provenance or {})
        self.pool, self.connection, self.owner, self.area = pool, connection, owner, area
        self.model, self.schema = model, schema
        self.clock = self.runtime = self.entry = self.scope = None
        self.replay_id, self.prepared_at, self.ready_at = uuid4().hex, None, None
        self.prepared_context, self.errors = None, []
        self.withdrawal = None
        self.focuses, self.briefs, self.frames, self.receipts = [], [], [], []
        self.closed, self.speech_attempts = False, 0
        self.sources = SourceRepository(pool)
        tasks = TaskRepository(pool, TaskLimits())
        self.context_repository = ContextRepository(tasks,
            publication=CanonicalSourcePublication(self.sources, area))
        self.publication = ObservationPublication(self.context_repository, owner_id=owner)
        settings = SimpleNamespace(VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS=15,
            VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES=4, VISUAL_SENSORY_WINDOW_TTL_SECONDS=30,
            VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=0.9,
            PROSPECTIVE_FIND_ONCE_DEFAULT_TTL_SECONDS=300, PROSPECTIVE_MONITOR_DEFAULT_TTL_SECONDS=300,
            PROSPECTIVE_MAX_CANDIDATES_PER_ANALYSIS=4, PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS=10,
            PROSPECTIVE_TRIGGER_CLAIM_LEASE_SECONDS=30, VISUAL_INTERRUPT_DEDUPE_SECONDS=30)
        manager = SimpleNamespace(attention_notifications_enabled=True,
            attention_repository=AttentionRepository(tasks), speak_prospective_interrupt=self.forbid_speech)
        self.prospective = ProspectiveMemoryService(repository=ProspectiveRepository(pool),
            realtime_session_manager=manager, settings=settings)
        self.window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=30, now=now)
        self.analyzer = CapturingAnalyzer(model)
        self.pipeline = VisualAnalysisPipeline(settings=settings, repository=VisualFrameRepository(pool),
            session_repository=object(), behavior_repository=object(),
            interrupt_service=SimpleNamespace(maybe_interrupt=self.forbid_speech),
            model_limiter=PriorityAsyncLimiter(1), fast_analyzer=self.analyzer,
            perception_repository=PerceptionRepository(pool, observation_publication=self.publication),
            sensory_frame_window=self.window, prospective_service=self.prospective,
            live_brief_bridge_for=lambda owner_id, session_id: self.runtime.bridge
                if self.runtime and (owner_id, session_id) == (self.owner, self.scope.session_id) else None)
        self.pipeline.set_observation_publication(self.publication)
        self.pipeline._context_builder.build = self.build_ambient_context

    async def forbid_speech(self, *args, **kwargs):
        self.speech_attempts += 1
        raise AssertionError('replay_cannot_speak_or_claim_delivery')

    def port(self, *, port_type=PipelineReplayPort, **kwargs):
        def traced(operation, callback):
            async def invoke(*args, **options):
                try:
                    return await callback(*args, **options)
                except Exception as error:
                    self.errors.append({'operation':operation, 'error_type':type(error).__name__,
                        'code':getattr(error, 'code', None), 'sqlstate':getattr(error, 'sqlstate', None),
                        'frames':[{'file':frame.filename.rsplit('/', 1)[-1], 'line':frame.lineno,
                                   'function':frame.name}
                                  for frame in traceback.extract_tb(error.__traceback__)[-5:]]})
                    raise
            return invoke
        return port_type(self.pipeline, execution_kind=self.execution_kind, begin=traced('open', self.open),
            apply_context=traced('context', self.context), make_frame=traced('make_frame', self.make_frame),
            collect=traced('collect', self.collect), cleanup=traced('close', self.close), **kwargs)

    async def append(self, *, event_id, occurred_at, text=None, receipt=None, provenance=None):
        payload = dict(receipt) if receipt else {'text': text}
        payload['replay_provenance'] = {
            **self.evaluation_provenance,
            'execution_kind': self.execution_kind,
            'controlled_model_outputs': self.execution_kind == 'synthetic',
            'recognition_qualified': False, 'replay_id':self.replay_id, 'event_id':event_id,
            'phase':'capture' if self.clock else 'preparation',
            'origin':self.clock.origin.isoformat() if self.clock else None,
            'prepared_at':self.prepared_at.isoformat(),
            **(provenance or {})}
        identity = f'{self.replay_id}/{self.scope.session_id}/{event_id}'
        try:
            record = await self.context_repository.append(self.owner, ContextIngress(
                session_id=self.scope.session_id, source_namespace='evaluation:video-replay',
                source_event_id=identity, receipt_id='replay:v1', expected_revision=0,
                kind='tool_outcome' if receipt else 'user_transcript',
                trust='tool_receipt' if receipt else 'user_input', source_time=occurred_at, payload=payload))
        except Exception as error:
            self.errors.append({'operation':'context_append', 'event_id':event_id,
                'error_type':type(error).__name__, 'code':getattr(error, 'code', None)})
            raise
        self.receipts.append(record)
        return record

    async def admit(self, target, occurred_at):
        focus = await self.prospective.set_visual_focus(owner_id=self.owner, target=target,
            description=('Explicit synthetic replay target' if self.execution_kind == 'synthetic'
                         else 'Explicit isolated hosted replay observation target'),
            purpose='Observe the requested replay target',
            mode='monitor', expires_in_seconds=300)
        self.focuses.append(focus)
        await self.append(event_id='focus:'+str(focus.id), occurred_at=occurred_at,
            receipt={'tool_name':'set_visual_focus', 'result':{
                'status':'success', 'visual_focus':focus.as_prompt_payload()}})
        return focus

    async def prepare(self, initial_context):
        """Admit initial context before the caller chooses the capture origin.

        Production rejects captures predating the focus or bound brief. An
        accelerated/historical origin is not a substitute for that authority.
        This preparation consumes no frames, labels, future events or model calls.
        """
        if not initial_context:
            raise ValueError('replay_requires_explicit_initial_focus')
        if self.scope is not None:
            raise ValueError('replay_scope_already_prepared')
        self.prepared_context, self.prepared_at = tuple(initial_context), now()
        current_schema = await self.connection.fetchval('select current_schema()')
        if current_schema != self.schema:
            raise ValueError('replay_database_scope_changed')
        session = await self.connection.fetchval(
            'insert into jarvis_sessions(owner_id) values($1) returning id', self.owner)
        self.scope = SessionScope(owner_id=self.owner, session_id=session, area_id=self.area)
        self.lease = self.window.begin_stream(self.owner, session, uuid4())
        for index, text in enumerate(initial_context):
            await self.append(event_id='initial:'+str(index), occurred_at=self.prepared_at, text=text)
        await self.admit(initial_context[0], self.prepared_at)
        self.canonical = CanonicalContextBinder(self.sources, self.scope, clock=now, timeout_seconds=18)
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
            as_of=now(), readiness='ready')
        self.runtime = SessionPerceptionRuntime(service=SimpleNamespace(snapshot=lambda:self.snapshot),
            canonical=self.canonical, bus=DomainEventBus(), clock=now, enabled=True)
        await self.refresh()
        self.ready_at = now()

    async def open(self, initial_context, clock, budget):
        if self.ready_at is None or tuple(initial_context) != self.prepared_context:
            raise ValueError('replay_requires_matching_prepared_context')
        if not clock.realtime or clock.origin < self.ready_at:
            raise ValueError('replay_requires_post_admission_wall_aligned_origin')
        self.clock = clock
        # Install the runner's exact budget before evidence resources are borrowed.
        self.analyzer._model = BudgetedTemporalModel(self.model, budget)
        self.entry = create_session_perception_evidence(
            SimpleNamespace(binder=self.canonical, episode_context=None), self.pipeline, clock=now)
        if self.entry is None or self.entry.focus_resolver is None:
            raise RuntimeError('canonical_focus_evidence_composition_unavailable')
        if self.entry.focus_resolver.model is not self.analyzer.temporal_model:
            raise RuntimeError('predicate_model_not_using_replay_budget')
        self.pipeline.register_perception_evidence_session(self.entry)

    async def refresh(self):
        self.snapshot = self.snapshot.model_copy(update={'as_of':now(), 'cursor':
            self.snapshot.cursor.model_copy(update={'revision':self.snapshot.cursor.revision+1})})
        await self.runtime._apply(await self.runtime._read())
        self.briefs.append(self.runtime.bridge.composition.brief)

    async def context(self, event, occurred_at):
        await self.append(event_id=event.event_id, occurred_at=occurred_at, text=event.text,
                          provenance={'context_kind':event.kind, 'offset_ms':event.offset_ms})
        if event.kind == 'goal':
            await self.admit(event.text, occurred_at)
        elif event.kind == 'cancel_goal':
            current = await self.prospective.get_visual_focus(owner_id=self.owner)
            if current:
                await self.prospective.clear_visual_focus(owner_id=self.owner, focus_id=current.id)
        await self.refresh()

    async def build_ambient_context(self, frame):
        active = await self.prospective.get_visual_focus(owner_id=self.owner)
        return VisualAnalyzerContext(owner_id=self.owner, session_id=self.scope.session_id,
            audio_session_id=None, frame_id=frame.id, frame_sequence=frame.frame_sequence,
            captured_at=frame.captured_at, current_utc_time=now(),
            behavior_instructions=(('Synthetic' if self.execution_kind == 'synthetic' else 'Isolated hosted')
                + ' replay evaluation; never claim physical actions or delivery.'),
            recent_audio_transcripts=[], recent_visual_interrupts=[], active_reminders=[],
            previous_working_state={}, active_focus=active.as_prompt_payload() if active else None)

    async def make_frame(self, frame):
        row = await self.connection.fetchrow("""insert into visual_frames
            (id,owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,
             content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
            values($1,$2,$3,$4,$5,'glasses',$6,$7,$8,$9,'fixture',$10,
                   clock_timestamp()+interval '1 hour') returning *""",
            frame.frame_id, self.owner, self.scope.session_id, self.lease.visual_session_id,
            frame.sequence, frame.captured_at, frame.content_type, frame.width, frame.height,
            'evaluation/replay/'+str(frame.frame_id)+'.jpg')
        canonical = VisualFrameRecord(**{name:row[name] for name in VisualFrameRecord.__dataclass_fields__})
        sensory = SensoryVisualFrame(frame_id=canonical.id, owner_id=self.owner,
            session_id=self.scope.session_id, audio_session_id=None,
            visual_session_id=self.lease.visual_session_id, frame_sequence=canonical.frame_sequence,
            source=canonical.source, captured_at=canonical.captured_at,
            server_received_at=canonical.server_received_at, content_type=canonical.content_type,
            width=canonical.width, height=canonical.height, storage_bucket=canonical.storage_bucket,
            storage_path=canonical.storage_path, raw_image_expires_at=canonical.raw_image_expires_at,
            image_bytes=frame.image_bytes)
        if not self.window.put(sensory, stream_lease=self.lease):
            raise RuntimeError('replay_camera_frame_not_admitted')
        self.frames.append(canonical)
        await self.append(event_id='frame:'+str(frame.frame_id), occurred_at=frame.captured_at,
            receipt={'tool_name':'evaluation_replay_capture', 'result':{'status':'captured'}},
            provenance={'frame_id':str(frame.frame_id), 'sequence':frame.sequence,
                'offset_ms':frame.offset_ms, 'sha256':hashlib.sha256(frame.image_bytes).hexdigest(),
                'camera_connection_id':str(self.lease.connection_id),
                'camera_visual_session_id':str(self.lease.visual_session_id)})
        return canonical

    async def payload_state(self, payload_id):
        """Read present storage and its canonical fence; never repair missing bytes."""
        payload_id = UUID(str(payload_id))
        async with self.connection.transaction():
            row = await self.connection.fetchrow('select id,availability,content from jarvis_context_payloads '
                'where owner_id=$1 and id=$2', self.owner, payload_id)
            fence_error = None
            try:
                await self.canonical._lock_roots(self.connection, (payload_id,))
            except SourceError as error:
                if error.code != 'live_state_context_revoked':
                    raise
                fence_error = error.code
        available = row is not None and row['availability']=='available' and fence_error is None
        stored_content = decoded(row['content']) if row is not None else None
        return {'payload_id':str(payload_id), 'payload_present':row is not None,
            'availability':row['availability'] if row is not None else None,
            'stored_content_present':stored_content is not None,
            'source_live':fence_error is None, 'fence_error_code':fence_error,
            'content_state':'available' if available and stored_content is not None else 'unavailable',
            'content':stored_content if available else None}

    async def evidence_rows(self):
        rows = await self.connection.fetch('select e.id,e.payload_id,e.source_namespace,e.source_event_id,e.revision '
            'from jarvis_context_events e '
            'where e.owner_id=$1 and e.session_id=$2 and e.source_namespace=$3 order by e.sequence',
            self.owner, self.scope.session_id, EVIDENCE_NAMESPACE)
        result = []
        for row in rows:
            result.append({'id':str(row['id']), 'source_namespace':row['source_namespace'],
                'source_event_id':row['source_event_id'], 'revision':row['revision'],
                **await self.payload_state(row['payload_id'])})
        return result

    async def join_owner(self):
        """Await the actual scheduled owner; never manufacture a settled result."""
        task = self.entry.focus_resolver._task if self.entry else None
        if task is not None and not task.cancelled():
            try:
                await asyncio.wait_for(asyncio.shield(task), 20)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise

    async def collect(self):
        await self.join_owner()
        c, owner, session = self.connection, self.owner, self.scope.session_id
        async def records(table, condition, *values):
            return [decoded(row['record']) for row in await c.fetch(
                f'select to_jsonb(t) as record from {table} t where owner_id=$1 and {condition}', owner, *values)]
        events = await records('jarvis_prospective_delivery_events', 'session_id=$2', session)
        intents = await records('jarvis_attention_intents', 'preferred_session_id=$2', session)
        observations = await records('perception_observations', 'session_id=$2', session)
        focuses = await records('jarvis_visual_focuses', 'id=any($2::uuid[])', [f.id for f in self.focuses])
        evidence = await self.evidence_rows()
        canonical_sources = []
        for row in await c.fetch('select payload_id from jarvis_observation_sources '
                                 'where owner_id=$1 and session_id=$2', owner, session):
            state = await self.payload_state(row['payload_id'])
            canonical_sources.append({key:value for key,value in state.items() if key!='content'}
                                     | {'live':state['source_live']})
        captured, input_sources, unavailable_evidence = {}, {}, []
        for item in evidence:
            saved = item['content']
            if saved is None:
                unavailable_evidence.append({key:value for key,value in item.items() if key!='content'})
                continue
            input_id = saved['analysis_input_id']
            state = await self.payload_state(input_id)
            captured[input_id] = state['content']
            input_sources[input_id] = {key:value for key,value in state.items() if key!='content'}
        frame_sources = []
        for receipt in self.receipts:
            if receipt.source_event_id.rsplit('/', 1)[-1].startswith('frame:'):
                frame_sources.append({'event_id':str(receipt.id),
                    'source_event_id':receipt.source_event_id, **await self.payload_state(receipt.payload_id)})
        withdrawal = None
        if self.withdrawal is not None:
            target = self.withdrawal
            source = await c.fetchval('select to_jsonb(s) from jarvis_knowledge_sources s '
                'where owner_id=$1 and area_id=$2 and id=$3', owner, self.area, UUID(target['source_id']))
            versions = await c.fetch('select source_id,revision,generation,payload_id '
                'from jarvis_knowledge_source_versions where owner_id=$1 and source_id=$2',
                owner, UUID(target['source_id']))
            withdrawal = {**target, 'source_metadata':decoded(source),
                'source_versions':[{'source_id':str(v['source_id']), 'revision':v['revision'],
                                    'generation':v['generation'], 'payload_id':str(v['payload_id'])}
                                   for v in versions],
                'payload_state':await self.payload_state(target['payload_id'])}
        outcome = self.entry.focus_resolver.last_result
        return {'observations':observations, 'evidence':evidence, 'events':events, 'intents':intents,
            'focuses':focuses, 'canonical_sources':canonical_sources, 'captured_inputs':captured,
            'input_sources':input_sources, 'unavailable_evidence':unavailable_evidence,
            'frame_sources':frame_sources, 'withdrawal':withdrawal,
            'resolver':None if outcome is None else {'status':outcome.status, 'reason':outcome.reason},
            'requests':[{'window_id':str(r.window_id), 'document':request_document(r),
                'brief':r.perception_brief.model_dump(mode='json') if r.perception_brief else None,
                'input_digest':r.perception_brief_input_digest,
                'camera_connection_id':str(r.camera_connection_id)} for r in self.analyzer.requests],
            'brief_versions':[b.version for b in self.briefs], 'speech_attempts':self.speech_attempts,
            'prepared_at':self.prepared_at.isoformat(), 'ready_at':self.ready_at.isoformat(),
            'capture_origin':self.clock.origin.isoformat(), 'fixture_errors':list(self.errors),
            'owner_joined_before_collect':self.entry.focus_resolver._task is None
                or self.entry.focus_resolver._task.done(),
            'last_brief_omission':self.runtime.bridge.last_omission,
            'controlled_model_outputs':self.execution_kind == 'synthetic',
            'execution_kind':self.execution_kind, 'recognition_qualified':False}

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                await self.join_owner()
            finally:
                if self.entry:
                    await self.entry.aclose()
        finally:
            if self.runtime:
                await self.runtime.close()
            if self.scope:
                # Replacing the lease also fences any retained old camera proof.
                self.window.begin_stream(self.owner, self.scope.session_id, uuid4())
