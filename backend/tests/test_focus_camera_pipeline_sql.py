"""Actual camera-window -> analyzer -> canonical evidence -> focus -> attention.

The only substituted inputs are a controlled model and a fixed context-data
builder. Request building, reference parsing, brief composition/binding, camera
lease capture, input publication, observation publication, evidence consumption,
focus predicate/adapter, prospective owner and attention repository all execute.
No consumer return, publication, effect guard, owner decision or intent is mocked.

Storage is actual PostgreSQL in a manifested disposable schema. No public
migration, real provider, image upload, hardware, voice write or service startup.
This qualifies the application path with controlled images/results, not visual
recognition correctness or delivery/playback. Cleanup independently audits a new
connection. It does not rerun the separate v41 schema qualification scenarios.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.attention.repository import AttentionRepository
from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import request_document
from app.knowledge.observations import ObservationPublication, OBSERVATION_NAMESPACE
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import LiveSnapshot, SessionScope, SourceDependency, StateCursor
from app.live_state.perception import CheckLifecycle, ObservationalQuestion
from app.live_state.perception_runtime import SessionPerceptionRuntime
from app.memory.prospective.repository import ProspectiveRepository, _focus_from_record
from app.memory.prospective.service import ProspectiveMemoryService
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.perception.contracts import FastProspectiveMatch
from app.perception.evidence_consumer import EVIDENCE_NAMESPACE, canonical_episode_snapshots
from app.perception.evidence_session import create_session_perception_evidence
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.repository import PerceptionRepository
from app.reminders.duties import ResponsibilityRepository
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_models import AdmitEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest
from app.tasks.repository import TaskRepository
from app.tools.invocation import ToolInvocationContext
from app.visual_memory.context import VisualAnalyzerContext
from app.visual_memory.models import VisualFrameRecord
from app.visual_memory.pipeline import VisualAnalysisPipeline
from app.visual_memory.repository import VisualFrameRepository
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources


def now():
    return datetime.now(UTC)


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def sanitized(error):
    return {'error_type':type(error).__name__, 'code':getattr(error, 'code', None),
        'sqlstate':getattr(error, 'sqlstate', None),
        'frames':[{'file':Path(frame.filename).name, 'line':frame.lineno, 'function':frame.name}
                  for frame in traceback.extract_tb(error.__traceback__)[-5:]]}


class ControlledModel:
    provider = 'fixture-no-provider'
    model = 'controlled-focus-camera'

    def __init__(self, pool, *, visibility='visible', uncertain=False, legacy=False):
        self.pool, self.visibility, self.uncertain, self.legacy = pool, visibility, uncertain, legacy
        self.analysis_entered, self.analysis_release = asyncio.Event(), asyncio.Event()
        self.predicate_entered, self.predicate_release = asyncio.Event(), asyncio.Event()
        self.analysis_release.set()
        self.calls, self.cancelled = [], False

    async def generate_temporal_json(self, *, prompt, images):
        predicate = prompt.startswith('Evaluate whether these original images')
        self.calls.append('predicate' if predicate else 'analysis')
        assert images and all(image.image_bytes for image in images)
        # One-connection pool: this would block if a model call retained a DB lease.
        async with self.pool.acquire() as connection:
            assert not connection.is_in_transaction()
            assert await connection.fetchval('select 1') == 1
        entered = self.predicate_entered if predicate else self.analysis_entered
        release = self.predicate_release if predicate else self.analysis_release
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if predicate:
            return json.dumps({'verdict':'uncertain' if self.uncertain else 'satisfied',
                'basis':'insufficient_evidence' if self.uncertain else 'condition_observed',
                'observed_condition':'The explicitly described synthetic red key ring is visible.',
                'confidence':0.99, 'uncertainties':['Identity is ambiguous.'] if self.uncertain else [],
                'supporting_frame_ids':[f'frame:{index}' for index in range(len(images))]})
        context = json.loads(prompt.split('Context JSON:\n```json\n')[1].split('\n```')[0])
        question, = context['perception_assignment']['questions']
        frames = [f'frame:{index}' for index in range(len(images))]
        output = {'summary':'A synthetic red key ring is on the table.', 'starts_new_event':False,
            'confidence':0.99, 'perception_findings':[{
                'question_id':question['id'], 'check_id':question['check_id'],
                'visibility':self.visibility, 'literal_finding':'A synthetic red key ring is on the table.',
                'confidence':0.99, 'evidence_frame_ids':frames}]}
        if self.legacy:
            output['prospective_matches'] = [{'subject_type':'visual_focus', 'subject_id':'focus:0',
                'confidence':0.99, 'observed_condition':'The synthetic red key ring is visible.',
                'supporting_frame_ids':frames, 'suggested_message':'The requested key ring is on the table.'}]
        return json.dumps(output)


class ObservedAnalyzer(FastPerceptionAnalyzer):
    """Telemetry only: the production mapper, prompt and parser still run."""
    def __init__(self, model):
        super().__init__(model)
        self.requests, self.outputs = [], []

    async def analyze(self, request):
        self.requests.append(request)
        output = await super().analyze(request)
        self.outputs.append(output)
        return output


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
                     os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'Explicit manifested database required')
class FocusCameraPipelineSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_camera_focus_pipeline(self):
        dsn = os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL']
        directory = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f'focus-camera-pipeline-{fixture.run_id}.json'
        report = dict(format='jarvis.focus_camera_pipeline.sql.v1', run_id=fixture.run_id,
            status='running', checks=[], cleanup=None, public_migration=False, provider_calls=False,
            camera='actual_in_process_lease_and_window', model='controlled_results', storage='PostgreSQL',
            consumer_return_mocked=False, actual_speech=False, recognition_qualified=False)
        connection = pool = None
        scenarios, failures = [], []

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2)
                handle.write('\n')

        async def check(name, operation):
            try:
                async with asyncio.timeout(90):
                    evidence = await operation()
            except Exception as error:
                entry = dict(name=name, status='failed', **sanitized(error))
                failures.append(entry)
                if isinstance(error, (OSError, asyncpg.ConnectionDoesNotExistError,
                                      asyncpg.CannotConnectNowError)) and not isinstance(error, TimeoutError):
                    report['checks'].append(entry)
                    save()
                    raise
            else:
                entry = dict(name=name, status='passed', evidence=evidence)
            finally:
                for scenario in scenarios:
                    if not scenario.closed:
                        scenario.model.analysis_release.set()
                        scenario.model.predicate_release.set()
                        await scenario.pipeline.close()
                        await scenario.entry.aclose()
                        await scenario.runtime.close()
                        scenario.closed = True
            report['checks'].append(entry)
            save()

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(item for item in load_migrations() if item.version <= 41)
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                                                   migrations=migrations), timeout=240)
            path_sql = f'set search_path to {quote_schema(schema)},pg_catalog,extensions'
            await connection.execute(path_sql)
            await connection.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                            command_timeout=25, setup=setup)
            owner = fixture.owners[0]
            area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Focus camera integration fixture','test') returning id", owner)
            sources = SourceRepository(pool)
            tasks = TaskRepository(pool, TaskLimits())
            context = ContextRepository(tasks, publication=CanonicalSourcePublication(sources, area))
            publication = ObservationPublication(context, owner_id=owner)
            prospective_repository = ProspectiveRepository(pool)
            attention = AttentionRepository(tasks)
            manager = SimpleNamespace(attention_notifications_enabled=True, attention_repository=attention,
                speak_prospective_interrupt=AsyncMock(side_effect=AssertionError('Direct speech is forbidden')))
            settings = SimpleNamespace(VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS=15,
                VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES=4, VISUAL_SENSORY_WINDOW_TTL_SECONDS=30,
                VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=0.9,
                PROSPECTIVE_FIND_ONCE_DEFAULT_TTL_SECONDS=300, PROSPECTIVE_MONITOR_DEFAULT_TTL_SECONDS=300,
                PROSPECTIVE_MAX_CANDIDATES_PER_ANALYSIS=4, PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS=10,
                PROSPECTIVE_TRIGGER_CLAIM_LEASE_SECONDS=30, VISUAL_INTERRUPT_DEDUPE_SECONDS=30)
            prospective = ProspectiveMemoryService(repository=prospective_repository,
                realtime_session_manager=manager, settings=settings)

            async def admit(mode='find_once'):
                return await prospective.set_visual_focus(owner_id=owner, target='synthetic red key ring',
                    description='A small synthetic silver key on a red ring', purpose='Find the requested target',
                    mode=mode, expires_in_seconds=300)

            async def scenario(*, mode='find_once', visibility='visible', uncertain=False, legacy=False,
                               pause_analysis=False):
                session = await connection.fetchval('insert into jarvis_sessions(owner_id) values($1) returning id', owner)
                scope = SessionScope(owner_id=owner, session_id=session, area_id=area)
                focus = await admit(mode)
                receipt = await context.append(owner, ContextIngress(session_id=session,
                    source_namespace='qualification:focus-camera', source_event_id=str(focus.id),
                    receipt_id=str(uuid4()), kind='tool_outcome', trust='tool_receipt',
                    payload={'tool_name':'set_visual_focus', 'result':{
                        'status':'success', 'visual_focus':focus.as_prompt_payload()}}))
                canonical = CanonicalContextBinder(sources, scope, clock=now, timeout_seconds=18)
                snapshot = LiveSnapshot(scope=scope, cursor=StateCursor(epoch=uuid4(), revision=1),
                    as_of=now(), readiness='ready')
                runtime = SessionPerceptionRuntime(service=SimpleNamespace(snapshot=lambda:snapshot),
                    canonical=canonical, bus=DomainEventBus(), clock=now, enabled=True)
                await runtime._apply(await runtime._read())
                brief = runtime.bridge.composition.brief
                self.assertEqual(len(brief.questions), 1, runtime.focus_omission)
                self.assertEqual(brief.questions[0].check_ref.source_id, focus.id)
                self.assertEqual(brief.questions[0].check_ref.source_namespace, 'visual_focus')
                window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=30, now=now)
                lease = window.begin_stream(owner, session, uuid4())
                records, images = [], []
                for sequence in (1,2):
                    row = await connection.fetchrow("""insert into visual_frames
                        (owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,
                         content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
                        values($1,$2,$3,$4,'glasses',$5,'image/jpeg',32,32,'fixture','not-uploaded',
                               clock_timestamp()+interval '1 hour') returning *""",
                        owner, session, lease.visual_session_id, sequence, now())
                    frame = VisualFrameRecord(**{field:row[field] for field in VisualFrameRecord.__dataclass_fields__})
                    image = SensoryVisualFrame(frame_id=frame.id, owner_id=owner, session_id=session,
                        audio_session_id=None, visual_session_id=lease.visual_session_id,
                        frame_sequence=sequence, source=frame.source, captured_at=frame.captured_at,
                        server_received_at=frame.server_received_at, content_type=frame.content_type,
                        width=frame.width, height=frame.height, storage_bucket=frame.storage_bucket,
                        storage_path=frame.storage_path, raw_image_expires_at=frame.raw_image_expires_at,
                        image_bytes=b'controlled-model-image-'+str(sequence).encode('ascii'))
                    self.assertTrue(window.put(image, stream_lease=lease))
                    records.append(frame)
                    images.append(image)
                model = ControlledModel(pool, visibility=visibility, uncertain=uncertain, legacy=legacy)
                if pause_analysis:
                    model.analysis_release.clear()
                analyzer = ObservedAnalyzer(model)
                limiter = PriorityAsyncLimiter(1)
                interrupt = SimpleNamespace(maybe_interrupt=AsyncMock(side_effect=AssertionError('No generic speech')))
                pipeline = VisualAnalysisPipeline(settings=settings, repository=VisualFrameRepository(pool),
                    session_repository=object(), behavior_repository=object(), interrupt_service=interrupt,
                    model_limiter=limiter, fast_analyzer=analyzer,
                    perception_repository=PerceptionRepository(pool, observation_publication=publication),
                    sensory_frame_window=window, prospective_service=prospective,
                    live_brief_bridge_for=lambda requested_owner, requested_session: runtime.bridge
                        if (requested_owner, requested_session) == (owner, session) else None)
                pipeline.set_observation_publication(publication)

                async def build_context(frame):
                    # Fixed ambient input boundary, not an evidence/owner shortcut:
                    # actual canonical history, capture and source checks still run.
                    active = await prospective.get_visual_focus(owner_id=owner)
                    return VisualAnalyzerContext(owner_id=owner, session_id=session, audio_session_id=None,
                        frame_id=frame.id, frame_sequence=frame.frame_sequence, captured_at=frame.captured_at,
                        current_utc_time=now(), behavior_instructions='Observe without claiming physical actions.',
                        recent_audio_transcripts=[], recent_visual_interrupts=[], active_reminders=[],
                        previous_working_state={}, active_focus=active.as_prompt_payload() if active else None)

                pipeline._context_builder.build = build_context
                entry = create_session_perception_evidence(
                    SimpleNamespace(binder=canonical, episode_context=None), pipeline, clock=now)
                self.assertIsNotNone(entry)
                self.assertIs(entry.focus_resolver.model, model)
                self.assertIs(entry.focus_resolver.limiter, limiter)
                self.assertIs(entry.focus_resolver.prospective, prospective)
                pipeline.register_perception_evidence_session(entry)
                s = SimpleNamespace(session=session, scope=scope, focus=focus, receipt=receipt, canonical=canonical,
                    runtime=runtime, brief=brief, window=window, lease=lease, frames=records, images=images,
                    model=model, analyzer=analyzer, pipeline=pipeline, entry=entry, diagnostics=[], closed=False,
                    interrupt=interrupt)
                scenarios.append(s)
                return s

            async def run(s, *, drain=True):
                async def diagnostic(value):
                    s.diagnostics.append(value)
                await s.pipeline.enqueue(frame=s.frames[-1], image_bytes=s.images[-1].image_bytes,
                                         send_diagnostic=diagnostic)
                if drain:
                    await asyncio.wait_for(s.pipeline.drain(), 20)
                    self.assertFalse(any(item['type']=='visual.frame.analysis_failed' for item in s.diagnostics),
                                     s.diagnostics)

            async def evidence_rows(s):
                return await connection.fetch("""select e.id,e.payload_id,p.content from jarvis_context_events e
                    join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                    where e.owner_id=$1 and e.session_id=$2 and e.source_namespace=$3 order by e.sequence""",
                    owner, s.session, EVIDENCE_NAMESPACE)

            async def focus_row(s):
                return dict(await connection.fetchrow('select * from jarvis_visual_focuses where owner_id=$1 and id=$2',
                                                     owner, s.focus.id))

            async def counts(s):
                return dict(await connection.fetchrow("""select
                    (select count(*) from perception_observations where owner_id=$1 and session_id=$2) as observations,
                    (select count(*) from jarvis_attention_intents where owner_id=$1 and preferred_session_id=$2) as intents,
                    (select count(*) from jarvis_prospective_delivery_events where owner_id=$1 and session_id=$2) as events""",
                    owner, s.session))

            async def recorded(s):
                self.assertEqual(len(s.analyzer.requests), 1)
                request = s.analyzer.requests[0]
                self.assertEqual(request.camera_connection_id, s.lease.connection_id)
                self.assertEqual(request.camera_visual_session_id, s.lease.visual_session_id)
                self.assertEqual(request.perception_brief, s.brief)
                self.assertEqual(request.perception_brief_input_digest, digest(request_document(request)))
                self.assertEqual(len(s.analyzer.outputs[0].perception_findings), 1)
                rows = await evidence_rows(s)
                self.assertEqual(len(rows), 1, s.diagnostics)
                saved = decoded(rows[0]['content'])
                self.assertEqual(saved['evidence']['brief'], s.brief.model_dump(mode='json'))
                self.assertEqual(saved['evidence']['window_id'], str(request.window_id))
                self.assertEqual(saved['camera_connection_id'], str(s.lease.connection_id))
                self.assertEqual(saved['request_digest'], request.perception_brief_input_digest)
                self.assertEqual(saved['evidence']['findings'],
                                 [finding.model_dump(mode='json') for finding in s.analyzer.outputs[0].perception_findings])
                source = await connection.fetchrow("""select e.source_namespace,e.source_event_id,p.content
                    from jarvis_context_events e join jarvis_context_payloads p
                    on p.owner_id=e.owner_id and p.id=e.payload_id where e.owner_id=$1 and e.id=$2""",
                    owner, saved['original_source']['source_id'])
                self.assertEqual(source['source_namespace'], OBSERVATION_NAMESPACE)
                original = decoded(source['content'])
                self.assertEqual(original['analysis_input_id'], saved['analysis_input_id'])
                captured = decoded(await connection.fetchval('select content from jarvis_context_payloads '
                    'where owner_id=$1 and id=$2', owner, saved['analysis_input_id']))
                self.assertEqual(captured['request'], request_document(request))
                observed_frames = tuple((item['frame_id'], item['frame_sequence'],
                    datetime.fromisoformat(item['captured_at'])) for item in saved['evidence']['frames'])
                expected_frames = tuple((str(frame.frame_id), frame.frame_sequence, frame.captured_at)
                                        for frame in request.frames)
                # Equivalent UTC spellings (Z and +00:00) describe the same
                # instant. Preserve exact identities, sequence and frame order.
                self.assertEqual(observed_frames, expected_frames)
                chronology = tuple((captured_at, sequence) for _, sequence, captured_at in observed_frames)
                self.assertEqual(chronology, tuple(sorted(chronology)))
                self.assertTrue(any(item.get('type')=='visual.perception_evidence' and item['status']=='recorded'
                                    for item in s.diagnostics))
                return rows[0]

            async def finish(s):
                s.model.predicate_release.set()
                task = s.entry.focus_resolver._task
                self.assertIsNotNone(task, s.diagnostics)
                await asyncio.wait_for(task, 15)
                return s.entry.focus_resolver.last_result

            async def no_effect(s, *, historical=True):
                row = await focus_row(s)
                self.assertIsNone(row['check_resolution'])
                self.assertIsNone(row['completed_at'])
                self.assertIsNone(row['completion_evidence'])
                values = await counts(s)
                self.assertEqual(values['intents'], 0)
                self.assertEqual(values['events'], 0)
                if historical:
                    self.assertEqual(values['observations'], 1)
                manager.speak_prospective_interrupt.assert_not_awaited()
                s.interrupt.maybe_interrupt.assert_not_awaited()

            async def positive(mode):
                s = await scenario(mode=mode)
                await run(s)
                await recorded(s)
                self.assertTrue(s.entry.last_focus_resolution_scheduled)
                await asyncio.wait_for(s.model.predicate_entered.wait(), 5)
                await no_effect(s)
                outcome = await finish(s)
                self.assertEqual(outcome.status, 'owner_processed', outcome.reason)
                row = _focus_from_record(await focus_row(s))
                self.assertEqual(row.revision, 2)
                self.assertIsNotNone(row.check_resolution)
                self.assertEqual(row.check_resolution.applicable_episode_or_object_ref,
                                 s.brief.questions[0].episode_or_object_ref)
                self.assertIsNone(row.completed_at)
                self.assertIsNone(row.completion_evidence)
                self.assertEqual(row.as_check(s.scope, now=now()).lifecycle,
                                 CheckLifecycle.TRACKING if mode=='monitor' else CheckLifecycle.RESOLVED)
                self.assertEqual(await counts(s), {'observations':1,'intents':1,'events':1})
                intent = await connection.fetchrow('select state,attempt_count from jarvis_attention_intents '
                    'where owner_id=$1 and preferred_session_id=$2', owner, s.session)
                self.assertEqual((intent['state'], intent['attempt_count']), ('pending',0))
                event = await connection.fetchrow('select event_type,metadata from jarvis_prospective_delivery_events '
                    'where owner_id=$1 and session_id=$2', owner, s.session)
                self.assertEqual(event['event_type'], 'generated')
                self.assertEqual(decoded(event['metadata'])['physical_action_status'], 'not_established')
                self.assertEqual(decoded(event['metadata'])['playback_status'], 'unconfirmed')
                manager.speak_prospective_interrupt.assert_not_awaited()
                self.assertEqual(s.model.calls, ['analysis','predicate'])
                return {'actual_chain':'admission/brief/camera/capture/analyzer/publication/evidence/predicate/owner/intent',
                        'mode':mode, 'intent_state':'pending', 'delivery_attempts':0, 'physical_completion':False}

            async def mutate(s, operation):
                if operation=='replace':
                    return await admit()
                if operation=='cancel':
                    return await prospective.clear_visual_focus(owner_id=owner, focus_id=s.focus.id)
                replacement = s.window.begin_stream(owner, s.session, s.lease.visual_session_id)
                self.assertNotEqual(replacement.connection_id, s.lease.connection_id)
                return replacement

            async def pending_predicate(operation):
                s = await scenario()
                await run(s)
                await recorded(s)
                await asyncio.wait_for(s.model.predicate_entered.wait(), 5)
                await mutate(s, operation)
                outcome = await finish(s)
                self.assertEqual(outcome.status, 'not_applied', outcome.reason)
                await no_effect(s)
                self.assertEqual(len(await evidence_rows(s)), 1)
                return {'recorded_before_change':True, 'late_resolution_rejected':True,
                        'historical_evidence_retained':True, 'intents':0}

            async def pending_analysis(operation):
                s = await scenario(pause_analysis=True)
                await run(s, drain=False)
                await asyncio.wait_for(s.model.analysis_entered.wait(), 5)
                await mutate(s, operation)
                s.model.analysis_release.set()
                await asyncio.wait_for(s.pipeline.drain(), 20)
                self.assertFalse(any(item['type']=='visual.frame.analysis_failed' for item in s.diagnostics), s.diagnostics)
                await no_effect(s)
                self.assertEqual(await evidence_rows(s), [])
                self.assertEqual(s.model.calls, ['analysis'])
                self.assertIsNone(s.entry.focus_resolver._task)
                return {'original_observation_retained':True, 'typed_evidence_not_admitted':True,
                        'predicate_calls':0, 'intents':0}

            async def uncertainty(visibility, predicate_uncertain=False):
                s = await scenario(visibility=visibility, uncertain=predicate_uncertain)
                await run(s)
                await recorded(s)
                outcome = await finish(s)
                self.assertEqual(outcome.status, 'observed_without_resolution', outcome.reason)
                await no_effect(s)
                self.assertEqual(s.model.calls.count('predicate'), 1 if predicate_uncertain else 0)
                return {'visibility':visibility, 'predicate_uncertain':predicate_uncertain,
                        'recorded_without_resolution':True, 'intents':0}

            async def duplicates(legacy_first):
                s = await scenario(legacy=legacy_first)
                await run(s)
                await recorded(s)
                request = s.analyzer.requests[0]
                skipped_redundant_predicate = False
                if legacy_first:
                    self.assertEqual(len(s.analyzer.outputs[0].prospective_matches), 1)
                    self.assertEqual((await counts(s))['intents'], 1)
                    # The existing route can reserve the watch while the typed
                    # evaluator waits for the limiter/owner. A finished refusal
                    # is not a missing model callback. Keep the original 5s bound
                    # and accept it ONLY with the exact persisted effect below.
                    task = s.entry.focus_resolver._task
                    self.assertIsNotNone(task)
                    entered = asyncio.create_task(s.model.predicate_entered.wait())
                    try:
                        done, _ = await asyncio.wait((task, entered), timeout=5,
                                                     return_when=asyncio.FIRST_COMPLETED)
                        self.assertTrue(done, 'Neither predicate entry nor resolver completion occurred')
                        if not s.model.predicate_entered.is_set():
                            self.assertIn(task, done)
                            await task
                            outcome = s.entry.focus_resolver.last_result
                            self.assertIsNotNone(outcome)
                            self.assertEqual((outcome.status, outcome.reason),
                                             ('not_applied', 'focus_predicate_changed'))
                            self.assertEqual(s.model.calls, ['analysis'])
                            skipped_redundant_predicate = True
                        else:
                            outcome = await finish(s)
                            self.assertEqual(outcome.status, 'owner_processed', outcome.reason)
                    finally:
                        if not entered.done():
                            entered.cancel()
                        await asyncio.gather(entered, return_exceptions=True)
                else:
                    await asyncio.wait_for(s.model.predicate_entered.wait(), 5)
                    outcome = await finish(s)
                    self.assertEqual(outcome.status, 'owner_processed', outcome.reason)
                original = await connection.fetchval('select payload_id from jarvis_observation_sources '
                    'where owner_id=$1 and session_id=$2', owner, s.session)
                if not legacy_first:
                    guard = s.runtime.bridge.effect_guard_for(request, original)
                    await s.pipeline._process_prospective_matches_best_effort(request=request,
                        matches=[FastProspectiveMatch(subject_type='visual_focus', subject_id=s.focus.id,
                            confidence=0.99, observed_condition='Synthetic ring visible.',
                            supporting_frame_ids=[frame.frame_id for frame in request.frames],
                            suggested_message='The ring is visible.')], frame_id=s.frames[-1].id,
                        source_payload_id=original, effect_guard=guard)
                self.assertEqual(await counts(s), {'observations':1,'intents':1,'events':1})
                row = await focus_row(s)
                question, = request.perception_brief.questions
                self.assertEqual(row['episode_revision'], question.episode_or_object_ref.revision)
                self.assertIsNone(row['completed_at'])
                self.assertIsNone(row['cleared_at'])
                self.assertIsNone(row['completion_evidence'])
                if skipped_redundant_predicate:
                    self.assertEqual(row['revision'], question.check_ref.revision)
                    self.assertIsNone(row['check_resolution'], 'No inference means no invented typed resolution')
                else:
                    self.assertEqual(row['revision'], question.check_ref.revision + 1)
                    self.assertIsNotNone(row['check_resolution'])
                self.assertTrue(s.entry.camera.is_current(request, s.lease.visual_session_id))
                self.assertLess(now(), request.perception_brief.expires_at)
                subject = await connection.fetchrow("select generation,state,claim_token from jarvis_attention_visual_subjects "
                    "where owner_id=$1 and subject_type='visual_focus' and subject_id=$2", owner, s.focus.id)
                self.assertEqual(subject['generation'], question.focus_ref.generation)
                self.assertEqual(subject['state'], 'active')
                self.assertIsNotNone(row['trigger_claim_token'])
                self.assertEqual(subject['claim_token'], row['trigger_claim_token'])
                event = await connection.fetchrow('select id,visual_focus_id,claim_token,event_type,evidence,metadata '
                    'from jarvis_prospective_delivery_events where owner_id=$1 and session_id=$2', owner, s.session)
                self.assertEqual(event['visual_focus_id'], s.focus.id)
                self.assertEqual(event['claim_token'], row['trigger_claim_token'])
                self.assertEqual(event['event_type'], 'generated')
                event_evidence, metadata = decoded(event['evidence']), decoded(event['metadata'])
                self.assertEqual(event_evidence['source_payload_id'], str(original))
                self.assertEqual(event_evidence['supporting_frame_ids'], [str(frame.frame_id) for frame in request.frames])
                self.assertEqual(datetime.fromisoformat(event_evidence['requested_at']), request.requested_at)
                self.assertTrue(metadata['attention_owned'])
                self.assertEqual(metadata['delivery_route'], 'attention')
                self.assertEqual(metadata['physical_action_status'], 'not_established')
                self.assertEqual(metadata['playback_status'], 'unconfirmed')
                intent = await connection.fetchrow('select id,source_id,source_payload_id,state,attempt_count '
                    'from jarvis_attention_intents where owner_id=$1 and preferred_session_id=$2', owner, s.session)
                self.assertEqual(intent['source_id'], event['id'])
                self.assertEqual(metadata['attention_intent_id'], str(intent['id']))
                self.assertEqual((intent['state'], intent['attempt_count']), ('pending',0))
                async with connection.transaction():
                    self.assertTrue(await connection.fetchval('select lock_jarvis_copy_sources($1,$2::uuid[])',
                        owner, [original, intent['source_payload_id']]))
                manager.speak_prospective_interrupt.assert_not_awaited()
                return {'order':'legacy_then_typed' if legacy_first else 'typed_then_legacy',
                        'events':1, 'intents':1, 'actual_speech':False, 'delivery_attempts':0,
                        'predicate_calls':s.model.calls.count('predicate'),
                        'resolver_result':{'status':outcome.status, 'reason':outcome.reason},
                        'resolver_task_done':s.entry.focus_resolver._task.done(),
                        'resolver_task_cancelled':s.entry.focus_resolver._task.cancelled(),
                        'camera_current':True, 'brief_version':request.perception_brief.version,
                        'assigned_check_revision':question.check_ref.revision,
                        'owner_check_revision':row['revision'], 'episode_revision':row['episode_revision'],
                        'subject_generation':subject['generation'],
                        'typed_resolution_written':row['check_resolution'] is not None,
                        'early_refusal_preserved_existing_effect':skipped_redundant_predicate}

            async def withdrawn_source():
                s = await scenario()
                await run(s)
                evidence = await recorded(s)
                await asyncio.wait_for(s.model.predicate_entered.wait(), 5)
                identity = await connection.fetchval('select source_id from jarvis_knowledge_source_versions '
                    'where owner_id=$1 and payload_id=$2', owner, evidence['payload_id'])
                self.assertIsNotNone(identity)
                await sources.delete_source(owner, area, identity, expected_revision=1)
                self.assertEqual((await finish(s)).status, 'not_applied')
                await no_effect(s)
                return {'real_source_withdrawal':True, 'late_resolution_rejected':True}

            async def retire_pending():
                s = await scenario()
                await run(s)
                await recorded(s)
                await asyncio.wait_for(s.model.predicate_entered.wait(), 5)
                await asyncio.wait_for(s.pipeline.close(), 10)
                self.assertTrue(s.entry._closed)
                self.assertTrue(s.entry.focus_resolver._task.done())
                self.assertTrue(s.model.cancelled)
                await no_effect(s)
                return {'session_retired':True, 'predicate_task_joined':True, 'intents':0}

            async def mixed_snapshot_partition():
                s = await scenario()
                episode_owner = EpisodeCheckOwner(ResponsibilityRepository(pool), enabled=True, clock=now)
                await episode_owner.initialize()
                source = await context.append(owner, ContextIngress(session_id=s.session,
                    source_namespace='qualification:mixed-focus', source_event_id=str(uuid4()), receipt_id=str(uuid4()),
                    kind='user_transcript', trust='user_input', source_time=now(), payload={'text':'Check the synthetic display.'}))
                admitted = await episode_owner.admit(s.scope, ToolInvocationContext(
                    request_event_id=source.id, request_revision=source.revision), AdmitEpisodeCheck(
                    spec=EpisodeCheckSpec(title='Synthetic episode'),
                    check=EpisodeCheckDraft(question='What does the synthetic display indicate?')))
                s.canonical.episode_context = SimpleNamespace(require_owner=lambda:episode_owner, limit=8)
                refs = (admitted.outcome.canonical_owner_ref, admitted.outcome.check_ref,
                        admitted.check.episode_or_object_ref)
                question = ObservationalQuestion(id=uuid4(), question=admitted.check.question,
                    check_ref=admitted.outcome.check_ref, episode_or_object_ref=admitted.check.episode_or_object_ref)
                brief = s.brief.model_copy(update={'id':uuid4(), 'questions':(*s.brief.questions,question),
                                                  'dependencies':(*s.brief.dependencies,*refs)})
                helper = canonical_episode_snapshots(s.canonical)
                async with s.canonical._transaction() as c:
                    snapshots = await helper(c, brief)
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(snapshots[0].check.id, admitted.check.id)
                foreign = SourceDependency(source_namespace='episode', source_id=uuid4(), revision=1)
                with self.assertRaises(SourceError):
                    async with s.canonical._transaction() as c:
                        await helper(c, brief.model_copy(update={'dependencies':(*brief.dependencies,foreign)}))
                await prospective.clear_visual_focus(owner_id=owner, focus_id=s.focus.id)
                with self.assertRaises(SourceError):
                    async with s.canonical._transaction() as c:
                        await helper(c, brief)
                return {'real_responsibility_owner_validated':True, 'only_exact_focus_group_partitioned':True,
                        'orphan_episode_rejected':True, 'cancelled_focus_rejected':True,
                        'scope':'snapshot_callback_not_a_second_camera_case'}

            for mode in ('find_once','monitor'):
                await check('actual_camera_to_focus_intent_'+mode, lambda mode=mode:positive(mode))
            for operation in ('replace','cancel','camera_switch'):
                await check('pending_predicate_'+operation, lambda operation=operation:pending_predicate(operation))
                await check('pending_analysis_'+operation, lambda operation=operation:pending_analysis(operation))
            for visibility in ('unclear','not_visible'):
                await check('uncertain_finding_'+visibility, lambda visibility=visibility:uncertainty(visibility))
            await check('uncertain_predicate', lambda:uncertainty('visible', True))
            await check('legacy_then_typed_duplicate', lambda:duplicates(True))
            await check('typed_then_legacy_duplicate', lambda:duplicates(False))
            await check('source_withdrawal_during_predicate', withdrawn_source)
            await check('session_retirement_joins_predicate', retire_pending)
            await check('mixed_owner_snapshot_partition', mixed_snapshot_partition)
            report['status'] = 'failed' if failures else 'passed'
            save()
        except Exception as error:
            report.update(status='failed', fatal=sanitized(error))
            save()
            raise AssertionError('Focus camera pipeline qualification stopped; see sanitized report') from None
        finally:
            try:
                for s in scenarios:
                    if not s.closed:
                        s.model.analysis_release.set()
                        s.model.predicate_release.set()
                        await s.pipeline.close()
                        await s.entry.aclose()
                        await s.runtime.close()
                        s.closed = True
            finally:
                try:
                    if fixture.manifest['schemas']:
                        await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
                    else:
                        if connection is not None:
                            await connection.close(timeout=5)
                        report['cleanup'] = {'status':'not_required_no_resources_allocated'}
                finally:
                    save()
                    print(json.dumps({'report':str(report_path), 'status':report['status'],
                        'checks':report['checks'], 'cleanup':report['cleanup'],
                        **({'fatal':report['fatal']} if 'fatal' in report else {})}))
        self.assertFalse(failures, json.dumps(failures))


if __name__ == '__main__':
    unittest.main()
