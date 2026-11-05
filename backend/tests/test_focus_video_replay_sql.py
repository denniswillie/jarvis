"""Replay runner -> real camera/pipeline/typed owner/attention, controlled outputs.

Actual SQL and independent teardown, no provider, hardware, public migration or
speech. Expected labels never reach the application port. A barrier controls only
model timing; port callbacks never fabricate observations, decisions or intents.
"""
import asyncio
import base64
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import UUID

import asyncpg

from app.knowledge.observations import OBSERVATION_NAMESPACE
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.tasks.models import digest
from evals.responsive_voice.focus_pipeline_replay import FocusPipelineReplay, decoded
from evals.responsive_voice.video_replay import (
    ContextEvent, FrameAsset, PermissionRecord, PipelineReplayPort, ReplayCase, ReplayInputs, run_replay)
from evals.second_brain.fixtures import DatabaseFixture
from test_focus_camera_pipeline_sql import ControlledModel, sanitized
from test_notification_ledger_database import cleanup_fixture_resources


# A local one-pixel JPEG fixture, not a user image or evidence of model quality.
JPEG = base64.b64decode(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////'
    '////////////////////////////////////////////2wBDAf//////////////////////////'
    '//////////////////////////////////////////////////////////wAARCAABAAEDASIA'
    'AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADA'
    'MBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAA'
    'AAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QA'
    'FBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgB'
    'AQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAU'
    'EQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAA'
    'E/EB//2Q==')


class ReplayControlledModel(ControlledModel):
    """Use original controlled predicate/parser inputs; tolerate a cleared goal."""
    async def generate_temporal_json(self, *, prompt, images):
        if not prompt.startswith('Evaluate whether these original images'):
            context = json.loads(prompt.split('Context JSON:\n```json\n')[1].split('\n```')[0])
            if not context.get('perception_assignment', {}).get('questions'):
                self.calls.append('analysis')
                self.analysis_entered.set()
                await self.analysis_release.wait()
                return json.dumps({'summary':'Synthetic replay with no active visual question.',
                    'starts_new_event':False, 'confidence':0.99, 'perception_findings':[]})
        output = json.loads(await super().generate_temporal_json(prompt=prompt, images=images))
        if prompt.startswith('Evaluate whether these original images'):
            output['observed_condition'] = 'Controlled fixture predicate: the requested target is visible.'
        else:
            # Preserve the whole original window, but do not claim that images
            # preceding a replacement goal support that new goal's match.
            for finding in output.get('perception_findings', []):
                finding['evidence_frame_ids'] = [f'frame:{len(images)-1}']
                finding['literal_finding'] = 'The controlled target is visible in the latest replay frame.'
        return json.dumps(output)


class BarrierReplayPort(PipelineReplayPort):
    """Telemetry and controlled timing only; every port operation calls production."""
    def __init__(self, *args, fixture, boundary='analysis', mutation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixture, self.boundary, self.mutation = fixture, boundary, mutation
        self.enqueues, self.drain_calls, self.pending_enqueues = [], 0, []
        self.before_release_task = None
        self.stale_outcome = None

    async def enqueue(self, frame):
        await super().enqueue(frame)
        self.enqueues.append(frame.sequence)
        model = self.fixture.model
        if len(self.enqueues) == 1:
            entered = model.analysis_entered if self.boundary == 'analysis' else model.predicate_entered
            await asyncio.wait_for(entered.wait(), 5)
            if self.boundary == 'predicate':
                self.before_release_task = self.fixture.entry.focus_resolver._task
                request = self.fixture.analyzer.requests[0]
                resolver = self.fixture.entry.focus_resolver
                def capture_old_result(task):
                    same_task = resolver._task is task
                    outcome = resolver.last_result if same_task else None
                    self.stale_outcome = {'status':outcome.status if outcome else None,
                        'reason':outcome.reason if outcome else None,
                        'window_id':str(request.window_id),
                        'camera_connection_id':str(request.camera_connection_id),
                        'task_done':task.done(), 'task_cancelled':task.cancelled(),
                        'same_task':same_task}
                self.before_release_task.add_done_callback(capture_old_result)
                # Later images deliberately do not authorize a fresh match.
                model.visibility = 'unclear'
        else:
            release = model.analysis_release if self.boundary == 'analysis' else model.predicate_release
            if not release.is_set():
                self.pending_enqueues.append(frame.sequence)

    async def context(self, event, occurred_at):
        await super().context(event, occurred_at)
        if self.mutation:
            await self.mutation(self.fixture)

    async def drain(self):
        self.drain_calls += 1
        assert self.enqueues == [1,2,3,4], 'drain occurred before all replay frames'
        self.fixture.model.analysis_release.set()
        self.fixture.model.predicate_release.set()
        if self.before_release_task is not None:
            await asyncio.wait_for(asyncio.shield(self.before_release_task), 10)
            await asyncio.sleep(0)
            assert self.stale_outcome is not None, 'Original owner task result was not captured'
        await super().drain()

    async def close(self):
        self.fixture.model.analysis_release.set()
        self.fixture.model.predicate_release.set()
        try:
            await self.fixture.join_owner()
        finally:
            await super().close()


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
                     os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'Explicit manifested database required')
class FocusVideoReplaySQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_replay_composition(self):
        dsn = os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL']
        directory = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'])
        database = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / ('focus-video-replay-'+database.run_id+'.json')
        report = {'format':'jarvis.focus_video_replay.sql.v1', 'run_id':database.run_id,
            'status':'running', 'checks':[], 'cleanup':None, 'public_migration':False,
            'provider_calls':False, 'controlled_model_outputs':True, 'recognition_qualified':False}
        connection = pool = None
        failures = []

        def save():
            fd = os.open(report_path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await database.create_schema(connection)
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                migrations=tuple(m for m in load_migrations() if m.version <= 41)), 240)
            path_sql = f'set search_path to {quote_schema(schema)},pg_catalog,extensions'
            await connection.execute(path_sql)
            await connection.execute("set lock_timeout='8s'; set statement_timeout='20s'")
            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                            command_timeout=25, setup=setup)
            owner = database.owners[0]
            area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Controlled video replay fixture','test') returning id", owner)

            async def revoke_source(fixture):
                rows = await fixture.evidence_rows()
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]['source_live'])
                self.assertEqual(rows[0]['availability'], 'available')
                self.assertIsNotNone(rows[0]['content'])
                identity = await connection.fetchval('select source_id from jarvis_knowledge_source_versions '
                    'where owner_id=$1 and payload_id=$2', owner, UUID(rows[0]['payload_id']))
                self.assertIsNotNone(identity)
                fixture.withdrawal = {'event_id':rows[0]['id'], 'payload_id':rows[0]['payload_id'],
                    'source_id':str(identity), 'source_revision':1,
                    'window_id':str(fixture.analyzer.requests[0].window_id), 'deletion_completed':False}
                await fixture.sources.delete_source(owner, area, identity, expected_revision=1)
                fixture.withdrawal['deletion_completed'] = True

            async def switch_camera(fixture):
                previous = fixture.lease
                fixture.lease = fixture.window.begin_stream(owner, fixture.scope.session_id,
                                                            previous.visual_session_id)
                self.assertNotEqual(previous.connection_id, fixture.lease.connection_id)

            async def exercise(name, kind='utterance', boundary='analysis', mutation=None, uncertain=False):
                model = ReplayControlledModel(pool, uncertain=uncertain)
                if boundary == 'analysis':
                    model.analysis_release.clear()
                fixture = FocusPipelineReplay(pool=pool, connection=connection, owner=owner,
                    area=area, model=model, schema=schema)
                port = fixture.port(port_type=BarrierReplayPort, fixture=fixture,
                                    boundary=boundary, mutation=mutation)
                offsets = (0, 200, 3000, 3500)
                event = ContextEvent(event_id='context-change', offset_ms=400, kind=kind,
                    text='Synthetic blue key ring' if kind=='goal' else 'The controlled replay context changed.')
                with tempfile.TemporaryDirectory(prefix='jarvis-replay-frames-') as root:
                    root = Path(root)
                    (root/'extracted').mkdir()
                    (root/'extracted'/'frame.jpg').write_bytes(JPEG)
                    inputs = ReplayInputs(duration_ms=4000, initial_context=('Synthetic red key ring',),
                        frames=tuple(FrameAsset(offset_ms=offset, relative_path='extracted/frame.jpg',
                            sha256=hashlib.sha256(JPEG).hexdigest(), width=1, height=1) for offset in offsets),
                        events=(event,))
                    case = ReplayCase(case_id=name, permission=PermissionRecord(
                        evidence='Repository-authored synthetic JPEG; no third-party or user media.',
                        local_evaluation=True, retain_derived_results=True), inputs=inputs,
                        expected_events=('evaluator-only-label',), forbidden_claims=('evaluator-only-forbidden',))
                    try:
                        await fixture.prepare(inputs.initial_context)
                    except BaseException:
                        await port.close()
                        raise
                    origin = datetime.now(UTC)
                    result = await run_replay(case, asset_root=root, port=port, origin=origin,
                        realtime=True, max_model_calls=12, timeout_seconds=60, cleanup_seconds=15)
                report['checks'].append({'name':name, 'status':'running', 'replay':result,
                    'fixture_errors':list(fixture.errors), 'stale_predicate_outcome':port.stale_outcome,
                    'queued_while_model_pending':list(port.pending_enqueues),
                    'drain_calls':port.drain_calls, 'actual_model_calls':list(model.calls)})
                entry = report['checks'][-1]
                self.assertEqual(result['execution_status'], 'completed', result)
                self.assertTrue(result['port_closed'])
                self.assertTrue(fixture.closed)
                self.assertTrue(fixture.entry.focus_resolver._closed)
                self.assertEqual(result['acceptance_status'], 'not_scored')
                self.assertEqual(port.drain_calls, 1)
                self.assertEqual(port.pending_enqueues, [2,3,4])
                self.assertEqual(result['model_calls_attempted'], len(model.calls))
                self.assertIsNone(result['first_useful_audio_ms'])
                self.assertFalse(root.exists(), 'temporary replay JPEG retained')
                out = result['outcomes']
                self.assertTrue(out['owner_joined_before_collect'])
                self.assertEqual(out['fixture_errors'], [])
                self.assertLessEqual(datetime.fromisoformat(out['ready_at']), origin)
                self.assertEqual(out['speech_attempts'], 0)
                self.assertGreaterEqual(len(out['observations']), 1)
                self.assertEqual(len(fixture.frames), 4)
                self.assertEqual([f.captured_at for f in fixture.frames],
                    [origin+timedelta(milliseconds=offset) for offset in offsets])
                self.assertEqual([str(f.id) for f in fixture.frames],
                                 [f['id'] for f in result['submitted_frames']])
                persisted_frames = await connection.fetch('select id,frame_sequence,captured_at from visual_frames '
                    'where owner_id=$1 and session_id=$2 order by frame_sequence', owner, fixture.scope.session_id)
                self.assertEqual([(r['id'],r['frame_sequence'],r['captured_at']) for r in persisted_frames],
                    [(f.id,f.frame_sequence,f.captured_at) for f in fixture.frames])
                released = {frame.id:frame for frame in fixture.frames}
                primary_ids = [r.primary_frame.frame_id for r in fixture.analyzer.requests]
                self.assertTrue(set(primary_ids) <= set(released))
                self.assertEqual(len(primary_ids), len(set(primary_ids)))
                sequences = [released[identity].frame_sequence for identity in primary_ids]
                self.assertEqual(sequences, sorted(sequences))
                self.assertEqual((sequences[0], sequences[-1]), (1,4))
                if boundary == 'analysis':
                    self.assertEqual(sequences, [1,4])
                provenance = {}
                for source in out['frame_sources']:
                    self.assertTrue(source['source_live'])
                    self.assertEqual(source['availability'], 'available')
                    self.assertIsNotNone(source['content'])
                    item = source['content']['replay_provenance']
                    identity = UUID(item['frame_id'])
                    self.assertNotIn(identity, provenance)
                    self.assertIn(identity, released)
                    self.assertEqual(item['sequence'], released[identity].frame_sequence)
                    self.assertEqual(item['offset_ms'], offsets[item['sequence']-1])
                    self.assertEqual(item['sha256'], hashlib.sha256(JPEG).hexdigest())
                    self.assertEqual(item['camera_visual_session_id'], str(released[identity].visual_session_id))
                    provenance[identity] = item
                self.assertEqual(set(provenance), set(released))
                if kind in {'goal', 'cancel_goal'}:
                    self.assertGreater(out['brief_versions'][-1], out['brief_versions'][0])
                else:
                    self.assertEqual(out['brief_versions'][-1], out['brief_versions'][0],
                                     'Unrelated context is not a semantic focus revision')
                for request in fixture.analyzer.requests:
                    self.assertIsNotNone(request.perception_brief, out['last_brief_omission'])
                    self.assertIsNotNone(request.perception_brief_input_digest)
                    self.assertLessEqual(request.perception_brief.created_at, request.primary_frame.captured_at)
                    self.assertLessEqual(request.primary_frame.captured_at, request.requested_at)
                    self.assertLessEqual(request.requested_at, datetime.now(UTC))
                    self.assertEqual(len(request.frames), len({f.frame_id for f in request.frames}))
                    self.assertEqual(request.perception_brief_input_digest, digest(
                        next(r['document'] for r in out['requests'] if r['window_id']==str(request.window_id))))
                    self.assertEqual([(f.captured_at,f.frame_sequence) for f in request.frames],
                                     sorted((f.captured_at,f.frame_sequence) for f in request.frames))
                    for frame in request.frames:
                        self.assertIn(frame.frame_id, released)
                        self.assertEqual(frame.frame_sequence, released[frame.frame_id].frame_sequence)
                        self.assertLessEqual(frame.frame_sequence, request.primary_frame.frame_sequence)
                        self.assertEqual(frame.captured_at, origin+timedelta(milliseconds=offsets[frame.frame_sequence-1]))
                        self.assertEqual(frame.image_bytes, JPEG)
                        if request.camera_connection_id is not None:
                            self.assertEqual(str(request.camera_connection_id),
                                             provenance[frame.frame_id]['camera_connection_id'])
                            self.assertEqual(str(request.camera_visual_session_id),
                                             provenance[frame.frame_id]['camera_visual_session_id'])
                for item in out['evidence']:
                    saved = item['content']
                    if saved is None:
                        self.assertIs(mutation, revoke_source)
                        self.assertIsNotNone(out['withdrawal'])
                        self.assertEqual(item['id'], out['withdrawal']['event_id'])
                        self.assertEqual(item['payload_id'], out['withdrawal']['payload_id'])
                        self.assertFalse(item['source_live'])
                        self.assertNotEqual(item['availability'], 'available')
                        self.assertEqual(item['fence_error_code'], 'live_state_context_revoked')
                        continue
                    self.assertTrue(item['source_live'])
                    self.assertEqual(item['availability'], 'available')
                    self.assertTrue(out['input_sources'][saved['analysis_input_id']]['source_live'])
                    self.assertEqual(out['input_sources'][saved['analysis_input_id']]['availability'], 'available')
                    self.assertIsNotNone(out['captured_inputs'][saved['analysis_input_id']])
                    request = next(r for r in out['requests'] if r['window_id']==saved['evidence']['window_id'])
                    self.assertEqual(saved['request_digest'], request['input_digest'])
                    self.assertEqual(saved['evidence']['brief'], request['brief'])
                    self.assertEqual(out['captured_inputs'][saved['analysis_input_id']]['request'], request['document'])
                    original = await connection.fetchrow('select e.source_namespace,p.content from jarvis_context_events e '
                        'join jarvis_context_payloads p on p.id=e.payload_id and p.owner_id=e.owner_id '
                        'where e.owner_id=$1 and e.id=$2', owner, UUID(saved['original_source']['source_id']))
                    self.assertEqual(original['source_namespace'], OBSERVATION_NAMESPACE)
                    self.assertEqual(decoded(original['content'])['analysis_input_id'], saved['analysis_input_id'])
                    actual = [(f['frame_id'],f['frame_sequence'],datetime.fromisoformat(f['captured_at']))
                              for f in saved['evidence']['frames']]
                    raw = next(r for r in fixture.analyzer.requests if str(r.window_id)==request['window_id'])
                    self.assertIsNotNone(raw.camera_connection_id)
                    self.assertEqual(saved['camera_connection_id'], str(raw.camera_connection_id))
                    self.assertEqual(actual, [(str(f.frame_id),f.frame_sequence,f.captured_at) for f in raw.frames])
                if mutation is revoke_source:
                    withdrawn = out['withdrawal']
                    self.assertIsNotNone(withdrawn)
                    self.assertTrue(withdrawn['deletion_completed'])
                    self.assertEqual(withdrawn['source_metadata']['status'], 'deleted')
                    self.assertTrue(any(v['source_id']==withdrawn['source_id']
                        and v['payload_id']==withdrawn['payload_id'] and v['revision']==withdrawn['source_revision']
                        for v in withdrawn['source_versions']))
                    state = withdrawn['payload_state']
                    self.assertTrue(state['payload_present'])
                    self.assertIsNone(state['content'])
                    self.assertFalse(state['stored_content_present'])
                    self.assertNotEqual(state['availability'], 'available')
                    self.assertFalse(state['source_live'])
                    self.assertEqual(state['fence_error_code'], 'live_state_context_revoked')
                    self.assertEqual([item['id'] for item in out['unavailable_evidence']], [withdrawn['event_id']])
                    self.assertEqual(withdrawn['window_id'], port.stale_outcome['window_id'])
                else:
                    self.assertEqual(out['unavailable_evidence'], [])
                    self.assertIsNone(out['withdrawal'])
                self.assertTrue(out['canonical_sources'])
                self.assertTrue(all(s['live'] for s in out['canonical_sources']))
                context_receipt = next(r for r in fixture.receipts
                    if r.content['replay_provenance']['event_id']==event.event_id)
                self.assertEqual(context_receipt.source_time, origin+timedelta(milliseconds=event.offset_ms))
                self.assertEqual(len({r.source_event_id for r in fixture.receipts}), len(fixture.receipts))
                for receipt in fixture.receipts:
                    self.assertEqual(receipt.revision, 1)
                    self.assertTrue(receipt.source_event_id.startswith(f'{fixture.replay_id}/{fixture.scope.session_id}/'))
                    self.assertTrue(receipt.content['replay_provenance']['controlled_model_outputs'])
                    self.assertNotIn('evaluator-only', json.dumps(receipt.content))
                for focus in out['focuses']:
                    self.assertIsNone(focus['completed_at'])
                    self.assertIsNone(focus['completion_evidence'])
                if kind=='cancel_goal' or mutation or uncertain:
                    self.assertEqual(out['events'], [])
                    self.assertEqual(out['intents'], [])
                    self.assertTrue(all(f['check_resolution'] is None for f in out['focuses']))
                else:
                    self.assertEqual(len(out['events']), 1)
                    self.assertEqual(len(out['intents']), 1)
                    intent, delivery = out['intents'][0], out['events'][0]
                    self.assertEqual((intent['state'],intent['attempt_count']), ('pending',0))
                    self.assertEqual(intent['source_id'], delivery['id'])
                    self.assertEqual(delivery['event_type'], 'generated')
                    self.assertEqual(delivery['metadata']['physical_action_status'], 'not_established')
                    self.assertEqual(delivery['metadata']['playback_status'], 'unconfirmed')
                    current = next(f for f in out['focuses'] if f['id']==str(fixture.focuses[-1].id))
                    self.assertIsNotNone(current['check_resolution'])
                    self.assertIsNone(current['cleared_at'])
                    self.assertEqual(current['mode'], 'monitor')
                if kind=='goal':
                    first, last = fixture.analyzer.requests
                    self.assertEqual(first.perception_brief.id, last.perception_brief.id)
                    self.assertEqual((first.perception_brief.version, last.perception_brief.version), (1,2))
                    self.assertEqual(first.perception_brief.questions[0].check_ref.source_id, fixture.focuses[0].id)
                    self.assertEqual(last.perception_brief.questions[0].check_ref.source_id, fixture.focuses[1].id)
                    old = next(f for f in out['focuses'] if f['id']==str(fixture.focuses[0].id))
                    self.assertIsNone(old['check_resolution'])
                    self.assertIsNotNone(old['cleared_at'])
                    self.assertTrue(all(e['content']['evidence']['brief']['questions'][0]['check_ref']['source_id']
                        ==str(fixture.focuses[1].id) for e in out['evidence']))
                if boundary=='predicate':
                    self.assertTrue(port.stale_outcome['same_task'])
                    self.assertTrue(port.stale_outcome['task_done'])
                    self.assertFalse(port.stale_outcome['task_cancelled'])
                    self.assertEqual(port.stale_outcome['status'], 'not_applied')
                    self.assertEqual(port.stale_outcome['window_id'], str(fixture.analyzer.requests[0].window_id))
                    expected_reason = ('live_state_context_revoked' if mutation is revoke_source
                                       else 'focus_evidence_assignment_or_camera')
                    self.assertEqual(port.stale_outcome['reason'], expected_reason)
                    if mutation is switch_camera:
                        self.assertNotEqual(port.stale_outcome['camera_connection_id'], str(fixture.lease.connection_id))
                        self.assertEqual(provenance[fixture.frames[0].id]['camera_connection_id'],
                                         port.stale_outcome['camera_connection_id'])
                        self.assertEqual(provenance[fixture.frames[-1].id]['camera_connection_id'],
                                         str(fixture.lease.connection_id))
                        for request in fixture.analyzer.requests:
                            if str(request.camera_connection_id)==port.stale_outcome['camera_connection_id']:
                                self.assertFalse(fixture.entry.camera.is_current(request, fixture.lease.visual_session_id))
                        old_camera_evidence = [item for item in out['evidence']
                            if item['content']['camera_connection_id']==port.stale_outcome['camera_connection_id']]
                        self.assertEqual(len(old_camera_evidence), 1)
                        self.assertEqual(old_camera_evidence[0]['content']['evidence']['window_id'],
                                         port.stale_outcome['window_id'])
                    entry['stale_predicate_outcome'] = port.stale_outcome
                entry.update(status='passed', queued_while_model_pending=port.pending_enqueues,
                    drain_calls=port.drain_calls, actual_model_calls=model.calls,
                    original_capture_identity_and_chronology=True, temporary_assets_removed=True)
                save()

            cases = (
                ('ongoing_frames_context_refresh', {}),
                ('goal_replaced_during_analysis', {'kind':'goal'}),
                ('goal_cancelled_during_analysis', {'kind':'cancel_goal'}),
                ('uncertain_predicate_no_intent', {'uncertain':True}),
                ('source_withdrawn_during_predicate', {'boundary':'predicate','mutation':revoke_source}),
                ('camera_switched_during_predicate', {'boundary':'predicate','mutation':switch_camera}),
            )
            for name, options in cases:
                try:
                    await exercise(name, **options)
                except Exception as error:
                    failure = dict(name=name, status='failed', **sanitized(error))
                    failures.append(failure)
                    if report['checks'] and report['checks'][-1]['name']==name:
                        report['checks'][-1].update(failure)
                    else:
                        report['checks'].append(failure)
                    save()
            report['status'] = 'failed' if failures else 'passed'
        except Exception as error:
            report.update(status='failed', fatal=sanitized(error))
            raise
        finally:
            try:
                if database.manifest['schemas']:
                    await cleanup_fixture_resources(database, report, save, pool=pool, connection=connection)
                    for check in report['checks']:
                        if 'replay' in check:
                            check['replay']['fixture_cleanup_audited'] = True
                elif connection is not None:
                    await connection.close(timeout=5)
            finally:
                save()
                print(json.dumps({'report':str(report_path), 'status':report['status'],
                    'checks':[{'name':c['name'], 'status':c['status'],
                               **({'error_type':c['error_type'], 'frames':c['frames']} if 'error_type' in c else {})}
                              for c in report['checks']], 'cleanup':report['cleanup']}))
        self.assertFalse(failures, json.dumps(failures))


if __name__ == '__main__':
    unittest.main()
