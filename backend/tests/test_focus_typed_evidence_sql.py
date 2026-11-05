"""Actual PostgreSQL focus schema/owner qualification, not full chain acceptance.

Opt in through both manifested DB environment gates. Only a new disposable
schema and generated fixture owners receive writes. No public migration, model,
service, device or provider runs. Existing cleanup audits committed removal on
an independent connection, even after failures.

Evidence/decision context events are explicitly authored test fixtures. They
exercise the real focus owner validator and SQL mutation, not analyzer accuracy,
PerceptionEvidenceConsumer capture, live brief binding, or spoken delivery.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import request_document
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder, context_dependency
from app.live_state.models import GuardedPayloadBinding, SessionScope, SourceDependency
from app.live_state.perception import (
    CheckLifecycle, CheckResolution, DurableResponsibilityCapability, EvidenceReference,
    PerceptionBrief, PerceptionEvidence, PerceptionFinding, PerceptionFrameReference,
)
from app.live_state.perception_focus import CanonicalFocusInputs
from app.memory.prospective.focus_evidence import (
    DECISION_NAMESPACE, FocusEvidenceDecision, FocusEvidencePolicy,
)
from app.memory.prospective.repository import ProspectiveRepository, _focus_from_record
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.evidence_consumer import EVIDENCE_NAMESPACE, EvidenceConsumption
from app.reminders.visual_resolution import VisualPredicateDecision
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest
from app.tasks.repository import TaskRepository
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources


def now():
    return datetime.now(UTC)


def sanitized(error):
    # Never put DSNs, driver messages, source bodies or settings in a report.
    return {'error_type': type(error).__name__, 'code': getattr(error, 'code', None),
            'sqlstate': getattr(error, 'sqlstate', None)}


class DeliberateRollback(Exception):
    pass


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
                     os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'Explicit manifested database required')
class FocusTypedEvidenceSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_focus_schema_owner_resolution_and_rollback(self):
        dsn = os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL']
        directory = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f'focus-typed-evidence-{fixture.run_id}.json'
        report = dict(format='jarvis.focus_typed_evidence.sql.v1', run_id=fixture.run_id,
            status='running', migration_ceiling=41, checks=[], cleanup=None,
            public_migration=False, provider_calls=False, full_chain_acceptance=False,
            evidence_ingress='explicit_test_fixture', network_retries=0)
        failures = []
        connection = pool = None

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2)
                handle.write('\n')

        async def check(name, operation):
            try:
                async with asyncio.timeout(80):
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
            report['checks'].append(entry)
            save()

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(item for item in load_migrations() if item.version <= 41)
            self.assertEqual(migrations[-1].name, 'visual_focus_checks')
            self.assertEqual(migrations[-1].version, 41)
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                migrations=tuple(item for item in migrations if item.version <= 40)), timeout=240)
            path_sql = f'set search_path to {quote_schema(schema)},pg_catalog,extensions'
            await connection.execute(path_sql)
            await connection.execute("set lock_timeout='8s'; set statement_timeout='20s'")
            owner, other_owner = fixture.owners

            # Pre-migration legacy delivery history must not invent typed proof.
            legacy_id = await connection.fetchval("""insert into jarvis_visual_focuses
                (owner_id,target,description,purpose,mode,expires_at,completed_at,completion_evidence)
                values($1,'fixture','legacy target','qualification','find_once',
                    clock_timestamp()+interval '10 minutes',clock_timestamp(),'{"legacy_delivery":true}')
                returning id""", owner)
            await asyncio.wait_for(apply_migrations(connection, schema=schema, migrations=migrations), timeout=90)
            await connection.execute(path_sql)

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                            command_timeout=25, setup=setup)
            repository = ProspectiveRepository(pool)
            sources = SourceRepository(pool)
            area = await connection.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain)
                values($1,'Focus SQL fixture','test') returning id""", owner)
            context = ContextRepository(TaskRepository(pool, TaskLimits()),
                publication=CanonicalSourcePublication(sources, area))

            async def row(identity, c=None):
                return dict(await (c or connection).fetchrow(
                    'select * from jarvis_visual_focuses where owner_id=$1 and id=$2', owner, identity))

            async def admit(mode='find_once'):
                return await repository.set_visual_focus(owner_id=owner, target='synthetic key ring',
                    description='A synthetic red ring; no provider image', purpose='SQL qualification only',
                    mode=mode, expires_at=now()+timedelta(minutes=10))

            def resolution(focus):
                episode = SourceDependency(source_namespace='episode', source_id=focus.id,
                                           revision=focus.episode_revision)
                return CheckResolution(basis='visual_observation', resolved_at=now(),
                    applicable_episode_or_object_ref=episode, source_refs=(EvidenceReference(
                        source=SourceDependency(source_namespace='context_event', source_id=uuid4(), revision=1),
                        captured_at=focus.created_at, episode_or_object_ref=episode,
                        frame_id=uuid4(), visibility='visible'),))

            async def decision_fixture(mode='find_once'):
                focus = await admit(mode)
                session = await connection.fetchval(
                    'insert into jarvis_sessions(owner_id) values($1) returning id', owner)
                scope = SessionScope(owner_id=owner, session_id=session, area_id=area)
                receipt = await context.append(owner, ContextIngress(session_id=session,
                    source_namespace='qualification:focus-sql', source_event_id=str(focus.id),
                    receipt_id=str(uuid4()), kind='tool_outcome', trust='tool_receipt',
                    payload={'tool_name':'set_visual_focus', 'result':{
                        'status':'success', 'visual_focus':focus.as_prompt_payload()}}))
                canonical = CanonicalContextBinder(sources, scope, clock=now, timeout_seconds=18)
                selected = await CanonicalFocusInputs(canonical).read()
                when = now()
                brief = PerceptionBrief(id=uuid4(), version=1, scope=scope, scope_epoch=uuid4(),
                    based_on_live_revision=1, questions=(selected.question,), dependencies=selected.dependencies,
                    durable_responsibility=DurableResponsibilityCapability(owner_namespace='visual_focus', status='available'),
                    created_at=when, fresh_until=when+timedelta(seconds=10), expires_at=when+timedelta(seconds=30))
                # Explicit owner-protocol fixture, NOT a CanonicalPerceptionBinder receipt.
                binding = GuardedPayloadBinding(scope=scope, payload_root_ids=(receipt.payload_id,),
                                                dependencies=brief.dependencies)
                frame = FastPerceptionFrame(uuid4(), 1, when, 'glasses', 'image/jpeg', b'not-a-provider-image')
                request = FastPerceptionRequest(owner_id=owner, session_id=session, window_id=uuid4(),
                    frames=(frame,), previous_working_state={}, existing_entity_ids=frozenset(),
                    recent_audio_transcripts=(), behavior_instructions='', active_reminders=(),
                    active_focus=focus.as_prompt_payload(), recent_interrupts=(), requested_at=when,
                    perception_brief=brief, perception_brief_binding=binding,
                    camera_connection_id=uuid4(), camera_visual_session_id=uuid4())
                finding = PerceptionFinding(question_id=selected.question.id, check_ref=selected.question.check_ref,
                    visibility='visible', literal_finding='Synthetic matching ring.', confidence=0.99,
                    evidence_frame_ids=(frame.frame_id,))
                envelope = PerceptionEvidence(brief=brief, window_id=request.window_id,
                    camera_stream_epoch=request.camera_visual_session_id, requested_at=when, analyzed_at=when,
                    frames=(PerceptionFrameReference(frame_id=frame.frame_id, frame_sequence=1, captured_at=when),),
                    findings=(finding,))
                evidence_record = await context.append(owner, ContextIngress(session_id=session,
                    source_namespace=EVIDENCE_NAMESPACE, source_event_id=str(uuid4()), receipt_id=str(uuid4()),
                    kind='observation', trust='derived', source_time=when, source_payload_ids=(receipt.payload_id,),
                    payload={'format':'jarvis.perception_evidence.v1', 'evidence':envelope.model_dump(mode='json'),
                        'request_digest':digest(request_document(request)),
                        'camera_connection_id':str(request.camera_connection_id)}))
                evidence = EvidenceConsumption('recorded', evidence=envelope, record=evidence_record)
                policy = FocusEvidencePolicy(max_evidence_age_seconds=30)
                verdict = VisualPredicateDecision(verdict='satisfied', basis='condition_observed',
                    observed_condition='Synthetic matching ring.', confidence=0.99, uncertainties=(),
                    supporting_frame_ids=(frame.frame_id,))
                decision_record = await context.append(owner, ContextIngress(session_id=session,
                    source_namespace=DECISION_NAMESPACE, source_event_id=str(uuid4()), receipt_id='decision',
                    kind='observation', trust='derived', source_time=now(),
                    source_payload_ids=(evidence_record.payload_id, receipt.payload_id),
                    payload={'format':'jarvis.focus_predicate.v1', 'policy':policy.model_dump(mode='json'),
                        'question':selected.question.model_dump(mode='json'), 'finding':finding.model_dump(mode='json'),
                        'decision':verdict.model_dump(mode='json'),
                        'evidence_source':context_dependency(evidence_record).model_dump(mode='json'),
                        'request_digest':digest(request_document(request))}))
                decision = FocusEvidenceDecision(canonical, request, evidence, decision_record, policy,
                    lambda request, stream: True, finding.question_id, focus.id)
                return focus, decision

            async def apply(decision, *, rollback=False):
                async with decision.canonical._transaction() as c:
                    await decision.canonical._lock_roots(c, (
                        *decision.request.perception_brief_binding.payload_root_ids,
                        decision.evidence.record.payload_id, decision.record.payload_id))
                    version = await repository.apply_focus_decision(c, owner_id=owner, decision=decision)
                    if rollback:
                        self.assertEqual((await row(decision.focus_id, c))['revision'], 2)
                        raise DeliberateRollback()
                    return version

            async def legacy_compatibility():
                saved = await row(legacy_id)
                self.assertEqual((saved['revision'], saved['episode_revision']), (1, 1))
                self.assertIsNone(saved['check_resolution'])
                self.assertIsNotNone(saved['completed_at'])
                self.assertEqual(json.loads(saved['completion_evidence']), {'legacy_delivery':True})
                return {'initial_versions':[1,1], 'invented_typed_resolution':False}

            async def admission_and_claim_churn():
                focus = await admit()
                self.assertEqual((focus.revision, focus.episode_revision), (1,1))
                token = await repository.try_claim_subject(owner_id=owner, subject_type='visual_focus',
                    subject_id=focus.id, claim_timeout_seconds=30)
                self.assertIsNotNone(token)
                await repository.release_subject_claim(owner_id=owner, subject_type='visual_focus',
                                                       subject_id=focus.id, claim_token=token)
                saved = await row(focus.id)
                self.assertEqual((saved['revision'], saved['episode_revision']), (1,1))
                await connection.execute('update jarvis_visual_focuses set revision=99,episode_revision=99 '
                                         'where owner_id=$1 and id=$2', owner, focus.id)
                saved = await row(focus.id)
                self.assertEqual((saved['revision'], saved['episode_revision']), (1,1))
                return {'claim_churn_bumps_revision':False, 'caller_cannot_assign_update_versions':True}

            async def owner_resolution(mode):
                focus, decision = await decision_fixture(mode)
                self.assertEqual(await apply(decision), 2)
                saved = await row(focus.id)
                stored = _focus_from_record(saved)
                self.assertEqual(stored.check_resolution.basis, 'visual_observation')
                self.assertEqual(stored.check_resolution.source_refs[0].captured_at, decision.request.frames[0].captured_at)
                self.assertIsNone(stored.completed_at)
                self.assertIsNone(stored.completion_evidence)
                expected = CheckLifecycle.TRACKING if mode == 'monitor' else CheckLifecycle.RESOLVED
                self.assertEqual(stored.as_check(decision.canonical.scope, now=now()).lifecycle, expected)
                generation = await connection.fetchval('select generation from jarvis_attention_visual_subjects '
                    "where owner_id=$1 and subject_type='visual_focus' and subject_id=$2", owner, focus.id)
                self.assertEqual(generation, 1)
                with self.assertRaises(SourceError):
                    await apply(decision)
                self.assertEqual((await row(focus.id))['revision'], 2)
                return {'revision':2, 'episode_revision':1, 'subject_generation':generation,
                        'lifecycle':expected.value, 'duplicate_old_decision_rejected':True,
                        'watch_completed':False, 'physical_action_completion':False}

            async def owner_rollback():
                focus, decision = await decision_fixture()
                with self.assertRaises(DeliberateRollback):
                    await apply(decision, rollback=True)
                saved = await row(focus.id)  # Separate connection observes committed state.
                self.assertEqual(saved['revision'], 1)
                self.assertIsNone(saved['check_resolution'])
                self.assertEqual(await apply(decision), 2)
                return {'separate_connection_observed_rollback':True, 'same_decision_retry_committed':True}

            async def obsolete_decision(operation):
                focus, decision = await decision_fixture()
                if operation == 'replace':
                    replacement = await admit()
                else:
                    await repository.clear_active_visual_focus(owner_id=owner, expected_focus_id=focus.id)
                with self.assertRaises(SourceError):
                    await apply(decision)
                saved = _focus_from_record(await row(focus.id))
                self.assertEqual(saved.as_check(decision.canonical.scope, now=now()).lifecycle, CheckLifecycle.DISMISSED)
                self.assertIsNone(saved.check_resolution)
                self.assertIsNone(saved.completed_at)
                if operation == 'replace':
                    await repository.clear_active_visual_focus(owner_id=owner, expected_focus_id=focus.id)
                    self.assertIsNone((await row(replacement.id))['cleared_at'])
                return {'stale_owner_resolution_rejected':True, 'projection':'dismissed'}

            async def corrected_episode():
                focus, decision = await decision_fixture('monitor')
                await apply(decision)
                await connection.execute('update jarvis_visual_focuses set description=$3 '
                    'where owner_id=$1 and id=$2', owner, focus.id, 'Corrected synthetic description')
                saved = await row(focus.id)
                self.assertEqual((saved['revision'], saved['episode_revision']), (3,2))
                self.assertIsNone(saved['check_resolution'])
                with self.assertRaises(SourceError):
                    await apply(decision)
                return {'revision':3, 'episode_revision':2, 'old_resolution_cleared':True}

            async def expired_and_dismissed():
                results = []
                for operation in ('expiry','dismissal'):
                    focus, decision = await decision_fixture('monitor')
                    await apply(decision)
                    if operation == 'expiry':
                        await connection.execute('update jarvis_visual_focuses set expires_at=clock_timestamp() '
                            "-interval '1 second' where owner_id=$1 and id=$2", owner, focus.id)
                    else:
                        await repository.clear_active_visual_focus(owner_id=owner, expected_focus_id=focus.id)
                    saved = _focus_from_record(await row(focus.id))
                    projected = saved.as_check(decision.canonical.scope, now=now())
                    expected = CheckLifecycle.EXPIRED if operation == 'expiry' else CheckLifecycle.DISMISSED
                    self.assertEqual(projected.lifecycle, expected)
                    self.assertIsNone(projected.resolution)
                    self.assertIsNotNone(saved.check_resolution)
                    self.assertIsNone(saved.completed_at)
                    self.assertIsNone(saved.completion_evidence)
                    results.append(expected.value)
                return {'noncompletion_lifecycles':results, 'historical_evidence_retained':True}

            async def delivery_separation(mode):
                focus, decision = await decision_fixture(mode)
                await apply(decision)
                before = await row(focus.id)
                token = await repository.try_claim_subject(owner_id=owner, subject_type='visual_focus',
                    subject_id=focus.id, claim_timeout_seconds=30)
                event = await repository.reserve_delivery_event(owner_id=owner,
                    session_id=decision.request.session_id, subject_type='visual_focus', subject_id=focus.id,
                    claim_token=token, occurrence_key=str(uuid4()), message='Synthetic fixture delivery.',
                    evidence={'fixture_delivery':True}, dedupe_window_seconds=1)
                self.assertIsNotNone(event)
                await repository.resolve_delivery_event(owner_id=owner, event_id=event.id,
                    outcome='delivered', provider_response_id=None)
                saved = await row(focus.id)
                self.assertEqual(saved['check_resolution'], before['check_resolution'])
                self.assertEqual(saved['episode_revision'], 1)
                self.assertEqual(saved['completed_at'] is not None, mode == 'find_once')
                self.assertEqual(saved['revision'], 3 if mode == 'find_once' else 2)
                return {'watch_completed':mode == 'find_once', 'typed_resolution_unchanged':True,
                        'actual_speech':False}

            async def malformed_resolution_constraints():
                focus = await admit()
                valid = resolution(focus).model_dump(mode='json')
                cases = [('empty_object', {}), ('array', []), ('string', 'wrong'), ('json_null', None)]
                for key in ('basis','source_refs','resolved_at','applicable_episode_or_object_ref'):
                    cases.append(('missing_'+key, {name:value for name,value in valid.items() if name != key}))
                    cases.append(('null_'+key, valid | {key:None}))
                cases.extend([('invalid_basis', valid | {'basis':'match'}),
                    ('nonarray_refs', valid | {'source_refs':{}}),
                    ('empty_refs', valid | {'source_refs':[]}),
                    ('too_many_refs', valid | {'source_refs':valid['source_refs'] * 9}),
                    ('nonobject_episode', valid | {'applicable_episode_or_object_ref':[]}),
                    ('nonstring_time', valid | {'resolved_at':1})])
                for label, value in cases:
                    with self.subTest(constraint=label):
                        with self.assertRaises(asyncpg.CheckViolationError):
                            async with connection.transaction():
                                await connection.execute('update jarvis_visual_focuses set check_resolution=$3::jsonb '
                                    'where owner_id=$1 and id=$2', owner, focus.id, json.dumps(value))
                        saved = await row(focus.id)
                        self.assertEqual(saved['revision'], 1)
                        self.assertIsNone(saved['check_resolution'])
                await connection.execute('update jarvis_visual_focuses set check_resolution=$3::jsonb '
                    'where owner_id=$1 and id=$2', owner, focus.id, json.dumps(valid))
                self.assertEqual((await row(focus.id))['revision'], 2)
                # SQL NULL is the valid absence of resolution; JSON null is not.
                await connection.execute('update jarvis_visual_focuses set check_resolution=null '
                    'where owner_id=$1 and id=$2', owner, focus.id)
                self.assertEqual((await row(focus.id))['revision'], 3)
                return {'rejected_cases':[label for label,value in cases], 'sql_null_allowed':True,
                        'constraint_errors_did_not_consume_revision':True}

            async def positive_version_constraints():
                for revision, episode in ((0,1),(-1,1),(1,0),(1,-1)):
                    with self.assertRaises(asyncpg.CheckViolationError):
                        async with connection.transaction():
                            await connection.execute("""insert into jarvis_visual_focuses
                                (owner_id,target,description,purpose,mode,expires_at,cleared_at,revision,episode_revision)
                                values($1,'fixture','fixture','fixture','monitor',
                                    clock_timestamp()+interval '1 minute',clock_timestamp(),$2,$3)""",
                                other_owner, revision, episode)
                return {'nonpositive_insert_versions_rejected':4}

            await check('legacy_upgrade_does_not_invent_resolution', legacy_compatibility)
            await check('admission_claim_churn_and_owner_issued_versions', admission_and_claim_churn)
            for mode in ('find_once','monitor'):
                await check('owner_resolution_'+mode, lambda mode=mode: owner_resolution(mode))
            await check('owner_resolution_transaction_rollback', owner_rollback)
            for operation in ('replace','cancel'):
                await check('obsolete_decision_'+operation, lambda operation=operation: obsolete_decision(operation))
            await check('descriptor_correction_rebinds_episode', corrected_episode)
            await check('expiry_and_dismissal_are_not_completion', expired_and_dismissed)
            for mode in ('find_once','monitor'):
                await check('delivery_separation_'+mode, lambda mode=mode: delivery_separation(mode))
            await check('malformed_resolution_constraints_fail_closed', malformed_resolution_constraints)
            await check('positive_version_constraints', positive_version_constraints)
            report['status'] = 'failed' if failures else 'passed'
            save()
        except Exception as error:
            report.update(status='failed', fatal=sanitized(error))
            save()
            raise AssertionError('Focus SQL qualification stopped; see sanitized report') from None
        finally:
            try:
                if fixture.manifest['schemas']:
                    await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
                else:
                    if connection is not None:
                        await connection.close(timeout=5)
                    report['cleanup'] = {'status':'not_required_no_resources_allocated',
                        'schemas_requested':0, 'independent_remote_audit':'not_attempted'}
            finally:
                save()
                print(json.dumps({'report':str(report_path), 'status':report['status'],
                    'checks':report['checks'], 'cleanup':report['cleanup'],
                    **({'fatal':report['fatal']} if 'fatal' in report else {})}))
        self.assertFalse(failures, json.dumps(failures))


if __name__ == '__main__':
    unittest.main()
