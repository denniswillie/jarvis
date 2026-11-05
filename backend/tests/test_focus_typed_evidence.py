"""Offline focus-owner qualification. No database/provider/hardware is used.

The SQL-shaped store exercises production admission, focus projection/validation,
predicate adapter, owner mutation, prospective claims/reservations and attention
enqueue. It is NOT PostgreSQL locking/migration qualification. The bound payload
fixture is explicit: the real brief binder still needs main's namespace hook.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import request_document
from app.live_state.canonical import CanonicalContextBinder, context_dependency
from app.live_state.models import GuardedPayloadBinding, SessionScope, SourceDependency
from app.live_state.perception import (
    CheckLifecycle, DurableResponsibilityCapability, PerceptionBrief,
    PerceptionEvidence, PerceptionFinding, PerceptionFrameReference,
)
from app.live_state.perception_focus import CanonicalFocusInputs, focus_dependencies
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.memory.prospective.focus_evidence import FocusEvidenceAdapter, FocusEvidencePolicy
from app.memory.prospective.repository import ProspectiveRepository, _focus_from_record
from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.applicability import PerceptionEffectGuard
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest, FastProspectiveMatch
from app.perception.evidence_consumer import EVIDENCE_NAMESPACE, EvidenceConsumption
from app.reminders.visual_resolution import VisualPredicateDecision
from app.tasks.models import digest


class Store:
    def __init__(self):
        self.now = datetime.now(UTC)
        self.rows, self.records, self.events, self.receipts = {}, {}, {}, []
        self.roots, self.intents, self.links = set(), [], []
        self.depth = 0

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self):
        outer = self.depth == 0
        before = deepcopy((self.rows, self.events, self.intents, self.links)) if outer else None
        self.depth += 1
        try:
            yield self
        except BaseException:
            if outer:
                self.rows, self.events, self.intents, self.links = before
            raise
        finally:
            self.depth -= 1

    def is_in_transaction(self):
        return self.depth > 0

    def active(self, row):
        return row['completed_at'] is None and row['cleared_at'] is None and row['expires_at'] > self.now

    def clear(self, row):
        row.update(cleared_at=self.now, trigger_claim_token=None, trigger_claimed_at=None,
                   revision=row['revision'] + 1, subject_generation=row['subject_generation'] + 1,
                   subject_state='cancelled')

    async def fetch(self, query, *args):
        if 's.generation as subject_generation' in query:
            return [dict(row) for row in self.rows.values() if row['owner_id'] == args[0] and self.active(row)]
        if "e.kind='tool_outcome'" in query:
            return [row for row in self.receipts if row.content['result']['visual_focus']['id'] == args[2]]
        raise AssertionError(query)

    async def fetchrow(self, query, *args):
        if 'insert into jarvis_visual_focuses' in query:
            owner, target, description, purpose, mode, message, expires = args
            identity = uuid4()
            row = dict(id=identity, owner_id=owner, target=target, description=description,
                purpose=purpose, mode=mode, created_from_message_id=message, created_at=self.now,
                expires_at=expires, completed_at=None, cleared_at=None, completion_evidence=None,
                trigger_claim_token=None, trigger_claimed_at=None, revision=1, episode_revision=1,
                check_resolution=None, subject_generation=1, subject_state='active',
                subject_expires_at=expires, descriptor_matches=True, descriptor_digest='a' * 64)
            self.rows[identity] = row
            return dict(row)
        if 'set cleared_at = coalesce' in query:
            row = next((row for row in self.rows.values() if self.active(row)
                        and (args[1] is None or row['id'] == args[1])), None)
            if row:
                self.clear(row)
            return dict(row) if row else None
        if 'select * from jarvis_visual_focuses' in query:
            row = self.rows.get(args[1])
            return dict(row) if row and row['owner_id'] == args[0] else None
        if 'insert into jarvis_prospective_delivery_events' in query:
            owner, session, focus, reminder, token, key, message, evidence = args
            if any(event['occurrence_key'] == key for event in self.events.values()):
                return None
            identity = uuid4()
            row = dict(id=identity, owner_id=owner, session_id=session, visual_focus_id=focus,
                reminder_id=reminder, claim_token=token, occurrence_key=key, message=message,
                evidence=evidence, event_type='generated', metadata={}, provider_response_id=None,
                created_at=self.now, updated_at=self.now)
            self.events[identity] = row
            return dict(row)
        raise AssertionError(query)

    async def fetchval(self, query, *args):
        if 'lock_jarvis_copy_sources' in query:
            return set(args[1]) <= self.roots
        if 'select p.content from jarvis_observation_sources' in query:
            return self.original
        if 'set trigger_claim_token = $3' in query:
            row = self.rows.get(args[1])
            if row is None or not self.active(row) or row['trigger_claim_token'] is not None:
                return None
            if any(event['visual_focus_id'] == row['id'] for event in self.events.values()):
                return None
            row.update(trigger_claim_token=args[2], trigger_claimed_at=self.now)
            return args[2]
        if 'set check_resolution=$4' in query:
            row = self.rows[args[1]]
            if row['revision'] != args[2] or not self.active(row):
                return None
            row.update(check_resolution=json.loads(args[3]), revision=row['revision'] + 1)
            return row['revision']
        if 'select exists' in query:
            return any(event['visual_focus_id'] == args[1] for event in self.events.values())
        if 'select true' in query:
            row = self.rows.get(args[1])
            return bool(row and self.active(row) and row['trigger_claim_token'] == args[2])
        raise AssertionError(query)

    async def execute(self, query, *args):
        if 'pg_advisory_xact_lock' in query:
            return
        if 'set cleared_at = coalesce' in query:
            for row in self.rows.values():
                if row['owner_id'] == args[0] and row['completed_at'] is None and row['cleared_at'] is None:
                    self.clear(row)
            return
        if 'set trigger_claim_token = null' in query:
            row = self.rows[args[1]]
            if row['trigger_claim_token'] == args[2]:
                row.update(trigger_claim_token=None, trigger_claimed_at=None)
            return
        if 'set metadata=metadata ||' in query:
            self.events[args[1]]['metadata'].update(json.loads(args[2]))
            return
        if 'insert into jarvis_visual_delivery_sources' in query or 'insert into jarvis_context_payload_dependencies' in query:
            self.links.append(args)
            return
        raise AssertionError(query)

    async def enqueue_visual(self, connection, **kwargs):
        assert connection is self and self.is_in_transaction()
        identity = uuid4()
        self.intents.append(dict(id=identity, **kwargs))
        return identity

    def record(self, scope, content, **changes):
        record = SimpleNamespace(id=uuid4(), payload_id=uuid4(), revision=1,
            owner_id=scope.owner_id, session_id=scope.session_id, is_current=True,
            availability='available', kind='observation', trust='derived', finality='final',
            recorded_at=self.now, source_time=self.now, source_namespace=EVIDENCE_NAMESPACE, content=content)
        for key, value in changes.items():
            setattr(record, key, value)
        self.records[record.id] = record
        self.roots.add(record.payload_id)
        return record


class FocusTypedEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store()
        self.scope = SessionScope(owner_id='focus-fixture', session_id=uuid4())
        self.repository = ProspectiveRepository(self.store)
        self.manager = SimpleNamespace(speak_prospective_interrupt=AsyncMock())
        self.settings = SimpleNamespace(PROSPECTIVE_FIND_ONCE_DEFAULT_TTL_SECONDS=300,
            PROSPECTIVE_MONITOR_DEFAULT_TTL_SECONDS=300, PROSPECTIVE_MAX_CANDIDATES_PER_ANALYSIS=4,
            VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=0.9, PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS=10,
            PROSPECTIVE_TRIGGER_CLAIM_LEASE_SECONDS=30, VISUAL_INTERRUPT_DEDUPE_SECONDS=30)
        self.service = ProspectiveMemoryService(repository=self.repository,
            realtime_session_manager=self.manager, settings=self.settings)
        self.canonical = SimpleNamespace(scope=self.scope, clock=lambda: self.store.now,
            _transaction=self.store.transaction, episode_context=None)

        async def lock_roots(connection, roots):
            if not set(roots) <= self.store.roots:
                raise SourceError('source_removed', 'Fixture source removed')

        async def append(connection, owner, ingress):
            return self.store.record(self.scope, ingress.payload, source_namespace=ingress.source_namespace,
                                     source_time=ingress.source_time)

        self.canonical._lock_roots = lock_roots
        self.canonical.context = SimpleNamespace(
            _record=AsyncMock(side_effect=lambda conn, owner, identity: self.store.records[identity]),
            append_in_transaction=AsyncMock(side_effect=append))
        self.inputs = CanonicalFocusInputs(self.canonical)
        self.patches = [patch('app.live_state.perception_focus.context_record', side_effect=lambda row: row),
                        patch('app.memory.prospective.service.selected_notification_repository', return_value=self.store)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.camera = True

    async def admit(self, mode='find_once'):
        focus = await self.service.set_visual_focus(owner_id=self.scope.owner_id, target='keys',
            description='A red key ring', purpose='Find the requested keys', mode=mode, expires_in_seconds=300)
        receipt = self.store.record(self.scope, {'tool_name': 'set_visual_focus',
            'result': {'status': 'success', 'visual_focus': focus.as_prompt_payload()}},
            kind='tool_outcome', trust='tool_receipt')
        self.store.receipts.append(receipt)
        return focus

    async def assignment(self, *, mode='find_once', visibility='visible', confidence=0.99):
        self.focus = await self.admit(mode)
        self.selected = await self.inputs.read()
        now = self.store.now
        self.brief = PerceptionBrief(id=uuid4(), version=1, scope=self.scope, scope_epoch=uuid4(),
            based_on_live_revision=1, questions=(self.selected.question,), dependencies=self.selected.dependencies,
            durable_responsibility=DurableResponsibilityCapability(owner_namespace='visual_focus', status='available'),
            created_at=now, fresh_until=now + timedelta(seconds=10), expires_at=now + timedelta(seconds=30))
        root, original = uuid4(), uuid4()
        self.store.roots.update((root, original))
        self.binding = GuardedPayloadBinding(scope=self.scope, dependencies=self.brief.dependencies, payload_root_ids=(root,))
        frame = FastPerceptionFrame(uuid4(), 1, now, 'glasses', 'image/jpeg', b'offline-only')
        self.store.original = {'frame_links': [{'frame_id': str(frame.frame_id)}]}
        self.request = FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(frame,), previous_working_state={}, existing_entity_ids=frozenset(),
            recent_audio_transcripts=(), behavior_instructions='', active_reminders=(),
            active_focus=self.focus.as_prompt_payload(), recent_interrupts=(), requested_at=now,
            perception_brief=self.brief, perception_brief_binding=self.binding,
            camera_connection_id=uuid4(), camera_visual_session_id=uuid4())
        self.finding = PerceptionFinding(question_id=self.selected.question.id, check_ref=self.selected.question.check_ref,
            visibility=visibility, literal_finding='A red key ring is on the table.', confidence=confidence,
            evidence_frame_ids=(frame.frame_id,))
        envelope = PerceptionEvidence(brief=self.brief, window_id=self.request.window_id,
            camera_stream_epoch=self.request.camera_visual_session_id, requested_at=now, analyzed_at=now,
            frames=(PerceptionFrameReference(frame_id=frame.frame_id, frame_sequence=1, captured_at=now),),
            findings=(self.finding,))
        record = self.store.record(self.scope, {'format': 'jarvis.perception_evidence.v1',
            'evidence': envelope.model_dump(mode='json'), 'request_digest': digest(request_document(self.request)),
            'camera_connection_id': str(self.request.camera_connection_id)})
        self.evidence = EvidenceConsumption('recorded', evidence=envelope, record=record)

        @asynccontextmanager
        async def bound_guard(brief, binding, *, effect_payload_ids=()):
            async with self.store.transaction() as connection:
                await self.canonical._lock_roots(connection, (*binding.payload_root_ids, *effect_payload_ids))
                await self.inputs.validate(connection, brief)
                yield connection

        self.guard = PerceptionEffectGuard(self.brief, self.binding, bound_guard, lambda: True, original)
        self.evaluate = AsyncMock(return_value=self.verdict())
        self.adapter = FocusEvidenceAdapter(canonical=self.canonical, prospective=self.service,
            evaluate=self.evaluate, enabled=True)

    def verdict(self, **changes):
        values = dict(verdict='satisfied', basis='condition_observed', observed_condition='The described keys are visible.',
            confidence=0.99, uncertainties=(), supporting_frame_ids=self.finding.evidence_frame_ids)
        return VisualPredicateDecision(**(values | changes))

    async def consume(self):
        return await self.adapter.consume(request=self.request, evidence=self.evidence, effect_guard=self.guard,
            camera_is_current=lambda request, stream: self.camera)

    def row(self):
        return self.store.rows[self.focus.id]

    def assert_no_match(self):
        self.assertIsNone(self.row()['check_resolution'])
        self.assertIsNone(self.row()['completed_at'])
        self.assertIsNone(self.row()['completion_evidence'])
        self.assertEqual(self.store.intents, [])
        self.manager.speak_prospective_interrupt.assert_not_awaited()

    async def legacy(self):
        await self.service.process_matches(request=self.request, matches=[FastProspectiveMatch(
            subject_type='visual_focus', subject_id=self.focus.id, confidence=0.99,
            observed_condition='Keys visible', supporting_frame_ids=list(self.finding.evidence_frame_ids),
            suggested_message='The keys are visible.')], source_payload_id=self.guard.observation_payload_id)

    async def test_admission_versions_and_distinct_subject_generation(self):
        await self.assignment()
        question = self.selected.question
        self.assertEqual((question.check_ref.source_namespace, question.check_ref.revision, question.check_ref.generation),
                         ('visual_focus', 1, None))
        self.assertEqual(question.episode_or_object_ref.source_id, self.focus.id)
        self.assertIsNone(question.episode_or_object_ref.generation)
        self.assertEqual(len(focus_dependencies(self.brief)), 3)
        self.assertEqual(question.focus_ref.source_namespace, 'visual_focus_subject')

    async def test_owner_match_uses_prospective_attention_without_completing_action_or_watch(self):
        await self.assignment()
        self.assertEqual((await self.consume()).status, 'owner_processed')
        row = self.row()
        self.assertEqual(row['revision'], 2)
        self.assertEqual(row['subject_generation'], 1)
        self.assertEqual(row['episode_revision'], 1)
        self.assertIsNone(row['completed_at'])
        self.assertIsNone(row['completion_evidence'])
        check = _focus_from_record(row).as_check(self.scope, now=self.store.now)
        self.assertEqual(check.lifecycle, CheckLifecycle.RESOLVED)
        self.assertEqual(check.resolution.basis, 'visual_observation')
        self.assertEqual(check.resolution.source_refs[0].captured_at, self.request.frames[0].captured_at)
        self.assertEqual(len(self.store.intents), 1)
        self.assertEqual(self.store.intents[0]['perception_focus'], self.guard.focus_fence)
        self.assertEqual(next(iter(self.store.events.values()))['metadata']['physical_action_status'], 'not_established')
        self.assertEqual(len(self.store.links), 2)
        self.manager.speak_prospective_interrupt.assert_not_awaited()

    async def test_monitor_retains_evidence_and_stays_active(self):
        await self.assignment(mode='monitor')
        self.assertEqual((await self.consume()).status, 'owner_processed')
        focus = _focus_from_record(self.row())
        check = focus.as_check(self.scope, now=self.store.now)
        self.assertEqual(check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(check.resolution)
        self.assertTrue(check.evidence_refs)
        self.assertIsNotNone(focus.check_resolution)
        self.assertTrue(self.store.active(self.row()))
        fresh = await self.inputs.read()
        self.assertEqual(fresh.question.check_ref.revision, 2)

    async def test_typed_then_legacy_then_replayed_typed_has_one_intent(self):
        await self.assignment()
        await self.consume()
        await self.legacy()
        self.assertEqual((await self.consume()).status, 'not_applied')
        self.assertEqual(len(self.store.intents), 1)
        self.assertEqual(len(self.store.events), 1)
        self.assertEqual(self.row()['revision'], 2)

    async def test_legacy_then_typed_records_resolution_without_duplicate_intent(self):
        await self.assignment()
        await self.legacy()
        self.assertEqual((await self.consume()).status, 'owner_processed')
        self.assertEqual(len(self.store.intents), 1)
        self.assertEqual(len(self.store.events), 1)
        self.assertIsNotNone(self.row()['check_resolution'])

    async def test_uncertainty_and_visibility_never_establish_match(self):
        for visibility, confidence in [('unclear', 0.99), ('not_visible', 0.99), ('visible', 0.3)]:
            with self.subTest(visibility=visibility, confidence=confidence):
                await self.assignment(visibility=visibility, confidence=confidence)
                self.assertEqual((await self.consume()).status, 'observed_without_resolution')
                self.evaluate.assert_not_awaited()
                self.assert_no_match()

    async def test_predicate_uncertainty_visibility_only_and_foreign_frame_fail_closed(self):
        for changes in [dict(verdict='uncertain'), dict(basis='visibility_only'),
                        dict(uncertainties=('Could be another key ring.',)), dict(supporting_frame_ids=(uuid4(),))]:
            with self.subTest(changes=changes):
                await self.assignment()
                self.evaluate.return_value = self.verdict(**changes)
                self.assertEqual((await self.consume()).status, 'observed_without_resolution')
                self.assert_no_match()

    async def test_replace_and_cancel_during_analysis_never_settle_old_focus(self):
        for operation in ['replace', 'cancel']:
            with self.subTest(operation=operation):
                await self.assignment()
                entered, release = asyncio.Event(), asyncio.Event()

                async def slow(*args):
                    self.assertFalse(self.store.is_in_transaction())
                    entered.set()
                    await release.wait()
                    return self.verdict()

                self.evaluate.side_effect = slow
                pending = asyncio.create_task(self.consume())
                await asyncio.wait_for(entered.wait(), 1)
                if operation == 'replace':
                    replacement = await self.admit()
                else:
                    await self.service.clear_visual_focus(owner_id=self.scope.owner_id, focus_id=self.focus.id)
                release.set()
                self.assertEqual((await pending).status, 'not_applied')
                self.assert_no_match()
                check = _focus_from_record(self.row()).as_check(self.scope, now=self.store.now)
                self.assertEqual(check.lifecycle, CheckLifecycle.DISMISSED)
                if operation == 'replace':
                    await self.service.clear_visual_focus(owner_id=self.scope.owner_id, focus_id=self.focus.id)
                    self.assertTrue(self.store.active(self.store.rows[replacement.id]))

    async def test_camera_retirement_source_withdrawal_and_expiry_during_analysis(self):
        for operation in ['camera', 'source', 'expiry', 'retire']:
            with self.subTest(operation=operation):
                await self.assignment()

                async def change(*args):
                    if operation == 'camera':
                        self.camera = False
                    elif operation == 'source':
                        self.store.roots.remove(self.evidence.record.payload_id)
                    elif operation == 'expiry':
                        self.store.now += timedelta(seconds=31)
                    else:
                        self.adapter.retire()
                    return self.verdict()

                self.evaluate.side_effect = change
                self.assertEqual((await self.consume()).status, 'not_applied')
                self.assert_no_match()
                self.camera = True

    async def test_expired_tracking_is_not_resolution(self):
        await self.assignment()
        self.store.now = self.focus.expires_at
        self.assertEqual((await self.consume()).status, 'not_applied')
        self.assertEqual(_focus_from_record(self.row()).as_check(self.scope, now=self.store.now).lifecycle,
                         CheckLifecycle.EXPIRED)
        self.assert_no_match()

    async def test_evidence_envelope_or_episode_version_tamper_is_rejected(self):
        for operation in ['digest', 'episode', 'revision', 'scope']:
            with self.subTest(operation=operation):
                await self.assignment()
                if operation == 'digest':
                    self.evidence.record.content['request_digest'] = '0' * 64
                elif operation == 'episode':
                    self.row()['episode_revision'] += 1
                elif operation == 'revision':
                    self.row()['revision'] += 1
                else:
                    self.evidence.record.owner_id = 'other-owner'
                self.assertEqual((await self.consume()).status, 'not_applied')
                self.assert_no_match()

    async def test_unrelated_live_revision_is_not_owner_version(self):
        await self.assignment()
        self.canonical.unrelated_live_revision = 1000
        self.assertEqual((await self.consume()).status, 'owner_processed')

    async def test_absent_evaluator_disabled_and_busy_are_honest(self):
        await self.assignment()
        self.adapter.evaluate = None
        self.assertEqual((await self.consume()).status, 'validator_unavailable')
        self.adapter.enabled = False
        self.assertEqual((await self.consume()).status, 'disabled')
        self.adapter.enabled, self.adapter.evaluate, self.adapter._busy = True, self.evaluate, True
        self.assertEqual((await self.consume()).status, 'busy')
        self.assert_no_match()

    async def test_attention_unavailable_cannot_fall_back_to_direct_speech(self):
        await self.assignment()
        with patch('app.memory.prospective.service.selected_notification_repository', return_value=None):
            self.assertEqual((await self.consume()).status, 'not_applied')
        self.assert_no_match()

    async def test_attention_failure_rolls_back_typed_owner_effect(self):
        await self.assignment()
        with patch.object(self.store, 'enqueue_visual', AsyncMock(side_effect=RuntimeError('fixture failure'))):
            self.assertEqual((await self.consume()).status, 'unavailable')
        self.assert_no_match()
        self.assertEqual(self.row()['revision'], 1)
        self.assertIsNone(self.row()['trigger_claim_token'])

    async def test_main_binder_binds_exact_focus_refs_and_original_receipt(self):
        from app.tasks.context_repository import context_record
        from test_session_perception_focus import FocusPersistence, NOW

        # Unlike the adapter's SQL-shaped fixture above, this exercises actual
        # CanonicalContextBinder, payload insertion, event decoding and brief
        # binding. Only persistence is in-memory; no runtime method is mocked.
        persistence = FocusPersistence(self.scope)
        row, receipt = persistence.add_focus(revision=5, episode_revision=2, generation=7)
        canonical = CanonicalContextBinder(SimpleNamespace(pool=None, transaction=persistence.transaction),
            self.scope, clock=lambda: NOW)
        inputs = CanonicalFocusInputs(canonical)
        with patch('app.live_state.perception_focus.context_record', side_effect=context_record):
            selected = await inputs.read()
            brief = PerceptionBrief(id=uuid4(), version=1, scope=self.scope, scope_epoch=uuid4(),
                based_on_live_revision=1, questions=(selected.question,), dependencies=selected.dependencies,
                durable_responsibility=DurableResponsibilityCapability(
                    owner_namespace='responsibility', status='unavailable'),
                created_at=NOW, fresh_until=NOW+timedelta(seconds=10), expires_at=NOW+timedelta(seconds=30))
            binder = CanonicalPerceptionBinder(canonical, focus=inputs)
            binding = await binder.bind(brief)
            question = brief.questions[0]
            expected = {
                SourceDependency(source_namespace='visual_focus', source_id=row['id'], revision=5),
                SourceDependency(source_namespace='episode', source_id=row['id'], revision=2),
                SourceDependency(source_namespace='visual_focus_subject', source_id=row['id'],
                                 revision=7, generation=7),
                context_dependency(receipt),
            }
            self.assertEqual(set(binding.dependencies), expected)
            self.assertEqual(binding.dependencies, brief.dependencies)
            self.assertEqual(set(focus_dependencies(brief)), expected - {context_dependency(receipt)})
            self.assertEqual(question.check_ref.revision, 5)
            self.assertEqual(question.episode_or_object_ref.revision, 2)
            self.assertIsNone(question.check_ref.generation)
            self.assertIsNone(question.episode_or_object_ref.generation)
            root, = binding.payload_root_ids
            self.assertEqual(persistence.payloads[root], {
                'format':'jarvis.perception_brief.v1', 'brief':brief.model_dump(mode='json')})
            self.assertEqual(persistence.parents[root], {receipt.payload_id})
            self.assertEqual(len(persistence.event_edges), 1)
            self.assertEqual(persistence.event_edges[0][2:],
                (receipt.id, receipt.revision, receipt.payload_id, 'current'))
            async with binder.guard(brief, binding):
                self.assertTrue(persistence.is_in_transaction())
            self.assertEqual(await binder.bind(brief), binding)
            for field in ('revision', 'episode_revision', 'subject_generation'):
                with self.subTest(changed_counter=field):
                    original = row[field]
                    row[field] += 1
                    try:
                        with self.assertRaises(SourceError):
                            async with binder.guard(brief, binding):
                                self.fail('Stale focus version reached the effect boundary')
                    finally:
                        row[field] = original
            persistence.payloads[receipt.payload_id] = None
            self.assertFalse(persistence.available(root))
            with self.assertRaises(SourceError):
                async with binder.guard(brief, binding):
                    self.fail('Removed original authorized the bound brief')


if __name__ == '__main__':
    unittest.main()
