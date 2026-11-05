"""Opt-in actual SQL composition, controlled model/images, never hosted evidence."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.responsive_voice import hosted_pipeline_replay as hosted_replay
from evals.responsive_voice.hosted_pipeline_replay import Limits, SharedModelGate, run_pair
from evals.responsive_voice.test_hosted_pipeline_replay import make_payload
from evals.responsive_voice.video_replay import PipelineReplayPort


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
                     os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'Explicit manifested isolated SQL configuration required')
class HostedPipelineSQLTests(unittest.IsolatedAsyncioTestCase):
    async def qualify_eligible_predicate(self, verdict):
        """Control model timing/results only, never canonical or owner outcomes."""
        import asyncio

        compositions, call_records = [], []
        active = maximum_active = 0

        class QueuedBeforeReleasePort(PipelineReplayPort):
            def __init__(port, *args, composition, **kwargs):
                super().__init__(*args, **kwargs)
                port.composition, port.enqueued = composition, []

            async def enqueue(port, frame):
                await super().enqueue(frame)
                port.enqueued.append(frame.sequence)

            async def drain(port):
                self.assertEqual(port.enqueued, [1, 2, 3])
                port.composition.analysis_release.set()
                await super().drain()
                # Admit the latest ORIGINAL committed evidence only after the
                # analyzer has released the shared limiter. No predicate waits
                # for another analysis while holding that limiter.
                self.assertTrue(port.composition.deferred_offers)
                accepted = port.composition.actual_offer(**port.composition.deferred_offers[-1])
                port.composition.entry.last_focus_resolution_scheduled = accepted
                self.assertTrue(accepted)
                port.composition.predicate_admitted_after_drain = True

            async def close(port):
                port.composition.analysis_release.set()
                await super().close()

        class ObservedComposition(hosted_replay.HostedPipelineComposition):
            def __init__(composition, **kwargs):
                composition.analysis_release = asyncio.Event()
                composition.deferred_offers = []
                composition.predicate_admitted_after_drain = False
                super().__init__(**kwargs)
                compositions.append(composition)

            async def open(composition, *args, **kwargs):
                await super().open(*args, **kwargs)
                composition.actual_offer = composition.entry.focus_resolver.offer

                def defer_offer(**original):
                    composition.deferred_offers.append(original)
                    # Truthful scheduling state: not admitted yet. The actual
                    # owner admission, guards and outcome run after drain.
                    return False
                composition.entry.focus_resolver.offer = defer_offer

            def port(composition):
                return super().port(port_type=QueuedBeforeReleasePort, composition=composition)

        async def generate(*, prompt, images):
            nonlocal active, maximum_active
            self.assertNotIn('EVALUATOR_SECRET', prompt)
            self.assertTrue(images)
            composition = compositions[-1]
            predicate = prompt.startswith('Evaluate whether these original images')
            active += 1
            maximum_active = max(maximum_active, active)
            call_records.append({'case_id': composition.gate.current_case.case_id,
                'phase': 'predicate' if predicate else 'analysis',
                'frame_count': len(images), 'session_id': str(composition.scope.session_id)})
            try:
                if predicate:
                    self.assertTrue(composition.predicate_admitted_after_drain)
                    return json.dumps({'verdict': verdict,
                        'basis': 'condition_observed' if verdict == 'satisfied' else 'insufficient_evidence',
                        'observed_condition': 'The controlled target is visible in the supplied original frames.'
                            if verdict == 'satisfied' else 'The supplied frames do not establish the full condition.',
                        'confidence': .99 if verdict == 'satisfied' else .4,
                        'uncertainties': [] if verdict == 'satisfied' else ['Controlled ambiguous identity.'],
                        'supporting_frame_ids': [f'frame:{len(images)-1}']})
                await asyncio.wait_for(composition.analysis_release.wait(), 15)
                context = json.loads(prompt.split('Context JSON:\n```json\n')[1].split('\n```')[0])
                question, = context['perception_assignment']['questions']
                return json.dumps({'summary': 'Controlled eligible target observation.',
                    'starts_new_event': False, 'confidence': .99,
                    'perception_findings': [{'question_id': question['id'], 'check_id': question['check_id'],
                        'visibility': 'visible', 'literal_finding': 'The controlled target is visible.',
                        'confidence': .99, 'evidence_frame_ids': [f'frame:{len(images)-1}']}]})
            finally:
                active -= 1

        output = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']) / ('eligible-' + verdict)
        with TemporaryDirectory() as directory:
            payload = make_payload(Path(directory), offsets=(0, 300, 600))
            gate = SharedModelGate(SimpleNamespace(provider='controlled-no-provider', model='fixture',
                generate_temporal_json=generate), limits=Limits(trial_seconds=90))
            with patch.object(hosted_replay, 'HostedPipelineComposition', ObservedComposition):
                result = await run_pair(payload, gate, dsn=os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
                                        output_dir=output)

        receipt = {'status': 'assertions_pending', 'predicate_verdict': verdict,
            'controlled_model_outputs': True, 'actual_recognition': False,
            'call_records': call_records, 'maximum_concurrent_model_calls': maximum_active,
            'predicate_scheduling': 'test-controlled postcommit admission after analysis drain',
            'deferred_offer_counts': [len(c.deferred_offers) for c in compositions],
            'all_predicates_admitted_after_drain': all(c.predicate_admitted_after_drain for c in compositions),
            'shared_model_calls': result['model_calls'], 'audit': result['audit'],
            'cases': [{'case_id': row['case_id'], 'resolver': row['outcomes'].get('resolver'),
                       'focuses': row['outcomes'].get('focuses'), 'events': row['outcomes'].get('events'),
                       'intents': row['outcomes'].get('intents'),
                       'evidence_records': row['outcomes'].get('evidence')}
                      for row in result['cases'] if 'outcomes' in row]}
        hosted_replay.write_private(output / 'predicate-integration.json', receipt)
        self.assertEqual(result['status'], 'completed_unscored', result.get('failure'))
        self.assertEqual(len(result['cases']), 2)
        self.assertEqual(maximum_active, 1)
        self.assertEqual(active, 0)
        self.assertEqual(len(call_records), len(result['model_calls']))
        self.assertLessEqual(len(call_records), gate.limits.calls)
        self.assertEqual(result['http_attempts'], [])
        self.assertEqual(result['cleanup']['manifest_statuses'], ['cleaned', 'cleaned'])
        self.assertEqual(result['audit']['status'], 'passed')
        self.assertEqual(result['audit']['remaining_schemas'], 0)
        self.assertEqual(result['audit']['remaining_public_owner_rows'], 0)
        for replay in result['cases']:
            identity, out = replay['case_id'], replay['outcomes']
            calls = [row for row in call_records if row['case_id'] == identity]
            self.assertGreaterEqual(sum(row['phase'] == 'predicate' for row in calls), 1)
            self.assertGreaterEqual(sum(row['phase'] == 'analysis' for row in calls), 1)
            self.assertTrue(replay['port_closed'])
            self.assertEqual(replay['execution_kind'], 'synthetic')
            self.assertTrue(out['owner_joined_before_collect'])
            self.assertFalse(out['actual_delivery'])
            self.assertEqual(out['recording_only_callbacks'], [])
            self.assertTrue(out['evidence'])
            self.assertTrue(out['canonical_sources'])
            self.assertTrue(all(source['live'] for source in out['canonical_sources']))
            requests = {request['window_id']: request for request in out['requests']}
            submitted = {frame['id'] for frame in replay['submitted_frames']}
            for item in out['evidence']:
                self.assertTrue(item['source_live'])
                self.assertEqual(item['availability'], 'available')
                saved = item['content']
                self.assertIsNotNone(saved)
                request = requests[saved['evidence']['window_id']]
                self.assertEqual(saved['request_digest'], request['input_digest'])
                self.assertEqual(saved['evidence']['brief'], request['brief'])
                self.assertEqual(saved['camera_connection_id'], request['camera_connection_id'])
                self.assertTrue(request['brief']['questions'])
                self.assertTrue(out['input_sources'][saved['analysis_input_id']]['source_live'])
                self.assertEqual(out['captured_inputs'][saved['analysis_input_id']]['request'], request['document'])
                self.assertTrue({frame['frame_id'] for frame in saved['evidence']['frames']} <= submitted)
                self.assertTrue(saved['original_source']['source_id'])
            focus, = out['focuses']
            self.assertEqual(focus['mode'], 'monitor')
            self.assertIsNone(focus['completed_at'])
            self.assertIsNone(focus['completion_evidence'])
            if verdict == 'satisfied':
                self.assertIsNotNone(focus['check_resolution'])
                self.assertEqual(len(out['events']), 1)
                self.assertEqual(len(out['intents']), 1)
                event, intent = out['events'][0], out['intents'][0]
                self.assertEqual((intent['state'], intent['attempt_count']), ('pending', 0))
                self.assertEqual(intent['source_id'], event['id'])
                self.assertEqual(event['event_type'], 'generated')
                self.assertEqual(event['metadata']['physical_action_status'], 'not_established')
                self.assertEqual(event['metadata']['playback_status'], 'unconfirmed')
            else:
                self.assertIsNone(focus['check_resolution'])
                self.assertEqual(out['events'], [])
                self.assertEqual(out['intents'], [])
        receipt['status'] = 'passed'
        hosted_replay.write_private(output / 'predicate-integration.json', receipt)

    async def test_eligible_true_predicate_uses_real_owner_and_single_pending_intent(self):
        await self.qualify_eligible_predicate('satisfied')

    async def test_eligible_uncertain_predicate_preserves_unresolved_owner_without_intent(self):
        await self.qualify_eligible_predicate('uncertain')

    async def test_pair_uses_real_canonical_pipeline_and_independent_cleanup(self):
        calls = []
        async def generate(*, prompt, images):
            self.assertNotIn('EVALUATOR_SECRET', prompt)
            calls.append(prompt)
            if prompt.startswith('Evaluate whether these original images'):
                return json.dumps({'verdict':'uncertain', 'basis':'insufficient_evidence',
                    'observed_condition':'The fixture view does not establish the full requested condition.',
                    'confidence':.4, 'uncertainties':['Controlled ambiguous evidence.'],
                    'supporting_frame_ids':[f'frame:{len(images)-1}']})
            context = json.loads(prompt.split('Context JSON:\n```json\n')[1].split('\n```')[0])
            question, = context['perception_assignment']['questions']
            return json.dumps({'summary':'Controlled partial target observation.', 'starts_new_event':False,
                'confidence':.5, 'perception_findings':[{'question_id':question['id'],
                    'check_id':question['check_id'], 'visibility':'visible',
                    'literal_finding':'Controlled target is partly visible; final state is unconfirmed.',
                    'confidence':.5, 'evidence_frame_ids':[f'frame:{len(images)-1}']}]})
        with TemporaryDirectory() as directory:
            payload = make_payload(Path(directory), offsets=(0, 300, 600))
            output = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']) / 'hosted-pair-report'
            gate = SharedModelGate(SimpleNamespace(provider='controlled-no-provider', model='fixture',
                generate_temporal_json=generate), limits=Limits(trial_seconds=180))
            result = await run_pair(payload, gate, dsn=os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
                                    output_dir=output)
        self.assertEqual(result['status'], 'completed_unscored', result.get('failure'))
        self.assertEqual(len(result['cases']), 2)
        self.assertTrue(calls)
        self.assertLessEqual(len(calls), 12)
        self.assertEqual(result['http_attempts'], [])
        self.assertEqual(result['audit']['remaining_schemas'], 0)
        self.assertEqual(result['audit']['remaining_public_owner_rows'], 0)
        self.assertEqual(result['cleanup']['manifest_statuses'], ['cleaned', 'cleaned'])
        contexts = []
        for replay in result['cases']:
            self.assertEqual(replay['execution_status'], 'completed')
            self.assertEqual(replay['execution_kind'], 'synthetic')
            self.assertTrue(replay['realtime'])
            self.assertEqual([f['offset_ms'] for f in replay['submitted_frames']], [0, 300, 600])
            out = replay['outcomes']
            self.assertTrue(out['controlled_model_outputs'])
            self.assertFalse(out['recognition_qualified'])
            self.assertTrue(out['observations'])
            self.assertTrue(out['evidence'])
            self.assertTrue(out['canonical_sources'])
            self.assertTrue(out['owner_joined_before_collect'])
            self.assertEqual(out['intents'], [])
            self.assertTrue(all(f['completed_at'] is None for f in out['focuses']))
            self.assertTrue(all(r['brief']['questions'] and r['input_digest']
                                and r['camera_connection_id'] for r in out['requests']))
            contexts.append(out['focuses'][0]['owner_id'])
        self.assertNotEqual(contexts[0], contexts[1])


if __name__ == '__main__':
    unittest.main()
