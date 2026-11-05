"""Actual retained runner ports, with the calendar-v3 terminal failure shape.

Synthetic existing media and process/host doubles only; not native acceptance.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from uuid import UUID, uuid4

from evals.responsive_voice import native_replay_run as runner
from tests.test_native_retained_sequence import RetainedFixture, run


def terminal_memory_failure(capture):
    capture = copy.deepcopy(capture)
    capture.update(final_ui_status='Evaluation replay capture failed.', callback_failure=None,
        task_barrier_failure_code=None,
        manual_input={'failure_code': None, 'memory_readiness': {
            'status': 'waiting', 'attempts': 1, 'retired': True, 'settlements': []}},
        post_media_capture={'status': 'not_started', 'required_inputs_complete': False},
        utterances=[
            {'ordinal': 0, 'clip_id': 'cal_a1', 'status': 'sdk_input_complete', 'failure_code': None},
            {'ordinal': 1, 'clip_id': 'cal_a2', 'status': 'failed',
             'failure_code': 'memory_readiness_unavailable', 'attempted_clip_samples': 0,
             'sdk_accepted_clip_samples': 0},
            {'ordinal': 2, 'clip_id': 'cal_a3', 'status': 'not_started',
             'failure_code': 'run_closed_before_injection', 'attempted_clip_samples': 0,
             'sdk_accepted_clip_samples': 0}])
    return capture


class TerminalFixture(RetainedFixture):
    def __init__(self, root, failure_stage=None):
        super().__init__(root)
        self.failure_stage = failure_stage

    async def command(self, argv, log_path, timeout=20):
        result = await super().command(argv, log_path, timeout)
        if len(argv) > 2 and argv[2] == 'launch' and result['returncode'] == 0:
            path = self.capture_dirs[-1]/'capture.json'
            capture = runner.read_json(path)
            capture['final_ui_status'] = 'Evaluation replay complete.'
            if self.failure_stage == 'capture' and self.launches == 1:
                capture = terminal_memory_failure(capture)
            runner.save(path, capture)
        elif len(argv) > 2 and argv[2] == 'terminate' and self.failure_stage == 'stop' and self.launches == 1:
            path = self.capture_dirs[-1]/'capture.json'
            runner.save(path, terminal_memory_failure(runner.read_json(path)))
        return result


class RetainedTerminalCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def run_fixture(self, failure_stage):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = TerminalFixture(Path(temporary.name), failure_stage)
        return fixture, await run(fixture)

    def assert_first_phase_failure_cleanly_stops_sequence(self, fixture, result):
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual((fixture.prepares, fixture.launches, fixture.cleanups), (1, 1, 1))
        self.assertEqual(len(result['phases']), 1)
        self.assertEqual(len(fixture.scope_bindings), 1)
        self.assertEqual(fixture.events.count('canonical_export'), 1)
        self.assertEqual(fixture.events.count('host_stopped'), 1)
        self.assertEqual(fixture.events[-1], 'fixture_cleanup')
        self.assertTrue(result['local_cleanup_complete'])
        self.assertTrue(Path(result['phases'][0]['canonical_evidence']).is_file())
        first, second = [runner.read_json(Path(p['risk_authorization']))['authorization_id']
                         for p in fixture.document['phases']]
        self.assertTrue((fixture.options.ledger_dir/f'authorization-{first}.spent').is_file())
        self.assertFalse((fixture.options.ledger_dir/f'authorization-{second}.spent').exists())
        self.assertFalse(fixture.token.exists())
        active = fixture.options.ledger_dir/f'simulator-{str(UUID(fixture.options.simulator_id)).upper()}.active.json'
        self.assertFalse(active.exists())
        self.assertEqual(result['memory_semantics'], 'not_evaluated')

    async def test_actual_bound_clock_failure_shape_stops_before_second_grant_or_host(self):
        fixture, result = await self.run_fixture('capture')
        self.assert_first_phase_failure_cleanly_stops_sequence(fixture, result)
        phase = result['phases'][0]
        failure = phase['native_capture_failure']
        self.assertEqual(failure['failure_code'], 'memory_readiness_unavailable')
        self.assertEqual(failure['source_clock']['status'], 'bound')
        self.assertEqual(failure['source_clock']['anchor_sha256'], phase['source_anchor']['sha256'])
        self.assertEqual(failure['observation_stage'], 'during_capture_wait')
        self.assertEqual(failure['failure_fields'], [
            {'field': 'utterances[1].failure_code', 'failure_code': 'memory_readiness_unavailable'}])
        self.assertTrue(any(x['path'].endswith('/capture.json') for x in phase['native_output']))

    async def test_failure_first_observed_after_stop_still_blocks_next_admission(self):
        fixture, result = await self.run_fixture('stop')
        self.assert_first_phase_failure_cleanly_stops_sequence(fixture, result)
        phase = result['phases'][0]
        self.assertNotIn('native_capture_failure', phase)
        failure = phase['native_capture_failure_after_stop']
        self.assertEqual(failure['failure_code'], 'memory_readiness_unavailable')
        self.assertEqual(failure['observation_stage'], 'cleanup_after_termination_attempt')

    async def test_successful_capture_continues_without_semantic_pass(self):
        fixture, result = await self.run_fixture(None)
        self.assertEqual(result['status'], 'lifecycle_complete', result)
        self.assertEqual((fixture.prepares, fixture.launches, fixture.cleanups), (1, 2, 1))
        self.assertEqual(len(result['phases']), 2)
        self.assertEqual(fixture.events.count('host_stopped'), 2)
        self.assertEqual(fixture.events.count('canonical_export'), 2)
        for phase in result['phases']:
            self.assertEqual(phase['status'], 'session_complete')
            self.assertNotIn('native_capture_failure', phase)
            self.assertTrue((fixture.options.ledger_dir/f"authorization-{phase['authorization_id']}.spent").exists())
        self.assertEqual(result['voice_qualification']['status'], 'not_evaluated')
        self.assertEqual(result['memory_semantics'], 'not_evaluated')


class TerminalFailureScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'capture.json'
        self.scope = {'run_id': uuid4().hex, 'replay_invocation_id': str(uuid4()),
                      'phase_start_seconds': 0, 'anchor_sha256': 'a'*64}
        self.capture = terminal_memory_failure({
            'run_id': str(UUID(self.scope['run_id'])), 'replay_invocation_id': self.scope['replay_invocation_id'],
            'replay_source_clock': {'version': 1, 'status': 'bound', 'failure_code': None,
                'phase_start_seconds': 0, 'anchor_sha256': self.scope['anchor_sha256']}})

    def classify(self, capture=None, scope=None):
        self.path.write_text(json.dumps(capture or self.capture))
        return runner.retained_capture_failure(self.path, scope or self.scope, observation_stage='offline_regression')

    def test_origin_hash_is_not_trusted_before_anchor_validation(self):
        self.assertIsNone(self.classify(scope={**self.scope, 'anchor_sha256': None}))
        self.assertEqual(self.classify()['failure_code'], 'memory_readiness_unavailable')

    def test_wrong_run_invocation_phase_anchor_or_protocol_fails_closed(self):
        changes = (('run_id', str(uuid4())), ('replay_invocation_id', str(uuid4())),
                   ('phase_start_seconds', 1), ('phase_start_seconds', True),
                   ('anchor_sha256', 'b'*64), ('version', 2), ('version', True))
        for key, value in changes:
            capture = copy.deepcopy(self.capture)
            target = capture if key in {'run_id', 'replay_invocation_id'} else capture['replay_source_clock']
            target[key] = value
            with self.subTest(field=key), self.assertRaisesRegex(Exception, 'retained_failure_capture_scope_mismatch'):
                self.classify(capture)

    def test_terminal_failure_without_specific_code_is_not_success(self):
        capture = copy.deepcopy(self.capture)
        capture['utterances'] = []
        self.assertEqual(self.classify(capture)['failure_code'], 'native_capture_failed')

    def test_distinct_failure_codes_do_not_guess_primary_cause(self):
        capture = copy.deepcopy(self.capture)
        capture['callback_failure'] = 'callback_failed'
        result = self.classify(capture)
        self.assertEqual(result['failure_code'], 'native_capture_failed')
        self.assertEqual(len(result['failure_fields']), 2)

    def test_nonterminal_and_successful_status_are_not_failures(self):
        for status in ('Running replay', 'Evaluation replay complete.'):
            with self.subTest(status=status):
                capture = {**self.capture, 'final_ui_status': status}
                self.assertIsNone(self.classify(capture))

    def test_private_text_cannot_be_promoted_to_failure_code(self):
        capture = copy.deepcopy(self.capture)
        capture['utterances'][1]['failure_code'] = 'private text / not a code'
        with self.assertRaisesRegex(Exception, 'retained_failure_capture_code_invalid'):
            self.classify(capture)

    def test_failure_utterance_inventory_stays_bounded(self):
        capture = copy.deepcopy(self.capture)
        capture['utterances'] = capture['utterances'] * 7
        with self.assertRaisesRegex(Exception, 'retained_failure_capture_utterance_bound'):
            self.classify(capture)


if __name__ == '__main__':
    unittest.main()
