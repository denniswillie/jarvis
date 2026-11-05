"""Offline retained deadlines and terminal capture detection; no native or SQL."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from evals.responsive_voice import native_replay_run as runner

RUN = '1c9e97b1dc08464ea6b5fc0419fbfa44'
INVOCATION = 'ced307a6-9f92-4106-b8c5-661ae3451ce2'
ANCHOR = '729759cf6615736a84d6f18ad6646f8bf4da6c3d38b39a2d72eecf604eafcd25'
SCOPE = dict(run_id=RUN, replay_invocation_id=INVOCATION,
             phase_start_seconds=67, anchor_sha256=ANCHOR)


def failure_capture():
    return dict(run_id=str(UUID(RUN)), replay_invocation_id=INVOCATION,
        clock_anchor=None, final_ui_status='Evaluation replay capture failed.',
        replay_source_clock=dict(version=1, status='failed', failure_code='sourceClockLate',
            phase_start_seconds=67, anchor_sha256=ANCHOR))


class TerminalCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.capture = self.root/'capture.json'
        self.anchor = self.root/'source-clock.json'

    def write_capture(self, value):
        self.capture.write_text(json.dumps(value))

    def read_failure(self):
        return runner.retained_capture_failure(self.capture, SCOPE, observation_stage='during_capture_wait')

    def test_matching_terminal_failure_preserves_native_code_and_uuid_identity(self):
        self.write_capture(failure_capture())
        result = self.read_failure()
        self.assertEqual(result['failure_code'], 'sourceClockLate')
        self.assertEqual(result['source_clock']['anchor_sha256'], ANCHOR)
        self.assertEqual(result['observation_stage'], 'during_capture_wait')
        self.assertEqual(len(result['sha256']), 64)

    def test_wrong_scope_or_future_protocol_is_never_the_expected_failure(self):
        mutations = [('run_id',str(uuid4())),('replay_invocation_id',str(uuid4())),
            ('version',2),('version',True),('phase_start_seconds',68),
            ('anchor_sha256','0'*64),('phase_start_seconds',True)]
        for key,value in mutations:
            with self.subTest(field=key,value=value):
                capture=failure_capture()
                target=capture if key in {'run_id','replay_invocation_id'} else capture['replay_source_clock']
                target[key]=value
                self.write_capture(capture)
                with self.assertRaisesRegex(Exception,'retained_failure_capture_scope_mismatch'):
                    self.read_failure()

    def test_nonterminal_successful_or_unbound_capture_is_not_a_scoped_terminal_failure(self):
        for location,key,value in [('top','final_ui_status','Running replay'),
                                   ('clock','status','pending'),
                                   ('top','final_ui_status','Evaluation replay complete.')]:
            with self.subTest(value=value):
                capture=failure_capture()
                (capture if location=='top' else capture['replay_source_clock'])[key]=value
                self.write_capture(capture)
                self.assertIsNone(self.read_failure())

    def test_missing_capture_is_not_failure_evidence(self):
        self.assertIsNone(self.read_failure())

    def test_failure_code_cannot_inject_private_text_into_diagnostics(self):
        capture=failure_capture();capture['replay_source_clock']['failure_code']='private payload / invalid'
        self.write_capture(capture)
        with self.assertRaisesRegex(Exception,'retained_failure_capture_code_invalid'):
            self.read_failure()

    async def test_existing_failure_wins_before_starting_an_anchor_wait(self):
        self.write_capture(failure_capture())
        class Process:
            async def wait_capture(self,*args):
                raise AssertionError('terminal evidence must win before waiting')
        observations=[]
        with self.assertRaises(runner.RetainedCaptureFailure) as raised:
            await runner.wait_retained_artifact(Process(),self.anchor,1,
                capture_path=self.capture,scope=SCOPE,observations=observations)
        self.assertEqual(raised.exception.code,'sourceClockLate')
        self.assertEqual(len(observations),1)
        self.assertFalse(self.anchor.exists())

    async def test_failure_appearing_during_wait_cancels_and_joins_waiter(self):
        owner=self
        class Process:
            joined=False
            async def wait_capture(self,*args):
                try:
                    owner.write_capture(failure_capture())
                    await asyncio.Event().wait()
                finally:self.joined=True
        process=Process();observations=[]
        with self.assertRaises(runner.RetainedCaptureFailure):
            await runner.wait_retained_artifact(process,self.anchor,1,
                capture_path=self.capture,scope=SCOPE,observations=observations)
        self.assertTrue(process.joined)

    async def test_nonterminal_file_does_not_mask_missing_anchor_timeout(self):
        capture=failure_capture();capture['final_ui_status']='Running replay'
        self.write_capture(capture);observations=[]
        with self.assertRaises(TimeoutError):
            await runner.wait_retained_artifact(runner.ProcessBoundary(),self.anchor,.01,
                capture_path=self.capture,scope=SCOPE,observations=observations)
        self.assertFalse(observations[-1]['terminal_failure_observed'])

    async def test_late_failure_is_not_claimed_to_have_existed_during_wait(self):
        observations=[]
        with self.assertRaises(TimeoutError):
            await runner.wait_retained_artifact(runner.ProcessBoundary(),self.anchor,.01,
                capture_path=self.capture,scope=SCOPE,observations=observations)
        self.write_capture(failure_capture())
        result=runner.retained_capture_failure(self.capture,SCOPE,
            observation_stage='after_app_stop')
        self.assertEqual(result['observation_stage'],'after_app_stop')
        self.assertFalse(observations[-1]['terminal_failure_observed'])

    async def test_normal_anchor_wait_remains_supported(self):
        self.anchor.write_text('{}')
        observations=[]
        await runner.wait_retained_artifact(runner.ProcessBoundary(),self.anchor,.1,
            capture_path=self.capture,scope=SCOPE,observations=observations)
        self.assertEqual(observations,[])


class SourceDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.boot=str(uuid4());self.origin=str(uuid4())
        self.anchor=dict(format='jarvis.replay.source_clock.v1',run_id=RUN,
            origin_invocation_id=self.origin,monotonic_before_seconds=100.0,
            monotonic_after_seconds=100.2,utc_unix_seconds=1700000000.0,boot_session_id=self.boot)
        self.path=self.root/'source-clock.json'
        self.retained=dict(index=1,scope=dict(run_id=RUN),phases=[dict(invocation_id=self.origin),
            dict(invocation_id=INVOCATION,phase_start_seconds=67)])
        self.plan=dict(duration_seconds=28,utterances=[dict(trigger=dict(kind='at_time',at_seconds=8))],
                       replay_invocation_id=INVOCATION)
        self.store_anchor()

    def store_anchor(self):
        raw=json.dumps(self.anchor).encode();self.path.write_bytes(raw)
        checksum=runner.scenarios.digest(raw)
        self.retained['source_anchor']=dict(path=str(self.path),sha256=checksum)
        self.plan['replay_source_clock']=dict(version=1,phase_start_seconds=67,
            anchor_file='source-clock.json',anchor_sha256=checksum)

    def sample(self, uptime):
        return dict(clock='ProcessInfo.systemUptime',boot_session_id=self.boot,uptime_seconds=uptime)

    def assess(self, uptime):
        return runner.retained_source_deadline(self.plan,self.retained,sample=self.sample(uptime))

    def test_original_fixed_deadline_expired_not_rebased(self):
        result=self.assess(176.723597)
        self.assertEqual(result['status'],'expired')
        self.assertAlmostEqual(result['deadline_uptime_seconds'],175.1)
        self.assertEqual(self.plan['utterances'][0]['trigger']['at_seconds'],8)
        self.assertEqual(self.retained['phases'][1]['phase_start_seconds'],67)

    def test_exact_deadline_is_expired(self):
        self.assertEqual(self.assess(175.1)['status'],'expired')

    def test_future_stimulus_is_not_a_startup_fit_claim(self):
        result=self.assess(175.0)
        self.assertEqual(result['status'],'not_yet_expired')
        self.assertIn('no UTC equality or startup-fit guarantee',result['meaning'])

    def test_future_anchor_is_rejected(self):
        with self.assertRaisesRegex(Exception,'retained_source_clock_future_anchor'):
            self.assess(100.1)

    def test_unknown_or_different_clock_is_rejected(self):
        for sample in ({},dict(clock='python.monotonic',boot_session_id=self.boot,uptime_seconds=176),
                       dict(clock='ProcessInfo.systemUptime',boot_session_id=self.boot,uptime_seconds=True)):
            with self.subTest(sample=sample),self.assertRaisesRegex(Exception,'retained_source_clock_unavailable'):
                runner.retained_source_deadline(self.plan,self.retained,sample=sample)

    def test_other_boot_is_rejected(self):
        sample=self.sample(176);sample['boot_session_id']=str(uuid4())
        with self.assertRaisesRegex(Exception,'retained_source_clock_boot_mismatch'):
            runner.retained_source_deadline(self.plan,self.retained,sample=sample)

    def test_wrong_anchor_scope_or_future_version_is_rejected(self):
        original=copy.deepcopy(self.anchor)
        for key,value in [('format','jarvis.replay.source_clock.v2'),('run_id',uuid4().hex),
                          ('origin_invocation_id',str(uuid4()))]:
            with self.subTest(field=key):
                self.anchor={**original,key:value};self.store_anchor()
                with self.assertRaisesRegex(Exception,'retained_source_anchor_identity_mismatch'):
                    self.assess(176)

    def test_changed_anchor_bytes_are_rejected(self):
        self.path.write_bytes(b'{}')
        with self.assertRaisesRegex(Exception,'retained_source_anchor_changed'):
            self.assess(176)

    def test_mismatched_manifest_clock_binding_is_rejected(self):
        self.plan['replay_source_clock']['anchor_sha256']='f'*64
        with self.assertRaisesRegex(Exception,'retained_source_clock_plan_mismatch'):
            self.assess(176)

    def test_utc_value_does_not_choose_deadline(self):
        one=self.assess(174)
        self.anchor['utc_unix_seconds']=1900000000;self.store_anchor()
        two=self.assess(174)
        self.assertEqual(one['deadline_uptime_seconds'],two['deadline_uptime_seconds'])
        self.assertEqual(one['status'],two['status'])

    def test_invalid_sampling_bracket_is_rejected(self):
        self.anchor['monotonic_after_seconds']=102;self.store_anchor()
        with self.assertRaisesRegex(Exception,'source_anchor_sampling_uncertainty'):
            self.assess(176)

    def test_sampled_clock_is_requested_only_when_not_injected(self):
        with patch.object(runner,'sample_replay_source_clock',return_value=self.sample(174)) as clock:
            self.assertEqual(runner.retained_source_deadline(self.plan,self.retained)['status'],'not_yet_expired')
        clock.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
