"""Reporting distinctions only; fixtures never establish real media acceptance."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.responsive_voice.native_replay_run import (
    read_json, run_once, unassessed_voice_qualification,
)
import test_native_replay_run as fixtures


class NativeVoiceQualificationStatusTests(unittest.IsolatedAsyncioTestCase):
    def assert_unassessed(self, report):
        self.assertIs(report['voice_pass'], False)
        self.assertEqual(report['voice_qualification'], {
            'status':'not_evaluated',
            'reason':'lifecycle_runner_does_not_evaluate_voice_acceptance',
            'voice_pass_semantics':'legacy_unqualified_sentinel_not_failure_verdict',
        })

    def test_metadata_is_fresh_and_never_claims_an_acceptance_verdict(self):
        first = unassessed_voice_qualification()
        first['status'] = 'foreign_mutation'
        self.assertEqual(unassessed_voice_qualification()['status'], 'not_evaluated')
        self.assertNotIn('passed', unassessed_voice_qualification())

    async def test_one_question_prepared_baseline_is_supported_but_not_self_qualified(self):
        for route in ('realtime', 'live'):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as directory:
                fixture = fixtures.Fixture(Path(directory))
                fixture.prepared('baseline', cuts=0, duration=45)
                if route == 'live':
                    fixture.live_baseline()
                plan = read_json(fixture.options.input_manifest)
                self.assertEqual(len(plan['utterances']), 1)
                # Controlled reader output tests reporting separation, not a
                # valid real capture or a semantic/audio quality oracle.
                evidence = {'evidence_kind':'OFFLINE_READER_RESULT_FIXTURE',
                    'capture_verdict':'structurally_valid_evidence',
                    'occurrences':[{'input_verdict':'sdk_completed_active_input'}],
                    'timing_blockers':[], 'answer_quality_judged':False,
                    'hearing_measured':False}
                with patch('evals.responsive_voice.native_capture_evidence.read_capture',
                           return_value=evidence) as reader:
                    report = await run_once(fixture.options, host=fixture, process=fixture)
                self.assertEqual(report['status'], 'lifecycle_complete')
                self.assertEqual((fixture.prepares, fixture.launches, fixture.cleanups), (1, 1, 1))
                self.assertEqual(len(fixture.plan['utterances']), 1)
                self.assertTrue(report['local_cleanup_complete'])
                reader.assert_called_once()
                self.assert_unassessed(report)
                self.assert_unassessed(read_json(fixture.options.artifact_dir/'native-run.json'))
                self.assertEqual(read_json(fixture.options.artifact_dir/'native-evidence-report.json'), evidence)

    async def test_actual_lifecycle_failure_remains_separate_from_unassessed_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = fixtures.Fixture(Path(directory))
            fixture.fail_launch = True
            report = await run_once(fixture.options, host=fixture, process=fixture)
            self.assertEqual(report['status'], 'failed')
            self.assertIn('fatal', report)
            self.assertEqual(fixture.launches, 1)
            self.assert_unassessed(report)

    async def test_capture_analysis_failure_is_preserved_not_converted_to_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = fixtures.Fixture(Path(directory))
            report = await run_once(fixture.options, host=fixture, process=fixture)
            self.assertEqual(report['status'], 'lifecycle_complete')
            self.assertIn('capture_analysis_error', report)
            self.assert_unassessed(report)


if __name__ == '__main__':
    unittest.main()
