"""Claim-gated cancellation stimuli; no semantic/tool/physical acceptance."""
from copy import deepcopy
from dataclasses import replace
import unittest

from evals.responsive_voice import native_interruption_scenarios as compiler
from evals.responsive_voice import test_native_interruption_scenarios as fixtures


class TaskCancelCompilationTests(unittest.TestCase):
    def setUp(self):
        fixtures.ScenarioTests.setUp(self)
        self.clips['conversation/post_cancel_question'] = replace(self.clips['conversation/continue_answer'],
            source_id='post_cancel_question')

    compile = fixtures.ScenarioTests.compile

    def cancellation(self, **options):
        return self.compile('explicit_cancel',1,task_aware=True,distinct_provider_turns=True,**options)

    def test_request_claim_cancel_and_two_part_probe_are_distinct_inputs(self):
        result=self.cancellation()
        rows=result.expectations['occurrences']
        self.assertEqual([row['intent'] for row in rows],[
            'request_one_bounded_read_only_background_task','query_setup_task_status_without_assuming_admission',
            'cancel_same_task_idempotently','canonical_status_and_unrelated_question_without_task_restart'])
        self.assertEqual([row['trigger']['kind'] for row in rows],['at_time','after_task_claim',
            'response_output_start','after_provider_input_finalized'])
        self.assertTrue(rows[2]['trigger']['require_task_claim'])
        self.assertTrue(all('require_task_claim' not in row['trigger'] for row in rows[3:]))
        for i in (3,):
            self.assertEqual(rows[i]['trigger']['after_input_ordinal'],i-1)
            self.assertAlmostEqual(rows[i]['latest_nominal_start_seconds'],
                rows[i-1]['latest_nominal_start_seconds']+rows[i-1]['input_duration_seconds']+10)
        self.assertEqual(sum(row['planned_cut'] for row in rows),1)
        self.assertNotIn('baseline/current_view',[row['source_clip'] for row in rows])
        self.assertEqual(result.manifest['frames'],self.compile(task_aware=True).manifest['frames'])

    def test_cancellation_is_not_pause_or_canonical_success(self):
        result=self.cancellation()
        expected=result.expectations
        self.assertEqual(expected['case_revision'],9)
        self.assertIsNone(expected['actual_cuts'])
        self.assertFalse(expected['task_scope_staged'])
        self.assertFalse(expected['late_callback_injection_supported'])
        self.assertEqual(expected['status_probe_ordinal'],expected['conversation_probe_ordinal'])
        self.assertTrue(expected['status_and_conversation_probe_share_user_turn'])
        self.assertTrue(all(row['verdict']=='unmeasured' for row in expected['assertions']))
        self.assertIn('cancelled_held_attempt',expected['result_control_expectation'])
        self.assertEqual(expected['prerequisites'][0]['setup_occurrence_id'],expected['occurrences'][0]['occurrence_id'])
        self.assertEqual(expected['occurrences'][2]['source_clip'],'conversation/explicit_cancel')
        self.assertNotIn('conversation/speech_only_pause',[row['source_clip'] for row in expected['occurrences']])
        self.assertNotIn('canonical_cancel',str(result.manifest))

    def test_no_implicit_escalation_or_unbounded_new_variant(self):
        for changes in ({'distinct_provider_turns':False},{'distinct_provider_turns':1},
                        {'cuts':2},{'cuts':3},{'duration':120.001},
                        {'binding':{**self.binding,'voice_route':'live'}}):
            with self.subTest(changes=changes):
                options=dict(case='explicit_cancel',cuts=1,task_aware=True,distinct_provider_turns=True)
                options.update(changes)
                with self.assertRaises(compiler.ScenarioError): self.compile(**options)

    def test_missing_probe_or_large_input_fails_instead_of_fake_conversation(self):
        clips=deepcopy(self.clips)
        clips.pop('conversation/post_cancel_question')
        with self.assertRaisesRegex(compiler.ScenarioError,'post_cancel_question_clip_required'):
            self.cancellation(clips=clips)
        clips=deepcopy(self.clips)
        clips['conversation/post_cancel_question']=replace(clips['conversation/post_cancel_question'],sample_count=61*48000)
        with self.assertRaises(compiler.ScenarioError): self.cancellation(clips=clips)

    def test_existing_pause_and_legacy_cancel_are_unchanged(self):
        for cuts in (1,2,3):
            with self.subTest(cuts=cuts):
                pause=self.compile(cuts=cuts,task_aware=True,distinct_provider_turns=True)
                self.assertEqual(pause.expectations['case_revision'],5)
                self.assertEqual(len(pause.manifest['utterances']),2+2*cuts)
                old=compiler._compile_legacy_scenario('explicit_cancel',cuts=cuts,binding=self.binding,
                    frame_plan=self.frames,frame_root=self.root,clips=self.clips)
                self.assertEqual(old,self.compile('explicit_cancel',cuts))
