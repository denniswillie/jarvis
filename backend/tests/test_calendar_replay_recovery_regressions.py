"""Offline regressions for the first actual calendar replay harness failures."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import test_native_memory_readiness_controls as controls
from evals.responsive_voice.native_memory_readiness_control import approved_deadline, RECEIPT_NAME
from evals.responsive_voice.native_replay_run import capture_replay_controls
from evals.responsive_voice.native_capture_evidence import analyze_capture, EvidenceError
from evals.responsive_voice.test_native_capture_evidence import fixture


def native_milliseconds(value):
    return value.replace(microsecond=0) + timedelta(milliseconds=(value.microsecond + 500) // 1000)


class CalendarWireDeadlineTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = controls.MemoryControlTests.asyncSetUp
    source_row = controls.MemoryControlTests.source_row
    put_request = controls.MemoryControlTests.put_request
    consume = controls.MemoryControlTests.consume

    def native_request(self, fraction=.068655):
        self.anchor['utc_unix_seconds'] = int(self.now.timestamp()) - 20 + fraction
        controls.private_json(self.operator/'source-clock.json', self.anchor)
        issued = datetime.fromtimestamp(self.anchor['utc_unix_seconds'] + 20, UTC)
        expires = datetime.fromtimestamp(self.anchor['utc_unix_seconds'] + 30, UTC)
        self.request = replace(self.request, issued_at=native_milliseconds(issued),
            expires_at=native_milliseconds(expires))
        self.observer.clock = lambda: issued + timedelta(seconds=5)
        return issued, expires

    async def test_actual_345_microsecond_native_wire_difference_reaches_observer(self):
        issued, _ = self.native_request()
        self.assertEqual(self.request.issued_at - issued, timedelta(microseconds=345))
        _, data, receipt = await self.consume()
        self.assertEqual(receipt['readiness']['status'], 'ready', receipt)
        self.assertEqual(receipt['request_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt['readiness']['source_clock']['deadline_uptime_seconds'], 130.1)
        self.assertEqual(self.observer.query_count, 9)

    async def test_native_rounding_across_second_keeps_original_uptime(self):
        self.native_request(.999700)
        _, _, receipt = await self.consume()
        self.assertEqual(receipt['readiness']['status'], 'ready', receipt)
        self.assertEqual(receipt['readiness']['source_clock']['arm_uptime_seconds'], 120.1)
        self.assertEqual(receipt['readiness']['source_clock']['deadline_uptime_seconds'], 130.1)

    async def test_arbitrary_microsecond_or_millisecond_shift_is_not_tolerance(self):
        issued, expires = self.native_request()
        changes = ({'issued_at': issued + timedelta(microseconds=1)},
                   {'expires_at': expires + timedelta(microseconds=1)},
                   {'issued_at': self.request.issued_at + timedelta(milliseconds=1)},
                   {'expires_at': self.request.expires_at + timedelta(milliseconds=1)},
                   {'expires_at': self.request.expires_at.replace(microsecond=0)})
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'memory_original_deadline_mismatch'):
                approved_deadline(self.config, self.observer.scope, self.anchor,
                    replace(self.request, **change), self.sample)
        self.assertEqual(self.pool.queries, [])

    async def test_rounded_expiry_never_extends_entry_deadline(self):
        self.native_request()
        self.sample['uptime_seconds'] = 130.10001
        _, _, receipt = await self.consume()
        self.assertEqual(receipt['readiness']['status'], 'expired')
        self.assertEqual(self.pool.queries, [])

    async def test_rounded_expiry_never_extends_publication_deadline(self):
        import asyncio
        self.native_request()
        self.pool.pause = asyncio.Event()
        self.put_request()
        task = asyncio.create_task(self.bridge.step())
        await self.pool.entered.wait()
        self.sample['uptime_seconds'] = 130.10001
        self.pool.pause.set()
        await task
        receipt = json.loads((self.controls/RECEIPT_NAME).read_bytes())
        self.assertEqual(receipt['readiness']['status'], 'expired')

    async def test_new_nonce_cannot_move_rounded_original_deadline(self):
        self.native_request()
        await self.consume()
        self.put_request(expires_at=self.request.expires_at + timedelta(milliseconds=1))
        with self.assertRaisesRegex(ValueError, 'memory_original_deadline_mismatch'):
            await self.bridge.step()
        self.assertEqual(self.observer.query_count, 9)


class CalendarControlCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controls = self.root/'controls'
        self.controls.mkdir(mode=0o700)
        self.output = self.root/'output'
        self.output.mkdir(mode=0o700)

    def write(self, name, data=b'{}'):
        path = self.controls/name
        path.write_bytes(data)
        path.chmod(0o600)

    def test_memory_only_never_requires_task_replay_report(self):
        self.write('memory-readiness-request.json')
        report = {'memory_readiness': {}}
        capture_replay_controls(report, self.controls, self.output)
        self.assertNotIn('task_control_capture_error', report)
        self.assertNotIn('task_replay', report)
        records = report['memory_readiness']['observations']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['sha256'], hashlib.sha256(b'{}').hexdigest())

    def test_task_and_memory_controls_have_separate_destinations(self):
        for name in ('session-binding.json', 'task-barrier.json', 'stimulus-completion.json',
                     'memory-readiness-request.json', 'memory-readiness-receipt.json'):
            self.write(name)
        report = {'task_replay': {}, 'memory_readiness': {}}
        capture_replay_controls(report, self.controls, self.output)
        self.assertEqual(len(report['task_replay']['observations']), 3)
        self.assertEqual(len(report['memory_readiness']['observations']), 2)
        self.assertTrue(all('/task-replay-observations/' in r['path'] for r in report['task_replay']['observations']))
        self.assertTrue(all('/memory-readiness-observations/' in r['path'] for r in report['memory_readiness']['observations']))

    def test_undeclared_control_scope_is_not_invented(self):
        self.write('memory-readiness-request.json')
        report = {}
        capture_replay_controls(report, self.controls, self.output)
        self.assertEqual(report, {})
        self.assertEqual(list(self.output.iterdir()), [])

    def test_memory_file_bound_failure_does_not_masquerade_as_task_error(self):
        self.write('memory-readiness-request.json', b'x' * 16385)
        report = {'memory_readiness': {}}
        capture_replay_controls(report, self.controls, self.output)
        self.assertIn('memory_control_capture_error', report)
        self.assertNotIn('task_control_capture_error', report)


class CalendarCaptureReaderTests(unittest.TestCase):
    def gate(self, capture):
        capture['utterances'][1]['trigger'] = {'kind': 'after_manual_response',
            'after_input_ordinal': 0, 'at_seconds': 15, 'timeout_seconds': 22,
            'canonical_requirement': 'retain_searchable', 'playback_requirement': 'authenticated_buffer_stop'}
        return capture['utterances'][1]

    def test_manual_gate_is_structural_not_memory_or_hearing_acceptance(self):
        capture, pcm, _ = fixture(2)
        self.gate(capture)
        result = analyze_capture(capture, pcm)
        self.assertEqual(result['occurrences'][1]['input_verdict'], 'sdk_completed_active_input')
        self.assertFalse(result['answer_quality_judged'])
        self.assertEqual(result['occurrences'][1]['input_end_to_annotated_useful_onset']['verdict'], 'not_annotated')

    def test_unstarted_manual_gate_never_gains_input_delivery(self):
        capture, pcm, _ = fixture(2)
        item = self.gate(capture)
        item.update(status='not_started', failure_code='run_closed_before_injection',
            attempted_clip_samples=0, sdk_accepted_clip_samples=0, callbacks=[])
        for key in ('triggered_uptime_seconds', 'enqueue_requested_uptime_seconds',
                    'first_sdk_callback_start_uptime_seconds', 'last_sdk_callback_end_uptime_seconds',
                    'first_successful_activity_uptime_seconds', 'last_successful_activity_uptime_seconds'):
            item[key] = None
        result = analyze_capture(capture, pcm)
        self.assertEqual(result['occurrences'][1]['input_verdict'], 'not_started')
        self.assertIsNone(result['occurrences'][1]['speech_onset_uptime_seconds'])

    def test_manual_trigger_stays_bounded_and_prior_only(self):
        changes = ({'after_input_ordinal': 1}, {'timeout_seconds': 121},
                   {'at_seconds': 179}, {'canonical_requirement': 'invented'},
                   {'playback_requirement': 'decoded_drain'}, {'extra': 'not_allowed'})
        for change in changes:
            capture, pcm, _ = fixture(2)
            item = self.gate(capture)
            item['trigger'].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(EvidenceError, 'invalid_manual_response_trigger'):
                analyze_capture(capture, pcm)


if __name__ == '__main__':
    unittest.main()
