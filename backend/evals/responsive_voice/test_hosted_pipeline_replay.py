"""Offline gates only: no external services, assets, credentials or SQL."""
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.visual_memory.gemini_client import GeminiVisualAnalyzerConfig
from evals.responsive_voice.hosted_pipeline_replay import (
    CASE_IDS, REVIEW, ExactWireTransport, Limits, SharedModelGate, consume_grant,
    create_gemini_gate, grant_template, load_payload, validate_grant,
    run_pair,
)
from evals.responsive_voice.video_replay import (
    FrameAsset, PermissionRecord, ReplayCase, ReplayInputs,
)


def make_payload(root, *, offsets=(0, 20, 40)):
    (root / 'cases').mkdir()
    frames = []
    for i, offset in enumerate(offsets):
        content = f'controlled image fixture {i}'.encode()
        name = f'opaque-{i}.jpg'
        (root / name).write_bytes(content)
        frames.append(FrameAsset(offset_ms=offset, relative_path=name,
            sha256=hashlib.sha256(content).hexdigest(), width=1, height=1))
    permission = PermissionRecord(evidence='Repository-authored synthetic fixture',
        local_evaluation=True, retain_derived_results=True)
    for name in CASE_IDS:
        case = ReplayCase(case_id=name, permission=permission,
            inputs=ReplayInputs(duration_ms=offsets[-1]+1,
                initial_context=('Observe the controlled target ' + name.rsplit('-', 1)[-1],),
                frames=tuple(frames)), expected_events=('EVALUATOR_SECRET_LABEL',),
            forbidden_claims=('EVALUATOR_SECRET_FORBIDDEN',))
        (root / 'cases' / (name + '.json')).write_text(case.model_dump_json())
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'asset_root': str(root),
        'cases': [{'id': 'q03', 'frames': [{'sha256': f.sha256} for f in frames]}]}))
    return load_payload(manifest)


def approved(payload):
    grant = grant_template(payload)
    grant.update(hosted_inference=True, approved_by='controlled test', approval_reference='offline fixture',
        permission_reference='synthetic fixture', provider_policy_reference='no external transport',
        expires_at=(datetime.now(UTC)+timedelta(minutes=5)).isoformat())
    return grant


def released_frame(payload, index=0):
    asset = payload.cases[0].inputs.frames[index]
    return SimpleNamespace(frame_id=uuid4(), offset_ms=asset.offset_ms,
        captured_at=datetime.now(UTC), image_bytes=(payload.asset_root / asset.relative_path).read_bytes())


def image(frame):
    return SimpleNamespace(image_bytes=frame.image_bytes, label='frame:0', content_type='image/jpeg')


class HostedPipelinePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = make_payload(self.root)

    def test_prepare_is_exact_pair_and_grant_is_not_permission(self):
        self.assertEqual(tuple(c.case_id for c in self.payload.cases), CASE_IDS)
        self.assertEqual(len(self.payload.image_hashes), 3)
        with self.assertRaisesRegex(ValueError, 'hosted_authorization_missing'):
            validate_grant(self.payload, grant_template(self.payload))

    def test_old_grant_format_is_not_reusable(self):
        grant = approved(self.payload)
        grant['format'] = 'jarvis.pov.narrow_hosted_permission.v1'
        with self.assertRaisesRegex(ValueError, 'fresh_pipeline_grant_required'):
            validate_grant(self.payload, grant)

    def test_fresh_exact_grant_consumed_once(self):
        grant = approved(self.payload)
        receipt = consume_grant(self.payload, grant, self.root / 'ledger')
        self.assertTrue(receipt.marker.exists())
        with self.assertRaises(FileExistsError):
            consume_grant(self.payload, grant, self.root / 'ledger')

    def test_changed_context_or_budget_or_expiry_rejected(self):
        for change in ('context', 'budget', 'expired', 'store', 'other_context'):
            with self.subTest(change=change):
                grant = approved(self.payload)
                if change == 'context': grant['context_sha256s']['q03-general'] = 'changed'
                if change == 'budget': grant['limits']['concurrency'] = 2
                if change == 'expired': grant['expires_at'] = '2020-01-01T00:00:00+00:00'
                if change == 'store': grant['request_store'] = True
                if change == 'other_context': grant['other_personal_context'] = True
                with self.assertRaises(ValueError): validate_grant(self.payload, grant)

    def test_main_review_is_new_and_not_keyword_scoring(self):
        self.assertIn('cutting existing strips', REVIEW['allowed_paraphrases'])
        self.assertIn('The word strips alone is not a contradiction.', REVIEW['scoring_rules'])
        self.assertIn('Do not require the word dice.', REVIEW['scoring_rules'])
        self.assertEqual(REVIEW['historical_labels_and_results'], 'unchanged')

    def test_changed_bytes_fail_before_execution(self):
        (self.root / 'opaque-0.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError): load_payload(self.root / 'manifest.json')


class HostedModelBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.payload = make_payload(Path(self.temp.name))

    def hosted_gate(self, grant):
        return SharedModelGate(SimpleNamespace(provider='google_gemini', model=grant['model']),
                               execution_kind='hosted_model')

    async def test_callable_hosted_run_requires_consumption_before_fixture_or_model(self):
        grant = approved(self.payload)
        gate = self.hosted_gate(grant)
        with patch('evals.responsive_voice.hosted_pipeline_replay.source_fixture') as fixture:
            with self.assertRaisesRegex(ValueError, 'consumed_grant_receipt_required'):
                await run_pair(self.payload, gate, dsn='never-connect',
                    output_dir=Path(self.temp.name) / 'unadmitted', grant=grant)
        fixture.assert_not_called()
        self.assertEqual(gate.calls, [])

    async def test_callable_failed_setup_keeps_one_use_execution_claim(self):
        grant = approved(self.payload)
        root = Path(self.temp.name)
        receipt = consume_grant(self.payload, grant, root / 'ledger')
        entered = []

        @asynccontextmanager
        async def unavailable(*args, **kwargs):
            entered.append(True)
            raise RuntimeError('controlled_setup_failure')
            yield  # pragma: no cover - makes the failure an async context manager

        gate = self.hosted_gate(grant)
        with patch('evals.responsive_voice.hosted_pipeline_replay.source_fixture', unavailable):
            result = await run_pair(self.payload, gate, dsn='never-connect', output_dir=root / 'first',
                                    grant=grant, admission=receipt)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(Path(result['grant_execution_marker']).exists())
            with self.assertRaises(FileExistsError):
                await run_pair(self.payload, gate, dsn='never-connect', output_dir=root / 'retry',
                               grant=grant, admission=receipt)
        self.assertEqual(entered, [True])
        self.assertEqual(gate.calls, [])

    async def test_concurrent_callable_claim_cannot_reenter_consumed_grant(self):
        grant = approved(self.payload)
        root = Path(self.temp.name)
        receipt = consume_grant(self.payload, grant, root / 'ledger')
        entered, release = asyncio.Event(), asyncio.Event()
        fixture_entries = []

        @asynccontextmanager
        async def held(*args, **kwargs):
            fixture_entries.append(True)
            entered.set()
            await release.wait()
            raise RuntimeError('controlled_setup_failure')
            yield  # pragma: no cover

        with patch('evals.responsive_voice.hosted_pipeline_replay.source_fixture', held):
            first = asyncio.create_task(run_pair(self.payload, self.hosted_gate(grant), dsn='never-connect',
                output_dir=root / 'first', grant=grant, admission=receipt))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                with self.assertRaises(FileExistsError):
                    await run_pair(self.payload, self.hosted_gate(grant), dsn='never-connect',
                        output_dir=root / 'concurrent', grant=grant, admission=receipt)
            finally:
                release.set()
                await first
        self.assertEqual(fixture_entries, [True])

    async def test_receipt_for_different_grant_is_rejected_before_fixture(self):
        original = approved(self.payload)
        receipt = consume_grant(self.payload, original, Path(self.temp.name) / 'ledger')
        replacement = approved(self.payload)
        with patch('evals.responsive_voice.hosted_pipeline_replay.source_fixture') as fixture:
            with self.assertRaisesRegex(ValueError, 'consumed_grant_binding_mismatch'):
                await run_pair(self.payload, self.hosted_gate(replacement), dsn='never-connect',
                    output_dir=Path(self.temp.name) / 'changed', grant=replacement, admission=receipt)
        fixture.assert_not_called()

    async def test_actual_gemini_client_wire_store_false_under_controlled_http(self):
        received = []
        async def handle(request):
            body = json.loads(request.content)
            received.append(body)
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': '{}'}]}}]})
        gate, close = await create_gemini_gate(GeminiVisualAnalyzerConfig('not-a-key', 'fixture-model'),
            limits=Limits(), controlled_transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(close)
        gate.start_case(self.payload.cases[0])
        frame = released_frame(self.payload)
        gate.release(frame)
        self.assertEqual(await gate.generate_temporal_json(prompt='Allowed context', images=(image(frame),)), '{}')
        self.assertIs(received[0]['store'], False)
        self.assertEqual(gate.execution_kind, 'synthetic')
        self.assertFalse(gate.http_attempts[0]['actual_network'])
        self.assertNotIn('EVALUATOR_SECRET', json.dumps(received))

    async def test_unreleased_future_image_never_reaches_model(self):
        calls = []
        async def raw(**kwargs): calls.append(kwargs); return '{}'
        gate = SharedModelGate(SimpleNamespace(provider='fixture', model='fixture', generate_temporal_json=raw))
        gate.start_case(self.payload.cases[0])
        with self.assertRaisesRegex(ValueError, 'image_not_yet_released'):
            await gate.generate_temporal_json(prompt='context', images=(image(released_frame(self.payload)),))
        self.assertEqual(calls, [])

    async def test_wire_records_exact_bytes_and_counts_before_dispatch(self):
        prompt = 'Approved context \u03b1'
        frames = tuple(released_frame(self.payload, i) for i in (0, 1))
        images = tuple(image(frame) for frame in frames)
        parts = [{'text': prompt}]
        for item in images:
            parts.extend([{'text': item.label}, {'inlineData': {
                'mimeType': item.content_type,
                'data': base64.b64encode(item.image_bytes).decode('ascii')}}])
        body = {'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'temperature': 0.2, 'responseMimeType': 'application/json'},
            'store': False}
        receipts = []
        for pretty in (False, True):
            with self.subTest(pretty=pretty):
                content = json.dumps(body, ensure_ascii=False,
                    indent=2 if pretty else None,
                    separators=None if pretty else (',', ':')).encode('utf-8')
                gate = SharedModelGate(None)
                gate.start_case(self.payload.cases[0])
                gate.ticket = {'prompt': prompt, 'images': images, 'used': False}
                forwarded = []

                async def handle(request):
                    forwarded.append(request.content)
                    row = gate.http_attempts[0]
                    self.assertEqual(row['status'], 'dispatched')
                    self.assertEqual(row['request_content_sha256'],
                        hashlib.sha256(request.content).hexdigest())
                    self.assertEqual(row['request_content_bytes'], len(request.content))
                    self.assertEqual(row['prompt_characters'], len(prompt))
                    self.assertEqual(row['prompt_utf8_bytes'], len(prompt.encode('utf-8')))
                    self.assertEqual(row['image_count'], 2)
                    self.assertNotIn(prompt, json.dumps(row, ensure_ascii=False))
                    return httpx.Response(200)

                wire = ExactWireTransport(gate, httpx.MockTransport(handle),
                    model='fixture', actual_network=False)
                self.addAsyncCleanup(wire.aclose)
                await wire.handle_async_request(httpx.Request('POST',
                    'https://generativelanguage.googleapis.com/v1beta/models/fixture:generateContent',
                    content=content))
                self.assertEqual(forwarded, [content])
                self.assertEqual(len(gate.http_attempts), 1)
                self.assertEqual(gate.http_attempts[0]['status'], 'response_transport_returned')
                receipts.append(gate.http_attempts[0]['request_content_sha256'])
        self.assertEqual(len(set(receipts)), 2)

    async def test_analyzer_predicate_share_cap_and_serialization(self):
        release, entered = asyncio.Event(), asyncio.Event()
        active, maximum = 0, 0
        async def raw(**kwargs):
            nonlocal active, maximum
            active += 1; maximum = max(maximum, active); entered.set()
            try: await release.wait(); return '{}'
            finally: active -= 1
        gate = SharedModelGate(SimpleNamespace(provider='fixture', model='fixture', generate_temporal_json=raw),
            limits=Limits(calls=2))
        gate.start_case(self.payload.cases[0]); frame = released_frame(self.payload); gate.release(frame)
        first = asyncio.create_task(gate.generate_temporal_json(prompt='analysis', images=(image(frame),)))
        await entered.wait()
        second = asyncio.create_task(gate.generate_temporal_json(
            prompt='Evaluate whether these original images', images=(image(frame),)))
        await asyncio.sleep(0)
        self.assertEqual(len(gate.calls), 1)
        release.set(); await asyncio.gather(first, second)
        self.assertEqual(maximum, 1)
        self.assertEqual([c['phase'] for c in gate.calls], ['analysis', 'predicate'])
        with self.assertRaisesRegex(ValueError, 'shared_model_budget_exhausted'):
            await gate.generate_temporal_json(prompt='extra', images=(image(frame),))

    async def test_error_stops_future_calls_without_retry(self):
        calls = []
        async def raw(**kwargs): calls.append(1); raise RuntimeError('controlled error')
        gate = SharedModelGate(SimpleNamespace(provider='fixture', model='fixture', generate_temporal_json=raw))
        gate.start_case(self.payload.cases[0]); frame = released_frame(self.payload); gate.release(frame)
        with self.assertRaises(RuntimeError): await gate.generate_temporal_json(prompt='a', images=(image(frame),))
        with self.assertRaisesRegex(ValueError, 'model_gate_retired'):
            await gate.generate_temporal_json(prompt='b', images=(image(frame),))
        self.assertEqual(calls, [1])

    async def test_cancellation_joins_and_does_not_replay(self):
        entered = asyncio.Event()
        async def raw(**kwargs): entered.set(); await asyncio.Event().wait()
        gate = SharedModelGate(SimpleNamespace(provider='fixture', model='fixture', generate_temporal_json=raw))
        gate.start_case(self.payload.cases[0]); frame = released_frame(self.payload); gate.release(frame)
        task = asyncio.create_task(gate.generate_temporal_json(prompt='a', images=(image(frame),)))
        await entered.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(gate.lock.locked()); self.assertIsNone(gate.ticket); self.assertTrue(gate.halted)

    async def test_wire_refuses_missing_store_before_forward(self):
        forwarded = []
        async def handle(request): forwarded.append(request); return httpx.Response(200)
        gate = SharedModelGate(None)
        gate.ticket = {'prompt': 'context', 'images': (), 'used': False}
        wire = ExactWireTransport(gate, httpx.MockTransport(handle), model='fixture', actual_network=False)
        self.addAsyncCleanup(wire.aclose)
        with self.assertRaisesRegex(ValueError, 'request_store_false_required'):
            await wire.handle_async_request(httpx.Request('POST',
                'https://generativelanguage.googleapis.com/v1beta/models/fixture:generateContent',
                json={'contents': [], 'generationConfig': {}}))
        self.assertEqual(forwarded, [])
        self.assertEqual(gate.http_attempts, [])


if __name__ == '__main__':
    unittest.main()
