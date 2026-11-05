"""Control/start versus admitted command waits; no SDK or external calls."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.media_decode import DECODE_MEDIA_INPUT, parse_decode_failure
from app.sandbox.models import E2BConfiguration, SandboxError

class NativeExit(Exception):
    def __init__(self, *, stderr='PRIVATE-REMOTE-TEXT', exit_code=2):
        self.exit_code, self.stderr, self.stdout = exit_code, stderr, 'PRIVATE-OUTPUT'
        super().__init__('PRIVATE-EXCEPTION-TEXT')

class CommandWaitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.run = AsyncMock(return_value=SimpleNamespace(exit_code=0, stdout=''))
        self.box = SimpleNamespace(commands=SimpleNamespace(run=self.run))
        self.provider = E2BProvider(E2BConfiguration(api_key='offline-unused', request_timeout_seconds=1),
                                   sdk=SimpleNamespace(CommandExitException=NativeExit))

    async def test_foreground_can_finish_after_control_ceiling_inside_command_budget(self):
        async def acknowledged(*_args, **_kwargs):
            await asyncio.sleep(3.1)
            return SimpleNamespace(exit_code=0, stdout='')
        self.run.side_effect = acknowledged
        await self.provider._run_privileged(self.box, 'fixed-offline-script', timeout=5)
        self.assertEqual(self.run.call_args.kwargs['request_timeout'], 1)
        self.assertEqual(self.run.call_args.kwargs['timeout'], 5)
        self.run.assert_awaited_once()

    async def test_background_start_keeps_short_control_ceiling_and_never_retries(self):
        async def never_acknowledged(*_args, **_kwargs):
            await asyncio.sleep(30)
        self.run.side_effect = never_acknowledged
        with self.assertRaises(SandboxError) as caught:
            await self.provider._run_privileged(self.box, 'fixed-offline-script', timeout=60, background=True)
        self.assertEqual(caught.exception.code, 'sandbox_provider_outcome_uncertain')
        self.assertEqual(caught.exception.diagnostic['outcome'], 'unknown')
        self.run.assert_awaited_once()

    async def test_acknowledged_failure_and_lost_response_are_distinct_and_redacted(self):
        for native, expected in [(NativeExit(), 'sandbox_privileged_command_failed'),
                                 (TimeoutError('PRIVATE-TIMEOUT'), 'sandbox_provider_outcome_uncertain')]:
            self.run.reset_mock();self.run.side_effect = native
            with self.subTest(kind=type(native).__name__), self.assertRaises(SandboxError) as caught:
                await self.provider._run_privileged(self.box, 'fixed-offline-script', timeout=5)
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn('PRIVATE', json.dumps(caught.exception.diagnostic))
            self.assertNotIn('stdout', caught.exception.diagnostic)
            self.assertNotIn('stderr', caught.exception.diagnostic)
            self.run.assert_awaited_once()
        self.run.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.provider._run_privileged(self.box, 'fixed-offline-script', timeout=5)

    async def test_only_allowlisted_inspection_failure_details_are_retained(self):
        receipt = {'format':'jarvis.media-decode-failure.v1','phase':'inventory','reason':'decode_timeout',
                   'elapsed_ms':57000,'phase_ms':56000}
        for payload, accepted in [(json.dumps(receipt), True), ('PRIVATE-STDERR', False),
                                  (json.dumps(receipt | {'reason':'PRIVATE-SECRET'}), False)]:
            self.run.side_effect = NativeExit(stderr=payload)
            with self.assertRaises(SandboxError) as caught:
                await self.provider._run_privileged(self.box, 'fixed-offline-script', timeout=60,
                    stage='media_input_decode', failure_parser=parse_decode_failure)
            diagnostic = caught.exception.diagnostic
            self.assertEqual(diagnostic['outcome'], 'completed_nonzero')
            self.assertEqual(diagnostic['exit_code'], 2)
            self.assertEqual('inspection' in diagnostic, accepted)
            self.assertNotIn('PRIVATE', json.dumps(diagnostic))

    def test_fixed_program_reports_phase_without_media_or_exception_content(self):
        program = {'__name__':'offline_program_test'}
        exec(compile(DECODE_MEDIA_INPUT, '<fixed-decode-program>', 'exec'), program)
        try:
            program['phase']('inventory', lambda: (_ for _ in ()).throw(program['DecodeFailure']('decode_timeout')))
        except ValueError as error:
            receipt = program['failure_receipt'](error)
        self.assertEqual(parse_decode_failure(json.dumps(receipt))['phase'], 'inventory')
        self.assertEqual(receipt['reason'], 'decode_timeout')
        receipt = program['failure_receipt'](OSError('PRIVATE-PATH-TOKEN'))
        self.assertEqual(receipt['reason'], 'decode_system_failure')
        self.assertNotIn('PRIVATE', json.dumps(receipt))
        for change in ({'phase':[]}, {'reason':'unknown-private'}, {'phase_ms':True}, {'elapsed_ms':65001},
                       {'phase_ms':60000}, {'extra':'PRIVATE'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_decode_failure(json.dumps(receipt | change))

if __name__ == '__main__':
    unittest.main()
