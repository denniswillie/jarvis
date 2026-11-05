"""Actual pipe drain/control parsing with offline in-memory process streams."""
import asyncio
import io
import json
from types import SimpleNamespace
import unittest

from evals.responsive_voice.frontend_host_qualification import ChildOutputDrain, QualificationFailure


class HeadTailEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def drain(self, payload, *, cap=1024*1024, stderr=b"", log=None):
        out, err = asyncio.StreamReader(), asyncio.StreamReader()
        log = log if log is not None else io.BytesIO()
        drain = ChildOutputDrain(SimpleNamespace(pid=4321, stdout=out, stderr=err), log, byte_limit=cap)
        for start in range(0, len(payload), 777):
            out.feed_data(payload[start:start+777])
        err.feed_data(stderr)
        out.feed_eof()
        err.feed_eof()
        result = await drain.wait_closed(timeout=2)
        return drain, log, result

    async def test_default_budget_keeps_fixed_head_and_terminal_tail(self):
        first = b"FIRST_HEAD\n"
        last = b'TERMINAL_RESUME_CHECK\nFRONTEND_HOST_EXIT {"status":"passed"}\n'
        payload = first + b"middle-line-0123456789\n"*65000 + last
        drain, log, result = await self.drain(payload)
        retained = log.getvalue()
        self.assertTrue(retained.startswith(first))
        self.assertTrue(retained.endswith(last))
        self.assertEqual(result["head_byte_limit"], 256*1024)
        self.assertLessEqual(len(retained), 1024*1024)
        self.assertEqual(retained.count(drain.TRUNCATION_MARKER), 1)
        self.assertEqual(result["retained_source_bytes"]+result["discarded_bytes"], len(payload))
        self.assertEqual(len(retained), result["retained_log_bytes"])
        self.assertTrue(result["retention_complete"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["exit_notice_status"], "passed")
        self.assertEqual(await drain.wait_closed(), result)
        self.assertEqual(log.getvalue(), retained)

    async def test_untruncated_bytes_are_exact_including_partial_final_line(self):
        payload = b'FRONTEND_HOST_READY {"pid":4321}\nunchanged partial'
        drain, log, result = await self.drain(payload)
        self.assertEqual(log.getvalue(), payload)
        self.assertEqual(result["discarded_bytes"], 0)
        self.assertFalse(result["truncated"])
        self.assertEqual(await drain.wait_ready(timeout=1), {"pid":4321})

    async def test_no_partial_json_lines_are_spliced_across_gap(self):
        payload = b'{"head":true}\n' + b"X"*8192 + b'\n{"terminal":true}\n'
        drain, log, result = await self.drain(payload, cap=512)
        left, right = log.getvalue().split(drain.TRUNCATION_MARKER)
        self.assertEqual(left, b'{"head":true}\n')
        self.assertEqual(right, b'{"terminal":true}\n')
        self.assertEqual(json.loads(right), {"terminal":True})
        self.assertGreater(result["line_boundary_discarded_bytes"], 0)
        self.assertEqual(result["retained_source_bytes"]+result["discarded_bytes"], len(payload))

    async def test_ready_and_exit_parser_survive_discarded_middle_and_both_pipes(self):
        payload = b"x\n"*1000 + b'FRONTEND_HOST_READY {"pid":4321}\n' + b"y\n"*1000
        payload += b'FRONTEND_HOST_EXIT {"status":"passed"}\n'
        drain, log, result = await self.drain(payload, cap=256, stderr=b"stderr\n"*200)
        self.assertEqual(await drain.wait_ready(timeout=1), {"pid":4321})
        self.assertEqual(result["exit_notice_status"], "passed")
        self.assertEqual(result["drained_to_eof"], {"stdout":True,"stderr":True})
        self.assertEqual(result["errors"], [])
        self.assertLessEqual(len(log.getvalue()), 256)
        self.assertEqual(result["retained_source_bytes"]+result["discarded_bytes"],
                         sum(result["bytes_read"].values()))

    async def test_short_log_write_is_reported_not_retried_or_called_complete(self):
        class ShortLog(io.BytesIO):
            writes = 0
            def write(self, data):
                self.writes += 1
                return super().write(data[:3])
        target = ShortLog()
        drain, _, result = await self.drain(b"line\n"*500, cap=256, log=target)
        self.assertTrue(result["complete"])
        self.assertFalse(result["retention_complete"])
        self.assertEqual(result["errors"][0]["stage"], "private_log_write")
        self.assertEqual(result["retained_log_bytes"], 3)
        await drain.wait_closed()
        self.assertEqual(target.writes, 1)

    async def test_eof_before_ready_is_preserved(self):
        drain, log, result = await self.drain(b"no ready\n")
        with self.assertRaisesRegex(QualificationFailure, "child_stdout_ended_before_ready"):
            await drain.wait_ready(timeout=1)
        self.assertTrue(result["complete"])
        self.assertEqual(log.getvalue(), b"no ready\n")


if __name__ == "__main__":
    unittest.main()
