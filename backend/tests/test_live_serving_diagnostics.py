"""Offline transport attribution; no browser, provider, or serving process."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close
from evals.responsive_voice.synthetic_webrtc import Browser, CDP, annotate_connection_error
from live_serving_smoke import ServingPeer


class ConnectionDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def test_unknown_close_reason_is_redacted(self):
        secret = "https://private.invalid/?token=do-not-record-this"
        error = ConnectionClosedError(Close(1008, secret), Close(1000, ""), True)
        detail = annotate_connection_error(error, connection="browser_cdp", operation="receive")
        self.assertEqual(detail["received_close"]["code"], 1008)
        self.assertEqual(detail["received_close"]["reason"], "[redacted]")
        self.assertTrue(detail["received_then_sent"])
        self.assertNotIn(secret, json.dumps(detail))
        self.assertEqual(len(detail["received_close"]["reason_sha256"]), 64)

    async def test_cdp_receive_preserves_original_exception_and_safe_operation(self):
        error = ConnectionClosedError(None, Close(1011, "keepalive ping timeout"), None)
        socket = SimpleNamespace(send=AsyncMock(), recv=AsyncMock(side_effect=error))
        with self.assertRaises(ConnectionClosedError) as caught:
            await CDP(socket).evaluate("acceptance.load('work','PRIVATE_AUDIO')")
        self.assertIs(caught.exception, error)
        detail = error._jarvis_connection_diagnostic
        self.assertEqual(detail["connection"], "browser_cdp")
        self.assertEqual(detail["operation"], "Runtime.evaluate:receive")
        self.assertEqual(detail["expression_kind"], "acceptance.load")
        self.assertEqual(detail["sent_close"]["reason"], "keepalive ping timeout")
        self.assertNotIn("PRIVATE_AUDIO", json.dumps(detail))

    async def test_cdp_send_is_distinguished(self):
        error = ConnectionClosedError(None, None, None)
        socket = SimpleNamespace(send=AsyncMock(side_effect=error), recv=AsyncMock())
        with self.assertRaises(ConnectionClosedError):
            await CDP(socket).evaluate("window.cleanup()")
        self.assertEqual(error._jarvis_connection_diagnostic["operation"], "Runtime.evaluate:send")
        socket.recv.assert_not_awaited()

    async def test_backend_receive_is_distinguished(self):
        error = ConnectionClosedError(Close(1011, "private backend details"), None, None)
        trace = Mock()
        peer = ServingPeer(None, trace, SimpleNamespace(recv=AsyncMock(side_effect=error)))
        with self.assertRaises(ConnectionClosedError):
            await peer.receive()
        self.assertEqual(error._jarvis_connection_diagnostic["connection"], "backend_audio_ws")
        self.assertEqual(error._jarvis_connection_diagnostic["operation"], "receive")
        trace.record.assert_called_once()

    async def test_backend_send_is_distinguished(self):
        error = ConnectionClosedError(None, None, None)
        peer = ServingPeer(None, Mock(), SimpleNamespace(send=AsyncMock(side_effect=error)))
        with self.assertRaises(ConnectionClosedError):
            await peer.respond({"type": "fixture"})
        self.assertEqual(error._jarvis_connection_diagnostic["connection"], "backend_audio_ws")
        self.assertEqual(error._jarvis_connection_diagnostic["operation"], "send")

    async def test_normal_backend_close_remains_normal(self):
        error = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
        trace = Mock()
        peer = ServingPeer(None, trace, SimpleNamespace(recv=AsyncMock(side_effect=error)))
        await peer.receive()
        trace.record.assert_not_called()

    async def test_browser_cleanup_records_preexisting_exit_and_cdp_failure(self):
        error = ConnectionClosedError(None, None, None)
        browser = Browser(Path("unused"), Path("unused.wav"))
        browser.process = SimpleNamespace(pid=123, poll=Mock(return_value=17), returncode=17)
        browser.cdp = CDP(SimpleNamespace(send=AsyncMock(side_effect=error)))
        with patch("evals.responsive_voice.synthetic_webrtc.os.killpg", side_effect=ProcessLookupError):
            result = await browser.close()
        self.assertEqual(result["process_before_cleanup"], {"pid": 123, "returncode": 17, "running": False})
        self.assertEqual(result["media_connection_error"]["expression_kind"], "window.cleanup")
        self.assertEqual(result["media_error"], "ConnectionClosedError")


if __name__ == "__main__":
    unittest.main()
