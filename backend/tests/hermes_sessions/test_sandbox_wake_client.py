import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from app.hermes_sessions.sandbox_wake_client import WakeDeliveryError, _open, notify


class WakeClientTests(unittest.TestCase):
    def test_real_redirect_never_reaches_destination_with_credential(self):
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "/destination")
                self.end_headers()

            def do_GET(self):
                self.server.destination_visits.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        server.destination_visits = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}/redirect", data=b"{}",
                              headers={"Authorization": "Bearer fixture-secret"})
            with self.assertRaises(HTTPError) as raised:
                _open(request, timeout=3)
            self.assertEqual(raised.exception.code, 302)
            raised.exception.close()
            self.assertEqual(server.destination_visits, [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "wake.json"
        self.config = {"endpoint": "https://callback.example/hermes/background-wake",
                       "session_id": "session-one", "token": "fixture-private-token"}
        self.path.write_text(json.dumps(self.config))

    def call(self, **changes):
        return notify(**({"event_id": "result-one", "message": "Saved result", "config_path": self.path} | changes))

    def response(self, **changes):
        data = {"delivery": "admitted", "session_id": "session-one", "admission_id": "native-input-one"} | changes
        response = io.BytesIO(json.dumps(data).encode())
        response.status = 202
        return response

    def test_request_is_scoped_and_credential_not_in_url_or_body(self):
        with patch("app.hermes_sessions.sandbox_wake_client._open", return_value=self.response()) as send:
            self.assertEqual(self.call()["admission_id"], "native-input-one")
        request = send.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer fixture-private-token")
        self.assertNotIn(b"fixture-private-token", request.data)
        self.assertNotIn("fixture-private-token", request.full_url)
        self.assertEqual(json.loads(request.data)["event_id"], "result-one")

    def test_no_automatic_retry_or_new_event_after_network_uncertainty(self):
        with patch("app.hermes_sessions.sandbox_wake_client._open", side_effect=URLError("fixture")) as send:
            with self.assertRaisesRegex(WakeDeliveryError, "same event ID and message"):
                self.call()
            self.assertEqual(send.call_count, 1)

    def test_refused_and_conflicting_callbacks_are_clear_without_secrets(self):
        for code, text in ((403, "refused"), (409, "different content"), (503, "unconfirmed")):
            error = HTTPError(self.config["endpoint"], code, "fixture", {}, None)
            with patch("app.hermes_sessions.sandbox_wake_client._open", side_effect=error):
                with self.assertRaisesRegex(WakeDeliveryError, text) as raised:
                    self.call()
            self.assertNotIn(self.config["token"], str(raised.exception))

    def test_response_must_acknowledge_the_same_session(self):
        for changes in ({"session_id": "sibling"}, {"delivery": "completed"}, {"admission_id": ""}):
            with patch("app.hermes_sessions.sandbox_wake_client._open", return_value=self.response(**changes)):
                with self.assertRaisesRegex(WakeDeliveryError, "unconfirmed"):
                    self.call()

    def test_invalid_content_never_sends(self):
        with patch("app.hermes_sessions.sandbox_wake_client._open") as send:
            for changes in ({"message": "x" * 4097}, {"message": " "}, {"event_id": ""}, {"event_id": "x" * 257}):
                with self.assertRaises(ValueError):
                    self.call(**changes)
            send.assert_not_called()

    def test_plain_https_endpoint_required(self):
        for endpoint in ("http://callback.example", "https://user:password@callback.example", "https://callback.example?token=oops"):
            self.path.write_text(json.dumps(self.config | {"endpoint": endpoint}))
            with patch("app.hermes_sessions.sandbox_wake_client._open") as send:
                with self.assertRaises(ValueError):
                    self.call()
                send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
