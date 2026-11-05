"""Bounded outbound transport and scoped credential storage."""
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from jarvis_project_connector.protocol import ConnectorError
from jarvis_project_connector.transport import (
    Connection, OutboundConnector, load_connection, origin, reserved_connection, save_connection,
)


@pytest.mark.parametrize("url", ["http://example.com", "https://user:pass@example.com",
                                "https://example.com/path", "https://example.com?token=secret",
                                "file:///etc/passwd", "https://example.com#fragment"])
def test_rejects_non_origin_or_unencrypted_remote_backend(url):
    with pytest.raises(ConnectorError):
        origin(url)


def test_credential_file_private_exclusive_and_redacted(tmp_path):
    config = Connection(url="https://example.com", connector_id=uuid4(), token="synthetic-token")
    path = tmp_path / "connection.json"
    save_connection(path, config)
    assert path.stat().st_mode & 0o077 == 0
    assert load_connection(path).token.get_secret_value() == "synthetic-token"
    assert "synthetic-token" not in repr(config)
    with pytest.raises(FileExistsError):
        save_connection(path, config)
    path.chmod(0o644)
    with pytest.raises(ConnectorError, match="credential_permissions"):
        load_connection(path)


def test_no_redirect_following_or_unbounded_server_response():
    seen = []
    def response(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://attacker.example/"})
    client = OutboundConnector("https://example.com", "synthetic-token")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(response), follow_redirects=False, trust_env=False)
    try:
        with pytest.raises(ConnectorError, match="backend_request_rejected"):
            client.post("/projects/connector/poll", {})
        assert seen == ["https://example.com/projects/connector/poll"]
    finally:
        client.close()


def test_reservation_does_not_overwrite_credentials_and_removes_owned_failed_file(tmp_path):
    path = tmp_path / "connection.json"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        with reserved_connection(path):
            raise AssertionError("Should fail before enrollment")
    assert path.read_text() == "existing"
    fresh = tmp_path / "fresh.json"
    with pytest.raises(RuntimeError):
        with reserved_connection(fresh):
            raise RuntimeError("Synthetic registration failure")
    assert not fresh.exists()
