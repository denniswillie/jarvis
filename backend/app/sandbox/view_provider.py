"""Non-resuming attachment to the standard protected E2B desktop transport.

The pinned SDK's connect() resumes and renews. This adapter deliberately uses
its generated GET-detail endpoint and constructs only a local command client.
"""
from dataclasses import dataclass, field
from datetime import UTC, datetime
import base64
import binascii
from io import BytesIO
import re

from app.runtime.http_privacy import private_http_logs
from app.sandbox.models import SandboxError


VIEW_PORT = 6087
THUMBNAIL_BYTES = 32768
# Both binaries are part of the existing desktop template. Capture only the
# running X display: no resume, input event, viewer startup or file creation.
CAPTURE_THUMBNAIL = r"""
set -eu
size="$(/usr/bin/xdpyinfo -display :0 | /usr/bin/awk '/dimensions:/{print $2; exit}')"
test -n "$size"
/usr/bin/ffmpeg -nostdin -hide_banner -loglevel error -f x11grab -video_size "$size" \
    -i :0 -frames:v 1 -vf 'scale=320:200:force_original_aspect_ratio=decrease' \
    -q:v 8 -f image2pipe -c:v mjpeg pipe:1 | /usr/bin/base64 -w0
"""


def decode_thumbnail(encoded):
    from PIL import Image
    try:
        if not isinstance(encoded, str) or len(encoded) > 45000:
            raise ValueError("thumbnail size")
        data = base64.b64decode(encoded.strip(), validate=True)
        if not 0 < len(data) <= THUMBNAIL_BYTES:
            raise ValueError("thumbnail size")
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG" or not (0 < image.width <= 320 and 0 < image.height <= 200):
                raise ValueError("thumbnail dimensions")
            image.verify()
        return data
    except (ValueError, OSError, binascii.Error):
        raise SandboxError("sandbox_thumbnail_invalid") from None
# The agent desktop is view-only, so watching it creates no keyboard/mouse
# activity. Configure only its existing unprivileged :0 session, not the host.
WAKE_DISPLAY = r"""
stage=desktop_unblank
desktop_pid="$(/usr/bin/pgrep -u "$(/usr/bin/id -u jarvis)" -x xfce4-session)"
desktop_bus="$(/usr/bin/tr '\000' '\n' < /proc/"$desktop_pid"/environ | /usr/bin/sed -n 's/^DBUS_SESSION_BUS_ADDRESS=//p')"
test -n "$desktop_bus"
desktop_view_command() {
    /usr/sbin/runuser -u jarvis -- /usr/bin/env -i PATH=/usr/bin:/bin \
        HOME=/home/jarvis/workspace USER=jarvis LOGNAME=jarvis DISPLAY=:0 \
        XDG_RUNTIME_DIR=/home/jarvis/workspace/.runtime \
        XDG_CONFIG_HOME=/home/jarvis/workspace/.config \
        DBUS_SESSION_BUS_ADDRESS="$desktop_bus" "$@"
}
if desktop_view_command /usr/bin/xfconf-query -c xfce4-screensaver -p /saver/enabled >/dev/null 2>&1; then
    desktop_view_command /usr/bin/xfconf-query -c xfce4-screensaver -p /saver/enabled -s false
else
    desktop_view_command /usr/bin/xfconf-query -c xfce4-screensaver -p /saver/enabled -n -t bool -s false
fi
desktop_view_command /usr/bin/xset s off
desktop_view_command /usr/bin/xfce4-screensaver-command --deactivate
"""

START_VIEW = r"""
set -eu
stage=lock
trap 'printf "jarvis.view.failed.%s\n" "$stage" >&2' EXIT
exec 9>/var/lib/jarvis/view.lock
/usr/bin/flock -x 9
stage=prepared_marker
test -f /var/lib/jarvis/ready
stage=x11vnc_binary
test -x /usr/bin/x11vnc
stage=novnc_binary
test -x /opt/noVNC/utils/novnc_proxy
stage=desktop_display
/usr/bin/pgrep -u "$(/usr/bin/id -u jarvis)" -x Xvfb >/dev/null
""" + WAKE_DISPLAY + r"""
if ! test -f /var/lib/jarvis/view.ready; then
    stage=vnc_start
    /usr/bin/x11vnc -bg -display :0 -forever -shared -viewonly -localhost -noremote -noshm \
        -nosel -noclipboard -nosetclipboard -noprimary -nosetprimary \
        -nopw -rfbport 5907 -wait 50 -o /var/lib/jarvis/view-vnc.log </dev/null >/dev/null 9>&-
    stage=novnc_start
    /usr/bin/nohup /opt/noVNC/utils/novnc_proxy --vnc localhost:5907 \
        --listen 6087 --web /opt/noVNC </dev/null >/var/lib/jarvis/view-proxy.log 2>&1 9>&- &
    i=0
    until /usr/bin/python3 -I -S -c 'import socket; s=socket.create_connection(("127.0.0.1",6087),.3); s.close()' >/dev/null 2>&1; do
        i=$((i+1)); test "$i" -lt 25; /usr/bin/sleep .2
    done
    /usr/bin/touch /var/lib/jarvis/view.ready
fi
stage=transport_check
/usr/bin/python3 -I -S -c 'import socket; s=socket.create_connection(("127.0.0.1",6087),.5); s.close()'
trap - EXIT
printf '%s\n' 'jarvis.view.ready.v1'
"""


def view_failure(stderr):
    stages = {"lock", "prepared_marker", "x11vnc_binary", "novnc_binary", "desktop_display", "desktop_unblank",
              "vnc_start", "novnc_start", "transport_check"}
    values = re.findall(r"(?m)^jarvis\.view\.failed\.([a-z_]+)$", stderr[-4096:])
    return {"view_stage": values[-1]} if values and values[-1] in stages else None


@dataclass(frozen=True)
class DesktopConnection:
    url: str = field(repr=False)
    token: str = field(repr=False)
    expires_at: datetime


class E2BDesktopViewer:
    def __init__(self, provider, secrets):
        self.provider, self.secrets = provider, secrets

    async def _attach(self, handle):
        from e2b.api.client.api.sandboxes import get_sandboxes_sandbox_id
        from e2b.api.client.models.sandbox_detail import SandboxDetail
        from e2b.api.client_async import get_api_client
        from e2b.connection_config import ConnectionConfig
        from e2b.sandbox.sandbox_api import SandboxInfo
        from packaging.version import Version

        options = self.provider._options()
        config = ConnectionConfig(**options)
        client = get_api_client(config)
        result = await self.provider._request(get_sandboxes_sandbox_id.asyncio_detailed,
                                             handle.sandbox_id, client=client)
        if result.status_code == 404:
            self.secrets.remove(handle.sandbox_id, handle.scope)
            raise SandboxError("sandbox_not_running")
        if result.status_code != 200 or not isinstance(result.parsed, SandboxDetail):
            raise SandboxError("sandbox_view_provider_inspection_failed")
        detail = result.parsed
        raw = SandboxInfo._from_sandbox_detail(detail)
        current = self.provider._admit(raw, handle.provider_scope, handle.policy)
        if current.sandbox_id != handle.sandbox_id:
            raise SandboxError("sandbox_scope_mismatch")
        if current.state != "running":
            raise SandboxError("sandbox_not_running")
        if current.expires_at <= datetime.now(UTC):
            raise SandboxError("sandbox_view_access_ended")
        token = self.secrets.read(handle)
        if not isinstance(detail.envd_access_token, str) or not detail.envd_access_token:
            raise SandboxError("sandbox_view_credential_unavailable")
        # Domain is provider-issued, never model input. Restrict forwarding to
        # the fixed protected E2B port/host rather than accepting arbitrary URLs.
        domain = detail.domain if isinstance(detail.domain, str) else "e2b.app"
        if domain != "e2b.app" or not re.fullmatch(r"[a-z0-9-]{1,200}", handle.sandbox_id):
            raise SandboxError("sandbox_view_host_unavailable")
        headers = {"E2b-Sandbox-Id": handle.sandbox_id,
                   "E2b-Sandbox-Port": str(ConnectionConfig.envd_port),
                   "X-Access-Token": detail.envd_access_token}
        box = self.provider.sdk.AsyncSandbox(sandbox_id=handle.sandbox_id, sandbox_domain=domain,
            envd_version=Version(detail.envd_version), envd_access_token=detail.envd_access_token,
            traffic_access_token=token,
            connection_config=ConnectionConfig(extra_sandbox_headers=headers, **options))
        connection = DesktopConnection(f"wss://{VIEW_PORT}-{handle.sandbox_id}.{domain}/websockify",
                                       token, current.expires_at)
        return box, connection

    async def inspect(self, handle):
        _, connection = await self._attach(handle)
        return connection

    async def thumbnail(self, handle):
        box, _ = await self._attach(handle)
        result = await self.provider._run_privileged(box, CAPTURE_THUMBNAIL, timeout=8)
        return decode_thumbnail(result.stdout)

    async def open(self, handle):
        box, connection = await self._attach(handle)
        result = await self.provider._run_privileged(box, START_VIEW, timeout=10, failure_parser=view_failure)
        if result.stdout.strip() != "jarvis.view.ready.v1":
            raise SandboxError("sandbox_view_start_receipt_invalid")
        # Recheck GET-only after setup; never return a view across shutdown.
        return await self.inspect(handle)
