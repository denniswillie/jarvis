"""Small policy checks for the existing desktop's view-start unblank step."""
import subprocess

from app.sandbox.view_provider import START_VIEW, WAKE_DISPLAY, view_failure


def test_view_start_unblanks_existing_transport_too():
    assert START_VIEW.index(WAKE_DISPLAY) < START_VIEW.index("if ! test -f /var/lib/jarvis/view.ready")
    assert "xfce4-screensaver-command --deactivate" in WAKE_DISPLAY
    assert "xfce4-screensaver-command --exit" not in WAKE_DISPLAY
    assert "kill" not in WAKE_DISPLAY


def test_unblank_uses_only_existing_jarvis_display_and_session_bus():
    assert 'pgrep -u "$(/usr/bin/id -u jarvis)" -x xfce4-session' in WAKE_DISPLAY
    assert "runuser -u jarvis -- /usr/bin/env -i" in WAKE_DISPLAY
    assert "DISPLAY=:0" in WAKE_DISPLAY
    assert 'DBUS_SESSION_BUS_ADDRESS="$desktop_bus"' in WAKE_DISPLAY
    assert "XDG_CONFIG_HOME=/home/jarvis/workspace/.config" in WAKE_DISPLAY
    assert "-p /saver/enabled -s false" in WAKE_DISPLAY
    assert "-p /saver/enabled -n -t bool -s false" in WAKE_DISPLAY
    assert "/usr/bin/xset s off" in WAKE_DISPLAY


def test_unblank_failure_keeps_an_explicit_stage():
    assert view_failure("jarvis.view.failed.desktop_unblank\n") == {"view_stage": "desktop_unblank"}


def test_composed_viewer_start_is_valid_shell_without_executing_it():
    result = subprocess.run(["/bin/sh", "-n"], input=START_VIEW, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
