"""Desktop environment is opt-in; no implicit elevation of old grants."""
import pytest
from pydantic import ValidationError

from app.sandbox.models import SandboxPolicy
from app.sandbox.desktop_environment import DISPLAY, PREPARE_DESKTOP, desktop_environment


def test_default_policy_shape_stays_compatible():
    policy = SandboxPolicy()
    assert policy.model_dump(mode="json") == {
        "template": "base", "lifetime_seconds": 300, "command_seconds": 60,
        "output_bytes": 64000, "allowed_domains": [],
    }
    assert policy.required_capabilities == ("sandbox.execute",)


def test_desktop_is_separate_from_network_and_root():
    desktop = SandboxPolicy(template="desktop")
    network = SandboxPolicy(template="desktop", internet_access=True)
    assert desktop.required_capabilities == ("sandbox.execute", "sandbox.desktop")
    assert network.required_capabilities == ("sandbox.execute", "sandbox.network", "sandbox.desktop")
    assert not desktop.within(SandboxPolicy(internet_access=True))
    assert SandboxPolicy().within(desktop)
    assert not network.within(desktop)
    assert desktop.within(network)
    assert desktop.fingerprint != SandboxPolicy().fingerprint


@pytest.mark.parametrize("template", ["untrusted-template", "", None])
def test_unconfigured_template_is_rejected(template):
    with pytest.raises(ValidationError):
        SandboxPolicy(template=template)


def test_desktop_environment_is_explicit_and_secret_free():
    env = desktop_environment()
    assert env["DISPLAY"] == DISPLAY == ":0"
    assert env["USER"] == env["LOGNAME"] == "jarvis"
    assert env["HOME"] == "/home/jarvis/workspace"
    assert set(env) == {"DISPLAY", "HOME", "USER", "LOGNAME",
                        "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"}


def test_preparation_uses_the_unprivileged_account_and_owned_receipts():
    assert "/usr/sbin/runuser -u jarvis" in PREPARE_DESKTOP
    assert "! /usr/bin/sudo -n true" in PREPARE_DESKTOP
    assert "/usr/bin/flock -x 9" in PREPARE_DESKTOP
    assert '/usr/bin/pgrep -u "$uid" -x Xvfb' in PREPARE_DESKTOP
    assert '/usr/bin/pgrep -u "$uid" -x xfce4-session' in PREPARE_DESKTOP
    assert "-nolisten tcp" in PREPARE_DESKTOP
    assert "jarvis.desktop.ready.v1" in PREPARE_DESKTOP


def test_privileged_desktop_utilities_use_absolute_system_paths():
    assert 'uid="$(/usr/bin/id -u jarvis)"' in PREPARE_DESKTOP
    assert "/usr/bin/nohup /usr/sbin/runuser -u jarvis -- /usr/bin/env -i" in PREPARE_DESKTOP
    assert '/usr/bin/sleep 0.2' in PREPARE_DESKTOP
    for unsafe in ("\nflock ", "\ninstall ", "$(id ", "$(pgrep ", "\n    runuser ", "\n    nohup ", "; sleep "):
        assert unsafe not in PREPARE_DESKTOP


def test_workspace_setup_and_application_lookup_happen_after_privilege_drop():
    assert "/usr/sbin/runuser -u jarvis -- /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin" in PREPARE_DESKTOP
    for directory in (".runtime", ".config", ".cache"):
        assert "desktop_run /usr/bin/install -d -m 0700 /home/jarvis/workspace/"+directory in PREPARE_DESKTOP
    assert "-o jarvis -g jarvis /home/jarvis/workspace/" not in PREPARE_DESKTOP
    assert "desktop_run /bin/sh -c 'for executable in Xvfb" in PREPARE_DESKTOP
    assert 'command -v "$executable" >/dev/null || exit 1' in PREPARE_DESKTOP
    assert "desktop_run xdpyinfo" in PREPARE_DESKTOP
    assert "desktop_run dbus-run-session -- startxfce4" in PREPARE_DESKTOP
