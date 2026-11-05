"""Trusted desktop startup inside an already owned, prepared E2B VM.

This is environment setup, not an agent loop or application workflow. The
ordinary unprivileged terminal remains the execution interface. Model commands
must never run as the template's passwordless-sudo account.
"""
from app.sandbox.e2b_provider import ACCOUNT, WORKSPACE

DISPLAY = ":0"
DESKTOP_CAPABILITY = "sandbox.desktop"

# Fixed paths/identity only. No task text or model arguments are interpolated.
# Existing /var/lib/jarvis ownership and allocation marker come from prepare().
# E2BProvider runs this under its clean, trusted-only root launch environment.
# Application lookup and writable workspace setup happen only after dropping uid.
PREPARE_DESKTOP = r"""
set -eu
exec 9>/var/lib/jarvis/desktop.lock
/usr/bin/flock -x 9
uid="$(/usr/bin/id -u jarvis)"
desktop_run() {
    /usr/sbin/runuser -u jarvis -- /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
        HOME=/home/jarvis/workspace USER=jarvis LOGNAME=jarvis DISPLAY=:0 \
        XDG_RUNTIME_DIR=/home/jarvis/workspace/.runtime \
        XDG_CONFIG_HOME=/home/jarvis/workspace/.config \
        XDG_CACHE_HOME=/home/jarvis/workspace/.cache "$@"
}
desktop_run /usr/bin/install -d -m 0700 /home/jarvis/workspace/.runtime
desktop_run /usr/bin/install -d -m 0700 /home/jarvis/workspace/.config
desktop_run /usr/bin/install -d -m 0700 /home/jarvis/workspace/.cache
desktop_run /bin/sh -c 'for executable in Xvfb xdpyinfo startxfce4 dbus-run-session xdotool scrot; do
    command -v "$executable" >/dev/null || exit 1
done'
if ! /usr/bin/pgrep -u "$uid" -x Xvfb >/dev/null; then
    /usr/bin/nohup /usr/sbin/runuser -u jarvis -- /usr/bin/env -i PATH=/usr/bin:/bin \
        Xvfb :0 -ac -screen 0 1024x768x24 -nolisten tcp \
        </dev/null >/var/lib/jarvis/desktop-display.log 2>&1 9>&- &
fi
i=0
until desktop_run xdpyinfo >/dev/null 2>&1; do
    i=$((i+1)); test "$i" -lt 30; /usr/bin/sleep 0.2
done
# A reachable display alone does not prove that it belongs to this account.
/usr/bin/pgrep -u "$uid" -x Xvfb >/dev/null
test "$(/usr/bin/pgrep -x Xvfb)" = "$(/usr/bin/pgrep -u "$uid" -x Xvfb)"
if ! /usr/bin/pgrep -u "$uid" -x xfce4-session >/dev/null; then
    (exec 9>&-; desktop_run dbus-run-session -- startxfce4) \
        </dev/null >/var/lib/jarvis/desktop-session.log 2>&1 &
fi
i=0
until /usr/bin/pgrep -u "$uid" -x xfwm4 >/dev/null; do
    i=$((i+1)); test "$i" -lt 40; /usr/bin/sleep 0.2
done
test "$(/usr/bin/pgrep -x xfwm4)" = "$(/usr/bin/pgrep -u "$uid" -x xfwm4)"
desktop_run /bin/sh -c 'test "$(/usr/bin/id -u)" -ne 0 && ! /usr/bin/sudo -n true 2>/dev/null'
printf '%s\n' 'jarvis.desktop.ready.v1'
"""

def desktop_environment():
    """Explicit environment for supported desktop CLI tools, without secrets."""
    return {"DISPLAY": DISPLAY, "HOME": WORKSPACE, "USER": ACCOUNT, "LOGNAME": ACCOUNT,
            "XDG_RUNTIME_DIR": WORKSPACE + "/.runtime",
            "XDG_CONFIG_HOME": WORKSPACE + "/.config",
            "XDG_CACHE_HOME": WORKSPACE + "/.cache"}
