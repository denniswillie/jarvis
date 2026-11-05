"""Explicit E2B desktop/provider compatibility probe, not model acceptance.

Uses the production allocation policy/preparation and existing resource manifest.
Only synthetic local files and GUI input are used. No personal account, database,
model, public screen stream or real external action is involved.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shlex

from dotenv import dotenv_values

from app.sandbox.desktop_environment import desktop_environment
from app.sandbox.e2b_provider import ACCOUNT, WORKSPACE, E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxPolicy
from evals.second_brain.e2b_probe import Probe


def scoped_desktop_class():
    # Eval-only maintained SDK adapter: never use the template's sudo account.
    from e2b_desktop import Sandbox

    class Commands:
        def __init__(self, native):
            self.native = native

        def run(self, command, **kwargs):
            env = " ".join(shlex.quote(key+"="+value) for key, value in desktop_environment().items())
            wrapped = ("env -i PATH=/usr/local/bin:/usr/bin:/bin " + env
                       + " /bin/bash --noprofile --norc -c " + shlex.quote(command))
            return self.native.run(wrapped, user=ACCOUNT, cwd=WORKSPACE, **kwargs)

    class Files:
        def __init__(self, native):
            self.native = native

        def read(self, path, **kwargs):
            return self.native.read(path, user=ACCOUNT, **kwargs)

        def remove(self, path, **kwargs):
            return self.native.remove(path, user=ACCOUNT, **kwargs)

    class ScopedDesktop(Sandbox):
        @property
        def commands(self):
            return Commands(super().commands)

        @property
        def files(self):
            return Files(super().files)

    return ScopedDesktop


class DesktopProbe(Probe):
    async def window(self, desktop, expected):
        # Poll the same launched application; never relaunch on observation loss.
        title = ""
        for _ in range(30):
            title = await asyncio.to_thread(lambda: desktop.get_window_title(desktop.get_current_window_id()))
            if expected in title:
                return title
            await asyncio.sleep(.5)
        raise AssertionError("application_window_not_observed")

    async def capture(self, desktop, filename):
        pixels = bytes(await asyncio.to_thread(desktop.screenshot))
        if not pixels.startswith(b"\x89PNG") or len(pixels) > 6000000:
            raise AssertionError("invalid_bounded_screenshot")
        (self.root / filename).write_bytes(pixels)
        self.record.setdefault("screenshots", []).append({
            "file":filename, "sha256":hashlib.sha256(pixels).hexdigest(), "bytes":len(pixels),
            "captured_at":datetime.now(UTC).isoformat(),
        })
        self.persist()

    async def exercise(self):
        try:
            handle = await self.create(SandboxPolicy(template="desktop", lifetime_seconds=300, command_seconds=30))
            self.check("owned_private_offline_desktop",
                handle.policy.required_capabilities == ("sandbox.execute", "sandbox.desktop"),
                template_id=handle.template_id)
            desktop = await asyncio.to_thread(scoped_desktop_class().connect, handle.sandbox_id,
                                               **self.provider._options())
            result = await asyncio.to_thread(desktop.commands.run,
                "id -u; if sudo -n true 2>/dev/null; then printf 'root-enabled'; else printf 'root-denied'; fi")
            uid, privilege = result.stdout.strip().splitlines()
            self.check("desktop_unprivileged", uid != "0" and privilege == "root-denied", uid=int(uid))
            result = await asyncio.to_thread(desktop.commands.run,
                "test ! -r /var/lib/jarvis/ready && test ! -w /var/lib/jarvis && "
                "test ! -w /home/jarvis && test -w /home/jarvis/workspace")
            self.check("protected_receipts_and_writable_workspace", result.exit_code == 0)

            before = (await asyncio.to_thread(desktop.commands.run,
                "pgrep -u $(id -u) -x Xvfb; pgrep -u $(id -u) -x xfce4-session")).stdout
            await self.provider.prepare(handle)
            after = (await asyncio.to_thread(desktop.commands.run,
                "pgrep -u $(id -u) -x Xvfb; pgrep -u $(id -u) -x xfce4-session")).stdout
            self.check("repeat_prepare_reuses_desktop", before == after and len(after.splitlines()) == 2)
            await self.capture(desktop, "desktop.png")

            native = await self.provider._connect(handle)
            page = WORKSPACE + "/fixture.html"
            await native.files.write(page,
                '<!doctype html><title>Jarvis desktop fixture</title>'
                '<body style="font:28px sans-serif;padding:40px;background:#eef7e9">'
                '<h1>Harmless desktop fixture</h1><p>Calibration value: 47</p></body>',
                user=ACCOUNT)
            launched = await asyncio.to_thread(desktop.commands.run,
                "google-chrome --no-sandbox --disable-dev-shm-usage --disable-gpu "
                "--no-first-run --no-default-browser-check " + shlex.quote("file://"+page),
                background=True, timeout=0)
            await asyncio.to_thread(launched.disconnect)
            title = await self.window(desktop, "Jarvis desktop fixture")
            await self.capture(desktop, "browser.png")
            self.check("real_browser_window", True, title=title)

            note = WORKSPACE + "/note.txt"
            await native.files.write(note, "", user=ACCOUNT)
            launched = await asyncio.to_thread(desktop.commands.run,
                "mousepad " + shlex.quote(note), background=True, timeout=0)
            await asyncio.to_thread(launched.disconnect)
            await self.window(desktop, "note.txt")
            contents = "Synthetic desktop note " + self.run_id + "\n"
            await asyncio.to_thread(desktop.write, contents.removesuffix("\n"), delay_in_ms=5)
            await asyncio.to_thread(desktop.press, "enter")
            await asyncio.to_thread(desktop.press, ["ctrl", "s"])
            actual = ""
            for _ in range(20):
                actual = await native.files.read(note, user=ACCOUNT)
                if actual == contents:
                    break
                await asyncio.sleep(.2)
            await self.capture(desktop, "editor.png")
            self.check("gui_typing_and_saved_bytes", actual == contents,
                       sha256=hashlib.sha256(actual.encode()).hexdigest())

            reconnected = await asyncio.to_thread(scoped_desktop_class().connect, handle.sandbox_id,
                                                   **self.provider._options())
            title = await asyncio.to_thread(lambda: reconnected.get_window_title(reconnected.get_current_window_id()))
            self.check("reconnect_observes_same_editor", "note.txt" in title, title=title)
            # Controlled negative fixture only: another account owns a display.
            other = await native.commands.run(
                "Xvfb :1 -screen 0 640x480x24 -nolisten tcp",
                user="user", background=True, timeout=0)
            await other.disconnect()
            for _ in range(20):
                state = await native.commands.run(
                    "xdpyinfo -display :1 >/dev/null 2>&1 && printf ready || true", user="user")
                if state.stdout.strip() == "ready":
                    break
                await asyncio.sleep(.2)
            else:
                raise AssertionError("conflicting_display_fixture_not_ready")
            try:
                await self.provider.prepare(handle)
            except SandboxError:
                self.check("foreign_display_preparation_rejected", True)
            else:
                self.check("foreign_display_preparation_rejected", False)
            self.record["result"] = "passed_provider_probe_not_native_model_acceptance"
            self.persist()
        except BaseException as error:
            self.record["failure"] = {"type":type(error).__name__, "code":getattr(error, "code", None)}
            self.persist()
            raise
        finally:
            await self.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    key = dotenv_values(args.credential_file).get("E2B_API_KEY")
    provider = E2BProvider(E2BConfiguration(api_key=key, request_timeout_seconds=30))
    probe = DesktopProbe(provider, args.directory)
    print(json.dumps({"manifest":str(probe.path)}), flush=True)
    try:
        asyncio.run(probe.exercise())
    except Exception as error:
        print(json.dumps({"passed":False, "error":getattr(error, "code", type(error).__name__),
                          "manifest":str(probe.path)}), flush=True)
        return 1
    print(json.dumps({"passed":True, "checks":probe.record["checks"], "cleanup":probe.record["cleanup"],
                      "manifest":str(probe.path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
