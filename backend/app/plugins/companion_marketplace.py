"""Instantaneous IDE-style extension catalog and root sandbox package installer.

Reads serve directly from owner-scoped RAM backed by working-memory persistence (<1ms).
Installs/uninstalls fence retained sandboxes, close active desktop viewer streams,
pause running VMs, execute noninteractive apt operations as root (keeping the
`jarvis` user zero-sudo), and restore running state cleanly.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import shlex
from uuid import NAMESPACE_URL, uuid5

from app.errors import JarvisHTTPError
from app.plugins.models import OperationView, PluginCard, PluginCatalog
from app.sandbox.models import SandboxError

logger = logging.getLogger(__name__)

PLUGIN_STATE_KEY = "companion_installed_plugins"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


@dataclass(frozen=True)
class ExtensionDescriptor:
    plugin_id: str
    title: str
    subtitle: str
    description: str
    symbol: str
    category_label: str
    core_builtin: bool
    apt_packages: tuple[str, ...]
    verify_bin: str
    install_size_hint: str
    default_version: str | None = None
    legacy_category: str | None = None


CURATED_EXTENSIONS: tuple[ExtensionDescriptor, ...] = (
    ExtensionDescriptor(
        plugin_id="video_editor",
        title="Video Tools (CLI)",
        subtitle="FFmpeg Command Line",
        description="Trim, stitch, caption, mix audio, and render MP4 videos with commands. Not a graphical editor.",
        symbol="film",
        category_label="Media & Video",
        core_builtin=True,
        apt_packages=(),
        verify_bin="ffmpeg",
        install_size_hint="Core",
        default_version="Sandbox-provided FFmpeg",
        legacy_category="video_editing",
    ),
    ExtensionDescriptor(
        plugin_id="browser",
        title="Chromium Browser",
        subtitle="XFCE Desktop Web",
        description="Browse live websites visually on the sandbox desktop.",
        symbol="globe",
        category_label="Web & Desktop",
        core_builtin=True,
        apt_packages=(),
        verify_bin="chromium",
        install_size_hint="Core",
        default_version="Chromium XFCE",
        legacy_category="web_browsing",
    ),
    ExtensionDescriptor(
        plugin_id="files_data",
        title="Python Data Suite",
        subtitle="Workspace & Tables",
        description="Analyze files, CSVs, spreadsheets, scripts, and calculations.",
        symbol="doc.on.doc",
        category_label="Data & Code",
        core_builtin=True,
        apt_packages=(),
        verify_bin="python3",
        install_size_hint="Core",
        default_version="Python 3.12",
        legacy_category="files_data",
    ),
    ExtensionDescriptor(
        plugin_id="shotcut",
        title="Shotcut Video Editor",
        subtitle="Visual Timeline Editor",
        description="Edit video visibly on the sandbox desktop with a timeline, preview, filters, and export controls.",
        symbol="film.stack",
        category_label="Creative Desktop",
        core_builtin=False,
        apt_packages=("shotcut",),
        verify_bin="shotcut",
        install_size_hint="Includes desktop dependencies",
    ),
    ExtensionDescriptor(
        plugin_id="gimp",
        title="GIMP Image Editor",
        subtitle="Desktop Graphics App",
        description="Full XFCE desktop raster photo retouching and layered graphics editor.",
        symbol="paintbrush.pointed",
        category_label="Creative Desktop",
        core_builtin=False,
        apt_packages=("gimp",),
        verify_bin="gimp",
        install_size_hint="~95 MB",
    ),
    ExtensionDescriptor(
        plugin_id="libreoffice",
        title="LibreOffice Suite",
        subtitle="Writer & Calc Desktop",
        description="Open, edit, and export Word documents, spreadsheets, and PDFs visually.",
        symbol="doc.richtext",
        category_label="Office Desktop",
        core_builtin=False,
        apt_packages=("libreoffice-writer", "libreoffice-calc"),
        verify_bin="libreoffice",
        install_size_hint="~140 MB",
    ),
    ExtensionDescriptor(
        plugin_id="vlc",
        title="VLC Media Player",
        subtitle="Desktop Audio/Video",
        description="Play and inspect video and audio streams on the sandbox desktop.",
        symbol="play.rectangle.fill",
        category_label="Media Desktop",
        core_builtin=False,
        apt_packages=("vlc",),
        verify_bin="vlc",
        install_size_hint="~65 MB",
    ),
    ExtensionDescriptor(
        plugin_id="imagemagick",
        title="ImageMagick Studio",
        subtitle="Batch Image CLI",
        description="Convert, resize, watermark, montage, and composite images.",
        symbol="wand.and.stars",
        category_label="Media CLI",
        core_builtin=False,
        apt_packages=("imagemagick",),
        verify_bin="convert",
        install_size_hint="~22 MB",
    ),
    ExtensionDescriptor(
        plugin_id="pandoc",
        title="Pandoc Converter",
        subtitle="Universal Docs CLI",
        description="Transform Markdown, DOCX, HTML, LaTeX, EPUB, and slide decks.",
        symbol="arrow.triangle.2.circlepath",
        category_label="Documents CLI",
        core_builtin=False,
        apt_packages=("pandoc",),
        verify_bin="pandoc",
        install_size_hint="~35 MB",
    ),
    ExtensionDescriptor(
        plugin_id="yt_dlp",
        title="Media Extractor (yt-dlp)",
        subtitle="Stream Fetcher CLI",
        description="Fetch public reference video and audio clips into the workspace.",
        symbol="arrow.down.circle",
        category_label="Media CLI",
        core_builtin=False,
        apt_packages=("yt-dlp",),
        verify_bin="yt-dlp",
        install_size_hint="~12 MB",
    ),
    ExtensionDescriptor(
        plugin_id="sqlite_jq",
        title="SQLite & jq Toolkit",
        subtitle="Database & JSON CLI",
        description="Query SQLite databases, inspect directory trees, and pipe JSON.",
        symbol="cylinder.split.1x2",
        category_label="Developer CLI",
        core_builtin=False,
        apt_packages=("sqlite3", "jq", "tree"),
        verify_bin="sqlite3",
        install_size_hint="~5 MB",
    ),
)

EXTENSIONS_BY_ID: dict[str, ExtensionDescriptor] = {item.plugin_id: item for item in CURATED_EXTENSIONS}


def build_root_install_script(desc: ExtensionDescriptor) -> str:
    if not desc.apt_packages:
        raise ValueError("Core builtins do not use apt installation")
    pkgs = " ".join(shlex.quote(pkg) for pkg in desc.apt_packages)
    primary = shlex.quote(desc.apt_packages[0])
    verify = shlex.quote(desc.verify_bin)
    pid = shlex.quote(desc.plugin_id)
    probe = (
        "/usr/sbin/runuser -u jarvis -- /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin "
        "/bin/sh -c " + shlex.quote("command -v " + verify)
    )
    return "\n".join((
        "set -eu",
        "export DEBIAN_FRONTEND=noninteractive",
        "/usr/bin/install -d -m 0755 -o root -g root /var/lib/jarvis/plugins",
        f"if ! {probe} >/dev/null 2>&1; then",
        "  /usr/bin/apt-get update -qq",
        f"  /usr/bin/apt-get install -y --no-install-recommends {pkgs} >/var/lib/jarvis/plugins/{pid}.install.log 2>&1",
        "fi",
        f"{probe} >/dev/null",
        "/usr/sbin/runuser -u jarvis -- /bin/sh -c 'test \"$(/usr/bin/id -u)\" -ne 0 && ! /usr/bin/sudo -n true 2>/dev/null'",
        f"ver=\"$(/usr/bin/dpkg-query -W -f='${{Version}}' {primary} 2>/dev/null || printf 'installed')\"",
        f"printf '%s\\n' \"$ver\" > /var/lib/jarvis/plugins/{pid}.version",
        "printf 'JARVIS_PLUGIN_OK:%s\\n' \"$ver\"",
    ))


def build_root_remove_script(desc: ExtensionDescriptor) -> str:
    if not desc.apt_packages:
        raise ValueError("Core builtins cannot be removed")
    pkgs = " ".join(shlex.quote(pkg) for pkg in desc.apt_packages)
    pid = shlex.quote(desc.plugin_id)
    return "\n".join((
        "set -eu",
        "export DEBIAN_FRONTEND=noninteractive",
        f"/usr/bin/apt-get remove -y {pkgs} >/var/lib/jarvis/plugins/{pid}.remove.log 2>&1 || true",
        f"/bin/rm -f /var/lib/jarvis/plugins/{pid}.version",
        "printf 'JARVIS_PLUGIN_REMOVED\\n'",
    ))


def _parse_version_marker(stdout: str) -> str:
    for line in (stdout or "").splitlines():
        if line.startswith("JARVIS_PLUGIN_OK:"):
            version = line.split(":", 1)[1].strip()
            if version:
                return version[:64]
    return "installed"


class CompanionPluginService:
    """Owner-scoped RAM + working-memory catalog and real root E2B package installer."""

    def __init__(self, *, owner_id: str, working_memory, runtime_services=None, clock=_now_utc):
        if not owner_id:
            raise ValueError("Owner ID is required")
        self.owner_id = owner_id
        self.working_memory = working_memory
        self.runtime_services = runtime_services
        self.clock = clock
        self._installed: dict[str, dict] = {}
        self._last_errors: dict[str, str] = {}
        self._maintenance: dict = self._idle_maintenance()
        self._persist_lock = asyncio.Lock()
        self._install_lock = asyncio.Lock()
        self._active_task: asyncio.Task | None = None
        self._ready = False

    @staticmethod
    def _idle_maintenance() -> dict:
        return {
            "active": False,
            "plugin_id": None,
            "action": None,
            "phase": None,
            "message": None,
            "paused_sandboxes": 0,
            "closed_streams": 0,
            "started_at": None,
        }

    async def initialize(self) -> None:
        async with self._persist_lock:
            if self._ready:
                return
            row = await self.working_memory.get_owner_working_memory(owner_id=self.owner_id)
            if row is not None and isinstance(row.state, dict):
                saved = row.state.get(PLUGIN_STATE_KEY, {})
                if isinstance(saved, dict) and isinstance(saved.get("installed"), dict):
                    self._installed = {
                        str(k): dict(v)
                        for k, v in saved["installed"].items()
                        if k in EXTENSIONS_BY_ID and not EXTENSIONS_BY_ID[k].core_builtin and isinstance(v, dict)
                    }
            self._ready = True

    async def close(self) -> None:
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _check_owner(self, owner_id: str) -> None:
        if owner_id != self.owner_id:
            raise JarvisHTTPError(403, "plugin_owner_mismatch", "Plugins belong to another account.")

    async def _persist_installed(self) -> None:
        async with self._persist_lock:
            payload = {
                "installed": deepcopy(self._installed),
                "updated_at": _iso(self.clock()),
            }
            for _ in range(5):
                row = await self.working_memory.get_owner_working_memory(owner_id=self.owner_id)
                expected_revision = row.revision if row is not None else 0
                root = deepcopy(dict(row.state)) if row is not None else {}
                root[PLUGIN_STATE_KEY] = payload
                result = await self.working_memory.compare_and_swap_owner_working_memory(
                    owner_id=self.owner_id,
                    expected_revision=expected_revision,
                    state=root,
                    expires_at=None,
                )
                if result.applied:
                    # Keep CompanionState's in-memory root cache aligned if present.
                    companion = getattr(self.runtime_services, "companion_state", None)
                    if companion is not None and getattr(companion, "_ready", False):
                        companion._root = deepcopy(dict(result.state.state))
                        companion._revision = result.state.revision
                    return
            raise RuntimeError("companion_plugin_state_write_conflict")

    def live_state_projection(self, owner_id: str) -> dict:
        """Pure RAM summary for read_live_state(section='system')."""
        self._check_owner(owner_id)
        installed_ids = [
            item.plugin_id
            for item in CURATED_EXTENSIONS
            if item.core_builtin or item.plugin_id in self._installed
        ]
        result = {"installed": installed_ids}
        if self._maintenance["active"]:
            result["maintenance"] = deepcopy(self._maintenance)
        return result

    def get_extensions_catalog(self, *, owner_id: str) -> dict:
        """Pure RAM read (<1ms) returning Installed and Available extensions."""
        self._check_owner(owner_id)
        now = self.clock()
        installed_cards = []
        available_cards = []
        active_id = self._maintenance["plugin_id"] if self._maintenance["active"] else None

        for desc in CURATED_EXTENSIONS:
            record = self._installed.get(desc.plugin_id)
            is_installed = desc.core_builtin or record is not None
            operation = None
            if active_id == desc.plugin_id:
                operation = {
                    "action": self._maintenance["action"],
                    "phase": self._maintenance["phase"],
                    "message": self._maintenance["message"],
                }
            elif desc.plugin_id in self._last_errors:
                operation = {
                    "action": "install",
                    "phase": "failed",
                    "message": self._last_errors[desc.plugin_id],
                }
            base = {
                "id": desc.plugin_id,
                "title": desc.title,
                "subtitle": desc.subtitle,
                "description": desc.description,
                "symbol": desc.symbol,
                "category_label": desc.category_label,
                "core_builtin": desc.core_builtin,
                "packages": list(desc.apt_packages),
                "install_size_hint": desc.install_size_hint,
                "operation": operation,
            }
            if is_installed:
                installed_cards.append({
                    **base,
                    "installed": True,
                    "version": desc.default_version if desc.core_builtin else (record or {}).get("version", "installed"),
                    "installed_at": None if desc.core_builtin else (record or {}).get("installed_at"),
                    "can_uninstall": not desc.core_builtin and not self._maintenance["active"],
                })
            else:
                available_cards.append({
                    **base,
                    "installed": False,
                    "version": None,
                    "installed_at": None,
                    "can_install": not self._maintenance["active"],
                })

        return {
            "schema_version": 2,
            "evaluated_at": _iso(now),
            "sandbox_maintenance": deepcopy(self._maintenance),
            "installed": installed_cards,
            "available": available_cards,
        }

    def get_legacy_catalog(self, *, owner_id: str) -> PluginCatalog:
        """Keep GET /plugins 200 OK and schema_version=1 compliant in companion mode."""
        self._check_owner(owner_id)
        now = self.clock()
        valid_until = now + timedelta(minutes=10)
        cards = []
        for desc in CURATED_EXTENSIONS:
            if desc.legacy_category is None:
                continue
            cfg_id = uuid5(NAMESPACE_URL, f"jarvis-companion-plugin:{self.owner_id}:{desc.plugin_id}")
            op = None
            if self._maintenance["active"]:
                op = OperationView(
                    operation_ref=f"maint-{self._maintenance['plugin_id']}",
                    configuration_id=cfg_id,
                    phase="installing",
                )
            cards.append(PluginCard(
                plugin_id=desc.plugin_id,  # type: ignore[arg-type]
                title=desc.title,
                description=desc.description,
                category=desc.legacy_category,  # type: ignore[arg-type]
                selected_default=True,
                desired_enabled=True,
                configuration_id=cfg_id,
                installed="installed",
                installed_version="core-1",
                readiness="unavailable" if op is not None else "ready",
                reason_code="plugin_change_pending" if op is not None else None,
                observations_valid_until=valid_until,
                operation=op,
                can_change=False,
            ))
        return PluginCatalog(
            evaluated_at=now,
            policy_revision=1,
            policy_available=True,
            items=tuple(cards),
        )

    async def start_install(self, *, owner_id: str, plugin_id: str) -> dict:
        self._check_owner(owner_id)
        desc = EXTENSIONS_BY_ID.get(plugin_id)
        if desc is None:
            raise JarvisHTTPError(404, "plugin_not_found", "Unknown plugin extension.")
        if desc.core_builtin or plugin_id in self._installed:
            return self.get_extensions_catalog(owner_id=owner_id)
        if self._install_lock.locked() or self._maintenance["active"]:
            raise JarvisHTTPError(409, "plugin_maintenance_active",
                                  "Another plugin installation is already pausing/updating sandboxes.")
        self._last_errors.pop(plugin_id, None)
        self._maintenance = {
            "active": True,
            "plugin_id": plugin_id,
            "action": "install",
            "phase": "pausing_sandboxes",
            "message": f"Pausing sandboxes & closing live streams to install {desc.title}…",
            "paused_sandboxes": 0,
            "closed_streams": 0,
            "started_at": _iso(self.clock()),
        }
        self._active_task = asyncio.create_task(
            self._execute_maintenance(desc=desc, action="install"),
            name=f"plugin-install-{plugin_id}",
        )
        return self.get_extensions_catalog(owner_id=owner_id)

    async def start_uninstall(self, *, owner_id: str, plugin_id: str) -> dict:
        self._check_owner(owner_id)
        desc = EXTENSIONS_BY_ID.get(plugin_id)
        if desc is None:
            raise JarvisHTTPError(404, "plugin_not_found", "Unknown plugin extension.")
        if desc.core_builtin:
            raise JarvisHTTPError(422, "plugin_core_builtin", "Core built-in capabilities cannot be removed.")
        if plugin_id not in self._installed:
            return self.get_extensions_catalog(owner_id=owner_id)
        if self._install_lock.locked() or self._maintenance["active"]:
            raise JarvisHTTPError(409, "plugin_maintenance_active",
                                  "Another plugin operation is currently updating sandboxes.")
        self._last_errors.pop(plugin_id, None)
        self._maintenance = {
            "active": True,
            "plugin_id": plugin_id,
            "action": "remove",
            "phase": "pausing_sandboxes",
            "message": f"Pausing sandboxes & closing live streams to remove {desc.title}…",
            "paused_sandboxes": 0,
            "closed_streams": 0,
            "started_at": _iso(self.clock()),
        }
        self._active_task = asyncio.create_task(
            self._execute_maintenance(desc=desc, action="remove"),
            name=f"plugin-remove-{plugin_id}",
        )
        return self.get_extensions_catalog(owner_id=owner_id)

    def _close_owner_desktop_streams(self) -> int:
        views = getattr(self.runtime_services, "sandbox_views", None)
        controls = getattr(self.runtime_services, "app_controls", None)
        if views is None:
            return 0
        closed_count = 0
        for lease in list(getattr(views, "leases", {}).values()):
            if getattr(lease.target, "owner_id", None) != self.owner_id:
                continue
            if views.stop(self.owner_id, lease.id):
                closed_count += 1
                if controls is not None:
                    controls._notify(
                        lease.target.session_id,
                        {"type": "workspace.sandbox.closed", "view_id": str(lease.id)},
                    )
        return closed_count

    @staticmethod
    def _remember_viewer_state(views, binding, info=None, *, state: str) -> None:
        access = getattr(views, "access", None)
        remember = getattr(access, "_remember", None)
        if callable(remember):
            try:
                remember(binding, info, state=state)
            except Exception:
                pass

    async def _execute_maintenance(self, *, desc: ExtensionDescriptor, action: str) -> None:
        async with self._install_lock:
            views = getattr(self.runtime_services, "sandbox_views", None)
            hermes_runtime = getattr(self.runtime_services, "hermes_profile_runtime", None)
            sandboxes = getattr(hermes_runtime, "sandboxes", None)
            live_targets: list[tuple[str, bool]] = []
            detected_version: str | None = None
            try:
                closed_streams = self._close_owner_desktop_streams()
                self._maintenance["closed_streams"] = closed_streams

                async with AsyncExitStack() as stack:
                    if sandboxes is not None:
                        session_ids = sorted(sandboxes.bindings.bound_session_ids())
                        for sid in session_ids:
                            await stack.enter_async_context(sandboxes._lock(sid))
                        for sid in session_ids:
                            try:
                                binding, info = await sandboxes._existing(sandboxes._required(sid))
                            except (SandboxError, TimeoutError, OSError):
                                continue
                            was_running = info.state == "running"
                            if was_running:
                                paused = await sandboxes._lifecycle(sid, "pause")
                                self._remember_viewer_state(views, sandboxes._required(sid), paused, state="paused")
                                self._maintenance["paused_sandboxes"] += 1
                            live_targets.append((sid, was_running))

                    self._maintenance["phase"] = "installing" if action == "install" else "removing"
                    self._maintenance["message"] = (
                        f"Installing {desc.title} in sandbox…"
                        if action == "install"
                        else f"Removing {desc.title} from sandbox…"
                    )
                    script = build_root_install_script(desc) if action == "install" else build_root_remove_script(desc)

                    try:
                        for sid, _ in live_targets:
                            resumed = await sandboxes._lifecycle(sid, "resume")
                            box = await sandboxes.provider._connect(resumed)
                            result = await sandboxes.provider._run_privileged(box, script, timeout=240)
                            if action == "install":
                                detected_version = _parse_version_marker(result.stdout)
                    finally:
                        if live_targets and sandboxes is not None:
                            self._maintenance["phase"] = "resuming_sandboxes"
                            self._maintenance["message"] = "Resuming sandboxes…"
                            for sid, was_running in live_targets:
                                try:
                                    if was_running:
                                        current = await sandboxes._lifecycle(sid, "resume")
                                        self._remember_viewer_state(
                                            views, sandboxes._required(sid), current, state="running"
                                        )
                                    else:
                                        paused = await sandboxes._lifecycle(sid, "pause")
                                        self._remember_viewer_state(
                                            views, sandboxes._required(sid), paused, state="paused"
                                        )
                                except Exception:
                                    logger.warning("Sandbox post-plugin state restore failed for session=%s", sid)

                if action == "install":
                    self._installed[desc.plugin_id] = {
                        "version": detected_version or "Ready on wake",
                        "installed_at": _iso(self.clock()),
                    }
                else:
                    self._installed.pop(desc.plugin_id, None)
                await self._persist_installed()
                self._last_errors.pop(desc.plugin_id, None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Plugin %s failed for %s (%s)", action, desc.plugin_id, type(exc).__name__)
                self._last_errors[desc.plugin_id] = f"Could not {action} {desc.title}. Tap Install to retry."
            finally:
                self._maintenance = self._idle_maintenance()

    async def reconcile_sandbox(self, handle) -> None:
        """Lazily install any user-selected extensions onto a newly created/woken sandbox."""
        if not self._installed:
            return
        hermes_runtime = getattr(self.runtime_services, "hermes_profile_runtime", None)
        sandboxes = getattr(hermes_runtime, "sandboxes", None)
        if sandboxes is None:
            return
        pending = [EXTENSIONS_BY_ID[pid] for pid in list(self._installed) if pid in EXTENSIONS_BY_ID]
        if not pending:
            return
        box = await sandboxes.provider._connect(handle)
        updated = False
        for desc in pending:
            if desc.core_builtin:
                continue
            script = build_root_install_script(desc)
            result = await sandboxes.provider._run_privileged(box, script, timeout=240)
            ver = _parse_version_marker(result.stdout)
            if self._installed.get(desc.plugin_id, {}).get("version") in {None, "Ready on wake"} and ver:
                self._installed[desc.plugin_id]["version"] = ver
                updated = True
        if updated:
            await self._persist_installed()
