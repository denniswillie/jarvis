"""Fixed maintained-SDK recipe; constructing it does not submit a build."""
from pathlib import Path
import hashlib
import json
import re
import shlex

ROOT = Path(__file__).resolve().parent
RECIPE_FILES = ("template.py", "build.py", "payload/install.sh", "payload/inventory.py", "payload/package_access.py", "payload/render-clip")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(65536):
            digest.update(block)
    return digest.hexdigest()


def preflight(root=ROOT, *, require_complete=False):
    pins = json.loads((root / "pins.json").read_text())
    lock = json.loads((root / "source-lock.json").read_text())
    if (pins["schema"] != "jarvis.clip.sandbox-pins.v1" or pins["platform"] != "linux/amd64"
            or pins["sdk"] != "2.49.1"
            or not re.fullmatch(r"debian:bookworm-slim@sha256:[0-9a-f]{64}", pins["base_image"])
            or not re.fullmatch(r"\d{8}T\d{6}Z", pins["debian_snapshot"])):
        raise ValueError("invalid_base_pin")
    if pins["proposed_build"] != {"count": 1, "cpu_count": 2, "memory_mb": 4096, "min_free_disk_mb": 4096, "observe_seconds": 900}:
        raise ValueError("build_budget_changed")
    for name in ("node", "chrome"):
        version = pins[name]["version"]
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:\.\d+)?", version):
            raise ValueError("invalid_version")
        expected = (f"https://nodejs.org/dist/v{version}/node-v{version}-linux-x64.tar.xz" if name == "node"
                    else f"https://storage.googleapis.com/chrome-for-testing-public/{version}/linux64/chrome-linux64.zip")
        if pins[name]["url"] != expected:
            raise ValueError("invalid_distribution_url")
        digest = pins[name]["sha256"]
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid_distribution_sha256")
    if int(pins["node"]["version"].split(".")[0]) < 22 or pins["node"]["sha256"] is None:
        raise ValueError("node_pin_required")
    if (not pins["chrome"]["generation"].isdigit() or pins["chrome"]["size_bytes"] != 200268898
            or pins["chrome"]["max_bytes"] != 210000000 or pins["node"]["max_bytes"] != 60000000):
        raise ValueError("distribution_bound")
    entries = lock["files"]
    seen = set()
    for entry in entries:
        relative = Path(entry["path"])
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.parts[0] not in {"clip", "font-proof"} or str(relative) in seen):
            raise ValueError("invalid_source_path")
        seen.add(str(relative))
        path = root / "payload" / relative
        if path.resolve() != path or not path.is_file() or path.stat().st_size != entry["size_bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError("source_hash_mismatch")
    for name in ("clip", "font-proof"):
        actual = {str(p.relative_to(root / "payload")) for p in (root / "payload" / name).rglob("*") if p.is_file() or p.is_symlink()}
        if actual != {value for value in seen if value.startswith(name + "/")}:
            raise ValueError("unexpected_source_file")
    copied_files = {str(p.relative_to(root / "payload")) for p in (root / "payload").rglob("*") if p.is_file() or p.is_symlink()}
    if copied_files != seen | {"install.sh", "inventory.py", "package_access.py", "render-clip"}:
        raise ValueError("unexpected_build_context_file")
    source_sha = hashlib.sha256(canonical(entries)).hexdigest()
    if source_sha != lock["source_bundle_sha256"]:
        raise ValueError("source_bundle_mismatch")
    packages = json.loads((root / "payload/clip/package-lock.json").read_text())["packages"]
    for name, version, integrity in ((pins["compositor_package"], pins["remotion"], pins["compositor_integrity"]),
                                    (pins["font_package"], pins["font_version"], pins["font_integrity"])):
        item = packages["node_modules/" + name]
        if item["version"] != version or item["integrity"] != integrity:
            raise ValueError("locked_dependency_mismatch")
    recipe = {"pins": pins, "source_bundle_sha256": source_sha,
              "recipe_files": [{"path": path, "sha256": file_digest(root / path)} for path in RECIPE_FILES]}
    missing = ["chrome_archive_sha256"] if pins["chrome"]["sha256"] is None else []
    if require_complete and missing:
        raise ValueError("chrome_archive_sha256_required_before_build")
    return {"pins": pins, "source_bundle_sha256": source_sha,
            "recipe_sha256": hashlib.sha256(canonical(recipe)).hexdigest(), "missing": missing,
            "runtime_qualified": False}


def make_template(Template, root=ROOT):
    data = preflight(root, require_complete=True)
    snapshot = data["pins"]["debian_snapshot"]
    # Remotion's maintained Debian recipe, fixed to a signed archive snapshot.
    packages = ("bash coreutils passwd ca-certificates curl xz-utils unzip python3 libatomic1 libstdc++6 zlib1g libnss3 libdbus-1-3 libatk1.0-0 "
                "libgbm1 libasound2 libxrandr2 libxkbcommon0 libxfixes3 libxcomposite1 libxdamage1 "
                "libatk-bridge2.0-0 libpango-1.0-0 libcairo2 libcups2")
    apt = ("set -eu; rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; "
           "printf '%s\\n' " + shlex.quote(f"deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/{snapshot}/ bookworm main")
           + " " + shlex.quote(f"deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/{snapshot}/ bookworm-security main")
           + " > /etc/apt/sources.list; apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=30 update; "
           "DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=30 install -y --no-install-recommends "
           + packages + "; rm -rf /var/lib/apt/lists/*")
    return (Template(file_context_path=str(root)).from_image(data["pins"]["base_image"])
            .run_cmd("timeout 240 /bin/sh -c " + shlex.quote(apt), user="root")
            .copy("payload/", "/opt/jarvis-render/", user="root:root", resolve_symlinks=False)
            .copy("pins.json", "/opt/jarvis-render/pins.json", user="root:root")
            .copy("source-lock.json", "/opt/jarvis-render/source-lock.json", user="root:root")
            .run_cmd("timeout 420 /bin/bash /opt/jarvis-render/install.sh", user="root"))
