#!/usr/bin/env python3
"""Install pinned upstream sources into a NEW private directory, never this repo.

No credentials, model calls, gateway startup or personal profile access. The
sources retain their upstream licenses. Use --download-only to inspect first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request

import certifi

from stage_hermes_static_prompts import patched

HERMES_SHA = "f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab"
HINDSIGHT_SHA = "d952b6e655fccab4581ee0722b51c5f103873d12"


def fetch(repo, revision, destination, *, subdir=None):
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    with tempfile.TemporaryDirectory(prefix="jarvis-source-") as temporary:
        temp = Path(temporary)
        archive = temp / "source.tar.gz"
        with urllib.request.urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive) as bundle:
            bundle.extractall(temp / "unpacked", filter="data")
        roots = list((temp / "unpacked").iterdir())
        if len(roots) != 1:
            raise ValueError("Unexpected upstream archive layout")
        source = roots[0] / subdir if subdir else roots[0]
        shutil.copytree(source, destination)
        if subdir:
            for license_file in roots[0].glob("LICENSE*"):
                if license_file.is_file():
                    shutil.copy2(license_file, destination / ("UPSTREAM_" + license_file.name))
    return {"repository": repo, "revision": revision, "archive_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--python", default="python3.14")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    root = args.destination.expanduser().resolve()
    if args.install_only:
        manifest = json.loads((root / "sources.json").read_text())
        if manifest["hermes"]["revision"] != HERMES_SHA or manifest["hindsight"]["revision"] != HINDSIGHT_SHA:
            raise ValueError("Source revision mismatch")
    else:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        manifest = {"hermes": fetch("NousResearch/hermes-agent", HERMES_SHA, root / "source"),
            "hindsight": fetch("vectorize-io/hindsight", HINDSIGHT_SHA, root / "hindsight-plugin",
                               subdir="hindsight-integrations/hermes")}
        # Same pinned static-prompt compatibility patch used by the working
        # system. It only changes these freshly downloaded source files.
        for name, path in (("hermes", root / "source/agent/system_prompt.py"),
                           ("hindsight", root / "hindsight-plugin/__init__.py")):
            original = path.read_bytes()
            replacement = patched(name, original)
            path.with_suffix(".py.upstream").write_bytes(original)
            path.write_bytes(replacement)
            manifest[name]["patched_sha256"] = hashlib.sha256(replacement).hexdigest()
        (root / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.download_only:
        print(f"Pinned sources prepared in {root}; nothing installed or started")
        return
    # The pinned upstream lock contains intentional dependency overrides that
    # plain pip cannot resolve. Use its native locked installer, not --no-deps.
    install_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(root / "runtime")}
    subprocess.run([sys.executable, "-m", "uv", "sync", "--project", str(root / "source"),
                    "--frozen", "--no-dev", "--python", args.python], env=install_env, check=True)
    python = root / "runtime/bin/python"
    with (root / "hindsight-plugin/pyproject.toml").open("rb") as stream:
        plugin = tomllib.load(stream)
    if plugin["project"]["version"] != "1.2.3":
        raise ValueError("Unexpected Hindsight plugin version")
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", str(python),
                    *plugin["project"]["dependencies"]], check=True)
    print(f"Native runtime prepared: {root}. No profile, model or server started.")


if __name__ == "__main__":
    main()
