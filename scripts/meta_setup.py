#!/usr/bin/env python3
"""Prepare an isolated Meta profile; explicitly launch existing services later.

No model calls, database access, background processes or changes to the template
profile occur during prepare. No history, sandbox bindings or .env is copied from
the template profile. This is configuration wiring, not another agent runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from dotenv import dotenv_values
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def private_write(path: Path, content: str | bytes, *, mode: int = 0o600):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        os.chmod(path, mode)
        stream.write(content.encode() if isinstance(content, str) else content)


def write_env(path: Path, values: dict):
    private_write(path, "".join(f"{key}={json.dumps(str(value))}\n"
                               for key, value in sorted(values.items()) if value is not None))


def read_key(path: Path) -> str:
    value = path.read_text().strip()
    if not value or value.startswith("REPLACE_"):
        raise ValueError(f"Credential is not configured: {path}")
    return value


def prepare(args):
    from app.shared_memory.client import BankScope

    home = args.home.resolve()
    template = args.template_profile.resolve() if args.template_profile else None
    source, python = args.hermes_source.resolve(), args.hermes_python.absolute()
    if args.port in (9000, 8888) or args.hindsight_port in (9000, 8888, args.port):
        raise ValueError("Choose separate candidate ports; 9000 and 8888 are reserved for the baseline")
    if not 1024 <= args.port <= 65535 or not 1024 <= args.hindsight_port <= 65535:
        raise ValueError("Ports must be between 1024 and 65535")
    if home.exists():
        raise ValueError("Candidate directory already exists; never overwrite retained state")
    if not (source / "gateway/run.py").is_file() or not python.is_file():
        raise ValueError("An installed, pinned Hermes source and interpreter are required")
    plugin = args.hindsight_plugin.resolve() if args.hindsight_plugin else template / "plugins/hindsight"
    if not plugin.is_dir():
        raise ValueError("Template needs the installed official Hindsight 1.2.3 plugin")
    env = {}
    for file in args.base_env:
        env.update({k: v for k, v in dotenv_values(file, interpolate=False).items() if v is not None})
    openai_key = env.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise ValueError("Base environment needs the existing OpenAI key for Voice and embeddings")
    meta_key = read_key(args.meta_key_file)
    config = yaml.safe_load((template / "config.yaml").read_text()) if template else {
        "approvals": {"mode": "off"}, "tools": {"tool_search": {"enabled": False}}}
    adapter = json.loads((template / "jarvis-adapter.json").read_text()) if template else {
        "launch": {"source": "cli", "toolsets": ["terminal", "jarvis_work", "jarvis_sandbox", "jarvis_domain", "memory"]}}
    model_config = yaml.safe_load((ROOT / "deployment/hermes-meta-provider.example.yaml").read_text())
    owner = args.owner_id or str(uuid4())
    ffprobe = Path(env.get("JARVIS_RECORDING_SYNC_FFPROBE") or shutil.which("ffprobe") or "")
    if not ffprobe.is_file():
        raise ValueError("Install FFmpeg first: ffprobe must be on PATH or set JARVIS_RECORDING_SYNC_FFPROBE")
    ffprobe = ffprobe.resolve()
    with ffprobe.open("rb") as stream:
        ffprobe_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    generation = "meta-demo-v1"
    bank = BankScope.for_owner(owner, generation=generation)
    profile = home / "profile"
    secret_dir = home / "secrets"
    api_key = secrets.token_urlsafe(32)

    # Only explicitly selected configuration and plugin code cross the boundary.
    # Do not inherit alternative endpoint credentials or automatic memory hooks.
    config.pop("providers", None)
    config.update(model_config)
    config["gateway"] = {"multiplex_profiles": False, "managed_workers": False}
    config["plugins"] = {"enabled": ["jarvis-retained-work", "hindsight"], "isolation": "in_process"}
    config["terminal"] = {"backend": "jarvis_e2b", "cwd": "/home/jarvis/workspace", "timeout": 120}
    config["memory"] = {"provider": "hindsight", "memory_enabled": False, "user_profile_enabled": False}
    if template is None:
        from app.hermes_sessions.profile_guidance import with_profile_guidance
        config = with_profile_guidance(config, callbacks_enabled=True, sandbox_controls=True)
    adapter.update(owner_id=owner, domain_tools=True, retained_terminal=True,
                   sandbox_controls=True, native_live_guidance=True)
    adapter["launch"].update(cwd=str(profile), model="muse-wonder-1.0", reasoning="low", max_turns=40)
    adapter["sandbox_policy"] = {"template": "desktop", "lifetime_seconds": 3600,
                                 "command_seconds": 120, "internet_access": True}

    # Deliberately do not inherit the working database, public callback, user ID,
    # bearer token, media process or mutable checkpoint/secret directories.
    env.update(
        JARVIS_OWNER_ID=owner, JARVIS_STATIC_BEARER_TOKEN=secrets.token_urlsafe(32),
        SUPABASE_DB_URL=read_key(args.database_url_file) if args.database_url_file else "",
        PUBLIC_BASE_URL=args.public_base_url or "https://meta-setup.invalid",
        JARVIS_ROUTER_PROVIDER="meta", JARVIS_META_ROUTER_MODEL="muse-spark-1.3",
        JARVIS_META_API_KEY_FILE=str(secret_dir / "meta-api-key"),
        JARVIS_PERCEPTION_PROVIDER="gemini", JARVIS_COMPANION_STATE_ENABLED="true",
        OPENAI_BASE_URL="https://api.openai.com/v1", OPENAI_LIVE_ENABLED="true",
        RESPONSIVE_VOICE_ENABLED="true", OPENAI_LIVE_MODEL="gpt-live-1",
        JARVIS_HERMES_PROFILE_HOME=str(profile), JARVIS_NATIVE_MEMORY_ENABLED="true",
        JARVIS_HINDSIGHT_URL=f"http://127.0.0.1:{args.hindsight_port}",
        JARVIS_HINDSIGHT_BANK_GENERATION=generation,
        JARVIS_HINDSIGHT_API_KEY_FILE=str(secret_dir / "api-key"),
        JARVIS_NATIVE_MEMORY_CHECKPOINT_FILE=str(home / "memory/cursor.sqlite"),
        SANDBOX_VIEW_SECRET_ROOT=str(home / "view-secrets"),
        JARVIS_STATIC_INSTRUCTIONS_ONLY="1", JARVIS_SOLOIST_HOME="",
        SECOND_BRAIN_ENABLED="true", SECOND_BRAIN_RECORDINGS_ENABLED="true",
        SECOND_BRAIN_TASKS_ENABLED="false", SECOND_BRAIN_FRONTIER_ENABLED="false",
        SECOND_BRAIN_KNOWLEDGE_ENABLED="false", SECOND_BRAIN_MEMORY_AUTHORITY="legacy",
        SECOND_BRAIN_RECORDING_AREA_ID=str(uuid4()),
        SECOND_BRAIN_RECORDING_FFPROBE_PATH=str(ffprobe),
        SECOND_BRAIN_RECORDING_FFPROBE_SHA256=ffprobe_sha256,
        SECOND_BRAIN_RECORDING_STORAGE_MAX_FILE_BYTES=env.get("SECOND_BRAIN_RECORDING_STORAGE_MAX_FILE_BYTES", "500000000"),
        JARVIS_RECORDING_SYNC_FFPROBE=str(ffprobe),
        MEMORY_JOB_RUNNER_ENABLED="false", MEMORY_MAINTENANCE_ENABLED="false",
        REMINDER_SCHEDULER_ENABLED="false",
    )
    home.mkdir(parents=True, mode=0o700)
    for directory in (profile, secret_dir, home / "user", home / "view-secrets"):
        directory.mkdir(mode=0o700)
    private_write(profile / "config.yaml", yaml.safe_dump(config, sort_keys=False))
    private_write(profile / "jarvis-adapter.json", json.dumps(adapter, indent=2) + "\n")
    private_write(profile / "jarvis/callback.key", os.urandom(32))
    private_write(profile / "hindsight/config.json", json.dumps({
        "mode": "local_external", "api_url": env["JARVIS_HINDSIGHT_URL"],
        "bank_id": bank.bank_id, "memory_mode": "tools", "auto_retain": False,
        "auto_recall": False, "retain_async": True, "timeout": 120,
        "mirror_to_own_bank": False, "additional_banks": [],
        "recall_additional_banks": [], "trusted_project_dirs": [],
    }, indent=2) + "\n")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".env")
    shutil.copytree(ROOT / "deployment/hermes_session_plugin", profile / "plugins/jarvis-retained-work", ignore=ignore)
    shutil.copytree(plugin, profile / "plugins/hindsight", ignore=ignore)
    soul = template / "SOUL.md" if template else None
    if soul is not None and soul.is_file():
        private_write(profile / "SOUL.md", soul.read_bytes())
    else:
        private_write(profile / "SOUL.md", "You are Jarvis's background work agent. Use your tools to inspect "
            "current state, relevant memories, files and ongoing work when needed. Preserve the user's "
            "original requests and apply live guidance. Publish an interim result when requested without "
            "closing the workspace. Report actual outcomes, not merely attempted actions.\n")
    # Container-readable mounts remain inside a mode-0700 host directory.
    for name, value in {"meta-api-key": meta_key, "openai-api-key": openai_key,
                        "api-key": api_key, "db-password": secrets.token_urlsafe(32)}.items():
        private_write(secret_dir / name, value + "\n", mode=0o444)
    write_env(home / "backend.env", env)
    project = "jarvis-meta-" + owner.replace("-", "")[:12].lower()
    write_env(home / "compose.env", {"JARVIS_HINDSIGHT_SECRET_DIR": secret_dir,
        "JARVIS_HINDSIGHT_PORT": args.hindsight_port, "JARVIS_KNOWLEDGE_NETWORK": project + "-clients"})
    private_write(home / "setup.json", json.dumps({
        "repo": str(ROOT), "source": str(source), "python": str(python), "bank_id": bank.bank_id,
        "pythonpath": [str(p.resolve()) for p in args.hermes_pythonpath],
        "port": args.port, "compose_project": project,
    }, indent=2) + "\n")
    print(f"Prepared {home}; no services started. Meta bank: {bank.bank_id}")
    print("Configure a separate database and callback origin in backend.env before launching the API.")
    print(f"Hindsight Compose project: {project}. See deployment/META_SETUP.md for explicit start commands.")


def load_setup(home: Path):
    config = json.loads((home / "setup.json").read_text())
    if Path(config["repo"]).resolve() != ROOT:
        raise ValueError("Run the launcher from the checkout that prepared this candidate")
    values = {k: v for k, v in dotenv_values(home / "backend.env", interpolate=False).items() if v is not None}
    return config, values


def process_base():
    # In particular, never inherit provider keys or another HERMES_HOME.
    return {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ", "TMPDIR") if key in os.environ}


def gateway(home, config, values):
    if not (home / "profile/jarvis/domain-tools.json").is_file():
        raise ValueError("Start the candidate API first and wait for startup before starting the Hermes gateway")
    env = process_base()
    env.update(HOME=str(home / "user"), HERMES_HOME=str(home / "profile"),
        XDG_CONFIG_HOME=str(home / "user/.config"), XDG_CACHE_HOME=str(home / "user/.cache"),
        PYTHONUNBUFFERED="1", JARVIS_STATIC_INSTRUCTIONS_ONLY="1", TERMINAL_ENV="jarvis_e2b",
        OPENAI_API_KEY=read_key(Path(values["JARVIS_META_API_KEY_FILE"])),
        OPENAI_BASE_URL="https://api.meta.ai/v1", HINDSIGHT_API_URL=values["JARVIS_HINDSIGHT_URL"],
        HINDSIGHT_API_KEY=read_key(Path(values["JARVIS_HINDSIGHT_API_KEY_FILE"])),
        HINDSIGHT_BANK_ID=config["bank_id"],
        PYTHONPATH=os.pathsep.join([*config["pythonpath"], config["source"], str(ROOT / "backend")]))
    os.chdir(config["source"])
    os.execve(config["python"], [config["python"], "-m", "gateway.run"], env)


def api(home, config, values):
    url = urlsplit(values.get("PUBLIC_BASE_URL", ""))
    if not values.get("SUPABASE_DB_URL") or url.scheme != "https" or not url.hostname or url.hostname.endswith(".invalid"):
        raise ValueError("Configure the candidate database and HTTPS callback origin before starting its API")
    env = process_base()
    env.update(values)
    env.update(PYTHONPATH=str(ROOT / "backend"), PYTHONUNBUFFERED="1")
    os.chdir(ROOT / "backend")
    os.execve(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(config["port"])], env)


async def init_db(values):
    """Initialize only an empty schema, or finish provisioning a ready one."""
    import asyncpg
    from app.migrations.runner import apply_migrations, read_schema_readiness
    from app.settings import is_configured_secret

    url = values.get("SUPABASE_DB_URL")
    if not is_configured_secret(url):
        raise ValueError("Configure the candidate database before init-db")
    connection = await asyncpg.connect(url, timeout=10, command_timeout=90)
    try:
        ready = await read_schema_readiness(connection)
        if not ready.ready:
            # The existing runner refuses occupied/partial schemas. Upgrading an
            # existing deployment remains a separate, explicit migration task.
            await apply_migrations(connection, fresh=True)
        owner, area = values["JARVIS_OWNER_ID"], UUID(values["SECOND_BRAIN_RECORDING_AREA_ID"])
        async with connection.transaction():
            await connection.execute("""insert into jarvis_knowledge_areas(id,owner_id,name,domain)
                values($1,$2,'Jarvis files','general') on conflict(id) do nothing""", area, owner)
            row = await connection.fetchrow("select owner_id,status from jarvis_knowledge_areas where id=$1", area)
            if row["owner_id"] != owner or row["status"] != "active":
                raise ValueError("Recording area belongs to another owner or is inactive")
        print("Candidate schema and private Files area are ready.")
    finally:
        await connection.close()


async def init_memory(config, values):
    from app.shared_memory.client import BankScope, MemoryRequestError, NativeMemory

    scope = BankScope.for_owner(values["JARVIS_OWNER_ID"], generation=values["JARVIS_HINDSIGHT_BANK_GENERATION"])
    if scope.bank_id != config["bank_id"]:
        raise ValueError("Bank identity changed; prepare another profile instead of splitting shared memory")
    memory = NativeMemory(scope=scope, base_url=values["JARVIS_HINDSIGHT_URL"],
                          api_key=read_key(Path(values["JARVIS_HINDSIGHT_API_KEY_FILE"])))
    try:
        try:
            await memory.create_fresh_bank()
        except MemoryRequestError as error:
            if error.code != "hindsight_bank_already_exists":
                raise
        await memory.seed_mental_models()
        print(f"Bank {scope.bank_id}: setup accepted; any model generation completes asynchronously.")
    finally:
        await memory.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "gateway", "api", "init-db", "init-memory"))
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--base-env", type=Path, action="append", default=[])
    parser.add_argument("--template-profile", type=Path)
    parser.add_argument("--hindsight-plugin", type=Path, help="Pinned plugin directory; avoids needing a template profile")
    parser.add_argument("--hermes-source", type=Path)
    parser.add_argument("--hermes-python", type=Path)
    parser.add_argument("--hermes-pythonpath", type=Path, action="append", default=[])
    parser.add_argument("--meta-key-file", type=Path)
    parser.add_argument("--database-url-file", type=Path)
    parser.add_argument("--public-base-url")
    parser.add_argument("--owner-id")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--hindsight-port", type=int, default=18889)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.base_env or any(getattr(args, key) is None for key in
                ("hermes_source", "hermes_python", "meta_key_file")) or not (args.template_profile or args.hindsight_plugin):
            parser.error("prepare needs --base-env, --hermes-source, --hermes-python, --meta-key-file and either --hindsight-plugin or --template-profile")
        prepare(args)
        return
    home = args.home.resolve()
    config, values = load_setup(home)
    if args.command == "gateway":
        gateway(home, config, values)
    elif args.command == "api":
        api(home, config, values)
    elif args.command == "init-db":
        asyncio.run(init_db(values))
    else:
        asyncio.run(init_memory(config, values))


if __name__ == "__main__":
    main()
