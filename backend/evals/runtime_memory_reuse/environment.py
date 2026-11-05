"""Run-owned local services only; no production application imports or DSNs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
import urllib.request

HONCHO = "ghcr.io/plastic-labs/honcho@sha256:b7b98f20c429758df433da14b26a7e9b6ed6540ea8ef020523d67e67217e1e8a"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def root_path(value):
    root = Path(value).resolve()
    if not root.name.startswith("jarvis-runtime-reuse.") or root.parent != Path("/tmp").resolve():
        raise ValueError("Expected a dedicated /tmp/jarvis-runtime-reuse.* directory")
    if not root.is_dir():
        raise ValueError("Create the private experiment root first")
    return root


def credentials():
    from dotenv import dotenv_values
    values = dotenv_values("/Users/denniswillie/gitbio/jarvis/backend/.env")
    key = values.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Existing OpenAI key unavailable")
    return key


def isolated_env(root):
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root / "home"), "XDG_CACHE_HOME": str(root / "cache"),
        "XDG_CONFIG_HOME": str(root / "config"), "TMPDIR": str(root),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    }


def docker(*args, env=None, timeout=180):
    result = subprocess.run(["docker", *args], env=env, text=True,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        # Docker errors can echo configuration; retain privately, not in console.
        raise RuntimeError("Docker operation failed: " + result.stderr[-1200:])
    return result.stdout.strip()


def start(root):
    manifest_path = root / "resources.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("containers") or manifest.get("processes"):
        raise RuntimeError("Root already contains recorded resources; cleanup before retry")
    prefix = "jarvis-reuse-" + root.name.split(".")[-1].lower()
    manifest.update(prefix=prefix, purpose="runtime_memory_reuse", status="allocating")
    save(manifest_path, manifest)
    key = credentials()
    password = secrets.token_urlsafe(18)
    env = os.environ.copy()
    env.update(POSTGRES_PASSWORD=password, LLM_OPENAI_API_KEY=key)

    def allocate(name, *args):
        manifest["containers"].append(name)
        save(manifest_path, manifest)
        return docker("run", "-d", "--name", name, "--label", "jarvis.reuse=" + prefix, *args, env=env)

    network = prefix + "-net"
    manifest["network"] = network
    save(manifest_path, manifest)
    docker("network", "create", "--label", "jarvis.reuse=" + prefix, network)
    pg = prefix + "-pg"
    allocate(pg, "--network", network, "--network-alias", "reuse-db", "--memory", "768m",
             "--cpus", "1", "--tmpfs", "/var/lib/postgresql/data:rw,size=512m",
             "-p", "127.0.0.1::5432", "-e", "POSTGRES_PASSWORD", "-e", "POSTGRES_DB=honcho",
             "pgvector/pgvector:pg16")
    for _ in range(40):
        try:
            docker("exec", pg, "pg_isready", "-U", "postgres", "-d", "honcho", timeout=10)
            break
        except RuntimeError:
            time.sleep(1)
    docker("exec", pg, "psql", "-U", "postgres", "-d", "honcho", "-c", "CREATE EXTENSION IF NOT EXISTS vector;")
    docker("exec", pg, "psql", "-U", "postgres", "-d", "honcho", "-c", "CREATE DATABASE hindsight;")
    docker("exec", pg, "psql", "-U", "postgres", "-d", "hindsight", "-c", "CREATE EXTENSION IF NOT EXISTS vector;")
    pg_port = docker("port", pg, "5432/tcp").rsplit(":", 1)[1]
    token = secrets.token_urlsafe(24)
    config = {
        "hindsight_url": "http://127.0.0.1:18888", "hindsight_token": token,
        "honcho_url": "http://127.0.0.1:18081", "model": "gpt-5.4-mini",
        "pg_port": int(pg_port), "pg_password": password,
    }
    save(root / "private-config.json", config)
    henv = isolated_env(root)
    henv.update({
        "HINDSIGHT_API_DATABASE_URL": f"postgresql://postgres:{password}@127.0.0.1:{pg_port}/hindsight",
        "HINDSIGHT_API_LLM_PROVIDER": "openai-responses", "HINDSIGHT_API_LLM_API_KEY": key,
        "HINDSIGHT_API_LLM_MODEL": config["model"], "HINDSIGHT_API_LLM_MAX_CONCURRENT": "1",
        "HINDSIGHT_API_LLM_MAX_RETRIES": "0", "HINDSIGHT_API_LLM_TIMEOUT": "90",
        "HINDSIGHT_API_LLM_TEMPERATURE": "none", "HINDSIGHT_API_LLM_REASONING_EFFORT": "low",
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai", "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": key,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": "text-embedding-3-small",
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf", "HINDSIGHT_API_ENABLE_RERANKING": "false",
        "HINDSIGHT_API_TENANT_EXTENSION": "hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension",
        "HINDSIGHT_API_TENANT_API_KEY": token,
    })
    log = open(root / "hindsight-service.log", "w")
    process = subprocess.Popen([str(root / "hindsight-env/bin/hindsight-api"), "--host", "127.0.0.1", "--port", "18888"],
                               cwd=root, env=henv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    manifest["processes"].append({"pid": process.pid, "executable": str(root / "hindsight-env/bin/hindsight-api")})
    save(manifest_path, manifest)
    db_url = f"postgresql+psycopg://postgres:{password}@reuse-db:5432/honcho"
    common = ["--network", network, "--memory", "1536m", "--cpus", "1", "-e", "LLM_OPENAI_API_KEY",
              "-e", "DB_CONNECTION_URI=" + db_url, "-e", "AUTH_USE_AUTH=false", "-e", "CACHE_ENABLED=false",
              "-e", "DERIVER_WORKERS=1", "-e", "DERIVER_FLUSH_ENABLED=true", "-e", "EMBEDDING_MAX_CONCURRENT_EMBEDDINGS=1",
              "-e", "SENTRY_ENABLED=false", "-e", "TELEMETRY_ENABLED=false"]
    allocate(prefix + "-honcho", *common, "-p", "127.0.0.1:18081:8000", "--entrypoint", "sh",
             HONCHO, "docker/entrypoint.sh")
    # API readiness follows its entrypoint migrations. Do not start the deriver
    # against a partially migrated database.
    for name in ("honcho", "hindsight"):
        deadline = time.monotonic() + 180
        while True:
            try:
                request = urllib.request.Request(config[name + "_url"] + "/openapi.json")
                if name == "hindsight":
                    request.add_header("Authorization", "Bearer " + token)
                with urllib.request.urlopen(request, timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(name + " API readiness deadline exceeded")
            time.sleep(1)
    allocate(prefix + "-deriver", *common, "--entrypoint", "/app/.venv/bin/python", HONCHO, "-m", "src.deriver")
    manifest["status"] = "started"
    save(manifest_path, manifest)
    print(json.dumps({"status": "started", "prefix": prefix, "services": [config["hindsight_url"], config["honcho_url"]]}))


def cleanup(root):
    path = root / "resources.json"
    manifest = json.loads(path.read_text())
    prefix = manifest.get("prefix")
    if prefix != "jarvis-reuse-" + root.name.split(".")[-1].lower():
        raise ValueError("Resource prefix does not match root")
    findings = []
    for item in manifest.get("processes", []):
        pid = int(item["pid"])
        check = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
        if check.returncode == 0:
            if item["executable"] not in check.stdout:
                findings.append("PID identity changed; refused to signal " + str(pid))
                continue
            os.killpg(pid, signal.SIGTERM)
    for name in reversed(manifest.get("containers", [])):
        if not name.startswith(prefix + "-"):
            raise ValueError("Unexpected container identity")
        inspected = subprocess.run(["docker", "inspect", "--format", '{{index .Config.Labels "jarvis.reuse"}}', name],
                                   capture_output=True, text=True)
        if inspected.returncode == 0:
            if inspected.stdout.strip() != prefix:
                raise ValueError("Container ownership label mismatch")
            docker("rm", "-f", "-v", name)
    if manifest.get("network"):
        network = manifest["network"]
        inspected = subprocess.run(["docker", "network", "inspect", "--format", '{{index .Labels "jarvis.reuse"}}', network],
                                   capture_output=True, text=True)
        if inspected.returncode == 0:
            if inspected.stdout.strip() != prefix:
                raise ValueError("Network ownership label mismatch")
            docker("network", "rm", network)
    time.sleep(2)
    for item in manifest.get("processes", []):
        checked = subprocess.run(["ps", "-p", str(item["pid"]), "-o", "command="], capture_output=True, text=True)
        if checked.returncode == 0 and item["executable"] in checked.stdout:
            findings.append("Process still alive: " + str(item["pid"]))
    residue = docker("ps", "-a", "--filter", "label=jarvis.reuse=" + prefix, "--format", "{{.Names}}")
    if residue:
        findings.append("Remaining containers: " + residue)
    manifest.update(status="cleanup_failed" if findings else "cleaned", cleanup_findings=findings,
                    retained="Private reports, dependency environments, downloaded images/source, and provider-home fixtures remain inventoried outside the repository")
    save(path, manifest)
    print(json.dumps({"status": manifest["status"], "findings": findings}))
    if findings:
        raise RuntimeError("Cleanup audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("command", choices=["start", "cleanup"])
    args = parser.parse_args()
    (start if args.command == "start" else cleanup)(root_path(args.root))
