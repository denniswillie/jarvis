"""Run the actual opt-in Docker deployment with owned secrets and test data."""
from contextlib import contextmanager
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import httpx

from app.settings import JarvisSettings
from evals.second_brain.hindsight_policy_trace import exercise
from evals.second_brain.hindsight_service import save


REPOSITORY = Path(__file__).resolve().parents[3]


def command(arguments, *, env=None, timeout=90):
    result = subprocess.run(arguments, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("owned_deployment_command_failed")
    return result.stdout


@contextmanager
def deployment_service(root, api_key):
    root = Path(root).resolve()
    if root.parent != Path("/tmp").resolve() or not root.name.startswith("jarvis-m02-deployment."):
        raise ValueError("Use a new private /tmp/jarvis-m02-deployment.* directory")
    root.mkdir(mode=0o700)
    project = "jarvis-m02-" + secrets.token_hex(8)
    manifest = {"format": "jarvis.m02.deployment.v1", "project": project, "status": "allocating",
                "containers": [], "secrets_removed": False}
    manifest_path = root/"deployment-resources.json"
    save(manifest_path, manifest)
    secret_dir = root/"secrets"
    secret_dir.mkdir(mode=0o700)
    token, password = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    secret_paths = []
    env = os.environ | {"JARVIS_HINDSIGHT_SECRET_DIR": str(secret_dir)}
    override = root/"ports.yml"
    override.write_text('services:\n  hindsight-api:\n    ports: !override\n      - "127.0.0.1::8888"\n')
    base = ["docker", "compose", "-p", project, "-f",
            str(REPOSITORY/"deployment/docker-compose.knowledge.yml"), "-f", str(override)]
    allocated = False
    down = False
    try:
        existing = command(["docker", "ps", "-a", "--filter", "label=com.docker.compose.project="+project, "--format", "{{.ID}}"])
        if existing.strip():
            raise RuntimeError("owned_deployment_project_already_exists")
        for name, value in (("db-password", password), ("api-key", token), ("openai-api-key", api_key)):
            path = secret_dir/name
            with path.open("x") as handle:
                secret_paths.append(path)
                handle.write(value+"\n")
            # The parent is 0700. The bind-mounted file must be readable by the
            # container's non-root UID; Compose does not remap bind-file owners.
            path.chmod(0o644)
        configuration = json.loads(command([*base, "config", "--format", "json"], env=env))
        ports = configuration["services"]["hindsight-api"]["ports"]
        if len(ports) != 1 or ports[0].get("host_ip") != "127.0.0.1" or configuration["services"]["hindsight-db"].get("ports"):
            raise RuntimeError("owned_deployment_network_not_private")
        manifest["image_id"] = command(["docker", "image", "inspect", "jarvis-hindsight:0.9.2-policy-v1",
                                         "--format", "{{.Id}}"]).strip()
        allocated = True
        command([*base, "up", "-d", "--no-build", "--pull", "never"], env=env, timeout=120)
        identities = command([*base, "ps", "-a", "-q"], env=env).split()
        inspected = json.loads(command(["docker", "inspect", *identities]))
        if len(inspected) != 2 or any(item["Config"]["Labels"].get("com.docker.compose.project") != project for item in inspected):
            raise RuntimeError("owned_deployment_identity_mismatch")
        manifest["containers"] = [
            {"id": item["Id"], "name": item["Name"], "image_id": item["Image"],
             "service": item["Config"]["Labels"]["com.docker.compose.service"]} for item in inspected
        ]
        address = command([*base, "port", "hindsight-api", "8888"], env=env).strip()
        manifest["url"] = "http://" + address
        save(manifest_path, manifest)
        deadline = time.monotonic()+180
        with httpx.Client(timeout=3, trust_env=False) as client:
            while True:
                try:
                    response = client.get(manifest["url"]+"/openapi.json", headers={"Authorization": "Bearer "+token})
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("owned_deployment_startup_deadline")
                time.sleep(0.5)
        manifest["status"] = "running"
        save(manifest_path, manifest)
        yield {"url": manifest["url"], "token": token}
    finally:
        try:
            if allocated:
                try:
                    log = command([*base, "logs", "--no-color"], env=env, timeout=30)
                    for value in (api_key, token, password):
                        log = log.replace(value, "[redacted]")
                    (root/"deployment.log").write_text(log)
                except Exception:
                    manifest["log_capture"] = "unavailable"
                try:
                    command([*base, "down", "--volumes", "--remove-orphans"], env=env, timeout=120)
                    down = True
                finally:
                    manifest["status"] = "cleaned" if down else "cleanup_unverified"
            else:
                down = True
                manifest["status"] = "startup_failed_before_allocation"
            if down:
                for path in secret_paths:
                    path.unlink(missing_ok=True)
                secret_dir.rmdir()
                manifest["secrets_removed"] = True
        finally:
            if down and not manifest["secrets_removed"]:
                manifest["status"] = "cleanup_unverified"
            save(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    settings = JarvisSettings()
    report = {"stage_accepted": False, "checks": {}}
    try:
        with deployment_service(args.root, settings.OPENAI_API_KEY) as service:
            report = asyncio.run(exercise(args.root, service, settings))
    except Exception as error:
        report["deployment_error"] = {"type": type(error).__name__}
    finally:
        if args.root.exists():
            save(args.root/"deployment-result.json", report)
    print(json.dumps({"root": str(args.root), "checks": report["checks"], "error": report.get("error"),
                      "deployment_error": report.get("deployment_error")}))
    return 0 if report["checks"] and all(report["checks"].values()) and not report.get("error") and not report.get("deployment_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
