"""Hermes native Docker backend + bounded agent artifact tool, no host shell."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from contracts import assert_tool_receipts, tool_result
from environment import root_path, save
from worker_acceptance import agent, setup


def run(root):
    setup(root)
    from tools.environments.docker import DockerEnvironment
    from tools.registry import registry
    report = {"native_backend": "Hermes DockerEnvironment 0.19.0", "errors": []}
    output = root / "reports/sandbox.json"
    workspace = root / "sandbox-workspace"
    workspace.mkdir(exist_ok=True)
    manifest_path = root / "resources.json"
    manifest = json.loads(manifest_path.read_text())
    name = manifest["prefix"] + "-sandbox"
    manifest["containers"].append(name)
    save(manifest_path, manifest)
    box = instance = None
    try:
        box = DockerEnvironment(
            image="pgvector/pgvector:pg16", cwd="/workspace", timeout=30, cpu=1, memory=256,
            task_id=name, volumes=[str(workspace) + ":/workspace:rw"],
            forward_env=[], env={}, network=False, auto_mount_cwd=False,
            persistent_filesystem=False, persist_across_processes=False,
            extra_args=["--name", name, "--label", "jarvis.reuse=" + manifest["prefix"],
                        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
                        "--tmpfs", "/var/lib/postgresql/data:rw,size=16m"])
        inspected = json.loads(subprocess.run(["docker", "inspect", name], capture_output=True, text=True, check=True).stdout)[0]
        report["container"] = {"name": name, "network": inspected["HostConfig"]["NetworkMode"],
                               "readonly_root": inspected["HostConfig"]["ReadonlyRootfs"],
                               "mounts": inspected["Mounts"],
                               "environment_names": [entry.split("=", 1)[0] for entry in inspected["Config"]["Env"]]}
        if report["container"]["network"] != "none" or not report["container"]["readonly_root"]:
            raise AssertionError("Container isolation configuration was not applied")
        dependencies = (root / "hermes-env").resolve()
        for mount in inspected["Mounts"]:
            if mount["Type"] == "bind":
                source = Path(mount["Source"]).resolve()
                if not source.is_relative_to(root) and not source.is_relative_to(dependencies):
                    raise AssertionError("Unexpected host bind mount")
            elif mount["Type"] == "volume":
                raise AssertionError("Unexpected persistent Docker volume")
        if any(key in report["container"]["environment_names"] for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "SUPABASE_DB_URL")):
            raise AssertionError("Credentials were forwarded into the sandbox")
        report["negative_controls"] = box.execute(
            "test ! -e /Users/denniswillie/gitbio/jarvis/backend/.env && "
            "test -z \"$OPENAI_API_KEY\" && "
            "if (printf denied > /forbidden-write) 2>/dev/null; then exit 9; else printf 'HOST_AND_ROOT_DENIED\\n'; fi",
            cwd="/workspace")
        # The model supplies content, never a host command, destination, or grant.
        def write(arguments, **kwargs):
            if set(arguments) != {"content"} or not isinstance(arguments["content"], str) or len(arguments["content"]) > 12000:
                raise ValueError("Only bounded fixture content is accepted")
            content = arguments["content"]
            encoded = base64.b64encode(content.encode()).decode()
            result = box.execute("printf '%s' '" + encoded + "' | base64 -d > /workspace/report.md", cwd="/workspace")
            artifact = workspace / "report.md"
            if not artifact.exists() or artifact.read_text() != content:
                raise RuntimeError("Sandbox did not create the requested artifact")
            receipt = {"path": "/workspace/report.md", "sha256": hashlib.sha256(content.encode()).hexdigest(),
                       "created": True, "emailed": False, "execution": result}
            report["receipt"] = receipt
            return tool_result(receipt)

        registry.register(name="create_sandbox_report", toolset="final_sandbox", handler=write, schema={
            "name": "create_sandbox_report", "description": "Create one text report through the native isolated Docker backend. No network or email.",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"], "additionalProperties": False}})
        instance = agent(root, "final_sandbox", {"create_sandbox_report"})
        result = instance.run_conversation(user_message="Create a short markdown file explaining why binary-search midpoint calculation needs parentheses around low + high. Use create_sandbox_report. Quote the returned path and SHA-256 afterward. Do not send anything.")
        assert_tool_receipts(result)
        report["output"] = result
        receipt = report.get("receipt", {})
        report["receipt_consumed"] = bool(receipt) and receipt["sha256"] in str(result.get("final_response", ""))
        report["artifact_content"] = (workspace / "report.md").read_text() if (workspace / "report.md").exists() else None
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        if instance is not None:
            instance.close()
        if box is not None:
            box.cleanup(force_remove=True)
            report["native_cleanup_joined"] = box.wait_for_cleanup(timeout=40)
        # Exact controller-owned name also covers constructor failure/recreation.
        subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, text=True, timeout=30)
        report["container_absent"] = subprocess.run(["docker", "inspect", name], capture_output=True, timeout=10).returncode != 0
        save(output, report)
    print(json.dumps({"report": str(output), "errors": report["errors"], "receipt_consumed": report.get("receipt_consumed"),
                      "container_absent": report["container_absent"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    run(root_path(parser.parse_args().root))
