"""Real local source and maintained-language-tool tests; no model or personal repo."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from jarvis_project_connector.__main__ import load_registry, main, save_registry
from jarvis_project_connector.filesystem import ProjectReader, register_root
from jarvis_project_connector.navigation import ProjectInspector
from jarvis_project_connector.protocol import ConnectorError, Limits, ReadRequest


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "controller.py").write_text("class Controller:\n    def move(self):\n        return 7\n")
    (root / "main.py").write_text("from controller import Controller\nrobot = Controller()\nrobot.move()\n")
    return root


def inspector(root, **kwargs):
    return ProjectInspector(ProjectReader(register_root(root, root.name)), **kwargs)


def request(operation, **kwargs):
    return ReadRequest(operation=operation, **kwargs)


@pytest.mark.parametrize("path", ["../other.py", "/etc/passwd", "./controller.py", "a//b.py",
                                   "a/../controller.py", "a\\b.py", ".env", "Credentials.json",
                                   ".git/config", ".SSH/key", "private.pem"])
def test_rejects_unscoped_and_secret_paths(project, path):
    with pytest.raises(ConnectorError):
        inspector(project).inspect(request("read", path=path))


def test_snapshot_omits_symlinks_hardlinks_binary_and_secrets(project, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("synthetic outside secret")
    (project / "escape.py").symlink_to(secret)
    (project / "outside").symlink_to(tmp_path, target_is_directory=True)
    os.link(secret, project / "hardlink.py")
    (project / ".env").write_text("SECRET=fixture")
    (project / "blob.dat").write_bytes(b"\x00\xff")
    value = inspector(project).inspect(request("files"))
    assert {row["path"] for row in value["items"]} == {"controller.py", "main.py"}
    assert not value["scope_complete"]
    assert value["snapshot"]["omitted"]["excluded_file_type"] == 3
    for path in ("escape.py", "outside/outside.txt", "hardlink.py", "blob.dat"):
        with pytest.raises(ConnectorError):
            inspector(project).inspect(request("read", path=path))


def test_case_exact_paths_even_on_case_insensitive_macos(project):
    with pytest.raises(ConnectorError):
        inspector(project).inspect(request("read", path="Controller.py"))


def test_replaced_or_offline_root_is_not_empty_memory(project):
    reader = ProjectReader(register_root(project, "original"))
    moved = project.with_name("moved")
    project.rename(moved)
    with pytest.raises(ConnectorError) as absent:
        reader.read("controller.py")
    assert absent.value.code == "root_unavailable"
    project.mkdir()
    (project / "controller.py").write_text("class Other: pass")
    with pytest.raises(ConnectorError) as replaced:
        reader.read("controller.py")
    assert replaced.value.code == "root_changed"


def test_digest_bound_read_paging_and_changed_snapshot(project):
    client = inspector(project)
    first = client.inspect(request("read", path="controller.py", max_chars=12))
    second = client.inspect(request("read", path="controller.py", offset=first["next_offset"],
                                    expected_digest=first["source"]["content_digest"]))
    assert first["content"] + second["content"] == (project / "controller.py").read_text()
    assert not first["complete"]
    with pytest.raises(ValidationError):
        request("read", path="controller.py", offset=12)
    snapshot = client.reader.snapshot()
    (project / "controller.py").write_text("class Updated: pass\n")
    with pytest.raises(ConnectorError) as changed:
        client.reader.verify(snapshot)
    assert changed.value.code == "source_changed"
    with pytest.raises(ConnectorError):
        client.inspect(request("read", path="controller.py", expected_digest=first["source"]["content_digest"]))
    with pytest.raises(ConnectorError):
        client.inspect(request("files", expected_snapshot=snapshot.snapshot_id))


def test_addition_after_snapshot_invalidates_reference_coverage(project):
    reader = ProjectReader(register_root(project, "project"))
    snapshot = reader.snapshot()
    (project / "new_reference.py").write_text("from controller import Controller\n")
    with pytest.raises(ConnectorError):
        reader.verify(snapshot)


def test_bounds_are_disclosed_not_reported_as_exhaustive(project):
    reader = ProjectReader(register_root(project, "bounded"), limits=Limits(files=1))
    result = ProjectInspector(reader).inspect(request("files"))
    assert len(result["items"]) == 1
    assert result["snapshot"]["omitted"]["file_limit"]
    assert not result["scope_complete"]
    (project / "large.txt").write_text("x" * 1025)
    reader = ProjectReader(register_root(project, "bounded"), limits=Limits(file_bytes=1024))
    with pytest.raises(ConnectorError) as too_large:
        reader.read("large.txt")
    assert too_large.value.code == "file_too_large"


def test_literal_search_reports_truncation_and_digest(project):
    result = inspector(project).inspect(request("search", query="Controller", limit=1))
    assert result["truncated"]
    assert len(result["items"][0]["content_digest"]) == 64


def test_real_jedi_definitions_references_and_two_root_identity(project, tmp_path):
    client = inspector(project)
    symbols = client.inspect(request("symbols", language="python", query="Controller"))
    assert any(row["name"] == "Controller" and row["path"] == "controller.py" for row in symbols["items"])
    definitions = client.inspect(request("definition", path="main.py", line=2, column=9))
    assert any(row["path"] == "controller.py" and row["line"] == 1 for row in definitions["items"])
    references = client.inspect(request("references", path="controller.py", line=1, column=7))
    assert any(row["path"] == "main.py" and row["line"] == 2 for row in references["items"])
    assert references["analysis_complete"] is False
    other = tmp_path / "other"
    other.mkdir()
    (other / "controller.py").write_text("class Controller:\n    def move(self): return 99\n")
    second = inspector(other).inspect(request("symbols", language="python", query="Controller"))
    assert symbols["root_id"] != second["root_id"]
    assert symbols["items"][0]["content_digest"] != second["items"][0]["content_digest"]


def test_jedi_does_not_execute_project_source(project, tmp_path):
    marker = tmp_path / "must_not_exist"
    (project / "bomb.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\nclass StaticOnly: pass\n")
    result = inspector(project).inspect(request("symbols", language="python", query="StaticOnly"))
    assert result["items"]
    assert not marker.exists()


@pytest.fixture
def typescript_runtime():
    node = os.environ.get("JARVIS_CONNECTOR_TEST_NODE")
    typescript = os.environ.get("JARVIS_CONNECTOR_TEST_TYPESCRIPT")
    if not node or not typescript:
        pytest.skip("Explicit isolated Node/TypeScript runtime required")
    return {"node": Path(node), "typescript": Path(typescript)}


def test_real_typescript_language_service_and_unicode_columns(project, typescript_runtime):
    (project / "controller.ts").write_text("export class Controller { move() { return 7; } }\n")
    line = "const face = '\U0001f642'; new Controller().move();"
    (project / "use.ts").write_text("import {Controller} from './controller';\n" + line + "\n")
    # A project-supplied plugin must not be loaded by the in-memory language service.
    (project / "tsconfig.json").write_text('{"compilerOptions":{"plugins":[{"name":"untrusted-plugin"}]}}')
    client = inspector(project, **typescript_runtime)
    symbols = client.inspect(request("symbols", language="typescript", query="Controller"))
    assert any(row["path"] == "controller.ts" for row in symbols["items"])
    definitions = client.inspect(request("definition", path="use.ts", line=2,
                                          column=line.index("Controller") + 1))
    assert any(row["path"] == "controller.ts" and row["line"] == 1 for row in definitions["items"])
    references = client.inspect(request("references", path="controller.ts", line=1, column=14))
    assert any(row["path"] == "use.ts" and row["line"] == 2
               and row["column"] == line.index("Controller") + 1 for row in references["items"])


def test_unavailable_language_is_not_an_empty_definition(project):
    (project / "source.swift").write_text("class Controller {}\n")
    assert inspector(project).inspect(request("read", path="source.swift"))["content"]
    with pytest.raises(ConnectorError) as unknown:
        inspector(project).inspect(request("definition", path="source.swift", line=1, column=7))
    assert unknown.value.code == "language_required"
    (project / "source.ts").write_text("class Controller {}\n")
    with pytest.raises(ConnectorError) as missing:
        inspector(project).inspect(request("definition", path="source.ts", line=1, column=7))
    assert missing.value.code == "language_runtime_unavailable"


def test_git_dirty_state_and_branch_are_bound_to_snapshot(project):
    def git(*args):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    git("init", "-b", "fixture")
    git("config", "user.name", "Synthetic fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("add", "controller.py", "main.py")
    git("commit", "-m", "Synthetic fixture")
    client = inspector(project)
    clean = client.inspect(request("status"))
    assert clean["snapshot"]["git"]["branch"] == "fixture"
    assert clean["snapshot"]["git"]["dirty"] is False
    (project / "controller.py").write_text("class Controller:\n    def move(self): return 11\n")
    dirty = client.inspect(request("changes"))
    assert dirty["snapshot"]["git"]["dirty"]
    assert any(row["path"] == "controller.py" for row in dirty["items"])
    assert dirty["snapshot"]["id"] != clean["snapshot"]["id"]
    assert dirty["snapshot"]["git"]["commit"] == clean["snapshot"]["git"]["commit"]


@pytest.mark.parametrize("body", [{"operation": "shell", "command": "rm -rf /"},
                                  {"operation": "read", "path": "x", "command": "cat /etc/passwd"},
                                  {"operation": "files", "limit": True}])
def test_no_shell_or_extra_inputs(body):
    with pytest.raises(ValidationError):
        ReadRequest.model_validate(body)


def test_private_registry_and_revocation_during_local_inspection(project, tmp_path, monkeypatch, capsys):
    registry = tmp_path / "registry.json"
    root = register_root(project, "local")
    save_registry(registry, [root])
    assert load_registry(registry) == [root]
    assert registry.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        save_registry(registry, [root])
    original = ProjectInspector.inspect
    def revoked(self, query):
        result = original(self, query)
        registry.write_text("[]")
        return result
    monkeypatch.setattr(ProjectInspector, "inspect", revoked)
    monkeypatch.setattr(sys, "argv", ["connector", "--registry", str(registry), "inspect",
                                    "--root-id", str(root.root_id), "--request",
                                    '{"operation":"read","path":"controller.py"}'])
    assert main() == 1
    assert json.loads(capsys.readouterr().out) == {"status": "unavailable", "error": "root_not_enrolled"}


def test_no_source_writes_from_any_read_operation(project):
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in project.iterdir()}
    client = inspector(project)
    for query in (request("files"), request("read", path="controller.py"),
                  request("search", query="Controller"), request("symbols", language="python", query="Controller"),
                  request("definition", path="main.py", line=2, column=9),
                  request("references", path="controller.py", line=1, column=7)):
        client.inspect(query)
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in project.iterdir()}
    assert before == after

def test_missing_file_does_not_claim_root_is_offline(project):
    with pytest.raises(ConnectorError) as missing:
        inspector(project).inspect(request("read", path="missing.py"))
    assert missing.value.code == "source_unavailable"


@pytest.mark.parametrize("body", [
    {"operation": "files", "path": "controller.py"},
    {"operation": "read", "path": "controller.py", "expected_snapshot": "a" * 64},
    {"operation": "files", "expected_digest": "a" * 64},
    {"operation": "search", "query": "x", "line": 1},
    {"operation": "symbols", "query": "Controller"},
])
def test_inapplicable_constraints_are_rejected_not_silently_ignored(body):
    with pytest.raises(ValidationError):
        ReadRequest.model_validate(body)


def test_missing_scoped_search_is_unavailable_not_empty(project):
    with pytest.raises(ConnectorError) as missing:
        inspector(project).inspect(request("search", path="missing.py", query="Controller"))
    assert missing.value.code == "source_not_in_snapshot"


def test_changed_source_during_language_analysis_is_not_released(project, monkeypatch):
    client = inspector(project)
    analyze = client._analyze
    def changed(*args):
        result = analyze(*args)
        (project / "controller.py").write_text("class Other: pass\n")
        return result
    monkeypatch.setattr(client, "_analyze", changed)
    with pytest.raises(ConnectorError) as invalid:
        client.inspect(request("symbols", language="python", query="Controller"))
    assert invalid.value.code == "source_changed"


def test_language_timeout_cleans_private_workspace_and_does_not_inherit_secrets(
        project, tmp_path, monkeypatch, typescript_runtime):
    module = tmp_path / "slow.cjs"
    module.write_text(
        "require('node:fs').writeFileSync(process.env.HOME+'/private-source','synthetic');"
        "while(true) {}")
    (project / "source.ts").write_text("class Controller {}\n")
    captured = {}
    native_popen = subprocess.Popen
    def observe(*args, **kwargs):
        process = native_popen(*args, **kwargs)
        if "start_new_session" in kwargs:
            captured.update(process=process, environment=kwargs["env"])
            communicate = process.communicate
            def bounded(*inputs, **options):
                if "timeout" in options:
                    options["timeout"] = 0.2
                return communicate(*inputs, **options)
            process.communicate = bounded
        return process
    monkeypatch.setenv("JARVIS_SYNTHETIC_SECRET", "must-not-be-inherited")
    monkeypatch.setattr(subprocess, "Popen", observe)
    client = inspector(project, node=typescript_runtime["node"], typescript=module)
    with pytest.raises(ConnectorError) as expired:
        client.inspect(request("symbols", language="typescript", query="Controller"))
    assert expired.value.code == "analysis_unavailable"
    assert captured["process"].poll() is not None
    assert "JARVIS_SYNTHETIC_SECRET" not in captured["environment"]
    assert not Path(captured["environment"]["HOME"]).exists()


def test_installed_cli_enrolls_and_reads_synthetic_root(project, tmp_path):
    registry = tmp_path / "enrolled.json"
    base = [sys.executable, "-m", "jarvis_project_connector", "--registry", str(registry)]
    enrolled = subprocess.run(base + ["enroll", "--root", str(project)], check=True,
                              text=True, capture_output=True)
    root_id = json.loads(enrolled.stdout)[0]["root_id"]
    inspected = subprocess.run(base + ["inspect", "--root-id", root_id, "--request",
                                       '{"operation":"read","path":"controller.py"}'],
                               check=True, text=True, capture_output=True)
    result = json.loads(inspected.stdout)
    assert result["root_id"] == root_id
    assert result["content"] == (project / "controller.py").read_text()
    assert result["complete"]
