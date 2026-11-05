"""Maintained language tooling operates on snapshots, never executing project code."""
from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .filesystem import ProjectReader
from .protocol import ConnectorError, ReadRequest


LANGUAGES = {".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "typescript",
             ".js": "typescript", ".jsx": "typescript", ".mjs": "typescript", ".cjs": "typescript"}


class ProjectInspector:
    def __init__(self, reader: ProjectReader, *, node: Path | None = None, typescript: Path | None = None):
        self.reader, self.node, self.typescript = reader, node, typescript
        if any(path is not None and not path.is_absolute() for path in (node, typescript)):
            raise ValueError("Language runtimes must be operator-selected absolute paths")

    def capabilities(self):
        return {"text": ["files", "read", "search", "status", "changes"],
                "navigation": {"python": "jedi-0.19.2",
                               "typescript": "typescript-language-service" if self.node and self.typescript else None},
                "positions": "one-based line and Unicode code-point column",
                "writes": False, "shell": False,
                "limitations": ["Static analysis can miss dynamic references.",
                                "Only policy-permitted files inside a bounded snapshot are analyzed.",
                                "Editor buffers not saved to disk are unavailable."]}

    def inspect(self, request: ReadRequest, *, include_originals=False):
        base = {"status": "available", "root_id": str(self.reader.registration.root_id),
                "label": self.reader.registration.label, "operation": request.operation,
                "capabilities": self.capabilities()}
        if request.operation == "read":
            original = self.reader.read(request.path, request.expected_digest)
            if request.offset > len(original.text):
                raise ConnectorError("invalid_offset")
            stop = min(len(original.text), request.offset + request.max_chars)
            base.update({"source": original.metadata(), "content": original.text[request.offset:stop],
                         "captured_at": datetime.now(UTC).isoformat(), "offset": request.offset,
                         "complete": request.offset == 0 and stop == len(original.text),
                         "next_offset": stop if stop < len(original.text) else None,
                         "git": self.reader.git_state()})
            # The content remains bound to its digest even when it changes immediately after return.
            self.reader.read(request.path, original.digest)
            if include_originals:
                base["_originals"] = [{"path": original.path, "text": original.text,
                                       "content_digest": original.digest}]
            return base
        snapshot = self.reader.snapshot()
        if request.expected_snapshot and snapshot.snapshot_id != request.expected_snapshot:
            raise ConnectorError("source_changed")
        base["snapshot"] = {"id": snapshot.snapshot_id, "captured_at": snapshot.captured_at,
                            "files": len(snapshot.files), "omitted": snapshot.omitted,
                            "git": snapshot.git}
        results, truncated = [], False
        if request.operation in {"status", "changes"}:
            if request.operation == "changes" and not snapshot.git["available"]:
                raise ConnectorError("git_status_unavailable")
            results = snapshot.git.get("changes", []) if request.operation == "changes" else []
        elif request.operation == "files":
            results = [source.metadata() for path, source in snapshot.files.items()
                       if request.query is None or request.query in path]
        elif request.operation == "search":
            if request.path is not None:
                if request.path not in snapshot.files:
                    raise ConnectorError("source_not_in_snapshot")
                if request.expected_digest and snapshot.files[request.path].digest != request.expected_digest:
                    raise ConnectorError("source_changed")
            for path, source in snapshot.files.items():
                if len(results) > request.limit:
                    break
                if request.path is not None and path != request.path:
                    continue
                for number, line in enumerate(source.text.splitlines(), 1):
                    if request.query in line:
                        results.append({"path": path, "line": number, "text": line[:2000],
                                        "line_truncated": len(line) > 2000, "content_digest": source.digest})
                        if len(results) > request.limit:
                            break
        else:
            language = request.language or LANGUAGES.get(Path(request.path or "").suffix)
            if language is None:
                raise ConnectorError("language_required")
            if request.path is not None:
                if request.path not in snapshot.files:
                    raise ConnectorError("source_not_in_snapshot")
                if LANGUAGES.get(Path(request.path).suffix) != language:
                    raise ConnectorError("unsupported_language")
                if request.expected_digest and snapshot.files[request.path].digest != request.expected_digest:
                    raise ConnectorError("source_changed")
                lines = snapshot.files[request.path].text.splitlines()
                if request.line is not None and (request.line > len(lines) or request.column > len(lines[request.line-1]) + 1):
                    raise ConnectorError("invalid_position")
            originals = {path: source.text for path, source in snapshot.files.items()
                         if LANGUAGES.get(Path(path).suffix) == language}
            results, truncated = self._analyze(language, request, originals)
            for row in results:
                source = snapshot.files.get(row["path"])
                if source is None:
                    raise ConnectorError("invalid_analysis_source")
                row["content_digest"] = source.digest
            base["analysis_complete"] = False
            base["analysis_limitations"] = [
                "Static results from bounded source files, not a proof of all possible runtime references.",
                "External dependency sources and repository-supplied plugins are not loaded."]
        self.reader.verify(snapshot)
        base.update({"items": results[:request.limit], "truncated": truncated or len(results) > request.limit,
                     "scope_complete": not snapshot.omitted and not truncated and len(results) <= request.limit})
        if include_originals and request.operation in {"search", "symbols", "definition", "references"}:
            paths = dict.fromkeys(item["path"] for item in base["items"])
            base["_originals"] = [{"path": path, "text": snapshot.files[path].text,
                                   "content_digest": snapshot.files[path].digest} for path in paths]
        return base

    def _analyze(self, language, request, files):
        directory = Path(__file__).parent
        if language == "python":
            command = [sys.executable, "-I", str(directory / "python_worker.py")]
        elif language == "typescript" and self.node and self.typescript:
            command = [str(self.node), "--max-old-space-size=192",
                       str(directory / "typescript_worker.cjs"), str(self.typescript)]
        else:
            raise ConnectorError("language_runtime_unavailable")
        try:
            with tempfile.TemporaryDirectory(prefix="jarvis-code-analysis-") as temporary:
                environment = {"PATH": "/usr/bin:/bin", "HOME": temporary, "TMPDIR": temporary,
                               "LANG": "en_US.UTF-8", "PYTHONIOENCODING": "utf-8"}
                if language == "python":
                    command.append(temporary)
                process = subprocess.Popen(command, text=True, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           cwd=directory, env=environment, start_new_session=True)
                try:
                    output, _ = process.communicate(json.dumps({"request": request.model_dump(), "files": files}),
                                                    timeout=20)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()
                    raise
                if process.returncode or len(output) > 1048576:
                    raise ConnectorError("analysis_unavailable")
                payload = json.loads(output)
                return payload["items"], payload["truncated"]
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
            raise ConnectorError("analysis_unavailable") from error
