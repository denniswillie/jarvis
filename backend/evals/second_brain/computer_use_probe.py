"""Harmless native Hermes browser/desktop cases, with independent owned fixtures."""
import argparse
import asyncio
from contextlib import contextmanager, aclosing
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import traceback
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from dotenv import dotenv_values

from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, SandboxConfiguration, TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxPolicy
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider, _NativeRun
from app.tasks.models import TaskCreate, TaskStatus, json_object
from app.tasks.model_gateway import OpenAIResponsesHTTP, TaskModelGateway
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.hermes_sandbox_probe import ManifestedProvider, ObservedBridge, cleanup_fixture
from evals.second_brain.source_fixture import source_fixture


NOTE = "Synthetic desktop note.\nThe calibration is still unfinished.\n"
FORM = {"name": ["Rowan Delta"], "color": ["Green"]}
REQUESTS = {
    "page": "Use the sandbox browser to open http://127.0.0.1:8765/, follow the Details link, "
            "and tell me the access code visible on that page.",
    "form": "Use the sandbox browser to open http://127.0.0.1:8765/form. Fill Name with Rowan Delta, "
            "choose Green for Color, and submit the form. Confirm the visible result.",
    "editor": "Using a desktop text editor, create workspace file note.txt with exactly these two lines, "
              "including a newline after the last line:\n" + NOTE +
              "Save the file, close it, reopen it in the editor, and confirm its contents.",
}

# Root-owned synthetic fixture state is not writable by the model's account.
# HTTP requests are local to the isolated VM; no real account or internet needed.
SERVER = r'''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs
ROOT = Path('/var/lib/jarvis/computer-fixture')
secret = (ROOT/'code').read_text()
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def page(self, body):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(('<html><head><title>Computer fixture</title></head>'
            '<body style="font:24px sans-serif;padding:40px">'+body+'</body></html>').encode())
    def do_GET(self):
        with (ROOT/'requests.jsonl').open('a') as stream:
            stream.write(json.dumps({'method':'GET','path':self.path})+'\n')
        if self.path == '/details':
            self.page('<h1>Details</h1><p>Access code: <strong>'+secret+'</strong></p>')
        elif self.path == '/form':
            self.page('<h1>Preferences</h1><form method="post" action="/submitted">'
                '<p><label>Name <input name="name" autocomplete="off"></label></p>'
                '<p><label>Color <select name="color"><option>Blue</option><option>Green</option>'
                '<option>Red</option></select></label></p><button>Submit</button></form>')
        else:
            self.page('<h1>Fixture home</h1><a href="/details">Details</a>')
    def do_POST(self):
        count = int(self.headers.get('Content-Length', '0'))
        if self.path != '/submitted' or count > 4096:
            self.send_error(400)
            return
        data = parse_qs(self.rfile.read(count).decode())
        with (ROOT/'submissions.jsonl').open('a') as stream:
            stream.write(json.dumps(data)+'\n')
        self.page('<h1>Saved</h1><p>Your preferences were recorded.</p>')
HTTPServer(('127.0.0.1',8765), Handler).serve_forever()
'''

# A not-yet-listening fixture is a readiness observation, not a failed native
# command. The SDK raises on nonzero command exit, so keep this check at exit 0.
SERVER_READY_CHECK = '''
import socket
try:
    connection = socket.create_connection(('127.0.0.1', 8765), 1)
except OSError:
    print('waiting')
else:
    connection.close()
    print('ready')
'''


class DesktopObservedBridge(ObservedBridge):
    async def resolve_image(self, name, result):
        resolved = await self.bridge.resolve_image(name, result)
        if name == "sandbox_view_image":
            suffix = ".png" if resolved.reference.content_type == "image/png" else ".jpg"
            target = self.probe.root / ("observation-" + result["selection_id"] + suffix)
            target.write_bytes(resolved.image_bytes)
            self.probe.record.setdefault("observations", []).append(
                {"path": str(target), "metadata": resolved.reference.to_metadata()})
            self.probe.persist()
        return resolved


class ObservedUpstream(OpenAIResponsesHTTP):
    """Keep bounded transport/usage diagnostics, never keys or request bodies."""
    def __init__(self, key, probe):
        super().__init__(key)
        self.probe = probe
        self.client.event_hooks["response"] = [self.response_observed]

    async def response_observed(self, response):
        entry = {"route": response.request.url.path, "http_status": response.status_code}
        if response.status_code >= 400:
            try:
                await response.aread()
                error = response.json().get("error", {})
                entry["error"] = {key: error.get(key) for key in ("type", "code", "param")}
            except (ValueError, AttributeError):
                entry["error"] = {"type": "unstructured_error"}
        self.probe.record.setdefault("upstream_transport", []).append(entry)
        self.probe.persist()

    async def count(self, payload):
        try:
            value = await super().count(payload)
            self.probe.record.setdefault("input_counts", []).append(value)
            self.probe.persist()
            return value
        except Exception as error:
            self.probe.record["upstream_failure"] = {"phase": "count", "code": getattr(error, "code", type(error).__name__)}
            self.probe.persist()
            raise

    async def events(self, payload):
        try:
            async for event in super().events(payload):
                if event.data.get("type") in {"response.completed", "response.failed", "response.incomplete", "error"}:
                    response = event.data.get("response", {})
                    self.probe.record.setdefault("upstream_outcomes", []).append({"type": event.data.get("type"),
                        "status": response.get("status"), "usage": response.get("usage"),
                        "incomplete_details": response.get("incomplete_details")})
                    self.probe.persist()
                yield event
        except Exception as error:
            self.probe.record["upstream_failure"] = {"phase": "events", "code": getattr(error, "code", type(error).__name__)}
            self.probe.persist()
            raise


async def trusted_fixture(provider, handle, code):
    box = await provider._connect(handle)
    root = "/var/lib/jarvis/computer-fixture"
    setup = "from pathlib import Path; p=Path(" + repr(root) + "); p.mkdir(mode=0o700); "
    setup += "(p/'code').write_text(" + repr(code) + "); (p/'server.py').write_text(" + repr(SERVER) + ")"
    await provider._request(box.commands.run, "python3 -c " + shlex.quote(setup), user="root", timeout=10)
    await provider._request(box.commands.run,
        "nohup python3 " + root + "/server.py </dev/null >" + root + "/server.log 2>&1 &",
        user="root", timeout=10)
    # This checks fixture readiness, not the model's navigation result.
    for _ in range(20):
        result = await provider._request(box.commands.run,
            "python3 -c " + shlex.quote(SERVER_READY_CHECK),
            user="root", timeout=3)
        if result.stdout.strip() == "ready":
            return
        await asyncio.sleep(0.1)
    raise RuntimeError("computer_fixture_unavailable")


@contextmanager
def native_diagnostics(probe):
    """Observe this synthetic run without changing policy or runtime decisions."""
    original_prepare = TaskModelGateway.prepare
    original_stream = TaskModelGateway.stream
    original_result = _NativeRun.result

    def prepare(gateway, body, **kwargs):
        try:
            return original_prepare(gateway, body, **kwargs)
        except Exception as error:
            shapes = []
            for item in body.get("input", []) if isinstance(body.get("input"), list) else []:
                parts = item.get("content", item.get("output", []))
                shapes.append({"type": item.get("type"), "role": item.get("role"), "keys": sorted(item),
                    "parts": [{"type": part.get("type"), "keys": sorted(part),
                               "image_registered": part.get("image_url") in gateway.images}
                              for part in parts if isinstance(part, dict)] if isinstance(parts, list) else []})
            probe.record["gateway_prepare_failure"] = {"code": getattr(error, "code", type(error).__name__),
                "keys": sorted(body), "max_output_tokens": body.get("max_output_tokens"), "input_shapes": shapes,
                "guard_trace": [{"line": frame.lineno, "function": frame.name} for frame in
                    traceback.extract_tb((error.__cause__ or error).__traceback__)],
                "tools": [{"name": tool.get("name"), "type": tool.get("type"), "keys": sorted(tool),
                           "known": tool.get("name") in gateway.tools,
                           "parameters_type": type(tool.get("parameters")).__name__}
                          for tool in body.get("tools", [])]}
            probe.persist()
            raise

    async def stream(gateway, body, **kwargs):
        try:
            async with aclosing(original_stream(gateway, body, **kwargs)) as events:
                async for event in events:
                    yield event
        except Exception as error:
            probe.record["gateway_stream_failure"] = {"code": getattr(error, "code", type(error).__name__)}
            probe.persist()
            raise

    async def result(run):
        value = await original_result(run)
        # Only this synthetic fixture's final result; never retain credentials.
        response = str(value.get("final_response", ""))[:4000]
        for token in (run.gateway.token, run.gateway.auxiliary_token):
            response = response.replace(token, "[redacted]")
        probe.record["native_final_diagnostic"] = response
        probe.record["native_result_state"] = {key: value.get(key) for key in
            ("completed", "failed", "interrupted", "partial", "api_calls", "turn_exit_reason", "context_compressions")}
        probe.record["native_result_at"] = datetime.now(UTC).isoformat()
        probe.persist()
        return value

    with patch.object(TaskModelGateway, "prepare", prepare), patch.object(TaskModelGateway, "stream", stream), \
            patch.object(_NativeRun, "result", result):
        yield


async def audit(provider, handle, case):
    box = await provider._connect(handle)
    if case == "editor":
        content = await provider.read_workspace_file(handle, "note.txt", max_bytes=4096)
        return {"text": content.decode(), "matches": content == NOTE.encode()}
    filename = "submissions.jsonl" if case == "form" else "requests.jsonl"
    result = await provider._request(box.commands.run,
        "cat /var/lib/jarvis/computer-fixture/" + filename, user="root", timeout=5)
    return {"rows": [json.loads(line) for line in result.stdout.splitlines()]}


async def exercise(probe, args, settings):
    frontier = FrontierConfiguration(model=args.model, reasoning_effort="medium", api_key=settings.OPENAI_API_KEY)
    policy = SandboxPolicy(template="desktop", lifetime_seconds=600, command_seconds=30)
    code = "VIEW-" + uuid4().hex[:8].upper()
    request = REQUESTS[args.case] + (
        " This is a synthetic basic computer-use check. Use the actual browser/desktop UI and inspect "
        "screenshots with sandbox_view_image. Do not replace UI interaction with direct HTTP requests, "
        "reading browser/fixture internals, or writing the requested output file with shell/file APIs. "
        "Terminal commands to launch applications, operate their windows and capture screenshots are available. "
        "Do not publish artifacts or make unrelated external changes. Report failures honestly.")
    async with source_fixture(settings.SUPABASE_DB_URL, args.manifest_dir,
                              test_name="native-computer-use-" + args.case) as source:
        probe.record.update(database_manifest=str(source.fixture.path), case=args.case, model=args.model,
                            fixture_protocol="computer-use.v2",
                            expected_code=code if args.case == "page" else None)
        probe.persist()
        tasks = TaskRepository(source.pool, TaskLimits(lease_seconds=600, active_seconds=600,
            tool_steps=80, request_output_tokens=4096, task_input_tokens=500000))
        service = TaskService(tasks, {"frontier": ConfiguredTaskProvider(frontier.model)},
                              sandbox_configuration=SandboxConfiguration(policy=policy))
        task = await service.create(source.owner, TaskCreate(request_id=uuid4(), title="Synthetic computer use",
            requested_outcome=request, execution_environment="desktop_sandbox"))
        claim = await tasks.claim_next("native-computer-use", ("frontier",),
                                      worker_capabilities=policy.required_capabilities)
        context = TaskRunner(tasks, {}).context(claim)
        fixture = SimpleNamespace(source=source, tasks=tasks, task=task, claim=claim,
                                  allocations=SandboxAllocationRepository(tasks))
        factory = WorkerContextFactory(source.pool, sandbox_provider=ManifestedProvider(probe), sandbox_policy=policy)

        async def prepare(ctx):
            probe.record["setup_stage"] = "worker_context"
            probe.persist()
            prepared = await factory(ctx)
            probe.record["setup_stage"] = "desktop_allocation"
            probe.persist()
            allocation = await prepared.tools.broker._ready(await prepared.tools.broker.allocation())
            if args.case != "editor":
                probe.record["setup_stage"] = "local_page_fixture"
                probe.persist()
                await trusted_fixture(probe.provider, fixture.allocations.handle(allocation), code)
            probe.record["setup_stage"] = "native_agent"
            probe.persist()
            return replace(prepared, tools=DesktopObservedBridge(prepared.tools, probe))

        provider = HermesTaskProvider(frontier, HermesRuntimeConfiguration(args.hermes_python,
            probe.root / "hermes", host_id="isolated-computer-use", max_iterations=32), prepare=prepare,
            upstream_factory=lambda: ObservedUpstream(frontier.api_key.get_secret_value(), probe))
        try:
            with native_diagnostics(probe):
                async with asyncio.timeout(540):
                    outcome = await provider.execute(context)
            probe.record["outcome"] = {"status": outcome.status.value, "payload": outcome.payload,
                                       "reason": outcome.reason}
            probe.persist()
            probe.check("native_task_succeeded", outcome.status == TaskStatus.SUCCEEDED)
            async with source.pool.acquire() as connection:
                allocation = await connection.fetchrow("select * from jarvis_sandbox_allocations where owner_id=$1", source.owner)
                rows = await connection.fetch("""select p.content as parameters,r.content as result
                    from jarvis_sandbox_commands c join jarvis_context_payloads p on p.id=c.parameters_payload_id
                    left join jarvis_context_payloads r on r.id=c.result_payload_id
                    where c.owner_id=$1 order by c.created_at""", source.owner)
            probe.record["command_receipts"] = [{"parameters": json_object(row["parameters"]),
                "result": json_object(row["result"]) if row["result"] is not None else None} for row in rows]
            observed = await audit(probe.provider, fixture.allocations.handle(allocation), args.case)
            probe.record["independent_outcome"] = observed
            probe.persist()
            if args.case == "page":
                probe.check("page_value_correct", code in outcome.payload["answer"]
                    and any(item["path"] == "/details" for item in observed["rows"]))
            elif args.case == "form":
                probe.check("one_exact_form_submission", observed["rows"] == [FORM])
            else:
                probe.check("exact_editor_file", observed["matches"])
            calls = probe.record.get("synthetic_tool_calls", [])
            probe.check("actual_pixels_observed", len(probe.record.get("observations", [])) >= 2)
            probe.check("native_input_commands_recorded", any(item["name"] == "sandbox_execute"
                and "xdotool" in item["arguments"].get("command", "") for item in calls))
            # These checks are necessary, not sufficient: inspect the actual
            # saved screenshots and traces for real navigation/editing/reopening.
            probe.record["manual_ui_trace_review"] = "required"
            probe.check("task_result_accepted", await tasks.finish(claim, outcome))
            probe.record["usage"] = await context.budgets.usage(source.owner, task.id)
            probe.persist()
        finally:
            await cleanup_fixture(fixture, probe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--case", choices=tuple(REQUESTS), required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    args = parser.parse_args()
    native = E2BProvider(E2BConfiguration(api_key=dotenv_values(args.credential_file).get("E2B_API_KEY")))
    probe = Probe(native, args.directory)
    print(json.dumps({"manifest": str(probe.path), "case": args.case}), flush=True)
    try:
        asyncio.run(exercise(probe, args, JarvisSettings()))
    except Exception as error:
        probe.record["failure"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
            "trace": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)]}
        probe.persist()
        print(json.dumps({"passed": False, "error": getattr(error, "code", type(error).__name__)}))
        return 1
    print(json.dumps({"passed": True, "checks": probe.record["checks"], "cleanup": probe.record.get("cleanup"),
                      "manifest": str(probe.path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
