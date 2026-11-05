"""Real pinned Hermes, OpenAI model and E2B with manifested synthetic SQL only."""
import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration
from app.sandbox.cleanup import SandboxCleaner, SandboxCleanupRepository
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxPolicy
from app.settings import JarvisSettings
from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.models import TaskStatus, json_object
from app.tasks.runner import WorkerContext
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


REQUEST = """In the authorized offline sandbox, create input.csv containing exactly:
project,hours
alpha,2
beta,5
alpha,3
Parse the CSV using Python and write summary.json containing an object with
by_project and total_hours. Then read summary.json back using a SECOND sandbox
command and report the totals. Execute the computation; do not substitute mental
arithmetic. A completed sandbox workspace file is not a published/downloadable
artifact, and no email or external action is requested. Use a new command key
only for genuinely different work, not to retry an unknown command. If a command
is still running, inspect its status rather than executing it again."""


class ManifestedProvider:
    """Save scoped cleanup identities before any native effect."""
    def __init__(self, probe):
        self.probe = probe

    def __getattr__(self, name):
        return getattr(self.probe.provider, name)

    async def create(self, scope, policy):
        self.probe.scopes.append((scope, policy))
        item = {"scope": scope.model_dump(mode="json"), "policy": policy.model_dump(mode="json"), "phase": "planned"}
        self.probe.record["resources"].append(item)
        self.probe.persist()
        handle = await self.probe.provider.create(scope, policy)
        item.update(sandbox_id=handle.sandbox_id, template_id=handle.template_id, phase="created")
        self.probe.persist()
        return handle


class ObservedBridge:
    """Synthetic fixture diagnostics only; production does not copy tool inputs."""
    def __init__(self, bridge, probe):
        self.bridge, self.probe = bridge, probe

    def __getattr__(self, name):
        return getattr(self.bridge, name)

    async def invoke(self, name, arguments, *, provider_call_id):
        result = await self.bridge.invoke(name, arguments, provider_call_id=provider_call_id)
        if name.startswith("sandbox_"):
            self.probe.record.setdefault("synthetic_tool_calls", []).append(
                {"name": name, "arguments": arguments, "result": result})
            self.probe.persist()
        return result


async def exercise(probe, args, settings):
    frontier = FrontierConfiguration(model=args.model, reasoning_effort="medium", api_key=settings.OPENAI_API_KEY)
    async with sandbox_task_fixture(settings.SUPABASE_DB_URL, args.manifest_dir,
        test_name="native-hermes-sandbox", policy=SandboxPolicy(lifetime_seconds=300, command_seconds=30),
        model=frontier.model, requested_outcome=REQUEST) as fixture:
        probe.record["database_manifest"] = str(fixture.source.fixture.path)
        probe.persist()
        context = WorkerContext(fixture.claim, fixture.tasks, TaskBudgetRepository(fixture.tasks),
                                TaskActionRepository(fixture.tasks))
        root = probe.path.parent/"hermes"
        factory = WorkerContextFactory(fixture.source.pool, sandbox_provider=ManifestedProvider(probe))
        async def prepare(context):
            prepared = await factory(context)
            return replace(prepared, tools=ObservedBridge(prepared.tools, probe))
        provider = HermesTaskProvider(frontier,
            HermesRuntimeConfiguration(args.hermes_python, root, host_id="isolated-hermes-e2b-probe", max_iterations=16),
            prepare=prepare)
        try:
            async with asyncio.timeout(240):
                outcome = await provider.execute(context)
            accepted = await fixture.tasks.finish(fixture.claim, outcome)
            probe.record["outcome"] = {"status": outcome.status.value, "reason": outcome.reason, "payload": outcome.payload}
            probe.persist()
            probe.check("native_expert_task_succeeded", accepted and outcome.status==TaskStatus.SUCCEEDED)
            async with fixture.source.pool.acquire() as connection:
                commands = await connection.fetch("""select c.*,p.content as parameters,r.content as result
                    from jarvis_sandbox_commands c
                    join jarvis_context_payloads p on p.id=c.parameters_payload_id
                    left join jarvis_context_payloads r on r.id=c.result_payload_id
                    where c.owner_id=$1 order by c.created_at""", fixture.source.owner)
                allocations = await connection.fetch("select * from jarvis_sandbox_allocations where owner_id=$1",
                                                     fixture.source.owner)
                unsettled = await connection.fetchval("""select count(*) from jarvis_task_reservations
                    where owner_id=$1 and status in ('reserved','submitted','unknown')""", fixture.source.owner)
            results = [json_object(row["result"]) for row in commands if row["result"] is not None]
            probe.check("one_native_allocation", len(allocations)==1)
            probe.check("separate_compute_and_read_commands", len(commands)>=2 and len(results)==len(commands))
            probe.check("all_command_receipts_successful", all(result["exit_code"]==0 and result["started"]
                and not result["timed_out"] and not result["output_limited"] and result["runner_error"] is None
                for result in results))
            # Independent native verification, not a restriction on how the
            # expert chooses to format stdout. This read-only fixture command
            # is deliberately separate from the model's two ledger operations.
            audit = await probe.provider.start(fixture.allocations.handle(allocations[0]),
                "cat summary.json", operation_id=uuid4())
            audited = await audit.wait(timeout_seconds=10)
            parsed = json.loads(audited.stdout)
            probe.record["independent_file_read"] = {"exit_code":audited.exit_code, "json":parsed}
            probe.persist()
            probe.check("native_file_round_trip_correct",
                audited.exit_code==0 and parsed=={"by_project": {"alpha":5,"beta":5},"total_hours":10})
            probe.check("command_and_model_usage_settled", unsettled==0)
            probe.check("expert_not_audio_model_authored_answer", outcome.payload.get("authorship",{}).get("kind")=="expert")
            probe.check("native_private_homes_removed", not any(root.glob("*/home")))
            probe.record["command_receipts"] = [{"id":str(row["id"]), "phase":row["phase"],
                "result":json_object(row["result"]) if row["result"] else None} for row in commands]
            probe.record["usage"] = await context.budgets.usage(fixture.source.owner, fixture.task.id)
            probe.persist()
        finally:
            await cleanup_fixture(fixture, probe)


async def cleanup_fixture(fixture, probe):
    try:
        async with fixture.source.pool.acquire() as connection:
            ids = await connection.fetch("select id from jarvis_sandbox_allocations where owner_id=$1",
                                        fixture.source.owner)
        for item in ids:
            await fixture.allocations.request_cleanup(fixture.source.owner, item["id"], reason="synthetic_probe_complete")
        cleaner = SandboxCleaner(SandboxCleanupRepository(fixture.allocations), probe.provider)
        for _ in ids:
            result = await cleaner.run_once(fixture.source.owner, "native-hermes-probe-cleaner")
            if result is not None:
                probe.check("task_cleanup_confirmed", result["state"]=="cleaned")
    finally:
        # A SQL/normal-cleaner failure must not skip independent native cleanup.
        if probe.scopes:
            await probe.cleanup()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    args = parser.parse_args()
    probe = None
    try:
        provider = E2BProvider(E2BConfiguration(api_key=dotenv_values(args.credential_file).get("E2B_API_KEY")))
        probe = Probe(provider, args.directory)
        print(json.dumps({"manifest":str(probe.path)}), flush=True)
        asyncio.run(exercise(probe, args, JarvisSettings()))
    except Exception as error:
        if probe is not None:
            probe.record["failure"] = {"type":type(error).__name__, "code":getattr(error,"code",None)}
            probe.persist()
        print(json.dumps({"passed":False,"error":getattr(error,"code",type(error).__name__)}))
        return 1
    print(json.dumps({"passed":True,"checks":len(probe.record["checks"]),
                      "cleanup":probe.record.get("cleanup"),"manifest":str(probe.path)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
