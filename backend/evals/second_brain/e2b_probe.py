"""Explicit real E2B boundary probe, with allocation-first manifests and cleanup.

Only synthetic data enters newly allocated base-template sandboxes. No model,
database, Gitbio sandbox, email, or Jarvis live runtime is involved.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import tempfile
import traceback
from uuid import uuid4

from app.sandbox.e2b_provider import E2BProvider, SDK_VERSION
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxPolicy, SandboxScope


class Probe:
    def __init__(self, provider, directory):
        self.provider = provider
        self.root = Path(tempfile.mkdtemp(prefix="jarvis-e2b-",dir=directory))
        self.path = self.root / "manifest.json"
        self.run_id = uuid4().hex
        self.scopes = []
        self.record = {"format":"jarvis.e2b.probe.v1", "run_id":self.run_id, "sdk":SDK_VERSION,
                       "status":"prepared", "resources":[], "checks":{}, "created_at":datetime.now(UTC).isoformat()}
        self.persist()

    def persist(self):
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
        with os.fdopen(fd,"w") as output:
            json.dump(self.record,output,indent=2)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)

    def check(self, name, passed, **evidence):
        self.record["checks"][name] = {"passed":bool(passed), **evidence}
        self.persist()
        if not passed:
            raise AssertionError(name)

    async def create(self, policy):
        scope = SandboxScope(owner_id="direct-image-eval-e2b-"+self.run_id,task_id=uuid4(),task_revision=1,
                             attempt_id=uuid4(),allocation_id=uuid4())
        self.scopes.append((scope,policy))
        resource = {"scope":scope.model_dump(mode="json"),"policy":policy.model_dump(mode="json"),"phase":"planned"}
        self.record["resources"].append(resource)
        self.persist()
        handle = await self.provider.create(scope,policy)
        resource.update(phase="created",sandbox_id=handle.sandbox_id,template_id=handle.template_id)
        self.persist()
        await self.provider.prepare(handle)
        resource["phase"] = "prepared"
        self.persist()
        return handle

    async def run(self, handle, command):
        pending = await self.provider.start(handle,command,operation_id=uuid4())
        return await pending.wait(timeout_seconds=handle.policy.command_seconds+2)

    async def python(self, handle, code):
        return await self.run(handle,"python3 -c "+shlex.quote(code))

    async def cleanup(self):
        failures,remaining = [],[]
        for scope,policy in self.scopes:
            resources = [item for item in self.record["resources"]
                         if item["scope"]["allocation_id"] == str(scope.allocation_id)]
            ids = {identity for item in resources for identity in
                   ([item["sandbox_id"]] if item.get("sandbox_id") else []) + item.get("cleanup_observed_ids",[])}
            try:
                ids.update(await self.provider.discover(scope,policy))
            except Exception as error:
                failures.append({"allocation_id":str(scope.allocation_id),"phase":"discover",
                                 "error":getattr(error,"code",type(error).__name__)})
            if not ids:
                # A planned create with no receipt cannot be declared rejected
                # merely because this particular native list was empty.
                if not any(item.get("sandbox_id") or item.get("cleanup_observed_ids") for item in resources):
                    failures.append({"allocation_id":str(scope.allocation_id),"phase":"unresolved_create",
                                     "error":"sandbox_creation_outcome_unresolved"})
            for item in self.record["resources"]:
                if item["scope"]["allocation_id"] == str(scope.allocation_id):
                    item["cleanup_observed_ids"] = sorted(set(item.get("cleanup_observed_ids",[])) | ids)
            self.persist()
            for identity in ids:
                try:
                    raw = await self.provider._request(self.provider.sdk.AsyncSandbox.get_info,identity,
                                                       **self.provider._options())
                    # Keep safe provider configuration evidence even if policy
                    # admission failed immediately after the create receipt.
                    self.record.setdefault("provider_policies",[]).append({
                        "allocation_id":str(scope.allocation_id), "sandbox_id":identity,
                        "internet":raw.allow_internet_access,"network":raw.network,"lifecycle":raw.lifecycle})
                    await self.provider.kill(identity,scope,policy)
                except Exception as error:
                    if getattr(error,"code",None) != "sandbox_not_found":
                        failures.append({"allocation_id":str(scope.allocation_id),"phase":"kill",
                                         "error":getattr(error,"code",type(error).__name__)})
            try:
                remaining.extend(await self.provider.discover(scope,policy))
            except Exception as error:
                failures.append({"allocation_id":str(scope.allocation_id),"phase":"verify",
                                 "error":getattr(error,"code",type(error).__name__)})
        self.record["cleanup"] = {"remaining_ids":remaining,"failures":failures}
        self.record["status"] = "cleaned" if not remaining and not failures else "cleanup_unresolved"
        self.persist()
        if remaining or failures:
            raise RuntimeError("Owned E2B resource cleanup remains unresolved: "+str(self.path))

    async def exercise(self):
        failure = None
        try:
            offline = await self.create(SandboxPolicy(lifetime_seconds=300,command_seconds=20))
            self.check("private_network_and_owned_allocation",len(await self.provider.discover(offline.scope,offline.policy))==1)
            value = await self.python(offline,
                "import os,json,pathlib; pathlib.Path('progress.txt').write_text('Synthetic calibration remains unfinished.'); "
                "print(json.dumps({'uid':os.getuid(),'secrets':[k for k in ('OPENAI_API_KEY','E2B_API_KEY','SUPABASE_DB_URL',"
                "'SUPABASE_SERVICE_ROLE_KEY') if k in os.environ]}))")
            data = json.loads(value.stdout)
            self.check("unprivileged_clean_environment",value.exit_code==0 and data["uid"]!=0 and not data["secrets"],uid=data["uid"])
            permissions = await self.run(offline,"test ! -w /home/jarvis && test ! -w /var/lib/jarvis && ! sudo -n true")
            self.check("no_root_or_profile_write",permissions.exit_code==0)
            network_inner = ("import json,urllib.request\ntry:\n response=urllib.request.urlopen('https://example.com',timeout=3); "
                             "print(json.dumps({'status':response.status}))\nexcept Exception as error:\n "
                             "print(json.dumps({'error':type(error).__name__}))")
            # urllib's socket timeout does not bound DNS resolution. Bound the
            # entire negative-control process without weakening the positive gate.
            network_code = ("import json,subprocess,sys\ntry:\n result=subprocess.run([sys.executable,'-c',"
                + repr(network_inner) + "],capture_output=True,text=True,timeout=6); print(result.stdout)\n"
                "except subprocess.TimeoutExpired:\n print(json.dumps({'error':'probe_deadline'}))")
            blocked = json.loads((await self.python(offline,network_code)).stdout)
            online = await self.create(SandboxPolicy(lifetime_seconds=180,command_seconds=20,allowed_domains=("example.com",)))
            allowed = json.loads((await self.python(online,network_code)).stdout)
            self.check("egress_denial_with_positive_control", "error" in blocked and allowed.get("status")==200,
                       offline=blocked,allowed=allowed)
            command = await self.provider.start(offline,
                "python3 -c "+shlex.quote("import time; time.sleep(3); print('same-process-result')"),operation_id=uuid4())
            try:
                await command.wait(timeout_seconds=.01)
                raise AssertionError("Expected observation timeout")
            except SandboxError as error:
                self.check("timeout_is_observation_loss",error.code=="sandbox_command_observation_lost",pid=command.pid)
            pids = [item.pid for item in await self.provider.processes(offline)]
            self.check("timed_out_process_still_running",command.pid in pids)
            same = await self.provider.reconnect_command(offline,command.pid)
            result = await same.wait(timeout_seconds=10)
            self.check("same_pid_reconnect",same.pid==command.pid and result.exit_code==0 and "same-process-result" in result.stdout)
            paused = await self.provider.pause(offline)
            self.check("native_pause",paused.state=="paused")
            try:
                await self.provider.processes(paused)
                raise AssertionError("Read silently resumed a paused sandbox")
            except SandboxError as error:
                self.check("paused_read_stays_paused",error.code=="sandbox_paused")
            resumed = await self.provider.resume(paused)
            contents = await self.python(resumed,"from pathlib import Path; print(Path('progress.txt').read_text())")
            self.check("resume_preserves_same_vm_and_file",resumed.sandbox_id==offline.sandbox_id and
                       "Synthetic calibration remains unfinished." in contents.stdout)
            pending = await self.provider.start(resumed,"sleep 30",operation_id=uuid4())
            accepted = await self.provider.cancel_command(resumed,pending.pid)
            await pending.disconnect()
            active = [item.pid for item in await self.provider.processes(resumed)]
            self.check("native_command_cancellation",accepted is True and pending.pid not in active)
            oversized = await self.provider.start(resumed,"python3 -c \"print('x'*200000)\"",operation_id=uuid4())
            try:
                await oversized.wait(timeout_seconds=20)
                raise AssertionError("Unbounded output was accepted")
            except SandboxError as error:
                self.check("native_output_bound",error.code=="sandbox_output_limit")
        except BaseException as error:
            failure = error
            self.record["failure"] = {"type":type(error).__name__,"code":getattr(error,"code",str(error) if isinstance(error,AssertionError) else None)}
            cause = error.__context__
            if cause is not None:
                self.record["failure"]["cause_type"] = type(cause).__name__
                self.record["failure"]["cause_frames"] = [{"file":Path(frame.filename).name,"line":frame.lineno,
                    "function":frame.name} for frame in traceback.extract_tb(cause.__traceback__)[-4:]]
            self.persist()
        finally:
            await self.cleanup()
        if failure is not None:
            raise failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-file",type=Path)
    parser.add_argument("--directory",type=Path,default=Path("/tmp"))
    args=parser.parse_args()
    key=os.getenv("E2B_API_KEY")
    if args.credential_file:
        from dotenv import dotenv_values
        key=dotenv_values(args.credential_file).get("E2B_API_KEY")
    provider=E2BProvider(E2BConfiguration(api_key=key))
    probe=Probe(provider,args.directory)
    print(json.dumps({"manifest":str(probe.path)}),flush=True)
    try:
        asyncio.run(probe.exercise())
    except Exception as error:
        print(json.dumps({"passed":False,"error":getattr(error,"code",type(error).__name__),"manifest":str(probe.path)}))
        return 1
    print(json.dumps({"passed":True,"checks":len(probe.record["checks"]),"cleanup":probe.record["cleanup"],"manifest":str(probe.path)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
