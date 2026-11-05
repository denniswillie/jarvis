"""Run-owned authenticated Hindsight fixture. No shared service is stopped."""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

import httpx


# The fixture data directory is a 512 MB tmpfs. Postgres's default 1 GB
# WAL target can exhaust it before a checkpoint during repeated migrations.
# Keep fsync, full-page writes and synchronous commit at their durable defaults.
POSTGRES_ARGUMENTS = (
    "postgres", "-c", "max_wal_size=64MB", "-c", "min_wal_size=32MB",
    "-c", "checkpoint_timeout=30s", "-c", "checkpoint_completion_target=0.5",
)


WORKER_ENV = {
    "HINDSIGHT_API_WORKER_MAX_SLOTS": "2",
    "HINDSIGHT_API_WORKER_CONSOLIDATION_RESERVED_SLOTS": "1",
}


from app.knowledge.service_policy import PRIVACY_ENV

PRIVACY_FIELDS = ("llm_trace_enabled", "audit_log_enabled", "enable_observation_history", "enable_mental_model_history")


def configured_privacy_policy(runtime, env):
    """Verify environment defaults; native bank audits cover effective behavior."""
    result = subprocess.run(
        [str(Path(runtime) / "bin/python"), "-c",
         "import json; from hindsight_api.config import HindsightConfig; c=HindsightConfig.from_env(); "
         + "print(json.dumps({k:getattr(c,k) for k in " + repr(PRIVACY_FIELDS) + "}))"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError("native_privacy_configuration_unavailable")
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or set(value) != set(PRIVACY_FIELDS) or any(flag is not False for flag in value.values()):
        raise RuntimeError("native_payload_history_must_be_disabled")
    return value


def configured_worker_allocation(runtime, env):
    # Ask the pinned runtime for its effective values; HTTP readiness alone
    # cannot establish that a retain job has usable worker capacity.
    result = subprocess.run(
        [str(Path(runtime) / "bin/python"), "-c",
         "import json; from hindsight_api.config import get_config; c=get_config(); "
         "print(json.dumps({'max_slots':c.worker_max_slots,'reservations':c.worker_slot_reservations}))"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError("native_worker_configuration_unavailable")
    value = json.loads(result.stdout)
    slots, reservations = value.get("max_slots"), value.get("reservations")
    if (type(slots) is not int or not 1 <= slots <= 2 or not isinstance(reservations, dict)
            or any(type(count) is not int or count < 0 for count in reservations.values())
            or slots - sum(reservations.values()) < 1):
        raise RuntimeError("native_retention_capacity_unavailable")
    return value | {"shared_slots": slots - sum(reservations.values())}


def save(path, value):
    temporary=path.with_suffix(".tmp")
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w") as handle:
        json.dump(value,handle,indent=2)
    temporary.replace(path)


def docker(*arguments, env=None):
    result=subprocess.run(["docker",*arguments],env=env,capture_output=True,text=True,timeout=90)
    if result.returncode:
        raise RuntimeError("isolated_docker_operation_failed")
    return result.stdout.strip()


def wait_for_postgres(name, *, timeout=45):
    """The image's temporary initialization server only accepts Unix sockets."""
    deadline=time.monotonic()+timeout
    while True:
        try:
            docker("exec",name,"pg_isready","-h","127.0.0.1","-U","postgres","-d","hindsight")
            return
        except RuntimeError:
            if time.monotonic()>=deadline:
                raise
            time.sleep(0.5)


@contextmanager
def hindsight_service(root, *, runtime, api_key, max_starts=3):
    if type(max_starts) is not int or not 1 <= max_starts <= 4:
        raise ValueError("Invalid native service restart budget")
    root=Path(root).resolve()
    if root.parent!=Path("/tmp").resolve() or not root.name.startswith("jarvis-m02-native."):
        raise ValueError("Use a new private /tmp/jarvis-m02-native.* directory")
    root.mkdir(mode=0o700)
    name="jarvis-m02-"+secrets.token_hex(8)
    manifest={"format":"jarvis.m02.native_service.v1","container":name,"status":"allocating",
              "process":None,"process_history":[],"restart_count":0,
              "postgres_arguments":list(POSTGRES_ARGUMENTS)}
    path=root/"service-resources.json"
    save(path,manifest)

    def remove_container():
        result=subprocess.run(["docker","inspect","--format",'{{index .Config.Labels "jarvis.m02"}}',name],
                              capture_output=True,text=True,timeout=20)
        if result.returncode==0:
            if result.stdout.strip()!=name: raise RuntimeError("native_container_owner_mismatch")
            docker("rm","-f","-v",name)

    def stop_process(process):
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    try:
        with ExitStack() as stack:
            stack.callback(remove_container)
            password,token=secrets.token_urlsafe(24),secrets.token_urlsafe(24)
            env=os.environ.copy()
            env["POSTGRES_PASSWORD"]=password
            docker("run","-d","--name",name,"--label","jarvis.m02="+name,
                   "--memory","768m","--cpus","1","--tmpfs","/var/lib/postgresql/data:rw,size=512m",
                   "-p","127.0.0.1::5432","-e","POSTGRES_PASSWORD","-e","POSTGRES_DB=hindsight",
                   "pgvector/pgvector:pg16",*POSTGRES_ARGUMENTS,env=env)
            wait_for_postgres(name)
            port=docker("port",name,"5432/tcp").rsplit(":",1)[1]
            docker("exec",name,"psql","-h","127.0.0.1","-U","postgres","-d","hindsight","-c","CREATE EXTENSION IF NOT EXISTS vector;")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0))
                api_port=sock.getsockname()[1]
            service_env={
                "PATH":os.environ.get("PATH","/usr/bin:/bin"),"HOME":str(root/"home"),
                "PYTHONPATH":str(Path(__file__).resolve().parents[2]),
                "XDG_CACHE_HOME":str(root/"cache"),"PYTHONUNBUFFERED":"1","PYTHONDONTWRITEBYTECODE":"1",
                "HINDSIGHT_API_DATABASE_URL":f"postgresql://postgres:{password}@127.0.0.1:{port}/hindsight",
                "HINDSIGHT_API_LLM_PROVIDER":"openai-responses","HINDSIGHT_API_LLM_API_KEY":api_key,
                "HINDSIGHT_API_LLM_MODEL":"gpt-5.4-mini","HINDSIGHT_API_LLM_MAX_CONCURRENT":"1",
                "HINDSIGHT_API_LLM_MAX_RETRIES":"0","HINDSIGHT_API_LLM_TIMEOUT":"90",
                "HINDSIGHT_API_LLM_TEMPERATURE":"none","HINDSIGHT_API_LLM_REASONING_EFFORT":"low",
                "HINDSIGHT_API_WORKER_MAX_RETRIES":"0", **WORKER_ENV, **PRIVACY_ENV,
                "HINDSIGHT_API_EMBEDDINGS_PROVIDER":"openai","HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY":api_key,
                "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL":"text-embedding-3-small",
                "HINDSIGHT_API_RERANKER_PROVIDER":"rrf","HINDSIGHT_API_ENABLE_RERANKING":"false",
                "HINDSIGHT_API_TENANT_EXTENSION":"hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension",
                "HINDSIGHT_API_TENANT_API_KEY":token,
            }
            manifest["worker_allocation"] = configured_worker_allocation(runtime, service_env)
            manifest["privacy_policy"] = configured_privacy_policy(runtime, service_env)
            save(path, manifest)
            log=stack.enter_context((root/"service.log").open("w"))
            process = None

            def stop_owned():
                if process is not None:
                    stop_process(process)

            def start_owned():
                nonlocal process
                if process is not None and process.poll() is None:
                    raise RuntimeError("native_process_already_running")
                if len(manifest["process_history"]) >= max_starts:
                    raise RuntimeError("native_restart_budget_exceeded")
                process = subprocess.Popen(
                    [str(Path(runtime)/"bin/python"),"-m","app.knowledge.service_entrypoint",
                     "--host","127.0.0.1","--port",str(api_port)],
                    cwd=root,env=service_env,stdout=log,stderr=subprocess.STDOUT,
                )
                manifest["process_history"].append(process.pid)
                manifest.update(process=process.pid,url=f"http://127.0.0.1:{api_port}",
                                status="starting",restart_count=len(manifest["process_history"])-1)
                save(path,manifest)
                deadline=time.monotonic()+180
                with httpx.Client(timeout=3,trust_env=False) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("native_hindsight_exited_during_startup")
                        try:
                            response=client.get(manifest["url"]+"/openapi.json",
                                                headers={"Authorization":"Bearer "+token})
                            if response.status_code==200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic()>=deadline:
                            raise TimeoutError("native_hindsight_startup_timeout")
                        time.sleep(0.5)
                manifest["status"]="running"
                save(path,manifest)

            def stopped_for_recovery():
                if process is not None and process.poll() is None:
                    raise RuntimeError("native_process_must_stop_before_database_recovery")
                label = docker("inspect", "--format", '{{index .Config.Labels "jarvis.m02"}}', name)
                if label != name:
                    raise RuntimeError("native_recovery_container_owner_mismatch")

            def snapshot_owned():
                stopped_for_recovery()
                if "database_snapshot" in manifest:
                    raise RuntimeError("native_snapshot_already_allocated")
                backup = "/tmp/jarvis-native-recovery/hindsight.dump"
                manifest["database_snapshot"] = {"path": backup, "status": "writing", "database": "hindsight"}
                save(path, manifest)
                docker("exec", name, "install", "-d", "-m", "700", "-o", "postgres",
                       "/tmp/jarvis-native-recovery")
                docker("exec", "--user", "postgres", name, "pg_dump", "-U", "postgres",
                       "-d", "hindsight", "-Fc", "-f", backup)
                docker("exec", name, "chmod", "600", backup)
                digest = docker("exec", name, "sha256sum", backup).split()[0]
                manifest["database_snapshot"].update(status="sealed", sha256=digest)
                save(path, manifest)
                return dict(manifest["database_snapshot"])

            def restore_owned():
                stopped_for_recovery()
                snapshot = manifest.get("database_snapshot")
                if not snapshot or snapshot["status"] != "sealed":
                    raise RuntimeError("native_snapshot_unavailable")
                if docker("exec", name, "sha256sum", snapshot["path"]).split()[0] != snapshot["sha256"]:
                    raise RuntimeError("native_snapshot_digest_mismatch")
                manifest["database_restore"] = "restoring"
                save(path, manifest)
                # Only this manifested fixture database; canonical fixture data
                # in its separate postgres database is deliberately unaffected.
                docker("exec", name, "dropdb", "-U", "postgres", "--force", "hindsight")
                docker("exec", name, "createdb", "-U", "postgres", "hindsight")
                docker("exec", name, "pg_restore", "-U", "postgres", "-d", "hindsight",
                       "--exit-on-error", "--no-owner", "--no-privileges", snapshot["path"])
                manifest["database_restore"] = "restored"
                save(path, manifest)

            # Resolve the current process at cleanup time, including any restart.
            stack.callback(stop_owned)
            start_owned()
            yield {"url":manifest["url"],"token":token,"container":name,
                   "stop_owned":stop_owned,"start_owned":start_owned,
                   "snapshot_owned":snapshot_owned,"restore_owned":restore_owned,
                   "source_database_url":f"postgresql://postgres:{password}@127.0.0.1:{port}/postgres"}
        residue=subprocess.run(["docker","inspect",name],capture_output=True,timeout=20)
        if residue.returncode==0: raise RuntimeError("native_container_cleanup_incomplete")
        if process.poll() is None: raise RuntimeError("native_process_cleanup_incomplete")
        manifest["status"]="cleaned"
    except BaseException:
        manifest["status"]="failed_or_cleanup_unverified"
        raise
    finally:
        save(path,manifest)
