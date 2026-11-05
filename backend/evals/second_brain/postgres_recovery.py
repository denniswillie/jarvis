"""Disposable native backup/WAL fixture, with no configured production DSN."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from evals.second_brain.hindsight_service import docker as run_docker, save


IMAGE = "pgvector/pgvector:pg16"
LABEL = "jarvis.s11.restore"


def docker(*arguments, **kwargs):
    try:
        return run_docker(*arguments, **kwargs)
    except RuntimeError:
        # The command family is safe; arguments may include private credentials.
        raise RuntimeError("isolated_docker_" + arguments[0] + "_failed") from None


class PostgresRecoveryFixture:
    def __init__(self, root, *, data_mebibytes=512):
        if type(data_mebibytes) is not int or not 256 <= data_mebibytes <= 2048:
            raise ValueError("Isolated PostgreSQL data capacity must be 256-2048 MiB")
        self.data_mebibytes = data_mebibytes
        self.root = Path(root).resolve()
        if self.root.parent != Path("/tmp").resolve() or not self.root.name.startswith("jarvis-s11-restore."):
            raise ValueError("Use a new private /tmp/jarvis-s11-restore.* directory")
        self.root.mkdir(mode=0o700)
        self.name = "jarvis-s11-" + secrets.token_hex(8)
        self.password = secrets.token_urlsafe(24)
        self.volume = self.name + "-archive"
        self.containers = {role: self.name + "-" + role for role in ("source", "old", "missing", "recovered")}
        self.manifest = {
            "format": "jarvis.postgres_restore_fixture.v1", "status": "allocating",
            "volume": self.volume, "containers": self.containers, "removed": [],
            "production_database_used": False,
            "limits": {"data_mebibytes": data_mebibytes, "memory_mebibytes": data_mebibytes + 256, "cpus": 1},
        }
        self.save()

    def save(self):
        save(self.root / "resources.json", self.manifest)

    def start(self):
        self.manifest["image_id"] = docker("image", "inspect", IMAGE, "--format", "{{.Id}}")
        self.save()
        docker("volume", "create", "--label", LABEL + "=" + self.name, self.volume)
        # A temporary initialization container uses the already-manifested source
        # name, so interruption cannot create an unowned anonymous helper.
        docker("run", "--name", self.containers["source"], "--label", LABEL + "=" + self.name,
               "--network", "none", "--mount", f"type=volume,src={self.volume},dst=/work",
               IMAGE, "sh", "-ec",
               "mkdir /work/base /work/wal /work/empty; chown postgres:postgres /work/base /work/wal /work/empty; "
               "chmod 700 /work/base /work/wal /work/empty")
        self.remove("source")
        archive = ('if test -f /work/wal/%f; then cmp -s %p /work/wal/%f; '
                   'else cp %p /work/wal/%f.tmp && sync /work/wal/%f.tmp && '
                   'mv /work/wal/%f.tmp /work/wal/%f && sync /work/wal; fi')
        env = os.environ.copy()
        env["POSTGRES_PASSWORD"] = self.password
        docker(*self._run_args("source", readonly=False), "-e", "POSTGRES_PASSWORD", IMAGE,
               "postgres", "-c", "archive_mode=on", "-c", "archive_command=" + archive,
               "-c", "max_wal_size=256MB", "-c", "shared_buffers=32MB", env=env)
        self.ready("source")
        # Match the maintained database fixture's prerequisite: global extension
        # objects belong to public, never to its disposable owner schema.
        docker("exec", self.containers["source"], "psql", "-U", "postgres", "-d", "postgres",
               "-v", "ON_ERROR_STOP=1", "-c",
               "create extension if not exists pgcrypto with schema public; "
               "create extension if not exists vector with schema public;")
        self.manifest["status"] = "running"
        self.save()

    def _run_args(self, role, *, readonly):
        mount = f"type=volume,src={self.volume},dst=/work" + (",readonly" if readonly else "")
        return ["run", "-d", "--name", self.containers[role], "--label", LABEL + "=" + self.name,
                "--memory", f"{self.data_mebibytes + 256}m", "--cpus", "1",
                "--tmpfs", f"/var/lib/postgresql/data:rw,size={self.data_mebibytes}m",
                "--mount", mount, "-p", "127.0.0.1::5432"]

    def dsn(self, role):
        port = docker("port", self.containers[role], "5432/tcp").rsplit(":", 1)[1]
        return f"postgresql://postgres:{self.password}@127.0.0.1:{port}/postgres"

    def ready(self, role, *, timeout=45):
        deadline = time.monotonic() + timeout
        while True:
            try:
                docker("exec", self.containers[role], "pg_isready", "-h", "127.0.0.1",
                       "-U", "postgres", "-d", "postgres")
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("isolated_postgres_readiness_timeout") from None
                time.sleep(0.25)

    def backup(self):
        docker("exec", "--user", "postgres", self.containers["source"], "pg_basebackup",
               "-D", "/work/base", "-U", "postgres", "-X", "stream", "-c", "fast")
        docker("exec", "--user", "postgres", self.containers["source"],
               "pg_verifybackup", "/work/base")
        self.manifest["physical_base_backup_verified"] = True
        self.save()

    def await_archive(self, wal_name):
        # Trusted PostgreSQL output, never model input.
        if len(wal_name) != 24 or any(c not in "0123456789ABCDEF" for c in wal_name):
            raise ValueError("Invalid WAL filename")
        deadline = time.monotonic() + 30
        while True:
            try:
                docker("exec", self.containers["source"], "test", "-f", "/work/wal/" + wal_name)
                self.manifest["target_wal_archived"] = True
                self.save()
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("isolated_wal_archive_timeout") from None
                time.sleep(0.25)

    def restore(self, role, checkpoint=None):
        if role not in ("old", "missing", "recovered"):
            raise ValueError("Invalid restore role")
        commands = ["postgres", "-c", "archive_mode=off", "-c", "shared_buffers=32MB"]
        if checkpoint is not None:
            archive = "empty" if role == "missing" else "wal"
            commands += ["-c", f"restore_command=cp /work/{archive}/%f %p",
                         "-c", "recovery_target_lsn=" + checkpoint.target_lsn,
                         "-c", "recovery_target_timeline=" + str(checkpoint.timeline),
                         "-c", "recovery_target_inclusive=on", "-c", "recovery_target_action=pause"]
        script = 'cp -a /work/base/. "$PGDATA"/; '
        if checkpoint is not None:
            script += 'touch "$PGDATA/recovery.signal"; '
        script += 'chown -R postgres:postgres "$PGDATA"; exec docker-entrypoint.sh "$@"'
        docker(*self._run_args(role, readonly=True), "--entrypoint", "sh", IMAGE,
               "-ec", script, "restore", *commands)
        if role != "missing":
            self.ready(role)

    def wait_missing_target_failure(self):
        deadline = time.monotonic() + 45
        while True:
            state = json.loads(docker("inspect", "--format", "{{json .State}}", self.containers["missing"]))
            if not state["Running"]:
                # docker logs uses stderr for postgres, so collect both streams.
                result = subprocess.run(["docker", "logs", self.containers["missing"]],
                                        capture_output=True, text=True, timeout=20)
                expected = "recovery ended before configured recovery target was reached"
                if state["ExitCode"] == 0 or expected not in result.stdout + result.stderr:
                    raise RuntimeError("isolated_restore_failed_for_unexpected_reason")
                return {"exit_code": state["ExitCode"], "missing_target_fatal": True}
            if time.monotonic() >= deadline:
                raise RuntimeError("isolated_missing_wal_did_not_fail_closed")
            time.sleep(0.25)

    def remove(self, role):
        name = self.containers[role]
        if name not in docker("ps", "-a", "--format", "{{.Names}}").splitlines():
            return
        label = docker("inspect", "--format", '{{index .Config.Labels "' + LABEL + '"}}', name)
        if label != self.name:
            raise RuntimeError("restore_fixture_container_owner_mismatch")
        docker("rm", "-f", "-v", name)
        if role not in self.manifest["removed"]:
            self.manifest["removed"].append(role)
        self.save()

    def cleanup(self):
        errors = []
        for role in reversed(tuple(self.containers)):
            try:
                self.remove(role)
            except Exception as error:
                errors.append(type(error).__name__)
        try:
            if self.volume in docker("volume", "ls", "--format", "{{.Name}}").splitlines():
                label = docker("volume", "inspect", "--format",
                               '{{index .Labels "' + LABEL + '"}}', self.volume)
                if label != self.name:
                    raise RuntimeError("restore_fixture_volume_owner_mismatch")
                docker("volume", "rm", self.volume)
            remaining_containers = set(docker("ps", "-a", "--format", "{{.Names}}").splitlines())
            remaining_volumes = set(docker("volume", "ls", "--format", "{{.Name}}").splitlines())
            clean = not (remaining_containers & set(self.containers.values())) and self.volume not in remaining_volumes
        except Exception as error:
            errors.append(type(error).__name__)
            clean = False
        self.manifest["status"] = "cleaned" if clean and not errors else "cleanup_unverified"
        self.manifest["cleanup_errors"] = errors
        self.save()
        if self.manifest["status"] != "cleaned":
            raise RuntimeError("restore_fixture_cleanup_unverified")


@contextmanager
def postgres_recovery_fixture(root, *, data_mebibytes=512):
    fixture = PostgresRecoveryFixture(root, data_mebibytes=data_mebibytes)
    try:
        fixture.start()
        yield fixture
    finally:
        fixture.cleanup()
