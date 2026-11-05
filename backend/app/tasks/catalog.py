"""Configured provider metadata and expiring worker readiness, without native imports."""
from __future__ import annotations

from dataclasses import dataclass

from app.tasks.models import TaskError


@dataclass(frozen=True)
class ConfiguredTaskProvider:
    model: str


class TaskWorkerRegistry:
    def __init__(self, pool):
        self.pool = pool

    async def advertise(self, *, worker_id, owner_id, host_id, providers, ttl_seconds, capabilities=()):
        if not 2 <= ttl_seconds <= 300:
            raise ValueError("Invalid readiness lease")
        async with self.pool.acquire() as connection, connection.transaction():
            for kind, provider in providers.items():
                identity = await connection.fetchval("""
                    insert into jarvis_task_workers(worker_id,owner_id,host_id,backend_kind,model,expires_at,capabilities)
                    values($1,$2,$3,$4,$5,clock_timestamp()+$6*interval '1 second',$7::text[])
                    on conflict(worker_id,backend_kind) do update set
                        expires_at=excluded.expires_at,updated_at=clock_timestamp(),draining=false,capabilities=excluded.capabilities
                    where jarvis_task_workers.owner_id=excluded.owner_id
                        and jarvis_task_workers.host_id=excluded.host_id
                        and jarvis_task_workers.model=excluded.model
                    returning worker_id
                """,worker_id,owner_id,host_id,kind,provider.model,float(ttl_seconds),list(capabilities))
                if identity is None:
                    raise TaskError("task_worker_identity_conflict","Worker identity is already bound to another scope.",409)

    async def withdraw(self, owner_id, worker_id):
        async with self.pool.acquire() as connection:
            await connection.execute("""
                update jarvis_task_workers set draining=true,expires_at=clock_timestamp(),
                    updated_at=clock_timestamp() where owner_id=$1 and worker_id=$2
            """,owner_id,worker_id)

    async def ready(self, owner_id, kind, model, *, worker_id=None, connection=None, required_capabilities=()):
        if connection is None:
            async with self.pool.acquire() as acquired:
                return await self.ready(owner_id, kind, model, worker_id=worker_id, connection=acquired,
                                        required_capabilities=required_capabilities)
        return bool(await connection.fetchval("""
                select exists(select 1 from jarvis_task_workers where owner_id=$1 and backend_kind=$2
                    and model=$3 and not draining and expires_at>clock_timestamp()
                    and ($4::text is null or worker_id=$4)
                    and capabilities @> $5::text[])
            """,owner_id,kind,model,worker_id,list(required_capabilities)))

    async def require(self, owner_id, kind, model, *, connection=None, required_capabilities=()):
        if not await self.ready(owner_id,kind,model,connection=connection,required_capabilities=required_capabilities):
            raise TaskError("task_worker_not_ready",
                            "No ready worker serves this owner and model; nothing new was accepted.",503)
