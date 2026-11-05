"""One synthetic task/grant fixture shared by SQL and native sandbox checks."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.models import SandboxPolicy
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@dataclass
class TaskSandboxFixture:
    source: object
    tasks: object
    allocations: object
    policy: object
    event: object
    grant: object
    task: object
    claim: object


@asynccontextmanager
async def sandbox_task_fixture(dsn,directory,*,test_name,policy=None,model="fixture-no-model",
                               requested_outcome="Compute synthetic output offline.",with_session=False):
    policy = policy or SandboxPolicy()
    async with source_fixture(dsn,directory,test_name=test_name) as source:
        tasks = TaskRepository(source.pool,TaskLimits(lease_seconds=300))
        async with source.pool.acquire() as connection:
            session = (await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id",source.owner)
                if with_session else None)
            payload = await tasks.payload(connection,source.owner,{"text":"Run a synthetic task in an offline sandbox."})
            event = await connection.fetchval("""insert into jarvis_context_events
                (owner_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,'fixture-sandbox',$2,'user_request','final','user_input',$3) returning id""",
                source.owner,str(uuid4()),payload)
            scope = await tasks.payload(connection,source.owner,policy.model_dump(mode="json"))
            grant = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'sandbox.execute',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                source.owner,str(uuid4()),event,scope)
        task = await tasks.create(source.owner,TaskCreate(request_id=uuid4(),title="Synthetic sandbox",
            requested_outcome=requested_outcome,knowledge_area_id=source.area,session_id=session,
            grant_ids=(grant,)),model=model)
        claim = await tasks.claim_next("fixture-sandbox-worker",("frontier",))
        yield TaskSandboxFixture(source,tasks,SandboxAllocationRepository(tasks),policy,event,grant,task,claim)
