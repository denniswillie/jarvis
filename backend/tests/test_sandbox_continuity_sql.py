"""Real PostgreSQL continuity guards, with controlled native responses only."""
import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.sandbox.cleanup import SandboxCleanupRepository, SandboxCleaner
from app.sandbox.models import CommandRequest, SandboxError
from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.models import TaskStatus, TaskMutation, WorkerOutcome, TaskError, ClaimLost
from app.tasks.runner import WorkerContext
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.sandbox_fixture import sandbox_task_fixture
from tests.test_sandbox_commands import CommandProvider


class RetainedProvider(CommandProvider):
    from app.sandbox.models import E2BConfiguration
    configuration = E2BConfiguration(api_key='offline-only', request_timeout_seconds=1)

    async def pause(self, handle):
        self.resources[handle.sandbox_id] = handle.model_copy(update={'state':'paused'})
        return self.resources[handle.sandbox_id]

    async def resume(self, handle, *, timeout_seconds=None):
        self.resources[handle.sandbox_id] = handle.model_copy(update={'state':'running'})
        return self.resources[handle.sandbox_id]

    async def info(self, identity, scope, policy):
        if identity not in self.resources:
            raise SandboxError('sandbox_not_found')
        return self.resources[identity]

    async def discover(self, scope, policy):
        return [i for i,h in self.resources.items() if h.provider_scope==scope and h.policy==policy]

    async def kill(self, identity, scope, policy):
        self.resources.pop(identity, None)
        return True


async def bridge_for(f, claim, provider):
    context = WorkerContext(claim,f.tasks,TaskBudgetRepository(f.tasks),TaskActionRepository(f.tasks))
    prepared = await WorkerContextFactory(f.source.pool, sandbox_provider=provider)(context)
    return prepared.tools


async def command(broker, request):
    result = await broker.execute(request)
    for _ in range(60):
        if result['sandbox_command']['state']=='completed':
            assert result['sandbox_command']['exit_code']==0, result
            return result
        await asyncio.sleep(.25)
        result = await broker.status(request.operation_key)
    raise AssertionError('command did not finish within bounded observation')


async def same_work_roundtrip(f, provider, evidence):
    """Shared real-SQL/native probe path, no model or Voice session involved."""
    first = await bridge_for(f, f.claim, provider)
    request = CommandRequest(operation_key='continuity-write',
        command="printf 'jarvis-continuity-proof' > continuity.txt; cat continuity.txt")
    written = await command(first.broker, request)
    row = await first.broker.allocation()
    evidence.update(allocation_id=str(row['id']), sandbox_id=row['sandbox_id'],
        native_scope=first.broker.allocations.native_scope(row).model_dump(mode='json'),
        policy=first.broker.allocations.policy(row).model_dump(mode='json'))
    await f.tasks.finish(f.claim, WorkerOutcome(TaskStatus.PAUSED, {'reason':'awaiting ordinary follow-up'}))
    cleaner = SandboxCleaner(SandboxCleanupRepository(first.broker.allocations), provider)
    paused = await cleaner.run_once(f.source.owner,'continuity-test',allocation_id=row['id'])
    assert paused['state']=='paused', paused
    evidence['paused']=True
    current = await f.tasks.mutate(f.source.owner,f.task.id,'resume',
        TaskMutation(request_id=uuid4(),expected_revision=f.task.revision))
    second_claim = await f.tasks.claim_next('continuity-followup',('frontier',),owner_id=f.source.owner,
        worker_capabilities=('sandbox.execute',))
    assert second_claim is not None
    second = await bridge_for(f, second_claim, provider)
    assert second.broker.continuation_id == row['id']
    read = await command(second.broker, CommandRequest(operation_key='continuity-read',command='cat continuity.txt'))
    resumed = await second.broker.allocation()
    assert row['id']==resumed['id'] and row['sandbox_id']==resumed['sandbox_id']
    assert row['deadline']==resumed['deadline']
    evidence.update(resumed=True, retained_output=read['sandbox_command']['stdout'],
        same_allocation=True,same_sandbox=True,deadline_unchanged=True)
    try:
        await first.broker.allocations.request_cleanup(f.source.owner,row['id'])
    except TaskError as error:
        assert error.code=='sandbox_cleanup_stale'
    else:
        raise AssertionError('stale cleanup was accepted')
    evidence['stale_cleanup_rejected']=True
    return first,second,cleaner,row


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
    os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),'Explicit manifested fixture required')
class ContinuitySQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_roundtrip_and_database_guards(self):
        async with sandbox_task_fixture(os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
            Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']),test_name=self.id()) as f:
            provider=RetainedProvider()
            evidence={}
            first,second,cleaner,row=await same_work_roundtrip(f,provider,evidence)
            self.assertEqual(provider.creates,1)
            self.assertEqual(provider.starts,2)
            # Native identity/deadline and cleanup tombstone are enforced in PostgreSQL.
            async with f.source.pool.acquire() as c:
                for sql in ("update jarvis_sandbox_allocations set deadline=deadline+interval '1 hour' where id=$1",
                            "update jarvis_sandbox_allocations set native_scope='{}'::jsonb where id=$1",
                            "update jarvis_sandbox_allocations set task_revision=task_revision+1 where id=$1"):
                    with self.assertRaises(asyncpg.CheckViolationError):
                        await c.execute(sql,row['id'])
            # An old generation cleanup claim must not become usable after transfer.
            with self.assertRaises(ClaimLost):
                await first.broker.allocations.current(f.claim,row['id'])
            await second.broker.allocations.request_cleanup(f.source.owner,row['id'])
            async with f.source.pool.acquire() as c:
                with self.assertRaises(asyncpg.CheckViolationError):
                    await c.execute("update jarvis_sandbox_allocations set phase='ready' where id=$1",row['id'])
            result=await cleaner.run_once(f.source.owner,'continuity-cleanup',allocation_id=row['id'])
            self.assertEqual(result['state'],'cleaned')
            self.assertFalse(provider.resources)
