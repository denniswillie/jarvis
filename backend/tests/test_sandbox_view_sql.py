"""One manifested real-SQL viewer authority proof; never calls E2B."""
from datetime import UTC,datetime,timedelta
import os
from pathlib import Path
import unittest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock
from app.errors import JarvisHTTPError
from app.runtime.feature_config import TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.models import SandboxInfo,SandboxPolicy
from app.sandbox.view_access import SandboxViewAccess
from app.sandbox.view_provider import DesktopConnection
from app.sandbox.view_service import SandboxViewService
from app.workspace.presentation import PresentationAck, PresentationRepository, PresentationService
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture

@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),'Explicit disposable schema required')
class ViewerSQLTests(unittest.IsolatedAsyncioTestCase):
 async def test_exact_owned_active_allocation_and_revocation(self):
  async with source_fixture(os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']),test_name=self.id()) as f:
   tasks=TaskRepository(f.pool,TaskLimits(lease_seconds=300));allocations=SandboxAllocationRepository(tasks)
   policy=SandboxPolicy(template='desktop')
   async with f.pool.acquire() as c:
    session=await c.fetchval('insert into jarvis_sessions(owner_id) values($1) returning id',f.owner)
    other=await c.fetchval('insert into jarvis_sessions(owner_id) values($1) returning id',f.owner)
    payload=await tasks.payload(c,f.owner,{'text':'View a synthetic desktop.'})
    event=await c.fetchval("insert into jarvis_context_events(owner_id,source_namespace,source_event_id,kind,finality,trust,payload_id) values($1,'fixture-view',$2,'user_request','final','user_input',$3) returning id",f.owner,str(uuid4()),payload)
    scope=await tasks.payload(c,f.owner,policy.model_dump(mode='json'))
    grant=await c.fetchval("insert into jarvis_operation_grants(owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at) values($1,'sandbox.execute',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id",f.owner,str(uuid4()),event,scope)
   task=await tasks.create(f.owner,TaskCreate(request_id=uuid4(),session_id=session,title='Fixture desktop',requested_outcome='View synthetic content',knowledge_area_id=f.area,grant_ids=(grant,)),model='fixture-no-model')
   claim=await tasks.claim_next('fixture-view-worker',('frontier',));self.assertIsNotNone(claim)
   row=await allocations.reserve(claim,grant);token=await allocations.begin_creation(claim,row['id'])
   receipt=SandboxInfo(sandbox_id='fixture-view-'+row['id'].hex,scope=allocations.scope(row),policy=policy,state='running',expires_at=datetime.now(UTC)+timedelta(minutes=4),template_id='desktop')
   await allocations.record_created(f.owner,row['id'],token,receipt);await allocations.ready(claim,row['id'])
   access=SandboxViewAccess(tasks,owner_id=f.owner)
   selected=await access.resolve(f.owner,session,task_id=task.id,allocation_id=row['id'])
   self.assertEqual(selected.task_id,task.id);self.assertEqual(selected.allocation_id,row['id'])
   self.assertEqual((await access.choices(f.owner,session))[0],[selected])
   for owner,selected_session,allocation,generation in [(f.other,session,row['id'],selected.generation),(f.owner,other,row['id'],selected.generation),(f.owner,session,uuid4(),selected.generation),(f.owner,session,row['id'],'b'*64)]:
    with self.assertRaises(JarvisHTTPError):await access.allocation(owner,selected_session,allocation,generation)
   async with f.pool.acquire() as c:
    before=await c.fetchrow('select phase,deadline,provider_expires_at,submission_token from jarvis_sandbox_allocations where id=$1',row['id'])
   await access.allocation(f.owner,session,row['id'],selected.generation)
   # The actual persisted phone journal keeps accepted/claimed/displayed distinct.
   views=SandboxViewService(access,SimpleNamespace(inspect=AsyncMock(return_value=DesktopConnection(
    'wss://fixture.invalid','synthetic-private-token',selected.deadline))))
   async def resolve(ctx,kind,identity,version):
    self.assertEqual(kind,'sandbox')
    return await views.resolve_presentation(ctx,identity,version)
   presentations=PresentationService(PresentationRepository(f.pool),resolver=resolve)
   ctx=SimpleNamespace(owner_id=f.owner,session_id=session,operation_id=uuid4())
   accepted=await views.present(ctx,presentations,task_id=task.id,allocation_id=row['id'])
   self.assertEqual(accepted['outcome'],'accepted')
   self.assertEqual(accepted['artifact']['kind'],'sandbox')
   offered=(await presentations.list_requests(ctx))['items']
   self.assertEqual(len(offered),1);self.assertEqual(offered[0]['version_id'],selected.generation)
   await presentations.acknowledge(ctx,ctx.operation_id,PresentationAck(session_id=session,state='claimed'))
   self.assertEqual((await presentations.operation_receipt(ctx,ctx.operation_id))['outcome'],'outcome_unknown')
   await presentations.acknowledge(ctx,ctx.operation_id,PresentationAck(session_id=session,state='displayed'))
   shown=await presentations.operation_receipt(ctx,ctx.operation_id)
   self.assertEqual(shown['outcome'],'completed');self.assertIn('live sandbox screen',shown['summary'])
   async with f.pool.acquire() as c:
    after=await c.fetchrow('select phase,deadline,provider_expires_at,submission_token from jarvis_sandbox_allocations where id=$1',row['id'])
    self.assertEqual(before,after)
    await c.execute("update jarvis_operation_grants set status='revoked' where id=$1",grant)
   with self.assertRaises(JarvisHTTPError):await access.allocation(f.owner,session,row['id'],selected.generation)
   self.assertEqual((await access.choices(f.owner,session))[0],[])
