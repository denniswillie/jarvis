"""Private sibling IO and approved original-clock bounds; no SQL/provider/native."""
import asyncio
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import test_native_memory_readiness_observer as composition
from evals.responsive_voice.native_memory_readiness_control import (
    CONFIG_FORMAT, REQUEST_FORMAT, RECEIPT_FORMAT, REQUEST_NAME, RECEIPT_NAME, SCOPE_NAME,
    MemoryReadinessFileBridge, parse_request,
)


def private_json(path, value):
    data = json.dumps(value, default=lambda x:x.isoformat() if isinstance(x,datetime) else str(x)).encode()
    path.write_bytes(data)
    path.chmod(0o600)
    return data


class MemoryControlTests(unittest.IsolatedAsyncioTestCase):
    source_row = composition.MemoryObserverTests.source_row

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        await composition.MemoryObserverTests.asyncSetUp(self)
        self.root = Path(self.temp.name).resolve()
        self.operator = self.root/'operator'; self.operator.mkdir(mode=0o700)
        self.inputs = self.root/'stage'/'inputs'; self.inputs.mkdir(parents=True,mode=0o700)
        self.controls = self.inputs.parent/'controls'; self.controls.mkdir(mode=0o700)
        self.gate = {'kind':'after_manual_response','after_input_ordinal':0,'at_seconds':20,
            'timeout_seconds':10,'canonical_requirement':'retain_searchable','playback_requirement':'none'}
        self.config = replace(self.config,configuration_path=self.operator/'configuration.json',
            status_path=self.operator/'status.json',origin_invocation_id=self.config.invocation_id,
            approved_gates=({'target_input_ordinal':1,'trigger':self.gate},))
        self.observer.config = self.config
        self.boot = str(uuid4())
        self.anchor = {'format':'jarvis.replay.source_clock.v1','run_id':self.host_scope.run_id,
            'origin_invocation_id':str(self.config.invocation_id),'monotonic_before_seconds':100.0,
            'monotonic_after_seconds':100.2,'utc_unix_seconds':self.now.timestamp()-20,
            'boot_session_id':self.boot}
        arm_utc = (self.anchor['utc_unix_seconds']+0)+20
        self.request = replace(self.request,issued_at=datetime.fromtimestamp(arm_utc,UTC),
            expires_at=datetime.fromtimestamp(arm_utc+10,UTC))
        self.sample = {'clock':'ProcessInfo.systemUptime','boot_session_id':self.boot,'uptime_seconds':125.0}
        self.bridge = MemoryReadinessFileBridge(self.observer,sample_clock=lambda:dict(self.sample))
        self.observer.bridge = self.bridge
        plan = {'version':1,'run_id':self.host_scope.run_id,'owner_id':self.host_scope.owner_id,
            'area_id':str(self.host_scope.area_id),'memory_bank_id':self.host_scope.memory_bank_id,
            'capture_mode':'manual','voice_route':'realtime','replay_invocation_id':str(self.config.invocation_id),
            'replay_source_clock':{'version':1,'phase_start_seconds':0,'anchor_file':None,'anchor_sha256':None},
            'memory_readiness':{'version':1,'scope_file':SCOPE_NAME,'manifest_sha256':self.config.manifest_sha256},
            'utterances':[{'clip_id':'a','trigger':{'kind':'at_time','at_seconds':0}},
                          {'clip_id':'b','trigger':self.gate}]}
        raw = private_json(self.inputs/'replay.json',plan)
        private_json(self.inputs/SCOPE_NAME,asdict(self.observer.scope))
        self.configuration = {'format':CONFIG_FORMAT,'run_id':self.host_scope.run_id,
            'invocation_id':str(self.config.invocation_id),'manifest_sha256':self.config.manifest_sha256,
            'scope_file':str(self.inputs/SCOPE_NAME),'runtime_manifest_file':str(self.inputs/'replay.json'),
            'runtime_manifest_sha256':hashlib.sha256(raw).hexdigest(),'control_root':str(self.controls),
            'source_anchor_path':str(self.operator/'source-clock.json')}
        private_json(self.config.configuration_path,self.configuration)
        await self.bridge.step()
        private_json(self.operator/'source-clock.json',self.anchor)

    def put_request(self, request=None, **changes):
        request = replace(request or self.request,nonce=uuid4().hex,**changes)
        data = private_json(self.controls/REQUEST_NAME,{'format':REQUEST_FORMAT,'request':asdict(request)})
        return request,data

    async def consume(self, **changes):
        request,data = self.put_request(**changes)
        await self.bridge.step()
        return request,data,json.loads((self.controls/RECEIPT_NAME).read_bytes())

    async def test_literal_request_bytes_full_scope_and_policy_projection(self):
        request,data,receipt = await self.consume()
        self.assertEqual(receipt['format'],RECEIPT_FORMAT)
        self.assertEqual(receipt['request_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt['readiness']['status'],'ready',receipt)
        self.assertEqual(receipt['readiness']['final_output_policy'],self.policy)
        self.assertEqual(parse_request(data),request)
        self.assertEqual(receipt['readiness']['source_clock']['deadline_uptime_seconds'],130.1)
        self.assertNotIn('PRIVATE',json.dumps(receipt))
        self.assertEqual(self.observer.query_count,9)

    async def test_nonce_refresh_cannot_extend_original_approved_deadline(self):
        await self.consume()
        self.put_request(expires_at=self.request.expires_at+timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError,'memory_original_deadline_mismatch'):
            await self.bridge.step()
        self.assertEqual(self.observer.query_count,9)

    async def test_changed_arm_time_cannot_move_fixed_window(self):
        self.put_request(issued_at=self.request.issued_at+timedelta(seconds=1),
                         expires_at=self.request.expires_at+timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError,'memory_original_deadline_mismatch'):
            await self.bridge.step()
        self.assertEqual(self.pool.queries,[])

    async def test_expired_earliest_and_future_unknown_other_boot_clocks(self):
        for uptime,status in ((119,'waiting'),(130.1,'expired'),(131,'expired')):
            with self.subTest(uptime=uptime):
                self.sample['uptime_seconds']=uptime
                _,_,receipt=await self.consume()
                self.assertEqual(receipt['readiness']['status'],status)
        self.assertEqual(self.pool.queries,[])
        for change in ({'clock':'python.monotonic'},{'boot_session_id':str(uuid4())},{'uptime_seconds':99}):
            old=dict(self.sample)
            with self.subTest(change=change):
                self.sample.update(change);self.put_request()
                with self.assertRaises(ValueError):await self.bridge.step()
            self.sample=old

    async def test_original_anchor_bytes_cannot_change_after_first_consume(self):
        await self.consume()
        private_json(self.operator/'source-clock.json',{**self.anchor,'utc_unix_seconds':self.anchor['utc_unix_seconds']+1})
        self.put_request()
        with self.assertRaisesRegex(ValueError,'memory_source_anchor_changed'):
            await self.bridge.step()

    async def test_unpublished_validated_anchor_is_pending_not_guessed(self):
        (self.operator/'source-clock.json').unlink()
        _,_,receipt=await self.consume()
        self.assertEqual(receipt['readiness']['reason'],'validated_source_anchor_pending')
        self.assertEqual(self.pool.queries,[])

    async def test_foreign_scope_mode_target_or_connection_rebinding_rejected(self):
        await self.consume()
        for change in ({'connection_id':'replacement'},{'target_input_ordinal':2},
                       {'canonical_requirement':'finalized_input'},
                       {'scope':replace(self.request.scope,memory_bank_id='jarvis_'+uuid4().hex)}):
            with self.subTest(change=change):
                self.put_request(**change)
                with self.assertRaises(ValueError):await self.bridge.step()
        self.assertEqual(self.observer.query_count,9)

    async def test_duplicate_nonce_rebound_bytes_fail_and_identical_file_never_requeries(self):
        request,data,_=await self.consume()
        await self.bridge.step()
        self.assertEqual(self.observer.query_count,9)
        private_json(self.controls/REQUEST_NAME,{'format':REQUEST_FORMAT,
            'request':asdict(replace(request,response_id='another'))})
        with self.assertRaisesRegex(ValueError,'memory_nonce_reused_or_rebound'):
            await self.bridge.step()

    async def test_new_request_supersedes_inflight_result_without_old_publication(self):
        self.pool.pause=asyncio.Event()
        self.put_request()
        task=asyncio.create_task(self.bridge.step())
        await self.pool.entered.wait()
        _,new_bytes=self.put_request(response_id='new-candidate')
        self.pool.pause.set()
        await task
        self.assertEqual(self.bridge.superseded,1)
        self.assertFalse((self.controls/RECEIPT_NAME).exists())
        self.assertEqual((self.controls/REQUEST_NAME).read_bytes(),new_bytes)

    async def test_clock_expiring_during_read_never_publishes_ready(self):
        self.pool.pause=asyncio.Event();self.put_request()
        task=asyncio.create_task(self.bridge.step())
        await self.pool.entered.wait()
        self.sample['uptime_seconds']=131;self.pool.pause.set()
        await task
        self.assertEqual(json.loads((self.controls/RECEIPT_NAME).read_bytes())['readiness']['status'],'expired')

    async def test_parser_permissions_bounds_and_input_replacement(self):
        _,data=self.put_request()
        for raw in (data[:-1]+b',"format":"duplicate"}',data+b' '*16385,
                    data.replace(b'"prior_input_ordinal": 0',b'"prior_input_ordinal": NaN')):
            with self.subTest(raw_bytes=len(raw)),self.assertRaises(Exception):parse_request(raw)
        (self.controls/REQUEST_NAME).chmod(0o644)
        with self.assertRaises(Exception):await self.bridge.step()
        (self.controls/REQUEST_NAME).chmod(0o600)
        private_json(self.inputs/SCOPE_NAME,{**asdict(self.observer.scope),'database_schema':'other'})
        with self.assertRaisesRegex(ValueError,'memory_staged_inputs_replaced'):await self.bridge.step()
        self.assertEqual(self.pool.queries,[])

    async def test_explicit_loop_cancellation_joins_and_records_terminal_status(self):
        self.pool.pause=asyncio.Event();self.put_request()
        self.bridge.task=asyncio.create_task(self.bridge._run())
        await self.pool.entered.wait()
        receipt=await self.observer.shutdown()
        self.assertTrue(receipt['observer_joined'] and receipt['io_loop_joined'])
        self.assertTrue(self.bridge.task.done())
        self.assertFalse(self.observer.inflight)
        self.assertEqual(json.loads(self.config.status_path.read_bytes())['status'],'closed')


if __name__=='__main__':unittest.main()
