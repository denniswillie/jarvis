"""Viewer access/lifetime and secret boundaries, with no provider allocation."""
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxInfo, SandboxPolicy, SandboxScope
from app.sandbox.view_access import DesktopTarget, SandboxViewAccess
from app.sandbox.view_provider import DesktopConnection, E2BDesktopViewer
from app.sandbox.view_secrets import SandboxViewSecrets
from app.sandbox.view_service import SandboxViewService
from app.workspace.presentation import PresentationRequest


def target():
    scope = SandboxScope(owner_id='viewer-owner', task_id=uuid4(), task_revision=1,
                         attempt_id=uuid4(), allocation_id=uuid4())
    info = SandboxInfo(sandbox_id='fixture-desktop', scope=scope, policy=SandboxPolicy(template='desktop'),
                       state='running', expires_at=datetime.now(UTC)+timedelta(minutes=4), template_id='desktop')
    return DesktopTarget(scope.owner_id, uuid4(), scope.task_id, scope.allocation_id, 'a'*64,
                         'Draft a design', info.expires_at, info)


class SecretTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'private'
        self.store = SandboxViewSecrets(self.root)
        self.target = target()

    def test_exact_holder_only_and_atomic_private_modes(self):
        h = self.target.handle
        self.store.capture(h, 'synthetic-traffic-token')
        self.assertEqual(self.store.read(h), 'synthetic-traffic-token')
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(list(self.root.iterdir())[0].stat().st_mode & 0o777, 0o600)
        for update in ({'owner_id':'different'}, {'attempt_id':uuid4()}, {'task_id':uuid4()}, {'allocation_id':uuid4()}):
            with self.subTest(update=update), self.assertRaises(SandboxError):
                self.store.read(h.model_copy(update={'scope':h.scope.model_copy(update=update)}))
        self.store.remove(h.sandbox_id, h.scope)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_no_symlink_or_public_directory(self):
        linked = self.root.parent / 'link'; linked.symlink_to(self.root)
        with self.assertRaises(OSError): SandboxViewSecrets(linked)
        self.root.chmod(0o755)
        with self.assertRaises(SandboxError): SandboxViewSecrets(self.root)

    def test_symlink_hardlink_and_expired_record_fail_closed(self):
        h = self.target.handle
        self.store.capture(h, 'synthetic-token')
        file = next(self.root.iterdir()); original = file.read_bytes(); file.unlink()
        outside = self.root.parent / 'outside'; outside.write_bytes(original); outside.chmod(0o600)
        file.symlink_to(outside)
        with self.assertRaises(SandboxError): self.store.read(h)
        file.unlink(); os.link(outside, file)
        with self.assertRaises(SandboxError): self.store.read(h)
        file.unlink(); outside.unlink()
        self.store.capture(h.model_copy(update={'expires_at':datetime.now(UTC)-timedelta(seconds=1)}), 'synthetic-token')
        with self.assertRaises(SandboxError): self.store.read(h)
        self.assertEqual(list(self.root.iterdir()), [])


class Access:
    def __init__(self, selected): self.selected = selected; self.revoked = False
    def check(self, owner):
        if owner != self.selected.owner_id or self.revoked:
            raise JarvisHTTPError(403,'sandbox_view_access_ended','Access ended')
    async def allocation(self, owner, session, allocation, generation=None):
        self.check(owner)
        if session != self.selected.session_id or allocation != self.selected.allocation_id:
            raise JarvisHTTPError(403,'sandbox_view_scope_mismatch','Wrong selection')
        if generation != self.selected.generation:
            raise JarvisHTTPError(409,'sandbox_view_replaced','Replaced')
        return self.selected
    async def close(self): self.revoked = True


class LeaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.t = target(); self.access = Access(self.t)
        self.connection = DesktopConnection('wss://protected.invalid', 'never-a-phone-token', self.t.deadline)
        self.provider = SimpleNamespace(open=AsyncMock(return_value=self.connection), inspect=AsyncMock(return_value=self.connection))
        self.views = SandboxViewService(self.access, self.provider)
    async def open(self):
        return await self.views.open(self.t.owner_id,self.t.session_id,self.t.allocation_id,self.t.generation)

    async def test_lease_is_bounded_and_has_no_provider_credentials(self):
        lease = await self.open()
        self.assertLessEqual(lease.expires_at,self.t.deadline)
        self.assertNotIn('never-a-phone-token',json.dumps(lease.wire()))
        self.assertNotIn('protected.invalid',json.dumps(lease.wire()))
        self.assertNotIn('ticket',lease.wire())
        await self.views.validate(lease)
        self.assertEqual(self.provider.open.await_count,1)
        self.assertEqual(self.provider.inspect.await_count,1)

    async def test_replaced_view_and_logout_revoke_without_stopping_work(self):
        first = await self.open(); second = await self.open()
        self.assertTrue(first.closed.is_set())
        with self.assertRaises(JarvisHTTPError): await self.views.validate(first)
        self.views.revoke(self.t.owner_id,self.t.session_id,second.id)
        with self.assertRaises(JarvisHTTPError): await self.views.validate(second)
        self.assertEqual(self.provider.open.await_count,2)

    async def test_phone_session_change_replaces_owner_view_and_old_stop_is_harmless(self):
        first = await self.open()
        replacement = replace(self.t, session_id=uuid4())
        self.access.selected = replacement
        second = await self.views.open(replacement.owner_id, replacement.session_id,
                                       replacement.allocation_id, replacement.generation)
        self.assertTrue(first.closed.is_set())
        self.views.revoke(self.t.owner_id, self.t.session_id, first.id)
        self.assertEqual(self.views.current(self.t.owner_id)["id"], str(second.id))
        self.assertFalse(second.closed.is_set())

    async def test_legacy_view_cannot_accidentally_resume(self):
        with self.assertRaises(JarvisHTTPError) as error:
            await self.views.open(self.t.owner_id, self.t.session_id, self.t.allocation_id,
                                  self.t.generation, resume=True)
        self.assertEqual(error.exception.code, "sandbox_resume_unavailable")
        self.provider.open.assert_not_awaited()

    async def test_wrong_owner_and_generation_never_reach_provider(self):
        for owner, generation in [('wrong',self.t.generation),(self.t.owner_id,'b'*64)]:
            with self.assertRaises(JarvisHTTPError):
                await self.views.open(owner,self.t.session_id,self.t.allocation_id,generation)
        self.provider.open.assert_not_awaited()

    async def test_authority_rechecked_after_startup(self):
        async def revoked(_): self.access.revoked=True; return self.connection
        self.provider.open.side_effect=revoked
        with self.assertRaises(JarvisHTTPError): await self.open()
        self.assertFalse(self.views.leases)

    async def test_outage_is_not_shutdown(self):
        self.provider.open.side_effect=SandboxError('sandbox_provider_outcome_uncertain')
        with self.assertRaises(JarvisHTTPError) as error: await self.open()
        self.assertEqual(error.exception.code,'sandbox_view_connection_failed')
        self.provider.open.side_effect=SandboxError('sandbox_not_running')
        with self.assertRaises(JarvisHTTPError) as error: await self.open()
        self.assertEqual(error.exception.code,'sandbox_not_running')

    async def test_expiry_rejects_without_provider_renewal(self):
        lease=await self.open(); lease.expires_at=datetime.now(UTC)-timedelta(seconds=1)
        with self.assertRaises(JarvisHTTPError): await self.views.validate(lease)
        self.provider.inspect.assert_not_awaited()
        self.assertTrue(lease.closed.is_set())

    def test_sandbox_cannot_be_saved_as_file(self):
        from pydantic import ValidationError
        fields=dict(request_id=uuid4(),artifact_kind='sandbox',artifact_id=self.t.allocation_id,
                    version_id=self.t.generation,created_at=datetime.now(UTC),expires_at=self.t.deadline,content_ready=True)
        self.assertEqual(PresentationRequest(action='open',**fields).artifact_kind,'sandbox')
        with self.assertRaises(ValidationError): PresentationRequest(action='download',**fields)


class ProviderAttachTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_pinned_sdk_get_constructs_only_local_handle(self):
        from e2b.api.client.models.sandbox_detail import SandboxDetail
        t=target(); h=t.handle
        raw={'templateID':'desktop','sandboxID':h.sandbox_id,'clientID':'fixture',
             'startedAt':datetime.now(UTC).isoformat(),'endAt':h.expires_at.isoformat(),
             'envdVersion':'0.5.0','cpuCount':2,'memoryMB':512,'diskSizeMB':1024,'state':'running',
             'envdAccessToken':'synthetic-envd','domain':'e2b.app','allowInternetAccess':False,
             'metadata':h.scope.metadata(h.policy),'network':{'allowPublicTraffic':False,'allowOut':[], 'denyOut':['0.0.0.0/0']},
             'lifecycle':{'onTimeout':'kill','autoResume':False},'volumeMounts':[]}
        sdk=SimpleNamespace(AsyncSandbox=Mock())
        provider=E2BProvider(E2BConfiguration(api_key='synthetic-api'),sdk=sdk)
        secrets=SimpleNamespace(read=Mock(return_value='synthetic-traffic'),remove=Mock())
        viewer=E2BDesktopViewer(provider,secrets)
        get=AsyncMock(return_value=SimpleNamespace(status_code=200,parsed=SandboxDetail.from_dict(raw)))
        with patch('e2b.api.client.api.sandboxes.get_sandboxes_sandbox_id.asyncio_detailed',get), patch('e2b.api.client_async.get_api_client'):
            _, connection=await viewer._attach(h)
        get.assert_awaited_once(); sdk.AsyncSandbox.assert_called_once()
        sdk.AsyncSandbox.connect.assert_not_called(); sdk.AsyncSandbox.create.assert_not_called()
        self.assertEqual(connection.url,'wss://6087-fixture-desktop.e2b.app/websockify')
        self.assertEqual(connection.token,'synthetic-traffic')
        self.assertNotIn('synthetic-traffic',repr(connection))
        raw['state']='paused'; get.return_value.parsed=SandboxDetail.from_dict(raw)
        sdk.AsyncSandbox.reset_mock()
        with patch('e2b.api.client.api.sandboxes.get_sandboxes_sandbox_id.asyncio_detailed',get), patch('e2b.api.client_async.get_api_client'):
            with self.assertRaises(SandboxError) as error: await viewer._attach(h)
        self.assertEqual(error.exception.code,'sandbox_not_running'); sdk.AsyncSandbox.assert_not_called()


class StreamBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.target = target()
        self.provider = SimpleNamespace(
            open=AsyncMock(return_value=DesktopConnection('wss://protected.invalid', 'private-provider-token', self.target.deadline)),
            inspect=AsyncMock(return_value=DesktopConnection('wss://protected.invalid', 'private-provider-token', self.target.deadline)))
        self.views = SandboxViewService(Access(self.target), self.provider)
        self.lease = await self.views.open(self.target.owner_id, self.target.session_id,
                                         self.target.allocation_id, self.target.generation)
        self.socket = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(runtime_services=SimpleNamespace(sandbox_views=self.views))),
            scope={'subprotocols': ['jarvis-view-v1', self.lease.ticket]},
            accept=AsyncMock(), close=AsyncMock())

    async def test_tls_verifies_trusted_roots_and_private_credentials_stay_upstream(self):
        import ssl
        from app.sandbox.view_router import stream
        @asynccontextmanager
        async def rejected_connection(*args, **kwargs):
            self.assertEqual(kwargs['ssl'].verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(kwargs['ssl'].check_hostname)
            self.assertGreater(kwargs['ssl'].cert_store_stats()['x509_ca'], 0)
            self.assertTrue(kwargs['logger'].disabled)
            self.assertEqual(kwargs['additional_headers'], {'e2b-traffic-access-token': 'private-provider-token'})
            self.assertIsNone(kwargs['proxy'])
            raise ssl.SSLCertVerificationError('untrusted upstream')
            yield
        with patch('app.sandbox.view_router.NoRedirectConnect', rejected_connection):
            await stream(self.socket, self.lease.id)
        self.socket.accept.assert_awaited_once_with(subprotocol='jarvis-view-v1')
        self.socket.close.assert_awaited_once_with(code=1011)
        self.assertFalse(self.lease.connected)

    async def test_failed_accept_releases_reservation_but_intruder_does_not(self):
        from app.sandbox.view_router import stream
        self.socket.accept.side_effect = RuntimeError('client disconnected')
        await stream(self.socket, self.lease.id)
        self.assertFalse(self.lease.connected)
        self.lease.connected = True
        self.socket.scope['subprotocols'][1] = 'wrong-ticket'
        self.socket.accept.reset_mock()
        await stream(self.socket, self.lease.id)
        self.socket.accept.assert_not_awaited()
        self.assertTrue(self.lease.connected)
