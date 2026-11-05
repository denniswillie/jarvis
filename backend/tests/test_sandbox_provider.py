"""Maintained SDK ownership/network/observation boundaries without cloud effects."""
import asyncio
import base64
import json
import hashlib
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from pydantic import ValidationError

from app.runtime.http_privacy import PRIVATE_HTTP_ACTIVE, private_http_logs
from app.sandbox.e2b_provider import E2BProvider, SandboxCommand, WORKSPACE, _Output, _privileged_command
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxPolicy, SandboxScope
from evals.second_brain.e2b_probe import Probe


class ExitError(Exception):
    def __init__(self, code):
        self.exit_code = code


class NativeCommand:
    pid = 31
    def __init__(self, result=None, error=None, delay=0):
        self.result = result or SimpleNamespace(exit_code=0)
        self.error, self.delay = error, delay
        self.disconnect = AsyncMock()

    async def wait(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


class SandboxPolicyTests(unittest.TestCase):
    def test_network_selectors_are_authority_not_urls_or_wildcards(self):
        self.assertEqual(SandboxPolicy(allowed_domains=("example.com",)).allowed_domains, ("example.com",))
        for domain in ("*", "*.example.com", "https://example.com", "example.com/path", "user@example.com",
                       "127.0.0.1", "0.0.0.0/0", "EXAMPLE.com", "example.com.", "x..com", "-x.com", "x_.com"):
            with self.subTest(domain=domain), self.assertRaises(ValidationError):
                SandboxPolicy(allowed_domains=(domain,))
        for values in ({"lifetime_seconds":0}, {"lifetime_seconds":3601}, {"template":"personal-secret-template"},
                       {"output_bytes":10000000}, {"allowed_domains":("example.com","example.com")}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                SandboxPolicy(**values)

    def test_offline_policy_keeps_legacy_serialization_and_cleanup_identity(self):
        legacy = {"template": "base", "lifetime_seconds": 300, "command_seconds": 60,
                  "output_bytes": 64000, "allowed_domains": []}
        policy = SandboxPolicy.model_validate(legacy)
        expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(policy.model_dump(mode="json"), legacy)
        self.assertEqual(policy.fingerprint, expected)
        self.assertFalse(policy.internet_access)
        self.assertEqual(policy.required_capabilities, ("sandbox.execute",))
        network = SandboxPolicy(internet_access=True)
        self.assertNotEqual(network.fingerprint, expected)
        self.assertEqual(network.required_capabilities, ("sandbox.execute", "sandbox.network"))

    def test_network_scope_is_explicit_and_bounded_by_worker_policy(self):
        offline = SandboxPolicy()
        scoped = SandboxPolicy(allowed_domains=("pypi.org",))
        network = SandboxPolicy(internet_access=True)
        self.assertTrue(offline.within(network))
        self.assertTrue(scoped.within(network))
        self.assertFalse(network.within(offline))
        self.assertFalse(scoped.within(offline))
        self.assertFalse(network.within(scoped))
        self.assertFalse(SandboxPolicy(lifetime_seconds=301).within(network))
        for kwargs in ({"internet_access": "true"}, {"internet_access": 1},
                       {"internet_access": True, "allowed_domains": ("pypi.org",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                SandboxPolicy(**kwargs)

    def test_credentials_do_not_enter_serializable_configuration(self):
        config = E2BConfiguration(api_key="synthetic-private-key")
        self.assertNotIn("synthetic-private-key", repr(config))
        self.assertNotIn("api_key", config.model_dump())
        for value in ("", "placeholder", " padded "):
            with self.assertRaises(ValidationError):
                E2BConfiguration(api_key=value)


class SandboxProviderTests(unittest.IsolatedAsyncioTestCase):
    def root_script(self, call):
        words = shlex.split(call.args[0])
        self.assertEqual(words[:-1], ["exec", "/usr/bin/env", "-i",
            "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "HOME=/root", "USER=root",
            "LOGNAME=root", "LC_ALL=C", "/bin/sh", "-eu", "-c"])
        self.assertEqual(call.kwargs["user"], "root")
        self.assertEqual(call.kwargs["cwd"], "/")
        self.assertNotIn("/usr/local", words[3])
        return words[-1]

    async def test_hermes_probe_still_cleans_native_resource_after_sql_cleanup_failure(self):
        from evals.second_brain.hermes_sandbox_probe import cleanup_fixture
        probe = SimpleNamespace(scopes=[object()], cleanup=AsyncMock())
        fixture = SimpleNamespace(source=SimpleNamespace(pool=SimpleNamespace(
            acquire=Mock(side_effect=RuntimeError("synthetic database outage")))))
        with self.assertRaises(RuntimeError):
            await cleanup_fixture(fixture, probe)
        probe.cleanup.assert_awaited_once()

    async def test_managed_command_dispatch_runs_only_fixed_protected_supervisor(self):
        from app.sandbox.models import CommandRequest
        from app.sandbox.remote_runner import RUNNER_PATH
        handle = await self.handle()
        identity = uuid4()
        request = CommandRequest(operation_key="one", command="printf synthetic")
        pid = await self.provider.start_managed(handle, request, operation_id=identity, request_digest="a"*64)
        command = self.commands.run.call_args.args[0]
        options = self.commands.run.call_args.kwargs
        self.assertEqual(pid, 31)
        script = self.root_script(self.commands.run.call_args)
        self.assertTrue(script.startswith("exec /usr/bin/python3 -I -S "+RUNNER_PATH+" "))
        payload = json.loads(base64.b64decode(shlex.split(script)[-1]))
        self.assertEqual(payload["command"], request.command)
        self.assertEqual(payload["operation_id"], str(identity))
        self.assertEqual(options["user"], "root")
        self.assertNotIn("synthetic-key", command)
        self.commands.run.return_value.disconnect.assert_awaited_once()

    async def test_managed_receipt_recovers_terminal_result_without_process_reconnect(self):
        from app.sandbox.models import CommandReceipt
        handle = await self.handle()
        identity = uuid4()
        receipt = CommandReceipt(format="jarvis.sandbox.command.v1", operation_id=identity,
            allocation_id=self.scope.allocation_id, request_digest="a"*64, pid=31, phase="completed",
            started=True, exit_code=0, stdout="first\\nlast\\n", stderr="", timed_out=False,
            output_limited=False, runner_error=None)
        self.commands.run.return_value = SimpleNamespace(
            stdout=base64.b64encode(receipt.model_dump_json().encode()).decode())
        self.assertEqual(await self.provider.command_receipt(handle, identity, "a"*64), receipt)
        self.commands.connect.assert_not_awaited()
        self.assertEqual(self.commands.run.call_args.kwargs["user"], "root")
        self.assertTrue(self.root_script(self.commands.run.call_args).startswith("exec /usr/bin/python3 -I -S -c "))
        for changed in ({"operation_id":uuid4()}, {"allocation_id":uuid4()}, {"request_digest":"b"*64},
                        {"stdout":"x"*(self.policy.output_bytes+1)}):
            self.commands.run.return_value.stdout = base64.b64encode(
                receipt.model_copy(update=changed).model_dump_json().encode()).decode()
            with self.subTest(changed=tuple(changed)), self.assertRaises(SandboxError) as caught:
                await self.provider.command_receipt(handle, identity, "a"*64)
            self.assertEqual(caught.exception.code, "sandbox_command_receipt_invalid")

    async def test_absent_managed_receipt_is_pending_not_permission_to_execute(self):
        self.commands.run.return_value = SimpleNamespace(stdout="pending")
        self.assertIsNone(await self.provider.command_receipt(await self.handle(), uuid4(), "a"*64))
        self.assertEqual(self.commands.run.await_count, 1)
        self.commands.connect.assert_not_awaited()

    def test_privileged_wrapper_quotes_one_script_without_ambient_expansion(self):
        script = "printf '%s' '$PATH'; printf '%s' \"a'b\"\n# unexpanded body"
        words = shlex.split(_privileged_command(script))
        self.assertEqual(words[-1], script)
        self.assertEqual(words[-4:-1], ["/bin/sh", "-eu", "-c"])
        self.assertNotIn("/usr/local", words[3])

    async def test_repeated_prepare_after_user_command_keeps_protected_root_launch(self):
        handle = await self.handle()
        await self.provider.prepare(handle)
        first = self.commands.run.call_args
        await self.provider.start(handle, "printf synthetic", operation_id=uuid4())
        user = self.commands.run.call_args
        await self.provider.prepare(handle)
        second = self.commands.run.call_args
        first_script = self.root_script(first)
        self.assertEqual(self.root_script(second), first_script)
        for expected in ("/usr/bin/install -d", "/usr/sbin/useradd --no-create-home",
                         "$(/usr/bin/cat /var/lib/jarvis/ready)", "/usr/bin/chmod 0600",
                         "exec /usr/bin/python3 -I -S -c "):
            self.assertIn(expected, first_script)
        self.assertIn("if test -f /var/lib/jarvis/ready", first_script)
        self.assertEqual(user.kwargs["user"], "jarvis")
        self.assertEqual(user.kwargs["cwd"], WORKSPACE)
        self.assertIn("PATH=/usr/local/bin:/usr/bin:/bin", user.args[0])
        self.api.create.assert_not_awaited()

    async def test_reused_vm_root_submission_and_polling_never_use_user_path(self):
        from app.sandbox.models import CommandRequest
        handle = await self.handle()
        await self.provider.start(handle, "printf synthetic", operation_id=uuid4())
        self.commands.run.reset_mock()
        for name in ("first", "second"):
            identity = uuid4()
            self.commands.run.return_value = NativeCommand()
            await self.provider.start_managed(handle, CommandRequest(operation_key=name, command="printf synthetic"),
                operation_id=identity, request_digest="a"*64)
            self.assertTrue(self.root_script(self.commands.run.call_args).startswith("exec /usr/bin/python3 -I -S "))
            self.commands.run.return_value = SimpleNamespace(stdout="pending", exit_code=0)
            self.assertIsNone(await self.provider.command_receipt(handle, identity, "a"*64))
            self.assertTrue(self.root_script(self.commands.run.call_args).startswith("exec /usr/bin/python3 -I -S -c "))
        self.assertEqual(self.commands.run.await_count, 4)
        self.api.create.assert_not_awaited()

    async def test_missing_root_system_command_fails_without_fallback_or_retry(self):
        self.commands.run.side_effect = ExitError(127)
        with self.assertRaises(SandboxError) as caught:
            await self.provider.prepare(await self.handle())
        self.assertEqual(caught.exception.code, "sandbox_privileged_command_failed")
        self.assertEqual(caught.exception.diagnostic["outcome"], "completed_nonzero")
        self.assertEqual(caught.exception.diagnostic["exit_code"], 127)
        self.commands.run.assert_awaited_once()
        self.root_script(self.commands.run.call_args)

    async def test_root_poll_failure_is_not_reported_as_pending(self):
        self.commands.run.side_effect = ExitError(127)
        with self.assertRaises(SandboxError):
            await self.provider.command_receipt(await self.handle(), uuid4(), "a"*64)
        self.commands.run.assert_awaited_once()

    async def test_returned_nonzero_root_exit_does_not_confirm_preparation(self):
        self.commands.run.return_value = SimpleNamespace(exit_code=127, stdout="")
        with self.assertRaises(SandboxError) as caught:
            await self.provider.prepare(await self.handle())
        self.assertEqual(caught.exception.code, "sandbox_privileged_command_failed")
        self.commands.run.assert_awaited_once()

    async def test_desktop_setup_uses_the_same_root_launch_policy(self):
        from app.sandbox.desktop_environment import PREPARE_DESKTOP
        self.policy = SandboxPolicy(template="desktop")
        self.raw.metadata = self.scope.metadata(self.policy)
        self.commands.run.return_value = SimpleNamespace(exit_code=0, stdout="jarvis.desktop.ready.v1\n")
        await self.provider.prepare(await self.handle())
        self.assertEqual(self.commands.run.await_count, 2)
        self.root_script(self.commands.run.call_args_list[0])
        self.assertEqual(self.root_script(self.commands.run.call_args_list[1]), PREPARE_DESKTOP)

    async def test_failed_base_preparation_never_attempts_desktop_setup(self):
        self.policy = SandboxPolicy(template="desktop")
        self.raw.metadata = self.scope.metadata(self.policy)
        self.commands.run.side_effect = ExitError(127)
        with self.assertRaises(SandboxError):
            await self.provider.prepare(await self.handle())
        self.commands.run.assert_awaited_once()

    async def test_privileged_input_and_output_helpers_share_python_isolation(self):
        from app.sandbox.file_transfer import INPUT_ROOT
        handle, content = await self.handle(), b"synthetic"
        identity = hashlib.sha256(content).hexdigest()
        self.box.files = SimpleNamespace(write=AsyncMock())
        receipt = {"path": INPUT_ROOT+"/"+identity, "sha256": identity, "size_bytes": len(content)}
        self.commands.run.return_value = SimpleNamespace(exit_code=0, stdout=json.dumps(receipt))
        self.assertEqual(await self.provider.stage_input(handle, identity, content), receipt)
        self.assertTrue(self.root_script(self.commands.run.call_args).startswith("exec /usr/bin/python3 -I -S -c "))
        output = {"content_base64": base64.b64encode(content).decode(), "sha256": identity, "size_bytes": len(content)}
        self.commands.run.return_value = SimpleNamespace(exit_code=0, stdout=json.dumps(output))
        self.assertEqual(await self.provider.read_workspace_file(handle, "result.txt"), content)
        self.assertTrue(self.root_script(self.commands.run.call_args).startswith("exec /usr/bin/python3 -I -S -c "))

    def setUp(self):
        self.scope = SandboxScope(owner_id="synthetic-owner", task_id=uuid4(), task_revision=1,
                                  attempt_id=uuid4(), allocation_id=uuid4())
        self.policy = SandboxPolicy()
        self.raw = SimpleNamespace(sandbox_id="synthetic-sandbox", template_id="base-id", state="running",
            metadata=self.scope.metadata(self.policy), end_at=datetime.now(UTC)+timedelta(seconds=300),
            allow_internet_access=False, network={"allow_out":[],"deny_out":["0.0.0.0/0"],
                "allow_public_traffic":False}, lifecycle={"on_timeout":"kill","auto_resume":False}, volume_mounts=[])
        self.commands = SimpleNamespace(run=AsyncMock(return_value=NativeCommand()),
            connect=AsyncMock(return_value=NativeCommand()), list=AsyncMock(return_value=[]), kill=AsyncMock(return_value=True))
        self.box = SimpleNamespace(sandbox_id=self.raw.sandbox_id, commands=self.commands)
        self.api = SimpleNamespace(create=AsyncMock(return_value=self.box), get_info=AsyncMock(return_value=self.raw),
            connect=AsyncMock(return_value=self.box), pause=AsyncMock(), kill=AsyncMock(return_value=True))
        self.sdk = SimpleNamespace(AsyncSandbox=self.api, CommandExitException=ExitError,
                                  SandboxState=SimpleNamespace(RUNNING="running", PAUSED="paused"), SandboxQuery=SimpleNamespace)
        self.provider = E2BProvider(E2BConfiguration(api_key="synthetic-key"), sdk=self.sdk)

    async def handle(self):
        return await self.provider.info(self.raw.sandbox_id, self.scope, self.policy)

    async def test_creation_uses_exact_scope_no_keys_in_vm_and_no_retry(self):
        await self.provider.create(self.scope, self.policy)
        values = self.api.create.call_args.kwargs
        self.assertTrue(values["secure"])
        self.assertFalse(values["allow_internet_access"])
        self.assertFalse(values["network"]["allow_public_traffic"])
        self.assertEqual(values["envs"], {})
        self.assertEqual(values["retries"], 0)
        self.assertEqual(values["lifecycle"], {"on_timeout":"kill", "auto_resume":False})
        self.assertNotIn(self.scope.owner_id, str(values["metadata"]))
        self.assertNotIn("synthetic-key", str(values["metadata"]))

    async def test_network_creation_matches_recorded_policy_without_public_inbound(self):
        self.policy = SandboxPolicy(internet_access=True)
        self.raw.metadata = self.scope.metadata(self.policy)
        self.raw.allow_internet_access = True
        self.raw.network["deny_out"] = []
        handle = await self.provider.create(self.scope, self.policy)
        options = self.api.create.call_args.kwargs
        self.assertTrue(options["allow_internet_access"])
        self.assertEqual(options["network"], {"allow_out": [], "deny_out": [], "allow_public_traffic": False})
        self.assertEqual(options["envs"], {})
        self.raw.allow_internet_access = False
        with self.assertRaises(SandboxError) as caught:
            await self.provider.start(handle, "pwd", operation_id=uuid4())
        self.assertEqual(caught.exception.code, "sandbox_policy_mismatch")
        self.commands.run.assert_not_awaited()
        await self.provider.kill(handle.sandbox_id, self.scope, self.policy)
        self.api.kill.assert_awaited_once()

    async def test_policy_mismatch_blocks_execution_but_not_owned_cleanup(self):
        handle = await self.handle()
        self.raw.network["allow_public_traffic"] = True
        with self.assertRaises(SandboxError) as caught:
            await self.provider.start(handle, "pwd", operation_id=uuid4())
        self.assertEqual(caught.exception.code, "sandbox_policy_mismatch")
        self.commands.run.assert_not_awaited()
        await self.provider.kill(handle.sandbox_id, self.scope, self.policy)
        self.api.kill.assert_awaited_once()

    async def test_unrequested_network_transforms_proxy_and_mounts_are_rejected(self):
        for key,value in (("rules",{"example.com":[{"transform":{}}]}),
                          ("egress_proxy",{"address":"proxy.example.com:1080"}),("https_ports",[3000])):
            self.raw.network[key] = value
            with self.subTest(key=key),self.assertRaises(SandboxError):
                await self.handle()
            del self.raw.network[key]
        self.raw.volume_mounts = [{"name":"another-owner-volume","path":"/workspace"}]
        with self.assertRaises(SandboxError):
            await self.handle()

    async def test_scope_mismatch_neither_connects_nor_kills_foreign_resource(self):
        handle = await self.handle()
        self.raw.metadata["jarvis_attempt"] = str(uuid4())
        for method in (lambda:self.provider.start(handle,"pwd",operation_id=uuid4()),
                       lambda:self.provider.kill(handle.sandbox_id,self.scope,self.policy)):
            with self.assertRaises(SandboxError) as caught:
                await method()
            self.assertEqual(caught.exception.code,"sandbox_scope_mismatch")
        self.api.connect.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_wrong_native_resource_and_process_identifiers_are_not_accepted(self):
        with self.assertRaises(SandboxError) as caught:
            await self.provider.info("another-sandbox",self.scope,self.policy)
        self.assertEqual(caught.exception.code,"sandbox_identity_mismatch")
        with self.assertRaises(SandboxError):
            await self.provider.kill("another-sandbox",self.scope,self.policy)
        self.api.kill.assert_not_awaited()
        with self.assertRaises(SandboxError) as caught:
            await self.provider.reconnect_command(await self.handle(),999)
        self.assertEqual(caught.exception.code,"sandbox_command_identity_mismatch")

    async def test_paused_observation_does_not_silently_resume(self):
        handle = await self.handle()
        self.raw.state = "paused"
        with self.assertRaises(SandboxError) as caught:
            await self.provider.processes(handle)
        self.assertEqual(caught.exception.code,"sandbox_paused")
        self.api.connect.assert_not_awaited()

    async def test_commands_are_unprivileged_and_clear_ambient_environment(self):
        handle = await self.handle()
        operation = uuid4()
        command = await self.provider.start(handle, "printf '%s' 'synthetic output'", operation_id=operation)
        self.assertEqual(command.pid,31)
        args,values = self.commands.run.call_args
        self.assertEqual(values["user"],"jarvis")
        self.assertEqual(values["cwd"],WORKSPACE)
        self.assertTrue(values["background"])
        self.assertTrue(args[0].startswith("env -i "))
        self.assertIn(str(operation),args[0])
        self.assertNotIn("synthetic-key",args[0])
        self.assertNotIn("envs",values)
        self.assertEqual(shlex.split(args[0]), [
            "env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/home/jarvis",
            "USER=jarvis", "LOGNAME=jarvis", "XDG_CACHE_HOME=" + WORKSPACE + "/.cache",
            "JARVIS_SANDBOX_OPERATION=" + str(operation),
            "/bin/bash", "--noprofile", "--norc", "-c", "printf '%s' 'synthetic output'",
        ])
        for cwd in ("/etc","../another", "nested/../../elsewhere"):
            with self.assertRaises(ValueError):
                await self.provider.start(handle,"pwd",operation_id=uuid4(),cwd=cwd)

    async def test_desktop_command_uses_prepared_home_and_display_without_elevation(self):
        from app.sandbox.desktop_environment import desktop_environment
        self.policy = SandboxPolicy(template="desktop")
        self.raw.metadata = self.scope.metadata(self.policy)
        operation = uuid4()
        source = "printf '%s' \"$HOME; literal 'text'\""
        command = await self.provider.start(await self.handle(), source, operation_id=operation, cwd="project")
        words = shlex.split(self.commands.run.call_args.args[0])
        shell_index = words.index("/bin/bash")
        self.assertEqual(words[:2], ["env", "-i"])
        self.assertEqual(dict(word.split("=", 1) for word in words[2:shell_index]), {
            "PATH": "/usr/local/bin:/usr/bin:/bin", "JARVIS_SANDBOX_OPERATION": str(operation),
            **desktop_environment(),
        })
        self.assertEqual(words[shell_index:], ["/bin/bash", "--noprofile", "--norc", "-c", source])
        options = self.commands.run.call_args.kwargs
        self.assertEqual(options["user"], "jarvis")
        self.assertEqual(options["cwd"], WORKSPACE + "/project")
        self.assertTrue(options["background"])
        self.assertEqual(options["timeout"], self.policy.command_seconds)
        self.assertEqual(options["request_timeout"], self.provider.configuration.request_timeout_seconds)
        self.assertNotIn("envs", options)
        self.assertNotIn("synthetic-key", self.commands.run.call_args.args[0])
        options["on_stdout"]("desktop stdout")
        options["on_stderr"]("desktop stderr")
        result = await command.wait(timeout_seconds=1)
        self.assertEqual(result.stdout, "desktop stdout")
        self.assertEqual(result.stderr, "desktop stderr")
        self.commands.run.assert_awaited_once()
        self.api.create.assert_not_awaited()

    async def test_desktop_launcher_clears_ambient_values_in_a_real_local_shell(self):
        self.policy = SandboxPolicy(template="desktop")
        self.raw.metadata = self.scope.metadata(self.policy)
        source = (
            "printf '%s\\n' \"$DISPLAY\" \"$HOME\" \"$XDG_RUNTIME_DIR\" \"$XDG_CONFIG_HOME\" "
            "\"$XDG_CACHE_HOME\" \"$USER\" \"$LOGNAME\" \"${JARVIS_TEST_SECRET-unset}\" "
            "\"${PYTHONPATH-unset}\" \"${BASH_ENV-unset}\""
        )
        await self.provider.start(await self.handle(), source, operation_id=uuid4())
        words = shlex.split(self.commands.run.call_args.args[0])
        # Execute only the emitted environment/printf locally: no E2B or GUI call.
        completed = subprocess.run(["/usr/bin/env", *words[1:]], capture_output=True, text=True,
            check=True, timeout=3, env={
                "DISPLAY": ":99", "HOME": "/root", "USER": "root", "LOGNAME": "root",
                "XDG_RUNTIME_DIR": "/untrusted", "XDG_CONFIG_HOME": "/untrusted",
                "XDG_CACHE_HOME": "/untrusted", "JARVIS_TEST_SECRET": "must-not-leak",
                "PYTHONPATH": "/untrusted", "BASH_ENV": "/not-a-profile",
            })
        self.assertEqual(completed.stdout.splitlines(), [
            ":0", WORKSPACE, WORKSPACE + "/.runtime", WORKSPACE + "/.config", WORKSPACE + "/.cache",
            "jarvis", "jarvis", "unset", "unset", "unset",
        ])

    async def test_network_control_plane_errors_are_not_absence_and_not_retried(self):
        self.api.create.side_effect = RuntimeError("possibly accepted; synthetic-secret")
        with self.assertRaises(SandboxError) as caught:
            await self.provider.create(self.scope,self.policy)
        self.assertEqual(caught.exception.code,"sandbox_provider_outcome_uncertain")
        self.assertNotIn("synthetic-secret",str(caught.exception))
        self.api.create.assert_awaited_once()

    async def test_native_missing_sandbox_is_absence_not_transport_uncertainty(self):
        class MissingSandbox(Exception):
            pass
        self.sdk.SandboxNotFoundException = MissingSandbox
        self.api.get_info.side_effect = MissingSandbox("synthetic native absence")
        with self.assertRaises(SandboxError) as caught:
            await self.provider.info("removed-sandbox",self.scope,self.policy)
        self.assertEqual(caught.exception.code,"sandbox_not_found")
        self.api.get_info.assert_awaited_once()

    async def test_same_pid_reconnect_does_not_start_another_command(self):
        command = await self.provider.reconnect_command(await self.handle(),31)
        self.assertEqual(command.pid,31)
        self.commands.connect.assert_awaited_once()
        self.commands.run.assert_not_awaited()

    async def test_discovery_uses_typed_query_and_all_native_pages(self):
        class Page:
            has_next = True
            async def next_items(page):
                page.has_next = False
                return [self.raw]
        seen = []
        def listing(**kwargs):
            seen.append(kwargs)
            return Page()
        self.api.list = listing
        ids = await self.provider.discover(self.scope,self.policy)
        self.assertEqual(ids,[self.raw.sandbox_id])
        self.assertEqual(seen[0]["query"].metadata,self.scope.metadata(self.policy))
        self.assertEqual(set(seen[0]["query"].state),{"running","paused"})

    async def test_probe_still_kills_recorded_resource_when_discovery_fails(self):
        self.provider.discover = AsyncMock(side_effect=SandboxError("synthetic_discovery_failure"))
        with tempfile.TemporaryDirectory() as directory:
            probe = Probe(self.provider,Path(directory))
            probe.scopes = [(self.scope,self.policy)]
            probe.record["resources"] = [{"scope":self.scope.model_dump(mode="json"),
                                         "sandbox_id":self.raw.sandbox_id}]
            with self.assertRaises(RuntimeError):
                await probe.cleanup()
            self.api.kill.assert_awaited_once()
            self.assertEqual(probe.record["status"],"cleanup_unresolved")

    async def test_known_exit_error_is_not_an_unknown_outcome(self):
        output = _Output(1024)
        output.capture("stderr","expected diagnostic")
        native = NativeCommand(error=ExitError(7))
        result = await SandboxCommand(native,output,exit_error=ExitError).wait(timeout_seconds=1)
        self.assertEqual(result.exit_code,7)
        self.assertEqual(result.stderr,"expected diagnostic")
        native.disconnect.assert_awaited_once()

    async def test_probe_reuses_persisted_discovery_identity_after_listing_changes(self):
        self.provider.discover = AsyncMock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            probe = Probe(self.provider,Path(directory))
            probe.scopes = [(self.scope,self.policy)]
            probe.record["resources"] = [{"scope":self.scope.model_dump(mode="json"),
                "cleanup_observed_ids":[self.raw.sandbox_id]}]
            await probe.cleanup()
            self.api.kill.assert_awaited_once()
            self.assertEqual(probe.record["status"],"cleaned")

    async def test_probe_empty_unacknowledged_create_is_not_confirmed_absent(self):
        self.provider.discover = AsyncMock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            probe = Probe(self.provider,Path(directory))
            probe.scopes = [(self.scope,self.policy)]
            probe.record["resources"] = [{"scope":self.scope.model_dump(mode="json"),"phase":"planned"}]
            with self.assertRaises(RuntimeError):
                await probe.cleanup()
            self.assertEqual(probe.record["status"],"cleanup_unresolved")
            self.api.kill.assert_not_awaited()

    async def test_wait_timeout_retains_pid_and_does_not_claim_cancellation(self):
        native = NativeCommand(delay=1)
        command = SandboxCommand(native,_Output(1024),exit_error=ExitError)
        with self.assertRaises(SandboxError) as caught:
            await command.wait(timeout_seconds=.01)
        self.assertEqual(caught.exception.code,"sandbox_command_observation_lost")
        self.assertEqual(command.pid,31)
        native.disconnect.assert_awaited_once()
        self.api.kill.assert_not_awaited()

    async def test_output_budget_is_combined_and_bounded_before_retention(self):
        output = _Output(1024)
        output.capture("stdout","x"*1000)
        with self.assertRaises(Exception):
            output.capture("stderr","y"*100)
        self.assertEqual(output.size,1000)
        self.assertEqual(output.stderr,[])
        with self.assertRaises(SandboxError) as caught:
            await SandboxCommand(NativeCommand(),output,exit_error=ExitError).wait(timeout_seconds=1)
        self.assertEqual(caught.exception.code,"sandbox_output_limit")

    async def test_transport_privacy_is_task_local_and_resets(self):
        entered,release = asyncio.Event(),asyncio.Event()
        async def private():
            with private_http_logs():
                self.assertTrue(PRIVATE_HTTP_ACTIVE.get())
                entered.set()
                await release.wait()
        running = asyncio.create_task(private())
        await entered.wait()
        self.assertFalse(PRIVATE_HTTP_ACTIVE.get())
        release.set()
        await running
        self.assertFalse(PRIVATE_HTTP_ACTIVE.get())
