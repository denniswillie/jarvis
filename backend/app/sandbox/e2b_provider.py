"""Thin maintained-SDK boundary. Durable grants, claims and recovery belong to Jarvis.

No API key enters the VM. Commands run as an unprivileged account in a disposable
microVM; a working directory is not a shell/chroot restriction. Output observation
loss is not command completion, cancellation or permission to run it again.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from importlib import import_module
from importlib.metadata import version
from math import ceil
from pathlib import PurePosixPath
import shlex
from urllib.parse import urlsplit

import httpx

from app.files.local_file import CHUNK_BYTES, LocalFileError, VerifiedLocalFile, _snapshot_chunks
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY
from app.runtime.http_privacy import private_http_logs
from app.sandbox.models import CommandReceipt, CommandRequest, CommandResult, E2BConfiguration, SandboxError, SandboxInfo, SandboxScope
from app.sandbox.remote_runner import RUNNER_PATH, SCRIPT
from app.sandbox.file_transfer import (
    FREEZE_OUTPUT, INPUT_ROOT, INSTALL_INPUT, INSTALL_INPUT_FILE, MAX_TRANSFER_BYTES, OUTPUT_ROOT,
    READ_OUTPUT, REMOVE_OUTPUT, workspace_file,
)

SDK_VERSION = "2.49.1"
WORKSPACE = "/home/jarvis/workspace"
ACCOUNT = "jarvis"
ROOT_PYTHON = "/usr/bin/python3 -I -S"
_DIAGNOSTIC_TYPES = frozenset({"CommandExitException", "CommandResult", "TimeoutError",
    "TimeoutException", "SandboxException", "SandboxNotFoundException", "NotFoundException",
    "InvalidArgumentException", "NotEnoughSpaceException", "FileNotFoundException", "AuthenticationException",
    "TemplateException", "RateLimitException", "ServiceBusyException", "ConnectError", "ReadError",
    "RemoteProtocolError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "OSError", "RuntimeError", "ValueError", "TypeError", "unrecognized"})


def _failure(code, *, stage, outcome, error_type, exit_code=None, detail=None):
    """Private bounded code evidence, never remote output or exception text."""
    error = SandboxError(code)
    kind = error_type if error_type in _DIAGNOSTIC_TYPES else "unrecognized"
    error.diagnostic = {"stage": stage if stage in {"provider_request", "privileged_command", "media_input_decode"} else "unrecognized",
        "outcome": outcome if outcome in {"not_found", "unknown", "completed_nonzero"} else "unknown", "error_type": kind}
    if type(exit_code) is int and -255 <= exit_code <= 255:
        error.diagnostic["exit_code"] = exit_code
    if detail is not None:
        error.diagnostic["inspection"] = detail
    return error


def _privileged_command(script):
    """Fixed root launch policy, separate from the model command environment.

    The image must provide protected system binaries, their loaders/libraries and
    Python's stdlib. Absolute paths and isolation do not attest their ownership.
    No PATH fallback is permitted if that image contract is unavailable. -S is
    intentional: isolated Python alone still processes system site packages.
    """
    return ("exec /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin "
            "HOME=/root USER=root LOGNAME=root LC_ALL=C "
            "/bin/sh -eu -c " + shlex.quote(script))


class _OutputBoundExceeded(Exception):
    pass


class _Output:
    def __init__(self, limit):
        self.limit, self.size = limit, 0
        self.stdout, self.stderr = [], []
        self.exceeded = False

    def capture(self, channel, text):
        length = len(text.encode("utf-8"))
        if self.size + length > self.limit:
            self.exceeded = True
            raise _OutputBoundExceeded()
        self.size += length
        getattr(self, channel).append(text)


class SandboxCommand:
    """The native PID survives observation loss. This wrapper never resubmits."""
    def __init__(self, native, output, *, exit_error):
        self._native, self._output, self._exit_error = native, output, exit_error
        self.pid = native.pid

    async def wait(self, *, timeout_seconds):
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 302:
            raise ValueError("Invalid command observation timeout")
        try:
            with private_http_logs():
                async with asyncio.timeout(timeout_seconds):
                    try:
                        result = await self._native.wait()
                        code = result.exit_code
                    except self._exit_error as error:
                        code = error.exit_code
            if self._output.exceeded:
                raise SandboxError("sandbox_output_limit")
            return CommandResult(pid=self.pid, exit_code=code,
                stdout="".join(self._output.stdout), stderr="".join(self._output.stderr))
        except asyncio.CancelledError:
            # The caller retains the sandbox and PID for explicit stop/reconcile.
            raise
        except SandboxError:
            raise
        except Exception:
            raise SandboxError("sandbox_output_limit" if self._output.exceeded else "sandbox_command_observation_lost") from None
        finally:
            await self.disconnect()

    async def disconnect(self):
        with private_http_logs():
            await self._native.disconnect()


class E2BProvider:
    # Subclasses may change resource ownership and timeout behavior without
    # copying SDK operations. Legacy task scope and kill-on-timeout stay default.
    _scope_type = SandboxScope
    _handle_type = SandboxInfo
    _on_timeout = "kill"

    def __init__(self, configuration: E2BConfiguration, *, sdk=None, view_secrets=None):
        self.configuration = configuration
        self.view_secrets = view_secrets
        # An unselected voice/API process need not install or import this SDK.
        if sdk is None:
            try:
                if version("e2b") != SDK_VERSION:
                    raise SandboxError("sandbox_sdk_version_mismatch")
                sdk = import_module("e2b")
            except ImportError:
                raise SandboxError("sandbox_sdk_unavailable") from None
        self.sdk = sdk

    def _options(self):
        return {"api_key": self.configuration.api_key.get_secret_value(), "retries": 0,
                "request_timeout": self.configuration.request_timeout_seconds, "debug": False}

    async def stage_input(self, handle, identity, content):
        from uuid import uuid4
        if (not isinstance(identity,str) or len(identity)!=64 or any(c not in "0123456789abcdef" for c in identity)
                or type(content) is not bytes or len(content)>MAX_TRANSFER_BYTES):
            raise ValueError("A stable input identity and bounded original bytes are required")
        box=await self._connect(handle)
        temporary="/var/lib/jarvis/incoming/"+uuid4().hex
        expected={"path":INPUT_ROOT+"/"+identity,"sha256":hashlib.sha256(content).hexdigest(),"size_bytes":len(content)}
        request={"identity":identity,"temporary":temporary,"limit":MAX_TRANSFER_BYTES,
                 "sha256":expected["sha256"],"size":len(content)}
        await self._request(box.files.write,temporary,content,user="root",
                            request_timeout=self.configuration.request_timeout_seconds)
        result=await self._run_privileged(box,
            "exec "+ROOT_PYTHON+" -c "+shlex.quote(INSTALL_INPUT)+" "+shlex.quote(json.dumps(request)),
            timeout=10)
        try:
            if len(result.stdout)>2048 or json.loads(result.stdout)!=expected:
                raise ValueError("Input receipt mismatch")
        except (ValueError,TypeError):
            raise SandboxError("sandbox_input_receipt_invalid") from None
        return expected

    async def stage_input_file(self, handle, identity, local_file: VerifiedLocalFile, *, policy: FileTransferPolicy):
        """Stream a verified media snapshot into this already-owned allocation.

        Caller retains file ownership and source/grant checks. After upload
        starts, any failure requires allocation cleanup, not an upload retry.
        Cancellation propagates; its owner must also retire the allocation.
        The existing 8 MiB document/bytes method is deliberately unchanged.
        """
        from uuid import uuid4
        if (not isinstance(identity, str) or len(identity) != 64
                or any(c not in "0123456789abcdef" for c in identity)
                or not isinstance(local_file, VerifiedLocalFile)
                or not isinstance(policy, FileTransferPolicy)
                or policy.policy_id not in {MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY}
                or not 0 < local_file.size_bytes <= policy.max_bytes):
            raise ValueError("A stable input identity, verified file and bounded media policy are required")

        async def checked_chunks():
            digest, size = hashlib.sha256(), 0
            async for block in local_file.iter_chunks():
                size += len(block)
                if size > policy.max_bytes or size > local_file.size_bytes:
                    raise LocalFileError("local_file_size_exceeded")
                digest.update(block)
                yield block
            if size != local_file.size_bytes or digest.hexdigest() != local_file.content_sha256:
                raise LocalFileError("local_file_changed")

        # Check immutable content before any remote write, as well as during
        # transfer and after installation. No movie-sized bytes object exists.
        async for _ in checked_chunks():
            pass
        box = await self._connect(handle)
        stage_id, boundary = uuid4().hex, "jarvis-" + uuid4().hex
        temporary = "/var/lib/jarvis/incoming/" + stage_id
        expected = {"path": INPUT_ROOT + "/" + identity, "sha256": local_file.content_sha256,
                    "size_bytes": local_file.size_bytes}
        request = {"identity": identity, "stage_id": stage_id, "policy": policy.policy_id,
                   "limit": policy.max_bytes, "sha256": local_file.content_sha256, "size": local_file.size_bytes}
        # SDK 2.49.1's upload_url publicly specifies multipart/form-data. Its
        # files.write accepts binary IO but not async iterables. Stream this
        # fixed multipart envelope through our existing verified async reader.
        prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                  f"filename=\"{stage_id}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        suffix = f"\r\n--{boundary}--\r\n".encode()
        consumed = False

        async def body():
            nonlocal consumed
            yield prefix
            async for block in checked_chunks():
                yield block
            yield suffix
            consumed = True

        started = False
        stream = body()
        try:
            with private_http_logs():
                seconds = self.configuration.transfer_timeout_seconds
                url = box.upload_url(path=temporary, user="root", use_signature_expiration=ceil(seconds))
                parsed = urlsplit(url)
                if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                        or parsed.path != "/files" or parsed.fragment):
                    raise SandboxError("sandbox_input_upload_invalid")
                async with asyncio.timeout(seconds):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(seconds, read=min(seconds, 30),
                            connect=self.configuration.request_timeout_seconds,
                            pool=self.configuration.request_timeout_seconds),
                            follow_redirects=False, trust_env=False) as client:
                        started = True
                        async with client.stream("POST", url, content=stream, headers={
                                "Content-Type": "multipart/form-data; boundary=" + boundary,
                                "Content-Length": str(len(prefix) + local_file.size_bytes + len(suffix)),
                                "Accept-Encoding": "identity"}) as response:
                            if response.status_code not in {200, 201} or not consumed:
                                raise ValueError("upload incomplete")
                            size = 0
                            async for block in response.aiter_raw(chunk_size=2048):
                                size += len(block)
                                if size > 4096:
                                    raise ValueError("upload receipt limit")
            result = await self._run_privileged(box,
                "exec " + ROOT_PYTHON + " -c " + shlex.quote(INSTALL_INPUT_FILE) + " " + shlex.quote(json.dumps(request)),
                timeout=60)
            if not isinstance(result.stdout, str) or len(result.stdout) > 2048:
                raise ValueError("input receipt mismatch")
            receipt = json.loads(result.stdout)
            if (not isinstance(receipt, dict) or type(receipt.get("size_bytes")) is not int
                    or receipt != expected):
                raise ValueError("input receipt mismatch")
            async for _ in checked_chunks():
                pass
            current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
            if current.state != "running":
                raise SandboxError("sandbox_paused")
            return expected
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if started:
                raise SandboxError("sandbox_input_cleanup_uncertain") from None
            if isinstance(error, SandboxError):
                raise
            raise SandboxError("sandbox_input_upload_invalid") from None
        finally:
            await stream.aclose()

    async def read_workspace_file(self, handle, path, *, max_bytes=MAX_TRANSFER_BYTES):
        path=workspace_file(path)
        if type(max_bytes) is not int or not 0<=max_bytes<=MAX_TRANSFER_BYTES:
            raise ValueError("Invalid output file byte bound")
        box=await self._connect(handle)
        result=await self._run_privileged(box,
            "exec "+ROOT_PYTHON+" -c "+shlex.quote(READ_OUTPUT)+" "
            +shlex.quote(json.dumps({"path":path,"limit":max_bytes})),
            timeout=10)
        try:
            if len(result.stdout)>((max_bytes+2)//3)*4+256:
                raise ValueError("Oversized output")
            receipt=json.loads(result.stdout)
            content=base64.b64decode(receipt["content_base64"],validate=True)
            if (len(content)>max_bytes or receipt["size_bytes"]!=len(content)
                    or receipt["sha256"]!=hashlib.sha256(content).hexdigest()):
                raise ValueError("Invalid output receipt")
        except (ValueError,TypeError,KeyError):
            raise SandboxError("sandbox_file_receipt_invalid") from None
        return content

    async def _request(self, fn, *args, **kwargs):
        try:
            with private_http_logs():
                async with asyncio.timeout(self.configuration.request_timeout_seconds + 2):
                    return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, SandboxError):
                raise
            missing = isinstance(error, getattr(self.sdk, "SandboxNotFoundException", ()))
            code = "sandbox_not_found" if missing or getattr(error, "status_code", None) == 404 else "sandbox_provider_outcome_uncertain"
            raise _failure(code, stage="provider_request", outcome="not_found" if code == "sandbox_not_found" else "unknown",
                           error_type=type(error).__name__) from None

    async def _run_privileged(self, box, script, *, timeout, background=False, failure_parser=None,
                              stage="privileged_command"):
        # No task-controlled cwd or inherited environment enters the root script.
        # A missing executable/failed script stays a failure; never retry via PATH
        # or infer successful preparation from a partially completed command.
        # SDK2.49.1 bounds opening the stream with request_timeout; timeout
        # bounds the whole running stream. Do not wrap a foreground wait in the
        # short control-request ceiling. Background calls still only await start.
        if type(timeout) not in (int, float) or not 0 < timeout <= 305:
            raise ValueError("An explicit bounded privileged command is required")
        wait_seconds = self.configuration.request_timeout_seconds if background else timeout
        try:
            with private_http_logs():
                async with asyncio.timeout(wait_seconds + 2):
                    result = await box.commands.run(_privileged_command(script),
                        user="root", cwd="/", background=background, timeout=timeout,
                        request_timeout=self.configuration.request_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, SandboxError):
                raise
            if isinstance(error, getattr(self.sdk, "CommandExitException", ())):
                detail = None
                if failure_parser is not None:
                    try:
                        detail = failure_parser(error.stderr)
                    except (ValueError, TypeError, AttributeError, RecursionError):
                        pass
                raise _failure("sandbox_privileged_command_failed", stage=stage,
                    outcome="completed_nonzero", error_type="CommandExitException",
                    exit_code=getattr(error, "exit_code", None), detail=detail) from None
            missing = isinstance(error, getattr(self.sdk, "SandboxNotFoundException", ()))
            code = "sandbox_not_found" if missing or getattr(error, "status_code", None) == 404 else "sandbox_provider_outcome_uncertain"
            raise _failure(code, stage=stage, outcome="not_found" if code == "sandbox_not_found" else "unknown",
                           error_type=type(error).__name__) from None
        if not background and getattr(result, "exit_code", 0) != 0:
            raise _failure("sandbox_privileged_command_failed", stage=stage, outcome="completed_nonzero",
                           error_type="CommandResult", exit_code=result.exit_code)
        return result

    async def decode_media_render_input(self, handle, *, input_path, content_sha256, size_bytes, binding):
        """Verify selected software and decode exact protected input, without retry.

        The caller retains task/source authority. No allocation, rendering or
        publication occurs here. Cancellation/lost response does not confirm
        remote completion and never manufactures a successful inspection.
        """
        from app.sandbox.media_decode import DECODE_MEDIA_INPUT, decode_request, parse_decode_receipt, parse_decode_failure

        if not isinstance(handle, SandboxInfo):
            raise ValueError("A trusted existing allocation is required")
        # One inspection is capped at60s of the admitted command policy, with
        # 3s reserved for child cleanup/receipt. Control requests stay unchanged.
        timeout = min(60, handle.policy.command_seconds)
        if timeout < 5:
            raise SandboxError("sandbox_media_decode_budget_unavailable")
        request = decode_request(handle, input_path=input_path, content_sha256=content_sha256,
            size_bytes=size_bytes, binding=binding, budget_seconds=timeout - 3)
        current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        # info() checks actual owner, lifecycle and network policy. Recheck the
        # actual selected template as well before accessing its protected input.
        decode_request(current, input_path=input_path, content_sha256=content_sha256,
            size_bytes=size_bytes, binding=binding, budget_seconds=timeout - 3)
        box = await self._connect(current)
        result = await self._run_privileged(box,
            "exec " + ROOT_PYTHON + " -c " + shlex.quote(DECODE_MEDIA_INPUT) + " "
            + shlex.quote(json.dumps(request, separators=(",", ":"))), timeout=timeout,
            failure_parser=parse_decode_failure, stage="media_input_decode")
        try:
            return parse_decode_receipt(result.stdout, request)
        except (AttributeError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            raise SandboxError("sandbox_media_decode_receipt_invalid") from None

    async def materialize_media_render(self, handle, plan):
        """Install exact render inputs, not render or publish an output.

        The task owner must revalidate its claim/source/grants and reconcile the
        original command before calling. This boundary adds no action authority
        or retry: incomplete/unknown preparation remains in that allocation.
        Image selection is not software/build attestation.
        """
        from app.workspace.media_render import MediaRenderPlan
        from app.workspace.media_staging import (
            INSTALL_RENDER_INPUTS, expected_materialization_receipt, materialization_request,
        )

        if not isinstance(handle, SandboxInfo) or not isinstance(plan, MediaRenderPlan):
            raise ValueError("A trusted allocation and prepared render plan are required")
        if (handle.scope.owner_id, handle.scope.task_id) != (plan.owner_id, plan.task_id):
            raise SandboxError("sandbox_media_scope_mismatch")
        plan.require_allocation(handle)
        request = json.dumps(materialization_request(plan), ensure_ascii=False,
                             allow_nan=False, separators=(",", ":"))
        if len(request.encode("utf-8")) > 131072:
            raise ValueError("The fixed materialization request exceeds its limit")
        current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        plan.require_allocation(current)
        box = await self._connect(current)
        result = await self._run_privileged(box,
            "exec " + ROOT_PYTHON + " -c " + shlex.quote(INSTALL_RENDER_INPUTS)
            + " " + shlex.quote(request), timeout=20)

        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate receipt field")
                value[key] = item
            return value

        try:
            raw = result.stdout
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > 8192:
                raise ValueError("receipt size")
            receipt = json.loads(raw, object_pairs_hook=unique)
            expected = expected_materialization_receipt(plan)
            if (not isinstance(receipt, dict) or set(receipt) != set(expected)
                    or any(type(receipt[key]) is not type(value) or receipt[key] != value
                           for key, value in expected.items())):
                raise ValueError("receipt mismatch")
        except (AttributeError, ValueError, TypeError, KeyError, UnicodeError):
            raise SandboxError("sandbox_media_receipt_invalid") from None
        return receipt

    async def read_workspace_file_snapshot(self, handle, path, *, policy, max_bytes):
        """Freeze an existing VM output, then stream into a caller-owned snapshot.

        No file content travels through command stdout or a buffered SDK read.
        Failed remote cleanup remains an allocation-cleanup responsibility; the
        caller must not interpret an uncertain transfer as a publication receipt.
        """
        from uuid import uuid4
        path = workspace_file(path)
        if (not isinstance(policy, FileTransferPolicy) or policy.policy_id != MP4_OUTPUT_POLICY
                or type(max_bytes) is not int or not 0 < max_bytes <= policy.max_bytes):
            raise ValueError("A trusted, bounded MP4 output transfer policy is required")
        box = await self._connect(handle)
        stage_id = uuid4().hex
        local = None
        cancelled = False
        try:
            result = await self._run_privileged(box,
                "exec " + ROOT_PYTHON + " -c " + shlex.quote(FREEZE_OUTPUT) + " " + shlex.quote(json.dumps(
                    {"stage_id": stage_id, "path": path, "limit": max_bytes})),
                timeout=60)
            try:
                if len(result.stdout) > 2048:
                    raise ValueError("receipt")
                receipt = json.loads(result.stdout)
                if (not isinstance(receipt, dict) or set(receipt) != {"stage_id", "size_bytes", "sha256"}
                        or receipt["stage_id"] != stage_id or type(receipt["size_bytes"]) is not int
                        or not 0 < receipt["size_bytes"] <= max_bytes
                        or not isinstance(receipt["sha256"], str) or len(receipt["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in receipt["sha256"])):
                    raise ValueError("receipt")
            except (ValueError, TypeError, KeyError):
                raise SandboxError("sandbox_file_receipt_invalid") from None
            with private_http_logs():
                # Supported SDK signing API; do not extract private SDK headers
                # or expose this short-lived root-scoped URL in logs/tools.
                seconds = self.configuration.transfer_timeout_seconds
                url = box.download_url(path=OUTPUT_ROOT + "/" + stage_id, user="root",
                                       use_signature_expiration=ceil(seconds))
                parsed = urlsplit(url)
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                    raise SandboxError("sandbox_file_download_invalid")
                async with asyncio.timeout(seconds):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(seconds, read=min(seconds, 30),
                            connect=self.configuration.request_timeout_seconds,
                            pool=self.configuration.request_timeout_seconds),
                            follow_redirects=False, trust_env=False) as client:
                        async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                            # Never buffer a provider error body or follow a URL
                            # redirect. Raw bytes preserve the measured digest.
                            if response.status_code != 200 or response.headers.get("content-encoding", "identity").lower() != "identity":
                                raise SandboxError("sandbox_file_download_rejected")
                            length = response.headers.get("content-length")
                            if length is not None and (not length.isascii() or not length.isdecimal()
                                    or int(length) != receipt["size_bytes"]):
                                raise SandboxError("sandbox_file_receipt_invalid")
                            local = await _snapshot_chunks(response.aiter_raw(chunk_size=CHUNK_BYTES),
                                max_bytes=max_bytes, expected_size=receipt["size_bytes"],
                                expected_sha256=receipt["sha256"])
            current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
            if current.state != "running":
                raise SandboxError("sandbox_paused")
        except asyncio.CancelledError:
            cancelled = True
            if local is not None:
                local.close()
            raise
        except Exception as error:
            if local is not None:
                local.close()
            if isinstance(error, SandboxError):
                raise
            if isinstance(error, LocalFileError):
                raise SandboxError("sandbox_file_receipt_invalid") from None
            raise SandboxError("sandbox_file_transfer_failed") from None
        finally:
            try:
                removed = await self._run_privileged(box,
                    "exec " + ROOT_PYTHON + " -c " + shlex.quote(REMOVE_OUTPUT) + " "
                    + shlex.quote(json.dumps({"stage_id": stage_id})),
                    timeout=5)
                if len(removed.stdout) > 256 or json.loads(removed.stdout) != {"stage_id": stage_id, "removed": True}:
                    raise ValueError("cleanup receipt")
            except BaseException:
                if local is not None:
                    local.close()
                if not cancelled:
                    raise SandboxError("sandbox_file_cleanup_uncertain") from None
        return local

    @classmethod
    def _scope_metadata(cls, scope, policy):
        if not isinstance(scope, cls._scope_type):
            raise SandboxError("sandbox_scope_mismatch")
        return scope.metadata(policy)

    @classmethod
    def _owned(cls, raw, scope, policy):
        if any(raw.metadata.get(key) != value for key, value in cls._scope_metadata(scope, policy).items()):
            raise SandboxError("sandbox_scope_mismatch")

    @classmethod
    def _admit(cls, raw, scope, policy):
        cls._owned(raw, scope, policy)
        network, lifecycle = raw.network or {}, raw.lifecycle or {}
        if (raw.allow_internet_access is not policy.internet_access or network.get("allow_public_traffic") is not False
                or set(network.get("allow_out", [])) != set(policy.allowed_domains)
                or set(network.get("deny_out", [])) != (set() if policy.internet_access else {"0.0.0.0/0"})
                or network.get("rules") or network.get("egress_proxy") or network.get("https_ports")
                or network.get("mask_request_host") or raw.volume_mounts
                or lifecycle.get("on_timeout") != cls._on_timeout or lifecycle.get("auto_resume") is not False):
            raise SandboxError("sandbox_policy_mismatch")
        state = raw.state.value if hasattr(raw.state, "value") else raw.state
        return cls._handle_type(sandbox_id=raw.sandbox_id, scope=scope, policy=policy,
            state=state, expires_at=raw.end_at, template_id=raw.template_id)

    async def info(self, sandbox_id, scope, policy):
        self._scope_metadata(scope, policy)
        raw = await self._request(self.sdk.AsyncSandbox.get_info, sandbox_id, **self._options())
        if raw.sandbox_id != sandbox_id:
            raise SandboxError("sandbox_identity_mismatch")
        return self._admit(raw, scope, policy)

    async def create(self, scope, policy):
        return await self._create(scope, policy, template=policy.template)

    async def create_bound(self, scope, policy, *, template_id):
        """Use a trusted job's physical image without changing its grant policy.

        This checks template selection, not installed software or build identity.
        The managed runtime still owns qualification and preparation checks.
        A lost/mismatched response is reconciled by the existing allocation owner;
        this method never retries creation or falls back to the generic template.
        """
        if not isinstance(template_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", template_id):
            raise ValueError("An exact trusted physical template identifier is required")
        handle = await self._create(scope, policy, template=template_id)
        if handle.template_id != template_id:
            raise SandboxError("sandbox_template_mismatch")
        return handle

    async def _create(self, scope, policy, *, template):
        box = await self._request(self.sdk.AsyncSandbox.create,
            template=template, timeout=policy.lifetime_seconds,
            metadata=self._scope_metadata(scope, policy), envs={}, secure=True,
            allow_internet_access=policy.internet_access,
            network={"allow_out": list(policy.allowed_domains),
                     "deny_out": [] if policy.internet_access else ["0.0.0.0/0"],
                     "allow_public_traffic": False},
            lifecycle={"on_timeout": self._on_timeout, "auto_resume": False}, **self._options())
        # The caller records allocation scope before create. If this read or the
        # create acknowledgement is lost, discover that allocation; do not retry.
        current = await self.info(box.sandbox_id, scope, policy)
        if self.view_secrets is not None:
            self.view_secrets.capture(current, box.traffic_access_token)
        return current

    async def discover(self, scope, policy):
        with private_http_logs():
            paginator = self.sdk.AsyncSandbox.list(query=self.sdk.SandboxQuery(metadata=self._scope_metadata(scope, policy),
                state=[self.sdk.SandboxState.RUNNING, self.sdk.SandboxState.PAUSED]), limit=100, **self._options())
        identities = []
        while paginator.has_next:
            rows = await self._request(paginator.next_items)
            for row in rows:
                self._owned(row, scope, policy)
                identities.append(row.sandbox_id)
            if len(identities) > 100:
                raise SandboxError("sandbox_allocation_ambiguous")
        return identities

    async def _connect(self, handle, *, resume=False, timeout_seconds=None):
        current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        if current.state != "running" and not resume:
            raise SandboxError("sandbox_paused")
        box = await self._request(self.sdk.AsyncSandbox.connect, handle.sandbox_id,
            timeout=(timeout_seconds or handle.policy.lifetime_seconds) if resume else None, **self._options())
        if resume and timeout_seconds is not None:
            # connect can extend rather than shorten: enforce the remaining ledger bound.
            await self._request(box.set_timeout, timeout_seconds, **self._options())
        await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        return box

    async def prepare(self, handle):
        box = await self._connect(handle)
        marker = handle.scope.allocation_id.hex + ":" + handle.policy.fingerprint
        # Root owns the home directory, preventing login-profile poisoning.
        # Only the workspace belongs to the execution account; it has no sudo group.
        script = "\n".join(("set -eu", "/usr/bin/install -d -m 0700 /var/lib/jarvis",
            "if test -f /var/lib/jarvis/ready; then",
            "  test \"$(/usr/bin/cat /var/lib/jarvis/ready)\" = " + shlex.quote(marker),
            "else", "  /usr/sbin/useradd --no-create-home --home-dir /home/jarvis --shell /bin/bash --user-group jarvis",
            "  /usr/bin/install -d -m 0755 -o root -g root /home/jarvis",
            "  /usr/bin/install -d -m 0700 -o jarvis -g jarvis " + WORKSPACE,
            "  printf %s " + shlex.quote(marker) + " > /var/lib/jarvis/ready",
            "  /usr/bin/chmod 0600 /var/lib/jarvis/ready", "fi"))
        encoded = base64.b64encode(SCRIPT.encode()).decode()
        install = ("import base64,os,pathlib,uuid; p=pathlib.Path("+repr(RUNNER_PATH)+"); "
            "b=base64.b64decode("+repr(encoded)+"); "
            "t=p.with_name(p.name+'.'+uuid.uuid4().hex); "
            "fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); "
            "s=os.fdopen(fd,'wb'); s.write(b); s.flush(); os.fsync(s.fileno()); s.close(); t.replace(p)")
        script += "\n/usr/bin/install -d -m 0755 -o root -g root /home/jarvis/inputs"
        script += "\n/usr/bin/install -d -m 0700 -o root -g root /var/lib/jarvis/incoming"
        script += "\nexec " + ROOT_PYTHON + " -c " + shlex.quote(install)
        await self._run_privileged(box, script, timeout=20)
        if handle.policy.template == "desktop":
            from app.sandbox.desktop_environment import PREPARE_DESKTOP
            ready = await self._run_privileged(box, PREPARE_DESKTOP, timeout=20)
            if ready.stdout.strip() != "jarvis.desktop.ready.v1":
                raise SandboxError("sandbox_desktop_unavailable")
        current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        current = current.model_copy(update={"scope": handle.scope, "native_scope": handle.native_scope})
        if self.view_secrets is not None:
            self.view_secrets.capture(current, box.traffic_access_token)
        return current

    async def start_managed(self, handle, request: CommandRequest, *, operation_id, request_digest):
        from uuid import UUID
        if not isinstance(operation_id,UUID) or not isinstance(request,CommandRequest):
            raise ValueError("Typed command identity and parameters are required")
        if not isinstance(request_digest,str) or len(request_digest)!=64 or any(c not in "0123456789abcdef" for c in request_digest):
            raise ValueError("Invalid durable command fingerprint")
        box = await self._connect(handle)
        payload = request.model_dump(mode="json") | {
            "operation_id":str(operation_id),"allocation_id":str(handle.scope.allocation_id),
            "request_digest":request_digest,"timeout_seconds":handle.policy.command_seconds,
            "output_bytes":handle.policy.output_bytes,
            "desktop":handle.policy.template == "desktop",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        # Only this fixed supervisor has root access. It drops all supplementary
        # groups and runs the model command as jarvis with an explicit clean env.
        native = await self._run_privileged(box,"exec "+ROOT_PYTHON+" "+shlex.quote(RUNNER_PATH)+" "+shlex.quote(encoded),
            background=True,timeout=handle.policy.command_seconds+5)
        pid = native.pid
        with private_http_logs():
            await native.disconnect()
        return pid

    async def command_receipt(self, handle, operation_id, request_digest):
        from uuid import UUID
        if not isinstance(operation_id,UUID):
            raise ValueError("Use the durable command identity")
        box = await self._connect(handle)
        path = "/var/lib/jarvis/commands/"+operation_id.hex+"/receipt.json"
        limit = handle.policy.output_bytes*6+8192
        read = ("import base64,os\ntry:\n fd=os.open("+repr(path)+",os.O_RDONLY|os.O_NOFOLLOW)\n"
            "except FileNotFoundError:\n print('pending')\nelse:\n"
            " with os.fdopen(fd,'rb') as source: data=source.read("+str(limit+1)+")\n"
            " print(base64.b64encode(data).decode())")
        value = await self._run_privileged(box,"exec "+ROOT_PYTHON+" -c "+shlex.quote(read), timeout=5)
        encoded = value.stdout.strip()
        if encoded=="pending":
            return None
        try:
            if len(encoded)>((limit+2)//3)*4:
                raise ValueError("Oversized receipt")
            raw = base64.b64decode(encoded,validate=True)
            if len(raw)>limit:
                raise ValueError("Oversized receipt")
            receipt = CommandReceipt.model_validate_json(raw)
            if (receipt.operation_id!=operation_id or receipt.allocation_id!=handle.scope.allocation_id
                    or receipt.request_digest!=request_digest
                    or len(receipt.stdout.encode())+len(receipt.stderr.encode())>handle.policy.output_bytes):
                raise ValueError("Receipt scope or output mismatch")
            return receipt
        except (ValueError,TypeError):
            raise SandboxError("sandbox_command_receipt_invalid") from None

    async def start(self, handle, command, *, operation_id, cwd="."):
        from uuid import UUID
        if not isinstance(operation_id, UUID):
            raise ValueError("A durable command operation identity is required")
        if not isinstance(command, str) or not command.strip() or len(command) > 16000 or "\x00" in command:
            raise ValueError("Invalid bounded sandbox command")
        path = PurePosixPath(cwd)
        if not isinstance(cwd, str) or len(cwd) > 1000 or "\x00" in cwd or path.is_absolute() or ".." in path.parts:
            raise ValueError("Use a workspace-relative working directory")
        box = await self._connect(handle)
        output = _Output(handle.policy.output_bytes)
        environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/jarvis",
            "USER": ACCOUNT, "LOGNAME": ACCOUNT, "XDG_CACHE_HOME": WORKSPACE + "/.cache",
            "JARVIS_SANDBOX_OPERATION": str(operation_id),
        }
        if handle.policy.template == "desktop":
            # Match prepare()'s unprivileged desktop session, not the template's
            # sudo-capable default account or the root-owned /home/jarvis.
            from app.sandbox.desktop_environment import desktop_environment
            environment.update(desktop_environment())
        wrapped = "env -i " + " ".join(shlex.quote(f"{key}={value}") for key, value in environment.items())
        wrapped += " /bin/bash --noprofile --norc -c " + shlex.quote(command)
        native = await self._request(box.commands.run, wrapped, background=True, user=ACCOUNT,
            cwd=str(PurePosixPath(WORKSPACE) / path), timeout=handle.policy.command_seconds,
            request_timeout=self.configuration.request_timeout_seconds,
            on_stdout=lambda text: output.capture("stdout", text),
            on_stderr=lambda text: output.capture("stderr", text))
        return SandboxCommand(native, output, exit_error=self.sdk.CommandExitException)

    async def reconnect_command(self, handle, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("Use the recorded native PID")
        box = await self._connect(handle)
        output = _Output(handle.policy.output_bytes)
        native = await self._request(box.commands.connect, pid, timeout=handle.policy.command_seconds,
            request_timeout=self.configuration.request_timeout_seconds,
            on_stdout=lambda text: output.capture("stdout", text),
            on_stderr=lambda text: output.capture("stderr", text))
        if native.pid != pid:
            with private_http_logs():
                await native.disconnect()
            raise SandboxError("sandbox_command_identity_mismatch")
        return SandboxCommand(native, output, exit_error=self.sdk.CommandExitException)

    async def processes(self, handle):
        box = await self._connect(handle)
        return await self._request(box.commands.list, request_timeout=self.configuration.request_timeout_seconds)

    async def cancel_command(self, handle, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("Use the recorded native PID")
        box = await self._connect(handle)
        return await self._request(box.commands.kill, pid, request_timeout=self.configuration.request_timeout_seconds)

    async def pause(self, handle):
        await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        await self._request(self.sdk.AsyncSandbox.pause, handle.sandbox_id, **self._options())
        return await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)

    async def resume(self, handle, *, timeout_seconds=None):
        if timeout_seconds is not None and (type(timeout_seconds) is not int or
                not 1 <= timeout_seconds <= handle.policy.lifetime_seconds):
            raise ValueError("Resume must fit the remaining allocation lifetime")
        box = await self._connect(handle, resume=True, timeout_seconds=timeout_seconds)
        current = await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        current = current.model_copy(update={"scope": handle.scope, "native_scope": handle.native_scope})
        if self.view_secrets is not None:
            self.view_secrets.capture(current, box.traffic_access_token)
        return current

    async def kill(self, sandbox_id, scope, policy):
        self._scope_metadata(scope, policy)
        try:
            raw = await self._request(self.sdk.AsyncSandbox.get_info, sandbox_id, **self._options())
        except SandboxError as error:
            if error.code == "sandbox_not_found" and self.view_secrets is not None:
                self.view_secrets.remove(sandbox_id, scope)
            raise
        if raw.sandbox_id != sandbox_id:
            raise SandboxError("sandbox_identity_mismatch")
        # Cleanup must remain possible if an owned VM's network policy is wrong.
        self._owned(raw, scope, policy)
        result = await self._request(self.sdk.AsyncSandbox.kill, sandbox_id, **self._options())
        if self.view_secrets is not None:
            self.view_secrets.remove(sandbox_id, scope)
        return result
