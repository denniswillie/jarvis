"""Trusted worker composition for opt-in generated MP4 publication.

No package installation, executable discovery, provider call or storage-limit
inference occurs here. The deployment operator supplies the storage ceiling.
Binary integrity is checked off the event loop before advertising capability
and before each inspection; this is not cloud or physical-media qualification.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import stat

from app.files.media_inspection import MediaInspectionError, inspect_mp4_file
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY


_MAX_EXECUTABLE_BYTES = 128 * 1024 * 1024


class PinnedMP4Inspector:
    def __init__(self, configuration, policy):
        if not isinstance(policy, FileTransferPolicy) or not policy.is_media:
            raise MediaInspectionError("media_inspection_policy_invalid")
        self.configuration = configuration
        self.policy = policy

    def _verify_binary(self):
        path = Path(self.configuration.ffprobe_path)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if (not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_EXECUTABLE_BYTES
                        or not os.access(path, os.X_OK)):
                    raise MediaInspectionError("media_probe_binary_invalid")
                digest = hashlib.sha256()
                size = 0
                while chunk := stream.read(64 * 1024):
                    size += len(chunk)
                    if size > _MAX_EXECUTABLE_BYTES:
                        raise MediaInspectionError("media_probe_binary_invalid")
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
            current = path.stat(follow_symlinks=False)
            signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                       value.st_mtime_ns, value.st_ctime_ns)
            if (size != before.st_size or signature(before) != signature(after)
                    or signature(after) != signature(current)
                    or digest.hexdigest() != self.configuration.ffprobe_sha256):
                raise MediaInspectionError("media_probe_binary_mismatch")
        except OSError:
            raise MediaInspectionError("media_probe_binary_unavailable") from None

    async def verify(self):
        await asyncio.to_thread(self._verify_binary)

    async def __call__(self, local_file, policy):
        if (not isinstance(policy, FileTransferPolicy) or policy.policy_id != self.policy.policy_id
                or policy.max_bytes > self.policy.max_bytes):
            raise MediaInspectionError("media_inspection_policy_exceeded")
        await self.verify()
        return await inspect_mp4_file(local_file, policy,
            ffprobe_path=self.configuration.ffprobe_path,
            timeout=self.configuration.probe_timeout_seconds)

    async def inspect_retained_input(self, local_file, admitted_policy):
        """Inspect an authorized retained source under its persisted policy.

        The caller must load this policy from authorized publication metadata,
        never model arguments. FileTransferPolicy and inspect_mp4_file enforce
        its deployment limit and existing capture/output hard size and duration
        ceilings. The configured binary pin and probe timeout remain shared;
        this does not change the publication policy enforced by __call__, grant
        source access, or establish successful decoding.
        """
        if (not isinstance(admitted_policy, FileTransferPolicy)
                or admitted_policy.policy_id not in (MP4_CAPTURE_POLICY, MP4_OUTPUT_POLICY)):
            raise MediaInspectionError("media_inspection_policy_exceeded")
        await self.verify()
        return await inspect_mp4_file(local_file, admitted_policy,
            ffprobe_path=self.configuration.ffprobe_path,
            timeout=self.configuration.probe_timeout_seconds)


async def prepare_media_inspector(configuration, policy):
    """Bind one concrete capture OR generated policy; callers cannot swap it."""
    inspector = PinnedMP4Inspector(configuration, policy)
    await inspector.verify()
    return inspector


async def prepare_artifact_media(configuration):
    """Return one shared trusted policy/inspector pair for work and recovery."""
    if configuration is None or not configuration.policy.supports_media:
        return None, None
    if configuration.media is None:
        raise MediaInspectionError("media_configuration_missing")
    policy = configuration.policy.for_format("mp4",
        deployment_limit_bytes=configuration.media.storage_max_file_bytes)
    inspector = await prepare_media_inspector(configuration.media, policy)
    return policy, inspector
