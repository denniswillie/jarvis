"""One Core-authorized managed media qualification, never an automatic test.

Core must supply an exact reviewed RunPlan, real configured clients, current
owner-scoped plugin evidence and an external 300s process-group watchdog. Call
execute(..., execute_once=True) only after a new shared-lane allocation. Nothing
here loads credentials, selects an image or installs software. The default mode
drives production tools deterministically. The explicit EnglishQualification
option uses real Hermes under separately reviewed 300/60/390s bounds. Neither
mode qualifies speech. SQL uses the existing disposable source fixture.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import signal
from types import SimpleNamespace
from uuid import UUID, uuid4

from app.files.generated import ArtifactPolicy
from app.files.local_file import VerifiedLocalFile, snapshot_file
from app.files.media_runtime import prepare_artifact_media, prepare_media_inspector
from app.files.recording_ingress import RecordingClientReport, RecordingIngress, RecordingUpload
from app.files.storage import JarvisFileStorageClient
from app.files.transfer import FileTransferPolicy
from app.files.tools import _file_payload
from app.files.workspace_access import WorkspaceFileAccess, file_version
from app.knowledge.files import SourceFileRepository
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission
from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.managed import ManagedPluginRuntime
from app.plugins.media import MediaRendererBundle
from app.plugins.repository import PluginPolicyRepository
from app.recordings.admission import RecordingAdmission
from app.runtime.feature_config import ArtifactConfiguration, ArtifactMediaConfiguration, SandboxConfiguration, TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.e2b_provider import E2BProvider, _DIAGNOSTIC_TYPES
from app.sandbox.media_decode import MediaDecodeReceipt
from app.sandbox.models import SandboxError, SandboxPolicy
from app.sandbox.tools import SandboxToolBridge
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskStatus, digest, json_object
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from app.workspace.media_publication import MediaPublicationStore
from app.workspace.media_recipe import MediaEditRecipeV1, RetainedRef
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.hermes_sandbox_probe import ManifestedProvider, cleanup_fixture
from evals.second_brain.sandbox_artifact_probe import TrackedStorage
from evals.second_brain.sandbox_media_input_probe import safe_error, verify_absent
from evals.second_brain.sandbox_renderer_probe import SOURCE_PATH, SOURCE_SHA256, SOURCE_SIZE
from evals.second_brain.source_fixture import source_fixture


MAIN_SECONDS, CLEANUP_SECONDS, PARENT_SECONDS = 210, 60, 300
BYTE_LIMIT, STATUS_POLLS = 20_000_000, 24
OPERATION_KEY = "managed-mr01"
NOT_PROVEN = ["speech or frontier reasoning", "physical capture or heard Jarvis speech",
              "visual cut/caption quality or audio alignment", "phone playback or offline access",
              "correction MR02", "full concert", "Instagram publication"]


def failure_location(error):
    """Bounded project code identities only; never messages, source text or locals."""
    frames = []
    traceback = error.__traceback__
    backend = Path(__file__).parent.parent.parent
    scanned = 0
    origin = "unavailable_or_external"
    while traceback is not None and scanned < 32:
        code = traceback.tb_frame.f_code
        try:
            relative = Path(code.co_filename).relative_to(backend)
        except ValueError:
            relative = None
        if (relative is not None and relative.parts[0] in {"app", "evals"}
                and ".." not in relative.parts and len(str(relative)) <= 160
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", code.co_name)
                and 1 <= traceback.tb_lineno <= 1_000_000):
            frames.append({"file": str(relative), "function": code.co_name, "line": traceback.tb_lineno})
            frames = frames[-8:]
            if traceback.tb_next is None:
                origin = "project_exception_origin"
        traceback = traceback.tb_next
        scanned += 1
    kind = type(error).__name__
    return {"error_type": kind if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", kind) else "unrecognized",
            "frames": frames, "frames_scanned": scanned, "truncated": traceback is not None,
            "origin": origin}


def source_knowledge(source):
    def authorize(owner):
        if owner != source.owner:
            raise ValueError("Only the manifested fixture owner is allowed")
    # Same source-bound constructor port used by sandbox_artifact_probe. No
    # vendor memory client is configured; this fixture qualifies media, not recall.
    return SimpleNamespace(context_for=authorize, configuration=SimpleNamespace(area_id=source.area),
        sources=source.sources, retrieval=SimpleNamespace(sources=source.sources), projects=None)


@dataclass(frozen=True)
class RunPlan:
    directory: Path
    bundle: MediaRendererBundle
    policy: SandboxPolicy
    media: ArtifactMediaConfiguration
    ffmpeg_path: Path
    ffmpeg_sha256: str
    build_receipt: Path
    watchdog_receipt: Path

    def validate(self, *, parent_seconds=PARENT_SECONDS):
        if parent_seconds not in (PARENT_SECONDS, 390):
            raise ValueError("Only the scripted or English-case watchdog bound is reviewed")
        if (not isinstance(self.bundle, MediaRendererBundle) or not isinstance(self.policy, SandboxPolicy)
                or self.policy.networked or self.policy.template != "base"
                or self.policy.lifetime_seconds > 300 or self.policy.command_seconds > 180
                or self.policy.output_bytes > 16384 or self.media.storage_max_file_bytes != BYTE_LIMIT
                or self.bundle.renderer.max_output_bytes > BYTE_LIMIT
                or self.bundle.renderer.policy.max_source_bytes > BYTE_LIMIT):
            raise ValueError("Use the reviewed offline single-VM bounds")
        if self.bundle.renderer.policy.max_source_bytes < SOURCE_SIZE:
            raise ValueError("The admitted source allowance is insufficient")
        for path in (self.directory, self.build_receipt, self.watchdog_receipt, self.ffmpeg_path):
            if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
                raise ValueError("Explicit absolute operator-owned paths are required")
        info = self.directory.stat()
        if not self.directory.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Use an existing private qualification directory")
        raw = bounded_bytes(self.build_receipt, 65536)
        receipt = json.loads(raw)
        binding = self.bundle.renderer
        if (hashlib.sha256(raw).hexdigest() != binding.build_receipt_sha256
                or receipt.get("template_id") != binding.physical_template_id
                or receipt.get("build_id") != binding.build_id or receipt.get("status") != "ready"):
            raise ValueError("The retained ready build receipt does not match the renderer")
        guard = json.loads(bounded_bytes(self.watchdog_receipt, 4096))
        # Receipt is written by Core's actual external wrapper, not this module.
        if (guard.get("child_pid") != os.getpid() or guard.get("child_pgid") != os.getpgrp()
                or guard.get("parent_pid") != os.getppid()
                or type(guard.get("hard_deadline_seconds")) is not int
                or not 1 <= guard["hard_deadline_seconds"] <= parent_seconds):
            raise ValueError("An active, bounded external process-group watchdog is required")
        os.kill(guard["parent_pid"], 0)
        pinned_binary(Path(self.media.ffprobe_path), self.media.ffprobe_sha256)
        pinned_binary(self.ffmpeg_path, self.ffmpeg_sha256)


def bounded_bytes(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        import stat
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("A regular file is required")
        value = stream.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError("Inspection input exceeds its bound")
    return value


def pinned_binary(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
            or not path.is_absolute() or not os.access(path, os.X_OK)):
        raise ValueError("A pinned executable is required")
    raw = bounded_bytes(path, 128 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("The local decoder changed")


class SingleVMProvider(ManifestedProvider):
    """Existing allocation-first manifest, with a strict one-create guard."""
    async def _stage(self, name, operation, *args, **kwargs):
        before = asyncio.get_running_loop().time()
        item = {"stage": name, "state": "started", "at": datetime.now(UTC).isoformat()}
        self.probe.record.setdefault("provider_stages", []).append(item)
        self.probe.persist()
        try:
            result = await operation(*args, **kwargs)
        except BaseException as error:
            item.update(state="failed", error=safe_error(error), failure_location=failure_location(error))
            # Only the provider's fixed diagnostic vocabulary is retained.
            diagnostic = getattr(error, "diagnostic", None)
            if isinstance(diagnostic, dict):
                selected = {}
                for key, allowed in {
                    "stage": {"provider_request", "privileged_command", "media_input_decode"},
                    "outcome": {"not_found", "unknown", "completed_nonzero"},
                    "error_type": _DIAGNOSTIC_TYPES,
                }.items():
                    if isinstance(diagnostic.get(key), str) and diagnostic[key] in allowed:
                        selected[key] = diagnostic[key]
                if type(diagnostic.get("exit_code")) is int and -255 <= diagnostic["exit_code"] <= 255:
                    selected["exit_code"] = diagnostic["exit_code"]
                if "inspection" in diagnostic:
                    from app.sandbox.media_decode import parse_decode_failure
                    try:
                        selected["inspection"] = parse_decode_failure(json.dumps(diagnostic["inspection"]))
                    except (ValueError, TypeError, RecursionError):
                        pass
                item["provider_diagnostic"] = selected
            raise
        else:
            item["state"] = "completed"
            return result
        finally:
            item["seconds"] = round(asyncio.get_running_loop().time() - before, 3)
            self.probe.persist()

    async def prepare(self, handle):
        return await self._stage("prepare", self.probe.provider.prepare, handle)

    async def stage_input_file(self, handle, identity, local_file, *, policy):
        return await self._stage("stage_input_file", self.probe.provider.stage_input_file,
                                 handle, identity, local_file, policy=policy)

    async def materialize_media_render(self, handle, plan):
        return await self._stage("materialize_media_render", self.probe.provider.materialize_media_render, handle, plan)

    async def create(self, *_args, **_kwargs):
        raise SandboxError("qualification_requires_bound_image")

    async def create_bound(self, scope, policy, *, template_id):
        if self.probe.record["resources"]:
            raise SandboxError("qualification_single_vm_consumed")
        if template_id != self.probe.record["renderer"]["physical_template_id"]:
            raise SandboxError("qualification_image_mismatch")
        self.probe.scopes.append((scope, policy))
        item = {"scope": scope.model_dump(mode="json"), "policy": policy.model_dump(mode="json"), "phase": "planned"}
        self.probe.record["resources"].append(item)
        self.probe.persist()
        handle = await self._stage("create_bound", self.probe.provider.create_bound,
                                   scope, policy, template_id=template_id)
        item.update(sandbox_id=handle.sandbox_id, template_id=handle.template_id, phase="created")
        self.probe.persist()
        return handle

    async def decode_media_render_input(self, handle, **kwargs):
        result = await self._stage("decode_media_render_input", self.probe.provider.decode_media_render_input, handle, **kwargs)
        if not isinstance(result, MediaDecodeReceipt) or result.inspection_timing is None:
            raise ValueError("Actual typed input-decode evidence is required")
        self.probe.record.setdefault("input_decodes", []).append(result.model_dump(mode="json"))
        self.probe.persist()
        return result


class OwnedCloudStorage(TrackedStorage):
    """Extend the existing object manifest to streamed MP4 uploads."""
    def __init__(self, storage, probe, owner):
        super().__init__(storage, probe)
        self.prefix = "owners/" + owner + "/"

    def check_upload(self, path, size):
        if (not path.startswith(self.prefix) or ".." in path.split("/")
                or not 0 < size <= BYTE_LIMIT
                or len(self.probe.record.get("storage_objects", [])) >= 3
                or any(item["path"] == path for item in self.probe.record.get("storage_objects", []))):
            raise ValueError("Only the three new fixture-owned objects may be uploaded once")

    async def upload_bytes(self, *, path, content, content_type):
        self.check_upload(path, len(content))
        return await super().upload_bytes(path=path, content=content, content_type=content_type)

    async def upload_file(self, *, path, local_file, content_type):
        if not isinstance(local_file, VerifiedLocalFile):
            raise ValueError("A verified snapshot is required")
        self.check_upload(path, local_file.size_bytes)
        item = {"bucket": self.storage.bucket, "path": path, "size": local_file.size_bytes,
                "sha256": local_file.content_sha256, "phase": "planned"}
        self.probe.record.setdefault("storage_objects", []).append(item)
        self.probe.persist()
        try:
            result = await self.storage.upload_file(path=path, local_file=local_file, content_type=content_type)
        except BaseException:
            item["phase"] = "unknown"
            self.probe.persist()
            raise
        item["phase"] = "uploaded"
        self.probe.persist()
        return result


async def retain(probe, name, local):
    """Preserve exact private review artifacts; never overwrite an earlier run."""
    if name not in {"original.mp4", "output.mp4"} or not isinstance(local, VerifiedLocalFile):
        raise ValueError("Only owned original/output snapshots may be retained")
    path = probe.root / name
    record = {"path": str(path), "state": "planned", "size_bytes": local.size_bytes,
              "content_sha256": local.content_sha256}
    probe.record.setdefault("review_artifacts", {})[name] = record
    probe.persist()
    checksum, size = hashlib.sha256(), 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as target:
        async for chunk in local.iter_chunks():
            size += len(chunk)
            if size > min(BYTE_LIMIT, local.size_bytes):
                raise ValueError("Retained copy exceeds its bound")
            checksum.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
        if size != local.size_bytes or checksum.hexdigest() != local.content_sha256:
            raise ValueError("Retained copy differs from its source")
        os.fchmod(target.fileno(), 0o400)
    record["state"] = "retained_exact_unreviewed"
    probe.persist()
    return path


async def bounded_process(argv, *, seconds=20):
    """Fixed local decoder child, bounded pipes and process-group cleanup."""
    process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    async def read(stream, maximum):
        result = bytearray()
        while chunk := await stream.read(4096):
            result.extend(chunk)
            if len(result) > maximum:
                raise ValueError("Decoder output exceeded its bound")
        return bytes(result)
    readers = [asyncio.create_task(read(process.stdout, 65536)), asyncio.create_task(read(process.stderr, 16384))]
    waiter = asyncio.create_task(process.wait())
    try:
        async with asyncio.timeout(seconds):
            stdout, stderr, code = await asyncio.gather(*readers, waiter)
        if code != 0 or stderr.strip():
            raise ValueError("Independent output decoder failed")
        return stdout
    finally:
        async def cleanup():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            async with asyncio.timeout(3):
                await process.wait()
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        task = asyncio.create_task(cleanup())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()


def output_metadata(raw):
    data = json.loads(raw)
    streams = data["streams"]
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    if len(streams) != 2 or len(videos) != 1 or len(audios) != 1:
        raise ValueError("Expected exactly one video and one audio stream")
    video, audio = videos[0], audios[0]
    duration = Fraction(data["format"]["duration"])
    if (video["codec_name"] != "h264" or audio["codec_name"] != "aac"
            or (video["width"], video["height"]) != (720, 1280)
            or Fraction(video["avg_frame_rate"]) != 24 or int(video["nb_read_frames"]) != 240
            or not Fraction(10) <= duration <= Fraction("10.05")
            or audio["sample_rate"] != "48000"):
        raise ValueError("Rendered output differs from the fixed MR01 frame/stream/duration contract")
    return {"video_frames": 240, "fps": 24, "width": 720, "height": 1280,
            "duration_seconds": str(duration), "video_codec": "h264", "audio_codec": "aac",
            "sample_rate": 48000}


def audio_sample_evidence(raw):
    """Count decoded mono PCM samples, independently of the video timeline."""
    samples = packets = 0
    time_base = None
    for line in raw.decode("ascii").splitlines():
        if line.startswith("#tb 0:"):
            if time_base is not None:
                raise ValueError("Duplicate audio sample clock")
            time_base = Fraction(line.split(":", 1)[1].strip())
        elif line.strip() and not line.startswith("#"):
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 6:
                raise ValueError("Invalid decoded audio packet evidence")
            stream, dts, pts, duration, size = map(int, fields[:5])
            checksum = fields[5]
            if (stream != 0 or dts != samples or pts != samples or duration <= 0
                    or size != duration * 2 or len(checksum) != 64
                    or any(char not in "0123456789abcdef" for char in checksum)):
                raise ValueError("Decoded audio is not continuous positive mono PCM")
            samples += duration
            packets += 1
    if time_base != Fraction(1, 48000) or not 475200 <= samples <= 489600 or packets == 0:
        raise ValueError("Independent audio decoding did not produce the expected sample duration")
    return {"decoded_audio_samples": samples, "decoded_audio_packets": packets,
            "decoded_audio_sample_rate": 48000, "decoded_audio_channels": 1,
            "decoded_audio_duration_seconds": str(Fraction(samples, 48000)),
            "decoded_audio_framehash_sha256": hashlib.sha256(raw).hexdigest()}


async def inspect_output(plan, local):
    pinned_binary(Path(plan.media.ffprobe_path), plan.media.ffprobe_sha256)
    pinned_binary(plan.ffmpeg_path, plan.ffmpeg_sha256)
    local._check()
    restriction = ["-protocol_whitelist", "file", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0"]
    raw = await bounded_process([plan.media.ffprobe_path, "-v", "error", *restriction,
        "-count_frames", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate",
        "-of", "json", "-i", str(local.path)])
    measured = output_metadata(raw)
    raw = await bounded_process([str(plan.ffmpeg_path), "-nostdin", "-v", "error", "-xerror",
        "-max_alloc", "268435456", "-threads", "1", "-err_detect", "explode", *restriction,
        "-i", str(local.path), "-map", "0:v:0", "-map", "0:a:0", "-threads", "1",
        "-progress", "pipe:1", "-nostats", "-f", "null", "-"])
    progress = dict(line.split("=", 1) for line in raw.decode("ascii").splitlines() if "=" in line)
    if (progress.get("progress") != "end" or int(progress.get("frame", "0")) != 240
            or not 9_900_000 <= int(progress.get("out_time_us", "0")) <= 10_200_000):
        raise ValueError("Full output decoding did not complete with the expected timeline")
    # Separate audio-only decoding: a valid video timeline cannot satisfy this.
    # Reset PCM timestamps to the actual sample count, not retained stream PTS.
    audio = await bounded_process([str(plan.ffmpeg_path), "-nostdin", "-v", "error", "-xerror",
        "-max_alloc", "268435456", "-threads", "1", "-err_detect", "explode", *restriction,
        "-i", str(local.path), "-map", "0:a:0", "-vn", "-sn", "-dn", "-threads", "1",
        "-ac", "1", "-af", "asetpts=N/SR/TB", "-c:a", "pcm_s16le",
        "-f", "framehash", "-hash", "sha256", "pipe:1"])
    measured.update(audio_sample_evidence(audio))
    audio_path = plan.directory / "independent-audio-framehash.txt"
    fd = os.open(audio_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(audio)
    audio_path.chmod(0o400)
    measured["decoded_audio_framehash_path"] = str(audio_path)
    # Re-read every exact byte after the subprocesses; file metadata alone is insufficient.
    checksum = hashlib.sha256()
    async for chunk in local.iter_chunks():
        checksum.update(chunk)
    if checksum.hexdigest() != local.content_sha256:
        raise ValueError("Inspected output bytes changed")
    return measured | {"full_video_audio_decode": True, "content_sha256": local.content_sha256,
                       "size_bytes": local.size_bytes, "playback_or_content_review": False}


def recipe_for(reference, template):
    return MediaEditRecipeV1.model_validate({"schema_version": 1, "edit_id": str(uuid4()),
        "previous_recipe": None, "template": template.model_dump(mode="json"),
        "sources": [{"source_id": "original", "file": reference.model_dump(mode="json")}],
        "segments": [{"segment_id": "first", "source_id": "original", "in_ms": 2000, "out_ms": 7000},
                     {"segment_id": "second", "source_id": "original", "in_ms": 12000, "out_ms": 17000}],
        "captions": [{"caption_id": "title", "start_frame": 24, "end_frame": 96,
                      "text": "Jarvis managed edit"}],
        "output_profile": "portrait_720p24_v1", "fit": "contain", "audio": "preserve"})


async def drive_tools(bridge, recipe, record, *, sleep=asyncio.sleep):
    async def call(name, arguments):
        result = await bridge.invoke(name, arguments, provider_call_id="qualification-" + uuid4().hex)
        record(name, result)
        if result.get("status") != "success":
            raise ValueError("A managed media tool did not succeed")
        return result
    value = await call("media_render", {"operation_key": OPERATION_KEY, "recipe": recipe.model_dump(mode="json")})
    for _ in range(STATUS_POLLS):
        state = value["media_operation"]["state"]
        if state == "rendered":
            break
        if state not in {"submitted", "unknown"}:
            raise ValueError("The original render is not progressing; no replacement is allowed")
        await sleep(5)
        value = await call("media_status", {"operation_key": OPERATION_KEY})
    if value["media_operation"]["state"] != "rendered":
        raise ValueError("Original render remains unresolved within the observation budget")
    first = await call("media_publish", {"operation_key": OPERATION_KEY})
    if first["media_operation"]["state"] != "published":
        raise ValueError("The original publication remains unresolved")
    second = await call("media_publish", {"operation_key": OPERATION_KEY})
    if (second["media_operation"].get("output") != first["media_operation"].get("output")
            or second["media_operation"]["state"] != "published"):
        raise ValueError("Same-key replay changed the accepted output")
    return first


async def _scenario(plan, probe, source, storage, observed_provider, observe, *, english=None):
    from evals.second_brain import managed_media_english as english_case
    limits = (english_case.task_limits() if english is not None else
              TaskLimits(lease_seconds=300, active_seconds=MAIN_SECONDS, max_attempts=1))
    tasks = TaskRepository(source.pool, limits)
    registry = TaskWorkerRegistry(source.pool)
    authority = PluginAuthority(PluginPolicyRepository(source.pool), ContextRepository(tasks), tasks,
                               deployment_baseline=CURATED_DEPLOYMENT_BASELINE_V1)
    runtime = ManagedPluginRuntime(authority, bundles=(plan.bundle,), observe=observe)
    artifact = ArtifactConfiguration(area_id=source.area, media=plan.media, policy=ArtifactPolicy(
        formats=("json", "mp4"), max_files=2, max_file_bytes=65536,
        mp4_max_file_bytes=BYTE_LIMIT, max_total_bytes=BYTE_LIMIT + 65536))
    output_policy, output_inspector = await prepare_artifact_media(artifact)
    capture_policy = FileTransferPolicy("mp4_capture_v1", BYTE_LIMIT)
    capture_inspector = await prepare_media_inspector(plan.media, capture_policy)
    async with source.pool.acquire() as connection:
        session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", source.owner)
    selections = SessionKnowledgeAreas(source.sources, tasks, source.area)
    admission = RecordingAdmission(source.sources, tasks, selections, owner_id=source.owner)
    ingress = RecordingIngress(source.sources, storage, admission, policy=capture_policy, inspect_file=capture_inspector)
    command = CommandAdmission(owner_id=source.owner, session_id=session,
        provider_session_key="synthetic-managed-media", provider_call_id="synthetic-recording-" + uuid4().hex,
        tool_name="fixture_recording", capability="recordings.start", normalized_arguments={"include_jarvis_audio": True})
    scope = (await admission.admit(command, area_id=source.area, policy=capture_policy)).scope
    original = await snapshot_file(SOURCE_PATH, max_bytes=BYTE_LIMIT,
                                  expected_size=SOURCE_SIZE, expected_sha256=SOURCE_SHA256)
    try:
        await retain(probe, "original.mp4", original)
        measured = await capture_inspector(original, capture_policy)
        upload = RecordingUpload(filename="synthetic-original.mp4",
            description="Synthetic marker/tone fixture, not physical capture or Jarvis speech.",
            size_bytes=SOURCE_SIZE, content_sha256=SOURCE_SHA256,
            client_report=RecordingClientReport(generation_id=uuid4(), timeline_end_ms=measured.duration_ms,
                                               video_frames=1, source_audio_frames=0, padded_audio_frames=0))
        saved = await ingress.finalize(owner_id=source.owner, session_id=session, recording_id=scope.recording_id,
                                       upload=upload, chunks=original.iter_chunks())
    finally:
        original.close()
    probe.check("synthetic_original_saved", saved.get("outcome") == "saved")
    files = SourceFileRepository(source.sources, storage, owner_id=source.owner, area_id=source.area,
                                media_policy=output_policy, media_inspector=output_inspector)
    probe.record["original_receipt"] = saved
    file = await files.get_file(owner_id=source.owner, file_id=UUID(saved["file_id"]))
    reference = RetainedRef(file_id=file.id, version_id=file_version(file), content_sha256=SOURCE_SHA256)
    knowledge = source_knowledge(source)
    factory = WorkerContextFactory(source.pool, knowledge=knowledge, file_storage=storage,
        sandbox_provider=observed_provider, sandbox_policy=plan.policy, artifact_configuration=artifact,
        media_policy=output_policy, media_inspector=output_inspector, managed_plugins=runtime)
    providers = {"frontier": ConfiguredTaskProvider(
        english.frontier.model if english is not None else "scripted-managed-media-no-model")}
    caps = tuple(dict.fromkeys(factory.capabilities + plan.policy.required_capabilities + artifact.required_capabilities))
    worker = "managed-media-qualification-" + probe.run_id
    await registry.advertise(worker_id=worker, owner_id=source.owner, host_id="qualification",
                             providers=providers, ttl_seconds=300, capabilities=caps)
    service = TaskService(tasks, providers, registry=registry, sandbox_configuration=SandboxConfiguration(policy=plan.policy),
                          artifact_configuration=artifact, managed_plugins=runtime)
    if english is None:
        task = await service.create_media(source.owner, TaskCreate(request_id=uuid4(), title="Synthetic MR01 edit",
            requested_outcome="Make the deterministic two-cut synthetic edit; no model is invoked by this fixture.",
            session_id=session, knowledge_area_id=source.area, execution_environment="offline_sandbox", publish_artifacts=True))
    else:
        task, receipt = await english_case.admit(service, files, admission, source.owner, session, scope.recording_id, saved)
        probe.record["english_admission"] = receipt
    claim = await tasks.claim_next(worker, ("frontier",), owner_id=source.owner, worker_capabilities=caps)
    if claim is None or claim.task.id != task.id:
        raise ValueError("The admitted media task was not claimed by its matching worker")
    probe.record.update(task_id=str(task.id), attempt_id=str(claim.attempt_id))
    probe.persist()
    def record(name, value):
        probe.record.setdefault("scripted_tools", []).append({"name": name, "result": value,
                                                             "at": datetime.now(UTC).isoformat()})
        probe.persist()
    if english is None:
        recipe = recipe_for(reference, plan.bundle.renderer.template)
        context = TaskRunner(tasks, {}).context(claim)
        bridge = (await factory(context)).tools
        if not isinstance(bridge, SandboxToolBridge) or bridge.media is None:
            raise ValueError("The normal managed media bridge is required")
        result = (await drive_tools(bridge, recipe, record))["media_operation"]
    else:
        bridge, result = await english_case.execute_claim(english, factory, tasks, claim, probe)
    store = MediaPublicationStore(tasks, files)
    saved_plan = await store.load_plan(source.owner, task.id, result["plan_sha256"])
    output = await store.load_accepted_output(source.owner, task.id, result["plan_sha256"])
    if saved_plan is None or output is None or output.model_dump(mode="json") != result["output"]:
        raise ValueError("Accepted source/action/version evidence is missing or mismatched")
    if english is not None:
        recipe = saved_plan.plan.recipe.recipe
        english_case.verify_recipe(recipe, reference, plan.bundle.renderer.template)
        probe.check("english_recipe_matches_request", True)
    probe.record.update(accepted_recipe=saved_plan.file.model_dump(mode="json"),
                        accepted_output=output.model_dump(mode="json"),
                        recipe=recipe.model_dump(mode="json"))
    async with tasks.transaction(source.owner) as connection:
        rows = await connection.fetch("""select c.*,p.content as parameters,r.content as result
            from jarvis_sandbox_commands c
            join jarvis_context_payloads p on p.owner_id=c.owner_id and p.id=c.parameters_payload_id
            left join jarvis_context_payloads r on r.owner_id=c.owner_id and r.id=c.result_payload_id
            where c.owner_id=$1 and c.task_id=$2 and p.availability='available'""", source.owner, task.id)
    if len(rows) != 1:
        raise ValueError("The fixture must execute exactly one durable render command")
    # Real TaskRunner has settled the English claim. Read its exact durable
    # receipt owner-scoped; do not manufacture a fresh claim to inspect history.
    command_row = rows[0] if english is not None else await bridge.broker.commands.current(claim, rows[0]["id"])
    expected = saved_plan.plan.command().model_dump(mode="json")
    actual = json_object(command_row["result"]) if command_row["result"] is not None else {}
    if (command_row["phase"] != "completed" or json_object(command_row["parameters"]) != expected
            or command_row["request_digest"] != digest(expected)
            or actual.get("started") is not True or actual.get("exit_code") != 0
            or actual.get("timed_out") is not False or actual.get("runner_error") is not None):
        raise ValueError("No exact successful fixed command receipt is available")
    probe.record["fixed_command_receipt"] = {"id": str(command_row["id"]),
        "allocation_id": str(command_row["allocation_id"]), "request": expected,
        "request_digest": command_row["request_digest"], "result": actual}
    probe.check("one_vm_and_three_uploads", len(probe.record["resources"]) == 1
                and len(probe.record.get("input_decodes", [])) == 1
                and len(probe.record.get("storage_objects", [])) == 3)
    # Reopen the accepted output via its actual source publication and authenticated storage read.
    reopened = await files.get_file(owner_id=source.owner, file_id=output.file.file_id)
    if file_version(reopened) != output.file.version_id:
        raise ValueError("The accepted output source version changed")
    manifest, transfer = await files.publications.read_manifest(owner_id=source.owner, file_id=reopened.id)
    if manifest.content_sha256 != output.file.content_sha256 or manifest.size_bytes != output.size_bytes:
        raise ValueError("Accepted output manifest changed")
    metadata_file, metadata_extra = await WorkspaceFileAccess(files).media_metadata(
        source.owner, session, reopened.id, output.file.version_id)
    metadata_path = probe.root / "output-file-metadata.json"
    metadata_raw = (json.dumps({"file": _file_payload(metadata_file) | metadata_extra},
                               sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(metadata_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(metadata_raw)
    metadata_path.chmod(0o400)
    probe.record["historical_output_metadata"] = {
        "path": str(metadata_path), "sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "meaning": "Ordinary Files serialization before fixture cleanup; historical presentation evidence only.",
        "authenticated_phone_delivery": False, "live_source_after_cleanup": False}
    probe.persist()
    local = await files.read_file(owner_id=source.owner, file_id=reopened.id)
    try:
        current = await files.get_file(owner_id=source.owner, file_id=reopened.id)
        if (local.content_sha256 != output.file.content_sha256 or local.size_bytes != output.size_bytes
                or current is None or file_version(current) != output.file.version_id):
            raise ValueError("Output readback no longer matches the exact accepted version")
        await retain(probe, "output.mp4", local)
        probe.record["independent_output_inspection"] = await inspect_output(plan, local)
    finally:
        local.close()
    if english is None:
        await tasks.abandon(claim, target=TaskStatus.PAUSED, reason="scripted_qualification_complete_not_expert_answer")
    await registry.withdraw(source.owner, worker)
    probe.persist()
    return files


async def execute(plan, *, execute_once=False, database_url=None, provider=None, storage=None, observe=None, english=None):
    """Operator-only entry point. False/missing opt-in causes no I/O at all."""
    if execute_once is not True:
        raise ValueError("A newly allocated execute_once=True run is required")
    if (type(provider) is not E2BProvider or type(storage) is not JarvisFileStorageClient
            or not isinstance(database_url, str) or not database_url or not callable(observe)):
        raise ValueError("Core must inject real configured clients and current scoped readiness evidence")
    from evals.second_brain import managed_media_english as english_case
    if english is not None:
        if type(english) is not english_case.EnglishQualification:
            raise ValueError("Use the explicit English qualification configuration")
        english.validate(plan.directory)
    main_seconds = english_case.MAIN_SECONDS if english is not None else MAIN_SECONDS
    parent_seconds = english_case.PARENT_SECONDS if english is not None else PARENT_SECONDS
    not_proven = english_case.NOT_PROVEN if english is not None else NOT_PROVEN
    plan.validate(parent_seconds=parent_seconds)
    original = await snapshot_file(SOURCE_PATH, max_bytes=BYTE_LIMIT,
                                  expected_size=SOURCE_SIZE, expected_sha256=SOURCE_SHA256)
    original.close()
    # An atomic marker prevents this reviewed allocation from being reused after failure/unknown.
    marker = os.open(plan.directory / "execute-once", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(marker)
    probe = Probe(provider, plan.directory)
    probe.record.update(format="jarvis.managed-media.qualification.v1",
        renderer=plan.bundle.renderer.model_dump(mode="json"), scripted=english is None, model_called=False,
        limits={"main_seconds": main_seconds, "cleanup_seconds": CLEANUP_SECONDS,
                "parent_seconds": parent_seconds, "vm_count": 1, "render_count": 1,
                "storage_uploads": 3, "bytes_per_mp4": BYTE_LIMIT}, not_proven=not_proven)
    if english is not None:
        probe.record["english_configuration"] = english.public_metadata()
    probe.persist()
    manager = source_fixture(database_url, probe.root / "database", test_name="managed-media-qualification")
    source = files = tracked = None
    failure = cleanup_failure = None
    try:
        async with asyncio.timeout(main_seconds):
            source = await manager.__aenter__()
            probe.record.update(database_manifest=str(source.fixture.path), fixture_owner=source.owner)
            probe.persist()
            tracked = OwnedCloudStorage(storage, probe, source.owner)
            observed = SingleVMProvider(probe)
            files = await _scenario(plan, probe, source, tracked, observed, observe, english=english)
    except BaseException as error:
        failure = error
        probe.record["failure"] = safe_error(error)
        probe.record["failure_location"] = failure_location(error)
        probe.persist()
    finally:
        async def cleanup():
            errors = []
            if english is not None:
                try:
                    english_case.verify_native_cleanup(english, probe)
                except BaseException as error:
                    errors.append(safe_error(error))
            if source is not None:
                tasks = TaskRepository(source.pool, TaskLimits(lease_seconds=300))
                fixture = SimpleNamespace(source=source, allocations=SandboxAllocationRepository(tasks))
                try:
                    await cleanup_fixture(fixture, probe)
                    await verify_absent(probe)
                except BaseException as error:
                    errors.append(safe_error(error))
                try:
                    # Reuse the existing cleanup owner even if scenario setup failed early.
                    if tracked is not None:
                        cleanup_files = files or SourceFileRepository(source.sources, tracked,
                            owner_id=source.owner, area_id=source.area)
                        await tracked.cleanup(cleanup_files)
                except BaseException as error:
                    errors.append(safe_error(error))
                # The existing fixture owns SQL teardown. Native scopes/IDs and
                # upload identities remain in Probe's separate on-disk manifest,
                # including unknown effects; never rely on async-generator GC.
                try:
                    await manager.__aexit__(None, None, None)
                except BaseException as error:
                    errors.append(safe_error(error))
            probe.record["cleanup_errors"] = errors
            probe.persist()
            if errors:
                raise ValueError("Cleanup remains unresolved; exact resource manifests retained")
            local = await snapshot_file(SOURCE_PATH, max_bytes=BYTE_LIMIT,
                expected_size=SOURCE_SIZE, expected_sha256=SOURCE_SHA256)
            local.close()
            probe.check("original_hash_unchanged", True)
        async def bounded_cleanup():
            async with asyncio.timeout(CLEANUP_SECONDS):
                await cleanup()
        cleanup_task = asyncio.create_task(bounded_cleanup())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as error:
                failure = failure or error
            except BaseException:
                # Record the terminal cleanup error below without losing the
                # primary scenario failure or the final qualification receipt.
                break
        try:
            cleanup_task.result()
        except BaseException as error:
            cleanup_failure = error
            probe.record["cleanup_failure"] = safe_error(error)
            probe.record["cleanup_failure_location"] = failure_location(error)
        probe.record["qualification"] = {"passed": failure is None and cleanup_failure is None,
            "scope": ("English admission and real Hermes/TaskRunner, SQL/cloud storage/E2B, independent output decode"
                      if english is not None else "scripted managed task, real SQL/cloud storage/E2B, independent output decode"),
            "full_demo_passed": False, "not_proven": not_proven}
        probe.persist()
    if failure is not None:
        raise failure
    if cleanup_failure is not None:
        raise cleanup_failure
    return probe.path


if __name__ == "__main__":
    print(json.dumps({"status": "source_only", "execution": "Core-reviewed Python injection with execute_once=True",
                      "live_run_authorized": False, "limits": {"main": MAIN_SECONDS, "cleanup": CLEANUP_SECONDS,
                      "parent": PARENT_SECONDS}, "not_proven": NOT_PROVEN}))
