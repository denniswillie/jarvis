"""Fixed decode program and exact receipt for the existing E2B provider.

This creates no allocation, render command, publication, retry or grant. The
task owner retains source/claim authority. A lost acknowledgement stays unknown.
"""
from __future__ import annotations

import json
import re
from typing import Literal
from uuid import UUID

from pydantic import Field, StrictBool, model_validator

from app.evidence.models import FrozenRecord
from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS
from app.sandbox.file_transfer import INPUT_ROOT
from app.sandbox.models import SandboxInfo
from app.workspace.media_render import RendererBinding


class MediaInspectionTiming(FrozenRecord):
    inventory_ms: int = Field(strict=True, ge=0, le=60000)
    metadata_ms: int = Field(strict=True, ge=0, le=60000)
    video_decode_ms: int = Field(strict=True, ge=0, le=60000)
    audio_decode_ms: int = Field(strict=True, ge=0, le=60000)
    total_ms: int = Field(strict=True, ge=0, le=60000)

    @model_validator(mode="after")
    def measured_components(self):
        if self.inventory_ms + self.metadata_ms + self.video_decode_ms + self.audio_decode_ms > self.total_ms:
            raise ValueError("Inspection phases cannot exceed their total duration")
        return self


class MediaDecodeReceipt(FrozenRecord):
    format: Literal["jarvis.media-decode.v1"]
    sandbox_id: str = Field(strict=True, min_length=1, max_length=200)
    allocation_id: UUID
    input_path: str = Field(strict=True, min_length=1, max_length=200)
    content_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, gt=0, le=CAPTURE_MAX_BYTES)
    software_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    duration_ms: int = Field(strict=True, gt=0, le=CAPTURE_MAX_DURATION_MS)
    video_streams: int = Field(strict=True, ge=1, le=1)
    audio_streams: int = Field(strict=True, ge=0, le=1)
    video_decoded: StrictBool
    audio_decoded: StrictBool
    video_frames: int = Field(strict=True, gt=0)
    audio_duration_us: int = Field(strict=True, ge=0)
    inspection_timing: MediaInspectionTiming | None = None

    @model_validator(mode="after")
    def successful_tracks(self):
        if (not self.video_decoded or self.audio_decoded != bool(self.audio_streams)
                or (self.audio_duration_us > 0) != bool(self.audio_streams)):
            raise ValueError("Successful decode observations must match actual track presence")
        return self


def decode_request(handle, *, input_path, content_sha256, size_bytes, binding, budget_seconds):
    if not isinstance(handle, SandboxInfo) or not isinstance(binding, RendererBinding):
        raise ValueError("A trusted allocation and qualified renderer binding are required")
    if (handle.template_id != binding.physical_template_id or handle.policy.networked
            or not isinstance(input_path, str)
            or re.fullmatch(re.escape(INPUT_ROOT) + r"/[0-9a-f]{64}", input_path) is None
            or not isinstance(content_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
            or type(size_bytes) is not int or not 0 < size_bytes <= min(binding.policy.max_source_bytes, CAPTURE_MAX_BYTES)
            or type(budget_seconds) is not int or not 1 <= budget_seconds <= 57):
        raise ValueError("Exact offline image, protected input and bounded decode are required")
    return {"format": "jarvis.media-decode-request.v1", "sandbox_id": handle.sandbox_id,
            "allocation_id": str(handle.scope.allocation_id), "input_path": input_path,
            "content_sha256": content_sha256, "size_bytes": size_bytes,
            "software_sha256": binding.software_sha256,
            "max_duration_ms": min(binding.policy.max_source_duration_ms, CAPTURE_MAX_DURATION_MS),
            "budget_seconds": budget_seconds}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate receipt field")
        result[key] = value
    return result


DECODE_PHASES = frozenset({"request", "input", "inventory_entrypoint", "privilege_drop", "inventory",
                         "metadata", "video_decode", "audio_decode", "input_recheck"})
DECODE_REASONS = frozenset({"container_invalid", "decode_cleanup_unconfirmed", "decode_empty_or_incomplete",
    "decode_interrupted", "decode_process_failed", "decode_timeout", "decoder_privilege_drop_unconfirmed",
    "directory_protection", "duplicate_key", "duration_exceeded", "duration_invalid", "input_changed",
    "input_protection_or_size", "input_size", "inventory_protection", "metadata_invalid", "process_output_limit",
    "progress_invalid", "request_identity", "request_limit", "request_shape", "software_identity_mismatch",
    "streams_invalid", "trusted_wrapper_required", "unprivileged_decoder_required", "video_invalid", "decode_system_failure"})


def parse_decode_failure(raw):
    """Only fixed inspection phase/reason codes and bounded measured durations."""
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 512:
        raise ValueError("A bounded inspection failure receipt is required")
    value = json.loads(raw, object_pairs_hook=_unique)
    if (not isinstance(value, dict) or set(value) != {"format", "phase", "reason", "elapsed_ms", "phase_ms"}
            or value["format"] != "jarvis.media-decode-failure.v1"
            or not isinstance(value["phase"], str) or not isinstance(value["reason"], str)
            or value["phase"] not in DECODE_PHASES or value["reason"] not in DECODE_REASONS
            or any(type(value[key]) is not int or not 0 <= value[key] <= 65000 for key in ("elapsed_ms", "phase_ms"))
            or value["phase_ms"] > value["elapsed_ms"]):
        raise ValueError("Invalid inspection failure receipt")
    return value


def parse_decode_receipt(raw, request):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 8192:
        raise ValueError("A bounded decode receipt is required")
    result = MediaDecodeReceipt.model_validate(json.loads(raw, object_pairs_hook=_unique))
    if result.inspection_timing is None or result.inspection_timing.total_ms > request["budget_seconds"] * 1000:
        raise ValueError("Separately measured inventory and decode phases are required")
    for key in ("sandbox_id", "input_path", "content_sha256", "size_bytes", "software_sha256"):
        if getattr(result, key) != request[key]:
            raise ValueError("Decode receipt does not match the requested input")
    if (str(result.allocation_id) != request["allocation_id"]
            or result.duration_ms > request["max_duration_ms"]):
        raise ValueError("Decode receipt exceeds its exact allocation or duration")
    return result


# Executed through the existing fixed privileged wrapper. Paths, binaries,
# privilege drop and subprocess arguments are fixed code, never model fields.
DECODE_MEDIA_INPUT = (r'''
import hashlib,json,os,pwd,re,selectors,signal,stat,subprocess,sys,time
from decimal import Decimal,ROUND_CEILING,localcontext
INPUTS="/home/jarvis/inputs"
ROOT="/opt/jarvis-render"
BIN=ROOT+"/clip/node_modules/@remotion/compositor-linux-x64-gnu"
KEYS={"format","sandbox_id","allocation_id","input_path","content_sha256","size_bytes",
      "software_sha256","max_duration_ms","budget_seconds"}
FIELDS="format=format_name,duration,size:format_tags=major_brand:stream=index,codec_type,codec_name,width,height,duration:stream_disposition=attached_pic"
class DecodeFailure(ValueError):pass
PHASES=__PHASES__
REASONS=__REASONS__
DIAGNOSTIC={"phase":"request","started":time.monotonic(),"phase_started":time.monotonic()}
def phase(name,operation):
    DIAGNOSTIC["phase"]=name;DIAGNOSTIC["phase_started"]=time.monotonic()
    return operation()
def failure_receipt(error):
    now=time.monotonic()
    reason=error.args[0] if isinstance(error,DecodeFailure) and error.args else "decode_system_failure"
    if reason not in REASONS:reason="decode_system_failure"
    return {"format":"jarvis.media-decode-failure.v1","phase":DIAGNOSTIC["phase"],"reason":reason,
            "elapsed_ms":max(0,int((now-DIAGNOSTIC["started"])*1000)),
            "phase_ms":max(0,int((now-DIAGNOSTIC["phase_started"])*1000))}
def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise DecodeFailure("duplicate_key")
        result[key]=value
    return result
def validate(request):
    if type(request) is not dict or set(request)!=KEYS or request["format"]!="jarvis.media-decode-request.v1":
        raise DecodeFailure("request_shape")
    patterns={"sandbox_id":r"[A-Za-z0-9_-]{1,200}","allocation_id":r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
              "input_path":re.escape(INPUTS)+r"/[0-9a-f]{64}","content_sha256":r"[0-9a-f]{64}","software_sha256":r"[0-9a-f]{64}"}
    if any(type(request[key]) is not str or re.fullmatch(pattern,request[key]) is None for key,pattern in patterns.items()):
        raise DecodeFailure("request_identity")
    for key,maximum in (("size_bytes",__CAPTURE_MAX_BYTES__),("max_duration_ms",__CAPTURE_MAX_DURATION_MS__),("budget_seconds",57)):
        if type(request[key]) is not int or not 0<request[key]<=maximum:raise DecodeFailure("request_limit")
def check_time(deadline):
    if time.monotonic()>=deadline:raise DecodeFailure("decode_timeout")
def protected_directory(path):
    fd=os.open("/",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in path.strip("/").split("/"):
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
            info=os.fstat(fd)
            if info.st_uid!=0 or info.st_mode&0o022:raise DecodeFailure("directory_protection")
        result,fd=fd,None
        return result
    finally:
        if fd is not None:os.close(fd)
def signature(info):
    return tuple(getattr(info,key) for key in ("st_dev","st_ino","st_mode","st_uid","st_size","st_mtime_ns","st_ctime_ns"))
def verify_input_at(directory,request,deadline,owner_uid=0):
    info=os.fstat(directory)
    if info.st_uid!=owner_uid or info.st_mode&0o022:raise DecodeFailure("directory_protection")
    fd=os.open(request["input_path"].rsplit("/",1)[1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
    with os.fdopen(fd,"rb") as source:
        before=os.fstat(source.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid!=owner_uid or before.st_mode&0o222
                or before.st_size!=request["size_bytes"]):raise DecodeFailure("input_protection_or_size")
        digest=hashlib.sha256();size=0
        while True:
            check_time(deadline)
            block=source.read(65536)
            if not block:break
            size+=len(block)
            if size>request["size_bytes"]:raise DecodeFailure("input_size")
            digest.update(block)
        if (size!=request["size_bytes"] or digest.hexdigest()!=request["content_sha256"]
                or signature(before)!=signature(os.fstat(source.fileno()))):raise DecodeFailure("input_changed")
    return signature(before)
def verify_input(request,deadline):
    directory=protected_directory(INPUTS)
    try:return verify_input_at(directory,request,deadline)
    finally:os.close(directory)
def verify_inventory_entrypoint():
    directory=protected_directory(ROOT)
    try:
        fd=os.open("inventory.py",os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_mode&0o222:raise DecodeFailure("inventory_protection")
        finally:os.close(fd)
    finally:os.close(directory)
def run_bounded(argv,deadline,limit=65536):
    check_time(deadline)
    process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        start_new_session=True,close_fds=True,cwd="/",env={"PATH":"/usr/bin:/bin","HOME":"/home/jarvis","LC_ALL":"C"})
    stdout=bytearray();stderr=bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            for pipe,target,ceiling in ((process.stdout,stdout,limit),(process.stderr,stderr,16384)):
                os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,(target,ceiling))
            while selector.get_map():
                check_time(deadline)
                for key,_ in selector.select(min(0.1,max(0,deadline-time.monotonic()))):
                    block=os.read(key.fd,8192)
                    if not block:selector.unregister(key.fileobj);continue
                    target,ceiling=key.data;target.extend(block)
                    if len(target)>ceiling:raise DecodeFailure("process_output_limit")
        check_time(deadline)
        code=process.wait(timeout=max(0.001,deadline-time.monotonic()))
        if code!=0 or stderr:raise DecodeFailure("decode_process_failed")
        return bytes(stdout)
    except subprocess.TimeoutExpired:
        raise DecodeFailure("decode_timeout") from None
    finally:
        # These fixed tools do not spawn background services. Kill/reap the
        # owned process group on every return, including timeout/cancellation.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:raise DecodeFailure("decode_cleanup_unconfirmed") from None
        finally:
            process.stdout.close();process.stderr.close()
def duration(value,maximum):
    if type(value) is not str or not 1<=len(value)<=64 or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?",value) is None:
        raise DecodeFailure("duration_invalid")
    number=Decimal(value)
    if not 0<number<=Decimal(maximum)/1000:raise DecodeFailure("duration_exceeded")
    return number
def metadata(raw,request):
    data=json.loads(raw,object_pairs_hook=unique)
    if type(data) is not dict or type(data.get("format")) is not dict:raise DecodeFailure("metadata_invalid")
    fmt=data["format"];tags=fmt.get("tags",{})
    if (fmt.get("format_name") not in ("mp4","mov,mp4,m4a,3gp,3g2,mj2") or type(tags) is not dict
            or tags.get("major_brand") not in {"isom","iso2","iso3","iso4","iso5","iso6","mp41","mp42","avc1"}
            or fmt.get("size")!=str(request["size_bytes"])):raise DecodeFailure("container_invalid")
    streams=data.get("streams")
    if type(streams) is not list or not 1<=len(streams)<=2:raise DecodeFailure("streams_invalid")
    videos=audios=0;indices=set();durations=[duration(fmt.get("duration"),request["max_duration_ms"])]
    for stream in streams:
        if type(stream) is not dict:raise DecodeFailure("streams_invalid")
        index=stream.get("index")
        if type(index) is not int or index<0 or index in indices:raise DecodeFailure("streams_invalid")
        indices.add(index);durations.append(duration(stream.get("duration"),request["max_duration_ms"]))
        if stream.get("codec_type")=="video" and stream.get("codec_name")=="h264":
            disposition=stream.get("disposition")
            if (type(disposition) is not dict or type(disposition.get("attached_pic")) is not int or disposition["attached_pic"]!=0
                    or any(type(stream.get(key)) is not int or not 1<=stream[key]<=7680 for key in ("width","height"))):
                raise DecodeFailure("video_invalid")
            videos+=1
        elif stream.get("codec_type")=="audio" and stream.get("codec_name")=="aac":audios+=1
        else:raise DecodeFailure("streams_invalid")
    if videos!=1 or audios>1:raise DecodeFailure("streams_invalid")
    with localcontext() as ctx:
        ctx.prec=80
        milliseconds=int((max(durations)*1000).to_integral_value(rounding=ROUND_CEILING))
    return audios,milliseconds
def progress(raw,field):
    text=raw.decode("ascii");values={}
    for line in text.splitlines():
        key,separator,value=line.partition("=")
        if not separator:raise DecodeFailure("progress_invalid")
        values[key]=value.strip()
    value=values.get(field,"")
    if values.get("progress")!="end" or re.fullmatch(r"[0-9]{1,18}",value) is None or int(value)<=0:
        raise DecodeFailure("decode_empty_or_incomplete")
    return int(value)
def decode_arguments(path,track):
    args=[BIN+"/ffmpeg","-nostdin","-v","error","-xerror","-max_alloc","268435456",
          "-threads","1","-filter_threads","1","-filter_complex_threads","1",
          "-protocol_whitelist","file","-f","mov","-enable_drefs","0","-use_absolute_path","0",
          "-err_detect","explode","-i",path,"-map","0:"+track+":0","-sn","-dn"]
    # The bundled FFmpeg omits null's default wrapped_avframe encoder. Explicit
    # raw sinks still decode every frame/sample; stream copy would not prove it.
    args+=( ["-an","-c:v","rawvideo","-fps_mode","passthrough"] if track=="v" else ["-vn","-c:a","pcm_s16le"] )
    return args+["-threads","1","-nostats","-progress","pipe:1","-f","null","/dev/null"]
def perform_decode(request,deadline,run=run_bounded):
    started=time.monotonic();timings={}
    def measured(name,operation):
        begin=time.monotonic()
        value=phase(name,operation)
        timings[name+"_ms"]=int((time.monotonic()-begin)*1000)
        return value
    # The inventory verifier measures actual installed contents/protection. Its
    # self-consistency alone is insufficient: compare to the admitted mapping.
    inventory=json.loads(measured("inventory",lambda:run(["/usr/bin/python3","-I","-S",ROOT+"/inventory.py","--verify"],deadline,8192)),object_pairs_hook=unique)
    if (type(inventory) is not dict or set(inventory)!={"software_sha256","render_qualified"}
            or inventory.get("software_sha256")!=request["software_sha256"] or inventory.get("render_qualified") is not False):
        raise DecodeFailure("software_identity_mismatch")
    path=request["input_path"]
    raw=measured("metadata",lambda:run([BIN+"/ffprobe","-v","error","-protocol_whitelist","file","-f","mov","-enable_drefs","0",
             "-use_absolute_path","0","-show_entries",FIELDS,"-of","json","-i",path],deadline))
    audios,milliseconds=metadata(raw,request)
    frames=progress(measured("video_decode",lambda:run(decode_arguments(path,"v"),deadline)),"frame")
    audio_us=progress(measured("audio_decode",lambda:run(decode_arguments(path,"a"),deadline)),"out_time_us") if audios else 0
    timings.setdefault("audio_decode_ms",0);timings["total_ms"]=int((time.monotonic()-started)*1000)
    return {"format":"jarvis.media-decode.v1",**{key:request[key] for key in
            ("sandbox_id","allocation_id","input_path","content_sha256","size_bytes","software_sha256")},
            "duration_ms":milliseconds,"video_streams":1,"audio_streams":audios,
            "video_decoded":True,"audio_decoded":bool(audios),"video_frames":frames,"audio_duration_us":audio_us,
            "inspection_timing":timings}
def interrupted(*_):raise DecodeFailure("decode_interrupted")
def main(request):
    DIAGNOSTIC.update(phase="request",started=time.monotonic(),phase_started=time.monotonic())
    validate(request)
    if os.geteuid()!=0:raise DecodeFailure("trusted_wrapper_required")
    deadline=time.monotonic()+request["budget_seconds"]
    before=phase("input",lambda:verify_input(request,deadline));phase("inventory_entrypoint",verify_inventory_entrypoint)
    DIAGNOSTIC.update(phase="privilege_drop",phase_started=time.monotonic())
    account=pwd.getpwnam("jarvis")
    if account.pw_uid<=0 or account.pw_gid<=0:raise DecodeFailure("unprivileged_decoder_required")
    # Every privileged input/directory descriptor is closed before this drop.
    os.setgroups([]);os.setgid(account.pw_gid);os.setuid(account.pw_uid)
    if (os.getuid()!=account.pw_uid or os.geteuid()!=account.pw_uid
            or os.getgid()!=account.pw_gid or os.getegid()!=account.pw_gid or os.getgroups()):
        raise DecodeFailure("decoder_privilege_drop_unconfirmed")
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    result=perform_decode(request,deadline)
    if phase("input_recheck",lambda:verify_input(request,deadline))!=before:raise DecodeFailure("input_changed")
    check_time(deadline)
    return result
if __name__=="__main__":
    try:
        if len(sys.argv)!=2 or len(sys.argv[1])>4096:raise DecodeFailure("request_limit")
        print(json.dumps(main(json.loads(sys.argv[1],object_pairs_hook=unique))))
    except (ValueError,OSError,KeyError,TypeError,RecursionError) as error:
        # Do not print media paths, bytes or subprocess error output.
        print(json.dumps(failure_receipt(error),separators=(",",":")),file=sys.stderr);sys.exit(2)
'''.replace("__PHASES__", repr(sorted(DECODE_PHASES)))
    .replace("__REASONS__", repr(sorted(DECODE_REASONS)))
    .replace("__CAPTURE_MAX_BYTES__", str(CAPTURE_MAX_BYTES))
    .replace("__CAPTURE_MAX_DURATION_MS__", str(CAPTURE_MAX_DURATION_MS)))
