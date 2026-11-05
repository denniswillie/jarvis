"""Small trusted E2B receipt writer, not a shell/model/agent framework.

The supervisor and receipts belong to root; only the requested command runs as
jarvis. A VM deletion remains the boundary for escaped background descendants.
"""
import hashlib


SCRIPT = r'''
import base64
import json
import os
from pathlib import Path
import pwd
import selectors
import signal
import subprocess
import sys
import time
from uuid import UUID

request = json.loads(base64.b64decode(sys.argv[1],validate=True))
operation = UUID(request["operation_id"])
root = Path("/var/lib/jarvis/commands")
root.mkdir(mode=0o700,exist_ok=True)
directory = root / operation.hex
try:
    directory.mkdir(mode=0o700)
except FileExistsError:
    # An existing accepted operation is never executed for a second time.
    raise SystemExit(0)

state = {
    "format":"jarvis.sandbox.command.v1",
    "operation_id":str(operation),"allocation_id":request["allocation_id"],
    "request_digest":request["request_digest"],"pid":os.getpid(),
    "phase":"running","started":False,"exit_code":None,
    "stdout":"","stderr":"","timed_out":False,"output_limited":False,"runner_error":None,
}
def persist():
    target = directory / "receipt.next"
    with open(target,"x",opener=lambda path,flags:os.open(path,flags,0o600)) as stream:
        json.dump(state,stream,ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    target.replace(directory / "receipt.json")

def stop_process(process):
    if process is not None:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass

process = None
streams = selectors.DefaultSelector()
buffers = {"stdout":bytearray(),"stderr":bytearray()}
try:
    persist()
    account = pwd.getpwnam("jarvis")
    environment = {
        "PATH":"/usr/local/bin:/usr/bin:/bin","HOME":"/home/jarvis",
        "USER":"jarvis","LOGNAME":"jarvis","XDG_CACHE_HOME":"/home/jarvis/workspace/.cache",
        "JARVIS_SANDBOX_OPERATION":str(operation),
    }
    # This switch comes from the parent's admitted policy, never model args.
    if request.get("desktop") is True:
        environment.update({"DISPLAY":":0","HOME":"/home/jarvis/workspace",
            "XDG_RUNTIME_DIR":"/home/jarvis/workspace/.runtime",
            "XDG_CONFIG_HOME":"/home/jarvis/workspace/.config"})
    process = subprocess.Popen(
        ["/bin/bash","--noprofile","--norc","-c",request["command"]],
        cwd="/home/jarvis/workspace/"+request["cwd"],env=environment,
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        user=account.pw_uid,group=account.pw_gid,extra_groups=(),
        start_new_session=True,close_fds=True,
    )
    state["started"] = True
    persist()
    for channel in buffers:
        stream = getattr(process,channel)
        os.set_blocking(stream.fileno(),False)
        streams.register(stream,selectors.EVENT_READ,channel)
    deadline = time.monotonic()+request["timeout_seconds"]
    size = 0
    while streams.get_map():
        remaining = deadline-time.monotonic()
        if remaining<=0:
            state["timed_out"] = True
            stop_process(process)
            break
        for key,_ in streams.select(min(.2,remaining)):
            chunk = os.read(key.fileobj.fileno(),8192)
            if not chunk:
                streams.unregister(key.fileobj)
                key.fileobj.close()
                continue
            available = max(0,request["output_bytes"]-size)
            buffers[key.data].extend(chunk[:available])
            size += min(available,len(chunk))
            if len(chunk)>available:
                state["output_limited"] = True
                stop_process(process)
                break
        if state["output_limited"]:
            break
    remaining = max(.001,deadline-time.monotonic())
    try:
        state["exit_code"] = process.wait(timeout=remaining if not (state["timed_out"] or state["output_limited"]) else 2)
    except subprocess.TimeoutExpired:
        state["timed_out"] = True
        stop_process(process)
        try:
            state["exit_code"] = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            state["runner_error"] = "process_termination_unconfirmed"
except Exception as error:
    stop_process(process)
    state["runner_error"] = type(error).__name__
finally:
    for key in list(streams.get_map().values()):
        key.fileobj.close()
    streams.close()
    # Bound the UTF-8 representation too, including replacement of invalid bytes.
    remaining = request["output_bytes"]
    for channel in buffers:
        encoded = bytes(buffers[channel]).decode("utf-8",errors="replace").encode("utf-8")
        state[channel] = encoded[:remaining].decode("utf-8",errors="ignore")
        if len(encoded)>remaining:
            state["output_limited"] = True
        remaining -= len(state[channel].encode("utf-8"))
    state["phase"] = "completed"
    persist()
'''

RUNNER_DIGEST = hashlib.sha256(SCRIPT.encode()).hexdigest()
RUNNER_PATH = "/var/lib/jarvis/runner-" + RUNNER_DIGEST + ".py"
