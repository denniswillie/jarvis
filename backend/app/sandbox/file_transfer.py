"""Bounded native file transfers; no shell authority or host filesystem paths."""
from pathlib import PurePosixPath

from app.files.transfer import CAPTURE_MAX_BYTES, MP4_OUTPUT_MAX_BYTES

MAX_TRANSFER_BYTES = 8 * 1024 * 1024
INPUT_ROOT = "/home/jarvis/inputs"
OUTPUT_ROOT = "/var/lib/jarvis/outgoing"


def workspace_file(value):
    if (not isinstance(value, str) or not value or len(value)>1000 or "\x00" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or not PurePosixPath(value).parts):
        raise ValueError("Use a nonempty workspace-relative file path")
    return str(PurePosixPath(value))


# Executed only by the trusted parent. The model cannot supply Python or an
# arbitrary destination. Incoming bytes remain under a root-only directory.
INSTALL_INPUT = r'''
import hashlib,json,os,stat,sys
request=json.loads(sys.argv[1])
identity=request["identity"]
if len(identity)!=64 or any(c not in "0123456789abcdef" for c in identity):
    raise ValueError("identity")
source=request["temporary"]
if not source.startswith("/var/lib/jarvis/incoming/") or "/" in source.removeprefix("/var/lib/jarvis/incoming/"):
    raise ValueError("incoming")
fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
with os.fdopen(fd,"rb") as stream:
    info=os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size>request["limit"]:
        raise ValueError("size or type")
    content=stream.read(request["limit"]+1)
if len(content)!=request["size"] or hashlib.sha256(content).hexdigest()!=request["sha256"]:
    raise ValueError("content")
target="/home/jarvis/inputs/"+identity
try:
    prior=os.open(target,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
except FileNotFoundError:
    os.chmod(source,0o444)
    # The root-only incoming directory is on the same VM filesystem. link is
    # exclusive and atomic; concurrent identical retries never overwrite input.
    try:
        os.link(source,target,follow_symlinks=False)
    except FileExistsError:
        pass
finally:
    if "prior" in locals():
        os.close(prior)
fd=os.open(target,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
with os.fdopen(fd,"rb") as stream:
    info=os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_mode&0o222:
        raise ValueError("input not protected")
    saved=stream.read(request["limit"]+1)
if saved!=content:
    raise ValueError("input conflict")
os.unlink(source)
print(json.dumps({"path":target,"sha256":request["sha256"],"size_bytes":len(saved)}))
'''

# Media-only counterpart; legacy document installation above stays unchanged.
# The trusted caller supplies policy/identity, never code or a destination path.
INSTALL_INPUT_FILE = r'''
import hashlib,json,os,stat,sys
INCOMING="/var/lib/jarvis/incoming"
INPUTS="/home/jarvis/inputs"
def fingerprint(info):
    return tuple(getattr(info,key) for key in
                 ("st_dev","st_ino","st_mode","st_uid","st_size","st_mtime_ns","st_ctime_ns"))
def protected_directory(path):
    fd=os.open("/",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in path.strip("/").split("/"):
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd)
            fd=child
            info=os.fstat(fd)
            if info.st_uid!=0 or info.st_mode&0o022:
                raise ValueError("directory protection")
        result,fd=fd,None
        return result
    finally:
        if fd is not None:
            os.close(fd)
def inspect(fd,request,*,readonly=False):
    before=os.fstat(fd)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid!=0
            or before.st_size!=request["size"] or before.st_mode&0o022
            or (readonly and before.st_mode&0o222)):
        raise ValueError("input protection or size")
    digest=hashlib.sha256()
    size=0
    while True:
        block=os.read(fd,65536)
        if not block:
            break
        size+=len(block)
        if size>request["size"] or size>request["limit"]:
            raise ValueError("size")
        digest.update(block)
    if (fingerprint(before)!=fingerprint(os.fstat(fd)) or size!=request["size"]
            or digest.hexdigest()!=request["sha256"]):
        raise ValueError("input changed or digest")
    return before
def install(request):
    ceilings={"mp4_capture_v1":__CAPTURE_MAX_BYTES__,"mp4_output_v1":__MP4_OUTPUT_MAX_BYTES__}
    limit=request["limit"]
    if (request["policy"] not in ceilings or type(limit) is not int
            or not 0<limit<=ceilings[request["policy"]]
            or type(request["size"]) is not int or not 0<request["size"]<=limit):
        raise ValueError("media limit")
    for field,length in (("identity",64),("stage_id",32),("sha256",64)):
        value=request[field]
        if not isinstance(value,str) or len(value)!=length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("identity")
    incoming=target_dir=source=target=None
    try:
        incoming=protected_directory(INCOMING)
        if os.fstat(incoming).st_mode&0o077:
            raise ValueError("incoming protection")
        target_dir=protected_directory(INPUTS)
        source=os.open(request["stage_id"],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=incoming)
        info=inspect(source,request)
        entry=os.stat(request["stage_id"],dir_fd=incoming,follow_symlinks=False)
        if fingerprint(info)!=fingerprint(entry):
            raise ValueError("input changed")
        os.fchmod(source,0o444)
        # Root-only incoming and root-owned target parents prevent the model
        # replacing either directory entry. Exclusive link never overwrites.
        try:
            os.link(request["stage_id"],request["identity"],src_dir_fd=incoming,
                    dst_dir_fd=target_dir,follow_symlinks=False)
        except FileExistsError:
            pass
        target=os.open(request["identity"],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=target_dir)
        inspect(target,request,readonly=True)
        os.unlink(request["stage_id"],dir_fd=incoming)
        return {"path":INPUTS+"/"+request["identity"],"sha256":request["sha256"],"size_bytes":request["size"]}
    finally:
        for fd in (target,source,target_dir,incoming):
            if fd is not None:
                os.close(fd)
        # Failure deliberately does not claim remote cleanup. The owner must
        # retire the allocation, never retry an unobserved upload/installation.
if __name__=="__main__":
    print(json.dumps(install(json.loads(sys.argv[1]))))
'''.replace("__CAPTURE_MAX_BYTES__", str(CAPTURE_MAX_BYTES)).replace("__MP4_OUTPUT_MAX_BYTES__", str(MP4_OUTPUT_MAX_BYTES))

# A descriptor walk prevents symlink/parent swaps. The reader drops to the
# execution account before touching workspace entries, rejects devices/FIFOs,
# and reads at most the supplied limit+1 even if the file grows concurrently.
READ_OUTPUT = r'''
import base64,hashlib,json,os,pwd,stat,sys
request=json.loads(sys.argv[1])
account=pwd.getpwnam("jarvis")
os.setgroups([])
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
parts=request["path"].split("/")
if not parts or any(part in {"","..","."} for part in parts):
    raise ValueError("path")
fd=os.open("/home/jarvis/workspace",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
    for part in parts[:-1]:
        new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        os.close(fd)
        fd=new
    leaf=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
finally:
    os.close(fd)
with os.fdopen(leaf,"rb") as stream:
    info=os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size>request["limit"]:
        raise ValueError("size or type")
    content=stream.read(request["limit"]+1)
    after=os.fstat(stream.fileno())
    if any(getattr(info,key)!=getattr(after,key) for key in
           ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")):
        raise ValueError("file changed during read")
if len(content)>request["limit"]:
    raise ValueError("size")
print(json.dumps({"sha256":hashlib.sha256(content).hexdigest(),"size_bytes":len(content),
                  "content_base64":base64.b64encode(content).decode("ascii")}))
'''

# Fixed trusted code, not a model-supplied command. Keep the real/saved UID root
# while temporarily dropping the effective UID for the descriptor walk/read.
# The workspace user cannot access the root-private output directory or ptrace
# this root-owned process. No untrusted code executes with either identity.
_OUTPUT_STAGE = r'''
import hashlib,json,os,pwd,stat,sys
PARENT="/var/lib/jarvis"
WORKSPACE="/home/jarvis/workspace"
def stage_directory(identity):
    if not isinstance(identity,str) or len(identity)!=32 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("identity")
    parent=os.open(PARENT,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        try:
            os.mkdir("outgoing",0o700,dir_fd=parent)
        except FileExistsError:
            pass
        directory=os.open("outgoing",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
    finally:
        os.close(parent)
    info=os.fstat(directory)
    if info.st_uid!=0 or info.st_mode&0o077:
        os.close(directory)
        raise ValueError("stage permissions")
    return directory
'''

FREEZE_OUTPUT = _OUTPUT_STAGE + r'''
def freeze(request):
    limit=request["limit"]
    if type(limit) is not int or not 0<limit<=__MP4_OUTPUT_MAX_BYTES__:
        raise ValueError("limit")
    parts=request["path"].split("/")
    if not parts or any(part in {"","..","."} for part in parts):
        raise ValueError("path")
    identity=request["stage_id"]
    directory=stage_directory(identity)
    output=leaf=walk=None
    completed=False
    try:
        output=os.open(identity,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        account=pwd.getpwnam("jarvis")
        os.setgroups([])
        os.setegid(account.pw_gid)
        os.seteuid(account.pw_uid)
        try:
            walk=os.open(WORKSPACE,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            for part in parts[:-1]:
                new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=walk)
                os.close(walk)
                walk=new
            leaf=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=walk)
            before=os.fstat(leaf)
            if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=limit:
                raise ValueError("size or type")
            digest=hashlib.sha256()
            size=0
            while True:
                block=os.read(leaf,min(65536,limit-size+1))
                if not block:
                    break
                size+=len(block)
                if size>limit:
                    raise ValueError("size")
                digest.update(block)
                remaining=memoryview(block)
                while remaining:
                    count=os.write(output,remaining)
                    if count<=0:
                        raise ValueError("write")
                    remaining=remaining[count:]
            after=os.fstat(leaf)
            if size!=before.st_size or any(getattr(before,key)!=getattr(after,key) for key in
                    ("st_dev","st_ino","st_mode","st_size","st_mtime_ns","st_ctime_ns")):
                raise ValueError("file changed during read")
        finally:
            os.seteuid(0)
            os.setegid(0)
        os.fsync(output)
        os.fchmod(output,0o400)
        completed=True
        return {"stage_id":identity,"size_bytes":size,"sha256":digest.hexdigest()}
    finally:
        for descriptor in (leaf,walk,output):
            if descriptor is not None:
                os.close(descriptor)
        if output is not None and not completed:
            os.unlink(identity,dir_fd=directory)
        os.close(directory)
if __name__=="__main__":
    print(json.dumps(freeze(json.loads(sys.argv[1]))))
'''.replace("__MP4_OUTPUT_MAX_BYTES__", str(MP4_OUTPUT_MAX_BYTES))

REMOVE_OUTPUT = _OUTPUT_STAGE + r'''
def remove(request):
    identity=request["stage_id"]
    directory=stage_directory(identity)
    try:
        try:
            os.unlink(identity,dir_fd=directory)
        except FileNotFoundError:
            pass
    finally:
        os.close(directory)
    return {"stage_id":identity,"removed":True}
if __name__=="__main__":
    print(json.dumps(remove(json.loads(sys.argv[1]))))
'''
