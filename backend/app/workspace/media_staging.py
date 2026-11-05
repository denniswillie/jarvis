"""Fixed materialization program/data for Core's existing privileged wrapper.

No provider, shell wrapper, grant, execution ledger or installation entry point is
exposed to model tools. Core owns allocation/source revalidation, deadlines and
cleanup on partial/uncertain outcomes. Do not retry by choosing a fresh identity.
"""
import hashlib

from app.files.transfer import CAPTURE_MAX_BYTES
from app.workspace.media_render import MediaRenderPlan


def materialization_request(plan: MediaRenderPlan):
    if not isinstance(plan, MediaRenderPlan):
        raise ValueError("A prepared render plan is required")
    staging = plan.staging()
    return {"schema_version": 1, "plan_sha256": plan.digest, "task_id": plan.task_id.hex,
            "source_identity": plan.input_path.rsplit("/", 1)[1],
            "source_sha256": staging["content_sha256"], "size_bytes": staging["size_bytes"],
            "limit_bytes": plan.binding.policy.max_source_bytes,
            "envelope": plan.envelope_bytes.decode("utf-8"),
            "envelope_sha256": hashlib.sha256(plan.envelope_bytes).hexdigest()}


def expected_materialization_receipt(plan: MediaRenderPlan):
    staging = plan.staging()
    return {"format": "jarvis.media-inputs.v1", "plan_sha256": plan.digest,
            "input_file": staging["input_file"], "staging_directory": staging["staging_directory"],
            "source_sha256": staging["content_sha256"], "size_bytes": staging["size_bytes"],
            "envelope_sha256": hashlib.sha256(plan.envelope_bytes).hexdigest(),
            "work_parent": plan.directory, "output_path": plan.output_path}


# Isolated stdlib-only program. Its production entry point fixes account/roots;
# JSON cannot override paths, uid/gid, code or the output name. Internal fd-based
# functions allow actual local filesystem tests without pretending to be root.
INSTALL_RENDER_INPUTS = r'''
import contextlib,hashlib,json,os,pwd,re,stat,sys
INPUTS="/home/jarvis/inputs"
WORKSPACE="/home/jarvis/workspace"
DIR_FLAGS=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
READ_FLAGS=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
KEYS={"schema_version","plan_sha256","task_id","source_identity","source_sha256",
      "size_bytes","limit_bytes","envelope","envelope_sha256"}
def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate_json_key")
        result[key]=value
    return result
def validate(request):
    if type(request) is not dict or set(request)!=KEYS or type(request["schema_version"]) is not int or request["schema_version"]!=1:
        raise ValueError("request_shape")
    for key,length in (("plan_sha256",64),("task_id",32),("source_identity",64),("source_sha256",64),("envelope_sha256",64)):
        if not isinstance(request[key],str) or re.fullmatch("[0-9a-f]{"+str(length)+"}",request[key]) is None:
            raise ValueError("request_identity")
    if (type(request["limit_bytes"]) is not int or not 0<request["limit_bytes"]<=__CAPTURE_MAX_BYTES__
            or type(request["size_bytes"]) is not int or not 0<request["size_bytes"]<=request["limit_bytes"]
            or type(request["envelope"]) is not str):
        raise ValueError("request_limit")
    raw=request["envelope"].encode("utf-8",errors="strict")
    if not 0<len(raw)<=65536 or hashlib.sha256(raw).hexdigest()!=request["envelope_sha256"]:
        raise ValueError("envelope_hash_or_size")
    envelope=json.loads(raw,object_pairs_hook=unique)
    if (not isinstance(envelope,dict) or set(envelope)!={"recipe","sources"}
            or not isinstance(envelope["sources"],list) or len(envelope["sources"])!=1
            or not isinstance(envelope["sources"][0],dict)
            or envelope["sources"][0].get("assetPath")!="inputs/"+request["source_sha256"]+".mp4"
            or envelope["sources"][0].get("contentSha256")!=request["source_sha256"]):
        raise ValueError("envelope_source_mismatch")
    return raw
def fingerprint(info):
    return tuple(getattr(info,key) for key in ("st_dev","st_ino","st_mode","st_uid","st_size","st_mtime_ns","st_ctime_ns"))
def open_root(path,*,leaf_uid):
    fd=os.open("/",DIR_FLAGS)
    try:
        parts=path.strip("/").split("/")
        for index,part in enumerate(parts):
            child=os.open(part,DIR_FLAGS,dir_fd=fd);os.close(fd);fd=child
            info=os.fstat(fd);expected=leaf_uid if index==len(parts)-1 else 0
            if info.st_uid!=expected or info.st_mode&0o022:
                raise ValueError("root_directory_protection")
        result,fd=fd,None
        return result
    finally:
        if fd is not None:os.close(fd)
def open_dir(stack,parent,name,*,uid,mode):
    fd=os.open(name,DIR_FLAGS,dir_fd=parent);stack.callback(os.close,fd)
    info=os.fstat(fd)
    if info.st_uid!=uid or stat.S_IMODE(info.st_mode)!=mode:
        raise ValueError("directory_protection")
    return fd
def read_checked(fd,*,uid,size,sha256,copy_to=None):
    before=os.fstat(fd)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid!=uid or before.st_mode&0o222
            or stat.S_IMODE(before.st_mode)&0o7000 or before.st_size!=size):
        raise ValueError("input_protection_or_size")
    digest=hashlib.sha256();count=0
    while True:
        block=os.read(fd,65536)
        if not block:break
        count+=len(block)
        if count>size:raise ValueError("input_size")
        digest.update(block)
        if copy_to is not None:
            remaining=memoryview(block)
            while remaining:
                written=os.write(copy_to,remaining)
                if written<=0:raise ValueError("input_copy_failed")
                remaining=remaining[written:]
    if fingerprint(before)!=fingerprint(os.fstat(fd)) or count!=size or digest.hexdigest()!=sha256:
        raise ValueError("input_changed_or_digest")
def read_file(stack,parent,name,*,uid,size,sha256):
    fd=os.open(name,READ_FLAGS,dir_fd=parent);stack.callback(os.close,fd)
    if stat.S_IMODE(os.fstat(fd).st_mode)!=0o444:
        raise ValueError("render_input_readability")
    read_checked(fd,uid=uid,size=size,sha256=sha256)
def verify_bundle(stack,inputs,name,request,raw,uid):
    root=open_dir(stack,inputs,name,uid=uid,mode=0o555)
    if set(os.listdir(root))!={"input.json","staging"}:raise ValueError("render_input_contents")
    staged=open_dir(stack,root,"staging",uid=uid,mode=0o555)
    if os.listdir(staged)!=["inputs"]:raise ValueError("render_staging_contents")
    files=open_dir(stack,staged,"inputs",uid=uid,mode=0o555)
    leaf=request["source_sha256"]+".mp4"
    if os.listdir(files)!=[leaf]:raise ValueError("render_source_contents")
    read_file(stack,root,"input.json",uid=uid,size=len(raw),sha256=request["envelope_sha256"])
    read_file(stack,files,leaf,uid=uid,size=request["size_bytes"],sha256=request["source_sha256"])
def create_bundle(stack,inputs,name,source,request,raw,uid):
    # Root-owned inputs prevents replacement of this entry. Failure deliberately
    # leaves a partial directory; no overwrite, repair, success or cleanup claim.
    os.mkdir(name,0o700,dir_fd=inputs)
    root=open_dir(stack,inputs,name,uid=uid,mode=0o700)
    os.mkdir("staging",0o700,dir_fd=root)
    staged=open_dir(stack,root,"staging",uid=uid,mode=0o700)
    os.mkdir("inputs",0o700,dir_fd=staged)
    files=open_dir(stack,staged,"inputs",uid=uid,mode=0o700)
    target=os.open(request["source_sha256"]+".mp4",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=files)
    stack.callback(os.close,target)
    read_checked(source,uid=uid,size=request["size_bytes"],sha256=request["source_sha256"],copy_to=target)
    os.fsync(target);os.fchmod(target,0o444)
    metadata=os.open("input.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=root)
    stack.callback(os.close,metadata)
    view=memoryview(raw)
    while view:
        written=os.write(metadata,view)
        if written<=0:raise ValueError("metadata_copy_failed")
        view=view[written:]
    os.fsync(metadata);os.fchmod(metadata,0o444)
    for directory in (files,staged,root):os.fchmod(directory,0o555)
def work_parent(stack,workspace,parts,uid):
    parent=workspace
    for name in parts:
        try:os.mkdir(name,0o700,dir_fd=parent)
        except FileExistsError:pass
        fd=os.open(name,DIR_FLAGS,dir_fd=parent);stack.callback(os.close,fd)
        info=os.fstat(fd)
        if info.st_uid!=uid or stat.S_IMODE(info.st_mode)!=0o700:
            raise ValueError("workspace_directory_protection")
        parent=fd
    # Never delete or adopt an existing render attempt. The command owner must
    # reconcile its original operation before requesting preparation again.
    try:os.stat("render",dir_fd=parent,follow_symlinks=False)
    except FileNotFoundError:return
    raise ValueError("render_already_present_reconcile_original_command")
def materialize(request,inputs,workspace,*,privileged_uid,worker_uid,worker_gid):
    raw=validate(request)
    with contextlib.ExitStack() as stack:
        source=os.open(request["source_identity"],READ_FLAGS,dir_fd=inputs);stack.callback(os.close,source)
        # Always recheck the original, including identical preparation replay.
        read_checked(source,uid=privileged_uid,size=request["size_bytes"],sha256=request["source_sha256"])
        os.lseek(source,0,os.SEEK_SET)
        name="render-"+request["plan_sha256"]
        try:os.stat(name,dir_fd=inputs,follow_symlinks=False)
        except FileNotFoundError:create_bundle(stack,inputs,name,source,request,raw,privileged_uid)
        verify_bundle(stack,inputs,name,request,raw,privileged_uid)
    # Close root-opened writable file descriptors before dropping authority.
    # Root is needed only for immutable inputs, never workspace mutations.
    if os.geteuid()!=worker_uid:
        os.setgroups([]);os.setgid(worker_gid);os.setuid(worker_uid)
    with contextlib.ExitStack() as stack:
        work_parent(stack,workspace,("media",request["task_id"],request["plan_sha256"]),worker_uid)
    protected=INPUTS+"/"+name
    work=WORKSPACE+"/media/"+request["task_id"]+"/"+request["plan_sha256"]
    return {"format":"jarvis.media-inputs.v1","plan_sha256":request["plan_sha256"],
        "input_file":protected+"/input.json","staging_directory":protected+"/staging",
        "source_sha256":request["source_sha256"],"size_bytes":request["size_bytes"],
        "envelope_sha256":request["envelope_sha256"],"work_parent":work,"output_path":work+"/render/output.mp4"}
def install(request):
    if os.geteuid()!=0:raise ValueError("trusted_root_wrapper_required")
    account=pwd.getpwnam("jarvis")
    if account.pw_uid==0:raise ValueError("unprivileged_renderer_required")
    with contextlib.ExitStack() as stack:
        inputs=open_root(INPUTS,leaf_uid=0);stack.callback(os.close,inputs)
        workspace=open_root(WORKSPACE,leaf_uid=account.pw_uid);stack.callback(os.close,workspace)
        return materialize(request,inputs,workspace,privileged_uid=0,worker_uid=account.pw_uid,worker_gid=account.pw_gid)
if __name__=="__main__":
    if len(sys.argv)!=2 or len(sys.argv[1].encode("utf-8"))>131072:raise ValueError("bounded_request_required")
    print(json.dumps(install(json.loads(sys.argv[1],object_pairs_hook=unique))))
'''.replace("__CAPTURE_MAX_BYTES__", str(CAPTURE_MAX_BYTES))
