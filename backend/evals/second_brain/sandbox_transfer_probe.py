"""Real E2B source/output transfer boundaries, with existing native cleanup manifests."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from dotenv import dotenv_values

from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxPolicy
from evals.second_brain.e2b_probe import Probe


async def exercise(probe):
    handle=await probe.create(SandboxPolicy(lifetime_seconds=180,command_seconds=20))
    content=b"project,hours\nalpha,2\nbeta,5\nalpha,3\n"
    identity=hashlib.sha256(content).hexdigest()
    receipt=await probe.provider.stage_input(handle,identity,content)
    again=await probe.provider.stage_input(handle,identity,content)
    probe.check("input_retry_reuses_exact_protected_path",receipt==again and receipt["sha256"]==identity)
    code=("import csv,json,pathlib\n"
          "source=pathlib.Path("+repr(receipt["path"])+")\n"
          "totals={}\n"
          "for row in csv.DictReader(source.read_text().splitlines()):\n"
          " totals[row['project']]=totals.get(row['project'],0)+int(row['hours'])\n"
          "pathlib.Path('summary.json').write_text(json.dumps(totals,sort_keys=True))\n"
          "try:\n source.write_bytes(b'altered')\nexcept PermissionError:\n print('protected')\n")
    result=await probe.python(handle,code)
    probe.check("unprivileged_computation_reads_but_cannot_rewrite_input",result.exit_code==0 and "protected" in result.stdout)
    output=await probe.provider.read_workspace_file(handle,"summary.json")
    probe.check("generated_output_readback_is_exact",json.loads(output)=={"alpha":5,"beta":5})
    await probe.python(handle,"import pathlib,os\n"
        "pathlib.Path('link').symlink_to('/etc/passwd')\n"
        "pathlib.Path('parent').symlink_to('/etc',target_is_directory=True)\n"
        "os.mkfifo('pipe')\n"
        "pathlib.Path('oversized').write_bytes(b'x'*1025)\n"
        "pathlib.Path('directory').mkdir()\n")
    for path,limit in (("link",4096),("parent/passwd",4096),("pipe",4096),("oversized",1024),("directory",4096)):
        try:
            await probe.provider.read_workspace_file(handle,path,max_bytes=limit)
        except SandboxError:
            blocked=True
        else:
            blocked=False
        probe.check("blocked_"+path.replace("/","_"),blocked)
    probe.check("input_content_remains_original",await probe.provider.stage_input(handle,identity,content)==receipt)


async def main(args):
    if not args.live:
        raise ValueError("Explicit --live is required")
    key=dotenv_values(args.credential_file).get("E2B_API_KEY")
    provider=E2BProvider(E2BConfiguration(api_key=key))
    probe=Probe(provider,args.manifest_dir)
    print(json.dumps({"manifest":str(probe.path)}),flush=True)
    try:
        await exercise(probe)
    finally:
        await probe.cleanup()
    print(json.dumps({"checks":probe.record["checks"],"cleanup":probe.record.get("cleanup")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--credential-file",type=Path,required=True)
    parser.add_argument("--manifest-dir",type=Path)
    asyncio.run(main(parser.parse_args()))
