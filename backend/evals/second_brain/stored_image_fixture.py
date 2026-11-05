"""Manifested synthetic storage objects using the real configured frame client."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx

from app.errors import JarvisHTTPError
from app.storage.supabase_config import get_supabase_storage_config
from app.visual_memory.models import ValidatedVisualFrame
from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.storage import VisualFrameStorageClient
from app.visual_memory.validation import jpeg_dimensions
from evals.second_brain.hindsight_service import save


async def require_absent_object(client, config, *, bucket, path):
    """Confirm backing-object absence, not immediate purge of every CDN copy."""
    url = (config.supabase_url.rstrip("/")+"/storage/v1/object/authenticated/"
           +quote(bucket,safe="")+"/"+quote(path,safe="/"))
    async with client.stream("GET",url,params={"jarvis_fixture_absence":uuid4().hex},headers={
        "apikey":config.service_role_key,"Authorization":"Bearer "+config.service_role_key,
        "Cache-Control":"no-cache, no-store",
    }) as response:
        if response.status_code not in (400,404):
            raise RuntimeError("Storage did not confirm the synthetic object's absence")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body)>2048:
                raise RuntimeError("Storage absence response exceeded its bound")
        try:
            error = json.loads(body)
        except (ValueError,UnicodeError) as cause:
            raise RuntimeError("Storage absence response was not structured") from cause
        if (not isinstance(error,dict) or error.get("code")!="NoSuchKey"
                or str(error.get("statusCode"))!="404"):
            raise RuntimeError("Storage did not return an exact missing-object code")


@asynccontextmanager
async def stored_image_fixture(fixture, root, settings, *, session_id, images):
    config = get_supabase_storage_config(settings)
    storage = VisualFrameStorageClient(config)
    absence_client = httpx.AsyncClient(timeout=10)
    repository = VisualFrameRepository(fixture.pool)
    identity = uuid4().hex
    manifest = {"format":"jarvis.s04.image_fixture.v1","owner":fixture.owner,
                "session_id":str(session_id),"status":"allocating","objects":[]}
    path = root/("image-objects-"+identity+".json")
    save(path,manifest)
    try:
        records = []
        for index,pixels in enumerate(images):
            visual_session_id = uuid4()
            object_path = storage.build_storage_path(owner_id=fixture.owner,session_id=session_id,
                                                       visual_session_id=visual_session_id,frame_sequence=index)
            item = {"bucket":storage.bucket,"path":object_path,"phase":"planned",
                    "sha256":hashlib.sha256(pixels).hexdigest(),"delete_allowed":False}
            manifest["objects"].append(item)
            save(path,manifest)
            # Refuse an existing object before claiming cleanup authority.
            await require_absent_object(absence_client,config,bucket=storage.bucket,path=object_path)
            item.update(phase="uploading",delete_allowed=True)
            save(path,manifest)
            now = datetime.now(UTC)
            width,height = jpeg_dimensions(pixels)
            frame = await repository.create_pending_frame(
                owner_id=fixture.owner,session_id=session_id,audio_session_id=None,
                visual_session_id=visual_session_id,
                frame=ValidatedVisualFrame(index,now,"glasses","image/jpeg",width,height,pixels),
                server_received_at=now,storage_bucket=storage.bucket,storage_path=object_path,
                raw_image_expires_at=now+timedelta(minutes=5),
            )
            try:
                stored = await storage.upload_frame(path=object_path,image_bytes=pixels,content_type="image/jpeg")
            except JarvisHTTPError as error:
                if error.details.get("status_code")==409:
                    item["delete_allowed"]=False
                    save(path,manifest)
                raise
            frame = await repository.mark_frame_available(owner_id=fixture.owner,frame_id=frame.id,stored_object=stored)
            item.update(phase="available",frame_id=str(frame.id))
            save(path,manifest)
            records.append(frame)
        manifest["status"]="available"
        save(path,manifest)
        yield records
    finally:
        failures = []
        try:
            for item in manifest["objects"]:
                if not item["delete_allowed"]:
                    continue
                try:
                    expected = "owners/"+fixture.owner.replace("/","_")+"/sessions/"+str(session_id)+"/frames/"
                    if not item["path"].startswith(expected) or item["bucket"]!=storage.bucket:
                        raise RuntimeError("Synthetic storage cleanup scope mismatch")
                    if item.get("frame_id"):
                        async with fixture.pool.acquire() as connection:
                            await connection.execute(
                                "update visual_frames set raw_image_status='deleted' where owner_id=$1 and id=$2",
                                fixture.owner,UUID(item["frame_id"]),
                            )
                    await storage.delete_objects(bucket=item["bucket"],paths=[item["path"]])
                    await require_absent_object(absence_client,config,bucket=item["bucket"],path=item["path"])
                    item.update(phase="deleted",backing_object_absence_verified=True)
                except Exception as error:
                    failures.append(type(error).__name__)
                    item.update(phase="cleanup_unverified",cleanup_error_type=type(error).__name__)
                save(path,manifest)
            manifest["status"]="cleanup_unverified" if failures else "cleaned"
            if failures:
                raise RuntimeError("Synthetic storage cleanup incomplete: "+", ".join(failures))
        except BaseException:
            manifest["status"]="cleanup_unverified"
            raise
        finally:
            save(path,manifest)
            await absence_client.aclose()
            await storage.close()
