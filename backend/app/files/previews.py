"""Small optional derivatives of bytes already fetched for publication/opening.

Listing and thumbnail delivery never read the original object. Preview failure
does not change the original's publication, access or version.
"""
import asyncio
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import tempfile

from PIL import Image, ImageOps

from app.files.local_file import VerifiedLocalFile

MAX_BYTES = 65536
MAX_EDGE = 480


@dataclass(frozen=True)
class Preview:
    jpeg: bytes
    width: int
    height: int

    @property
    def sha256(self):
        return hashlib.sha256(self.jpeg).hexdigest()


def _image(content):
    with Image.open(BytesIO(content)) as source:
        if source.width * source.height > 20_000_000:
            return None
        source.draft("RGB", (MAX_EDGE, MAX_EDGE))
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        for quality in (80, 65, 45):
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            if output.tell() <= MAX_BYTES:
                return Preview(output.getvalue(), image.width, image.height)
    return None


async def build_preview(content, content_type):
    """Bounded best effort. Cancellation still propagates to the caller."""
    try:
        if content_type.startswith("image/") and isinstance(content, bytes) and len(content) <= 8 * 1024 * 1024:
            return await asyncio.to_thread(_image, content)
        if content_type != "video/mp4" or not isinstance(content, VerifiedLocalFile):
            return None
        executable = shutil.which("ffmpeg")
        if executable is None:
            return None
        # Only the already verified private MP4 is decoded; no network inputs.
        # One bounded output file avoids unbounded stdout/stderr buffering.
        with tempfile.TemporaryDirectory(prefix="jarvis-preview-") as directory:
            output = Path(directory) / "frame.jpg"
            process = await asyncio.create_subprocess_exec(executable,
                "-nostdin", "-v", "error", "-threads", "1", "-protocol_whitelist", "file",
                "-i", str(content.path), "-map", "0:v:0", "-frames:v", "1",
                "-vf", "scale=480:480:force_original_aspect_ratio=decrease", "-threads", "1",
                "-q:v", "5", "-fs", "131072", str(output),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                async with asyncio.timeout(8):
                    code = await process.wait()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            if code != 0 or not output.exists() or output.stat().st_size > 131072:
                return None
            return await asyncio.to_thread(_image, output.read_bytes())
    except Exception:
        return None


async def store_preview(connection, owner_id, file_id, content_sha256, preview):
    if preview is None:
        return
    await connection.execute("""update jarvis_file_publications
        set preview_jpeg=$4,preview_sha256=$5,preview_width=$6,preview_height=$7
        where owner_id=$1 and file_id=$2 and content_sha256=$3
            and phase='available' and not delete_requested and preview_jpeg is null""",
        owner_id, file_id, content_sha256, preview.jpeg, preview.sha256, preview.width, preview.height)


async def preview_references(repository, owner_id, files):
    if not files:
        return {}
    async with repository.sources.pool.acquire() as connection:
        rows = await connection.fetch("""select file_id,preview_sha256,preview_width,preview_height,
            octet_length(preview_jpeg) as size_bytes from jarvis_file_publications
            where owner_id=$1 and file_id=any($2::uuid[]) and phase='available'
                and not delete_requested and preview_jpeg is not null""", owner_id, [file.id for file in files])
    from app.files.workspace_access import file_version
    versions = {file.id: file_version(file) for file in files}
    return {row["file_id"]: {"path": f'/files/{row["file_id"]}/thumbnail',
        "version_id": versions[row["file_id"]], "content_type": "image/jpeg",
        "content_sha256": row["preview_sha256"], "size_bytes": row["size_bytes"],
        "width": row["preview_width"], "height": row["preview_height"]} for row in rows}


async def read_preview(repository, owner_id, file_id):
    async with repository.sources.pool.acquire() as connection:
        row = await connection.fetchrow("""select preview_jpeg,preview_sha256
            from jarvis_file_publications where owner_id=$1 and file_id=$2
                and phase='available' and not delete_requested and preview_jpeg is not null""", owner_id, file_id)
    if row is None:
        return None
    data = bytes(row["preview_jpeg"])
    if not 0 < len(data) <= MAX_BYTES or hashlib.sha256(data).hexdigest() != row["preview_sha256"]:
        return None
    return data, row["preview_sha256"]
