"""Authenticated local-journal scope; not an account system or voice gate."""
import asyncio
import hashlib
import json
from typing import Annotated
from urllib.parse import unquote, urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError


router = APIRouter(tags=["client-context"])


def storage_namespace(database_url, *, schema="public"):
    """Opaque route identity, stable across bearer/password and tunnel changes.

    Include the SQL user because shared Supabase pooler hosts use it to select
    the project. A route change may conservatively lock an old namespace; it
    never automatically unlocks a different account or exposes routing values.
    """
    try:
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.path.strip("/"):
            raise ValueError()
        if not isinstance(schema, str) or not schema or len(schema) > 256 or "\x00" in schema:
            raise ValueError()
        identity = [parsed.hostname.lower(), parsed.port or 5432,
                    unquote(parsed.path), unquote(parsed.username or ""), schema]
        encoded = json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()
    except (ValueError, TypeError, AttributeError):
        raise JarvisHTTPError(503, "client_context_unavailable", "The authenticated storage context is unavailable.") from None
    return "jarvis-storage-v1:" + hashlib.sha256(encoded).hexdigest()


@router.get("/client-context")
async def client_context(request: Request, session_id: UUID,
                         principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)]):
    services = getattr(request.app.state, "runtime_services", None)
    pool = getattr(services, "db_pool", None)
    settings = getattr(services, "settings", None)
    if settings is None or settings.JARVIS_OWNER_ID != principal.owner_id or pool is None:
        raise JarvisHTTPError(503, "client_context_unavailable", "The authenticated source context is not ready.")
    try:
        async with asyncio.timeout(2):
            async with pool.acquire() as connection:
                owned = await connection.fetchrow(
                    "select current_schema() as schema_name from jarvis_sessions where owner_id=$1 and id=$2",
                    principal.owner_id, session_id)
    except (TimeoutError, OSError):
        raise JarvisHTTPError(503, "client_context_unavailable", "The owned session context could not be verified.") from None
    if not owned:
        raise JarvisHTTPError(404, "client_session_unavailable", "The session is outside this authenticated owner scope.")
    namespace = storage_namespace(settings.SUPABASE_DB_URL, schema=owned["schema_name"])
    return JSONResponse({"owner_id": principal.owner_id, "backend_namespace": namespace, "session_id": str(session_id)},
        headers={"Cache-Control": "private, no-store", "Vary": "Authorization", "X-Content-Type-Options": "nosniff"})
