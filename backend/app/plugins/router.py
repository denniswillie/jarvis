"""Authenticated account routes; Core owns production runtime registration."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.plugins.companion_marketplace import CompanionPluginService
from app.plugins.policy import PluginPolicyChange
from app.plugins.service import PluginPolicyService


router = APIRouter(prefix="/plugins", tags=["plugins"])
PRIVATE_HEADERS = {
    "Cache-Control": "private, no-store",
    "Vary": "Authorization",
    "X-Content-Type-Options": "nosniff",
}


def get_plugin_policy_service(request: Request) -> PluginPolicyService:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "plugin_policy_service", None)
    if not isinstance(service, PluginPolicyService):
        raise JarvisHTTPError(503, "plugin_catalog_unavailable", "Plugin choices are not available right now.")
    return service


def get_companion_plugin_service(request: Request) -> CompanionPluginService:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "companion_plugin_service", None)
    if not isinstance(service, CompanionPluginService):
        raise JarvisHTTPError(503, "plugin_extensions_unavailable", "Plugin extensions are not available right now.")
    return service


@router.get("")
async def get_plugins(
    request: Request,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
) -> JSONResponse:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "plugin_policy_service", None)
    if isinstance(service, PluginPolicyService):
        catalog = await service.get(owner_id=principal.owner_id)
        return JSONResponse(catalog.model_dump(mode="json"), headers=PRIVATE_HEADERS)
    companion = getattr(runtime, "companion_plugin_service", None)
    if isinstance(companion, CompanionPluginService):
        catalog = companion.get_legacy_catalog(owner_id=principal.owner_id)
        return JSONResponse(catalog.model_dump(mode="json"), headers=PRIVATE_HEADERS)
    raise JarvisHTTPError(503, "plugin_catalog_unavailable", "Plugin choices are not available right now.")


@router.get("/extensions")
async def get_plugin_extensions(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[CompanionPluginService, Depends(get_companion_plugin_service)],
) -> JSONResponse:
    return JSONResponse(service.get_extensions_catalog(owner_id=principal.owner_id), headers=PRIVATE_HEADERS)


@router.post("/extensions/{plugin_id}/install", status_code=202)
async def install_plugin_extension(
    plugin_id: str,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[CompanionPluginService, Depends(get_companion_plugin_service)],
) -> JSONResponse:
    catalog = await service.start_install(owner_id=principal.owner_id, plugin_id=plugin_id)
    return JSONResponse(catalog, status_code=202, headers=PRIVATE_HEADERS)


@router.delete("/extensions/{plugin_id}/install", status_code=202)
async def remove_plugin_extension(
    plugin_id: str,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[CompanionPluginService, Depends(get_companion_plugin_service)],
) -> JSONResponse:
    catalog = await service.start_uninstall(owner_id=principal.owner_id, plugin_id=plugin_id)
    return JSONResponse(catalog, status_code=202, headers=PRIVATE_HEADERS)


@router.put("/policy")
async def put_policy(
    command: PluginPolicyChange,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[PluginPolicyService, Depends(get_plugin_policy_service)],
) -> JSONResponse:
    result = await service.update(owner_id=principal.owner_id, command=command)
    return JSONResponse(result, headers=PRIVATE_HEADERS)
