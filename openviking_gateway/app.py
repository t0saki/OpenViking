# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Standalone ASGI host: authentication, routing, management and byte streaming."""

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .capture import CaptureWorker
from .client import VikingClient, VikingError
from .config import GatewayConfig, StandaloneGatewayConfig
from .kernel import MemoryKernel
from .management import (
    delete_account_data as delete_account_data,
)
from .management import (
    delete_user_data as delete_user_data,
)
from .management import (
    register_management,
    request_credential,
)
from .proxy import ProxyRequest, filtered_headers
from .storage import ManagementStore, SQLiteKernelStore

logger = logging.getLogger(__name__)


def create_app(
    config: GatewayConfig,
    *,
    openviking_url: str,
    vlm_source=None,
    studio_dir=None,
    background_first_probe=False,
):
    encryption_key = config.encryption_key()
    store = SQLiteKernelStore(config.directory / "kernel.sqlite3", encryption_key)
    management = ManagementStore(config.directory / "management.sqlite3", encryption_key)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await store.initialize()
            await management.initialize()
            async with aiohttp.ClientSession(
                auto_decompress=False,
                trust_env=False,
                connector=aiohttp.TCPConnector(limit=1024, limit_per_host=512),
            ) as http:
                app.state.http = http
                app.state.viking = VikingClient(http, openviking_url, config.min_server_version)
                app.state.kernel = MemoryKernel(store, app.state.viking)
                app.state.worker = CaptureWorker(store, management, app.state.viking)
                app.state.health = {"status": "starting"}
                if not background_first_probe:
                    await refresh_health(app)
                if isinstance(config, StandaloneGatewayConfig):
                    mode = app.state.health.get("auth_mode")
                    if (
                        mode == "trusted"
                        or app.state.health.get("reason") == "trusted_mode_unsupported"
                    ):
                        raise ValueError(
                            "OpenViking Gateway does not support trusted authentication"
                        )
                    if config.host not in {"127.0.0.1", "::1", "localhost"} and mode != "api_key":
                        raise ValueError(
                            "A publicly bound OpenViking Gateway requires a successful synchronous "
                            "probe confirming API-key authentication; use a loopback host until "
                            "OpenViking is reachable and its authentication mode is known"
                        )
                task = asyncio.create_task(maintenance(app))
                try:
                    yield
                finally:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        finally:
            store.close()
            management.close()

    app = FastAPI(
        title="OpenViking Gateway",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store, app.state.management, app.state.config = store, management, config
    app.state.openviking_url = openviking_url.rstrip("/")
    app.state.vlm_source = vlm_source

    async def refresh_health(app):
        try:
            app.state.health = await app.state.viking.health(require_identity=False)
            app.state.viking.unavailable_reason = ""
        except VikingError as error:
            app.state.health = {"status": "degraded", "reason": error.reason}
            app.state.viking.unavailable_reason = error.reason

    async def maintenance(app):
        health_at = 0
        cleanup_at = 0
        while True:
            try:
                if time.monotonic() >= health_at:
                    await refresh_health(app)
                    health_at = time.monotonic() + (
                        1
                        if app.state.health.get("status") == "degraded"
                        else config.health_interval_seconds
                    )
                if time.monotonic() >= cleanup_at:
                    await store.expire(time.time() - config.session_ttl_days * 86400)
                    await management.expire_logs(time.time() - config.log_retention_days * 86400)
                    cleanup_at = time.monotonic() + 3600
                if not await app.state.worker.once():
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("OpenViking Gateway maintenance failed")
                await asyncio.sleep(1)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        return JSONResponse(
            {
                "reason": "invalid_configuration",
                "detail": "; ".join(item["msg"] for item in error.errors()),
            },
            status_code=422,
        )

    @app.exception_handler(VikingError)
    async def viking_error(_request, error):
        return JSONResponse(
            {
                "reason": error.reason,
                "error": {"message": error.reason, "reason": error.reason},
                **(
                    {"required_version": config.min_server_version}
                    if error.reason == "openviking_version_mismatch"
                    else {}
                ),
            },
            status_code=error.status,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, error):
        detail = (
            error.detail
            if isinstance(error.detail, dict)
            else {"reason": "gateway_http_" + str(error.status_code), "detail": error.detail}
        )
        return JSONResponse(detail, status_code=error.status_code, headers=error.headers)

    @app.get("/health")
    async def health(request: Request):
        result = {
            "status": "ok",
            "service": "openviking-gateway",
            "auth_mode": app.state.health.get("auth_mode", "api_key"),
            "openviking_url": app.state.openviking_url,
            "openviking": app.state.health,
        }
        if key := request_credential(request):
            identity = await app.state.viking.identify(key)
            result.update(
                {
                    field: identity[field]
                    for field in ("account_id", "user_id", "role", "auth_mode")
                    if field in identity
                }
            )
        return result

    async def authenticate(request):
        key = request_credential(request)
        if key.startswith("sk-ant-oat"):
            raise HTTPException(403, "Claude subscription OAuth credentials are not supported")
        credential = await management.authenticate(key)
        invalid = "Invalid gateway key. It may have been revoked, or removed because its OpenViking API key was regenerated. Issue a new key in OpenViking Studio → Gateway → Keys."
        if not credential:
            raise HTTPException(401, invalid)
        try:
            identity = await app.state.viking.identify(credential["openviking_key"])
            if (
                identity["account_id"] != credential["account"]
                or identity["user_id"] != credential["user_id"]
            ):
                raise VikingError("openviking_http_401", 401)
        except VikingError as error:
            if error.status == 401:
                await management.delete(credential["account"], "keys", credential["id"])
                raise HTTPException(401, invalid) from error
            raise
        return credential

    register_management(app)

    @app.get("/api/compatible/v1/models")
    @app.get("/api/v3/models")
    @app.get("/v1/models")
    async def models(request: Request):
        credential = await authenticate(request)
        values = await management.list(credential["account"], "upstreams")
        names = sorted(
            {
                model
                for u in values
                if u["id"] in credential["upstream_ids"] and u.get("enabled", True)
                for model in [*u.get("models", []), *u.get("aliases", {})]
                if not credential["models"] or model in credential["models"]
            }
        )
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "openviking-gateway"} for name in names
            ],
        }

    @app.websocket("/api/v3/responses")
    @app.websocket("/v1/responses")
    async def websocket_fallback(websocket: WebSocket):
        await websocket.send_denial_response(
            Response(status_code=426, headers={"Upgrade": "HTTP/1.1"})
        )

    @app.post("/gateway/uploads")
    async def proxy_upload(request: Request):
        token = request.query_params.get("token", "")
        if not token or len(token) > 16384:
            raise HTTPException(400, "A signed upload token is required")
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data;"):
            raise HTTPException(415, "Expected a multipart file upload")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > config.max_body_bytes:
                raise HTTPException(413, "Upload exceeds configured body limit")
        try:
            async with app.state.http.post(
                openviking_url + "/api/v1/resources/temp_upload",
                params={"token": token},
                data=bytes(raw),
                headers={"content-type": content_type, "accept-encoding": "identity"},
                timeout=aiohttp.ClientTimeout(total=120),
                allow_redirects=False,
            ) as response:
                return Response(
                    await response.read(),
                    status_code=response.status,
                    headers=filtered_headers(response.headers),
                )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HTTPException(502, "OpenViking upload is unavailable")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    @app.api_route("/api/v3/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    @app.api_route(
        "/api/compatible/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
    )
    async def proxy(path: str, request: Request):
        credential = await authenticate(request)
        return await ProxyRequest(app, request, request.url.path.lstrip("/"), credential).run()

    if isinstance(config, StandaloneGatewayConfig) or studio_dir is not None:
        register_standalone(app, studio_dir)
    return app


def register_standalone(app, studio_dir=None):
    """Serve Studio and the two read-only OV administration resources it needs."""
    from pathlib import Path
    from urllib.parse import quote, urlencode

    @app.get("/api/v1/admin/accounts")
    @app.get("/api/v1/admin/accounts/{account_id}/users")
    async def administration(request: Request, account_id: str | None = None):
        credential = request_credential(request)
        await app.state.viking.identify(credential)
        path = "/api/v1/admin/accounts"
        if account_id is not None:
            path += "/" + quote(account_id, safe="") + "/users"
        params = dict(request.query_params)
        if account_id is not None:
            params["include_credentials"] = "false"
        if params:
            path += "?" + urlencode(params)
        result = await app.state.viking.request("GET", path, credential)
        return {"status": "ok", "result": result}

    from .cli import studio_directory

    directory = studio_dir or studio_directory()
    root = Path(directory).resolve() if directory else None

    @app.get("/")
    async def home():
        return RedirectResponse("/studio/")

    @app.get("/studio")
    async def studio_redirect():
        return RedirectResponse("/studio/")

    @app.get("/studio/{path:path}")
    async def studio(path: str):
        if root is None or not (root / "index.html").is_file():
            raise HTTPException(503, "OpenViking Studio bundle is unavailable")
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise HTTPException(404)
        if candidate.is_file():
            return FileResponse(candidate)
        if path.startswith("assets/"):
            raise HTTPException(404)
        return FileResponse(root / "index.html")
