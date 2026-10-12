# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lifecycle and routing for the optional embedded gateway."""

from contextlib import asynccontextmanager

from starlette.responses import JSONResponse
from starlette.routing import Match, Route, WebSocketRoute

from openviking.server.config import get_server_url_from_server_data

GATEWAY_PREFIXES = ("/v1", "/api/v3", "/api/compatible/v1", "/gateway", "/api/v1/admin/gateway")
_gateway_app = None


def is_gateway_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in GATEWAY_PREFIXES)


def create_embedded_gateway(config, server_config, service_source):
    if not config.enabled:
        return None
    from openviking_gateway.app import create_app

    async def vlm_source(account_id):
        from openviking.server.gateway_vlm import vlm_upstreams

        resolver = service_source().vlm_resolver
        if resolver is None:
            raise RuntimeError("Account VLM configuration is not initialized")
        return vlm_upstreams(await resolver.effective_vlm_config(account_id))

    return create_app(
        config,
        openviking_url=get_server_url_from_server_data(server_config),
        vlm_source=vlm_source,
        background_first_probe=True,
    )


class _GatewayRoute(Route):
    def matches(self, scope):
        match, child_scope = super().matches(scope)
        if match != Match.NONE:
            child_scope["route"] = self
        return match, child_scope


def mount_gateway(app, gateway):
    """Forward untouched scopes, retaining gateway errors and arbitrary HTTP methods."""

    async def reject_websocket(websocket):
        await websocket.send_denial_response(
            JSONResponse(
                {
                    "error": {
                        "message": "Use HTTP streaming for Responses; WebSocket is unsupported."
                    }
                },
                status_code=426,
            )
        )

    for path in ("/v1/responses", "/api/v3/responses"):
        app.routes.append(WebSocketRoute(path, endpoint=reject_websocket))
    for prefix in GATEWAY_PREFIXES:
        app.routes.append(_GatewayRoute(prefix + "/{_gw:path}", endpoint=gateway))


@asynccontextmanager
async def gateway_lifespan(gateway):
    global _gateway_app
    if gateway is None:
        yield
        return
    async with gateway.router.lifespan_context(gateway):
        previous = _gateway_app
        _gateway_app = gateway
        try:
            yield
        finally:
            _gateway_app = previous


async def seed_dev_defaults(gateway):
    if gateway is None:
        return
    from openviking_gateway.management import import_vlm_objects
    from openviking_gateway.models import Policy

    management = gateway.state.management
    if not await management.list("default", "upstreams"):
        await import_vlm_objects(gateway, "default")
    if not await management.get("default", "policies", "default"):
        await management.save("default", "policies", "default", Policy().model_dump())


async def delete_gateway_data(account_id, user_id=None):
    """Durable deletion retries if the embedded gateway is not ready."""
    if _gateway_app is None:
        raise RuntimeError("Embedded gateway is not initialized")
    from openviking_gateway.management import delete_account_data, delete_user_data

    if user_id is None:
        await delete_account_data(_gateway_app, account_id)
    else:
        await delete_user_data(_gateway_app, account_id, user_id)
