# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Construction and routing for the optional embedded gateway."""

from starlette.routing import Match, Route, WebSocketRoute

from openviking.server.config import get_server_url_from_server_data

GATEWAY_PREFIXES = ("/v1", "/api/v3", "/api/compatible/v1", "/gateway", "/api/v1/admin/gateway")


def is_gateway_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in GATEWAY_PREFIXES)


class ScopedRoute(Route):
    """Expose the selected route through ``scope["route"]``, matching
    ``APIRoute.matches``, so outer observability can resolve the static route template."""

    def matches(self, scope):
        match, child_scope = super().matches(scope)
        if match != Match.NONE:
            child_scope["route"] = self
        return match, child_scope


def create_embedded_gateway(openviking_config, server_config, service_source):
    config = openviking_config.gateway
    if not config.enabled:
        return None
    from openviking.crypto.config import bootstrap_encryption
    from openviking.utils.agfs_utils import _run_coro_blocking
    from openviking_gateway.app import create_app

    async def vlm_source(account_id):
        from openviking.server.gateway_vlm import vlm_upstreams

        resolver = service_source().vlm_resolver
        if resolver is None:
            raise RuntimeError("Account VLM configuration is not initialized")
        return vlm_upstreams(await resolver.effective_vlm_config(account_id))

    # The gateway shares the server's encryption settings; None when OV encryption is disabled.
    encryption = {"encryption": openviking_config.encryption.model_dump()}
    # The embedded gateway ships with this server, so the minimum-version gate never applies.
    return create_app(
        config.model_copy(update={"min_server_version": "0"}),
        openviking_url=get_server_url_from_server_data(server_config),
        vlm_source=vlm_source,
        encryptor=_run_coro_blocking(bootstrap_encryption(encryption)),
    )


def mount_gateway(app, gateway):
    """Forward untouched scopes, retaining gateway errors and arbitrary HTTP methods."""
    # The gateway rejects Responses WebSocket upgrades itself (426 with an Upgrade header).
    for path in ("/v1/responses", "/api/v3/responses"):
        app.routes.append(WebSocketRoute(path, endpoint=gateway))
    for prefix in GATEWAY_PREFIXES:
        app.routes.append(ScopedRoute(prefix + "/{_gw:path}", endpoint=gateway))
