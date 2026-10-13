# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Construction of the optional embedded gateway, and the route class /mcp shares."""

from starlette.routing import Match, Route

from openviking.server.config import get_server_url_from_server_data


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
