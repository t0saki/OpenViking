# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Translate account-effective VLM credentials into gateway upstream candidates."""

import re
from urllib.parse import urlsplit

from openviking_cli.utils.config.vlm_config import VLMConfig


def _vendor(base: str) -> str:
    host = (urlsplit(base).hostname or "").lower()
    for suffix, vendor in (
        ("volces.com", "ark"),
        ("bytepluses.com", "byteplus"),
        ("byteplus.com", "byteplus"),
        ("openai.com", "openai"),
        ("anthropic.com", "anthropic"),
        ("deepseek.com", "deepseek"),
    ):
        if host == suffix or host.endswith("." + suffix):
            return vendor
    return "generic"


def vlm_upstreams(config: VLMConfig) -> dict:
    """Produce deterministic candidates; no credentials are written here."""
    from openviking_gateway.models import Upstream

    upstreams, skipped = [], []
    for credential in config.credentials:
        identifier = "vlm-" + credential.id
        settings = config.build_vlm_config_dict_for_credential(credential)
        provider = (settings.get("provider") or "litellm").lower()
        model = settings.get("model") or ""
        base = settings.get("api_base") or ""
        api_key = settings.get("api_key") or ""
        headers = dict(settings.get("extra_headers") or {})
        protocol = "chat"
        reason = None
        if provider == "openai-codex":
            reason = "openai_codex"
        elif provider == "azure" or (provider == "litellm" and model.startswith("azure/")):
            reason = "azure"
        elif provider == "volcengine":
            base = base or "https://ark.cn-beijing.volces.com/api/v3"
            headers = {k: v for k, v in headers.items() if k.lower() != "x-client-request-id"}
        elif provider == "openai":
            base = base or "https://api.openai.com/v1"
        elif provider == "kimi":
            from openviking.models.vlm.backends.kimi_vlm import (
                DEFAULT_KIMI_MODEL,
                DEFAULT_KIMI_USER_AGENT,
                KIMI_LEGACY_MODEL_ALIASES,
                _normalize_kimi_api_base,
            )

            base = _normalize_kimi_api_base(base)
            model = model.strip() or DEFAULT_KIMI_MODEL
            model = KIMI_LEGACY_MODEL_ALIASES.get(model, model)
            if not any(key.lower() == "user-agent" for key in headers):
                headers["User-Agent"] = DEFAULT_KIMI_USER_AGENT
        elif provider == "glm":
            from openviking.models.vlm.backends.glm_vlm import (
                DEFAULT_GLM_API_BASE,
                DEFAULT_GLM_MODEL,
            )

            base = base or DEFAULT_GLM_API_BASE
            model = model or DEFAULT_GLM_MODEL
        elif model.startswith(("ollama/", "ollama_chat/")):
            model = model.split("/", 1)[1]
            base = base or "http://localhost:11434"
            if not urlsplit(base).path.strip("/"):
                base = base.rstrip("/") + "/v1"
            api_key = api_key or "ollama"
        elif model.startswith("anthropic/"):
            model = model.split("/", 1)[1]
            protocol = "anthropic"
            base = base or "https://api.anthropic.com"
        elif not base:
            reason = "litellm_no_api_base"
        if reason:
            skipped.append({"id": identifier, "reason": reason})
            continue
        try:
            upstream = Upstream(
                name=credential.id,
                protocol=protocol,
                base_url=base,
                api_key=api_key,
                headers=headers,
                models=[model] if model else [],
                vendor=_vendor(base),
                coding_plan=bool(re.search(r"(^|/)(coding|plan)(/|$)", urlsplit(base).path)),
            )
        except ValueError:
            # Validation errors can include secrets from the input; never expose them.
            skipped.append({"id": identifier, "reason": "invalid_configuration"})
            continue
        upstreams.append({"id": identifier, **upstream.model_dump()})
    return {"upstreams": upstreams, "skipped": skipped}
