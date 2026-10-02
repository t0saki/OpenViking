# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Vendor differences without converting or dropping protocol fields."""

import hashlib
from urllib.parse import urlsplit

import orjson

from .records import RecordKind as K

ARK_PATHS = {
    "/api/v3/chat/completions": "/v1/chat/completions",
    "/api/v3/responses": "/v1/responses",
    "/api/compatible/v1/messages": "/v1/messages",
    "/api/compatible/v1/messages/count_tokens": "/v1/messages/count_tokens",
}
STABLE_PARAMETERS = (
    "model",
    "thinking",
    "reasoning_effort",
    "temperature",
    "top_p",
    "seed",
    "tools",
    "system",
    "instructions",
)


def parameter_fingerprint(body):
    return hashlib.sha256(
        orjson.dumps(
            {key: body[key] for key in STABLE_PARAMETERS if key in body},
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()


def ark_url(upstream, path):
    base = upstream["base_url"].rstrip("/")
    base_path = urlsplit(base).path
    if path.startswith("/v1/messages"):
        suffix = path.removeprefix("/v1")
        return (
            base + suffix
            if base_path.endswith("/api/compatible/v1")
            else base + "/api/compatible/v1" + suffix
        )
    suffix = path.removeprefix("/v1")
    return base + suffix if base_path.endswith("/api/v3") else base + "/api/v3" + suffix


async def apply_vendor(body, upstream, prepared, store):
    if upstream.get("vendor") != "ark" or prepared is None:
        return body
    body = dict(body)
    if prepared.protocol in {"chat", "responses"}:
        cache = await store.put(
            prepared.scope,
            prepared.session,
            K.VENDOR,
            "ark",
            {
                "prompt_cache_key": body.get("prompt_cache_key")
                or "ovcg-"
                + hashlib.sha256((prepared.scope + prepared.session).encode()).hexdigest()[:40],
                "parameters": parameter_fingerprint(body),
            },
        )
        body["prompt_cache_key"] = cache["prompt_cache_key"]
        if parameter_fingerprint(body) != cache["parameters"]:
            prepared.metrics["degradation"] = "ark_cache_parameters_changed"
    return body
