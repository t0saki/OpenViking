# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""HTTP routing, forwarding and response observation for one gateway request."""

import asyncio
import json
import logging
import time
import uuid
from urllib.parse import urlsplit

import aiohttp
import orjson
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from .protocols import ResponseCapture, SSEDecoder, enhanced_supported, parse_body
from .storage import digest
from .tool_executor import ToolExecutor
from .tool_loop import ChatToolLoop, ToolLoopError, sse
from .vendors import ARK_PATHS, apply_vendor, ark_url

logger = logging.getLogger(__name__)
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
PATH_PROTOCOL = {
    "/v1/messages": "anthropic",
    "/v1/messages/count_tokens": "anthropic",
    "/v1/chat/completions": "chat",
    "/v1/responses": "responses",
}


def filtered_headers(headers, request=False):
    blocked = HOP_HEADERS | {
        part.strip().lower() for part in headers.get("connection", "").split(",")
    }
    if request:
        blocked |= {
            "host",
            "content-length",
            "authorization",
            "x-api-key",
            "cookie",
            "accept-encoding",
        }
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in blocked and not (request and k.lower().startswith("x-openviking-"))
    }


def upstream_url(upstream, path):
    if upstream.get("vendor") == "ark":
        return ark_url(upstream, path)
    base = upstream["base_url"].rstrip("/")
    # Accept both https://host and https://host/v1 as a configured base URL.
    if urlsplit(base).path.endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    return base + path


def upstream_headers(upstream, incoming):
    headers = filtered_headers(incoming, request=True)
    headers.update(upstream.get("headers", {}))
    key = (
        upstream.get("api_key", "")
        if upstream.get("auth_mode") != "passthrough"
        else incoming.get("x-openviking-upstream-key", "")
    )
    if key.startswith("sk-ant-oat"):
        raise HTTPException(
            403, "Claude subscription OAuth credentials are not supported; use an API key"
        )
    if not key:
        raise HTTPException(401, "Upstream API key is missing")
    if upstream["protocol"] == "anthropic":
        headers["x-api-key"] = key
    else:
        headers["authorization"] = "Bearer " + key
    headers["accept-encoding"] = "identity"
    return headers


def matches(upstream, protocol, model):
    return (
        upstream.get("enabled", True)
        and (not protocol or upstream["protocol"] == protocol)
        and (
            not model
            or not upstream.get("models")
            or model in upstream["models"]
            or model in upstream.get("aliases", {})
        )
    )


class ProxyRequest:
    def __init__(self, app, request, path, credential):
        self.app, self.request, self.credential = app, request, credential
        self.config, self.store, self.management = (
            app.state.config,
            app.state.store,
            app.state.management,
        )
        self.path = "/" + path
        if self.path.startswith("/api/v3/responses/"):
            self.path = "/v1/responses/" + self.path.removeprefix("/api/v3/responses/")
        self.path = ARK_PATHS.get(self.path, self.path)
        self.protocol = PATH_PROTOCOL.get(self.path)
        self.prepared = None
        self.metrics = {"request_id": uuid.uuid4().hex, "kind": "passthrough"}

    async def run(self):
        if self.request.headers.get("upgrade", "").lower() == "websocket":
            return Response(status_code=426, headers={"Upgrade": "HTTP/1.1"})
        if self.path.startswith("/admin") or self.path == "/health":
            raise HTTPException(404)
        raw = bytearray()
        async for chunk in self.request.stream():
            raw.extend(chunk)
            if len(raw) > self.config.max_body_bytes:
                raise HTTPException(413, "Request exceeds configured body limit")
        self.raw = bytes(raw)
        self.body = await asyncio.to_thread(parse_body, self.raw) if raw else None
        original = self.body
        await self.route()
        await self.prepare()
        if self.body != original:
            self.raw = await asyncio.to_thread(orjson.dumps, self.body)
        self.target = upstream_url(self.upstream, self.path)
        if self.request.url.query:
            self.target += "?" + self.request.url.query
        self.headers = upstream_headers(self.upstream, self.request.headers)
        if self.raw:
            self.headers.setdefault("content-type", "application/json")
        self.started = time.monotonic()
        try:
            self.response = await self.app.state.http.request(
                self.request.method,
                self.target,
                data=self.raw or None,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.upstream_timeout_seconds),
                allow_redirects=False,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HTTPException(502, "Model upstream is unavailable")
        self.metrics.update(status=self.response.status, upstream_id=self.upstream["id"])
        self.capture = ResponseCapture(self.protocol or self.upstream["protocol"])
        if self.prepared and self.prepared.tools_active and self.response.status < 300:
            return await self.tool_response()
        return await self.observe_response()

    async def route(self):
        routing = self.body
        if routing is None and self.raw:
            # Unsafe numbers/duplicate keys can still be forwarded byte for byte;
            # the model ACL must also apply on this uncommon fallback path.
            try:
                routing = json.loads(self.raw)
            except (ValueError, RecursionError):
                pass
        self.model = routing.get("model", "") if isinstance(routing, dict) else ""
        if self.credential["models"] and self.model and self.model not in self.credential["models"]:
            raise HTTPException(403, "Model is not allowed by this key")
        self.metrics["model"] = self.model
        self.upstreams = [
            u
            for u in await self.management.list(self.credential["account"], "upstreams")
            if u["id"] in self.credential["upstream_ids"]
        ]
        candidates = sorted(
            [u for u in self.upstreams if matches(u, self.protocol, self.model)],
            key=lambda u: (-u["priority"], u["id"]),
        )
        if not candidates:
            raise HTTPException(404, "No allowed upstream matches this protocol and model")
        self.upstream = candidates[0]
        if self.path.startswith("/v1/responses/"):
            identifier = digest(self.credential["id"] + self.path.split("/")[3])
            mapping = await self.management.get(self.credential["account"], "responses", identifier)
            if not mapping:
                raise HTTPException(404, "Unknown response for this key")
            self.upstream = next(
                (u for u in self.upstreams if u["id"] == mapping["upstream_id"]), None
            )
            if not self.upstream or not self.upstream.get("enabled", True):
                raise HTTPException(403, "Response upstream is no longer allowed")
        # Pinned upstream selection in the kernel is restricted to allowed models.
        self.candidates = candidates

    async def prepare(self):
        if (
            self.protocol
            and self.body
            and enhanced_supported(self.body, self.protocol)
            and not self.request.headers.get("content-encoding")
        ):
            policy = await self.management.get(
                self.credential["account"], "policies", self.credential["policy_id"]
            )
            if not policy:
                raise HTTPException(403, "Context policy is unavailable")
            self.body, self.prepared, metrics = await self.app.state.kernel.enhance(
                self.body,
                self.protocol,
                dict(self.request.headers),
                self.credential,
                self.upstream,
                policy,
                self.path.endswith("count_tokens"),
                self.candidates,
            )
            self.metrics.update(metrics)
            if self.prepared:
                self.upstream = self.prepared.upstream
        elif self.raw and self.body is None:
            self.metrics["degradation"] = "unsafe_json"
        if self.upstream.get("coding_plan") and not self.upstream.get("allow_coding_plan"):
            raise HTTPException(
                403, "Coding Plan upstreams are disabled; configure a model API key"
            )
        if self.body:
            self.body = await apply_vendor(self.body, self.upstream, self.prepared, self.store)
            mapped = self.upstream.get("aliases", {}).get(self.model)
            if mapped:
                self.body = {**self.body, "model": mapped}
            if self.prepared:
                self.prepared.body = self.body
                self.metrics.update(self.prepared.metrics)

    async def finish(self):
        self.metrics.update(self.capture.usage or {})
        if self.upstream.get("vendor") == "ark":
            self.metrics["cache_min_tokens"] = self.upstream.get("cache_min_tokens", 1024)
            self.metrics["cache_eligible"] = (
                self.metrics.get("input_tokens", 0) >= self.metrics["cache_min_tokens"]
            )
        self.metrics["duration_ms"] = round((time.monotonic() - self.started) * 1000, 2)
        try:
            if (
                self.response.status < 300
                and self.prepared
                and await self.management.get(
                    self.credential["account"], "keys", self.credential["id"]
                )
            ):
                await self.app.state.kernel.completed(self.prepared, self.credential, self.capture)
            if (
                self.capture.response_id
                and self.path == "/v1/responses"
                and self.body
                and self.body.get("store") is not False
            ):
                await self.management.save(
                    self.credential["account"],
                    "responses",
                    digest(self.credential["id"] + self.capture.response_id),
                    {"upstream_id": self.upstream["id"]},
                    ttl=self.config.response_ttl_seconds,
                )
            await self.management.log(self.credential["account"], self.metrics)
        except Exception:
            logger.exception("Context Gateway response bookkeeping failed")

    async def tool_response(self):
        response, prepared, capture = self.response, self.prepared, self.capture
        executor = ToolExecutor(
            self.app.state.viking,
            self.store,
            prepared,
            self.credential,
            self.config.public_url,
            self.config.max_body_bytes,
        )
        loop = ChatToolLoop(prepared, executor, self.store, capture)
        headers = {
            k.lower(): v
            for k, v in filtered_headers(response.headers).items()
            if k.lower() not in {"content-length", "etag", "content-encoding"}
        }

        async def send(payload, timeout):
            return await self.app.state.http.post(
                self.target,
                data=orjson.dumps(payload),
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=False,
            )

        async def stream():
            finished = False
            try:
                async for chunk in loop.run(response, send):
                    yield chunk
                finished = True
            except (ToolLoopError, aiohttp.ClientError) as error:
                self.metrics["degradation"] = "hidden_tool_loop_failed"
                yield sse({"error": {"message": str(error), "type": "gateway_tool_error"}})
                yield b"data: [DONE]\n\n"
            finally:
                if not finished:
                    capture.complete = False
                self.metrics.update(prepared.metrics)
                await self.finish()

        if self.body.get("stream"):
            headers["content-type"] = "text/event-stream"
            return StreamingResponse(stream(), headers=headers)
        try:
            async for _ in loop.run(response, send):
                pass
        except (ToolLoopError, aiohttp.ClientError) as error:
            capture.complete = False
            self.metrics.update(
                degradation="hidden_tool_loop_failed", status=getattr(error, "status", 502)
            )
            await self.finish()
            if isinstance(error, ToolLoopError) and error.content is not None:
                return Response(
                    error.content, status_code=error.status, headers=filtered_headers(error.headers)
                )
            raise HTTPException(
                getattr(error, "status", 502), str(error), headers=getattr(error, "headers", None)
            )
        self.metrics.update(prepared.metrics)
        await self.finish()
        return Response(orjson.dumps(loop.final), headers=headers, media_type="application/json")

    async def observe_response(self):
        response, capture = self.response, self.capture
        headers = filtered_headers(response.headers)
        if "text/event-stream" in response.headers.get("content-type", ""):

            async def stream():
                decoder = SSEDecoder()
                observe = not response.headers.get("content-encoding")
                finished = False
                try:
                    async for chunk in response.content.iter_any():
                        yield chunk
                        if observe:
                            try:
                                for frame in decoder.feed(chunk):
                                    data = decoder.data(frame)
                                    if data:
                                        capture.event(data)
                            except Exception:
                                observe = False
                                capture.complete = False
                                self.metrics["degradation"] = "capture_parse_failure"
                    finished = True
                finally:
                    response.close()
                    if not finished:
                        capture.complete = False
                    await self.finish()

            return StreamingResponse(stream(), status_code=response.status, headers=headers)
        try:
            content = await response.read()
            if response.status < 300 and not response.headers.get("content-encoding"):
                try:
                    capture.nonstream(orjson.loads(content))
                except (ValueError, TypeError, KeyError):
                    pass
        finally:
            response.close()
        await self.finish()
        return Response(content, status_code=response.status, headers=headers)
