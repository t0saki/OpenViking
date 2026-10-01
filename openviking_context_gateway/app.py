# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Standalone ASGI host: authentication, routing, management and byte streaming."""

import asyncio
import contextlib
import copy
import hmac
import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import aiohttp
import orjson
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .client import VikingClient, VikingError
from .config import ContextGatewayConfig
from .kernel import CaptureWorker, MemoryKernel
from .models import KeyRequest, Policy, Upstream
from .protocols import (
    ResponseCapture,
    SSEDecoder,
    enhanced_supported,
    parse_body,
    strip_thinking,
)
from .storage import ManagementStore, SQLiteKernelStore, digest
from .tool_catalog import ToolPolicyConflict, tool_block_reason
from .tool_executor import ToolExecutor
from .tool_loop import ChatToolLoop, ToolLoopError, limit_completion, sse
from .vendors import ARK_PATHS, VendorRateLimit, apply_vendor, ark_url, reserve_vendor

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


def public_object(kind, value):
    value = copy.deepcopy(value)
    if kind == "upstreams":
        value["has_api_key"] = bool(value.pop("api_key", ""))
        value["header_names"] = list(value.pop("headers", {}))
    if kind == "keys":
        value.pop("openviking_key", None)
    return value


def create_app(config: ContextGatewayConfig | None = None):
    if config is None:
        from .cli import load_config

        config = load_config()
    encryption_key, admin_token = config.secrets()
    store = SQLiteKernelStore(config.directory / "kernel.sqlite3", encryption_key)
    management = ManagementStore(config.directory / "management.sqlite3", encryption_key)

    @asynccontextmanager
    async def lifespan(app):
        await store.initialize()
        await management.initialize()
        async with aiohttp.ClientSession(
            auto_decompress=False,
            trust_env=False,
            connector=aiohttp.TCPConnector(limit=1024, limit_per_host=512),
        ) as http:
            app.state.http = http
            app.state.viking = VikingClient(http, config.openviking_url, config.min_server_version)
            app.state.kernel = MemoryKernel(store, app.state.viking)
            app.state.worker = CaptureWorker(store, management, app.state.viking)
            app.state.health = {"status": "starting"}
            await refresh_health(app)
            if app.state.health.get("auth_mode") == "dev" and config.host not in {
                "127.0.0.1",
                "::1",
                "localhost",
            }:
                raise ValueError(
                    "Context Gateway must bind to loopback when OpenViking uses dev authentication"
                )
            task = asyncio.create_task(maintenance(app))
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="OpenViking Context Gateway",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store, app.state.management, app.state.config = store, management, config

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
                    health_at = time.monotonic() + config.health_interval_seconds
                if time.monotonic() >= cleanup_at:
                    await store.expire(time.time() - config.session_ttl_days * 86400)
                    await management.expire_logs(time.time() - config.log_retention_days * 86400)
                    cleanup_at = time.monotonic() + 3600
                if not await app.state.worker.once():
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Context Gateway maintenance failed")
                await asyncio.sleep(1)

    @app.exception_handler(VikingError)
    async def viking_error(_request, error):
        return JSONResponse({"error": {"message": error.reason}}, status_code=error.status)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "context-gateway", "openviking": app.state.health}

    async def authenticate(request):
        if app.state.health.get("auth_mode") == "dev" and config.host not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise HTTPException(503, "Dev authentication requires a loopback gateway")
        key = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        if key.startswith("sk-ant-oat"):
            raise HTTPException(403, "Claude subscription OAuth credentials are not supported")
        credential = await management.authenticate(key)
        if not credential:
            raise HTTPException(401, "Invalid or revoked Context Gateway key")
        return credential

    def admin_account(request):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, admin_token):
            raise HTTPException(401, "Invalid gateway management credential")
        account = request.headers.get("x-openviking-account", "")
        if not account:
            raise HTTPException(400, "Management requires a verified account")
        return account

    @app.get("/admin/overview")
    async def overview(request: Request):
        account = admin_account(request)
        logs = await management.logs(account, 10000)
        groups = {}
        for label, kinds in (("first_call", {"user"}), ("continuation", {"continuation"})):
            subset = [log for log in logs if log.get("kind") in kinds]
            total = sum(
                log.get("first_upstream_input_tokens", log.get("input_tokens", 0)) for log in subset
            )
            cached = sum(
                log.get("first_upstream_cached_tokens", log.get("cached_tokens", 0))
                for log in subset
            )
            hidden_calls = 0
            if label == "continuation":
                total += sum(log.get("hidden_upstream_input_tokens", 0) for log in logs)
                cached += sum(log.get("hidden_upstream_cached_tokens", 0) for log in logs)
                hidden_calls = sum(log.get("hidden_upstream_calls", 0) for log in logs)
            groups[label] = {
                "requests": len(subset) + hidden_calls,
                "input_tokens": total,
                "cached_tokens": cached,
                "cache_hit_ratio": cached / total if total else 0,
            }
        reasons = {}
        for log in logs:
            reason = log.get("degradation")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "requests": len(logs),
            "output_tokens": sum(log.get("output_tokens", 0) for log in logs),
            "cache": groups,
            "degradations": reasons,
            "openviking": app.state.health,
            "recall_count": sum(log.get("recall_count", 0) for log in logs),
            "recall_ms": sum(log.get("recall_ms", 0) for log in logs) / max(1, len(logs)),
            "sample_limit": 10000,
        }

    @app.get("/admin/logs")
    async def logs(request: Request, limit: int = 200):
        return await management.logs(admin_account(request), max(1, min(limit, 1000)))

    @app.get("/admin/guides")
    async def guides(request: Request):
        admin_account(request)
        from .guides import connection_guides

        return connection_guides(config.public_url or config.url)

    @app.get("/admin/{kind}")
    async def list_objects(kind: str, request: Request):
        if kind not in {"upstreams", "policies", "keys"}:
            raise HTTPException(404)
        return [public_object(kind, v) for v in await management.list(admin_account(request), kind)]

    @app.api_route("/admin/{kind}/{identifier}", methods=["PUT", "DELETE"])
    async def change_object(kind: str, identifier: str, request: Request):
        account = admin_account(request)
        if kind not in {"upstreams", "policies", "keys"}:
            raise HTTPException(404)
        if request.method == "DELETE":
            if kind != "keys":
                keys = await management.list(account, "keys")
                if any(
                    k["policy_id"] == identifier
                    if kind == "policies"
                    else identifier in k["upstream_ids"]
                    for k in keys
                ):
                    raise HTTPException(
                        409, "Revoke or update dependent keys before deleting this object"
                    )
            await management.delete(account, kind, identifier)
            return {"deleted": True}
        if kind == "keys":
            raise HTTPException(405, "Issue a replacement key using POST /admin/keys")
        try:
            payload = await request.json()
            previous = await management.get(account, kind, identifier)
            if kind == "upstreams":
                if previous:
                    for field in ("api_key", "headers"):
                        if not payload.get(field):
                            payload[field] = previous.get(field, "" if field == "api_key" else {})
                value = Upstream.model_validate(payload).model_dump()
            else:
                value = Policy.model_validate(payload).model_dump()
        except (ValidationError, ValueError):
            # Validation messages can contain the submitted secret. Do not echo input.
            raise HTTPException(
                422, "Invalid gateway configuration; check field names, types and limits"
            )
        return public_object(kind, await management.save(account, kind, identifier, value))

    @app.post("/admin/keys")
    async def issue_key(request: Request):
        account = admin_account(request)
        try:
            value = KeyRequest.model_validate(await request.json())
        except (ValidationError, ValueError):
            raise HTTPException(422, "Invalid key configuration")
        identity = await app.state.viking.health(value.openviking_key)
        if identity["account_id"] != account:
            raise HTTPException(403, "OpenViking key belongs to another account")
        if not await management.get(account, "policies", value.policy_id):
            raise HTTPException(400, "Unknown context policy")
        for identifier in value.upstream_ids:
            if not await management.get(account, "upstreams", identifier):
                raise HTTPException(400, "Unknown upstream")
        secret = "ovcg_" + secrets.token_urlsafe(32)
        stored = {
            **value.model_dump(),
            "user_id": identity["user_id"],
            "prefix": secret[:12],
            "created_at": time.time(),
        }
        result = await management.save(account, "keys", digest(secret), stored)
        for protocol in ("anthropic", "chat", "responses"):
            await store.allow_scope(digest(account + "\0" + identity["user_id"] + "\0" + protocol))
        await management.save(
            account, "users", identity["user_id"], {"user_id": identity["user_id"]}
        )
        return {**public_object("keys", result), "key": secret}

    @app.post("/admin/upstreams/{identifier}/test")
    async def test_upstream(identifier: str, request: Request):
        upstream = await management.get(admin_account(request), "upstreams", identifier)
        if not upstream:
            raise HTTPException(404)
        try:
            async with app.state.http.get(
                upstream_url(upstream, "/v1/models"),
                headers=upstream_headers(upstream, {}),
                timeout=aiohttp.ClientTimeout(total=10),
                allow_redirects=False,
            ) as response:
                return {"ok": response.status < 400, "status": response.status}
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {"ok": False, "reason": "upstream_unavailable"}

    @app.delete("/admin/users/{user_id}/data")
    async def delete_user_data(user_id: str, request: Request):
        account = admin_account(request)
        for key in await management.list(account, "keys"):
            if key["user_id"] == user_id:
                await management.delete(account, "keys", key["id"])
        for protocol in ("anthropic", "chat", "responses"):
            await store.expire(0, digest(account + "\0" + user_id + "\0" + protocol))
        await management.delete(account, "users", user_id)
        return {"deleted": True}

    @app.delete("/admin/account/data")
    async def delete_account_data(request: Request):
        account = admin_account(request)
        users = {user["user_id"] for user in await management.list(account, "users")}
        for user in users:
            await delete_user_data(user, request)
        await management.delete_account(account)
        return {"deleted": True}

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
                {"id": name, "object": "model", "owned_by": "context-gateway"} for name in names
            ],
        }

    @app.websocket("/api/v3/responses")
    @app.websocket("/v1/responses")
    async def websocket_fallback(websocket: WebSocket):
        await websocket.send_denial_response(
            Response(status_code=426, headers={"Upgrade": "HTTP/1.1"})
        )

    @app.post("/context-gateway/uploads")
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
                config.openviking_url + "/api/v1/resources/temp_upload",
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

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request):
        credential = await authenticate(request)
        path = "/" + path
        if path.startswith("/api/v3/responses/"):
            path = "/v1/responses/" + path.removeprefix("/api/v3/responses/")
        else:
            path = ARK_PATHS.get(path, path)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return Response(status_code=426, headers={"Upgrade": "HTTP/1.1"})
        if path.startswith("/admin") or path == "/health":
            raise HTTPException(404)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > config.max_body_bytes:
                raise HTTPException(413, "Request exceeds configured body limit")
        raw = bytes(raw)
        body = await asyncio.to_thread(parse_body, raw) if raw else None
        protocol = PATH_PROTOCOL.get(path)
        routing_body = body
        if routing_body is None and raw:
            # Route and enforce model ACLs even when enhancement is unsafe.
            # The original bytes, including wide numbers, remain untouched.
            try:
                routing_body = json.loads(raw)
            except (ValueError, RecursionError):
                pass
        model = routing_body.get("model", "") if isinstance(routing_body, dict) else ""
        if credential["models"] and model and model not in credential["models"]:
            raise HTTPException(403, "Model is not allowed by this key")
        upstreams = [
            u
            for u in await management.list(credential["account"], "upstreams")
            if u["id"] in credential["upstream_ids"]
        ]
        candidates = sorted(
            [u for u in upstreams if matches(u, protocol, model)],
            key=lambda u: (-u["priority"], u["id"]),
        )
        if not candidates:
            raise HTTPException(404, "No allowed upstream matches this protocol and model")
        upstream = candidates[0]
        # Stateful Responses lookup/cancel/delete must use the account that created it.
        if path.startswith("/v1/responses/"):
            response_id = path.split("/")[3]
            mapping = await management.get(
                credential["account"], "responses", digest(credential["id"] + response_id)
            )
            if not mapping:
                raise HTTPException(404, "Unknown response for this key")
            upstream = next((u for u in upstreams if u["id"] == mapping["upstream_id"]), None)
            if not upstream or not upstream.get("enabled", True):
                raise HTTPException(403, "Response upstream is no longer allowed")
        metrics = {"request_id": uuid.uuid4().hex, "kind": "passthrough", "model": model}
        prepared = None
        if (
            protocol
            and body
            and enhanced_supported(body, protocol)
            and not request.headers.get("content-encoding")
        ):
            policy = await management.get(
                credential["account"], "policies", credential["policy_id"]
            )
            if not policy:
                raise HTTPException(403, "Context policy is unavailable")
            try:
                prepared = await app.state.kernel.prepare(
                    body,
                    protocol,
                    dict(request.headers),
                    credential,
                    upstream,
                    policy,
                    path.endswith("count_tokens"),
                )
                pinned = next(
                    (
                        u
                        for u in upstreams
                        if u["id"] == prepared.root["upstream_id"] and matches(u, protocol, model)
                    ),
                    None,
                )
                if pinned:
                    upstream = pinned
                elif prepared.root["upstream_id"] != upstream["id"]:
                    key = "input" if protocol == "responses" else "messages"
                    prepared.body[key] = strip_thinking(prepared.body[key])
                    prepared.metrics["degradation"] = "upstream_changed"
                metrics.update(prepared.metrics)
                body = prepared.body
            except ToolPolicyConflict as error:
                raise HTTPException(409, str(error))
            except Exception:
                # Storage loss is different from recall unavailability. Anthropic
                # cannot verify old signatures without the missing injected prefix.
                logger.exception("Context Gateway preparation failed")
                metrics["degradation"] = "memory_store_failure"
                if protocol == "chat":
                    raise HTTPException(
                        503, "Conversation replay storage is unavailable; retry later"
                    )
                if protocol == "anthropic":
                    body = copy.deepcopy(body)
                    body["messages"] = strip_thinking(body.get("messages", []))
                prepared = None
        elif raw and body is None:
            metrics["degradation"] = "unsafe_json"
        if upstream.get("coding_plan") and not upstream.get("allow_coding_plan"):
            raise HTTPException(
                403, "Coding Plan upstreams are disabled; configure a model API key"
            )
        if prepared and prepared.root.get("tools") and tool_block_reason(body, protocol, upstream):
            raise HTTPException(
                409,
                "This upstream cannot use the session's frozen gateway tools; start a new session",
            )
        if body:
            body = await apply_vendor(body, upstream, prepared, store)
            if prepared:
                prepared.body = body
                metrics.update(prepared.metrics)
            mapped = upstream.get("aliases", {}).get(model)
            if mapped:
                body = {**body, "model": mapped}
                if prepared:
                    prepared.body = body
            if prepared and prepared.root.get("tools"):
                try:
                    limit_completion(body, prepared.root["policy"].get("tool_total_tokens", 100000))
                except ToolLoopError as error:
                    raise HTTPException(error.status, str(error))
            original = await asyncio.to_thread(parse_body, raw)
            if body != original:
                raw = await asyncio.to_thread(orjson.dumps, body)
        target = upstream_url(upstream, path)
        if request.url.query:
            target += "?" + request.url.query
        headers = upstream_headers(upstream, request.headers)
        if raw:
            headers.setdefault("content-type", "application/json")
        metrics["upstream_id"] = upstream["id"]
        started = time.monotonic()
        try:
            await reserve_vendor(upstream, body or {}, prepared, store)
        except VendorRateLimit as error:
            raise HTTPException(429, str(error), headers={"Retry-After": error.retry_after})
        try:
            response = await app.state.http.request(
                request.method,
                target,
                data=raw or None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=min(
                        config.upstream_timeout_seconds,
                        prepared.root["policy"].get("tool_total_seconds", 120),
                    )
                    if prepared and prepared.root.get("tools")
                    else config.upstream_timeout_seconds
                ),
                allow_redirects=False,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HTTPException(502, "Model upstream is unavailable")
        metrics["status"] = response.status
        out_headers = filtered_headers(response.headers)
        capture = ResponseCapture(protocol or upstream["protocol"])

        async def finish():
            metrics.update(capture.usage or {})
            if upstream.get("vendor") == "ark":
                metrics["cache_min_tokens"] = upstream.get("cache_min_tokens", 1024)
                metrics["cache_eligible"] = (
                    metrics.get("input_tokens", 0) >= metrics["cache_min_tokens"]
                )
            metrics["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
            try:
                if (
                    response.status < 300
                    and prepared
                    and await management.get(credential["account"], "keys", credential["id"])
                ):
                    await app.state.kernel.completed(prepared, credential, capture)
                if capture.response_id and (
                    protocol == "responses" or path.startswith("/v1/responses")
                ):
                    await management.save(
                        credential["account"],
                        "responses",
                        digest(credential["id"] + capture.response_id),
                        {"upstream_id": upstream["id"]},
                    )
                await management.log(credential["account"], metrics)
            except Exception:
                logger.exception("Context Gateway response bookkeeping failed")

        if prepared and prepared.root.get("tools") and response.status < 300:
            executor = ToolExecutor(
                app.state.viking,
                store,
                prepared,
                credential,
                config.public_url,
                config.max_body_bytes,
            )
            loop = ChatToolLoop(prepared, executor, store, capture)
            loop.deadline = started + prepared.root["policy"].get("tool_total_seconds", 120)
            out_headers = {
                k.lower(): v
                for k, v in out_headers.items()
                if k.lower() not in {"content-length", "etag", "content-encoding"}
            }

            async def send_tool_continuation(payload, timeout):
                try:
                    await reserve_vendor(upstream, payload, prepared, store)
                except VendorRateLimit as error:
                    raise ToolLoopError(str(error), 429, headers={"Retry-After": error.retry_after})
                return await app.state.http.post(
                    target,
                    data=orjson.dumps(payload),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    allow_redirects=False,
                )

            async def tool_stream():
                finished = False
                try:
                    async for chunk in loop.run(response, send_tool_continuation):
                        yield chunk
                    finished = True
                except (ToolLoopError, aiohttp.ClientError) as error:
                    capture.complete = False
                    metrics["degradation"] = "hidden_tool_loop_failed"
                    yield sse({"error": {"message": str(error), "type": "gateway_tool_error"}})
                    yield b"data: [DONE]\n\n"
                finally:
                    if not finished:
                        capture.complete = False
                    metrics.update(prepared.metrics)
                    await finish()

            if body.get("stream"):
                out_headers["content-type"] = "text/event-stream"
                return StreamingResponse(tool_stream(), headers=out_headers)
            try:
                async for _ in loop.run(response, send_tool_continuation):
                    pass
            except (ToolLoopError, aiohttp.ClientError) as error:
                capture.complete = False
                metrics["degradation"] = "hidden_tool_loop_failed"
                metrics["status"] = getattr(error, "status", 502)
                await finish()
                if isinstance(error, ToolLoopError) and error.content is not None:
                    return Response(
                        error.content,
                        status_code=error.status,
                        headers=filtered_headers(error.headers),
                    )
                raise HTTPException(
                    getattr(error, "status", 502),
                    str(error),
                    headers=getattr(error, "headers", None),
                )
            metrics.update(prepared.metrics)
            await finish()
            return Response(
                orjson.dumps(loop.final), headers=out_headers, media_type="application/json"
            )

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
                                metrics["degradation"] = "capture_parse_failure"
                    finished = True
                finally:
                    response.close()
                    if not finished:
                        capture.complete = False
                    await finish()

            return StreamingResponse(stream(), status_code=response.status, headers=out_headers)
        try:
            content = await response.read()
            if response.status < 300 and not response.headers.get("content-encoding"):
                try:
                    capture.nonstream(orjson.loads(content))
                except (ValueError, TypeError, KeyError):
                    pass
        finally:
            response.close()
        await finish()
        return Response(content, status_code=response.status, headers=out_headers)

    return app
