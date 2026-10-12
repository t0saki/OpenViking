# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Gateway management: OV identity and independent object-write authorization."""

import asyncio
import copy
import hmac
import inspect
import secrets
import time
from dataclasses import dataclass
from urllib.parse import quote

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from .capture import reset_capture
from .client import VikingError
from .config import PUBLIC_OPENVIKING_URLS, normalize_url
from .models import CaptureReset, KeyRequest, Policy, Upstream
from .proxy import upstream_headers, upstream_url
from .records import RecordKind as K
from .storage import digest

PREFIX = "/api/v1/admin/gateway"
MANAGEMENT_HEADER = "x-openviking-gateway-management-key"


def fail(status, reason, **extra):
    raise HTTPException(status, {"reason": reason, "detail": reason, **extra})


def request_credential(request):
    return request.headers.get("x-api-key") or request.headers.get(
        "authorization", ""
    ).removeprefix("Bearer ")


@dataclass
class Caller:
    account_id: str
    credential: str
    identity: dict | None
    manager: bool
    can_write: bool

    @property
    def role(self):
        return self.identity.get("role") if self.identity else None

    @property
    def user_id(self):
        return self.identity.get("user_id") if self.identity else None

    @property
    def log_user(self):
        return self.user_id if self.role == "user" else None


async def caller(request, *, require_identity=True, write=False):
    state = request.app.state
    configured = state.config.management_key
    supplied = request.headers.get(MANAGEMENT_HEADER)
    manager = bool(
        configured and supplied and hmac.compare_digest(supplied.encode(), configured.encode())
    )
    if supplied is not None and not manager:
        fail(401, "management_key_invalid")
    credential = request_credential(request)
    identity = None
    if credential or require_identity or not manager:
        try:
            identity = await state.viking.identify(credential)
        except VikingError as error:
            if error.reason == "openviking_unavailable":
                raise VikingError("openviking_unreachable", 503) from error
            raise
    account = request.query_params.get("account_id") or (identity or {}).get(
        "account_id", "default"
    )
    # A management key may select any account for objects only. It cannot elevate OV operations.
    if identity and account != identity["account_id"] and identity["role"] != "root":
        if require_identity or not manager:
            fail(403, "other_account")
    can_write = manager or not configured
    if write and not can_write:
        fail(403, "management_key_required")
    return Caller(account, credential, identity, manager, can_write)


async def delete_user_data(app, account_id, user_id):
    from .storage import delete_user_data as delete

    await delete(app.state.management, app.state.store, account_id, user_id)
    return {"deleted": True}


async def delete_account_data(app, account_id):
    from .storage import delete_account_data as delete

    await delete(app.state.management, app.state.store, account_id)
    return {"deleted": True}


async def import_vlm_objects(app, account_id):
    source = app.state.vlm_source
    if source is None:
        fail(404, "vlm_import_unavailable")
    result = source(account_id)
    if inspect.isawaitable(result):
        result = await result
    imported, confirmation = [], []
    for candidate in result.get("upstreams", []):
        payload = dict(candidate)
        identifier = payload.pop("id")
        value = Upstream.model_validate(payload).model_dump()
        previous = await app.state.management.get(account_id, "upstreams", identifier)
        if previous and previous.get("allow_coding_plan"):
            value["allow_coding_plan"] = True
        await app.state.management.save(account_id, "upstreams", identifier, value)
        row = {"id": identifier, "name": value["name"]}
        (
            confirmation if value["coding_plan"] and not value["allow_coding_plan"] else imported
        ).append(row)
    return {
        "imported": imported,
        "needs_confirmation": confirmation,
        "skipped": result.get("skipped", []),
    }


def can_import(app, who):
    return bool(
        app.state.vlm_source
        and (who.manager if app.state.config.management_key else who.role == "root")
    )


def register_management(app):
    router = APIRouter(prefix=PREFIX, tags=["gateway"])
    management, store = app.state.management, app.state.store

    @router.get("/access")
    async def access(request: Request):
        who = await caller(request, require_identity=False)
        warnings = []
        if (
            not app.state.config.management_key
            and normalize_url(app.state.openviking_url) in PUBLIC_OPENVIKING_URLS
        ):
            warnings.append("public_openviking_without_management_key")
        return {
            "managed": bool(app.state.config.management_key),
            "can_write_objects": who.can_write,
            "can_import_vlm": can_import(app, who),
            "can_issue_for_others": who.role in {"root", "admin"}
            and (who.identity or {}).get("auth_mode") != "dev",
            "role": who.role,
            "account_id": who.account_id,
            "user_id": who.user_id,
            "warnings": warnings,
        }

    @router.get("/overview")
    async def overview(request: Request):
        who = await caller(request)
        logs = await management.logs(who.account_id, 10000, user_id=who.log_user)
        return {
            **overview_summary(logs),
            "openviking": app.state.health,
            "log_retention_days": app.state.config.log_retention_days,
        }

    @router.get("/logs")
    async def logs(request: Request, limit: int = 200):
        who = await caller(request)
        return await management.logs(who.account_id, max(1, min(limit, 1000)), user_id=who.log_user)

    @router.get("/guides")
    async def guides(request: Request):
        await caller(request)
        return {
            "base_url": app.state.config.public_url or str(request.base_url).rstrip("/"),
            "public_url_configured": bool(app.state.config.public_url),
        }

    @router.get("/tools")
    async def tools(request: Request):
        who = await caller(request)
        catalog = await app.state.viking.tools(who.credential)
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                **({"annotations": t["annotations"]} if "annotations" in t else {}),
            }
            for t in catalog
        ]

    @router.delete("/users/{user_id}/data")
    async def delete_user(user_id: str, request: Request):
        who = await caller(request)
        if who.role not in {"root", "admin"}:
            fail(403, "forbidden")
        return await delete_user_data(app, who.account_id, user_id)

    @router.delete("/account/data")
    async def delete_account(request: Request):
        who = await caller(request)
        if who.role != "root":
            fail(403, "forbidden")
        return await delete_account_data(app, who.account_id)

    @router.post("/upstreams/import-vlm")
    async def import_vlm(request: Request):
        who = await caller(request, require_identity=False)
        if app.state.vlm_source is None:
            fail(404, "vlm_import_unavailable")
        if not can_import(app, who):
            fail(403, "vlm_import_forbidden")
        return await import_vlm_objects(app, who.account_id)

    @router.get("/{kind}")
    async def list_objects(kind: str, request: Request):
        if kind not in {"upstreams", "policies", "keys"}:
            fail(404, "not_found")
        who = await caller(request, require_identity=kind == "keys")
        values = await management.list(
            who.account_id, kind, user_id=who.log_user if kind == "keys" else None
        )
        return [public_object(kind, value, read_only=not who.can_write) for value in values]

    @router.get("/{kind}/{identifier}")
    async def get_object(kind: str, identifier: str, request: Request):
        if kind not in {"upstreams", "policies", "keys"}:
            fail(404, "not_found")
        who = await caller(request, require_identity=kind == "keys")
        value = await management.get(who.account_id, kind, identifier)
        if not value or kind == "keys" and who.role == "user" and value["user_id"] != who.user_id:
            fail(404, "not_found")
        return public_object(kind, value, read_only=not who.can_write)

    @router.api_route("/{kind}/{identifier}", methods=["PUT", "DELETE"])
    async def change_object(kind: str, identifier: str, request: Request):
        if kind not in {"upstreams", "policies", "keys"}:
            fail(404, "not_found")
        who = await caller(request, require_identity=kind == "keys", write=kind != "keys")
        previous = await management.get(who.account_id, kind, identifier)
        if (
            kind == "keys"
            and who.role == "user"
            and previous
            and previous["user_id"] != who.user_id
        ):
            fail(404, "not_found")
        if request.method == "DELETE":
            if kind != "keys":
                keys = await management.list(who.account_id, "keys")
                count = sum(
                    k["policy_id"] == identifier
                    if kind == "policies"
                    else identifier in k["upstream_ids"]
                    for k in keys
                )
                if count:
                    fail(409, "in_use", key_count=count)
            await management.delete(who.account_id, kind, identifier)
            return {"deleted": True}
        if kind == "keys":
            fail(405, "replacement_key_required")
        try:
            payload = await request.json()
            if kind == "upstreams":
                payload = keep_upstream_secrets(payload, previous or {})
                value = Upstream.model_validate(payload).model_dump()
            else:
                value = Policy.model_validate(payload).model_dump()
        except ValidationError as error:
            fail(
                422,
                "invalid_configuration",
                detail="; ".join(
                    item["msg"] for item in error.errors(include_input=False, include_context=False)
                ),
            )
        except ValueError:
            fail(422, "invalid_configuration")
        return public_object(kind, await management.save(who.account_id, kind, identifier, value))

    @router.post("/keys")
    async def issue_key(request: Request):
        who = await caller(request)
        try:
            value = KeyRequest.model_validate(await request.json())
        except ValidationError as error:
            fail(
                422,
                "invalid_configuration",
                detail="; ".join(
                    item["msg"] for item in error.errors(include_input=False, include_context=False)
                ),
            )
        except ValueError:
            fail(422, "invalid_configuration")
        user_id = value.user_id or who.user_id
        if who.role == "user" and user_id != who.user_id:
            fail(403, "forbidden")
        key = value.openviking_key
        if not key:
            if (
                user_id == who.user_id
                and who.identity["account_id"] == who.account_id
                and (who.role != "root" or who.identity.get("auth_mode") == "dev")
            ):
                key = who.credential
            else:
                rows = await app.state.viking.request(
                    "GET",
                    f"/api/v1/admin/accounts/{quote(who.account_id, safe='')}/users?include_credentials=true",
                    who.credential,
                )
                if isinstance(rows, dict):
                    rows = rows.get("users", [])
                selected = next(
                    (
                        r
                        for r in rows
                        if r["user_id"] == user_id and r.get("role") in {"admin", "user"}
                    ),
                    None,
                )
                if not selected:
                    fail(400, "unknown_user")
                key = selected.get("api_key")
                if not key:
                    fail(409, "user_key_unreadable")
        identity = await app.state.viking.health(key)
        if identity["account_id"] != who.account_id:
            fail(403, "other_account")
        if (value.user_id or who.role == "user") and identity["user_id"] != user_id:
            fail(403, "forbidden")
        if not await management.get(who.account_id, "policies", value.policy_id):
            fail(400, "unknown_policy")
        for identifier in value.upstream_ids:
            if not await management.get(who.account_id, "upstreams", identifier):
                fail(400, "unknown_upstream")
        secret = "ovgw_" + secrets.token_urlsafe(32)
        stored = {
            **value.model_dump(),
            "openviking_key": key,
            "user_id": identity["user_id"],
            "prefix": secret[:12],
            "created_at": time.time(),
        }
        result = await management.save(who.account_id, "keys", digest(secret), stored)
        for protocol in ("anthropic", "chat", "responses"):
            await store.allow_scope(
                digest(who.account_id + "\0" + identity["user_id"] + "\0" + protocol)
            )
        await management.save(
            who.account_id, "users", identity["user_id"], {"user_id": identity["user_id"]}
        )
        return {**public_object("keys", result), "key": secret}

    @router.post("/keys/{identifier}/capture/reset")
    async def reset(identifier: str, data: CaptureReset, request: Request):
        who = await caller(request)
        key = await management.get(who.account_id, "keys", identifier)
        if not key or who.role == "user" and key["user_id"] != who.user_id:
            fail(404, "not_found")
        scope = digest(who.account_id + "\0" + key["user_id"] + "\0" + data.protocol)
        for session in (digest(data.session), data.session):
            if (K.ROOT, "") in await store.replay.read(scope, session, [""]):
                break
        else:
            fail(404, "not_found")
        await reset_capture(store.capture, scope, session, "manual_reset")
        return {"status": "ready", "message": "Capture will resync from the next request history"}

    @router.post("/upstreams/{identifier}/test")
    async def test_upstream(identifier: str, request: Request):
        who = await caller(request, require_identity=False)
        upstream = await management.get(who.account_id, "upstreams", identifier)
        if not upstream:
            fail(404, "not_found")
        url = upstream_url(upstream, "/v1/models")
        try:
            async with app.state.http.get(
                url,
                headers=upstream_headers(upstream, {}),
                timeout=aiohttp.ClientTimeout(total=10),
                allow_redirects=False,
            ) as response:
                status = response.status
                if status in {404, 405}:
                    reason, ok = "models_unavailable", True
                elif status == 429:
                    reason, ok = "rate_limited", True
                elif 200 <= status < 300:
                    try:
                        body = await response.json(content_type=None)
                    except ValueError:
                        body = {}
                    ok = (
                        isinstance(body, dict)
                        and isinstance(body.get("data"), list)
                        and not body.get("error")
                    )
                    reason = "passed" if ok else "upstream_key_rejected"
                else:
                    reason, ok = (
                        (
                            "upstream_key_rejected"
                            if status in {401, 403}
                            else "upstream_unavailable"
                        ),
                        False,
                    )
                return {"ok": ok, "status": status, "reason": reason, "url": url}
        except HTTPException as error:
            # Provider credential configuration must not look like a failed OV login.
            return {
                "ok": False,
                "status": error.status_code,
                "reason": "upstream_key_rejected",
                "url": url,
            }
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {"ok": False, "reason": "upstream_unavailable", "url": url}

    app.include_router(router)


def public_object(kind, value, *, read_only=False):
    value = copy.deepcopy(value)
    if kind == "upstreams":
        value["has_api_key"] = bool(value.pop("api_key", ""))
        headers = value.pop("headers", {})
        if not read_only:
            value["header_names"] = list(headers)
    if kind == "keys":
        value.pop("openviking_key", None)
    return value


def keep_upstream_secrets(payload, previous):
    """Preserve omitted secrets only while the destination remains unchanged.

    A blank ``api_key`` keeps the stored key. ``headers`` omitted keeps every stored header;
    otherwise the submitted names are the new header set, and a blank value keeps the stored
    value for that name (or drops the name when nothing is stored).
    """
    if not isinstance(payload, dict):
        raise ValueError("expected an upstream object")
    same_base = payload.get("base_url", "").rstrip("/") == previous.get("base_url", "").rstrip("/")
    if not payload.get("api_key"):
        payload["api_key"] = previous.get("api_key", "") if same_base else ""
    stored = previous.get("headers", {}) if same_base else {}
    headers = payload.get("headers")
    if headers is None:
        payload["headers"] = stored
    elif isinstance(headers, dict):
        payload["headers"] = {
            name: value or stored[name]
            for name, value in headers.items()
            if value or name in stored
        }
    return payload


def overview_summary(logs):
    """Aggregate newest-first request-log records into the management overview."""
    requests = [log for log in logs if log.get("kind") != "capture"]
    groups = {}
    for label, kinds in (("first_call", {"user"}), ("continuation", {"continuation"})):
        subset = [log for log in requests if log.get("kind") in kinds]
        total = sum(
            log.get("first_upstream_input_tokens", log.get("input_tokens", 0)) for log in subset
        )
        cached = sum(
            log.get("first_upstream_cached_tokens", log.get("cached_tokens", 0)) for log in subset
        )
        hidden_calls = 0
        if label == "continuation":
            total += sum(log.get("hidden_upstream_input_tokens", 0) for log in requests)
            cached += sum(log.get("hidden_upstream_cached_tokens", 0) for log in requests)
            hidden_calls = sum(log.get("hidden_upstream_calls", 0) for log in requests)
        groups[label] = {
            "requests": len(subset) + hidden_calls,
            "input_tokens": total,
            "cached_tokens": cached,
            "cache_hit_ratio": cached / total if total else 0,
        }
    degradations = {}
    for log in logs:
        if reason := log.get("degradation"):
            degradations[reason] = degradations.get(reason, 0) + 1
    # "disabled" means recall was skipped (off, budget spent or query too short).
    recalls = [log for log in requests if log.get("recall_reason", "disabled") != "disabled"]
    capture = {}
    for log in logs:
        if log.get("session") and log.get("capture_status"):
            capture.setdefault((log["session"], log.get("protocol")), log["capture_status"])
    return {
        "requests": len(requests),
        "last_request_at": requests[0].get("time") if requests else None,
        "output_tokens": sum(log.get("output_tokens", 0) for log in requests),
        "cache": groups,
        "degradations": degradations,
        "recall_count": sum(log.get("recall_count", 0) for log in requests),
        "recall_requests": len(recalls),
        "recall_ms": sum(log.get("recall_ms", 0) for log in recalls) / len(recalls)
        if recalls
        else 0,
        "capture_issues": {
            status: sum(1 for value in capture.values() if value == status)
            for status in ("retrying", "paused")
        },
        "sample_limit": 10000,
    }
