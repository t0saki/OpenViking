"""OpenViking fixture with per-library data and inspectable fault controls.

GET /_records exposes requests and libraries; POST /_reset clears data and faults.
POST /_fail accepts on, status, key and path; /_revoke and /_unorder accept key
and enabled. /_auth_mode accepts mode; /_tool_delay accepts seconds. /_keys accepts
a key-to-library mapping. Every library deliberately identifies as default/default.
"""

import argparse
import asyncio
import copy
import json
import os
import time

from aiohttp import web

DEFAULT_KEYS = {"key-a": "library-a", "key-a-new": "library-a", "key-b": "library-b"}
TOOLS = [
    {
        "name": "search",
        "description": "Search this library's memories.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]


def ok(value):
    return web.json_response({"status": "ok", "result": value})


def failure(status, code):
    return web.json_response({"status": "error", "error": {"code": code}}, status=status)


def create_app():
    keys = {**DEFAULT_KEYS, **json.loads(os.environ.get("MOCK_OV_KEYS", "{}"))}
    requests, libraries = [], {}
    revoked, unordered = set(), set()
    control = {"auth_mode": "api_key", "fail": None, "tool_delay": 0}
    counter = 0

    def library(name):
        return libraries.setdefault(
            name, {"sessions": {}, "recalls": [], "tool_calls": [], "archives": {}}
        )

    def identity(key):
        value = keys.get(key)
        if isinstance(value, dict):
            return value.get("library", value.get("account_id", "library-a")), value
        return value, {"account_id": "default", "user_id": "default", "role": "admin"}

    async def controls(request):
        if request.path in {"/_records", "/_events"}:
            records = requests
            for name in ("key", "path", "source_pod", "library"):
                if name in request.query:
                    records = [r for r in records if r.get(name) == request.query[name]]
            if "since" in request.query:
                records = [r for r in records if r["time"] >= float(request.query["since"])]
            selected = libraries
            if "library" in request.query:
                name = request.query["library"]
                selected = {name: libraries[name]} if name in libraries else {}
            return web.json_response(
                {
                    "requests": records,
                    "events": records,
                    "libraries": selected,
                    "control": control,
                    "revoked": sorted(revoked),
                    "unordered": sorted(unordered),
                }
            )
        if request.method != "POST":
            raise web.HTTPMethodNotAllowed(request.method, ["POST"])
        body = await request.json() if request.can_read_body else {}
        if request.path == "/_reset":
            requests.clear()
            libraries.clear()
            revoked.clear()
            unordered.clear()
            control.update(auth_mode="api_key", fail=None, tool_delay=0)
        elif request.path == "/_fail":
            status = int(body.get("status", 503))
            if not 400 <= status <= 599:
                raise web.HTTPBadRequest(text="status must be 400..599")
            control["fail"] = {**body, "status": status} if body.get("on", True) else None
        elif request.path in {"/_revoke", "/_unorder"}:
            selected = revoked if request.path == "/_revoke" else unordered
            (selected.add if body.get("enabled", body.get("on", True)) else selected.discard)(
                body["key"]
            )
        elif request.path == "/_auth_mode":
            mode = body.get("mode", body.get("auth_mode"))
            if mode not in {"api_key", "dev", "trusted"}:
                raise web.HTTPBadRequest(text="mode must be api_key, dev or trusted")
            control["auth_mode"] = mode
        elif request.path in {"/_tool_delay", "/_delay"}:
            seconds = float(body.get("seconds", body.get("tools", 0)))
            if not 0 <= seconds <= 120:
                raise web.HTTPBadRequest(text="seconds must be 0..120")
            control["tool_delay"] = seconds
        elif request.path == "/_keys":
            keys.update(body.get("keys", body))
        else:
            raise web.HTTPNotFound()
        return web.json_response({"status": "ok", **control})

    @web.middleware
    async def record_and_auth(request, handler):
        nonlocal counter
        if request.path.startswith("/_"):
            return await controls(request)
        body = await request.json() if request.can_read_body else {}
        key = request.headers.get("X-API-Key", "")
        name, who = identity(key)
        counter += 1
        record = {
            "id": counter,
            "time": time.time(),
            "source_pod": request.headers.get("X-Source-Pod", request.remote),
            "key": key,
            "library": name,
            "method": request.method,
            "path": request.path,
            "query": dict(request.query),
            "body": body,
            "status": None,
        }
        requests.append(record)
        request["record"], request["body"], request["identity"] = record, body, who
        fault = control["fail"]
        known_path = request.path in {
            "/health",
            "/mcp",
            "/api/v1/system/status",
            "/api/v1/skills",
            "/api/v1/fs/ls",
            "/api/v1/content/read",
            "/api/v1/search/search",
            "/api/v1/sessions",
        } or request.path.startswith("/api/v1/sessions/")
        if (
            fault
            and (not fault.get("key") or fault["key"] == key)
            and (not fault.get("path") or request.path.startswith(fault["path"]))
        ):
            response = failure(fault["status"], "UNAVAILABLE")
        elif not known_path:
            response = failure(404, "NOT_FOUND")
        elif control["auth_mode"] != "dev" and (key in revoked or (not name and key)):
            response = failure(401, "UNAUTHENTICATED")
        elif control["auth_mode"] != "dev" and key in unordered and request.path == "/health":
            response = failure(402, "ProductUnordered")
        elif not name and not key and request.path != "/health" and control["auth_mode"] != "dev":
            response = failure(401, "UNAUTHENTICATED")
        else:
            request["library"] = library(name or "library-a")
            request["library_name"] = name or "library-a"
            try:
                response = await handler(request)
            except web.HTTPException as error:
                record["status"] = error.status
                raise
        record["status"] = response.status
        record["finished_at"] = time.time()
        return response

    def recall_result(request, body):
        name = request["library_name"]
        entry = {
            "uri": f"viking://user/default/memories/{name}.md",
            "text": f"Memory from {name}: blue cluster.",
            "score": 1.0,
        }
        entries = [] if entry["uri"] in body.get("exclude_uris", []) else [entry]
        return {
            "entries": entries,
            "rendered": "\n\n".join(e["uri"] + "\n" + e["text"] for e in entries),
        }

    async def handle(request):
        path, body, data = request.path, request["body"], request["library"]
        if path == "/health":
            result = {
                "status": "ok",
                "healthy": True,
                "version": "0.5.2",
                "auth_mode": control["auth_mode"],
            }
            if request.headers.get("X-API-Key") and identity(request.headers["X-API-Key"])[0]:
                result.update(
                    {
                        k: request["identity"].get(k, default)
                        for k, default in (
                            ("account_id", "default"),
                            ("user_id", "default"),
                            ("role", "admin"),
                        )
                    }
                )
            return web.json_response(result)
        if path == "/api/v1/system/status":
            return ok({"user": "default"})
        if path == "/api/v1/skills":
            return ok({"skills": []})
        if path == "/api/v1/fs/ls":
            return ok(
                [{"name": "default", "isDir": True}]
                if request.query.get("uri") == "viking://user"
                else []
            )
        if path == "/api/v1/search/search":
            result = recall_result(request, body)
            data["recalls"].append({**request["record"], "result": result, "status": 200})
            return ok(result)
        if path == "/api/v1/sessions" and request.method == "POST":
            sid = body["session_id"]
            request["record"]["session"] = sid
            if sid in data["sessions"]:
                return failure(409, "CONFLICT")
            data["sessions"][sid] = {
                "session_id": sid,
                "messages": [],
                "pending": [],
                "commits": [],
                "created_by": request["record"]["key"],
                "create_body": body,
            }
            return ok({"session_id": sid})
        if path.startswith("/api/v1/sessions/"):
            parts = path.split("/")
            sid = parts[4]
            request["record"]["session"] = sid
            session = data["sessions"].get(sid)
            if session is None:
                return failure(404, "NOT_FOUND")
            if path.endswith("/messages/batch") and request.method == "POST":
                session["messages"].extend(copy.deepcopy(body["messages"]))
                session["pending"].extend(copy.deepcopy(body["messages"]))
                request["record"]["count"] = len(body["messages"])
                return ok(
                    {
                        "added": len(body["messages"]),
                        "pending_tokens": len(json.dumps(session["pending"])) // 3,
                    }
                )
            if path.endswith("/commit") and request.method == "POST":
                keep = max(0, int(body.get("keep_recent_count", 0)))
                archived = session["pending"][:-keep] if keep else session["pending"][:]
                session["pending"] = session["pending"][-keep:] if keep else []
                uri = f"viking://user/default/sessions/{sid}/history/archive_{len(session['commits']) + 1:03d}"
                commit = {
                    "archive_uri": uri,
                    "messages": archived,
                    "keep_recent_count": keep,
                    "key": request["record"]["key"],
                    "time": time.time(),
                }
                session["commits"].append(commit)
                data["archives"][uri] = archived
                return ok({"archive_uri": uri})
            if request.method == "GET":
                return ok(
                    {
                        "session_id": sid,
                        "pending_tokens": len(json.dumps(session["pending"])) // 3
                        if session["pending"]
                        else 0,
                        "messages": session["pending"],
                    }
                )
        if path == "/api/v1/content/read":
            uri = request.query.get("uri", "")
            if uri.removesuffix("/.done") in data["archives"] and uri.endswith("/.done"):
                return ok("")
            if uri in data["archives"]:
                return ok(json.dumps(data["archives"][uri]))
            return failure(404, "NOT_FOUND")
        if path == "/mcp":
            method, params = body.get("method"), body.get("params", {})
            request["record"]["rpc"] = method
            if method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                call = {
                    **request["record"],
                    "name": params.get("name"),
                    "arguments": params.get("arguments", {}),
                }
                data["tool_calls"].append(call)
                await asyncio.sleep(control["tool_delay"])
                if params.get("name") != "search":
                    return web.json_response(
                        {
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "error": {"code": -32602, "message": "Unknown tool"},
                        }
                    )
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": recall_result(request, params.get("arguments", {}))["rendered"],
                        }
                    ],
                    "isError": False,
                }
                call.update(result=result, finished_at=time.time(), status=200)
            elif method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-ov", "version": "1"},
                }
            else:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                )
            return web.json_response({"jsonrpc": "2.0", "id": body.get("id"), "result": result})
        raise web.HTTPNotFound()

    app = web.Application(middlewares=[record_and_auth], client_max_size=32 * 1024**2)
    app.router.add_route("*", "/{path:.*}", handle)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "1933")))
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port, access_log=None)
