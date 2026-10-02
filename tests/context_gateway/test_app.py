import asyncio
import json
from pathlib import Path

import httpx
import orjson
import pytest
import pytest_asyncio
from aiohttp import web
from cryptography.fernet import Fernet

from openviking_context_gateway.app import create_app
from openviking_context_gateway.config import ContextGatewayConfig
from openviking_context_gateway.protocols import normalize


@pytest_asyncio.fixture
async def running_gateway(tmp_path, monkeypatch):
    captured = []
    writes = []
    override = {}

    async def backend(request):
        if override.get("handler"):
            response = await override["handler"](request)
            if response is not None:
                return response
        raw = await request.read()
        body = json.loads(raw) if raw else {}
        if request.path == "/health":
            key = request.headers.get("X-API-Key")
            return web.json_response(
                {
                    "version": "0.4.16",
                    "status": "ok",
                    "auth_mode": "api_key",
                    "role": "root" if key == "root" else "user",
                    "account_id": "other" if key == "other" else "tenant",
                    "user_id": "alice",
                }
            )
        if request.path == "/api/v1/search/search":
            assert body["mode"] == "context" and "session_id" not in body
            return web.json_response(
                {
                    "status": "ok",
                    "result": {
                        "entries": [
                            {"uri": "viking://user/alice/memories/x", "text": "Memory data"}
                        ]
                    },
                }
            )
        if "/messages/batch" in request.path:
            writes.append(body)
            return web.json_response({"status": "ok", "result": {"pending_tokens": 10}})
        if "/commit" in request.path and request.path.startswith("/api/v1/sessions"):
            return web.json_response({"status": "ok", "result": {}})
        captured.append((request.path, raw, dict(request.headers)))
        if body.get("model") == "error":
            return web.Response(
                body=b'{"error":{"raw":"provider error"}}',
                status=429,
                headers={
                    "Retry-After": "4",
                    "X-Should-Retry": "true",
                    "Request-Id": "upstream-id",
                    "Content-Type": "application/json",
                },
            )
        if body.get("stream"):
            response = web.StreamResponse(
                headers={"Content-Type": "text/event-stream", "Request-Id": "stream-id"}
            )
            await response.prepare(request)
            stream = b'data: {"id":"r-1","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\r\n\r\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":123,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":100}}}\n\ndata: [DONE]\n\n'
            for start in range(0, len(stream), 7):
                await response.write(stream[start : start + 7])
            await response.write_eof()
            return response
        if "/responses" in request.path:
            return web.json_response(
                {
                    "id": "resp-1",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "role": "assistant",
                            "type": "message",
                            "content": [{"type": "output_text", "text": "hello"}],
                        }
                    ],
                }
            )
        if request.path.endswith("count_tokens"):
            return web.json_response({"input_tokens": 20})
        if request.path.endswith("messages"):
            return web.json_response(
                {
                    "id": "msg-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn",
                }
            )
        return web.json_response(
            {
                "id": "r-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
            }
        )

    server = web.Application()
    server.router.add_route("*", "/{path:.*}", backend)
    runner = web.AppRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    monkeypatch.setenv("OPENVIKING_CONTEXT_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("OPENVIKING_CONTEXT_GATEWAY_ADMIN_TOKEN", "admin-" + "x" * 32)
    config = ContextGatewayConfig(enabled=True, storage_path=str(tmp_path), openviking_url=base)
    app = create_app(config)
    app.state.test_backend = override
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            admin = {"Authorization": "Bearer admin-" + "x" * 32, "X-OpenViking-Account": "tenant"}
            for protocol in ("chat", "anthropic", "responses"):
                response = await client.put(
                    f"/admin/upstreams/{protocol}",
                    headers=admin,
                    json={
                        "name": protocol,
                        "protocol": protocol,
                        "base_url": base,
                        "api_key": "model-secret",
                        "models": ["model", "error"],
                    },
                )
                assert response.status_code == 200, response.text
            assert (
                await client.put("/admin/policies/default", headers=admin, json={"name": "Default"})
            ).status_code == 200
            minted = await client.post(
                "/admin/keys",
                headers=admin,
                json={
                    "name": "test",
                    "openviking_key": "user-key",
                    "policy_id": "default",
                    "upstream_ids": ["chat", "anthropic", "responses"],
                },
            )
            assert minted.status_code == 200, minted.text
            yield app, client, admin, minted.json(), captured, writes
    await runner.cleanup()


@pytest.mark.parametrize(
    ("path", "protocol"),
    [
        ("/v1/chat/completions", "chat"),
        ("/v1/messages", "anthropic"),
        ("/v1/responses", "responses"),
    ],
)
async def test_http_replay_and_client_visibility(running_gateway, path, protocol):
    _, client, _, key, seen, _ = running_gateway
    headers = {
        "Authorization": "Bearer " + key["key"],
        "X-OpenViking-Session": "chat-1",
        "anthropic-version": "2023-06-01",
    }
    field = "input" if protocol == "responses" else "messages"
    body = {
        "model": "model",
        "store": False,
        field: [{"role": "user", "content": "Remember blue?"}],
        "unknown": {"preserve": [1, 2]},
    }
    response = await client.post(path, headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert "openviking-context" not in response.text
    original = orjson.loads(seen[-1][1])
    assert "openviking-context" in str(original[field])
    body[field] += [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "Continue please"},
    ]
    response = await client.post(path, headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert orjson.loads(seen[-1][1])[field][0] == original[field][0]
    assert seen[-1][2].get(
        "Authorization", seen[-1][2].get("x-api-key", seen[-1][2].get("X-Api-Key"))
    ) in {"Bearer model-secret", "model-secret"}


@pytest.mark.parametrize("client_name", ["claude-code", "codex-cli", "openai-python"])
@pytest.mark.parametrize("session_header", [True, False])
async def test_recorded_client_prefixes(running_gateway, client_name, session_header):
    app, client, admin, key, seen, writes = running_gateway
    fixture = json.loads((Path(__file__).parent / "fixtures" / f"{client_name}.json").read_text())
    upstream = []
    for recorded in fixture["requests"]:
        headers = {"Authorization": "Bearer " + key["key"]}
        headers.update(
            {k: v for k, v in recorded["headers"].items() if session_header or "session" not in k}
        )
        # Keep the captured message representation, but use the local JSON mock
        # response. SSE framing is exercised separately and in live acceptance.
        body = {**recorded["body"], "stream": False}
        response = await client.post(recorded["path"], headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert "openviking-context" not in response.text
        forwarded = [raw for path, raw, _ in seen if path == recorded["path"]][-1]
        upstream.append(orjson.loads(forwarded))
    field = "input" if client_name == "codex-cli" else "messages"
    old, new = (body[field] for body in upstream)
    assert "openviking-context" in str(old)
    assert [normalize(m) for m in old] == [normalize(m) for m in new[: len(old)]]
    assert "openviking-context" not in str(fixture)
    for field_name in ("tools", "system", "instructions"):
        if field_name in fixture["requests"][1]["body"]:
            assert upstream[1][field_name] == fixture["requests"][1]["body"][field_name]

    # A client retry and simultaneous duplicate requests must reuse the same
    # injection and must not capture the confirmed first turn more than once.
    repeated = await asyncio.gather(
        *(client.post(recorded["path"], headers=headers, json=body) for _ in range(3))
    )
    assert all(r.status_code == 200 for r in repeated)
    forwarded = [raw for path, raw, _ in seen if path == recorded["path"]][-3:]
    assert all(orjson.loads(raw) == upstream[1] for raw in forwarded)
    for _ in range(20):
        await app.state.worker.once()
        if writes:
            break
        await asyncio.sleep(0.01)
    source_ids = [
        source_id
        for batch in writes
        for message in batch["messages"]
        for source_id in message["source_message_ids"]
    ]
    assert source_ids and len(source_ids) == len(set(source_ids))
    logs = (await client.get("/admin/logs", headers=admin)).json()
    assert logs[0]["replay_hits"] >= 1


@pytest.mark.parametrize("client_name", ["claude-code", "codex-cli", "openai-python"])
async def test_recorded_client_disabled_passthrough(running_gateway, client_name):
    _, client, admin, _, seen, _ = running_gateway
    fixture = json.loads((Path(__file__).parent / "fixtures" / f"{client_name}.json").read_text())
    await client.put(
        "/admin/policies/off",
        headers=admin,
        json={"name": "Off", "recall": False, "capture": False, "takeover": False},
    )
    minted = await client.post(
        "/admin/keys",
        headers=admin,
        json={
            "name": "off",
            "openviking_key": "user-key",
            "policy_id": "off",
            "upstream_ids": ["chat", "anthropic", "responses"],
        },
    )
    assert minted.status_code == 200
    for recorded in fixture["requests"]:
        raw = json.dumps({**recorded["body"], "stream": False}, indent=2).encode()
        response = await client.post(
            recorded["path"],
            headers={"Authorization": "Bearer " + minted.json()["key"]},
            content=raw,
        )
        assert response.status_code == 200
        assert [data for path, data, _ in seen if path == recorded["path"]][-1] == raw


async def test_raw_passthrough_errors_sse_and_large_numbers(running_gateway):
    _, client, admin, key, seen, _ = running_gateway
    headers = {"Authorization": "Bearer " + key["key"]}
    await client.put(
        "/admin/policies/off",
        headers=admin,
        json={"name": "Off", "recall": False, "capture": False, "takeover": False},
    )
    raw = b'{ "model":"model", "messages":[{"role":"user","content":"hi"}], "arg":922337203685477580922 }'
    response = await client.post("/v1/chat/completions", headers=headers, content=raw)
    assert response.status_code == 200
    assert seen[-1][1] == raw
    body = {"model": "error", "messages": [{"role": "user", "content": "hello"}]}
    response = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 429 and response.headers["retry-after"] == "4"
    assert response.content == b'{"error":{"raw":"provider error"}}'
    response = await client.post(
        "/v1/chat/completions", headers=headers, json={**body, "model": "model", "stream": True}
    )
    assert response.headers["request-id"] == "stream-id"
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert b"openviking-context" not in response.content
    logs = (await client.get("/admin/logs", headers=admin)).json()
    assert logs[0]["cached_tokens"] == 100
    assert "model-secret" not in str(logs) and "user-key" not in str(logs)


async def test_key_scope_revocation_and_root_rejection(running_gateway):
    _, client, admin, key, _, _ = running_gateway
    for secret, expected in [("root", 403), ("other", 403)]:
        response = await client.post(
            "/admin/keys",
            headers=admin,
            json={
                "name": "bad",
                "openviking_key": secret,
                "policy_id": "default",
                "upstream_ids": ["chat"],
            },
        )
        assert response.status_code == expected
    other_admin = {**admin, "X-OpenViking-Account": "other"}
    assert (await client.get("/admin/keys", headers=other_admin)).json() == []
    listed = (await client.get("/admin/keys", headers=admin)).json()
    assert key["key"] not in str(listed) and "user-key" not in str(listed)
    await client.delete("/admin/keys/" + key["id"], headers=admin)
    response = await client.get("/v1/models", headers={"Authorization": "Bearer " + key["key"]})
    assert response.status_code == 401


async def test_stateful_responses_and_counting(running_gateway):
    _, client, _, key, seen, _ = running_gateway
    headers = {"Authorization": "Bearer " + key["key"]}
    raw = b'{"model":"model", "previous_response_id":"old", "input":"new"}'
    response = await client.post("/v1/responses", headers=headers, content=raw)
    assert response.status_code == 200 and seen[-1][1] == raw
    assert (await client.get("/v1/responses/resp-1", headers=headers)).status_code == 200
    assert (await client.get("/v1/responses/unknown", headers=headers)).status_code == 404
    body = {"model": "model", "messages": [{"role": "user", "content": "Remember this request"}]}
    await client.post("/v1/messages", headers=headers, json=body)
    first = orjson.loads(seen[-1][1])
    await client.post("/v1/messages/count_tokens", headers=headers, json=body)
    assert orjson.loads(seen[-1][1]) == first


async def enable_tools(client, admin, **policy):
    response = await client.put(
        "/admin/policies/default",
        headers=admin,
        json={"name": "Tools", "gateway_tools": True, "recall": False, "takeover": False, **policy},
    )
    assert response.status_code == 200, response.text


def tool_call(name="openviking_search", identifier="g-1", arguments=None):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"query": "blue"})},
    }


def completion(message, finish="stop", **extra):
    return {
        "id": "upstream-id",
        "object": "chat.completion",
        "model": "model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
        **extra,
    }


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_hidden_tools_mixed_replay_and_usage(running_gateway, streaming, mixed):
    app, client, admin, key, _, _ = running_gateway
    await enable_tools(client, admin)
    calls, model_requests = [], []

    async def backend(request):
        if request.path == "/mcp":
            payload = await request.json()
            calls.append(payload)
            assert request.headers["X-API-Key"] == "user-key"
            assert payload["params"]["name"] == "find"
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "the answer is blue"}]},
                }
            )
        if request.path != "/v1/chat/completions":
            return None
        payload = await request.json()
        model_requests.append(payload)
        first = len(model_requests) == 1
        message = {
            "role": "assistant",
            "content": "Looking. " if first else "Blue.",
            "reasoning_content": "private-1" if first else "private-2",
            "vendor_extension": {"signature": "opaque"},
        }
        if first:
            message["tool_calls"] = [tool_call()]
            if mixed:
                message["tool_calls"].append(tool_call("client_weather", "c-1", {"city": "Paris"}))
        value = completion(message, "tool_calls" if first else "stop", provider_field="keep")
        if not streaming:
            return web.json_response(value)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        content = dict(message)
        tools = content.pop("tool_calls", [])
        events = [
            {
                "id": "u-id",
                "provider_field": "keep",
                "choices": [{"index": 0, "delta": content, "finish_reason": None}],
            }
        ]
        # Split names and arguments across SSE fragments, including both kinds.
        for index, call in enumerate(tools):
            function = call["function"]
            events += [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call["id"],
                                        "type": "function",
                                        "function": {"name": function["name"][:5], "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "function": {
                                            "name": function["name"][5:],
                                            "arguments": function["arguments"],
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            ]
        events.append(
            {
                "choices": [{"delta": {}, "finish_reason": value["choices"][0]["finish_reason"]}],
                "usage": value["usage"],
            }
        )
        raw = (
            b"".join(b"data: " + orjson.dumps(event) + b"\n\n" for event in events)
            + b"data: [DONE]\n\n"
        )
        for i in range(0, len(raw), 13):
            await response.write(raw[i : i + 13])
        await response.write_eof()
        return response

    app.state.test_backend["handler"] = backend
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": "Find blue"}],
        "stream": streaming,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {"name": "client_weather", "parameters": {"type": "object"}},
            }
        ],
    }
    headers = {"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "tool-session"}
    response = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert "openviking_search" not in response.text
    from openviking_context_gateway.storage import digest

    stored = await app.state.store.read(digest("tenant\0alice\0chat"), digest("tool-session"))
    assert [v["input_tokens"] for (kind, _), v in stored.items() if kind == "usage"] == [100]
    assert len(calls) == 1 and len(model_requests) == (1 if mixed else 2)
    if streaming:
        from openviking_context_gateway.protocols import SSEDecoder
        from openviking_context_gateway.tool_loop import merge_delta

        visible = {"role": "assistant", "content": ""}
        events = [SSEDecoder.data(f) for f in SSEDecoder().feed(response.content)]
        events = [event for event in events if event]
        for event in events:
            for choice in event.get("choices", []):
                delta = dict(choice.get("delta", {}))
                if delta.get("tool_calls"):
                    visible["tool_calls"] = [
                        {k: v for k, v in c.items() if k != "index"}
                        for c in delta.pop("tool_calls")
                    ]
                merge_delta(visible, delta)
        assert response.content.count(b"data: [DONE]") == 1
        assert sum(bool(c.get("finish_reason")) for e in events for c in e.get("choices", [])) == 1
        assert events[-1]["usage"]["prompt_tokens"] == (100 if mixed else 200)
    else:
        visible = response.json()["choices"][0]["message"]
        assert response.json()["provider_field"] == "keep"
    assert visible["content"] == ("Looking. " if mixed else "Looking. Blue.")
    # A typical chat UI drops reasoning/unknown fields from its returned history.
    visible = {k: v for k, v in visible.items() if k in {"role", "content", "tool_calls"}}
    body["messages"].append(visible)
    if mixed:
        body["messages"].append({"role": "tool", "tool_call_id": "c-1", "content": "Sunny"})
    else:
        body["messages"].append({"role": "user", "content": "Continue"})
    # Change the live policy; the existing root must retain its frozen tools.
    await enable_tools(client, admin, tool_allowlist=["list"])
    response = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    replay = model_requests[-1]
    assert replay["tools"] == model_requests[0]["tools"]
    hidden = replay["messages"][1]
    assert hidden["reasoning_content"] == "private-1"
    assert hidden["vendor_extension"] == {"signature": "opaque"}
    assert hidden["tool_calls"][0]["function"]["name"] == "openviking_search"
    assert replay["messages"][2]["tool_call_id"] == "g-1"
    if mixed:
        assert replay["messages"][3]["tool_call_id"] == "c-1"
    assert len(calls) == 1
    logs = (await client.get("/admin/logs", headers=admin)).json()
    assert any(log.get("hidden_rounds") == 1 for log in logs)
    assert not any("private-1" in str(log) for log in logs)


@pytest.mark.parametrize(
    "extra",
    [
        {"n": 2},
        {"response_format": {"type": "json_object"}},
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "function": {"name": "foo"}}},
    ],
)
async def test_tool_capability_gate_and_frozen_conflict(running_gateway, extra):
    _, client, admin, key, seen, _ = running_gateway
    await enable_tools(client, admin)
    headers = {"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "blocked"}
    body = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}
    response = await client.post("/v1/chat/completions", headers=headers, json={**body, **extra})
    assert response.status_code == 200
    assert not orjson.loads(seen[-1][1]).get("tools")
    headers["X-OpenViking-Session"] = "enabled"
    response = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200
    assert len(orjson.loads(seen[-1][1])["tools"]) == 3
    response = await client.post("/v1/chat/completions", headers=headers, json={**body, **extra})
    assert response.status_code == 200
    assert not orjson.loads(seen[-1][1]).get("tools")


async def test_tool_round_limit_disables_without_removing_definitions(running_gateway):
    app, client, admin, key, _, _ = running_gateway
    await enable_tools(client, admin, tool_max_rounds=1)
    requests, mcp_calls = [], []

    async def backend(request):
        if request.path == "/mcp":
            mcp_calls.append(await request.json())
            return web.json_response({"id": 1, "result": {"content": []}})
        if request.path == "/v1/chat/completions":
            requests.append(await request.json())
            return web.json_response(
                completion(
                    {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
                    "tool_calls",
                )
            )
        return None

    app.state.test_backend["handler"] = backend
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "tools"},
        json={"model": "model", "messages": [{"role": "user", "content": "find blue"}]},
    )
    assert response.status_code == 502
    assert len(requests) == 2 and len(mcp_calls) == 1
    assert requests[1]["tool_choice"] == "none"
    assert requests[0]["tools"] == requests[1]["tools"]


@pytest.mark.parametrize(
    "path,protocol,target",
    [
        ("/api/v3/chat/completions", "chat", "/api/v3/chat/completions"),
        ("/api/v3/responses", "responses", "/api/v3/responses"),
        ("/api/compatible/v1/messages", "anthropic", "/api/compatible/v1/messages"),
    ],
)
async def test_ark_paths_unknown_fields_and_cache_key(running_gateway, path, protocol, target):
    _, client, admin, key, seen, _ = running_gateway
    upstream = (await client.get("/admin/upstreams", headers=admin)).json()
    configured = next(u for u in upstream if u["id"] == protocol)
    payload = {
        k: v
        for k, v in configured.items()
        if k not in {"id", "revision", "has_api_key", "header_names", "account"}
    }
    payload["vendor"] = "ark"
    response = await client.put("/admin/upstreams/" + protocol, headers=admin, json=payload)
    assert response.status_code == 200, response.text
    field = "input" if protocol == "responses" else "messages"
    body = {
        "model": "model",
        "store": False,
        field: [{"role": "user", "content": "Remember blue"}],
        "thinking": {"type": "enabled"},
        "caching": {"type": "enabled"},
        "expire_at": 1893456000,
        "encrypted_content": "opaque",
    }
    headers = {"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "ark"}
    for _ in range(2):
        response = await client.post(path, headers=headers, json=body)
        assert response.status_code == 200, response.text
        forwarded = orjson.loads(seen[-1][1])
        assert seen[-1][0] == target
        assert all(
            forwarded[k] == body[k]
            for k in ["thinking", "caching", "expire_at", "encrypted_content"]
        )
    if protocol != "anthropic":
        assert orjson.loads(seen[-2][1])["prompt_cache_key"] == forwarded["prompt_cache_key"]
    payload["coding_plan"] = True
    assert (
        await client.put("/admin/upstreams/" + protocol, headers=admin, json=payload)
    ).status_code == 200
    assert (await client.post(path, headers=headers, json=body)).status_code == 403


@pytest.mark.parametrize("name", ["add_resource", "add_skill"])
async def test_attachment_import_and_signed_upload_proxy(running_gateway, name):
    import base64

    app, client, admin, key, _, _ = running_gateway
    app.state.config.public_url = "https://gateway.example"
    await enable_tools(client, admin, allow_write_tools=True, tool_allowlist=[name])
    model_requests, uploads, mcp_requests = [], [], []

    async def backend(request):
        if request.path == "/mcp":
            payload = await request.json()
            mcp_requests.append(payload)
            assert payload["params"]["name"] == name
            assert payload["params"]["arguments"]["path"].startswith("/client-upload/")
            return web.Response(
                content_type="text/event-stream",
                text="data: "
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "POST multipart field file to http://private-ov/api/v1/resources/temp_upload?token=signed-once",
                                }
                            ]
                        },
                    }
                )
                + "\n\n",
            )
        if request.path == "/api/v1/resources/temp_upload":
            assert request.query["token"] == "signed-once"
            assert "Authorization" not in request.headers and "X-API-Key" not in request.headers
            multipart = await request.multipart()
            part = await multipart.next()
            uploads.append((part.name, part.filename, bytes(await part.read())))
            return web.json_response(
                {"status": "ok", "result": {"uri": "viking://resources/imported"}}
            )
        if request.path == "/v1/chat/completions":
            payload = await request.json()
            model_requests.append(payload)
            if len(model_requests) == 1:
                return web.json_response(
                    completion(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                tool_call("openviking_" + name, arguments={"attachment_index": 0})
                            ],
                        },
                        "tool_calls",
                    )
                )
            return web.json_response(completion({"role": "assistant", "content": "Imported."}))
        return None

    app.state.test_backend["handler"] = backend
    body = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Import the attached file"},
                    {
                        "type": "file",
                        "file": {
                            "filename": "SKILL.md",
                            "file_data": "data:text/plain;base64,"
                            + base64.b64encode(b"# Test skill\ncontent").decode(),
                        },
                    },
                ],
            }
        ],
    }
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "tools"},
        json=body,
    )
    assert response.status_code == 200, response.text
    assert len(mcp_requests) == 1 and uploads == [("file", "SKILL.md", b"# Test skill\ncontent")]
    assert "signed-once" not in response.text
    assert "viking://resources/imported" in model_requests[1]["messages"][-1]["content"]
    # The token is the sole upload authorization, and the destination is fixed.
    response = await client.post(
        "/context-gateway/uploads?token=signed-once&url=http://attacker",
        files={"file": ("notes.txt", b"upload")},
        headers={"Authorization": "Bearer not-forwarded"},
    )
    assert response.status_code == 200, response.text
    assert uploads[-1] == ("file", "notes.txt", b"upload")
    assert (
        await client.post("/context-gateway/uploads", files={"file": ("x.txt", b"x")})
    ).status_code == 400


async def test_shell_upload_instructions_use_public_proxy(running_gateway):
    app, client, admin, key, _, _ = running_gateway
    app.state.config.public_url = "https://gateway.example"
    await enable_tools(client, admin, allow_write_tools=True, tool_allowlist=["add_resource"])
    requests = []

    async def backend(request):
        if request.path == "/mcp":
            return web.json_response(
                {
                    "id": 1,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "POST to http://private-ov/api/v1/resources/temp_upload?token=opaque%2Btoken",
                            }
                        ]
                    },
                }
            )
        if request.path == "/v1/chat/completions":
            payload = await request.json()
            requests.append(payload)
            if len(requests) == 1:
                return web.json_response(
                    completion(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                tool_call(
                                    "openviking_add_resource",
                                    arguments={"path": "/home/client/file.md"},
                                )
                            ],
                        },
                        "tool_calls",
                    )
                )
            result = payload["messages"][-1]["content"]
            assert "https://gateway.example/context-gateway/uploads?token=opaque%2Btoken" in result
            assert "private-ov" not in result and "user-key" not in result
            return web.json_response(
                completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call("bash", "shell-call", {"command": "upload"})],
                    },
                    "tool_calls",
                )
            )
        return None

    app.state.test_backend["handler"] = backend
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "tools"},
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "import file"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"


async def test_ark_responses_stream_keeps_native_fields_and_done(running_gateway):
    app, client, admin, key, _, _ = running_gateway
    upstreams = (await client.get("/admin/upstreams", headers=admin)).json()
    upstream = next(u for u in upstreams if u["id"] == "responses")
    allowed = {"name", "protocol", "base_url", "models", "aliases", "vendor"}
    payload = {k: v for k, v in upstream.items() if k in allowed}
    payload["vendor"] = "ark"
    assert (
        await client.put("/admin/upstreams/responses", headers=admin, json=payload)
    ).status_code == 200
    native = {
        "type": "response.completed",
        "response": {
            "id": "ark-response",
            "status": "completed",
            "output": [
                {"type": "reasoning", "encrypted_content": "opaque-signature"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            ],
            "caching": {"type": "enabled"},
            "expire_at": 1893456000,
            "usage": {
                "input_tokens": 3000,
                "output_tokens": 12,
                "input_tokens_details": {"cached_tokens": 2048},
            },
        },
    }
    raw = b"data: " + orjson.dumps(native) + b"\n\ndata: [DONE]\n\n"

    async def backend(request):
        if request.path == "/api/v3/responses":
            return web.Response(body=raw, content_type="text/event-stream")
        return None

    app.state.test_backend["handler"] = backend
    response = await client.post(
        "/api/v3/responses",
        headers={"Authorization": "Bearer " + key["key"]},
        json={"model": "model", "store": False, "stream": True, "input": "hello there"},
    )
    assert response.content == raw
    log = (await client.get("/admin/logs", headers=admin)).json()[0]
    assert log["cached_tokens"] == 2048 and log["input_tokens"] == 3000
    assert log["cache_eligible"] is True


@pytest.mark.parametrize("max_field", ["max_tokens", "max_completion_tokens"])
async def test_large_first_tool_request_keeps_client_token_limit(running_gateway, max_field):
    _, client, admin, key, seen, _ = running_gateway
    await enable_tools(client, admin, tool_total_tokens=1024)
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "system": "fixed instructions " * 20000,
        max_field: 6000,
    }
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "large-tool"},
        json=body,
    )
    assert response.status_code == 200
    forwarded = orjson.loads(seen[-1][1])
    assert forwarded[max_field] == 6000
    assert ("max_completion_tokens" if max_field == "max_tokens" else "max_tokens") not in forwarded


async def test_ark_burst_is_forwarded_without_local_429(running_gateway):
    app, client, admin, key, seen, _ = running_gateway
    upstream = await app.state.management.get("tenant", "upstreams", "chat")
    await app.state.management.save("tenant", "upstreams", "chat", {**upstream, "vendor": "ark"})
    headers = {"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "ark-burst"}
    for _ in range(20):
        response = await client.post(
            "/api/v3/chat/completions",
            headers=headers,
            json={"model": "model", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200
    forwarded = [orjson.loads(raw) for path, raw, _ in seen if path == "/api/v3/chat/completions"]
    assert len(forwarded) == 20
    assert len({body["prompt_cache_key"] for body in forwarded}) == 1


async def test_responses_store_false_has_no_routing_mapping(running_gateway):
    app, client, _, key, _, _ = running_gateway
    response = await client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer " + key["key"]},
        json={"model": "model", "store": False, "input": "hello"},
    )
    assert response.status_code == 200
    assert await app.state.management.list("tenant", "responses") == []


async def test_storage_failure_still_forwards_plain_chat(running_gateway, monkeypatch):
    app, client, _, key, seen, _ = running_gateway

    async def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(app.state.store, "read", unavailable)
    raw = b'{"model":"model", "messages":[{"role":"user","content":"hello"}]}'
    response = await client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer " + key["key"]}, content=raw
    )
    assert response.status_code == 200
    assert seen[-1][1] == raw


@pytest.mark.parametrize("limit", [{}, {"max_tokens": 1024}, {"max_completion_tokens": 2048}])
async def test_long_context_tool_continuation_preserves_output_limit(running_gateway, limit):
    app, client, admin, key, _, _ = running_gateway
    await enable_tools(client, admin, allow_write_tools=True, tool_allowlist=["write"])
    requests, writes = [], []

    async def backend(request):
        if request.path not in {"/mcp", "/v1/chat/completions"}:
            return None
        payload = await request.json()
        if request.path == "/mcp":
            writes.append(payload)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "saved"}]},
                }
            )
        if request.path != "/v1/chat/completions":
            return None
        requests.append(payload)
        if len(requests) == 1:
            value = completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "openviking_write",
                            arguments={"uri": "viking://~/notes/test", "content": "saved"},
                        )
                    ],
                },
                "tool_calls",
            )
        else:
            value = completion({"role": "assistant", "content": "saved"})
        value["usage"] = {"prompt_tokens": 110000, "completion_tokens": 20}
        return web.json_response(value)

    app.state.test_backend["handler"] = backend
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "long-tools"},
        json={"model": "model", "messages": [{"role": "user", "content": "save this"}], **limit},
    )
    assert response.status_code == 200, response.text
    assert len(writes) == 1 and len(requests) == 2
    assert {
        k: v for k, v in requests[1].items() if k in {"max_tokens", "max_completion_tokens"}
    } == limit


@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_budget_finishes_after_results_without_usage(running_gateway, streaming):
    app, client, admin, key, _, _ = running_gateway
    await enable_tools(client, admin, tool_total_tokens=1024)
    requests, calls = [], []

    async def backend(request):
        if request.path not in {"/mcp", "/v1/chat/completions"}:
            return None
        payload = await request.json()
        if request.path == "/mcp":
            calls.append(payload)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "result " * 1000}]},
                }
            )
        if request.path != "/v1/chat/completions":
            return None
        requests.append(payload)
        first = len(requests) == 1
        message = (
            {"role": "assistant", "content": None, "tool_calls": [tool_call()]}
            if first
            else {"role": "assistant", "content": "done"}
        )
        finish = "tool_calls" if first else "stop"
        if not streaming:
            value = completion(message, finish)
            value.pop("usage")
            return web.json_response(value)
        if first:
            message["tool_calls"][0]["index"] = 0
        event = {"choices": [{"index": 0, "delta": message, "finish_reason": finish}]}
        return web.Response(
            body=b"data: " + orjson.dumps(event) + b"\n\ndata: [DONE]\n\n",
            content_type="text/event-stream",
        )

    app.state.test_backend["handler"] = backend
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "budget-tools"},
        json={
            "model": "model",
            "stream": streaming,
            "messages": [{"role": "user", "content": "search"}],
        },
    )
    assert response.status_code == 200 and "gateway_tool_error" not in response.text
    assert len(calls) == 1 and len(requests) == 2
    assert requests[1]["tool_choice"] == "none"
    assert "max_tokens" not in requests[1]
    logs = (await client.get("/admin/logs", headers=admin)).json()
    assert logs[0]["tool_stop_reason"] == "token_budget"


async def test_capture_reset_is_scoped_to_the_key_account(running_gateway):
    app, client, admin, key, _, _ = running_gateway
    await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key["key"], "X-OpenViking-Session": "reset"},
        json={"model": "model", "messages": [{"role": "user", "content": "hello"}]},
    )
    log = (await client.get("/admin/logs", headers=admin)).json()[0]
    path = "/admin/keys/" + log["credential_id"] + "/capture/reset"
    body = {"session": log["session"], "protocol": "chat"}
    wrong = await client.post(path, headers={**admin, "X-OpenViking-Account": "other"}, json=body)
    assert wrong.status_code == 404
    assert (await client.post(path, headers=admin, json=body)).status_code == 200
    assert (
        await client.post(path, headers=admin, json={**body, "session": "unknown"})
    ).status_code == 404
