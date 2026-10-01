import json

import httpx
import orjson
import pytest
import pytest_asyncio
from aiohttp import web
from cryptography.fernet import Fernet

from openviking_context_gateway.app import create_app
from openviking_context_gateway.config import ContextGatewayConfig


@pytest_asyncio.fixture
async def running_gateway(tmp_path, monkeypatch):
    captured = []
    writes = []

    async def backend(request):
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
