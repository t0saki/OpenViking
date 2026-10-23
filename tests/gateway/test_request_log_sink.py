"""Request logs select storage or promptly flushed JSON on stdout."""

import asyncio
import json
import os
import sys
import time

import pytest
from aiohttp import web
from conftest import make_due
from pydantic import ValidationError

from openviking_gateway.cli import load_config
from openviking_gateway.config import GatewayConfig


@pytest.fixture(params=["store", "stdout"])
def gateway_config(request):
    return {"request_log_sink": request.param, "openviking_keys": True}


@pytest.mark.parametrize("direct_key", [False, True])
async def test_requests_use_selected_sink(running_gateway, capsys, direct_key):
    app, client, admin, issued, captured, _ = running_gateway
    stdout = app.state.config.request_log_sink == "stdout"
    key = "user-key" if direct_key else issued["key"]
    request_ids = set()
    for index, (stream, model, status) in enumerate(
        [(False, "model", 200), (True, "model", 200), (False, "error", 429)]
    ):
        session = f"private-session-{index}"
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "X-OpenViking-Session": session},
            json={
                "model": model,
                "stream": stream,
                "messages": [{"role": "user", "content": "private prompt\n你好"}],
            },
        )
        assert response.status_code == status, response.text
        assert captured[-1][2]["Authorization"] == "Bearer model-secret"
        if stream:
            assert "data: [DONE]" in response.text
        elif status == 200:
            assert response.json()["choices"][0]["message"]["content"] == "hello"
        else:
            assert response.headers["retry-after"] == "4"
        output = capsys.readouterr().out
        stored = (await client.get("/admin/logs", headers=admin)).json()
        if stdout:
            assert len(output.splitlines()) == 1
            record = json.loads(output)
            assert record["account"] == "tenant" and record["user_id"] == "alice"
            assert stored == []
            assert await app.state.management.port.logs("tenant") == []
        else:
            assert output == ""
            assert len(stored) == index + 1
            record = stored[0]
        assert 0 <= time.time() - record["time"] < 10
        assert record["model"] == model and record["status"] == status
        assert record["kind"] == "user" and record["protocol"] == "chat"
        assert record["duration_ms"] >= 0
        if stream:
            assert record["input_tokens"] == 123 and record["cached_tokens"] == 100
        request_ids.add(record["request_id"])
        for secret in (key, issued["key"], "user-key", "model-secret", session, "private prompt"):
            assert secret not in output + json.dumps(stored)
    assert len(request_ids) == 3
    overview = (await client.get("/admin/overview", headers=admin)).json()
    assert overview["requests"] == (0 if stdout else 3)


async def test_capture_events_use_selected_sink(running_gateway, capsys):
    app, client, _, issued, _, writes = running_gateway
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + issued["key"], "X-OpenViking-Session": "capture"},
        json={"model": "model", "messages": [{"role": "user", "content": "Remember this"}]},
    )
    assert response.status_code == 200, response.text
    capsys.readouterr()

    async def unavailable(request):
        if request.path.startswith("/api/v1/sessions"):
            return web.Response(status=503)

    for handler, status, reason in [
        (unavailable, "retrying", "openviking_http_503"),
        (None, "active", "delivery_recovered"),
    ]:
        app.state.test_backend["handler"] = handler
        await make_due(app.state.store)
        assert await app.state.worker.once()
        output = capsys.readouterr().out
        stored = await app.state.management.logs("tenant")
        if app.state.config.request_log_sink == "stdout":
            assert len(output.splitlines()) == 1
            record = json.loads(output)
            assert record["account"] == "tenant" and record["user_id"] == "alice"
            assert stored == []
        else:
            assert output == ""
            record = stored[0]
        assert record["kind"] == "capture"
        assert record["capture_status"] == status and record["capture_reason"] == reason
        for secret in (issued["key"], "user-key", "model-secret", "Remember this"):
            assert secret not in output + json.dumps(stored)
    assert writes


async def test_stdout_flushes_each_line_before_process_exit():
    script = """
import asyncio
import sys
from openviking_gateway.app import stdout_log
for index in range(2):
    asyncio.run(stdout_log('tenant\\n你好', {'kind': 'capture', 'index': index}))
    sys.stdin.buffer.readline()
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={name: value for name, value in os.environ.items() if name != "PYTHONUNBUFFERED"},
    )
    try:
        for index in range(2):
            line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
            assert line.endswith(b"\n") and line.count(b"\n") == 1
            record = json.loads(line)
            assert record["index"] == index and record["account"] == "tenant\n你好"
            assert record["user_id"] is None and record["time"] <= time.time()
            assert process.returncode is None
            process.stdin.write(b"continue\n")
            await process.stdin.drain()
        output, errors = await asyncio.wait_for(process.communicate(), timeout=10)
        assert process.returncode == 0, errors.decode()
        assert output == b""
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.parametrize("sink", ["db", "both", "STDOUT", "", None])
def test_invalid_sink_fails_when_loading_config(tmp_path, monkeypatch, sink):
    path = tmp_path / "gateway.conf"
    path.write_text(json.dumps({"openviking_url": "https://ov.test", "request_log_sink": sink}))
    monkeypatch.setenv("OPENVIKING_GATEWAY_CONFIG_FILE", str(path))
    with pytest.raises(ValidationError, match="request_log_sink"):
        load_config()


def test_sink_default_and_environment_override(tmp_path, monkeypatch):
    assert GatewayConfig().request_log_sink == "store"
    path = tmp_path / "gateway.conf"
    path.write_text(json.dumps({"openviking_url": "https://ov.test", "request_log_sink": "stdout"}))
    monkeypatch.setenv("OPENVIKING_GATEWAY_CONFIG_FILE", str(path))
    assert load_config().request_log_sink == "stdout"
    monkeypatch.setenv("OPENVIKING_GATEWAY_REQUEST_LOG_SINK", "store")
    assert load_config().request_log_sink == "store"
    monkeypatch.setenv("OPENVIKING_GATEWAY_REQUEST_LOG_SINK", "both")
    with pytest.raises(ValidationError, match="request_log_sink"):
        load_config()
