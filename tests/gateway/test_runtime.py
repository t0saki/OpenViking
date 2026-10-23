"""Capture runs concurrently and drains before gateway resources close."""

import asyncio

import pytest
import pytest_asyncio
from conftest import make_due
from pydantic import ValidationError

from openviking_gateway.cli import load_config
from openviking_gateway.config import GatewayConfig
from openviking_gateway.storage import digest


@pytest_asyncio.fixture
async def shutdown_checks(monkeypatch, gateway_backend):
    """Check shutdown after running_gateway has finished its lifespan."""
    checks = []
    yield checks
    for check in checks:
        await check()


@pytest.fixture
def gateway_config(shutdown_checks):
    return {"capture_concurrency": 2, "health_interval_seconds": 1}


async def queue_conversation(gateway, session):
    app, client, _, minted, _, _ = gateway
    response = await client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + minted["key"],
            "X-OpenViking-Session": session,
            "X-OpenViking-Context": '{"recall":false,"profile":false}',
        },
        json={"model": "model", "messages": [{"role": "user", "content": session}]},
    )
    assert response.status_code == 200, response.text
    await make_due(app.state.store)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_capture_concurrency_is_validated_when_loading_config(tmp_path, monkeypatch, value):
    path = tmp_path / "gateway.conf"
    path.write_text('{"openviking_url":"http://localhost:1933"}')
    monkeypatch.setenv("OPENVIKING_GATEWAY_CONFIG_FILE", str(path))
    monkeypatch.setenv("OPENVIKING_GATEWAY_CAPTURE_CONCURRENCY", str(value))
    with pytest.raises(ValidationError, match="capture_concurrency"):
        load_config()


def test_capture_concurrency_defaults_to_one_and_accepts_environment(monkeypatch):
    assert GatewayConfig().capture_concurrency == 1
    monkeypatch.setenv("OPENVIKING_GATEWAY_CAPTURE_CONCURRENCY", "3")
    assert GatewayConfig().capture_concurrency == 3


async def test_two_commits_are_in_flight_together(running_gateway):
    app = running_gateway[0]
    entered, release = asyncio.Event(), asyncio.Event()
    health = asyncio.Event()
    sessions = set()

    async def handler(request):
        if request.path == "/health" and not request.headers.get("X-API-Key") and entered.is_set():
            health.set()
        if request.path.endswith("/commit"):
            sessions.add(request.path.split("/")[4])
            if len(sessions) == 2:
                entered.set()
            await release.wait()

    app.state.test_backend["handler"] = handler
    try:
        await queue_conversation(running_gateway, "first")
        await queue_conversation(running_gateway, "second")
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(health.wait(), 3)
        assert len(sessions) == 2
        assert len(running_gateway[5]) == 2
    finally:
        release.set()


@pytest.mark.parametrize("stuck", [False, True], ids=["drain", "timeout"])
async def test_shutdown_stops_claiming_and_closes_after_delivery(
    running_gateway, shutdown_checks, monkeypatch, stuck
):
    app = running_gateway[0]
    entered, release = asyncio.Event(), asyncio.Event()
    commits, cancelled, claims = [], [], []
    completed = []
    commit, claim, wait = app.state.viking.commit, app.state.store.capture.claim, asyncio.wait

    async def handler(request):
        if request.path.endswith("/commit"):
            commits.append(request.path.split("/")[4])
            if len(commits) == 2:
                entered.set()
            await release.wait()

    async def observed_commit(*args, **kwargs):
        try:
            result = await commit(*args, **kwargs)
            completed.append(args[1])
            return result
        except asyncio.CancelledError:
            cancelled.append(args[1])
            raise

    async def observed_claim(*args, **kwargs):
        claims.append(True)
        return await claim(*args, **kwargs)

    app.state.test_backend["handler"] = handler
    monkeypatch.setattr(app.state.viking, "commit", observed_commit)
    monkeypatch.setattr(app.state.store.capture, "claim", observed_claim)
    try:
        await queue_conversation(running_gateway, "first")
        await queue_conversation(running_gateway, "second")
        await asyncio.wait_for(entered.wait(), 3)
        await queue_conversation(running_gateway, "unclaimed")
    except BaseException:
        release.set()
        raise
    claims_before_shutdown = len(claims)
    pending_at_timeout = []

    async def drain(tasks, *, timeout):
        assert timeout == 100
        assert not app.state.http.closed
        if not stuck:
            release.set()
        done, pending = await wait(tasks, timeout=0.02 if stuck else 3)
        pending_at_timeout.extend(pending)
        if not stuck:
            assert not pending
            assert len(completed) == 2
        assert len(claims) == claims_before_shutdown
        doc = await app.state.store.capture.get(digest("tenant\0alice\0chat"), digest("unclaimed"))
        assert doc.value["pending"] and not doc.value["delivered"]
        return done, pending

    async def after_shutdown():
        release.set()
        assert app.state.http.closed
        assert len(commits) == 2
        assert len(claims) == claims_before_shutdown
        assert len(cancelled) == (2 if stuck else 0)
        assert len(completed) == (0 if stuck else 2)
        assert len(pending_at_timeout) == (2 if stuck else 0)
        assert all(task.cancelled() for task in pending_at_timeout)

    monkeypatch.setattr(asyncio, "wait", drain)
    shutdown_checks.append(after_shutdown)
