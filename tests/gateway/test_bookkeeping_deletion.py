"""A completed upstream stream must not recreate deleted gateway data."""

import sqlite3
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from openviking_gateway.config import GatewayConfig
from openviking_gateway.proxy import ProxyRequest
from openviking_gateway.storage import (
    ManagementStore,
    SQLiteKernelStore,
    delete_account_data,
    delete_user_data,
    digest,
)


@pytest_asyncio.fixture
async def stores(tmp_path):
    encryption_key = Fernet.generate_key().decode()
    path = tmp_path / "management.sqlite3"
    stream_worker = ManagementStore(path, encryption_key)
    deletion_worker = ManagementStore(path, encryption_key)
    kernel = SQLiteKernelStore(tmp_path / "kernel.sqlite3", encryption_key)
    await stream_worker.initialize()
    await deletion_worker.initialize()
    await kernel.initialize()
    try:
        yield stream_worker, deletion_worker, kernel
    finally:
        stream_worker.close()
        deletion_worker.close()
        kernel.close()


def pending_response(management, kernel):
    config = GatewayConfig()
    app = SimpleNamespace(state=SimpleNamespace(config=config, management=management, store=kernel))
    request = ProxyRequest(
        app, None, "v1/responses", {"account": "a", "id": "key", "user_id": "alice"}
    )
    request.capture = SimpleNamespace(usage={}, response_id="response")
    request.upstream = {"id": "upstream"}
    request.body = {"model": "model"}
    request.response = SimpleNamespace(status=200)
    request.started = time.monotonic()
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["user", "account", "revoke"])
async def test_stream_finishes_after_deletion_in_another_worker(stores, operation):
    stream_worker, deletion_worker, kernel = stores
    stream_worker.cache.ttl = float("inf")
    await stream_worker.save("a", "keys", "key", {"user_id": "alice"})
    await stream_worker.save("a", "users", "alice", {"user_id": "alice"})
    # A different account's live requests must remain writable.
    await stream_worker.save("b", "keys", "other", {"user_id": "bob"})
    request = pending_response(stream_worker, kernel)
    assert await stream_worker.get("a", "keys", "key")  # Prime the reader's stale key cache.

    if operation == "user":
        await delete_user_data(deletion_worker, kernel, "a", "alice")
    elif operation == "account":
        await delete_account_data(deletion_worker, kernel, "a")
    else:
        await deletion_worker.delete("a", "keys", "key")
    assert await stream_worker.get("a", "keys", "key")  # Cache is intentionally still warm.
    await request.finish()

    assert await stream_worker.logs("a") == []
    assert await stream_worker.list("a", "responses") == []
    with stream_worker.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM object_expiry WHERE account='a'").fetchone()[0]
            == 0
        )
    assert await stream_worker.record_request("b", "other", {"kind": "capture"})
    assert len(await stream_worker.logs("b", user_id="bob")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("store_response", [True, False])
async def test_live_stream_records_user_log_and_optional_response(stores, store_response):
    management, _, kernel = stores
    await management.save("a", "keys", "key", {"user_id": "alice"})
    request = pending_response(management, kernel)
    request.body["store"] = store_response
    await request.finish()
    assert len(await management.logs("a", user_id="alice")) == 1
    response = await management.get("a", "responses", digest("keyresponse"))
    if store_response:
        assert response["user_id"] == "alice"
        assert response["upstream_id"] == "upstream"
        assert response["expires_at"] > time.time()
    else:
        assert response is None


@pytest.mark.asyncio
async def test_capture_log_guard_skips_revoked_key(stores):
    management, deletion_worker, _ = stores
    await management.save("a", "keys", "key", {"user_id": "alice"})
    await deletion_worker.delete("a", "keys", "key")
    assert not await management.record_request("a", "key", {"kind": "capture"})
    assert await management.logs("a") == []


@pytest.mark.asyncio
async def test_response_and_log_are_one_transaction(stores):
    management, _, _ = stores
    await management.save("a", "keys", "key", {"user_id": "alice"})
    with management.connect() as connection:
        connection.execute("""
            CREATE TRIGGER reject_log BEFORE INSERT ON request_logs
            BEGIN SELECT RAISE(ABORT, 'log write failed'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="log write failed"):
        await management.record_request(
            "a",
            "key",
            {"kind": "user"},
            response_id="response",
            upstream_id="upstream",
            response_ttl=60,
        )
    assert await management.list("a", "responses") == []
    with management.connect() as connection:
        assert connection.execute("SELECT count(*) FROM object_expiry").fetchone()[0] == 0
    assert await management.logs("a") == []
