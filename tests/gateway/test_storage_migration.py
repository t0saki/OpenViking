"""Upgrade real v5 SQLite layouts and preserve tenant/user boundaries."""

import sqlite3
import time

import orjson
import pytest
from cryptography.fernet import Fernet, InvalidToken

from openviking_gateway.storage import (
    ManagementStore,
    SQLiteKernelStore,
    delete_account_data,
    delete_user_data,
    digest,
)


@pytest.fixture
def v5_database(tmp_path):
    key = Fernet.generate_key().decode()
    cipher = Fernet(key.encode())
    path = tmp_path / "management.sqlite3"
    records = [
        ("openai", "chat", "https://api.openai.com", "https://api.openai.com/v1"),
        ("ark", "chat", "https://ark.test", "https://ark.test/api/v3"),
        ("byteplus", "responses", "https://byteplus.test", "https://byteplus.test/api/v3"),
        ("ark", "anthropic", "https://ark.test", "https://ark.test/api/compatible"),
        ("byteplus", "anthropic", "https://byteplus.test", "https://byteplus.test/api/compatible"),
        ("anthropic", "anthropic", "https://anthropic.test/v1", "https://anthropic.test"),
        (
            "ark",
            "anthropic",
            "https://ark.test/api/compatible/v1",
            "https://ark.test/api/compatible",
        ),
        # v5 requested base + /v1/...; this GLM base was already wrong and stays so.
        ("generic", "chat", "https://glm.test/api/paas/v4", "https://glm.test/api/paas/v4"),
        ("generic", "chat", "https://openrouter.ai/api", "https://openrouter.ai/api/v1"),
        ("generic", "responses", "https://api.groq.com/openai", "https://api.groq.com/openai/v1"),
        ("openai", "chat", "https://proxy.test/v1", "https://proxy.test/v1"),
        ("ark", "chat", "https://ark.test/custom", "https://ark.test/custom"),
        ("generic", "anthropic", "https://anthropic.test", "https://anthropic.test"),
    ]
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            PRAGMA user_version=5;
            CREATE TABLE objects (account TEXT,kind TEXT,id TEXT,revision INTEGER,value BLOB,
                PRIMARY KEY(account,kind,id));
            CREATE TABLE request_logs (id INTEGER PRIMARY KEY,account TEXT,time REAL,value BLOB);
            CREATE INDEX log_account ON request_logs(account,time);
        """)
        for index, (vendor, protocol, base, _) in enumerate(records):
            connection.execute(
                "INSERT INTO objects VALUES ('a','upstreams',?,7,?)",
                (
                    str(index),
                    cipher.encrypt(
                        orjson.dumps(
                            {
                                "vendor": vendor,
                                "protocol": protocol,
                                "base_url": base,
                                "api_key": "kept-secret",
                            }
                        )
                    ),
                ),
            )
        connection.execute(
            "INSERT INTO request_logs(account,time,value) VALUES ('a',?,?)",
            (time.time(), cipher.encrypt(orjson.dumps({"request_id": "old"}))),
        )
    return path, key, records


def migrated(values):
    return {v["id"]: (v["base_url"], v["revision"]) for v in values}


def expected(records):
    return {str(i): (r[3], 7 + (r[2] != r[3])) for i, r in enumerate(records)}


@pytest.mark.asyncio
async def test_v5_migration_is_idempotent_and_preserves_encrypted_secrets(v5_database):
    path, key, records = v5_database
    store = ManagementStore(path, key)
    try:
        await store.initialize()
        values = await store.list("a", "upstreams")
        assert migrated(values) == expected(records)
        assert all(v["api_key"] == "kept-secret" for v in values)
        await store.initialize()
        assert await store.list("a", "upstreams") == values
        assert len(await store.logs("a")) == 1
        assert await store.logs("a", user_id="alice") == []
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
            columns = [r[2] for r in connection.execute("PRAGMA index_info(log_account_user)")]
            assert columns == ["account", "user", "time"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_failed_migration_rolls_back_schema_and_objects(v5_database):
    path, key, _ = v5_database
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE objects SET value=? WHERE id='8'", (b"corrupt",))
    store = ManagementStore(path, key)
    try:
        with pytest.raises(InvalidToken):
            await store.initialize()
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
            assert "user" not in [
                r[1] for r in connection.execute("PRAGMA table_info(request_logs)")
            ]
            row = connection.execute("SELECT value,revision FROM objects WHERE id='0'").fetchone()
            assert store.decode(row["value"])["base_url"] == "https://api.openai.com"
            assert row["revision"] == 7
    finally:
        store.close()


@pytest.mark.asyncio
async def test_user_filtering_and_deletion_preserves_other_users(tmp_path):
    key = Fernet.generate_key().decode()
    management = ManagementStore(tmp_path / "management.sqlite3", key)
    kernel = SQLiteKernelStore(tmp_path / "kernel.sqlite3", key)
    try:
        await management.initialize()
        await kernel.initialize()
        await management.save("a", "upstreams", "shared", {"name": "shared"})
        for account, user in [("a", "alice"), ("a", "bob"), ("b", "alice")]:
            await management.save(account, "keys", user, {"user_id": user})
            await management.save(account, "users", user, {"user_id": user})
            await management.save(account, "responses", user, {"user_id": user}, ttl=3600)
            await management.log(account, {"request_id": user}, user_id=user)
            for protocol in ("chat", "responses", "anthropic"):
                scope = digest(account + "\0" + user + "\0" + protocol)
                with kernel.connect() as connection:
                    kernel.touch(connection, scope, "session")
        with management.connect() as connection:
            users = [r[0] for r in connection.execute("SELECT user FROM request_logs ORDER BY id")]
        assert users == ["alice", "bob", "alice"]
        assert len(await management.list("a", "keys", user_id="alice")) == 1
        assert len(await management.logs("a", user_id="alice")) == 1
        await delete_user_data(management, kernel, "a", "alice")
        assert await management.list("a", "keys", user_id="alice") == []
        assert await management.logs("a", user_id="alice") == []
        assert await management.get("a", "responses", "alice") is None
        assert len(await management.logs("a")) == 1
        assert len(await management.logs("b")) == 1
        assert len(await management.list("a", "upstreams")) == 1
        with kernel.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 6
        await delete_account_data(management, kernel, "a")
        assert await management.list("a", "upstreams") == []
        assert await management.logs("a") == []
        assert len(await management.list("b", "keys")) == 1
        with kernel.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 3
    finally:
        management.close()
        kernel.close()


@pytest.mark.asyncio
async def test_workers_can_migrate_same_v5_database(v5_database):
    import asyncio

    path, key, records = v5_database
    workers = [ManagementStore(path, key) for _ in range(6)]
    try:
        await asyncio.gather(*(worker.initialize() for worker in workers))
        for worker in workers:
            assert migrated(await worker.list("a", "upstreams")) == expected(records)
    finally:
        for worker in workers:
            worker.close()


@pytest.mark.asyncio
async def test_v5_kernel_upgrade_keeps_sessions(tmp_path):
    path = tmp_path / "kernel.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            PRAGMA user_version=5;
            CREATE TABLE sessions (scope TEXT,session TEXT,touched REAL NOT NULL,
                PRIMARY KEY(scope,session));
            INSERT INTO sessions VALUES ('scope','session',1000);
        """)
    store = SQLiteKernelStore(path, Fernet.generate_key().decode())
    try:
        await store.initialize()
        with store.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
            assert tuple(connection.execute("SELECT * FROM sessions").fetchone()) == (
                "scope",
                "session",
                1000,
            )
    finally:
        store.close()
