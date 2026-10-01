# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Encrypted, cross-process storage. SQLite connections never cross threads.

KernelStore is the replaceable boundary for Redis/internal KV implementations.
All correctness decisions are atomic in storage; no process-local session locks.
"""

import asyncio
import hashlib
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import orjson
from cryptography.fernet import Fernet


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class KernelStore(Protocol):
    async def read(self, scope: str, session: str, anchors: list[str]) -> dict: ...
    async def put(self, scope: str, session: str, kind: str, anchor: str, value: dict) -> dict: ...
    async def enqueue(
        self, scope: str, session: str, anchor: str, value: dict, ready: float
    ) -> None: ...
    async def claim(self, lease_seconds: float = 60) -> dict | None: ...
    async def ack(self, item: dict, success: bool) -> None: ...
    async def reconcile(self, scope: str, session: str, anchors: list[str]) -> None: ...
    async def expire(self, before: float, scope: str | None = None) -> None: ...


class Database:
    def __init__(self, path: Path, encryption_key: str):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        self.cipher = Fernet(encryption_key.encode())

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        os.chmod(self.path, 0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def encode(self, value: Any) -> bytes:
        return self.cipher.encrypt(orjson.dumps(value))

    def decode(self, value: bytes) -> Any:
        return orjson.loads(self.cipher.decrypt(value))

    async def run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)


class SQLiteKernelStore(Database):
    async def initialize(self):
        await self.run(self._initialize)

    def _initialize(self):
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    scope TEXT, session TEXT, touched REAL NOT NULL,
                    PRIMARY KEY(scope,session));
                CREATE TABLE IF NOT EXISTS records (
                    scope TEXT, session TEXT, kind TEXT, anchor TEXT, value BLOB NOT NULL,
                    PRIMARY KEY(scope,session,kind,anchor),
                    FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS prefix_lookup ON records(scope,kind,anchor);
                CREATE TABLE IF NOT EXISTS queue (
                    id TEXT PRIMARY KEY, scope TEXT, session TEXT, anchor TEXT,
                    value BLOB NOT NULL, ready REAL, lease REAL DEFAULT 0, owner TEXT,
                    UNIQUE(scope,anchor),
                    FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS queue_ready ON queue(ready,lease);
                CREATE TABLE IF NOT EXISTS deleted_scopes (scope TEXT PRIMARY KEY);
                PRAGMA user_version=1;
            """)

    def _touch(self, c, scope, session):
        if c.execute("SELECT 1 FROM deleted_scopes WHERE scope=?", (scope,)).fetchone():
            raise RuntimeError("Gateway user data has been deleted")
        c.execute(
            "INSERT INTO sessions VALUES (?,?,?) ON CONFLICT(scope,session) "
            "DO UPDATE SET touched=excluded.touched",
            (scope, session, time.time()),
        )

    async def read(self, scope, session, anchors):
        return await self.run(self._read, scope, session, anchors)

    def _read(self, scope, session, anchors):
        with self.connect() as c:
            self._touch(c, scope, session)
            rows = c.execute(
                "SELECT * FROM records WHERE scope=? AND session=?", (scope, session)
            ).fetchall()
            result = {(r["kind"], r["anchor"]): self.decode(r["value"]) for r in rows}
            # One indexed join, independent of history length. Never cross a user/protocol scope.
            c.execute("CREATE TEMP TABLE wanted(anchor TEXT PRIMARY KEY)")
            c.executemany("INSERT OR IGNORE INTO wanted VALUES (?)", ((a,) for a in anchors))
            inherited = c.execute(
                "SELECT r.* FROM records r JOIN wanted w ON r.anchor=w.anchor "
                "WHERE r.scope=? AND r.kind IN ('injection','hidden') "
                "ORDER BY r.session",
                (scope,),
            ).fetchall()
            for row in inherited:
                result.setdefault((row["kind"], row["anchor"]), self.decode(row["value"]))
            return result

    async def put(self, scope, session, kind, anchor, value):
        return await self.run(self._put, scope, session, kind, anchor, value)

    def _put(self, scope, session, kind, anchor, value):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            self._touch(c, scope, session)
            if kind == "injection":
                prior = c.execute(
                    "SELECT value FROM records WHERE scope=? AND kind=? AND anchor=? LIMIT 1",
                    (scope, kind, anchor),
                ).fetchone()
                if prior:
                    value = self.decode(prior[0])
                else:
                    value = dict(value)
                    budget = value.pop("_budget", None)
                    used = sum(
                        self.decode(row[0]).get("tokens", 0)
                        for row in c.execute(
                            "SELECT value FROM records WHERE scope=? AND session=? AND kind='injection'",
                            (scope, session),
                        )
                    )
                    if budget is not None and used + value.get("tokens", 0) > budget:
                        value = {"text": "", "uris": [], "tokens": 0, "reason": "session_budget"}
            c.execute(
                "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)",
                (scope, session, kind, anchor, self.encode(value)),
            )
            row = c.execute(
                "SELECT value FROM records WHERE scope=? AND session=? AND kind=? AND anchor=?",
                (scope, session, kind, anchor),
            ).fetchone()
            c.commit()
            return self.decode(row[0])

    async def enqueue(self, scope, session, anchor, value, ready):
        await self.run(self._enqueue, scope, session, anchor, value, ready)

    def _enqueue(self, scope, session, anchor, value, ready):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            self._touch(c, scope, session)
            # Done rows are retained for user-wide dedup, including copied conversations.
            if not value.get("confirmed") and value.get("user_anchor"):
                for row in c.execute(
                    "SELECT id,value FROM queue WHERE scope=? AND session=? AND ready IS NOT NULL AND lease=0",
                    (scope, session),
                ).fetchall():
                    previous = self.decode(row["value"])
                    if (
                        not previous.get("confirmed")
                        and previous.get("user_anchor") == value["user_anchor"]
                    ):
                        c.execute("DELETE FROM queue WHERE id=?", (row["id"],))
            c.execute(
                "INSERT OR IGNORE INTO queue(id,scope,session,anchor,value,ready) VALUES (?,?,?,?,?,?)",
                (uuid.uuid4().hex, scope, session, anchor, self.encode(value), ready),
            )
            if value.get("confirmed"):
                c.execute(
                    "UPDATE queue SET ready=?,value=? WHERE scope=? AND anchor=? AND ready IS NOT NULL AND lease=0",
                    (ready, self.encode(value), scope, anchor),
                )
            c.commit()

    async def claim(self, lease_seconds=60):
        return await self.run(self._claim, lease_seconds)

    def _claim(self, lease_seconds):
        now, owner = time.time(), uuid.uuid4().hex
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT * FROM queue WHERE ready<=? AND lease<? "
                "AND NOT EXISTS (SELECT 1 FROM queue q WHERE q.scope=queue.scope "
                "AND q.session=queue.session AND q.lease>?) ORDER BY ready,rowid LIMIT 1",
                (now, now, now),
            ).fetchone()
            if row is None:
                c.commit()
                return None
            c.execute(
                "UPDATE queue SET lease=?,owner=? WHERE id=?",
                (now + lease_seconds, owner, row["id"]),
            )
            c.commit()
            return {**dict(row), "owner": owner, "payload": self.decode(row["value"])}

    async def ack(self, item, success):
        await self.run(self._ack, item, success)

    async def reconcile(self, scope, session, anchors):
        def reconcile():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                for row in c.execute(
                    "SELECT * FROM queue WHERE scope=? AND session=? AND ready IS NOT NULL AND lease=0",
                    (scope, session),
                ).fetchall():
                    value = self.decode(row["value"])
                    if not value.get("confirmed"):
                        if row["anchor"] in anchors:
                            value["confirmed"] = True
                            c.execute(
                                "UPDATE queue SET ready=?,value=? WHERE id=?",
                                (time.time(), self.encode(value), row["id"]),
                            )
                        else:
                            c.execute("DELETE FROM queue WHERE id=?", (row["id"],))
                c.commit()

        await self.run(reconcile)

    def _ack(self, item, success):
        with self.connect() as c:
            c.execute(
                "UPDATE queue SET ready=?,lease=0,owner=NULL WHERE id=? AND owner=?",
                (None if success else time.time() + 10, item["id"], item["owner"]),
            )

    async def expire(self, before, scope=None):
        def expire():
            with self.connect() as c:
                if scope is not None:
                    c.execute("INSERT OR IGNORE INTO deleted_scopes VALUES (?)", (scope,))
                    c.execute("DELETE FROM sessions WHERE scope=?", (scope,))
                else:
                    c.execute("DELETE FROM sessions WHERE touched<?", (before,))

        await self.run(expire)

    async def allow_scope(self, scope):
        def allow():
            with self.connect() as c:
                c.execute("DELETE FROM deleted_scopes WHERE scope=?", (scope,))

        await self.run(allow)


class ManagementStore(Database):
    async def initialize(self):
        def initialize():
            with self.connect() as c:
                c.execute("PRAGMA journal_mode=WAL")
                c.executescript("""
                    CREATE TABLE IF NOT EXISTS objects (
                        account TEXT, kind TEXT, id TEXT, revision INTEGER, value BLOB,
                        PRIMARY KEY(account,kind,id));
                    CREATE TABLE IF NOT EXISTS request_logs (
                        id INTEGER PRIMARY KEY, account TEXT, time REAL, value BLOB);
                    CREATE INDEX IF NOT EXISTS log_account ON request_logs(account,time);
                    PRAGMA user_version=1;
                """)

        await self.run(initialize)

    async def list(self, account, kind):
        def read():
            with self.connect() as c:
                return [
                    {**self.decode(r["value"]), "id": r["id"], "revision": r["revision"]}
                    for r in c.execute(
                        "SELECT * FROM objects WHERE account=? AND kind=? ORDER BY id",
                        (account, kind),
                    )
                ]

        return await self.run(read)

    async def get(self, account, kind, identifier):
        values = await self.list(account, kind)
        return next((v for v in values if v["id"] == identifier), None)

    async def save(self, account, kind, identifier, value):
        def save():
            with self.connect() as c:
                c.execute(
                    "INSERT INTO objects VALUES (?,?,?,1,?) ON CONFLICT(account,kind,id) "
                    "DO UPDATE SET revision=revision+1,value=excluded.value",
                    (account, kind, identifier, self.encode(value)),
                )

        await self.run(save)
        return await self.get(account, kind, identifier)

    async def delete(self, account, kind, identifier):
        def delete():
            with self.connect() as c:
                c.execute(
                    "DELETE FROM objects WHERE account=? AND kind=? AND id=?",
                    (account, kind, identifier),
                )

        await self.run(delete)

    async def authenticate(self, key):
        identifier = digest(key)

        def read():
            with self.connect() as c:
                row = c.execute(
                    "SELECT account,value FROM objects WHERE kind='keys' AND id=?", (identifier,)
                ).fetchone()
                return (
                    {**self.decode(row["value"]), "account": row["account"], "id": identifier}
                    if row
                    else None
                )

        return await self.run(read)

    async def log(self, account, value):
        def log():
            with self.connect() as c:
                c.execute(
                    "INSERT INTO request_logs(account,time,value) VALUES (?,?,?)",
                    (account, time.time(), self.encode(value)),
                )

        await self.run(log)

    async def logs(self, account, limit=200):
        def read():
            with self.connect() as c:
                return [
                    {"time": r["time"], **self.decode(r["value"])}
                    for r in c.execute(
                        "SELECT time,value FROM request_logs WHERE account=? ORDER BY id DESC LIMIT ?",
                        (account, limit),
                    )
                ]

        return await self.run(read)

    async def expire_logs(self, before):
        def expire():
            with self.connect() as c:
                c.execute("DELETE FROM request_logs WHERE time<?", (before,))

        await self.run(expire)

    async def delete_account(self, account):
        def delete():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                c.execute("DELETE FROM objects WHERE account=?", (account,))
                c.execute("DELETE FROM request_logs WHERE account=?", (account,))
                c.commit()

        await self.run(delete)
