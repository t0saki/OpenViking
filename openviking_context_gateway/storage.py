# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Encrypted records and an ordered, leased queue.

The store only implements indexed reads, first-writer inserts, conditional batch
updates and queue leases. Recall budgets and capture reconciliation live in the
kernel. Connections are pooled and used by one thread at a time.
"""

import asyncio
import hashlib
import os
import queue
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
    async def read(self, scope, session, anchors=None, kinds=None) -> dict: ...
    async def lookup(self, scope, anchors, kinds) -> dict: ...
    async def owners(self, scope, anchors, kind) -> dict: ...
    async def put(self, scope, session, kind, anchor, value) -> dict: ...
    async def put_many(self, scope, session, values) -> None: ...
    async def commit(
        self, scope, session, values, *, expected=None, shared=(), jobs=(), cancel=()
    ) -> bool: ...
    async def enqueue(self, scope, session, anchor, value, ready, position=0) -> None: ...
    async def claim(self, lease_seconds=60) -> dict | None: ...
    async def ack(self, item, success, *, retry_at=None, failed=False) -> None: ...
    async def expire(self, before, scope=None) -> None: ...


class Database:
    def __init__(self, path: Path, encryption_key: str):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        # Protect the file before SQLite opens it, not once per operation.
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        self.cipher = Fernet(encryption_key.encode())
        self.pool = queue.LifoQueue(maxsize=8)

    @contextmanager
    def connect(self):
        try:
            connection = self.pool.get_nowait()
        except queue.Empty:
            connection = sqlite3.connect(
                self.path, timeout=30, isolation_level=None, check_same_thread=False
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            if connection.in_transaction:
                connection.rollback()
            try:
                self.pool.put_nowait(connection)
            except queue.Full:
                connection.close()

    def close(self):
        while True:
            try:
                self.pool.get_nowait().close()
            except queue.Empty:
                return

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
                    UNIQUE(scope,session,anchor),
                    FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS queue_ready ON queue(ready,lease);
                CREATE TABLE IF NOT EXISTS deleted_scopes (scope TEXT PRIMARY KEY);
                DROP TABLE IF EXISTS rate_reservations;
            """)
            columns = {r["name"] for r in c.execute("PRAGMA table_info(queue)")}
            for name in ("position", "attempts", "failed"):
                if name not in columns:
                    c.execute(f"ALTER TABLE queue ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0")
            if c.execute("PRAGMA user_version").fetchone()[0] < 4:
                # Rebuild the old cross-session unique key without losing leases
                # or delivery progress. All workers must run the same schema.
                c.executescript("""
                    BEGIN IMMEDIATE;
                    ALTER TABLE queue RENAME TO queue_old;
                    CREATE TABLE queue (
                        id TEXT PRIMARY KEY, scope TEXT, session TEXT, anchor TEXT,
                        value BLOB NOT NULL, ready REAL, lease REAL DEFAULT 0, owner TEXT,
                        position INTEGER NOT NULL DEFAULT 0,
                        attempts INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(scope,session,anchor),
                        FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
                    INSERT INTO queue SELECT id,scope,session,anchor,value,ready,lease,owner,
                        position,attempts,failed FROM queue_old;
                    DROP TABLE queue_old;
                    PRAGMA user_version=4;
                    COMMIT;
                """)
            c.execute("CREATE INDEX IF NOT EXISTS queue_ready ON queue(ready,lease)")
            c.execute("CREATE INDEX IF NOT EXISTS queue_order ON queue(scope,session,position)")

    def _touch(self, c, scope, session):
        if c.execute("SELECT 1 FROM deleted_scopes WHERE scope=?", (scope,)).fetchone():
            raise RuntimeError("Gateway user data has been deleted")
        now = time.time()
        c.execute(
            "INSERT INTO sessions VALUES (?,?,?) ON CONFLICT(scope,session) "
            "DO UPDATE SET touched=excluded.touched WHERE sessions.touched<?",
            (scope, session, now, now - 60),
        )

    async def read(self, scope, session, anchors=None, kinds=None):
        return await self.run(self._read, scope, session, anchors, kinds)

    def _read(self, scope, session, anchors, kinds):
        sql, args = "SELECT kind,anchor,value FROM records WHERE scope=?", [scope]
        if session is not None:
            sql += " AND session=?"
            args.append(session)
        for name, items in (("kind", kinds), ("anchor", anchors)):
            if items is not None:
                sql += f" AND {name} IN (SELECT value FROM json_each(?))"
                args.append(orjson.dumps(items).decode())
        if session is None:
            sql += " ORDER BY session"
        with self.connect() as c:
            result = {}
            for r in c.execute(sql, args):
                key = r["kind"], r["anchor"]
                if key not in result:
                    result[key] = self.decode(r["value"])
            return result

    async def lookup(self, scope, anchors, kinds):
        return await self.read(scope, None, anchors, kinds)

    async def owners(self, scope, anchors, kind):
        """Find record owners without reading/decrypting their values."""

        def read():
            with self.connect() as c:
                result = {}
                for row in c.execute(
                    "SELECT anchor,session FROM records WHERE scope=? AND kind=? "
                    "AND anchor IN (SELECT value FROM json_each(?))",
                    (scope, kind, orjson.dumps(anchors).decode()),
                ):
                    owners = result.setdefault(row["anchor"], [])
                    if len(owners) < 2:
                        owners.append(row["session"])
                return result

        return await self.run(read)

    async def put(self, scope, session, kind, anchor, value):
        def put():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                self._touch(c, scope, session)
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

        return await self.run(put)

    async def put_many(self, scope, session, values):
        def put():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                self._touch(c, scope, session)
                c.executemany(
                    "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)",
                    ((scope, session, k, a, self.encode(v)) for (k, a), v in values.items()),
                )
                c.commit()

        if values:
            await self.run(put)

    async def commit(self, scope, session, values, *, expected=None, shared=(), jobs=(), cancel=()):
        """Atomically compare records, then replace records and pending queue entries.

        A shared key additionally requires an identical value across the scope.
        This is a generic first-writer constraint, independent of record kind.
        """

        def commit():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                self._touch(c, scope, session)
                for (kind, anchor), value in (expected or {}).items():
                    row = c.execute(
                        "SELECT value FROM records WHERE scope=? AND session=? AND kind=? AND anchor=?",
                        (scope, session, kind, anchor),
                    ).fetchone()
                    if (self.decode(row[0]) if row else None) != value:
                        return False
                for kind, anchor in shared:
                    row = c.execute(
                        "SELECT value FROM records WHERE scope=? AND kind=? AND anchor=? LIMIT 1",
                        (scope, kind, anchor),
                    ).fetchone()
                    if row and self.decode(row[0]) != values[kind, anchor]:
                        return False
                for (kind, anchor), value in values.items():
                    if value is None:
                        c.execute(
                            "DELETE FROM records WHERE scope=? AND session=? AND kind=? AND anchor=?",
                            (scope, session, kind, anchor),
                        )
                    else:
                        c.execute(
                            "INSERT INTO records VALUES (?,?,?,?,?) ON CONFLICT(scope,session,kind,anchor) "
                            "DO UPDATE SET value=excluded.value",
                            (scope, session, kind, anchor, self.encode(value)),
                        )
                for anchor in cancel:
                    c.execute(
                        "DELETE FROM queue WHERE scope=? AND session=? AND anchor=? AND ready IS NOT NULL AND lease=0 AND failed=0",
                        (scope, session, anchor),
                    )
                for job in jobs:
                    c.execute(
                        "INSERT INTO queue(id,scope,session,anchor,value,ready,position) VALUES (?,?,?,?,?,?,?) "
                        "ON CONFLICT(scope,session,anchor) DO UPDATE SET value=excluded.value,ready=excluded.ready,position=excluded.position "
                        "WHERE queue.session=excluded.session AND queue.ready IS NOT NULL AND queue.lease=0 AND queue.failed=0",
                        (
                            uuid.uuid4().hex,
                            scope,
                            session,
                            job["anchor"],
                            self.encode(job["value"]),
                            job["ready"],
                            job["position"],
                        ),
                    )
                c.commit()
                return True

        return await self.run(commit)

    async def enqueue(self, scope, session, anchor, value, ready, position=0):
        await self.commit(
            scope,
            session,
            {},
            jobs=[{"anchor": anchor, "value": value, "ready": ready, "position": position}],
        )

    async def claim(self, lease_seconds=60):
        def claim():
            now, owner = time.time(), uuid.uuid4().hex
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute(
                    "SELECT * FROM queue WHERE ready<=? AND lease<? AND failed=0 "
                    "AND NOT EXISTS (SELECT 1 FROM queue q WHERE q.scope=queue.scope AND q.session=queue.session "
                    "AND (q.lease>? OR ((q.ready IS NOT NULL OR q.failed=1) "
                    "AND (q.position<queue.position OR (q.position=queue.position AND q.rowid<queue.rowid))))) "
                    "ORDER BY ready,rowid LIMIT 1",
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

        return await self.run(claim)

    async def ack(self, item, success, *, retry_at=None, failed=False):
        def ack():
            with self.connect() as c:
                c.execute(
                    "UPDATE queue SET ready=?,lease=0,owner=NULL,attempts=attempts+?,failed=? WHERE id=? AND owner=?",
                    (
                        None if success or failed else retry_at,
                        int(not success),
                        int(failed),
                        item["id"],
                        item["owner"],
                    ),
                )

        await self.run(ack)

    async def expire(self, before, scope=None):
        def expire():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                if scope is not None:
                    c.execute("INSERT OR IGNORE INTO deleted_scopes VALUES (?)", (scope,))
                    c.execute("DELETE FROM sessions WHERE scope=?", (scope,))
                else:
                    c.execute("DELETE FROM sessions WHERE touched<?", (before,))
                c.commit()

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
                    CREATE INDEX IF NOT EXISTS object_identity ON objects(kind,id);
                    CREATE TABLE IF NOT EXISTS request_logs (
                        id INTEGER PRIMARY KEY, account TEXT, time REAL, value BLOB);
                    CREATE INDEX IF NOT EXISTS log_account ON request_logs(account,time);
                    CREATE TABLE IF NOT EXISTS object_expiry (
                        account TEXT, kind TEXT, id TEXT, expires REAL,
                        PRIMARY KEY(account,kind,id));
                    CREATE INDEX IF NOT EXISTS expiry_time ON object_expiry(expires);
                    PRAGMA user_version=2;
                """)
                # Preserve pre-TTL response routes for one bounded migration window.
                c.execute(
                    "INSERT OR IGNORE INTO object_expiry SELECT account,kind,id,? "
                    "FROM objects WHERE kind='responses'",
                    (time.time() + 30 * 86400,),
                )

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
        def read():
            with self.connect() as c:
                row = c.execute(
                    "SELECT o.value,o.revision,e.expires FROM objects o LEFT JOIN object_expiry e "
                    "ON (o.account,o.kind,o.id)=(e.account,e.kind,e.id) "
                    "WHERE o.account=? AND o.kind=? AND o.id=?",
                    (account, kind, identifier),
                ).fetchone()
                if not row or (row["expires"] is not None and row["expires"] <= time.time()):
                    return None
                value = self.decode(row["value"])
                if value.get("expires_at", float("inf")) <= time.time():
                    return None
                return {**value, "id": identifier, "revision": row["revision"]}

        return await self.run(read)

    async def save(self, account, kind, identifier, value, ttl=None):
        if ttl is not None:
            value = {**value, "expires_at": time.time() + ttl}

        def save():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                c.execute(
                    "INSERT INTO objects VALUES (?,?,?,1,?) ON CONFLICT(account,kind,id) "
                    "DO UPDATE SET revision=revision+1,value=excluded.value",
                    (account, kind, identifier, self.encode(value)),
                )
                if ttl is not None:
                    c.execute(
                        "INSERT OR REPLACE INTO object_expiry VALUES (?,?,?,?)",
                        (account, kind, identifier, value["expires_at"]),
                    )
                c.commit()

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
                c.execute("BEGIN IMMEDIATE")
                c.execute("DELETE FROM request_logs WHERE time<?", (before,))
                c.execute(
                    "DELETE FROM objects WHERE (account,kind,id) IN "
                    "(SELECT account,kind,id FROM object_expiry WHERE expires<=?)",
                    (time.time(),),
                )
                c.execute("DELETE FROM object_expiry WHERE expires<=?", (time.time(),))
                c.commit()

        await self.run(expire)

    async def delete_account(self, account):
        def delete():
            with self.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                c.execute("DELETE FROM objects WHERE account=?", (account,))
                c.execute("DELETE FROM request_logs WHERE account=?", (account,))
                c.execute("DELETE FROM object_expiry WHERE account=?", (account,))
                c.commit()

        await self.run(delete)
