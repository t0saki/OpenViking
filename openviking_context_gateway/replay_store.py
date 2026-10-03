# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Immutable replay storage: indexed batch reads and put-if-absent only."""

from typing import Protocol

import orjson

from .records import RecordKind as K

SHARED = {K.INJECTION, K.HIDDEN}


class ReplayStore(Protocol):
    async def read(self, scope, session, anchors) -> dict: ...
    async def put(self, scope, session, kind: K, anchor, value) -> dict: ...


class SQLiteReplayStore:
    def __init__(self, db):
        self.db = db

    def read_in(self, c, scope, session, anchors):
        return {
            (row["kind"], row["anchor"]): self.db.decode(row["value"])
            for row in c.execute(
                "SELECT kind,anchor,value FROM replay WHERE scope=? AND session IN (?, '*') "
                "AND anchor IN (SELECT value FROM json_each(?))",
                (scope, session, orjson.dumps(anchors).decode()),
            )
        }

    async def read(self, scope, session, anchors):
        def read():
            with self.db.connect() as c:
                return self.read_in(c, scope, session, anchors)

        return await self.db.run(read)

    async def put(self, scope, session, kind, anchor, value):
        kind = K(kind)
        owner = "*" if kind in SHARED else session

        def put():
            with self.db.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                self.db.touch(c, scope, owner)
                c.execute(
                    "INSERT OR IGNORE INTO replay VALUES (?,?,?,?,?)",
                    (scope, owner, kind, anchor, self.db.encode(value)),
                )
                row = c.execute(
                    "SELECT value FROM replay WHERE scope=? AND session=? AND kind=? AND anchor=?",
                    (scope, owner, kind, anchor),
                ).fetchone()
                c.commit()
                return self.db.decode(row[0])

        return await self.db.run(put, write=True)
