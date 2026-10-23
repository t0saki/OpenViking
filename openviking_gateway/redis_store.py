# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Redis adapters: every write that must be atomic is one Lua script on one slot."""

import time
import uuid
from collections import defaultdict
from pathlib import Path

from redis.asyncio.cluster import RedisCluster
from redis.crc import key_slot

from .storage import AsyncCodec, digest

LOG_LIMIT = 10000
INDEXED_KINDS = {"keys", "upstreams", "policies", "users"}


def h(value):
    return digest(value)[:32]


class RedisBackend:
    def __init__(self, client, key_prefix, codec: AsyncCodec):
        self.client, self.prefix, self.codec = client, key_prefix + "v1:", codec
        self.scripts = {}

    def script(self, name):
        if name not in self.scripts:
            self.scripts[name] = self.client.register_script(
                (Path(__file__).parent / "lua" / f"{name}.lua").read_text()
            )
        return self.scripts[name]

    async def scan(self, pattern):
        if not isinstance(self.client, RedisCluster):
            async for key in self.client.scan_iter(match=pattern, count=500):
                yield key
            return
        await self.client.initialize()
        nodes = (
            [self.client.get_node_from_key(pattern)]
            if "{" in pattern
            else self.client.get_primaries()
        )
        for node in nodes:
            cursor = 0
            while True:
                cursors, keys = await self.client.scan(
                    cursor, match=pattern, count=500, target_nodes=node
                )
                for key in keys:
                    yield key
                cursor = cursors[node.name]
                if not cursor:
                    break

    async def unlink(self, keys):
        slots = defaultdict(list)
        for key in keys:
            slots[key_slot(key.encode() if isinstance(key, str) else key)].append(key)
        for batch in slots.values():
            for start in range(0, len(batch), 500):
                # The Cluster convenience method fans out; explicit commands stay in one slot.
                await self.client.execute_command("UNLINK", *batch[start : start + 500])


class RedisManagementStore:
    def __init__(self, backend: RedisBackend, log_ttl=7 * 86400):
        self.backend, self.client, self.codec = backend, backend.client, backend.codec
        self.log_ttl = log_ttl

    def account_prefix(self, account):
        return self.backend.prefix + "m:{a" + h(account) + "}:"

    def object_key(self, account, kind, identifier):
        return self.account_prefix(account) + f"o:{kind}:{h(identifier)}"

    def credential_key(self, identifier):
        return self.backend.prefix + "m:{c" + h(identifier) + "}"

    def object_keys(self, account, kind, identifier):
        keys = [self.object_key(account, kind, identifier)]
        if kind in INDEXED_KINDS:
            keys.append(self.account_prefix(account) + f"ix:{kind}")
        return keys

    async def initialize(self):
        await self.client.ping()

    async def get(self, account, kind, identifier):
        value, revision = await self.client.hmget(
            self.object_key(account, kind, identifier), "v", "rev"
        )
        if value is None:
            return None
        value = await self.codec.decode(value)
        if value.get("expires_at", float("inf")) <= time.time():
            return None
        return {**value, "id": identifier, "revision": int(revision)}

    async def list(self, account, kind, user_id=None):
        identifiers = await self.client.zrange(self.account_prefix(account) + f"ix:{kind}", 0, -1)
        values = []
        for identifier in sorted(identifiers):
            value = await self.get(account, kind, identifier.decode())
            if value is not None and (user_id is None or value.get("user_id") == user_id):
                values.append(value)
        return values

    async def save(self, account, kind, identifier, value, ttl=None):
        if ttl is not None:
            value = {**value, "expires_at": time.time() + ttl}
        revision, remaining = await self.backend.script("obj_save")(
            keys=self.object_keys(account, kind, identifier),
            args=[identifier, await self.codec.encode(value), h(value.get("user_id") or ""),
                  "" if ttl is None else int(ttl * 1000)],
        )
        if kind == "keys" and remaining != -2:
            await self.client.set(
                self.credential_key(identifier), account,
                **({"px": max(1, remaining)} if remaining >= 0 else {}),
            )
        return None if remaining == -2 else {**value, "id": identifier, "revision": revision}

    async def delete(self, account, kind, identifier):
        await self.backend.script("obj_delete")(
            keys=self.object_keys(account, kind, identifier), args=[identifier]
        )
        if kind == "keys":
            await self.client.delete(self.credential_key(identifier))

    async def authenticate(self, key):
        return await self.credential(digest(key))

    async def credential(self, identifier):
        account = await self.client.get(self.credential_key(identifier))
        if account is None:
            return None
        value = await self.get(account.decode(), "keys", identifier)
        if value is None:
            return None
        value.pop("revision")
        return {**value, "account": account.decode()}

    async def log(self, account, value, user_id=None):
        key = self.account_prefix(account) + "log"
        prefix = f"{int(time.time() * 1000)}\0{h(user_id) if user_id is not None else ''}\0"
        async with self.client.pipeline(transaction=False) as pipeline:
            pipeline.lpush(key, prefix.encode() + await self.codec.encode(value))
            pipeline.ltrim(key, 0, LOG_LIMIT - 1)
            pipeline.pexpire(key, int(self.log_ttl * 1000))
            await pipeline.execute()

    async def logs(self, account, limit=200, user_id=None):
        key, values = self.account_prefix(account) + "log", []
        for start in range(0, LOG_LIMIT, 1000):
            if len(values) >= limit:
                break
            rows = await self.client.lrange(key, start, start + 999)
            for row in rows:
                timestamp, user, value = row.split(b"\0", 2)
                if user_id is None or user.decode() == h(user_id):
                    values.append({"time": int(timestamp) / 1000, **await self.codec.decode(value)})
                    if len(values) >= limit:
                        break
            if len(rows) < 1000:
                break
        return values

    async def delete_user(self, account, user_id):
        for kind in ("keys", "users"):
            index = self.account_prefix(account) + f"ix:{kind}"
            async for identifier, _ in self.client.zscan_iter(index, count=500):
                identifier = identifier.decode()
                user = await self.client.hget(self.object_key(account, kind, identifier), "u")
                if user == h(user_id).encode():
                    await self.delete(account, kind, identifier)
        key = self.account_prefix(account) + "log"
        await self.backend.script("log_delete_user")(
            keys=[key, key + ":" + uuid.uuid4().hex], args=[h(user_id)]
        )

    async def delete_account(self, account):
        identifiers = await self.client.zrange(self.account_prefix(account) + "ix:keys", 0, -1)
        batch = []
        async for key in self.backend.scan(self.account_prefix(account) + "*"):
            batch.append(key)
            if len(batch) == 500:
                await self.backend.unlink(batch)
                batch.clear()
        await self.backend.unlink(batch)
        await self.backend.unlink([self.credential_key(identifier.decode()) for identifier in identifiers])

    async def expire_logs(self, before):
        pass
