# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Redis adapters: every write that must be atomic is one Lua script on one slot."""

import asyncio
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path

import orjson
from redis.asyncio.cluster import RedisCluster
from redis.crc import key_slot
from redis.exceptions import ResponseError

from .capture_store import Document, LeaseLost
from .records import RecordKind as K
from .replay_store import INHERITED
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
            args=[
                identifier,
                await self.codec.encode(value),
                h(value.get("user_id") or ""),
                "" if ttl is None else int(ttl * 1000),
            ],
        )
        if kind == "keys" and remaining != -2:
            await self.client.set(
                self.credential_key(identifier),
                account,
                **({"px": max(1, remaining)} if remaining >= 0 else {}),
            )
        return None if remaining == -2 else {**value, "id": identifier, "revision": revision}

    async def delete(self, account, kind, identifier):
        await self.backend.script("obj_delete")(
            keys=self.object_keys(account, kind, identifier), args=[identifier]
        )
        if kind == "keys":
            await self.client.execute_command("DEL", self.credential_key(identifier))

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
        await self.backend.script("log_push")(
            keys=[key],
            args=[
                prefix.encode() + await self.codec.encode(value),
                LOG_LIMIT,
                int(self.log_ttl * 1000),
            ],
        )

    async def logs(self, account, limit=200, user_id=None):
        key, values = self.account_prefix(account) + "log", []
        for start in range(0, LOG_LIMIT, 1000):
            if len(values) >= limit:
                break
            rows = await self.client.lrange(key, start, start + 999)
            for row in rows:
                timestamp, user, value = row.split(b"\0", 2)
                if int(timestamp) / 1000 < time.time() - self.log_ttl:
                    continue
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
        await self.backend.unlink(
            [self.credential_key(identifier.decode()) for identifier in identifiers]
        )

    async def expire_logs(self, before):
        pass


class RedisKernelStore:
    def __init__(self, backend, ttl=7 * 86400):
        self.backend, self.client, self.codec = backend, backend.client, backend.codec
        self.prefix, self.ttl = backend.prefix, max(1, int(ttl * 1000))
        self.replay, self.state, self.capture = (
            RedisReplayStore(self),
            RedisStateStore(self),
            RedisCaptureQueue(self),
        )
        self.touched = OrderedDict()

    async def initialize(self):
        await self.client.ping()

    def tombstone(self, scope, capture=False):
        return f"{self.prefix}kd:{{cap}}:{scope}" if capture else f"{self.prefix}kd:{{{scope}}}"

    def replay_key(self, scope, session):
        return f"{self.prefix}k:{{{scope}}}:rp:{h(session)}"

    def state_key(self, scope, key):
        return f"{self.prefix}k:{{{scope}}}:st:{h(key)}"

    def capture_key(self, scope, session):
        return f"{self.prefix}k:{{cap}}:{scope}:{h(session)}"

    async def script(self, name, keys, args):
        try:
            return await self.backend.script(name)(keys=keys, args=args)
        except ResponseError as error:
            if "OVGW_DELETED" in str(error):
                raise RuntimeError("Gateway user data has been deleted") from error
            raise

    async def touch(self, key):
        now = time.monotonic()
        if now - self.touched.get(key, float("-inf")) < 60:
            return
        self.touched[key] = now
        self.touched.move_to_end(key)
        if len(self.touched) > 2048:
            self.touched.popitem(last=False)
        try:
            await self.client.pexpire(key, self.ttl)
        except BaseException:
            self.touched.pop(key, None)
            raise

    async def load(self, scope, session, anchors, ancestors=()):
        replay, document = await asyncio.gather(
            self.replay.load(scope, session, anchors, ancestors, state=True),
            self.capture.get(scope, session),
        )
        return replay[0], replay[1], document

    async def keep(self, scope, sessions, keys):
        sessions = list(sessions)
        await self.script(
            "kernel_keep",
            [
                self.tombstone(scope),
                *[self.replay_key(scope, sid) for sid in sessions],
                *[self.state_key(scope, key) for key in keys],
            ],
            [self.ttl],
        )
        await asyncio.gather(*(self.touch(self.capture_key(scope, sid)) for sid in sessions))

    async def expire(self, before, scope=None):
        if scope is None:
            return
        await self.client.set(self.tombstone(scope), 1)
        await self.client.set(self.tombstone(scope, capture=True), 1)
        for pattern in (f"{self.prefix}k:{{{scope}}}:*", f"{self.prefix}k:{{cap}}:{scope}:*"):
            batch = []
            async for key in self.backend.scan(pattern):
                batch.append(key)
                if len(batch) == 256:
                    await self.backend.unlink(batch)
                    batch = []
            if batch:
                await self.backend.unlink(batch)

    async def allow_scope(self, scope):
        await self.client.execute_command("DEL", self.tombstone(scope))
        await self.client.execute_command("DEL", self.tombstone(scope, capture=True))


class RedisReplayStore:
    def __init__(self, store):
        self.store = store

    def index(self, scope, anchor):
        return f"{self.store.prefix}k:{{{scope}}}:r:{anchor}"

    async def load(self, scope, session, anchors, ancestors=(), state=False):
        owners = [*sorted(set(ancestors) - {session}), session]
        keys = [self.store.replay_key(scope, sid) for sid in owners]
        if state:
            keys.append(self.store.state_key(scope, session))
        rows, observation = await self.store.script(
            "kernel_load",
            keys,
            [len(owners), orjson.dumps(list(anchors)), orjson.dumps(sorted(INHERITED)), int(state)],
        )
        values = {}
        for field, value in rows:
            kind, anchor = field.decode().split(":", 1)
            values[kind, anchor] = await self.store.codec.decode(value)
        states = {}
        if observation:
            states[session] = Document(
                await self.store.codec.decode(observation[0]), int(observation[1])
            )
        return values, states

    async def read(self, scope, session, anchors, ancestors=()):
        return (await self.load(scope, session, anchors, ancestors))[0]

    async def replies(self, scope, anchors):
        owners = {}
        for anchor in anchors:
            index = self.index(scope, anchor)
            members = await self.store.client.smembers(index)
            if not members:
                continue
            members = sorted(members)
            live = await self.store.script(
                "kernel_replies_prune",
                [index, *[self.store.replay_key(scope, sid.decode()) for sid in members]],
                ["reply:" + anchor, self.store.ttl, *members],
            )
            if live:
                owners[anchor] = {sid.decode() for sid in live}
        return owners

    async def put(self, scope, session, kind, anchor, value):
        kind = K(kind)
        keys = [self.store.tombstone(scope), self.store.replay_key(scope, session)]
        if kind == K.REPLY:
            keys.append(self.index(scope, anchor))
        result = await self.store.script(
            "kernel_replay_put",
            keys,
            [
                f"{kind.value}:{anchor}",
                await self.store.codec.encode(value),
                self.store.ttl,
                session,
            ],
        )
        await self.store.touch(self.store.capture_key(scope, session))
        return await self.store.codec.decode(result)


class RedisStateStore:
    def __init__(self, store):
        self.store = store

    async def read(self, scope, keys):
        keys = list(keys)
        if not keys:
            return {}
        rows = await self.store.script(
            "kernel_state_read", [self.store.state_key(scope, key) for key in keys], []
        )
        return {
            keys[index - 1]: Document(await self.store.codec.decode(value), int(version))
            for index, value, version in rows
        }

    async def swap(self, scope, key, previous, value):
        return bool(
            await self.store.script(
                "kernel_state_swap",
                [self.store.tombstone(scope), self.store.state_key(scope, key)],
                [previous.version, await self.store.codec.encode(value), self.store.ttl],
            )
        )


class RedisCaptureQueue:
    def __init__(self, store):
        self.store = store
        self.ready = f"{store.prefix}k:{{cap}}:ready"

    def keys(self, scope, session):
        return [
            self.store.tombstone(scope, capture=True),
            self.store.capture_key(scope, session),
            self.ready,
        ]

    async def get(self, scope, session):
        value, version = await self.store.client.hmget(
            self.store.capture_key(scope, session), "v", "ver"
        )
        return (
            Document(await self.store.codec.decode(value), int(version))
            if value is not None
            else Document()
        )

    async def swap(self, scope, session, previous, value, ready, owner=None):
        result = await self.store.script(
            "kernel_capture_swap",
            self.keys(scope, session),
            [
                previous.version,
                await self.store.codec.encode(value),
                "" if ready is None else repr(ready),
                owner or "",
                self.store.ttl,
                scope + "\0" + session,
            ],
        )
        if result == [-1]:
            raise LeaseLost
        if result:
            await self.store.touch(self.store.replay_key(scope, session))
        return bool(result)

    async def claim(self, lease_seconds=120):
        candidates = await self.store.client.zrangebyscore(
            self.ready, "-inf", time.time() * 1000, start=0, num=8
        )
        for member in candidates:
            scope, session = member.decode().split("\0", 1)
            owner = uuid.uuid4().hex
            result = await self.store.script(
                "kernel_capture_claim",
                self.keys(scope, session),
                [owner, lease_seconds * 1000, member],
            )
            if result:
                return {
                    "scope": scope,
                    "session": session,
                    "owner": owner,
                    "document": Document(await self.store.codec.decode(result[0]), int(result[1])),
                }
        return None

    async def release(self, item):
        await self.store.script(
            "kernel_capture_release",
            self.keys(item["scope"], item["session"]),
            [item["owner"], item["scope"] + "\0" + item["session"]],
        )
