"""Standalone gateway tests intentionally need no vector engine or LLM SDK."""

import copy

import orjson
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from openviking_context_gateway.kernel import MemoryKernel
from openviking_context_gateway.models import Policy
from openviking_context_gateway.storage import SQLiteKernelStore


class FakeViking:
    def __init__(self):
        self.recalls = []
        self.writes = []
        self.write_sessions = []
        self.archive_status = "pending"
        self.live = {}
        self.archived = {}
        self.commits = []
        self.entries = [
            {
                "uri": "viking://user/alice/memories/deploy.md",
                "text": "Deploy using the blue cluster.",
            }
        ]
        self.failure = None
        self.summary = "Previously deployed to the blue cluster."

    async def recall(self, key, query, policy, exclude, budget):
        self.recalls.append((key, query, copy.deepcopy(exclude), budget))
        if self.failure:
            raise self.failure
        return {"entries": [x for x in self.entries if x["uri"] not in exclude]}

    async def health(self, key="", require_identity=True):
        return {"version": "0.4.16", "role": "user", "account_id": "tenant", "user_id": "alice"}

    async def create_session(self, key, session):
        return {}

    async def write(self, key, session, messages):
        self.writes.append(copy.deepcopy(messages))
        self.write_sessions.append(session)
        self.live.setdefault(session, []).extend(copy.deepcopy(messages))
        return {
            "pending_tokens": sum(len(orjson.dumps(m["parts"])) // 3 for m in self.live[session])
        }

    async def capture_status(self, key, session):
        live = self.live.get(session, [])
        index = sum(1 for a in self.archived if f"/{session}/" in a) + 1
        return {
            "messages": copy.deepcopy(live),
            "pending_tokens": sum(len(orjson.dumps(m["parts"])) // 3 for m in live),
            "next_archive_uri": f"viking://user/alice/sessions/{session}/history/archive_{index:03d}",
        }

    async def commit(self, key, session, keep=0):
        self.commits.append((session, keep))
        info = await self.capture_status(key, session)
        uri = info["next_archive_uri"]
        messages = self.live.get(session, [])
        self.archived[uri] = messages[:-keep] if keep else messages
        self.live[session] = messages[-keep:] if keep else []
        return {"archive_uri": uri}

    async def resolve_commit(self, key, session, intent):
        uri = intent["archive_uri"]
        if uri not in self.archived:
            uri = (await self.commit(key, session, intent["keep"]))["archive_uri"]
        return {
            **intent,
            "status": "pending",
            "committed": True,
            "archive_uri": uri,
            "archive_id": uri.rsplit("/", 1)[-1],
        }

    async def archive_state(self, key, session, archive, uri=""):
        return self.archive_status

    async def overview(self, key, session, archive):
        return self.summary


@pytest.fixture
def credential():
    return {
        "account": "tenant",
        "user_id": "alice",
        "id": "key-id",
        "openviking_key": "user-secret",
    }


@pytest_asyncio.fixture
async def setup_kernel(tmp_path):
    key = Fernet.generate_key().decode()
    store = SQLiteKernelStore(tmp_path / "kernel.sqlite3", key)
    await store.initialize()
    viking = FakeViking()
    kernel = MemoryKernel(store, viking)
    yield kernel, store, viking, key
    store.close()


@pytest.fixture
def policy():
    return {**Policy().model_dump(), "id": "policy", "revision": 1}


async def replay_records(store, request, kind):
    values = await store.replay.read(
        request.scope, request.session, ["", *request.chain, *request.body_chain]
    )
    return {key: value for key, value in values.items() if key[0] == kind}


async def update_capture(store, request, **changes):
    from openviking_context_gateway.capture import ready_at

    old = await store.capture.get(request.scope, request.session)
    value = {**old.value, **changes}
    assert await store.capture.swap(request.scope, request.session, old, value, ready_at(value))
    return value


async def make_due(store):
    from openviking_context_gateway.capture import ready_at

    with store.connect() as c:
        keys = list(c.execute("SELECT scope,session FROM capture"))
    for scope, session in keys:
        old = await store.capture.get(scope, session)
        value = copy.deepcopy(old.value)
        for turn in value.get("pending", []):
            turn["ready"] = 0
        if value.get("error"):
            value["error"]["retry_at"] = 0
        if value.get("archive"):
            value["archive"]["next_check"] = 0
        assert await store.capture.swap(scope, session, old, value, ready_at(value))
