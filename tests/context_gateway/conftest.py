"""Standalone gateway tests intentionally need no vector engine or LLM SDK."""

import copy

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
        return {"pending_tokens": 22000}

    async def commit(self, key, session, keep=0):
        self.commits.append((session, keep))
        return {"archive_uri": "viking://user/alice/sessions/s/history/archive_001"}

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
    return kernel, store, viking, key


@pytest.fixture
def policy():
    return {**Policy().model_dump(), "id": "policy", "revision": 1}
