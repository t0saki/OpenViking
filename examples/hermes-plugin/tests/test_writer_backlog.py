"""Failed turn uploads are kept and resent in order, through the public hooks only."""

import logging

import pytest


class FakeServer:
    """One OpenViking server as seen by one identity. ``fail`` is None (up), "down" or an HTTP status."""

    def __init__(self, http_error):
        self.http_error = http_error
        self.fail = None
        self.pending = {}
        self.commits = []
        self.requests = []

    def _check(self):
        if self.fail == "down":
            raise ConnectionError("connection refused")
        if self.fail is not None:
            raise self.http_error(f"HTTP {self.fail}", self.fail)

    def post(self, path, payload=None, **_kwargs):
        self.requests.append(path)
        self._check()
        sid = path.split("/")[4]
        if path.endswith("/commit"):
            self.commits.append((sid, self.pending.pop(sid, [])))
            return {"result": {}}
        self.pending.setdefault(sid, []).extend(payload["messages"] if path.endswith("/batch") else [payload])
        return {"result": {}}

    def get(self, path, **_kwargs):
        self._check()
        return {"result": {"pending_tokens": 10 if self.pending.get(path.split("/")[4]) else 0}}

    def texts(self, sid):
        return [m["parts"][0]["text"] for m in self.pending.get(sid, [])]

    def committed(self, sid):
        return [[m["parts"][0]["text"] for m in messages] for s, messages in self.commits if s == sid]


@pytest.fixture
def writer(external_provider, monkeypatch, inject_deps, core_module):
    home, provider, module, _ = external_provider("writer-backlog")
    monkeypatch.setenv("HERMES_HOME", str(home))
    http_error = core_module(module, "http")._OpenVikingHTTPError
    servers = {name: FakeServer(http_error) for name in ("alice", "bob")}

    class Client:
        def __init__(self, endpoint, api_key="", *, account="", user="", agent=""):
            self._conn_snapshot = (endpoint, api_key, account, user, agent)
            self.get = servers[user].get
            self.post = servers[user].post

    inject_deps(module, provider, client=Client, health=lambda *_: ("healthy", ""))

    def connect(user):
        monkeypatch.setenv("OPENVIKING_USER", user)
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:19531")
        monkeypatch.setenv("OPENVIKING_API_KEY", "private-test-key")
        if not provider._env_refresh_enabled:
            provider.initialize("sid-1", hermes_home=str(home))
        else:
            assert provider._ensure_client() is not None

    connect("alice")
    yield provider, servers, connect, core_module(module, "session_writer")
    provider.shutdown()


def turn(provider, text, sid="sid-1"):
    provider.sync_turn(f"u{text}", f"a{text}", session_id=sid)
    assert provider._drain_writers(sid, timeout=5)
    assert provider._drain_finalizers(timeout=5)


def pair(*turns):
    return [t for n in turns for t in (f"u{n}", f"a{n}")]


def test_server_down_then_back_keeps_every_turn_in_order(writer):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    alice.fail = "down"
    for n in (1, 2, 3):
        turn(provider, n)
    assert alice.texts("sid-1") == []
    alice.fail = None
    turn(provider, 4)
    assert alice.texts("sid-1") == pair(1, 2, 3, 4)
    provider.on_session_end([])
    assert alice.committed("sid-1") == [pair(1, 2, 3, 4)]


def test_retryable_status_is_backlogged_and_resent(writer):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    alice.fail = 503
    turn(provider, 1)
    alice.fail = None
    turn(provider, 2)
    assert alice.texts("sid-1") == pair(1, 2)


def test_commit_waits_for_backlog(writer):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    turn(provider, 1)
    alice.fail = "down"
    turn(provider, 2)
    marker = provider._state_path("pending", "sid-1")
    provider.on_session_end([])
    assert alice.commits == []
    assert marker.exists()
    alice.fail = None
    alice.requests.clear()
    provider.on_session_end([])
    # The backlog goes out before the commit, and the commit covers it.
    assert alice.requests == ["/api/v1/sessions/sid-1/messages/batch", "/api/v1/sessions/sid-1/commit"]
    assert alice.committed("sid-1") == [pair(1, 2)]
    assert not marker.exists()


def test_session_switch_with_backlog_commits_old_session_after_resend(writer):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    alice.fail = "down"
    turn(provider, 1, sid="sid-1")
    provider.on_session_switch("sid-2")
    assert provider._drain_finalizers(timeout=5)
    assert alice.commits == []
    assert provider._state_path("pending", "sid-1").exists()
    alice.fail = None
    turn(provider, 2, sid="sid-2")
    assert alice.committed("sid-1") == [pair(1)]
    assert alice.texts("sid-2") == pair(2)
    assert not provider._state_path("pending", "sid-1").exists()


def test_backlog_keeps_its_identity_across_reload(writer):
    provider, servers, connect, _ = writer
    alice, bob = servers["alice"], servers["bob"]
    alice.fail = "down"
    turn(provider, 1)
    connect("bob")
    alice.fail = None
    turn(provider, 2)
    # Alice's retryable backlog is sent with Alice's client, never with Bob's.
    assert alice.texts("sid-1") == pair(1)
    assert bob.texts("sid-1") == pair(2)


def test_auth_failure_is_not_retried_and_resent_within_the_same_connection(writer):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    alice.fail = 401
    turn(provider, 1)
    # No immediate retry and no per-message fallback.
    assert alice.requests == ["/api/v1/sessions/sid-1/messages/batch"]
    alice.fail = None
    turn(provider, 2)
    assert alice.texts("sid-1") == pair(1, 2)


def test_auth_failure_backlog_is_dropped_when_credentials_change(writer, caplog):
    provider, servers, connect, _ = writer
    alice, bob = servers["alice"], servers["bob"]
    alice.fail = 403
    turn(provider, 1)
    connect("bob")
    alice.fail = None
    with caplog.at_level(logging.WARNING):
        turn(provider, 2)
        provider.on_session_end([])
    assert alice.texts("sid-1") == []
    assert bob.committed("sid-1") == [pair(2)]
    assert any("rejected with 401/403" in r.getMessage() for r in caplog.records)


def test_other_client_errors_are_dropped_with_a_warning(writer, caplog):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    alice.fail = 400
    with caplog.at_level(logging.WARNING):
        turn(provider, 1)
    alice.fail = None
    turn(provider, 2)
    assert alice.texts("sid-1") == pair(2)


def test_backlog_bound_drops_oldest_with_warning(writer, monkeypatch, caplog):
    provider, servers, _, session_writer = writer
    alice = servers["alice"]
    monkeypatch.setattr(session_writer, "_BACKLOG_MAX_MESSAGES", 4)
    alice.fail = "down"
    with caplog.at_level(logging.WARNING):
        for n in (1, 2, 3):
            turn(provider, n)
    assert any("dropped the 2 oldest" in r.getMessage() for r in caplog.records)
    alice.fail = None
    turn(provider, 4)
    assert alice.texts("sid-1") == pair(2, 3, 4)


def test_backlog_is_resent_in_batches_of_at_most_100(writer):
    provider, servers, _, _ = writer
    alice = servers["alice"]
    alice.fail = "down"
    for n in range(60):
        turn(provider, n)
    alice.fail = None
    alice.requests.clear()
    turn(provider, "last")
    assert alice.texts("sid-1") == pair(*range(60), "last")
    assert alice.requests == ["/api/v1/sessions/sid-1/messages/batch"] * 3
