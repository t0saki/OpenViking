"""Records follow the conversation whose replies a request continues, and expire with it."""

import time

import orjson
import pytest
from aiohttp import web
from conftest import make_due
from test_review_regressions import worker_for

from openviking_context_gateway.protocols import ResponseCapture
from openviking_context_gateway.records import RecordKind as K
from openviking_context_gateway.state_store import get_state
from openviking_context_gateway.storage import SQLiteKernelStore
from openviking_context_gateway.tool_protocols import hidden_chain

QUESTION = {"role": "user", "content": "How do I deploy?"}


async def prepare(kernel, credential, policy, messages, session=None, protocol="chat", **upstream):
    return await kernel.prepare(
        {"model": "model", "messages": messages},
        protocol,
        {"x-openviking-session": session} if session else {},
        credential,
        {"id": "upstream", **upstream},
        policy,
    )


async def test_same_opening_in_another_conversation_inherits_nothing(
    setup_kernel, credential, policy
):
    kernel, _, viking, _ = setup_kernel
    viking.profile = "Alice deploys on Fridays."
    a = await prepare(kernel, credential, policy, [QUESTION], "a")
    assert "Deploy using the blue cluster." in a.body["messages"][0]["content"]
    policy.update(recall=False, profile=False)
    b = await prepare(kernel, credential, policy, [QUESTION], "b", id="other")
    content = b.body["messages"][0]["content"]
    assert "blue cluster" not in content and "Fridays" not in content
    assert b.records[K.INJECTION, b.chain[0]]["reason"] == "disabled"


async def test_forked_conversation_replays_the_replies_it_continues(
    setup_kernel, credential, policy
):
    kernel, store, viking, _ = setup_kernel
    policy.update(gateway_tools=True)
    a = await prepare(kernel, credential, policy, [QUESTION], "a")
    reply = {"role": "assistant", "content": "Use blue."}
    call = {
        "id": "g",
        "type": "function",
        "function": {"name": "openviking_find", "arguments": "{}"},
    }
    transcript = [
        {"role": "assistant", "content": "checking", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "g", "content": "blue"},
        reply,
    ]
    await store.replay.put(
        a.scope,
        a.session,
        K.HIDDEN,
        hidden_chain([QUESTION, reply], "chat")[-1],
        {"messages": transcript, "visible_count": 1, "upstream_id": "upstream"},
    )
    await kernel.completed(a, credential, ResponseCapture("chat", reply, complete=True))
    # Client-side compaction or a subagent forks the history under a new session id.
    fork = await prepare(
        kernel, credential, policy, [QUESTION, reply, {"role": "user", "content": "Next?"}], "fork"
    )
    assert fork.session != a.session
    assert fork.body["messages"][0] == a.body["messages"][0]
    assert fork.body["messages"][1:4] == transcript
    assert [query for _, query, _, _ in viking.recalls] == ["How do I deploy?", "Next?"]


def tool_call_stream(protocol):
    if protocol == "chat":
        start = {"index": 0, "id": "c1", "type": "function"}
        return [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"tool_calls": [{**start, "function": {"name": "weather"}}]}}]},
            *(
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": a}}]}}
                    ]
                }
                for a in ('{"city":', '"Paris"}')
            ),
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    return [
        {"type": "message_start", "message": {"id": "m1", "usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"text": "Let me check."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "weather", "input": {}},
        },
        *(
            {"type": "content_block_delta", "index": 1, "delta": {"partial_json": part}}
            for part in ('{"city":', '"Paris"}')
        ),
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
    ]


@pytest.mark.parametrize("protocol", ["chat", "anthropic"])
async def test_anonymous_client_tool_handoff_keeps_the_session_and_captures_the_turn(
    setup_kernel, credential, policy, protocol
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False)
    question = {"role": "user", "content": "Weather in Paris?"}
    first = await prepare(kernel, credential, policy, [question], protocol=protocol)
    capture = ResponseCapture(protocol)
    for event in tool_call_stream(protocol):
        capture.event(event)
    assert capture.finished and not capture.complete
    await kernel.completed(first, credential, capture)
    if protocol == "chat":
        arguments = orjson.dumps({"city": "Paris"}).decode()
        function = {"name": "weather", "arguments": arguments}
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": function}],
        }
        result = {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}
        final = {"choices": [{"message": {"role": "assistant", "content": "It is sunny."}}]}
        final["choices"][0]["finish_reason"] = "stop"
    else:
        call = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "Paris"}},
            ],
        }
        result = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}],
        }
        final = {"content": [{"type": "text", "text": "It is sunny."}], "stop_reason": "end_turn"}
    continued = await prepare(
        kernel, credential, policy, [question, call, result], protocol=protocol
    )
    assert continued.session == first.session and continued.kind == "continuation"
    answer = ResponseCapture(protocol)
    answer.nonstream(final)
    await kernel.completed(continued, credential, answer)
    await make_due(store)
    worker = await worker_for(store, encryption, credential, viking)
    while await worker.once():
        pass
    assert "It is sunny." in str(viking.writes) and "Weather in Paris?" in str(viking.writes)
    assert len(set(viking.write_sessions)) == 1


def tool_handoff(protocol):
    if protocol == "chat":
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        }
        return call, {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}
    call = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "Paris"}},
        ],
    }
    result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}],
    }
    return call, result


@pytest.mark.parametrize("protocol", ["chat", "anthropic"])
async def test_streamed_handoff_is_matched_before_bookkeeping_runs(
    running_gateway, monkeypatch, protocol
):
    """A client tool loop can resend the history before the response's background work."""
    app, client, admin, key, _, _ = running_gateway
    path = "/v1/chat/completions" if protocol == "chat" else "/v1/messages"
    streamed = []

    async def backend(request):
        if request.path != path or streamed:
            return None
        streamed.append(True)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in tool_call_stream(protocol):
            await response.write(b"data: " + orjson.dumps(event) + b"\n\n")
        await response.write_eof()
        return response

    async def late(*_):
        pass

    app.state.test_backend["handler"] = backend
    monkeypatch.setattr(app.state.kernel, "completed", late)
    question = {"role": "user", "content": "Weather in Paris?"}
    headers = {"Authorization": "Bearer " + key["key"]}
    body = {"model": "model", "max_tokens": 64, "messages": [question], "stream": True}
    response = await client.post(path, headers=headers, json=body)
    assert response.status_code == 200, response.text
    body = {**body, "stream": False, "messages": [question, *tool_handoff(protocol)]}
    response = await client.post(path, headers=headers, json=body)
    assert response.status_code == 200, response.text
    first, continued = (await client.get("/admin/logs", headers=admin)).json()[1::-1]
    assert continued["kind"] == "continuation"
    assert continued["session"] == first["session"]


async def test_live_fork_keeps_what_it_inherits_from_expiring(setup_kernel, credential, policy):
    kernel, store, _, _ = setup_kernel
    a = await prepare(kernel, credential, policy, [QUESTION], "a", protocol="anthropic")
    thinking = {"type": "thinking", "thinking": "Blue is the safe choice."}
    reply = {"role": "assistant", "content": [thinking, {"type": "text", "text": "Use blue."}]}
    await kernel.completed(a, credential, ResponseCapture("anthropic", reply, complete=True))
    stale = time.time() - 200000

    def age():
        with store.connect() as c:
            c.execute("UPDATE sessions SET touched=? WHERE session=?", (stale, a.session))
            c.execute("UPDATE state SET touched=? WHERE key LIKE 'thinking:%'", (stale,))

    await store.run(age, write=True)
    resent = {
        "role": "assistant",
        "content": [{"type": "text", "text": thinking["thinking"]}, reply["content"][1]],
    }
    history = [QUESTION, resent, {"role": "user", "content": "Next?"}]
    fork = await prepare(kernel, credential, policy, history, "fork", protocol="anthropic")
    assert fork.ancestors == [a.session] and fork.thinking
    answer = {"role": "assistant", "content": [{"type": "text", "text": "Then green."}]}
    await kernel.completed(fork, credential, ResponseCapture("anthropic", answer, complete=True))
    await store.expire(time.time() - 100000)
    again = await prepare(kernel, credential, policy, history, "fork", protocol="anthropic")
    assert again.ancestors == [a.session] and again.thinking == fork.thinking
    assert again.body["messages"][0] == a.body["messages"][0]


async def test_expiry_removes_stale_sessions_records_and_documents(
    setup_kernel, credential, policy
):
    kernel, store, _, _ = setup_kernel
    a = await prepare(kernel, credential, policy, [QUESTION], "a")
    b = await prepare(kernel, credential, policy, [QUESTION], "b")
    stale = time.time() - 1000

    def age():
        with store.connect() as c:
            c.execute("UPDATE sessions SET touched=? WHERE session=?", (stale, a.session))
            c.execute("UPDATE state SET touched=? WHERE key=?", (stale, a.session))

    await store.run(age, write=True)
    await store.expire(time.time() - 100)
    assert not await store.replay.read(a.scope, a.session, ["", a.chain[0]])
    assert (K.INJECTION, b.chain[0]) in await store.replay.read(b.scope, b.session, b.chain)
    assert not (await get_state(store.state, a.scope, a.session)).value
    assert (await get_state(store.state, b.scope, b.session)).value["recall"]


V1 = """
    CREATE TABLE sessions (
        scope TEXT, session TEXT, touched REAL NOT NULL, PRIMARY KEY(scope,session));
    CREATE TABLE replay (
        scope TEXT, session TEXT, kind TEXT, anchor TEXT, value BLOB NOT NULL,
        PRIMARY KEY(scope,session,kind,anchor),
        FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
    CREATE INDEX replay_anchors ON replay(scope,session,anchor);
    CREATE TABLE state (
        scope TEXT, key TEXT, value BLOB NOT NULL, version INTEGER NOT NULL,
        PRIMARY KEY(scope,key));
    CREATE TABLE capture (
        scope TEXT, session TEXT, value BLOB NOT NULL, version INTEGER NOT NULL,
        ready REAL, lease REAL NOT NULL DEFAULT 0, owner TEXT,
        PRIMARY KEY(scope,session),
        FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
    CREATE INDEX capture_ready ON capture(ready,lease);
    CREATE TABLE deleted_scopes (scope TEXT PRIMARY KEY);
    PRAGMA user_version=1;
"""


async def test_version_one_storage_migrates_in_place(tmp_path, setup_kernel):
    _, _, _, encryption = setup_kernel
    store = SQLiteKernelStore(tmp_path / "v1" / "kernel.sqlite3", encryption)
    try:
        value = store.encode({"text": "kept"})
        with store.connect() as c:
            c.executescript(V1)
            c.executemany(
                "INSERT INTO sessions VALUES ('scope',?,?)", [("*", time.time()), ("s", 0)]
            )
            c.executemany(
                "INSERT INTO replay VALUES ('scope',?,?,'',?)",
                [("*", "injection", value), ("s", "root", value)],
            )
            c.executemany(
                "INSERT INTO state VALUES ('scope',?,?,1)",
                [("prefix:anchor", value), ("s", value)],
            )
        await store.initialize()
        await store.initialize()
        with store.connect() as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == 2
            assert [r[0] for r in c.execute("SELECT session FROM sessions")] == ["s"]
            assert [r[0] for r in c.execute("SELECT session FROM replay")] == ["s"]
            assert [tuple(r) for r in c.execute("SELECT key,touched>0 FROM state")] == [("s", 1)]
        assert await store.replay.read("scope", "s", [""]) == {(K.ROOT, ""): {"text": "kept"}}
        await store.expire(time.time() - 100)
        assert (await get_state(store.state, "scope", "s")).value == {"text": "kept"}
        with store.connect() as c:
            c.execute("PRAGMA user_version=3")
        with pytest.raises(ValueError):
            await store.initialize()
    finally:
        store.close()
