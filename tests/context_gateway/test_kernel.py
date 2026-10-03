import asyncio
import copy
import time

import pytest
from conftest import replay_records

from openviking_context_gateway.capture import CaptureWorker, capture_messages
from openviking_context_gateway.capture_store import Document
from openviking_context_gateway.client import VikingError
from openviking_context_gateway.protocols import (
    ResponseCapture,
    classify,
    normalize,
    parse_body,
    plugin_present,
    prefix_chain,
)
from openviking_context_gateway.storage import ManagementStore, SQLiteKernelStore


async def prepare(kernel, body, credential, policy, protocol="chat", session="session", **kwargs):
    return await kernel.prepare(
        body,
        protocol,
        {"x-openviking-session": session},
        credential,
        {"id": "upstream"},
        policy,
        **kwargs,
    )


@pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
async def test_replay_survives_restart_and_preserves_signed_prefix(
    setup_kernel, credential, policy, protocol
):
    kernel, store, viking, key = setup_kernel
    field = "input" if protocol == "responses" else "messages"
    first = {
        "model": "test",
        "store": False,
        field: [{"role": "user", "content": "How do I deploy?"}],
        "unknown_provider_option": {"opaque": "preserve"},
        "tools": [],
        "system": "fixed",
    }
    before = copy.deepcopy(first)
    one = await prepare(kernel, first, credential, policy, protocol)
    assert first == before
    assert "openviking-context" in str(one.body[field][0])
    store2 = SQLiteKernelStore(store.path, key)
    await store2.initialize()
    kernel.store = store2
    second = copy.deepcopy(first)
    second[field] += [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private", "signature": "bound"},
                {"type": "text", "text": "Use blue."},
            ],
        },
        {"role": "user", "content": "What next?"},
    ]
    two = await prepare(kernel, second, credential, policy, protocol)
    assert normalize(two.body[field][0]) == normalize(one.body[field][0])
    assert two.body[field][1] == second[field][1]
    assert two.body["system"] == first["system"]
    assert two.body["unknown_provider_option"] == first["unknown_provider_option"]
    assert viking.recalls[1][2] == [viking.entries[0]["uri"]]
    # A simulator validates the exact normalized prefix that signed thinking saw.
    assert prefix_chain(two.body[field])[0] == prefix_chain(one.body[field])[0]


async def test_empty_decision_and_concurrent_first_writer(setup_kernel, credential, policy):
    kernel, _, viking, _ = setup_kernel
    body = {"messages": [{"role": "user", "content": "How do I deploy?"}]}
    viking.failure = VikingError("recall_timeout")
    one = await prepare(kernel, body, credential, policy)
    viking.failure = None
    many = await asyncio.gather(*(prepare(kernel, body, credential, policy) for _ in range(12)))
    assert all(x.body == one.body == body for x in many)
    assert len(viking.recalls) == 1
    fresh = await asyncio.gather(
        *(prepare(kernel, body, credential, policy, session="different") for _ in range(10))
    )
    assert all(x.body == body for x in fresh)  # inherited empty decisions are also sticky


async def test_scope_fork_and_policy_snapshot(setup_kernel, credential, policy):
    kernel, _, viking, _ = setup_kernel
    body = {"messages": [{"role": "user", "content": "How do I deploy?"}]}
    first = await prepare(kernel, body, credential, policy)
    policy["max_tokens"] = 64
    second = await prepare(kernel, body, credential, policy)
    assert second.root["policy"]["max_tokens"] == 1600
    fork = await prepare(kernel, body, credential, policy, session="fork")
    assert fork.body == first.body
    assert len(viking.recalls) == 1
    other = await prepare(kernel, body, {**credential, "user_id": "bob"}, policy)
    assert other.scope != first.scope
    assert len(viking.recalls) == 2


async def test_plugin_stop_is_sticky_but_replays_existing_prefix(setup_kernel, credential, policy):
    kernel, _, viking, _ = setup_kernel
    body = {"messages": [{"role": "user", "content": "How do I deploy?"}]}
    one = await prepare(kernel, body, credential, policy)
    two_body = copy.deepcopy(body)
    two_body["messages"] += [
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Next question"},
    ]
    two_body["tools"] = [{"type": "namespace", "name": "mcp__openviking", "tools": []}]
    two = await prepare(kernel, two_body, credential, policy)
    assert two.disabled
    assert two.body["messages"][0] == one.body["messages"][0]
    del two_body["tools"]
    three = await prepare(kernel, two_body, credential, policy)
    assert three.disabled and len(viking.recalls) == 1


async def test_capture_confirms_branch_and_isolates_forks(setup_kernel, credential, policy):
    kernel, store, viking, key = setup_kernel
    management = ManagementStore(store.path.parent / "management.sqlite3", key)
    await management.initialize()
    await management.save("tenant", "keys", credential["id"], credential)
    body = {"messages": [{"role": "user", "content": "How do I deploy?"}]}
    one = await prepare(kernel, body, credential, policy)
    response = ResponseCapture(
        "chat", {"role": "assistant", "content": "Discarded answer"}, complete=True
    )
    await kernel.completed(one, credential, response)
    assert await store.capture.claim() is None  # idle delay
    body["messages"] += [
        {"role": "assistant", "content": "Kept answer"},
        {"role": "user", "content": "Next question"},
    ]
    await prepare(kernel, body, credential, policy)
    worker = CaptureWorker(store, management, viking)
    assert await worker.once()
    assert "Kept answer" in str(viking.writes) and "Discarded answer" not in str(viking.writes)
    assert "openviking-context" not in str(viking.writes)
    await prepare(kernel, body, credential, policy, session="fork")
    assert await worker.once()
    assert len(viking.writes) == 2
    assert viking.write_sessions[0] != viking.write_sessions[1]


async def test_lease_is_exclusive_and_old_owner_cannot_ack(setup_kernel):
    _, store, _, _ = setup_kernel
    await store.capture.swap("scope", "session", Document(), {"work": "pending"}, 0)
    results = await asyncio.gather(*(store.capture.claim() for _ in range(8)))
    claimed = [x for x in results if x]
    assert len(claimed) == 1
    await store.capture.release({**claimed[0], "owner": "old"})
    assert await store.capture.claim() is None
    old = await store.capture.get("scope", "session")
    await store.capture.swap("scope", "session", old, {"work": "done"}, None)
    await store.capture.release(claimed[0])
    assert await store.capture.claim() is None


async def test_archive_replacement_is_immutable(setup_kernel, credential, policy):
    kernel, store, viking, _ = setup_kernel
    policy["keep_recent_turns"] = 1
    body = {
        "messages": [
            {"role": "user", "content": "How do I deploy?"},
            {"role": "assistant", "content": "Blue cluster"},
            {"role": "user", "content": "What next?"},
        ]
    }
    one = await prepare(kernel, body, credential, policy)
    await store.replay.put(
        one.scope,
        one.session,
        "replacement",
        one.chain[1],
        {"text": "[OpenViking Session Context]\nverified summary"},
    )
    two = await prepare(kernel, body, credential, policy)
    assert two.body["messages"][0]["content"].startswith("[OpenViking Session Context]")
    await store.replay.put(
        one.scope, one.session, "replacement", one.chain[1], {"text": "Changed summary"}
    )
    three = await prepare(kernel, body, credential, policy)
    assert two.body == three.body


async def test_encryption_and_whole_session_expiry(setup_kernel):
    _, store, _, _ = setup_kernel
    await store.replay.put(
        "scope", "session", "injection", "anchor", {"text": "VERY_PRIVATE_MEMORY"}
    )
    assert b"VERY_PRIVATE_MEMORY" not in store.path.read_bytes()
    await store.expire(time.time() - 1)
    assert ("injection", "anchor") in await store.replay.read("scope", "session", ["anchor"])
    await store.expire(time.time() + 1)
    assert not await store.replay.read("scope", "session", ["anchor"])


def test_normalization_client_equivalence():
    a = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": " hello ", "cache_control": {"type": "ephemeral"}}
            ],
        }
    ]
    b = [{"role": "system", "content": "transient"}, {"role": "user", "content": "hello"}]
    assert prefix_chain(a)[0] == prefix_chain(b)[1]
    a += [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "ok"}],
        }
    ]
    b += [{"role": "assistant", "content": "ok"}]
    assert prefix_chain(a)[-1] == prefix_chain(b)[-1]
    assert normalize(
        {"role": "user", "content": "<context>RAG</context><user_query>hello</user_query>"}
    ) == normalize(b[1])


@pytest.mark.parametrize(
    "raw",
    [
        b'{"arg":9223372036854775809}',
        b'{"arg":0.123456789123456789}',
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e400}',
    ],
)
def test_lossy_input_is_not_modified(raw):
    assert parse_body(raw) is None


def test_classification_and_plugin_namespaces():
    user = {"role": "user", "content": "Deploy please"}
    tool = {"role": "tool", "tool_call_id": "1", "content": "ok"}
    assert classify({}, {}, [user, tool])[0] == "continuation"
    assert classify({}, {"x-claude-code-request-class": "compaction"}, [user])[0] == "auxiliary"
    assert classify({}, {"x-claude-code-request-class": "auxiliary"}, [user])[0] == "auxiliary"
    assert (
        classify({}, {"x-codex-turn-metadata": '{"request_kind":"compact"}'}, [user])[0]
        == "auxiliary"
    )
    assert classify({}, {"x-claude-code-agent-id": "child"}, [user])[0] == "subagent"
    assert plugin_present(
        {
            "additional_tools": [
                {"namespace": "mcp__openviking_memory", "tools": [{"name": "read"}]}
            ]
        },
        {},
    )


def test_capture_pairs_tools_and_strips_noise():
    messages = [
        {"role": "user", "content": "<system-reminder>noise</system-reminder>please deploy"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1", "function": {"name": "shell", "arguments": '{"cmd":"deploy"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "success"},
        {"role": "assistant", "content": "Done"},
    ]
    output = capture_messages(messages, prefix_chain(messages))
    assert output[1]["parts"][0]["tool_output"] == "success"
    assert "noise" not in str(output)
    assert len(output) == 3


def test_sse_chunk_boundaries_and_anthropic_usage():
    from openviking_context_gateway.protocols import SSEDecoder

    raw = 'data: {"text":"中文"}\r\n\r\ndata: [DONE]\n\n'.encode()
    decoder = SSEDecoder()
    frames = []
    for byte in raw:
        frames += decoder.feed(bytes([byte]))
    assert b"".join(frames) == raw
    capture = ResponseCapture("anthropic")
    capture.event(
        {
            "type": "message_start",
            "message": {"id": "x", "usage": {"input_tokens": 12, "cache_read_input_tokens": 88}},
        }
    )
    capture.event(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 20},
        }
    )
    assert capture.usage == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_tokens": 88,
        "cache_write_tokens": 0,
    }


async def test_known_missing_record_drops_old_thinking(setup_kernel, credential, policy):
    kernel, store, _, _ = setup_kernel
    body = {"messages": [{"role": "user", "content": "How do I deploy?"}]}
    first = await prepare(kernel, body, credential, policy, "anthropic")
    await kernel.completed(first, credential, ResponseCapture("anthropic"))
    with store.connect() as connection:
        connection.execute("DELETE FROM replay WHERE kind='injection'")
    body["messages"] += [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret", "signature": "bound"},
                {"type": "text", "text": "blue"},
            ],
        },
        {"role": "user", "content": "Continue please"},
    ]
    second = await prepare(kernel, body, credential, policy, "anthropic")
    assert second.metrics["degradation"] == "missing_injection_record"
    assert all(block["type"] != "thinking" for block in second.body["messages"][1]["content"])


async def test_large_body_does_not_trigger_placeholder_archive(setup_kernel, credential, policy):
    kernel, store, viking, _ = setup_kernel
    policy.update(context_window=1024, archive_wait_seconds=0, keep_recent_turns=1, recall=False)
    viking.summary = ""
    body = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "Large old history " * 1000},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "Keep going"},
        ],
    }
    first = await prepare(kernel, body, credential, policy)
    assert first.body == body and "degradation" not in first.metrics
    await kernel.completed(first, credential, ResponseCapture("chat", usage={"input_tokens": 1000}))
    second = await prepare(kernel, body, credential, policy)
    assert "degradation" not in second.metrics
    assert second.body == body
    assert not await replay_records(store, first, "replacement")


def test_tool_input_whitespace_is_significant():
    before = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "1", "name": "write", "input": {"content": "a "}}
            ],
        }
    ]
    after = copy.deepcopy(before)
    after[0]["content"][0]["input"]["content"] = "a"
    assert prefix_chain(before) != prefix_chain(after)
