import asyncio
import base64
import copy
from types import SimpleNamespace

import orjson
import pytest

from openviking_context_gateway.models import Policy, Upstream
from openviking_context_gateway.protocols import ResponseCapture
from openviking_context_gateway.storage import SQLiteKernelStore
from openviking_context_gateway.tool_catalog import attachments, hidden_chain, select_tools
from openviking_context_gateway.tool_executor import ToolExecutor, attachment_bytes
from openviking_context_gateway.tool_loop import ChatToolLoop, sse
from openviking_context_gateway.vendors import ark_url


@pytest.mark.parametrize(
    "vendor,extra,expected",
    [
        ("generic", {}, 3),
        ("deepseek", {}, 0),
        ("deepseek", {"thinking": {"type": "enabled"}}, 0),
        ("deepseek", {"thinking": {"type": "disabled"}}, 3),
        ("ark", {"tools": [{"type": "custom", "name": "x"}]}, 0),
    ],
)
def test_vendor_tool_capabilities(vendor, extra, expected):
    policy = Policy(gateway_tools=True).model_dump()
    upstream = Upstream(
        name="x", base_url="http://model", protocol="chat", vendor=vendor
    ).model_dump()
    assert len(select_tools(extra, "chat", upstream, policy)) == expected


def test_write_tools_require_permission_and_client_capability():
    policy = Policy(
        gateway_tools=True, tool_allowlist=["write", "add_resource", "add_skill"]
    ).model_dump()
    assert select_tools({}, "chat", {}, policy) == []
    policy["allow_write_tools"] = True
    assert len(select_tools({}, "chat", {}, policy)) == 1
    shell = {"tools": [{"type": "function", "function": {"name": "exec_command"}}]}
    assert len(select_tools(shell, "chat", {}, policy)) == 3
    assert select_tools(shell, "responses", {}, policy) == []


def test_file_bytes_and_extracted_text():
    name, data = attachment_bytes(
        {
            "filename": "../../notes.txt",
            "file_data": "data:text/plain;base64," + base64.b64encode(b"hello").decode(),
        },
        100,
    )
    assert name == "notes.txt" and data == b"hello"
    with pytest.raises(ValueError):
        attachment_bytes({"file_data": "a" * 1000}, 10)
    with pytest.raises(ValueError):
        attachment_bytes({"file_id": "remote-file"}, 100)
    parts = attachments(
        {
            "messages": [
                {
                    "role": "system",
                    "content": '<context><source id="1" name="report.pdf">hello &amp; world</source></context>',
                }
            ]
        }
    )
    assert parts == [{"filename": "report.pdf.txt", "text": "hello & world"}]


async def test_tool_claim_timeout_and_byte_bound(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    policy.update(
        gateway_tools=True, allow_write_tools=True, tool_allowlist=["write"], tool_result_bytes=1024
    )
    prepared = await kernel.prepare(
        {"messages": [{"role": "user", "content": "Save this"}]},
        "chat",
        {"x-openviking-session": "tools"},
        credential,
        {"id": "upstream"},
        policy,
    )
    executed = []

    async def mcp(name, key, args):
        executed.append((name, key, args))
        await asyncio.sleep(0.03)
        return {"content": [{"type": "text", "text": '"' * 10000}]}

    viking.mcp = mcp
    call = {
        "id": "write-1",
        "function": {
            "name": "openviking_write",
            "arguments": '{"uri":"viking://~/notes/a","content":"note"}',
        },
    }
    another = SQLiteKernelStore(store.path, encryption)
    executors = [
        ToolExecutor(viking, s, prepared, credential, "http://gateway", 1024)
        for s in [store, another]
    ]
    results = await asyncio.gather(*(e.execute(call) for e in executors))
    assert len(executed) == 1
    assert results[0] == results[1]
    assert len(results[0]["content"].encode()) <= 1024
    assert orjson.loads(results[0]["content"])["truncated"]
    # A timed out write is remembered, including across a new store instance.
    policy["tool_timeout_seconds"] = 0.001
    prepared.root["policy"]["tool_timeout_seconds"] = 0.001
    call["id"] = "timeout-write"
    result = await executors[0].execute(call)
    assert "timed out" in result["content"]
    again = await executors[1].execute(call)
    assert again == result and len(executed) == 2


async def test_hidden_branch_restart_and_archive_mapping(setup_kernel, credential, policy):
    policy.update(gateway_tools=True)
    kernel, store, _, encryption = setup_kernel
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "blue"},
    ]
    history = [
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "g",
                    "type": "function",
                    "function": {"name": "openviking_search", "arguments": "{}"},
                }
            ],
            "reasoning_content": "opaque",
        },
        {"role": "tool", "tool_call_id": "g", "content": "blue"},
        {"role": "assistant", "content": "blue"},
    ]
    p = await kernel.prepare({"messages": messages}, "chat", {}, credential, {"id": "u"}, policy)
    await store.replay.put(
        p.scope, p.session, "hidden", hidden_chain(messages)[-1], {"messages": history}
    )
    kernel.store = SQLiteKernelStore(store.path, encryption)
    p = await kernel.prepare(
        {"messages": [*messages, {"role": "user", "content": "follow up"}]},
        "chat",
        {"x-openviking-session": "tools"},
        credential,
        {"id": "u"},
        policy,
    )
    assert p.body["messages"][2:5] == history
    edited = copy.deepcopy(messages)
    edited[-1]["content"] = "red"
    p = await kernel.prepare({"messages": edited}, "chat", {}, credential, {"id": "u"}, policy)
    assert len(p.body["messages"]) == 3
    # Replacing an archived prefix must not use expanded transcript indices.
    raw = [
        *messages,
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "later"},
    ]
    p = await kernel.prepare({"messages": raw}, "chat", {}, credential, {"id": "u"}, policy)
    p.body["messages"] = copy.deepcopy(raw)
    p.body_chain = hidden_chain(raw)
    from openviking_context_gateway.models import Policy
    from openviking_context_gateway.tool_catalog import replay_hidden

    p.records["replacement", p.chain[2]] = {"text": "summary"}
    assert kernel.replace_archive(p, Policy(keep_recent_turns=1))
    p.body["messages"] = replay_hidden(p.body["messages"], p.body_chain, p.records)
    assert p.body["messages"] == [raw[0], {"role": "user", "content": "summary"}, *raw[3:]]


async def test_stream_text_precedes_tool_completion_and_cancel_closes():
    waiting = asyncio.Event()
    closed = []

    class Content:
        async def iter_any(self):
            yield sse({"choices": [{"delta": {"content": "live text"}, "finish_reason": None}]})
            await waiting.wait()
            pytest.fail("The client cancelled before the tool round completed")

    response = SimpleNamespace(
        status=200,
        headers={"content-type": "text/event-stream"},
        content=Content(),
        close=lambda: closed.append(True),
    )
    prepared = SimpleNamespace(
        body={"messages": [], "stream": True}, root={"policy": {}}, metrics={}
    )
    loop = ChatToolLoop(
        prepared, SimpleNamespace(allowed={"openviking_search"}), None, ResponseCapture("chat")
    )
    stream = loop.run(response, None)
    event = await asyncio.wait_for(stream.__anext__(), 0.5)
    assert b"live text" in event and not waiting.is_set()
    await stream.aclose()
    assert closed and not loop.capture.complete


@pytest.mark.parametrize(
    "base,path,expected",
    [
        ("https://ark/api/v3", "/v1/chat/completions", "https://ark/api/v3/chat/completions"),
        ("https://ark", "/v1/responses/resp_1", "https://ark/api/v3/responses/resp_1"),
        ("https://ark/api/compatible/v1", "/v1/messages", "https://ark/api/compatible/v1/messages"),
    ],
)
def test_ark_base_paths(base, path, expected):
    assert ark_url({"base_url": base}, path) == expected


@pytest.mark.parametrize("json_response", [False, True])
async def test_real_stateless_fastmcp_transport(json_response):
    import socket

    import aiohttp
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    from openviking_context_gateway.client import VikingClient

    mcp = FastMCP(
        "gateway-test",
        stateless_http=True,
        json_response=json_response,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @mcp.tool()
    async def find(query: str) -> str:
        return "found: " + query

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(mcp.streamable_http_app(), access_log=False, log_level="error")
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(200):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert server.started
        async with aiohttp.ClientSession(auto_decompress=False) as http:
            adapter = VikingClient(http, f"http://127.0.0.1:{port}", "0.4.16")
            result = await adapter.mcp("find", "test-key", {"query": "blue"})
            assert result["content"][0]["text"] == "found: blue"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        listener.close()


async def test_explicit_session_keeps_initial_tool_snapshot(setup_kernel, credential, policy):
    kernel, _, _, _ = setup_kernel
    policy.update(gateway_tools=True, tool_allowlist=["search"])
    body = {"messages": [{"role": "user", "content": "Find blue"}]}
    first = await kernel.prepare(
        body, "chat", {"x-openviking-session": "tools"}, credential, {"id": "upstream"}, policy
    )
    policy = {**policy, "tool_allowlist": ["list"]}
    body["messages"].extend(
        [{"role": "assistant", "content": "blue"}, {"role": "user", "content": "Follow up"}]
    )
    second = await kernel.prepare(
        body, "chat", {"x-openviking-session": "tools"}, credential, {"id": "upstream"}, policy
    )
    assert first.session == second.session
    assert first.body["tools"] == second.body["tools"]


async def test_incompatible_tools_keep_visible_history(setup_kernel, credential, policy):
    kernel, store, _, _ = setup_kernel
    policy.update(gateway_tools=True, recall=False)
    messages = [{"role": "user", "content": "search"}, {"role": "assistant", "content": "answer"}]
    headers = {"x-openviking-session": "downgrade"}
    first = await kernel.prepare(
        {"messages": messages}, "chat", headers, credential, {"id": "u"}, policy
    )
    transcript = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "owned",
                    "type": "function",
                    "function": {"name": "openviking_search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "owned", "content": "result"},
        messages[1],
    ]
    await store.replay.put(
        first.scope, first.session, "hidden", hidden_chain(messages)[-1], {"messages": transcript}
    )
    body = {
        "messages": [*messages, {"role": "user", "content": "continue"}],
        "response_format": {"type": "json_object"},
    }
    prepared = await kernel.prepare(body, "chat", headers, credential, {"id": "u"}, policy)
    assert not prepared.tools_active
    assert prepared.body == body
    assert all(not message.get("tool_calls") for message in prepared.body["messages"])
