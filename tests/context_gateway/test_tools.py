import asyncio
import base64
import copy
from types import SimpleNamespace

import orjson
import pytest

from openviking_context_gateway.capture import capture_messages
from openviking_context_gateway.models import Policy, Upstream
from openviking_context_gateway.protocols import ResponseCapture
from openviking_context_gateway.proxy import upstream_url
from openviking_context_gateway.storage import SQLiteKernelStore
from openviking_context_gateway.tool_catalog import attachments, select_tools
from openviking_context_gateway.tool_executor import ToolExecutor, attachment_bytes
from openviking_context_gateway.tool_loop import HiddenToolLoop
from openviking_context_gateway.tool_protocols import hidden_chain, tool_protocol
from openviking_context_gateway.tool_protocols.common import (
    NOTICE,
    notice_head,
    notice_tail,
    sse,
)
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
    assert len(select_tools({**shell, "store": False}, "responses", {}, policy)) == 3


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
        p.scope,
        p.session,
        "hidden",
        hidden_chain(messages, "chat")[-1],
        {"messages": history, "visible_count": 1},
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
    p.body_chain = hidden_chain(raw, "chat")
    from openviking_context_gateway.kernel import gateway_note
    from openviking_context_gateway.models import Policy
    from openviking_context_gateway.tool_protocols import replay_hidden

    p.records["replacement", p.chain[2]] = {"text": "summary"}
    assert kernel.replace_archive(p, Policy(keep_recent_turns=1))
    p.body["messages"] = replay_hidden(p.body["messages"], p.body_chain, p.records, "chat")
    note = gateway_note(Policy(keep_recent_turns=1), p.root["tools"])
    summary = "summary\n\n<openviking-context>\n" + note + "\n</openviking-context>"
    assert p.body["messages"] == [raw[0], {"role": "user", "content": summary}, *raw[3:]]


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
        body={"messages": [], "stream": True}, protocol="chat", root={"policy": {}}, metrics={}
    )
    loop = HiddenToolLoop(
        prepared, SimpleNamespace(allowed={"openviking_search"}), None, ResponseCapture("chat")
    )
    stream = loop.run(response, None)
    event = await asyncio.wait_for(stream.__anext__(), 0.5)
    assert b"live text" in event and not waiting.is_set()
    await stream.aclose()
    assert closed and not loop.capture.complete


def gateway_call(name, arguments, identifier="g-1"):
    if not isinstance(arguments, str):
        arguments = orjson.dumps(arguments).decode()
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": "openviking_" + name, "arguments": arguments},
    }


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        ("search", {"query": " release\n  date "}, '> OpenViking search: "release date"'),
        ("search", {"query": "q" * 200}, '> OpenViking search: "' + "q" * 79 + '…"'),
        ("read", {"uris": ["viking://a"]}, "> OpenViking read: viking://a"),
        (
            "read",
            {"uris": ["viking://a", "viking://b", 3]},
            "> OpenViking read: viking://a (+1 more)",
        ),
        ("list", {"uri": "viking://resources"}, "> OpenViking list: viking://resources"),
        ("list", {}, "> OpenViking list"),
        ("write", {"uri": "viking://n.md", "content": "x"}, "> OpenViking write: viking://n.md"),
        ("add_resource", {"path": "https://x/a.pdf"}, "> OpenViking add_resource: https://x/a.pdf"),
        ("add_resource", {"attachment_index": 0}, "> OpenViking add_resource: attachment 0"),
        ("add_resource", {"attachment_index": True}, "> OpenViking add_resource"),
        ("add_skill", {"path": "./s", "target_uri": "viking://t"}, "> OpenViking add_skill: ./s"),
        ("add_skill", {"target_uri": "viking://t"}, "> OpenViking add_skill: viking://t"),
        ("add_skill", {"attachment_index": 1}, "> OpenViking add_skill: attachment 1"),
        ("add_skill", {"data": "---\nname: s"}, "> OpenViking add_skill: SKILL.md text"),
        ("add_skill", {}, "> OpenViking add_skill"),
        ("search", "not json", "> OpenViking search"),
        ("search", "[1]", "> OpenViking search"),
        ("search", {"query": "  "}, "> OpenViking search"),
    ],
)
def test_notice_names_each_call_target(name, arguments, expected):
    head = notice_head(gateway_call(name, arguments))
    assert head == expected
    # Capture must recognize every rendered line, whatever its outcome.
    for outcome in ("done", "failed", "skipped"):
        assert NOTICE.fullmatch(head + " — " + outcome)


@pytest.mark.parametrize(
    "content,skipped,expected",
    [
        ('{"content":[{"type":"text","text":"ok"}]}', False, " — done"),
        ('{"content":[{"type":"text","text":"denied"}],"isError":true}', False, " — failed"),
        ('{"error":"Tool is not allowed"}', False, " — failed"),
        ('{"truncated":true,"text":"{\\"error\\""}', False, " — done"),
        ("plain text", False, " — done"),
        ("Gateway tool budget reached; answer using the available results.", True, " — skipped"),
    ],
)
def test_notice_outcome(content, skipped, expected):
    assert notice_tail(content, skipped) == expected


async def test_notices_stream_around_each_gateway_call():
    contents = iter(['{"content":[]}', '{"error":"' + "x" * 4000 + '"}'])
    deltas, seen = [], []

    async def execute(call):
        # The call's head has already streamed when the call starts.
        seen.append(deltas[-1])
        return {"role": "tool", "tool_call_id": call["id"], "content": next(contents)}

    prepared = SimpleNamespace(
        body={"messages": [], "stream": True},
        protocol="chat",
        root={"policy": {"tool_total_tokens": 1000}},
        metrics={},
    )
    executor = SimpleNamespace(allowed={"openviking_search"}, execute=execute)
    loop = HiddenToolLoop(prepared, executor, None, ResponseCapture("chat"))
    loop.adapter.begin()
    calls = [
        gateway_call("search", {"query": "blue"}, "g-1"),
        gateway_call("add_resource", {"attachment_index": 0}, "g-2"),
        gateway_call("read", {"uris": ["viking://a", "viking://b"]}, "g-3"),
    ]
    async for chunk in loop.execute(calls):
        deltas.append(orjson.loads(chunk.removeprefix(b"data: "))["choices"][0]["delta"]["content"])
    assert deltas == [
        '\n\n> OpenViking search: "blue"',
        " — done",
        "\n\n> OpenViking add_resource: attachment 0",
        " — failed",
        "\n\n> OpenViking read: viking://a (+1 more)",
        " — skipped",
        "\n\n",
    ]
    assert seen == [deltas[0], deltas[2]]
    assert loop.adapter.visible[0]["content"] == "".join(deltas)
    # The model receives the real results, never the notices.
    assert "OpenViking" not in orjson.dumps(loop.body["messages"]).decode()


async def test_notices_off_leave_the_reply_unchanged():
    async def execute(call):
        return {"role": "tool", "tool_call_id": call["id"], "content": "{}"}

    prepared = SimpleNamespace(
        body={"messages": [], "stream": True},
        protocol="chat",
        root={"policy": {"show_tool_calls": False}},
        metrics={},
    )
    executor = SimpleNamespace(allowed={"openviking_search"}, execute=execute)
    loop = HiddenToolLoop(prepared, executor, None, ResponseCapture("chat"))
    loop.adapter.begin()
    events = [e async for e in loop.execute([gateway_call("search", {"query": "blue"})])]
    assert events == [] and loop.adapter.visible == [{"role": "assistant", "content": ""}]


def test_capture_drops_tool_notices_from_assistant_text():
    notice = "\n\n> OpenViking add_resource: https://x/a.pdf — done\n\n"
    two = '\n\n> OpenViking search: "a — b" — done\n\n> OpenViking read: viking://a — failed\n\n'
    messages = [
        {"role": "user", "content": '> OpenViking search: "quoted by the user" — done'},
        {"role": "assistant", "content": "Let me call it." + notice + "Added."},
        {"role": "user", "content": "Again"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking."},
                {"type": "text", "text": two},
                {"type": "text", "text": "Found it."},
            ],
        },
        {"role": "user", "content": "Import it"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": notice}],
        },
    ]
    captured = capture_messages(messages, [str(i) for i in range(len(messages))])
    texts = [m["parts"][0]["text"] for m in captured]
    assert texts == [
        messages[0]["content"],
        "Let me call it.\n\nAdded.",
        "Again",
        "Looking.\n\n\nFound it.",
        "Import it",
    ]


@pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
def test_omitted_hidden_history_drops_tool_notices(protocol):
    notice = '\n\n> OpenViking search: "blue" — done\n\n'
    messages = [
        {"role": "user", "content": notice},
        {"role": "assistant", "content": "Looking." + notice + "Blue."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking."},
                {"type": "text", "text": notice},
                {"type": "tool_use", "id": "c-1", "name": "shell", "input": {}},
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": notice}],
        },
        {"type": "function_call", "call_id": "c-1", "name": "shell", "arguments": "{}"},
    ]
    cleaned = tool_protocol(protocol).omit_hidden_history(messages)
    # Only assistant text changes; a part or item left empty is dropped.
    assert cleaned == [
        messages[0],
        {"role": "assistant", "content": "Looking.\n\nBlue."},
        {"role": "assistant", "content": [messages[2]["content"][0], messages[2]["content"][2]]},
        messages[4],
    ]


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


@pytest.mark.parametrize("vendor", ["ark", "byteplus"])
@pytest.mark.parametrize(
    "base,path,expected",
    [
        ("https://ark", "/v1/chat/completions", "https://ark/api/v3/chat/completions"),
        ("https://ark", "/v1/responses", "https://ark/api/v3/responses"),
        ("https://ark", "/v1/messages", "https://ark/api/compatible/v1/messages"),
        (
            "https://ark",
            "/v1/messages/count_tokens",
            "https://ark/api/compatible/v1/messages/count_tokens",
        ),
        ("https://ark", "/v1/models", "https://ark/api/v3/models"),
        ("https://ark/api/v3", "/v1/responses", "https://ark/api/v3/responses"),
        ("https://ark/api/compatible/v1", "/v1/messages", "https://ark/api/compatible/v1/messages"),
    ],
)
def test_ark_vendors_share_upstream_paths(vendor, base, path, expected):
    assert upstream_url({"base_url": base, "vendor": vendor}, path) == expected


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
        first.scope,
        first.session,
        "hidden",
        hidden_chain(messages, "chat")[-1],
        {"messages": transcript, "visible_count": 1},
    )
    body = {
        "messages": [*messages, {"role": "user", "content": "continue"}],
        "response_format": {"type": "json_object"},
    }
    prepared = await kernel.prepare(body, "chat", headers, credential, {"id": "u"}, policy)
    assert not prepared.tools_active
    # Only the opening note joins the new user turn; the visible history is untouched.
    assert prepared.body["messages"][:-1] == messages
    assert prepared.body["messages"][-1]["content"].startswith("continue\n\n<openviking-context>")
    assert prepared.body["response_format"] == body["response_format"]
    assert all(not message.get("tool_calls") for message in prepared.body["messages"])
