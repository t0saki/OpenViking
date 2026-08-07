# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Experience attribution across the tool-output shapes real harnesses record.

The Experience tools live on the server MCP endpoint, so every harness sees
them. Each harness namespaces the tool name differently and records the MCP
result in a different envelope; attribution has to survive all of them.
"""

import json

import pytest

from openviking.core.experience import (
    load_tool_output_mapping,
    normalize_experience_tool_name,
)
from openviking.message import Message, ToolPart
from openviking.server.identity import RequestContext, Role
from openviking.session.memory.experience_lineage import collect_read_experience_uris
from openviking.usage_reporter import MemoryUsageExtractor, UsageContext
from openviking_cli.session.user_id import UserIdentifier

USER_ID = "test"
EXPERIENCE_URI = f"viking://user/{USER_ID}/memories/experiences/no-order-exchange.md"

# Literal payloads the server tools emit, before any harness wraps them.
SEARCH_OUTPUT = json.dumps(
    {
        "results": [
            {"uri": EXPERIENCE_URI, "title": "no-order-exchange", "score": 0.61, "snippet": ""}
        ]
    },
    ensure_ascii=False,
    separators=(",", ":"),
)
READ_OUTPUT = json.dumps(
    {"uri": EXPERIENCE_URI, "content": "## Situation\n用户未提供订单号但要求换货。"},
    ensure_ascii=False,
    separators=(",", ":"),
)


def _content_block_envelope(text: str) -> str:
    """Claude Code records `tool_result.content`, an MCP content-block array."""
    return json.dumps([{"type": "text", "text": text}])


def _structured_content_envelope(text: str) -> str:
    """FastMCP wraps a `-> str` tool result as structuredContent {"result": ...}."""
    return json.dumps(
        {"content": [{"type": "text", "text": text}], "structuredContent": {"result": text}}
    )


# (label, tool-name prefix, output envelope)
HARNESS_SHAPES = [
    ("codex", "", lambda text: text),
    ("claude-code", "mcp__openviking__", _content_block_envelope),
    ("opencode", "openviking_", lambda text: text),
    ("openclaw", "", lambda text: text),
    ("structured-content", "mcp__openviking__", _structured_content_envelope),
]


def _usage_context() -> UsageContext:
    return UsageContext(
        account_id="new",
        user_id=USER_ID,
        session_id="session-1",
        archive_uri=f"viking://user/{USER_ID}/sessions/session-1/history/archive_001",
        task_id="task-1",
    )


def _request_context() -> RequestContext:
    return RequestContext(user=UserIdentifier("new", USER_ID), role=Role.USER)


def _messages(prefix: str, envelope) -> list[Message]:
    return [
        Message(
            id="msg-1",
            role="user",
            parts=[
                ToolPart(
                    tool_id="call-search",
                    tool_name=f"{prefix}search_experience",
                    tool_status="completed",
                    tool_input={"query": "无订单号换货"},
                    tool_output=envelope(SEARCH_OUTPUT),
                ),
                ToolPart(
                    tool_id="call-read",
                    tool_name=f"{prefix}read_experience",
                    tool_status="completed",
                    tool_input={"uri": EXPERIENCE_URI},
                    tool_output=envelope(READ_OUTPUT),
                ),
            ],
        )
    ]


@pytest.mark.parametrize(
    "label,prefix,envelope", HARNESS_SHAPES, ids=[s[0] for s in HARNESS_SHAPES]
)
@pytest.mark.asyncio
async def test_usage_events_survive_every_harness_shape(label, prefix, envelope):
    events = await MemoryUsageExtractor().extract(
        messages=_messages(prefix, envelope),
        context=_usage_context(),
    )

    assert [event.event_type for event in events] == ["memory.recalled", "memory.injected"]
    assert [event.resource_uri for event in events] == [EXPERIENCE_URI, EXPERIENCE_URI]
    assert events[0].evidence["tool_name"] == f"{prefix}search_experience"


@pytest.mark.parametrize(
    "label,prefix,envelope", HARNESS_SHAPES, ids=[s[0] for s in HARNESS_SHAPES]
)
def test_lineage_survives_every_harness_shape(label, prefix, envelope):
    uris = collect_read_experience_uris(_messages(prefix, envelope), ctx=_request_context())

    assert uris == [EXPERIENCE_URI]


@pytest.mark.asyncio
async def test_recall_events_survive_without_the_call_side_tool_input():
    """Harnesses that only record the result block leave the read URI in the output."""
    messages = [
        Message(
            id="msg-1",
            role="user",
            parts=[
                ToolPart(
                    tool_id="call-read",
                    tool_name="mcp__openviking__read_experience",
                    tool_status="completed",
                    tool_output=_content_block_envelope(READ_OUTPUT),
                )
            ],
        )
    ]

    events = await MemoryUsageExtractor().extract(messages=messages, context=_usage_context())

    assert [event.resource_uri for event in events] == [EXPERIENCE_URI]
    assert collect_read_experience_uris(messages, ctx=_request_context()) == [EXPERIENCE_URI]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("search_experience", "search_experience"),
        ("read_experience", "read_experience"),
        ("mcp__openviking__search_experience", "search_experience"),
        ("mcp__plugin_openviking-memory_openviking__read_experience", "read_experience"),
        ("openviking_search_experience", "search_experience"),
        ("  read_experience  ", "read_experience"),
        ("", ""),
        (None, ""),
        ("read", "read"),
        ("search_experiences", "search_experiences"),
        ("__search_experience", "search_experience"),
    ],
)
def test_normalize_experience_tool_name(raw, expected):
    assert normalize_experience_tool_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        '{"uri":"truncated',
        "[]",
        "[{}]",
        '["plain text"]',
        "42",
    ],
)
def test_load_tool_output_mapping_yields_empty_on_junk(raw):
    assert load_tool_output_mapping(raw) == {}


def test_load_tool_output_mapping_keeps_read_output_content_field():
    """`read_experience` output has its own `content` key — it must not be unwrapped."""
    assert load_tool_output_mapping(READ_OUTPUT)["uri"] == EXPERIENCE_URI
    assert load_tool_output_mapping(READ_OUTPUT)["content"].startswith("## Situation")
