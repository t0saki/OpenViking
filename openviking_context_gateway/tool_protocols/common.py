# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Wire adapters preserve native history; only executable calls are normalized.

Adapters assemble one upstream response and project visible events. They do not
execute tools, own budgets, or access storage. The shared loop owns those steps.
"""

import copy
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import orjson

from ..protocols import prefix_chain

PREFIX = "openviking_"
# None is the Chat Completions [DONE] marker; all other events are native JSON.
StreamEvent = dict | None
# One rendered notice line, with the blank lines that separate it. Capture
# strips exactly these lines, so this pattern must follow notice_head/tail.
NOTICE = re.compile(r"^> OpenViking \w+(?:: [^\n]+)? — (?:done|failed|skipped)$\n*", re.M)


class ToolLoopError(Exception):
    def __init__(self, reason, status=502, content=None, headers=None):
        super().__init__(reason)
        self.status, self.content, self.headers = status, content, headers or {}


def merge_delta(target, delta):
    for key, value in delta.items():
        if value is None:
            target.setdefault(key, None)
        elif isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            merge_delta(target[key], value)
        elif isinstance(value, str) and key not in {"role", "type"}:
            target[key] = (target.get(key) or "") + value
        elif isinstance(value, list):
            target.setdefault(key, []).extend(copy.deepcopy(value))
        else:
            target[key] = copy.deepcopy(value)


def add_usage(total, usage):
    for key, value in usage.items():
        if isinstance(value, dict):
            add_usage(total.setdefault(key, {}), value)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


def sse(value):
    return b"data: " + orjson.dumps(value) + b"\n\n"


def call(identifier, name, arguments):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def clip(value, limit=80):
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def notice_head(item):
    """The visible line for one gateway-run call, shown before it runs."""
    short = item["function"]["name"].removeprefix(PREFIX)
    try:
        args = orjson.loads(item["function"].get("arguments") or "{}")
    except (TypeError, ValueError):
        args = {}
    args = args if isinstance(args, dict) else {}

    def text(key):
        value = args.get(key)
        return clip(value) if isinstance(value, str) else ""

    index = args.get("attachment_index")
    attachment = f"attachment {index}" if type(index) is int else ""
    uris = args.get("uris") if isinstance(args.get("uris"), list) else []
    uris = [clip(u) for u in uris if isinstance(u, str) and u.strip()]
    target = {
        "search": text("query") and f'"{text("query")}"',
        "read": uris[0] + (f" (+{len(uris) - 1} more)" if len(uris) > 1 else "") if uris else "",
        "list": text("uri"),
        "write": text("uri"),
        "add_resource": text("path") or attachment,
        "add_skill": text("path")
        or text("target_uri")
        or attachment
        or ("SKILL.md text" if text("data") else ""),
    }.get(short, "")
    return "> OpenViking " + short + (": " + target if target else "")


def notice_tail(content, skipped):
    """The outcome that completes a notice line once the call has run."""
    if skipped:
        return " — skipped"
    try:
        value = orjson.loads(content)
    except (TypeError, ValueError):
        value = None
    failed = isinstance(value, dict) and ("error" in value or value.get("isError") is True)
    return " — failed" if failed else " — done"


def strip_notices(messages):
    """Remove tool notices from echoed assistant text, dropping emptied text parts."""
    result = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str):
            message = {**message, "content": NOTICE.sub("", content)}
        elif message.get("role") == "assistant" and isinstance(content, list):
            parts = []
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str) and NOTICE.search(text):
                    text = NOTICE.sub("", text)
                    if not text.strip():
                        continue
                    part = {**part, "text": text}
                parts.append(part)
            if not parts and content:
                continue
            message = {**message, "content": parts}
        result.append(message)
    return result


@dataclass(frozen=True)
class ToolRound:
    """The validated result of one upstream call, independent of wire format."""

    output: list[dict]
    calls: list[dict]
    usage: dict
    tool_handoff: bool


class ToolProtocol(ABC):
    """Native request/history rules and a per-request response assembler.

    ``visible`` is the cumulative client-visible history. ``end`` returns the
    completed round; the loop never inspects the assembler's partial state.
    Event producers return native JSON (or the Chat DONE marker); only encode
    turns events into bytes. Adapters never execute tools or access storage.
    """

    field = "messages"
    id_prefix = ""
    canonicalizes_history = False

    def __init__(self, body: dict):
        self.body = body
        self.identifier = self.id_prefix + "ovcg-" + uuid.uuid4().hex
        self.visible: list[dict] = []
        self.started = False

    @staticmethod
    def tool_choice(body: dict):
        return body.get("tool_choice", "auto")

    @classmethod
    def block_reason(cls, body: dict) -> str:
        return "" if cls.tool_choice(body) in ("auto", "none") else "tools_forced_choice"

    @staticmethod
    @abstractmethod
    def wire_tools(tools: list[dict]) -> list[dict]:
        """Encode the frozen function catalogue for this protocol."""

    @classmethod
    def add_tools(cls, body: dict, tools: list[dict]) -> None:
        body["tools"] = [*body.get("tools", []), *cls.wire_tools(tools)]

    @staticmethod
    def disable_tools(body: dict) -> None:
        body["tool_choice"] = "none"

    @staticmethod
    def history_chain(messages: list[dict]) -> list[str]:
        return prefix_chain(messages)

    @staticmethod
    def join_replayed_history(messages: list[dict]) -> list[dict]:
        return messages

    @staticmethod
    def omit_hidden_history(messages: list[dict]) -> list[dict]:
        """Drop what cannot be used without the omitted tool history."""
        return strip_notices(messages)

    def begin(self) -> None:
        self.output, self.calls, self.usage, self.envelope = [], [], {}, {}
        self.finish = None

    @abstractmethod
    def event(self, value: dict) -> list[dict]:
        """Consume an upstream event and return any client-visible events."""

    @abstractmethod
    def load(self, value: dict) -> None:
        """Consume a nonstreaming response."""

    @abstractmethod
    def end(self) -> ToolRound:
        """Validate the assembled round before exposing executable calls."""

    @abstractmethod
    def publish_calls(self, client: list[dict]) -> list[dict]:
        """Add client-owned calls to visible history and return their events."""

    def open_notice(self) -> list[dict]:
        """Start visible text for the calls the gateway runs and return its events."""
        return []

    @abstractmethod
    def notice(self, text: str) -> list[dict]:
        """Append text to the open notice in visible history and return its events."""

    def close_notice(self) -> list[dict]:
        return []

    @abstractmethod
    def results(self, receipts: list[dict]) -> list[dict]:
        """Encode executor receipts as native tool-result messages."""

    @abstractmethod
    def final(self, usage: dict) -> dict:
        """Build the combined client response."""

    @abstractmethod
    def terminal(self, final: dict) -> list[StreamEvent]:
        """Return the protocol's success terminal events."""

    def error(self, message: str) -> list[StreamEvent]:
        return [{"type": "error", "error": {"type": "gateway_tool_error", "message": message}}]

    def encode(self, value: StreamEvent) -> bytes:
        if value is None:
            return b"data: [DONE]\n\n"
        return b"event: " + value["type"].encode() + b"\n" + sse(value)
