# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Wire adapters preserve native history; only executable calls are normalized.

Adapters assemble one upstream response and project visible events. They do not
execute tools, own budgets, or access storage. The shared loop owns those steps.
"""

import copy
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import orjson

from ..protocols import prefix_chain

PREFIX = "openviking_"
# None is the Chat Completions [DONE] marker; all other events are native JSON.
StreamEvent = dict | None


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
        """Drop reasoning that cannot be used without the omitted tool history."""
        return messages

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
