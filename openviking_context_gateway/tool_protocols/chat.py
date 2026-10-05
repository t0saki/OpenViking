# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Chat Completions messages and delta projection."""

import copy

from ..protocols import prefix_chain, text_content
from .common import (
    SUMMARY_HEADROOM,
    SummaryError,
    ToolLoopError,
    ToolProtocol,
    ToolRound,
    merge_delta,
    sse,
)


class ChatProtocol(ToolProtocol):
    id_prefix = "chatcmpl-"
    canonicalizes_history = True

    @staticmethod
    def wire_tools(tools):
        return tools

    @staticmethod
    def block_reason(body):
        if any(t.get("type", "function") != "function" for t in body.get("tools", [])):
            return "tools_non_function"
        return ToolProtocol.block_reason(body)

    @staticmethod
    def history_chain(messages):
        # Chat clients often omit reasoning/vendor metadata. Match visible
        # content and calls; the stored transcript retains the original objects.
        canonical = []
        for message in messages:
            if message.get("role") == "assistant":
                message = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}),
                }
            canonical.append(message)
        return prefix_chain(canonical)

    @classmethod
    def summary_request(cls, body, messages, instruction, max_tokens):
        request = super().summary_request(body, messages, instruction, max_tokens)
        request.pop("stream_options", None)
        cap = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        request[cap] = max_tokens + (SUMMARY_HEADROOM if body.get("reasoning_effort") else 0)
        return request

    @staticmethod
    def summary_text(response):
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if message.get("tool_calls"):
            raise SummaryError("summary_tool_call")
        if choice.get("finish_reason") != "stop":
            raise SummaryError("summary_incomplete")
        return text_content(message)

    def __init__(self, body):
        super().__init__(body)
        self.visible = [{"role": "assistant", "content": ""}]

    def begin(self):
        super().begin()
        self.message, self.fragments = {"role": "assistant"}, {}

    def encode(self, value):
        return super().encode(value) if value is None else sse(value)

    def event(self, value):
        self.envelope.update({k: v for k, v in value.items() if k not in {"choices", "usage"}})
        if value.get("usage"):
            self.usage.update(value["usage"])
        choices = value.get("choices") or []
        if not choices:
            return [] if value.get("usage") else [{**value, "id": self.identifier}]
        if len(choices) != 1 or choices[0].get("index", 0) != 0:
            raise ToolLoopError("Tool mode requires one completion")
        choice = choices[0]
        delta = copy.deepcopy(choice.get("delta", {}))
        for item in delta.pop("tool_calls", []) or []:
            index = item.pop("index", 0)
            merge_delta(self.fragments.setdefault(index, {}), item)
        merge_delta(self.message, delta)
        merge_delta(self.visible[0], delta)
        self.finish = choice.get("finish_reason") or self.finish
        if delta or any(k not in {"delta", "finish_reason", "index"} for k in choice):
            return [
                {
                    **value,
                    "id": self.identifier,
                    "usage": None,
                    "choices": [{**choice, "index": 0, "delta": delta, "finish_reason": None}],
                }
            ]
        return []

    def load(self, value):
        choices = value["choices"]
        if len(choices) != 1:
            raise ToolLoopError("Tool mode requires one completion")
        self.envelope = value
        self.message = copy.deepcopy(choices[0]["message"])
        self.finish = choices[0]["finish_reason"]
        self.usage = value.get("usage") or {}
        merge_delta(self.visible[0], {k: v for k, v in self.message.items() if k != "tool_calls"})

    def end(self):
        if not self.finish:
            raise ToolLoopError("Model response ended before finish_reason")
        if self.fragments:
            self.message["tool_calls"] = [self.fragments[i] for i in sorted(self.fragments)]
        self.message.setdefault("content", None)
        self.output = [self.message]
        self.calls = self.message.get("tool_calls") or []
        return ToolRound(self.output, self.calls, self.usage, self.finish == "tool_calls")

    def publish_calls(self, client):
        if not client:
            return []
        self.visible[0]["tool_calls"] = client
        return [self.chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(client)]})]

    def notice(self, text):
        self.visible[0]["content"] = (self.visible[0].get("content") or "") + text
        return [self.chunk({"content": text})]

    def results(self, results):
        return [{key: value for key, value in r.items() if key != "failed"} for r in results]

    def chunk(self, delta, finish=None):
        return {
            **self.envelope,
            "id": self.identifier,
            "object": "chat.completion.chunk",
            "usage": None,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def final(self, usage):
        return {
            **self.envelope,
            "id": self.identifier,
            "object": "chat.completion",
            "usage": usage,
            "choices": [
                {
                    **((self.envelope.get("choices") or [{}])[0]),
                    "index": 0,
                    "message": self.visible[0],
                    "finish_reason": self.finish,
                }
            ],
        }

    def terminal(self, final):
        events = [self.chunk({}, self.finish)]
        if (self.body.get("stream_options") or {}).get("include_usage"):
            events.append({**self.chunk({}), "choices": [], "usage": final["usage"]})
        return [*events, None]

    def error(self, message):
        return [{"error": {"type": "gateway_tool_error", "message": message}}, None]
