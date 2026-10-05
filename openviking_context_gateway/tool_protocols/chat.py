# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Chat Completions messages and delta projection."""

import copy

from .common import ToolLoopError, ToolProtocol, merge_delta, sse


class ChatProtocol(ToolProtocol):
    id_prefix = "chatcmpl-"
    call_finish = "tool_calls"

    def __init__(self, body):
        super().__init__(body)
        self.visible = [{"role": "assistant", "content": ""}]

    def begin(self):
        super().begin()
        self.message, self.fragments = {"role": "assistant"}, {}

    def encode(self, value):
        return sse(value)

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

    def publish_calls(self, client):
        if not client:
            return []
        self.visible[0]["tool_calls"] = client
        return [self.chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(client)]})]

    def results(self, results):
        return results

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
        events = [sse(self.chunk({}, self.finish))]
        if self.body.get("stream_options", {}).get("include_usage"):
            events.append(sse({**self.chunk({}), "choices": [], "usage": final["usage"]}))
        return [*events, b"data: [DONE]\n\n"]

    def error(self, message):
        return (
            sse({"error": {"type": "gateway_tool_error", "message": message}}) + b"data: [DONE]\n\n"
        )
