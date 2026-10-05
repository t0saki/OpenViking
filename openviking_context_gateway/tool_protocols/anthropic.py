# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Messages content blocks, preserving thinking/signatures in native history."""

import copy

import orjson

from .common import ToolLoopError, ToolProtocol, call


class AnthropicProtocol(ToolProtocol):
    id_prefix = "msg_"
    call_finish = "tool_use"

    def __init__(self, body):
        super().__init__(body)
        self.visible = [{"role": "assistant", "content": []}]

    def begin(self):
        super().begin()
        self.blocks, self.arguments, self.indices = {}, {}, {}
        self.stopped, self.closed = False, set()

    def event(self, value):
        kind = value["type"]
        if kind == "message_start":
            self.envelope = copy.deepcopy(value["message"])
            self.usage.update(self.envelope.get("usage") or {})
            if not self.started:
                self.started = True
                return [
                    {**value, "message": {**self.envelope, "id": self.identifier, "content": []}}
                ]
        elif kind == "content_block_start":
            index, block = value["index"], copy.deepcopy(value["content_block"])
            self.blocks[index] = block
            # Buffer tool blocks until the full call is validated. All other
            # blocks stream immediately, including opaque/redacted thinking.
            if block["type"] != "tool_use":
                self.indices[index] = len(self.visible[0]["content"])
                self.visible[0]["content"].append(block)
                return [{**value, "index": self.indices[index]}]
        elif kind == "content_block_delta":
            index, delta = value["index"], value["delta"]
            block = self.blocks[index]
            if delta["type"] == "input_json_delta":
                self.arguments[index] = self.arguments.get(index, "") + delta["partial_json"]
            else:
                for key in ("text", "thinking", "signature"):
                    if key in delta:
                        block[key] = block.get(key, "") + delta[key]
                if delta["type"] == "citations_delta":
                    block.setdefault("citations", []).append(copy.deepcopy(delta["citation"]))
            if index in self.indices:
                return [{**value, "index": self.indices[index]}]
        elif kind == "content_block_stop":
            index = value["index"]
            self.closed.add(index)
            if index in self.arguments:
                self.blocks[index]["input"] = orjson.loads(self.arguments[index])
            if index in self.indices:
                return [{**value, "index": self.indices[index]}]
        elif kind == "message_delta":
            self.finish = value.get("delta", {}).get("stop_reason") or self.finish
            self.envelope.update(value.get("delta", {}))
            self.usage.update(value.get("usage") or {})
        elif kind == "message_stop":
            self.stopped = True
        else:
            return [value]
        return []

    def load(self, value):
        self.envelope = copy.deepcopy(value)
        self.finish, self.usage = value["stop_reason"], value.get("usage") or {}
        self.blocks = dict(enumerate(copy.deepcopy(value["content"])))
        self.visible[0]["content"].extend(
            b for b in self.blocks.values() if b["type"] != "tool_use"
        )
        self.stopped, self.closed = True, set(self.blocks)

    def end(self):
        if not self.stopped or not self.finish or self.closed != set(self.blocks):
            raise ToolLoopError("Incomplete Anthropic message")
        blocks = [self.blocks[i] for i in sorted(self.blocks)]
        self.output = [{"role": "assistant", "content": blocks}]
        self.calls = [
            call(b["id"], b["name"], orjson.dumps(b["input"]).decode())
            for b in blocks
            if b["type"] == "tool_use"
        ]

    def publish_calls(self, client):
        ids = {c["id"] for c in client}
        events = []
        for block in self.blocks.values():
            if block["type"] != "tool_use" or block["id"] not in ids:
                continue
            index = len(self.visible[0]["content"])
            self.visible[0]["content"].append(block)
            events.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": orjson.dumps(block["input"]).decode(),
                        },
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
        return events

    def results(self, results):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["tool_call_id"],
                        "content": r["content"],
                    }
                    for r in results
                ],
            }
        ]

    def final(self, usage):
        return {
            **self.envelope,
            "id": self.identifier,
            "type": "message",
            "role": "assistant",
            "content": self.visible[0]["content"],
            "stop_reason": self.finish,
            "usage": usage,
        }

    def terminal(self, final):
        return [
            self.encode(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self.finish,
                        "stop_sequence": final.get("stop_sequence"),
                    },
                    "usage": final["usage"],
                }
            ),
            self.encode({"type": "message_stop"}),
        ]
