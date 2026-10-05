# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Stateless Responses output items and a single numbered event stream."""

import copy
import uuid

from ..protocols import messages_of
from .common import PREFIX, ToolLoopError, ToolProtocol, ToolRound, call

CLIENT_CALLS = {"function_call", "custom_tool_call"}


class ResponsesProtocol(ToolProtocol):
    field = "input"
    id_prefix = "resp_"

    @staticmethod
    def wire_tools(tools):
        return [{"type": "function", **t["function"], "strict": False} for t in tools]

    @staticmethod
    def block_reason(body):
        if any(m.get("type") == "item_reference" for m in messages_of(body, "responses")):
            return "tools_require_full_history"
        return ToolProtocol.block_reason(body)

    @classmethod
    def add_tools(cls, body, tools):
        super().add_tools(body, tools)
        body["include"] = list(
            dict.fromkeys([*body.get("include", []), "reasoning.encrypted_content"])
        )

    def __init__(self, body):
        super().__init__(body)
        self.sequence = 0

    def begin(self):
        super().begin()
        self.indices, self.buffered, self.items = {}, {}, {}
        self.announce, self.closed = not self.started, set()
        self.completed = False

    def encode(self, value):
        value = {**value, "sequence_number": self.sequence}
        if "response_id" in value:
            value["response_id"] = self.identifier
        if "response" in value:
            value["response"] = self.public_response(value["response"])
        self.sequence += 1
        return super().encode(value)

    def public_response(self, value):
        value = {**value, "id": self.identifier}
        if "tools" in value:
            value["tools"] = [t for t in value["tools"] if not t.get("name", "").startswith(PREFIX)]
        return value

    def event(self, value):
        kind = value["type"]
        if kind in {"response.created", "response.in_progress"}:
            if self.announce:
                self.envelope = copy.deepcopy(value["response"])
                self.started = True
                return [{**value, "response": {**value["response"], "output": []}}]
            return []
        if kind in {"response.completed", "response.incomplete"}:
            self.envelope = copy.deepcopy(value["response"])
            self.finish = self.envelope["status"]
            self.usage = self.envelope.get("usage") or {}
            self.output = self.envelope["output"]
            self.completed = True
            return []
        if kind == "response.failed":
            raise ToolLoopError("Model response failed")
        if "output_index" in value:
            index = value["output_index"]
            if kind == "response.output_item.added":
                item = value["item"]
                self.items[index] = copy.deepcopy(item)
                if item["type"] in CLIENT_CALLS:
                    self.buffered[index] = []
                else:
                    self.indices[index] = len(self.visible)
                    self.visible.append(copy.deepcopy(item))
            if kind == "response.output_item.done":
                self.closed.add(index)
                self.items[index] = copy.deepcopy(value["item"])
                if index in self.indices:
                    self.visible[self.indices[index]] = self.items[index]
            if index in self.buffered:
                self.buffered[index].append(value)
                return []
            return [{**value, "output_index": self.indices[index]}]
        # Events without an output item cannot expose a gateway function call.
        return [{**value, **({"response_id": self.identifier} if "response_id" in value else {})}]

    def load(self, value):
        self.envelope = copy.deepcopy(value)
        self.finish, self.usage = value["status"], value.get("usage") or {}
        self.output = copy.deepcopy(value["output"])
        self.visible.extend(item for item in self.output if item["type"] not in CLIENT_CALLS)
        self.completed = True

    def end(self):
        if not self.completed or self.finish not in {"completed", "incomplete"}:
            raise ToolLoopError("Incomplete Responses event stream")
        if self.body.get("stream"):
            # The terminal object is authoritative, including opaque reasoning
            # data that need not appear in deltas. Never reconstruct signatures.
            if set(self.items) != set(range(len(self.output))) or self.closed != set(self.items):
                raise ToolLoopError("Responses output items are missing")
            for index, item in enumerate(self.output):
                if index in self.indices:
                    self.visible[self.indices[index]] = item
        self.calls = [
            call(
                item["call_id"],
                (item.get("namespace", "") + "." if item.get("namespace") else "") + item["name"],
                item.get("arguments", item.get("input", "")),
            )
            for item in self.output
            if item["type"] in CLIENT_CALLS
        ]

        return ToolRound(self.output, self.calls, self.usage, self.finish == "completed")

    def publish_calls(self, client):
        ids, events = {c["id"] for c in client}, []
        for original_index, item in enumerate(self.output):
            if item["type"] not in CLIENT_CALLS or item["call_id"] not in ids:
                continue
            index = len(self.visible)
            self.visible.append(item)
            events.extend(
                {**event, "output_index": index} for event in self.buffered.get(original_index, [])
            )
        return events

    def open_notice(self):
        # Notices are a gateway-written assistant message, streamed like a model's.
        item = {
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "status": "in_progress",
            "content": [],
            "role": "assistant",
        }
        part = {"type": "output_text", "annotations": [], "text": ""}
        index = len(self.visible)
        events = [
            {"type": "response.output_item.added", "output_index": index, "item": dict(item)},
            {
                "type": "response.content_part.added",
                "output_index": index,
                "item_id": item["id"],
                "content_index": 0,
                "part": dict(part),
            },
        ]
        self.visible.append({**item, "content": [part]})
        return events

    def notice(self, text):
        item = self.visible[-1]
        item["content"][0]["text"] += text
        return [
            {
                "type": "response.output_text.delta",
                "output_index": len(self.visible) - 1,
                "item_id": item["id"],
                "content_index": 0,
                "delta": text,
            }
        ]

    def close_notice(self):
        item = self.visible[-1]
        item["status"] = "completed"
        part = item["content"][0]
        place = {"output_index": len(self.visible) - 1, "item_id": item["id"], "content_index": 0}
        return [
            {"type": "response.output_text.done", **place, "text": part["text"]},
            {"type": "response.content_part.done", **place, "part": dict(part)},
            {
                "type": "response.output_item.done",
                "output_index": place["output_index"],
                "item": copy.deepcopy(item),
            },
        ]

    def results(self, results):
        return [
            {"type": "function_call_output", "call_id": r["tool_call_id"], "output": r["content"]}
            for r in results
        ]

    def final(self, usage):
        return self.public_response({**self.envelope, "output": self.visible, "usage": usage})

    def terminal(self, final):
        return [{"type": "response." + self.finish, "response": final}]

    def error(self, message):
        return [
            {
                "type": "response.failed",
                "response": {
                    **self.envelope,
                    "object": "response",
                    "model": self.body.get("model"),
                    "status": "failed",
                    "error": {"code": "server_error", "message": message},
                    "incomplete_details": None,
                    "output": self.visible,
                },
            }
        ]
