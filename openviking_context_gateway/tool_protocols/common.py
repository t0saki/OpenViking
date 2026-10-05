# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Wire adapters preserve native history; only executable calls are normalized.

Adapters assemble one upstream response and project visible events. They do not
execute tools, own budgets, or access storage. The shared loop owns those steps.
"""

import copy
import uuid

import orjson


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


class ToolProtocol:
    """Per-request adapter contract.

    begin/event/load/end assemble a round's native output, executable calls,
    usage and stop reason. publish_calls adds only client-owned calls to visible
    history. results encodes executor receipts; final/terminal publish the one
    combined response. No adapter makes a network or persistence decision.
    """

    field = "messages"
    id_prefix = ""

    def __init__(self, body):
        self.body = body
        self.identifier = self.id_prefix + "ovcg-" + uuid.uuid4().hex
        self.visible = []
        self.started = False

    def begin(self):
        self.output, self.calls, self.usage, self.envelope = [], [], {}, {}
        self.finish = None

    def encode(self, value):
        return b"event: " + value["type"].encode() + b"\n" + sse(value)

    def error(self, message):
        return self.encode(
            {"type": "error", "error": {"type": "gateway_tool_error", "message": message}}
        )
