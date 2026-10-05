# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native protocol adapters for the shared hidden tool loop."""

from ..records import RecordKind as K
from .anthropic import AnthropicProtocol
from .chat import ChatProtocol
from .common import ToolProtocol
from .responses import ResponsesProtocol

PROTOCOLS = {"chat": ChatProtocol, "responses": ResponsesProtocol, "anthropic": AnthropicProtocol}


def tool_protocol(protocol: str) -> type[ToolProtocol]:
    return PROTOCOLS[protocol]


def hidden_chain(messages: list[dict], protocol: str) -> list[str]:
    return tool_protocol(protocol).history_chain(messages)


def replay_hidden(
    messages: list[dict], chain: list[str], records: dict, protocol: str
) -> list[dict]:
    """Expand each matching visible span into its immutable native transcript."""
    result, offsets = [], []
    for index, (message, anchor) in enumerate(zip(messages, chain, strict=True)):
        offsets.append(len(result))
        result.append(message)
        record = records.get((K.HIDDEN, anchor))
        if record:
            count = record["visible_count"]
            if 0 < count <= index + 1:
                result[offsets[index + 1 - count] :] = record["messages"]
    return tool_protocol(protocol).join_replayed_history(result)
