# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native protocol adapters for the shared hidden tool loop."""

from .anthropic import AnthropicProtocol
from .chat import ChatProtocol
from .responses import ResponsesProtocol

PROTOCOLS = {"chat": ChatProtocol, "responses": ResponsesProtocol, "anthropic": AnthropicProtocol}


def tool_protocol(protocol, body):
    return PROTOCOLS[protocol](body)
