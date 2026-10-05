# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Freeze MCP tool definitions with small gateway-specific overrides."""

import copy
import html
import re

from .protocols import enhanced_supported, messages_of
from .tool_protocols import tool_protocol
from .tool_protocols.common import PREFIX

TOOL_VERSION = 2
TOOL_OVERRIDES = {
    "find": {"notice": ("query",)},
    "search": {"notice": ("query",)},
    "read": {"notice": ("uris",)},
    "list": {"notice": ("uri",)},
    "write": {"notice": ("uri",)},
    "add_resource": {
        "notice": ("path", "attachment_index"),
        "attachment": True,
        "description": (
            " For a file attached to this conversation, set attachment_index (zero-based). "
            "For a local path, follow the returned upload instructions using the client's "
            "shell tool. The gateway cannot read local paths."
        ),
    },
    "add_skill": {
        "notice": ("path", "target_uri", "attachment_index", "data"),
        "attachment": True,
        "description": (
            " For a file attached to this conversation, set attachment_index (zero-based). "
            "Local directories must be zipped and uploaded using the client's shell tool "
            "and the returned signed instructions. The gateway cannot read local paths."
        ),
    },
}

def has_shell(body):
    return any(
        re.search(
            r"(^|[_-])(bash|shell|exec_command|terminal|run_command)([_-]|$)",
            tool.get("function", tool).get("name", ""),
            re.I,
        )
        for tool in body.get("tools", [])
        if isinstance(tool, dict)
    )


def attachments(body):
    output = []
    for message in messages_of(body, "responses" if "input" in body else "chat"):
        if message.get("role") not in {"user", "system"}:
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"file", "input_file"}:
                    output.append(part.get("file", part))
                elif isinstance(part, dict) and part.get("type") == "document":
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        output.append(
                            {
                                "filename": part.get("title", "attachment.pdf"),
                                "file_data": source.get("data", ""),
                            }
                        )
                    elif source.get("type") == "text":
                        output.append(
                            {
                                "filename": part.get("title", "attachment.txt"),
                                "text": source.get("data", ""),
                            }
                        )
        # Open WebUI can send extracted documents as source tags instead of bytes.
        # Only marked source text is importable, never arbitrary conversation text.
        texts = (
            [content]
            if isinstance(content, str)
            else [p.get("text", "") for p in content if isinstance(p, dict)]
        )
        for text in texts:
            for context in re.findall(r"<context>([\s\S]*?)</context>", text):
                for index, (attrs, source) in enumerate(
                    re.findall(r"<source\b([^>]*)>([\s\S]*?)</source>", context)
                ):
                    name = re.search(r'name=["\']([^"\']+)["\']', attrs)
                    filename = html.unescape(name[1]) if name else f"source-{index}.txt"
                    output.append(
                        {
                            "filename": filename + ".txt"
                            if not filename.endswith(".txt")
                            else filename,
                            "text": html.unescape(source),
                        }
                    )
    return output


def tool_block_reason(body, protocol, upstream):
    if not enhanced_supported(body, protocol):
        return "tools_require_full_history"
    if not upstream.get("allow_gateway_tools", True):
        return "upstream_tools_disabled"
    if body.get("n", 1) != 1:
        return "tools_multiple_choices"
    output_format = (
        body.get("response_format")
        or body.get("text", {}).get("format")
        or body.get("output_config", {}).get("format")
        or {}
    )
    if output_format.get("type", "text") != "text":
        return "tools_structured_output"
    reason = tool_protocol(protocol).block_reason(body)
    if reason:
        return reason
    if upstream.get("vendor") == "deepseek":
        thinking = body.get("thinking") or {}
        if thinking.get("type") != "disabled":
            return "deepseek_reasoning_history_required"
    return ""


def select_tools(body, protocol, upstream, policy, catalog):
    if not policy.get("gateway_tools") or tool_block_reason(body, protocol, upstream):
        return []
    disabled = set(policy.get("disabled_tools", []))
    selected = []
    for tool in catalog:
        name = tool["name"]
        if name in disabled:
            continue
        override = TOOL_OVERRIDES.get(name, {})
        schema = copy.deepcopy(tool["inputSchema"])
        if override.get("attachment"):
            schema.setdefault("properties", {})["attachment_index"] = {
                "type": "integer",
                "minimum": 0,
                "description": "Zero-based index of a file attached to this conversation.",
            }
        selected.append(
            {
                "type": "function",
                "function": {
                    "name": PREFIX + name,
                    "description": tool.get("description", "") + override.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return selected
