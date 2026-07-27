# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Flat XML rendering of assembled context.

Metadata lives in node attributes and the body is the tag content, so the
envelope costs a fraction of what recall v1's nested groups did.
"""

from __future__ import annotations

from typing import Sequence

from openviking.retrieve.context_assembler.models import AssembledEntry
from openviking.utils.token_estimation import estimate_text_tokens

FRAGMENT_SEPARATOR = "\n"


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _body(text: str) -> str:
    # Protect the envelope without escaping the body wholesale: agents read this
    # text as markdown, and escaping would also mangle the viking:// citations.
    return text.replace("</memory>", "<\\/memory>")


def render_entry(entry: AssembledEntry) -> str:
    head = (
        f'<memory uri="{_attr(entry.uri)}" type="{_attr(entry.category)}"'
        f' score="{entry.score:.2f}" detail="{entry.detail}"'
    )
    if not entry.text:
        return f"{head} />"
    return f"{head}>\n{_body(entry.text)}\n</memory>"


def render_context(entries: Sequence[AssembledEntry]) -> str:
    return FRAGMENT_SEPARATOR.join(render_entry(entry) for entry in entries)


def fragment_tokens(entry: AssembledEntry) -> int:
    """Token cost of one rendered fragment, envelope included."""
    return estimate_text_tokens(render_entry(entry))
