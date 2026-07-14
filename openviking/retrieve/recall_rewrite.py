# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Optional LLM rewrite for bounded recall results."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from openviking.prompts import render_prompt
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


def server_rewrite_enabled(mode: bool | Literal["auto"]) -> bool:
    """Resolve the API mode without constructing a model client."""
    if mode is True:
        return True
    if mode != "auto":
        return False
    planner = get_openviking_config().query_planner
    return planner is not None and planner._has_any_config()


def normalize_recall_digest(raw: Any, max_bullets: int = 6) -> str:
    """Accept only the small, cited digest contract emitted by the rewrite prompt."""
    text = str(raw or "").strip()
    if not text or "NO_RELEVANT_MEMORY" in text.upper():
        return ""

    bullets: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not re.match(r"^[-*]\s+", cleaned):
            continue
        cleaned = re.sub(r"^[-*]\s+", "- ", cleaned)
        if "viking://" not in cleaned:
            continue
        bullets.append(cleaned[:500].rstrip())
        if len(bullets) >= max(1, max_bullets):
            break
    if not bullets:
        return ""
    return "OpenViking memory digest:\n" + "\n".join(bullets)


async def rewrite_recall(
    *,
    query: str,
    rendered: str,
    max_bullets: int = 6,
    timeout_s: float | None = None,
) -> tuple[str, str]:
    """Return ``(digest, status)`` while containing every model failure."""
    if not rendered.strip():
        return "", "empty"

    config = get_openviking_config()
    timeout = timeout_s or config.retrieval.recall_rewrite_timeout_s
    prompt = render_prompt(
        "retrieval.recall_rewrite",
        {
            "query": query,
            "rendered": rendered,
            "max_bullets": max(1, max_bullets),
        },
    )
    try:
        response = await asyncio.wait_for(
            config.get_query_planner().get_completion_async(prompt),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning("Recall rewrite timed out after %.2fs", timeout)
        return "", "timeout"
    except Exception as exc:
        logger.warning("Recall rewrite failed: %s", exc)
        return "", "failed"

    digest = normalize_recall_digest(response, max_bullets=max_bullets)
    return (digest, "ok") if digest else ("", "empty")
