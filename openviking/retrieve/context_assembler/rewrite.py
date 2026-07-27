# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Optional LLM digest over assembled context.

Opt-in and fail-closed: when the model is slow, absent, or off-contract the
caller still gets the unrewritten ``rendered`` block.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Literal, Optional, Tuple

from openviking.prompts import render_prompt
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

DIGEST_HEADER = "OpenViking memory digest:"
MAX_BULLET_CHARS = 500


def server_rewrite_enabled(mode: bool | Literal["auto"]) -> bool:
    """Resolve the API mode without constructing a model client."""
    if mode is True:
        return True
    if mode != "auto":
        return False
    planner = get_openviking_config().query_planner
    return planner is not None and planner._has_any_config()


def normalize_digest(raw: Any, max_bullets: int = 6) -> str:
    """Accept only the small, cited digest contract emitted by the prompt."""
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
        bullets.append(cleaned[:MAX_BULLET_CHARS].rstrip())
        if len(bullets) >= max(1, max_bullets):
            break
    if not bullets:
        return ""
    return f"{DIGEST_HEADER}\n" + "\n".join(bullets)


def _usage_snapshot(planner: Any) -> Tuple[int, int]:
    try:
        total = planner.token_tracker.get_total_usage()
        return int(total.prompt_tokens), int(total.completion_tokens)
    except Exception:
        return 0, 0


async def rewrite_context(
    *,
    query: str,
    rendered: str,
    max_bullets: int = 6,
    timeout_s: Optional[float] = None,
) -> Tuple[str, str, Optional[Dict[str, int]]]:
    """Return ``(digest, status, usage)`` while containing every model failure."""
    if not rendered.strip():
        return "", "empty", None

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
        planner = config.get_query_planner()
    except Exception as exc:
        logger.warning("Rewrite planner unavailable: %s", exc)
        return "", "failed", None

    before = _usage_snapshot(planner)
    try:
        response = await asyncio.wait_for(
            planner.get_completion_async(prompt),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning("Context rewrite timed out after %.2fs", timeout)
        return "", "timeout", None
    except Exception as exc:
        logger.warning("Context rewrite failed: %s", exc)
        return "", "failed", None

    after = _usage_snapshot(planner)
    usage: Optional[Dict[str, int]] = None
    if after != before:
        usage = {
            "prompt_tokens": max(0, after[0] - before[0]),
            "completion_tokens": max(0, after[1] - before[1]),
        }

    digest = normalize_digest(response, max_bullets=max_bullets)
    return (digest, "ok", usage) if digest else ("", "empty", usage)
