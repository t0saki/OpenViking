# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Archive observation runs under the capture mailbox lease, outside HTTP prepare."""

import asyncio
import time

from .client import VikingError

POLL_SECONDS = 5
MAX_PENDING_SECONDS = 900
TERMINAL = {"ready", "completed", "failed", "abandoned"}


async def observe_archive(viking, archive, token, session):
    now = time.time()
    if archive["status"] in TERMINAL or archive.get("next_check", 0) > now:
        return archive

    async def probe():
        try:
            summary = await viking.overview(token, session, archive["archive_id"])
        except VikingError:
            summary = ""
        if summary:
            return {"status": "ready", "summary": summary}
        state = await viking.archive_state(
            token, session, archive["archive_id"], archive["archive_uri"]
        )
        if state == "completed":
            try:
                summary = await viking.overview(token, session, archive["archive_id"])
            except VikingError:
                summary = ""
            if summary:
                return {"status": "ready", "summary": summary}
        return {"status": state}

    try:
        update = await asyncio.wait_for(probe(), timeout=POLL_SECONDS)
    except (VikingError, asyncio.TimeoutError):
        update = {"status": "unknown"}
    if update["status"] not in TERMINAL and now - archive["created"] >= MAX_PENDING_SECONDS:
        update = {"status": "abandoned"}
    return {**archive, **update, "next_check": now + POLL_SECONDS}
