# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared, bounded archive observation for capture and request preparation."""

import asyncio
import time

from .client import VikingError
from .records import RecordKind as K

POLL_SECONDS = 5
MAX_PENDING_SECONDS = 900
TERMINAL = {"ready", "completed", "failed", "abandoned"}


async def refresh_archive(store, viking, scope, session, anchor, archive, token, ov_session):
    now = time.time()
    if archive.get("status") in TERMINAL or archive.get("next_check", 0) > now:
        return archive
    key = K.ARCHIVE, anchor
    claimed = {**archive, "created": archive.get("created", now), "next_check": now + POLL_SECONDS}
    if not await store.commit(scope, session, {key: claimed}, expected={key: archive}):
        fresh = await store.read(scope, session, [anchor], [K.ARCHIVE])
        return fresh.get(key, archive)

    async def probe():
        try:
            summary = await viking.overview(token, ov_session, archive["archive_id"])
        except VikingError:
            summary = ""
        if summary:
            return {"status": "ready", "summary": summary}
        state = await viking.archive_state(
            token, ov_session, archive["archive_id"], archive.get("archive_uri", "")
        )
        # A summary may land between the overview read and the terminal marker.
        if state == "completed":
            try:
                summary = await viking.overview(token, ov_session, archive["archive_id"])
            except VikingError:
                summary = ""
            if summary:
                return {"status": "ready", "summary": summary}
        return {"status": state}

    try:
        update = await asyncio.wait_for(probe(), timeout=POLL_SECONDS)
    except (VikingError, asyncio.TimeoutError):
        update = {"status": "unknown"}
    if update["status"] not in TERMINAL and now - claimed["created"] >= MAX_PENDING_SECONDS:
        update = {"status": "abandoned"}
    result = {**claimed, **update}
    await store.commit(scope, session, {key: result}, expected={key: claimed})
    return result
