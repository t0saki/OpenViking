# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
from types import SimpleNamespace

from openviking.retrieve.recall_rewrite import (
    normalize_recall_digest,
    rewrite_recall,
)


def test_normalize_recall_digest_requires_cited_bullets():
    assert normalize_recall_digest("NO_RELEVANT_MEMORY") == ""
    assert normalize_recall_digest("- uncited fact") == ""
    assert normalize_recall_digest("- fact (viking://user/a.md)") == (
        "OpenViking memory digest:\n- fact (viking://user/a.md)"
    )


async def test_rewrite_statuses_fail_open(monkeypatch):
    class Planner:
        async def get_completion_async(self, _prompt):
            return "NO_RELEVANT_MEMORY"

    config = SimpleNamespace(
        retrieval=SimpleNamespace(recall_rewrite_timeout_s=0.01),
        get_query_planner=lambda: Planner(),
    )
    monkeypatch.setattr("openviking.retrieve.recall_rewrite.get_openviking_config", lambda: config)
    assert await rewrite_recall(query="q", rendered="memory") == ("", "empty")

    class FailingPlanner:
        async def get_completion_async(self, _prompt):
            raise RuntimeError("boom")

    config.get_query_planner = lambda: FailingPlanner()
    assert await rewrite_recall(query="q", rendered="memory") == ("", "failed")

    class SlowPlanner:
        async def get_completion_async(self, _prompt):
            await asyncio.sleep(0.05)

    config.get_query_planner = lambda: SlowPlanner()
    assert await rewrite_recall(query="q", rendered="memory") == ("", "timeout")
    assert await rewrite_recall(query="q", rendered="") == ("", "empty")


async def test_recall_rewrite_keeps_rendered_on_success(client, service, monkeypatch):
    class FindResult:
        memories = [
            {
                "uri": "viking://user/test_user/memories/entities/a.md",
                "score": 0.9,
                "abstract": "A",
            }
        ]

    async def fake_find(**kwargs):
        return FindResult() if kwargs["target_uri"].endswith("/entities") else SimpleFind()

    class SimpleFind:
        memories = []

    async def fake_read(*_args, **_kwargs):
        return "full A"

    async def fake_rewrite(**kwargs):
        assert "viking://user/test_user/memories/entities/a.md" in kwargs["rendered"]
        return (
            "OpenViking memory digest:\n- A (viking://user/test_user/memories/entities/a.md)",
            "ok",
        )

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)
    monkeypatch.setattr(
        "openviking.server.routers.search.server_rewrite_enabled", lambda _mode: True
    )
    monkeypatch.setattr("openviking.server.routers.search.rewrite_recall", fake_rewrite)
    response = await client.post(
        "/api/v1/search/recall",
        json={
            "query": "A",
            "quotas": {"events": 0, "entities": 1, "preferences": 0},
            "rewrite": True,
        },
    )
    result = response.json()["result"]
    assert result["stats"]["rewrite"] == "ok"
    assert result["digest"].startswith("OpenViking memory digest:")
    assert "<memory" in result["rendered"]
