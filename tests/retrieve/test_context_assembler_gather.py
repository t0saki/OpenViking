# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
from types import SimpleNamespace

from openviking.retrieve.context_assembler.gather import (
    category_for,
    category_targets,
    dedupe_keep_best,
    gather_candidates,
    strip_level_suffix,
)
from openviking.retrieve.context_assembler.params import (
    PURPOSE_PRESETS,
    normalize_penalties,
    normalize_quotas,
)
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


class _FakeFindResult:
    def __init__(self, memories=None, resources=None, skills=None):
        self.memories = memories or []
        self.resources = resources or []
        self.skills = skills or []


def _ctx(actor_peer_id="current"):
    return RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id=actor_peer_id,
    )


def _service(find):
    return SimpleNamespace(search=SimpleNamespace(find=find), fs=SimpleNamespace())


def test_normalize_quotas_off_by_default():
    assert normalize_quotas(None) is None
    assert normalize_quotas(None, "coding") == PURPOSE_PRESETS["coding"]
    assert normalize_quotas({"events": 3, "bogus": 5}) == {"events": 3}
    assert normalize_quotas({"events": "nope"}) == {"events": 0}


def test_normalize_quotas_explicit_wins_over_purpose():
    assert normalize_quotas({"skills": 2}, "coding") == {"skills": 2}


def test_strip_level_suffix_detects_directory_hits():
    assert strip_level_suffix("viking://user/u/memories/events/.abstract.md") == (
        "viking://user/u/memories/events",
        True,
    )
    assert strip_level_suffix("viking://user/u/memories/events/x/.overview.md") == (
        "viking://user/u/memories/events/x",
        True,
    )
    assert strip_level_suffix("viking://user/u/memories/events/x.md") == (
        "viking://user/u/memories/events/x.md",
        False,
    )


def test_category_for_prefers_bucket_then_uri():
    item = {"uri": "viking://user/u/memories/entities/a.md"}
    assert category_for(item, "events") == "events"
    assert category_for(item, None) == "entities"
    assert category_for({"uri": "viking://resources/docs/a.md"}, None) == "resources"
    assert category_for({"uri": "viking://user/u/skills/a/SKILL.md"}, None) == "skills"


def test_category_targets_cover_peer_and_global_roots():
    ctx = _ctx()
    assert category_targets("events", ctx) == [
        "viking://user/test_user/memories/events",
        "viking://user/test_user/peers/current/memories/events",
    ]
    assert category_targets("resources", ctx) == [
        "viking://user/test_user/resources",
        "viking://resources",
    ]
    assert category_targets("skills", ctx) == [
        "viking://user/test_user/skills",
        "viking://agent/skills",
    ]


def test_dedupe_keep_best_keeps_highest_score():
    items = [
        {"uri": "viking://a", "score": 0.2},
        {"uri": "viking://a", "score": 0.7},
        {"uri": "viking://b", "score": 0.1},
    ]
    deduped = dedupe_keep_best(items)
    assert [item["uri"] for item in deduped] == ["viking://a", "viking://b"]
    assert deduped[0]["score"] == 0.7


async def test_quota_buckets_search_concurrently():
    started: list[str] = []
    all_started = asyncio.Event()

    async def fake_find(**kwargs):
        started.append(kwargs["target_uri"])
        if len(started) == 4:
            all_started.set()
        await all_started.wait()
        return _FakeFindResult()

    candidates, stats = await asyncio.wait_for(
        gather_candidates(
            service=_service(fake_find),
            ctx=_ctx(),
            queries=["parallel"],
            quotas={"events": 1, "entities": 1},
            limit=10,
            score_threshold=0.1,
            peer_scope="actor",
        ),
        timeout=1.0,
    )

    assert candidates == []
    assert set(started) == set(
        category_targets("events", _ctx()) + category_targets("entities", _ctx())
    )
    assert stats["searched"] == {"events": 0, "entities": 0}


async def test_flat_mode_merges_all_categories():
    async def fake_find(**kwargs):
        assert kwargs["target_uri"] == ""
        return _FakeFindResult(
            memories=[{"uri": "viking://user/test_user/memories/events/a.md", "score": 0.5}],
            resources=[{"uri": "viking://resources/doc.md", "score": 0.4}],
            skills=[{"uri": "viking://agent/skills/s/SKILL.md", "score": 0.3}],
        )

    candidates, stats = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["flat"],
        quotas=None,
        limit=10,
        score_threshold=None,
    )

    assert [c.category for c in candidates] == ["events", "resources", "skills"]
    assert stats["quotas"] is None
    assert stats["candidates"] == 3


async def test_flat_mode_keeps_the_owning_bucket_over_the_uri_shape():
    trap = "viking://resources/backup/memories/events/log.md"

    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult(resources=[{"uri": trap, "score": 0.5}])

    candidates, _ = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["flat"],
        quotas=None,
        limit=10,
        score_threshold=None,
    )

    assert [c.category for c in candidates] == ["resources"]


async def test_excluded_uris_are_compensated_with_extra_rows():
    limits: list[int] = []

    async def fake_find(**kwargs):
        limits.append(kwargs["limit"])
        return _FakeFindResult()

    await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["cooled"],
        quotas={"events": 4},
        limit=10,
        score_threshold=None,
        peer_scope="actor",
        excluded={f"viking://cooled/{i}" for i in range(3)},
    )

    assert limits and all(limit == 7 for limit in limits)


async def test_exclude_uris_filtered_and_counted():
    async def fake_find(**kwargs):
        return _FakeFindResult(
            memories=[
                {"uri": "viking://user/test_user/memories/events/a.md", "score": 0.5},
                {"uri": "viking://user/test_user/memories/events/b.md", "score": 0.4},
            ]
        )

    candidates, stats = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["excluded"],
        quotas=None,
        limit=10,
        score_threshold=None,
        excluded={"viking://user/test_user/memories/events/a.md"},
    )

    assert [c.base_uri for c in candidates] == ["viking://user/test_user/memories/events/b.md"]
    assert stats["excluded"] == 1


async def test_other_peer_penalty_demotes_foreign_hits():
    actor_uri = "viking://user/test_user/peers/current/memories/events/mine.md"
    other_uri = "viking://user/test_user/peers/other/memories/events/theirs.md"

    async def fake_find(**kwargs):
        target = kwargs["target_uri"]
        if target.endswith("/peers"):
            return _FakeFindResult(memories=[{"uri": other_uri, "score": 0.6}])
        if "/peers/current/" in target:
            return _FakeFindResult(memories=[{"uri": actor_uri, "score": 0.55}])
        return _FakeFindResult()

    candidates, stats = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["peer"],
        quotas={"events": 5},
        limit=10,
        score_threshold=None,
        peer_scope="all",
        penalties=normalize_penalties({"events": 0.1}),
    )

    assert [c.base_uri for c in candidates] == [actor_uri, other_uri]
    assert stats["origins"]["actor_peer"] == 1
    assert stats["origins"]["other_peer"] == 1


async def test_profile_memory_is_never_a_candidate():
    async def fake_find(**kwargs):
        return _FakeFindResult(
            memories=[{"uri": "viking://user/test_user/memories/profile.md", "score": 0.9}]
        )

    candidates, _ = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["profile"],
        quotas=None,
        limit=10,
        score_threshold=None,
    )
    assert candidates == []


async def test_failing_scope_does_not_break_assembly():
    async def fake_find(**kwargs):
        if "entities" in kwargs["target_uri"]:
            raise RuntimeError("index offline")
        return _FakeFindResult(
            memories=[{"uri": "viking://user/test_user/memories/events/a.md", "score": 0.5}]
        )

    candidates, stats = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["resilient"],
        quotas={"events": 2, "entities": 2},
        limit=10,
        score_threshold=None,
        peer_scope="actor",
    )

    assert [c.category for c in candidates] == ["events"]
    assert stats["searched"]["entities"] == 0
    # A broken scope must be visible, so an empty block is not mistaken for
    # "nothing relevant".
    assert any("index offline" in error for error in stats["retrieval_errors"])


async def test_healthy_retrieval_reports_no_errors():
    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult()

    _candidates, stats = await gather_candidates(
        service=_service(fake_find),
        ctx=_ctx(),
        queries=["healthy"],
        quotas=None,
        limit=10,
        score_threshold=None,
    )
    assert "retrieval_errors" not in stats
