# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import httpx

from openviking.retrieve.context_assembler.params import DEFAULT_QUOTAS, PURPOSE_PRESETS
from openviking.retrieve.context_assembler.recall_preset import (
    RECALL_SCORE_THRESHOLD,
    fold_recall_request,
)
from openviking_cli.retrieve import ContextType, MatchedContext


class _FakeFindResult:
    def __init__(self, memories):
        self.memories = memories


def _memory(uri: str, score: float = 0.9, abstract: str = ""):
    return MatchedContext(
        uri=uri,
        context_type=ContextType.MEMORY,
        level=2,
        score=score,
        abstract=abstract,
        category=uri.split("/memories/", 1)[-1].split("/", 1)[0],
    )


def test_v1_max_chars_folds_into_a_token_budget():
    params, aliases = fold_recall_request({"query": "q", "max_chars": 6500}, {"max_chars"})
    assert params.max_tokens == 1625
    assert aliases == ["max_chars"]

    explicit, _ = fold_recall_request(
        {"query": "q", "max_chars": 6500, "max_tokens": 900}, {"max_chars", "max_tokens"}
    )
    assert explicit.max_tokens == 900


def test_v1_min_score_folds_and_preset_applies_when_absent():
    params, aliases = fold_recall_request({"query": "q", "min_score": 0.1}, {"min_score"})
    assert params.score_threshold == 0.1
    assert aliases == ["min_score"]

    preset, aliases = fold_recall_request({"query": "q"}, set())
    assert preset.score_threshold == RECALL_SCORE_THRESHOLD
    assert aliases == []


def test_v1_render_tristate_maps_onto_detail():
    rendered, _ = fold_recall_request({"query": "q", "render": True}, {"render"})
    assert (rendered.render, rendered.detail) == (True, None)

    entries_only, _ = fold_recall_request({"query": "q", "render": False}, {"render"})
    assert (entries_only.render, entries_only.detail) == (False, None)

    compact, _ = fold_recall_request({"query": "q", "render": "compact"}, {"render"})
    assert (compact.render, compact.detail) == (True, "abstract")

    override, _ = fold_recall_request(
        {"query": "q", "render": "compact", "detail": "full"}, {"render", "detail"}
    )
    assert override.detail == "full"


def test_quotas_default_to_v1_buckets_and_null_opts_into_purpose():
    omitted, _ = fold_recall_request({"query": "q"}, set())
    assert omitted.quotas == DEFAULT_QUOTAS
    assert omitted.purpose == "coding"

    explicit_null, _ = fold_recall_request({"query": "q", "quotas": None}, {"quotas"})
    assert explicit_null.quotas is None
    assert PURPOSE_PRESETS[explicit_null.purpose]["experiences"] == 2


def test_dedup_turns_only_default_on_with_a_session():
    stateless, _ = fold_recall_request({"query": "q"}, set())
    assert stateless.dedup_turns == 0

    stateful, _ = fold_recall_request({"query": "q", "session_id": "s1"}, {"session_id"})
    assert stateful.dedup_turns == 5

    pinned, _ = fold_recall_request(
        {"query": "q", "session_id": "s1", "dedup_turns": 0}, {"session_id", "dedup_turns"}
    )
    assert pinned.dedup_turns == 0


async def test_recall_endpoint_assembles_tiers_and_signals_deprecation(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    calls = []

    async def fake_find(**kwargs):
        calls.append(kwargs)
        target_uri = kwargs["target_uri"]
        if target_uri.endswith("/events"):
            return _FakeFindResult(
                [
                    _memory(
                        "viking://user/default/memories/events/launch.md",
                        0.91,
                        "Launch decision",
                    )
                ]
            )
        if target_uri.endswith("/entities"):
            return _FakeFindResult(
                [
                    _memory(
                        "viking://user/default/memories/entities/openviking.md",
                        0.82,
                        "OpenViking project",
                    )
                ]
            )
        return _FakeFindResult([])

    async def fake_read(uri, **kwargs):
        del kwargs
        if uri.endswith("/launch.md"):
            return "# Summary\nShip stdio MCP proxy.\n\n# ChatLog:\n" + "x" * 2000
        return "OpenViking is the target project."

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    resp = await client.post(
        "/api/v1/search/recall",
        json={
            "query": "what should I remember",
            "quotas": {"events": 1, "entities": 1, "preferences": 0, "experiences": 0},
            "max_chars": 1600,
            "min_score": 0.1,
            "render": True,
        },
    )

    assert resp.status_code == 200
    assert resp.headers["Deprecation"] == "true"
    body = resp.json()
    assert body["status"] == "ok"
    result = body["result"]
    assert result["stats"]["returned"] == 2
    assert {entry["category"] for entry in result["entries"]} == {"events", "entities"}
    assert all(entry["detail"] != "uri" for entry in result["entries"])
    assert result["rendered"].count("<memory ") == 2
    assert result["stats"]["deprecated"]["aliases_used"] == ["max_chars", "min_score", "render"]
    assert [call["target_uri"].rsplit("/", 1)[-1] for call in calls] == [
        "events",
        "peers",
        "entities",
        "peers",
    ]


async def test_recall_endpoint_keeps_every_entry_readable_under_a_tight_budget(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    async def fake_find(**kwargs):
        if kwargs["target_uri"].endswith("/events"):
            return _FakeFindResult(
                [
                    _memory("viking://user/default/memories/events/big.md", 0.9, "big event"),
                    _memory("viking://user/default/memories/events/big2.md", 0.8, "second"),
                ]
            )
        return _FakeFindResult([])

    async def fake_read(uri, **kwargs):
        del kwargs
        return "# Summary\n" + ("s" * 500) + "\n\n# ChatLog:\n" + ("x" * 2000) + f"\n{uri}"

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    # A budget that fits the abstract floor but nothing deeper: every entry stays
    # readable instead of degrading to a bare URI.
    resp = await client.post(
        "/api/v1/search/recall",
        json={
            "query": "budget",
            "quotas": {"events": 2, "entities": 0, "preferences": 0},
            "max_tokens": 120,
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert [entry["detail"] for entry in result["entries"]] == ["abstract", "abstract"]
    assert result["stats"]["used_tokens"] <= 120


async def test_recall_endpoint_degrades_then_drops_when_nothing_fits(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    long_abstract = "s" * 500
    long_name = "n" * 200

    async def fake_find(**kwargs):
        if kwargs["target_uri"].endswith("/events"):
            return _FakeFindResult(
                [
                    _memory(
                        f"viking://user/default/memories/events/{long_name}-1.md",
                        0.9,
                        long_abstract,
                    ),
                    _memory(
                        f"viking://user/default/memories/events/{long_name}-2.md",
                        0.8,
                        long_abstract,
                    ),
                ]
            )
        return _FakeFindResult([])

    async def fake_read(uri, **kwargs):
        del kwargs
        return f"# Summary\n{'x' * 4000}\n\n# ChatLog:\n{uri}"

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    # Abstract does not fit, so the entry degrades to the URI fragment.
    resp = await client.post(
        "/api/v1/search/recall",
        json={
            "query": "budget",
            "quotas": {"events": 2, "entities": 0, "preferences": 0},
            "max_tokens": 200,
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert [entry["detail"] for entry in result["entries"]] == ["uri", "uri"]
    assert result["stats"]["used_tokens"] <= 200

    # Not even one URI fragment fits: drop rather than overrun the contract.
    resp = await client.post(
        "/api/v1/search/recall",
        json={
            "query": "budget",
            "quotas": {"events": 2, "entities": 0, "preferences": 0},
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["entries"] == []
    assert result["rendered"] == ""
    assert result["stats"]["dropped"] == 2


async def test_recall_endpoint_sanitizes_nonfinite_scores(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    async def fake_find(**kwargs):
        if kwargs["target_uri"].endswith("/events"):
            return _FakeFindResult(
                [_memory("viking://user/default/memories/events/inf.md", float("inf"), "bad score")]
            )
        return _FakeFindResult([])

    async def fake_read(uri, **kwargs):
        del uri, kwargs
        return "small content"

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    resp = await client.post(
        "/api/v1/search/recall",
        json={"query": "inf", "quotas": {"events": 1, "entities": 0, "preferences": 0}},
    )

    assert resp.status_code == 200
    entries = resp.json()["result"]["entries"]
    assert entries and entries[0]["score"] == 0.0


async def test_recall_endpoint_rejects_unknown_fields(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/recall",
        json={"query": "hello", "unexpected": "value"},
    )

    assert resp.status_code == 400


async def test_recall_endpoint_filters_profile_and_duplicates(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    async def fake_find(**kwargs):
        del kwargs
        duplicate = _memory("viking://user/default/memories/events/dup.md", 0.8, "same")
        profile = _memory("viking://user/default/memories/profile.md", 0.99, "profile")
        return _FakeFindResult([profile, duplicate, duplicate])

    async def fake_read(uri, **kwargs):
        del kwargs
        if uri.endswith("profile.md"):
            return "profile"
        return "duplicate content"

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    resp = await client.post(
        "/api/v1/search/recall",
        json={"query": "hello", "quotas": {"events": 3, "entities": 0, "preferences": 0}},
    )

    assert resp.status_code == 200
    entries = resp.json()["result"]["entries"]
    assert [entry["uri"] for entry in entries] == ["viking://user/default/memories/events/dup.md"]
