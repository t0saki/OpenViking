# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json

import httpx

from openviking.server.identity import RequestContext, Role
from openviking_cli.retrieve import ContextType, MatchedContext
from openviking_cli.session.user_id import UserIdentifier


class _FakeFindResult:
    def __init__(self, memories=None, resources=None, skills=None):
        self.memories = memories or []
        self.resources = resources or []
        self.skills = skills or []


def _memory(uri: str, score: float = 0.9, abstract: str = "", level: int = 2):
    return MatchedContext(
        uri=uri,
        context_type=ContextType.MEMORY,
        level=level,
        score=score,
        abstract=abstract,
        category=uri.split("/memories/", 1)[-1].split("/", 1)[0],
    )


async def test_context_mode_returns_flat_entries_with_no_bare_uris(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult(
            [
                _memory("viking://user/test_user/memories/events/launch.md", 0.71, "Launch note"),
                _memory("viking://user/test_user/memories/entities/ov.md", 0.55, "OpenViking"),
            ]
        )

    async def fake_read(uri, **kwargs):
        del kwargs
        return f"# Summary\ngist of {uri}\n\n# 2026-07-14 ChatLog:\n" + "x" * 300

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "what changed", "mode": "context", "max_tokens": 1200},
    )

    assert resp.status_code == 200
    result = resp.json()["result"]
    assert [entry["category"] for entry in result["entries"]] == ["events", "entities"]
    assert all(entry["detail"] != "uri" for entry in result["entries"])
    assert all(entry["uri"].startswith("viking://") for entry in result["entries"])
    assert result["rendered"].count("<memory ") == 2
    assert "<memory_group" not in result["rendered"]
    assert result["digest"] == ""
    assert result["stats"]["used_tokens"] <= 1200
    assert result["stats"]["rewrite"] == "off"


async def test_context_mode_quotas_fan_out_per_category(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    targets = []

    async def fake_find(**kwargs):
        targets.append(kwargs["target_uri"])
        return _FakeFindResult()

    monkeypatch.setattr(service.search, "find", fake_find)

    resp = await client.post(
        "/api/v1/search/search",
        json={
            "query": "quota",
            "mode": "context",
            "quotas": {"events": 2, "skills": 1},
            "peer_scope": "actor",
        },
    )

    assert resp.status_code == 200
    assert any(target.endswith("/memories/events") for target in targets)
    assert any(target.endswith("/user/default/skills") for target in targets)
    assert "viking://agent/skills" in targets
    stats = resp.json()["result"]["stats"]
    assert stats["quotas"] == {"events": 2, "skills": 1}
    assert stats["ignored"] == ["limit"]


async def test_context_mode_directory_hit_uses_overview_sidecar(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    directory = "viking://user/test_user/memories/events"

    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult([_memory(f"{directory}/.abstract.md", 0.45, "", level=0)])

    async def fake_read(uri, **kwargs):
        del kwargs
        assert uri == f"{directory}/.overview.md"
        return "events directory overview"

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "dirs", "mode": "context"},
    )

    assert resp.status_code == 200
    entries = resp.json()["result"]["entries"]
    assert [entry["uri"] for entry in entries] == [directory]
    assert entries[0]["detail"] == "overview"
    assert entries[0]["text"] == "events directory overview"


async def test_context_mode_sanitizes_nonfinite_scores(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult(
            [_memory("viking://user/test_user/memories/events/inf.md", float("inf"), "bad")]
        )

    async def fake_read(uri, **kwargs):
        del uri, kwargs
        return "small"

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.fs, "read", fake_read)

    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "inf", "mode": "context"},
    )

    assert resp.status_code == 200
    entries = resp.json()["result"]["entries"]
    assert entries and entries[0]["score"] == 0.0


async def test_context_params_are_rejected_in_list_mode(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "hello", "max_tokens": 1000},
    )
    assert resp.status_code == 400
    assert "mode='context'" in resp.text


async def test_target_uri_is_rejected_in_context_mode(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "hello", "mode": "context", "target_uri": "viking://user/test_user"},
    )
    assert resp.status_code == 400
    assert "target_uri" in resp.text


async def test_unknown_quota_keys_are_rejected(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "hello", "mode": "context", "quotas": {"bogus": 1}},
    )
    assert resp.status_code == 400
    assert "bogus" in resp.text


async def test_out_of_range_context_params_are_rejected(client: httpx.AsyncClient):
    for payload in (
        {"mode": "context", "max_tokens": 1},
        {"mode": "context", "full_score_threshold": 2.0},
        {"mode": "context", "detail": "everything"},
        {"mode": "context", "rewrite_max_bullets": 0},
        {"mode": "context", "exclude_uris": [f"viking://{i}" for i in range(201)]},
    ):
        resp = await client.post("/api/v1/search/search", json={"query": "hi", **payload})
        assert resp.status_code == 400, payload


async def test_list_mode_response_is_unchanged(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    hit = _memory("viking://user/test_user/memories/events/a.md", 0.5, "abs")

    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult([hit])

    async def fake_search(**kwargs):
        del kwargs
        return _FakeFindResult([hit])

    monkeypatch.setattr(service.search, "find", fake_find)
    monkeypatch.setattr(service.search, "search", fake_search)

    default = await client.post("/api/v1/search/search", json={"query": "plain"})
    explicit = await client.post("/api/v1/search/search", json={"query": "plain", "mode": "list"})

    assert default.status_code == 200
    assert default.json() == explicit.json()
    assert "rendered" not in default.json()["result"]


async def test_context_mode_reads_real_agfs_content_and_sidecars(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    """Only vector retrieval is faked here: reads go through the real AGFS path."""
    ctx = RequestContext(user=UserIdentifier.the_default_user("default"), role=Role.ROOT)
    root = "viking://user/default/memories"
    file_uri = f"{root}/events/launch.md"
    directory_uri = f"{root}/entities"

    await service.viking_fs.write_file(
        uri=file_uri,
        content=(
            "# Summary\nShipped the stdio MCP proxy.\n\n"
            "# 2026-07-14 ChatLog:\n" + ("chatter " * 400) + "\n\n"
            '<!-- MEMORY_FIELDS\n{"memory_type": "events"}\n-->'
        ),
        ctx=ctx,
    )
    await service.viking_fs.write_file(
        uri=f"{directory_uri}/.overview.md",
        content="Entities directory: projects, people and software.",
        ctx=ctx,
    )

    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult(
            [
                _memory(file_uri, 0.62, "launch abstract"),
                _memory(f"{directory_uri}/.abstract.md", 0.44, "", level=0),
            ]
        )

    monkeypatch.setattr(service.search, "find", fake_find)

    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "what shipped", "mode": "context", "max_tokens": 1200},
    )

    assert resp.status_code == 200
    result = resp.json()["result"]
    by_uri = {entry["uri"]: entry for entry in result["entries"]}

    # File hit: real read, memory metadata stripped, Summary section extracted.
    assert by_uri[file_uri]["detail"] in ("overview", "full")
    assert "Shipped the stdio MCP proxy." in by_uri[file_uri]["text"]
    assert "MEMORY_FIELDS" not in by_uri[file_uri]["text"]

    # Directory hit: starts at overview and reads the sidecar from AGFS.
    assert by_uri[directory_uri]["detail"] == "overview"
    assert by_uri[directory_uri]["text"] == "Entities directory: projects, people and software."

    assert result["stats"]["used_tokens"] <= 1200


async def test_context_mode_dedup_ledger_round_trips_through_agfs(
    client: httpx.AsyncClient,
    service,
    monkeypatch,
):
    file_uri = "viking://user/default/memories/events/repeat.md"
    ctx = RequestContext(user=UserIdentifier.the_default_user("default"), role=Role.ROOT)
    await service.viking_fs.write_file(uri=file_uri, content="# Summary\nrepeated fact", ctx=ctx)

    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult([_memory(file_uri, 0.7, "repeat abstract")])

    monkeypatch.setattr(service.search, "find", fake_find)

    create = await client.post("/api/v1/sessions", json={})
    assert create.status_code == 200
    session_id = create.json()["result"]["session_id"]

    payload = {
        "query": "repeat",
        "mode": "context",
        "session_id": session_id,
        "dedup_turns": 3,
        "query_expansion": "off",
    }

    first = await client.post("/api/v1/search/search", json=payload)
    assert first.status_code == 200
    assert [entry["uri"] for entry in first.json()["result"]["entries"]] == [file_uri]

    ledger_uri = f"viking://user/default/sessions/{session_id}/.recall_log.json"
    ledger = json.loads(await service.viking_fs.read_file(ledger_uri, ctx=ctx))
    assert file_uri in ledger["entries"]

    second = await client.post("/api/v1/search/search", json=payload)
    assert second.status_code == 200
    result = second.json()["result"]
    assert result["entries"] == []
    assert result["stats"]["dedup"]["cooled"] == 1
    assert result["stats"]["excluded"] == 1
