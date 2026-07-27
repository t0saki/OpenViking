# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
import re
from types import SimpleNamespace

from openviking.retrieve.context_assembler import pipeline as pipeline_module
from openviking.retrieve.context_assembler.models import AssembledEntry
from openviking.retrieve.context_assembler.params import AssembleParams
from openviking.retrieve.context_assembler.pipeline import assemble_context
from openviking.retrieve.context_assembler.render import render_context, render_entry
from openviking.retrieve.context_assembler.rewrite import normalize_digest
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier

USER_ROOT = "viking://user/test_user"
SESSION_URI = f"{USER_ROOT}/sessions/s1"


def _ctx():
    return RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )


class _FakeFindResult:
    def __init__(self, memories=None, resources=None, skills=None):
        self.memories = memories or []
        self.resources = resources or []
        self.skills = skills or []


class _FakeVikingFS:
    def __init__(self):
        self.files = {}

    async def read_file(self, uri, ctx=None):
        del ctx
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None, lock_handle=None):
        del ctx, lock_handle
        self.files[uri] = content


def _service(*, hits, bodies, session=None, viking_fs=None):
    async def fake_find(**kwargs):
        del kwargs
        return _FakeFindResult(memories=list(hits))

    async def fake_read(uri, **kwargs):
        del kwargs
        if uri not in bodies:
            raise FileNotFoundError(uri)
        return bodies[uri]

    sessions = SimpleNamespace(session=lambda ctx, session_id: session)
    return SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=fake_read),
        sessions=sessions,
        viking_fs=viking_fs,
    )


def _fake_session(turn=6, overview="", messages=None):
    async def load():
        return None

    async def get_context_for_search(query, max_messages=20):
        del query, max_messages
        return {"latest_archive_overview": overview, "current_messages": messages or []}

    return SimpleNamespace(
        uri=SESSION_URI,
        meta=SimpleNamespace(total_message_count=turn),
        messages=[],
        load=load,
        get_context_for_search=get_context_for_search,
    )


def test_render_entry_is_flat_and_carries_metadata_as_attributes():
    entry = AssembledEntry(
        uri=f"{USER_ROOT}/memories/events/a.md",
        category="events",
        score=0.7239,
        detail="full",
        text="body text",
    )
    rendered = render_entry(entry)

    assert rendered.startswith(f'<memory uri="{USER_ROOT}/memories/events/a.md" type="events"')
    assert 'score="0.72"' in rendered
    assert 'detail="full"' in rendered
    assert rendered.endswith("</memory>")
    assert "<memory_group" not in rendered
    assert "<memory_section" not in rendered


def test_render_entry_without_body_is_self_closing():
    entry = AssembledEntry(uri="viking://x", category="events", score=0.1, detail="uri")
    assert render_entry(entry).endswith("/>")


def test_render_protects_the_envelope_from_body_content():
    entry = AssembledEntry(
        uri="viking://x", category="events", score=0.1, detail="full", text="a </memory> b"
    )
    rendered = render_entry(entry)
    assert len(re.findall(r"</memory>", rendered)) == 1


def test_render_context_joins_fragments_with_newlines():
    entries = [
        AssembledEntry(uri="viking://a", category="events", score=0.5, detail="uri"),
        AssembledEntry(uri="viking://b", category="entities", score=0.4, detail="uri"),
    ]
    assert render_context(entries).count("<memory ") == 2


def test_normalize_digest_requires_cited_bullets():
    assert normalize_digest("NO_RELEVANT_MEMORY") == ""
    assert normalize_digest("- a bullet without a citation") == ""

    digest = normalize_digest("preamble\n- fact 来源：viking://a\n- other 来源：viking://b", 1)
    assert digest == "OpenViking memory digest:\n- fact 来源：viking://a"


async def test_end_to_end_assembly_has_no_bare_uris_and_respects_budget():
    hits = [
        {"uri": f"{USER_ROOT}/memories/events/{name}.md", "score": score, "abstract": f"abs {name}"}
        for name, score in (("a", 0.62), ("b", 0.55), ("c", 0.41))
    ]
    bodies = {
        f"{USER_ROOT}/memories/events/{name}.md": f"# Summary\ngist {name}\n\n# ChatLog:\n{'x' * 300}"
        for name in ("a", "b", "c")
    }

    result = await assemble_context(
        service=_service(hits=hits, bodies=bodies),
        ctx=_ctx(),
        params=AssembleParams(query="what changed", max_tokens=1600),
    )

    assert len(result.entries) == 3
    assert all(entry.uri.startswith("viking://") for entry in result.entries)
    assert all(entry.detail != "uri" for entry in result.entries)
    assert result.stats["used_tokens"] <= 1600
    assert result.rendered.count("<memory ") == 3
    assert result.digest == ""
    assert result.stats["rewrite"] == "off"
    assert result.stats["query_expansion"] == "off"


async def test_directory_hit_starts_at_overview_instead_of_dumping_the_sidecar():
    directory = f"{USER_ROOT}/memories/events"
    hits = [{"uri": f"{directory}/.abstract.md", "score": 0.45, "abstract": ""}]
    bodies = {f"{directory}/.overview.md": "events directory overview"}

    result = await assemble_context(
        service=_service(hits=hits, bodies=bodies),
        ctx=_ctx(),
        params=AssembleParams(query="dir", max_tokens=1600),
    )

    assert [entry.uri for entry in result.entries] == [directory]
    assert result.entries[0].detail == "overview"
    assert result.entries[0].text == "events directory overview"


async def test_dedup_turns_excludes_uris_served_recently():
    uri = f"{USER_ROOT}/memories/events/a.md"
    hits = [{"uri": uri, "score": 0.6, "abstract": "abs a"}]
    viking_fs = _FakeVikingFS()
    viking_fs.files[f"{SESSION_URI}/.recall_log.json"] = json.dumps({"entries": {uri: {"turn": 5}}})

    result = await assemble_context(
        service=_service(hits=hits, bodies={}, session=_fake_session(turn=6), viking_fs=viking_fs),
        ctx=_ctx(),
        params=AssembleParams(query="again", session_id="s1", dedup_turns=3, query_expansion="off"),
    )

    assert result.entries == []
    assert result.stats["excluded"] == 1
    assert result.stats["dedup"]["cooled"] == 1


async def test_served_entries_are_recorded_in_the_ledger():
    uri = f"{USER_ROOT}/memories/events/a.md"
    hits = [{"uri": uri, "score": 0.6, "abstract": "abs a"}]
    viking_fs = _FakeVikingFS()

    result = await assemble_context(
        service=_service(hits=hits, bodies={}, session=_fake_session(turn=4), viking_fs=viking_fs),
        ctx=_ctx(),
        params=AssembleParams(query="first", session_id="s1", dedup_turns=3, query_expansion="off"),
    )

    assert len(result.entries) == 1
    stored = json.loads(viking_fs.files[f"{SESSION_URI}/.recall_log.json"])
    assert stored["entries"][uri]["turn"] == 4


async def test_query_expansion_fans_out_planned_queries(monkeypatch):
    queries_seen: list[str] = []

    async def fake_expand(*, query, session, mode, timeout_s=None):
        del session, mode, timeout_s
        return [query, "expanded query"], "used"

    monkeypatch.setattr(pipeline_module, "expand_queries", fake_expand)

    async def fake_find(**kwargs):
        queries_seen.append(kwargs["query"])
        return _FakeFindResult()

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=None),
        sessions=SimpleNamespace(session=lambda ctx, sid: _fake_session()),
        viking_fs=None,
    )

    result = await assemble_context(
        service=service,
        ctx=_ctx(),
        params=AssembleParams(query="short", session_id="s1", query_expansion="auto"),
    )

    assert queries_seen == ["short", "expanded query"]
    assert result.stats["planned_queries"] == ["short", "expanded query"]
    assert result.stats["query_expansion"] == "used"


async def test_rewrite_failure_keeps_rendered_context(monkeypatch):
    hits = [{"uri": f"{USER_ROOT}/memories/events/a.md", "score": 0.6, "abstract": "abs a"}]

    monkeypatch.setattr(pipeline_module, "server_rewrite_enabled", lambda mode: True)

    async def failing_rewrite(**kwargs):
        del kwargs
        return "", "timeout", None

    monkeypatch.setattr(pipeline_module, "rewrite_context", failing_rewrite)

    result = await assemble_context(
        service=_service(hits=hits, bodies={}),
        ctx=_ctx(),
        params=AssembleParams(query="rewrite me", rewrite=True),
    )

    assert result.digest == ""
    assert result.rendered.count("<memory ") == 1
    assert result.stats["rewrite"] == "timeout"


async def test_successful_rewrite_reports_digest_and_usage(monkeypatch):
    hits = [{"uri": f"{USER_ROOT}/memories/events/a.md", "score": 0.6, "abstract": "abs a"}]

    monkeypatch.setattr(pipeline_module, "server_rewrite_enabled", lambda mode: True)

    async def ok_rewrite(**kwargs):
        del kwargs
        return (
            "OpenViking memory digest:\n- fact 来源：viking://a",
            "ok",
            {"prompt_tokens": 120, "completion_tokens": 30},
        )

    monkeypatch.setattr(pipeline_module, "rewrite_context", ok_rewrite)

    result = await assemble_context(
        service=_service(hits=hits, bodies={}),
        ctx=_ctx(),
        params=AssembleParams(query="rewrite me", rewrite="auto"),
    )

    assert result.digest.startswith("OpenViking memory digest:")
    assert result.rendered
    assert result.stats["rewrite_usage"] == {"prompt_tokens": 120, "completion_tokens": 30}


async def test_purpose_preset_activates_bucketed_quotas():
    targets: list[str] = []

    async def fake_find(**kwargs):
        targets.append(kwargs["target_uri"])
        return _FakeFindResult()

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=None),
        sessions=SimpleNamespace(),
        viking_fs=None,
    )

    result = await assemble_context(
        service=service,
        ctx=_ctx(),
        params=AssembleParams(query="coding", purpose="coding", peer_scope="actor"),
    )

    assert f"{USER_ROOT}/memories/experiences" in targets
    assert result.stats["quotas"]["experiences"] == 2
    assert result.stats["purpose"] == "coding"
