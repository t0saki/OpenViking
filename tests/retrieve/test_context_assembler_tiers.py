# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace

from openviking.retrieve.context_assembler.gather import Candidate
from openviking.retrieve.context_assembler.tiers import (
    content_uri_for,
    doc_overview,
    extract_summary_section,
    floor_tier,
    overview_from_content,
    prefetch_contents,
    tier_text,
)
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


def _ctx():
    return RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )


def _candidate(uri, *, category="events", abstract="", is_directory=False, score=0.5, level=2):
    return Candidate(
        uri=uri,
        base_uri=uri,
        category=category,
        score=score,
        ranked_score=score,
        level=level,
        abstract=abstract,
        origin="self",
        is_directory=is_directory,
        read_ctx=_ctx(),
    )


def test_extract_summary_section_handles_heading_and_legacy_prefix():
    heading = "# Summary\n2026-07-14, shipped the thing.\n\n# 2026-07-14 ChatLog:\nnoise"
    assert extract_summary_section(heading) == "2026-07-14, shipped the thing."

    legacy = "Summary: an older event body\n2026-06-30 (Tuesday) ChatLog:\nnoise"
    assert extract_summary_section(legacy) == "an older event body"

    assert extract_summary_section("no summary anywhere") == ""


def test_doc_overview_returns_heading_tree_and_first_paragraph():
    content = "# Title\n\nIntro paragraph text.\n\n## Section A\n\nbody\n\n### Deep\n"
    overview = doc_overview(content)
    assert "# Title" in overview
    assert "  ## Section A" in overview
    assert "    ### Deep" in overview
    assert "Intro paragraph text." in overview


def test_doc_overview_without_headings_uses_paragraph():
    assert doc_overview("just prose here\n\nmore prose") == "just prose here"


def test_overview_dispatch_uses_code_outline_for_code_files():
    candidate = _candidate("viking://resources/pkg/mod.py", category="resources")
    content = "class Widget:\n    def render(self):\n        return 1\n\n\ndef helper(a, b):\n    return a\n"
    overview = overview_from_content(candidate, content)
    assert "class Widget" in overview
    assert "helper" in overview
    assert "return 1" not in overview


def test_overview_dispatch_uses_summary_for_memory_files():
    candidate = _candidate("viking://user/test_user/memories/events/a.md")
    content = "# Summary\nthe gist\n\n# 2026-07-14 ChatLog:\nlots of noise"
    assert overview_from_content(candidate, content) == "the gist"


def test_directory_overview_is_the_sidecar_body():
    candidate = _candidate("viking://user/test_user/memories/events", is_directory=True, level=0)
    assert overview_from_content(candidate, "  directory overview  ") == "directory overview"


def test_directory_floor_is_overview_and_full_is_capped():
    candidate = _candidate("viking://user/test_user/memories/events", is_directory=True, level=0)
    assert floor_tier(candidate) == "overview"
    assert content_uri_for(candidate).endswith("/.overview.md")

    contents = {content_uri_for(candidate): "dir overview"}
    assert tier_text(candidate, "overview", contents=contents) == "dir overview"
    assert tier_text(candidate, "full", contents=contents) is None


def test_file_floor_is_abstract_and_missing_abstract_is_unavailable():
    candidate = _candidate("viking://user/test_user/memories/events/a.md", abstract="short")
    assert floor_tier(candidate) == "abstract"
    assert tier_text(candidate, "abstract", contents={}) == "short"

    bare = _candidate("viking://user/test_user/memories/events/b.md")
    assert tier_text(bare, "abstract", contents={}) is None
    assert tier_text(bare, "uri", contents={}) == ""


async def test_prefetch_reads_memory_body_without_metadata_and_tolerates_failures():
    good = "viking://user/test_user/memories/events/a.md"
    bad = "viking://user/test_user/memories/events/b.md"
    raw = 'Visible body\n\n<!-- MEMORY_FIELDS\n{"event_name": "internal"}\n-->'

    async def fake_read(uri, **kwargs):
        del kwargs
        if uri == bad:
            raise RuntimeError("gone")
        return raw

    service = SimpleNamespace(fs=SimpleNamespace(read=fake_read))
    contents = await prefetch_contents(
        service=service,
        candidates=[_candidate(good), _candidate(bad)],
    )

    assert contents[good] == "Visible body"
    assert "MEMORY_FIELDS" not in contents[good]
    assert contents[bad] == ""


async def test_prefetch_reads_each_content_uri_once():
    reads: list[str] = []

    async def fake_read(uri, **kwargs):
        del kwargs
        reads.append(uri)
        return "body"

    service = SimpleNamespace(fs=SimpleNamespace(read=fake_read))
    uri = "viking://user/test_user/memories/events/a.md"
    await prefetch_contents(service=service, candidates=[_candidate(uri), _candidate(uri)])
    assert reads == [uri]
