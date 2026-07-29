# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from openviking.retrieve.context_assembler.budget import per_entry_cap, plan_entries
from openviking.retrieve.context_assembler.gather import Candidate
from openviking.retrieve.context_assembler.render import render_context
from openviking.retrieve.context_assembler.tiers import content_uri_for
from openviking.server.identity import RequestContext, Role
from openviking.utils.token_estimation import estimate_text_tokens
from openviking_cli.session.user_id import UserIdentifier


def _ctx():
    return RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )


def _candidate(
    name,
    *,
    score=0.5,
    category="events",
    abstract=None,
    is_directory=False,
):
    uri = f"viking://user/test_user/memories/{category}/{name}"
    abstract = f"abstract for {name}" if abstract is None else abstract
    return Candidate(
        uri=uri,
        base_uri=uri,
        category=category,
        score=score,
        ranked_score=score,
        level=0 if is_directory else 2,
        abstract=abstract,
        origin="self",
        is_directory=is_directory,
        read_ctx=_ctx(),
    )


def _memory_body(summary: str, filler: str = "") -> str:
    return f"# Summary\n{summary}\n\n# 2026-07-14 ChatLog:\n{filler}"


def test_per_entry_cap_is_twice_the_average_share():
    assert per_entry_cap(3000, 13) == 3000 // 13 * 2
    assert per_entry_cap(1600, 1) == 3200
    assert per_entry_cap(100, 0) == 200


def test_default_pass_leaves_no_bare_uri():
    candidates = [
        _candidate(f"{i}.md", score=0.4 + i / 100, category="entities", abstract=f"abstract {i}")
        for i in range(6)
    ]
    plan = plan_entries(candidates, {}, max_tokens=1600)

    assert len(plan.entries) == 6
    assert {entry.detail for entry in plan.entries} == {"abstract"}
    assert all(entry.text for entry in plan.entries)
    assert plan.stats["dropped"] == 0


def test_default_fill_upgrades_within_budget_and_reports_tiers():
    candidates = [_candidate(f"{i}.md", score=0.6 - i / 100) for i in range(3)]
    contents = {
        content_uri_for(candidate): _memory_body(f"summary for {candidate.base_uri}", "x" * 200)
        for candidate in candidates
    }

    plan = plan_entries(candidates, contents, max_tokens=1600)

    assert plan.stats["tier_counts"].get("full", 0) >= 1
    assert plan.stats["used_tokens"] <= 1600
    assert estimate_text_tokens(render_context(plan.entries)) <= 1600 + len(plan.entries)


def test_oversized_tier_falls_back_instead_of_truncating():
    small = _candidate("small.md", score=0.9)
    huge = _candidate("huge.md", score=0.95)
    contents = {
        content_uri_for(small): _memory_body("small gist"),
        content_uri_for(huge): _memory_body("huge gist", "y" * 40000),
    }

    plan = plan_entries([huge, small], contents, max_tokens=800)
    by_uri = {entry.uri: entry for entry in plan.entries}

    assert by_uri[huge.base_uri].detail in ("abstract", "overview")
    assert "y" * 100 not in by_uri[huge.base_uri].text
    assert by_uri[small.base_uri].detail == "full"


def test_leftover_budget_deepens_without_a_score_threshold():
    low = _candidate("low.md", score=0.02)
    contents = {content_uri_for(low): _memory_body("gist", "z" * 4000)}

    plan = plan_entries([low], contents, max_tokens=1600)
    assert plan.entries[0].detail == "full"

    crowded = plan_entries([low], contents, max_tokens=200)
    assert crowded.entries[0].detail == "overview"


def test_explicit_detail_pins_every_entry():
    candidate = _candidate("a.md", score=0.9)
    contents = {content_uri_for(candidate): _memory_body("gist", "w" * 100)}

    plan = plan_entries([candidate], contents, max_tokens=1600, detail="abstract")
    assert plan.entries[0].detail == "abstract"

    plan = plan_entries([candidate], contents, max_tokens=1600, detail="overview")
    assert plan.entries[0].detail == "overview"


def test_detail_accepts_a_per_category_map():
    event = _candidate("e.md", score=0.9)
    entity = _candidate("n.md", category="entities", score=0.8)
    contents = {
        content_uri_for(event): _memory_body("event gist", "w" * 100),
        content_uri_for(entity): _memory_body("entity gist", "w" * 100),
    }

    plan = plan_entries(
        [event, entity],
        contents,
        max_tokens=1600,
        detail={"events": "abstract", "entities": "full"},
    )
    by_uri = {entry.uri: entry.detail for entry in plan.entries}
    assert by_uri[event.base_uri] == "abstract"
    assert by_uri[entity.base_uri] == "full"


def test_unknown_detail_value_falls_back_to_the_defaults():
    candidate = _candidate("guide.md", category="resources", score=0.9, abstract="short abstract")
    contents = {content_uri_for(candidate): "# Title\n\nintro text\n\n## Body\n\nmore"}

    plan = plan_entries([candidate], contents, max_tokens=1600, detail="summary")
    assert plan.entries[0].detail == "abstract"
    assert plan.stats["detail"] is None


def test_resources_stay_at_their_abstract_until_asked_otherwise():
    candidate = _candidate("guide.md", category="resources", score=0.9, abstract="short abstract")
    contents = {content_uri_for(candidate): "# Title\n\nintro text\n\n## Body\n\nmore"}

    default = plan_entries([candidate], contents, max_tokens=1600)
    assert default.entries[0].detail == "abstract"

    explicit = plan_entries([candidate], contents, max_tokens=1600, detail="full")
    assert explicit.entries[0].detail == "full"


def test_directory_candidate_starts_at_overview():
    directory = _candidate("events", is_directory=True, abstract="", score=0.5)
    contents = {content_uri_for(directory): "directory overview body"}

    plan = plan_entries([directory], contents, max_tokens=1600)
    assert plan.entries[0].detail == "overview"
    assert plan.entries[0].text == "directory overview body"


def test_oversized_abstract_is_capped_like_any_other_tier():
    huge = _candidate("huge.md", category="entities", score=0.9, abstract="q" * 20000)
    small = _candidate("small.md", category="entities", score=0.8, abstract="tiny")

    plan = plan_entries([huge, small], {}, max_tokens=1600)
    by_uri = {entry.uri: entry for entry in plan.entries}

    assert by_uri[huge.base_uri].detail == "uri"
    assert by_uri[small.base_uri].detail == "abstract"
    assert plan.stats["used_tokens"] <= 1600


def test_oversized_abstract_prefers_overview_over_a_bare_uri():
    huge = _candidate("huge.md", category="entities", score=0.9, abstract="q" * 20000)
    contents = {content_uri_for(huge): _memory_body("the gist", "q" * 20000)}

    plan = plan_entries([huge], contents, max_tokens=1600)
    assert plan.entries[0].detail == "overview"
    assert plan.entries[0].text == "the gist"


def test_tiny_budget_degrades_then_drops():
    candidates = [
        _candidate(f"{i}.md", score=0.5, category="entities", abstract=f"{i}" + "a" * 200)
        for i in range(3)
    ]
    plan = plan_entries(candidates, {}, max_tokens=60)

    assert plan.stats["returned"] + plan.stats["dropped"] == 3
    assert plan.stats["dropped"] >= 1
    assert plan.stats["used_tokens"] <= 60
    assert all(entry.detail in ("abstract", "uri") for entry in plan.entries)


def test_identical_bodies_are_deduplicated():
    first = _candidate("a.md", score=0.6, abstract="same body")
    second = _candidate("b.md", score=0.5, abstract="same body")

    plan = plan_entries([first, second], {}, max_tokens=1600)
    assert [entry.uri for entry in plan.entries] == [first.base_uri]


def test_cjk_bodies_cost_more_than_ascii_of_equal_length():
    ascii_candidate = _candidate("ascii.md", abstract="a" * 200)
    cjk_candidate = _candidate("cjk.md", abstract="中" * 200)

    ascii_plan = plan_entries([ascii_candidate], {}, max_tokens=100000)
    cjk_plan = plan_entries([cjk_candidate], {}, max_tokens=100000)

    assert cjk_plan.stats["used_tokens"] > ascii_plan.stats["used_tokens"] * 4
