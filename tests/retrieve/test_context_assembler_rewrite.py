# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
from types import SimpleNamespace

from openviking.retrieve.context_assembler import expansion as expansion_module
from openviking.retrieve.context_assembler import rewrite as rewrite_module
from openviking.retrieve.context_assembler.expansion import expand_queries
from openviking.retrieve.context_assembler.rewrite import (
    normalize_digest,
    rewrite_context,
    server_rewrite_enabled,
)


def _config(planner_factory, *, rewrite_timeout=0.01, intent_timeout=0.01, planner=None):
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            recall_rewrite_timeout_s=rewrite_timeout,
            recall_intent_timeout_s=intent_timeout,
        ),
        get_query_planner=planner_factory,
        query_planner=planner,
    )


def test_normalize_digest_keeps_only_cited_bullets():
    assert normalize_digest("NO_RELEVANT_MEMORY") == ""
    assert normalize_digest("- uncited fact") == ""
    assert normalize_digest("* fact (viking://user/a.md)") == (
        "OpenViking memory digest:\n- fact (viking://user/a.md)"
    )


def test_server_rewrite_enabled_resolves_auto_from_planner_config(monkeypatch):
    configured = SimpleNamespace(_has_any_config=lambda: True)
    monkeypatch.setattr(
        rewrite_module, "get_openviking_config", lambda: _config(None, planner=configured)
    )
    assert server_rewrite_enabled(True) is True
    assert server_rewrite_enabled(False) is False
    assert server_rewrite_enabled("auto") is True

    monkeypatch.setattr(
        rewrite_module, "get_openviking_config", lambda: _config(None, planner=None)
    )
    assert server_rewrite_enabled("auto") is False
    assert server_rewrite_enabled(True) is True


async def test_rewrite_statuses_fail_closed(monkeypatch):
    class Empty:
        async def get_completion_async(self, _prompt):
            return "NO_RELEVANT_MEMORY"

    config = _config(lambda: Empty())
    monkeypatch.setattr(rewrite_module, "get_openviking_config", lambda: config)
    assert await rewrite_context(query="q", rendered="memory") == ("", "empty", None)

    class Failing:
        async def get_completion_async(self, _prompt):
            raise RuntimeError("boom")

    config.get_query_planner = lambda: Failing()
    assert await rewrite_context(query="q", rendered="memory") == ("", "failed", None)

    class Slow:
        async def get_completion_async(self, _prompt):
            await asyncio.sleep(0.05)

    config.get_query_planner = lambda: Slow()
    assert await rewrite_context(query="q", rendered="memory") == ("", "timeout", None)
    assert await rewrite_context(query="q", rendered="") == ("", "empty", None)


class _Tracker:
    """Shape of the tracker the real VLM instance exposes."""

    def __init__(self):
        self.total = SimpleNamespace(prompt_tokens=10, completion_tokens=2, call_count=1)

    def get_total_usage(self):
        return self.total


class _Planner:
    """VLMConfig-shaped planner: the tracker lives on the model instance."""

    def __init__(self, extra_calls=0):
        self._instance = SimpleNamespace(token_tracker=_Tracker())
        self._extra_calls = extra_calls

    def get_vlm_instance(self):
        return self._instance

    async def get_completion_async(self, _prompt):
        calls = 1 + self._extra_calls
        self._instance.token_tracker.total = SimpleNamespace(
            prompt_tokens=210, completion_tokens=52, call_count=1 + calls
        )
        return "- fact (viking://user/a.md)"


async def test_rewrite_reports_usage_delta(monkeypatch):
    planner = _Planner()
    monkeypatch.setattr(
        rewrite_module, "get_openviking_config", lambda: _config(lambda: planner, rewrite_timeout=5)
    )

    digest, status, usage = await rewrite_context(query="q", rendered="memory")
    assert status == "ok"
    assert digest.startswith("OpenViking memory digest:")
    assert usage == {"prompt_tokens": 200, "completion_tokens": 50}


async def test_rewrite_usage_is_dropped_when_another_call_shared_the_tracker(monkeypatch):
    planner = _Planner(extra_calls=1)
    monkeypatch.setattr(
        rewrite_module, "get_openviking_config", lambda: _config(lambda: planner, rewrite_timeout=5)
    )

    _digest, status, usage = await rewrite_context(query="q", rendered="memory")
    assert status == "ok"
    assert usage is None


async def test_rewrite_usage_is_none_when_the_planner_has_no_tracker(monkeypatch):
    class Bare:
        async def get_completion_async(self, _prompt):
            return "- fact (viking://user/a.md)"

    monkeypatch.setattr(
        rewrite_module, "get_openviking_config", lambda: _config(lambda: Bare(), rewrite_timeout=5)
    )

    _digest, status, usage = await rewrite_context(query="q", rendered="memory")
    assert status == "ok"
    assert usage is None


async def test_expansion_is_off_without_session_or_mode():
    assert await expand_queries(query="q", session=None, mode="auto") == (["q"], "off")
    assert await expand_queries(query="q", session=object(), mode="off") == (["q"], "off")


async def test_expansion_skips_when_session_has_no_context():
    async def get_context_for_search(query, max_messages=20):
        del query, max_messages
        return {"latest_archive_overview": "", "current_messages": []}

    session = SimpleNamespace(get_context_for_search=get_context_for_search)
    assert await expand_queries(query="q", session=session, mode="auto") == (["q"], "off")


async def test_expansion_caps_planned_queries_and_keeps_original_first(monkeypatch):
    async def get_context_for_search(query, max_messages=20):
        del query, max_messages
        return {"latest_archive_overview": "prior work", "current_messages": ["m"]}

    class FakeAnalyzer:
        def __init__(self, max_recent_messages=5):
            del max_recent_messages

        async def analyze(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                queries=[
                    SimpleNamespace(query="second"),
                    SimpleNamespace(query="third"),
                    SimpleNamespace(query="fourth"),
                ]
            )

    monkeypatch.setattr(expansion_module, "IntentAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(
        expansion_module, "get_openviking_config", lambda: _config(None, intent_timeout=5)
    )

    session = SimpleNamespace(get_context_for_search=get_context_for_search)
    queries, status = await expand_queries(query="first", session=session, mode="auto")

    assert queries == ["first", "second", "third"]
    assert status == "used"


async def test_expansion_failure_falls_back_to_original_query(monkeypatch):
    async def get_context_for_search(query, max_messages=20):
        del query, max_messages
        return {"latest_archive_overview": "prior", "current_messages": ["m"]}

    class SlowAnalyzer:
        def __init__(self, max_recent_messages=5):
            del max_recent_messages

        async def analyze(self, **kwargs):
            del kwargs
            await asyncio.sleep(0.05)

    monkeypatch.setattr(expansion_module, "IntentAnalyzer", SlowAnalyzer)
    monkeypatch.setattr(
        expansion_module, "get_openviking_config", lambda: _config(None, intent_timeout=0.01)
    )

    session = SimpleNamespace(get_context_for_search=get_context_for_search)
    assert await expand_queries(query="q", session=session, mode="auto") == (["q"], "failed")
