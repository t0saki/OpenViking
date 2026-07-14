# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
from types import SimpleNamespace

from openviking.server.identity import RequestContext, Role
from openviking.server.routers.search import RecallRequest, _recall_query_plan
from openviking_cli.session.user_id import UserIdentifier


def _ctx():
    return RequestContext(user=UserIdentifier.the_default_user("test_user"), role=Role.USER)


async def test_missing_or_empty_session_degrades_without_planner(monkeypatch):
    class MissingSession:
        async def load(self):
            raise FileNotFoundError("missing")

    service = SimpleNamespace(sessions=SimpleNamespace(session=lambda *_args: MissingSession()))
    called = False

    async def analyze(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("openviking.server.routers.search.IntentAnalyzer.analyze", analyze)
    queries, status = await _recall_query_plan(
        service=service,
        ctx=_ctx(),
        request=RecallRequest(query="hello", session_id="missing", query_expansion="auto"),
    )
    assert queries == ["hello"]
    assert status == "off"
    assert called is False


async def test_query_expansion_calls_planner_once_and_caps_queries(monkeypatch):
    class Session:
        async def load(self):
            return None

        async def get_context_for_search(self, query):
            assert query == "hello"
            return {
                "latest_archive_overview": "prior work",
                "current_messages": [SimpleNamespace(role="user", content="context")],
            }

    service = SimpleNamespace(sessions=SimpleNamespace(session=lambda *_args: Session()))
    calls = 0

    async def analyze(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            queries=[
                SimpleNamespace(query="planned one"),
                SimpleNamespace(query="planned two"),
                SimpleNamespace(query="planned three"),
            ]
        )

    monkeypatch.setattr("openviking.server.routers.search.IntentAnalyzer.analyze", analyze)
    queries, status = await _recall_query_plan(
        service=service,
        ctx=_ctx(),
        request=RecallRequest(query="hello", session_id="session", query_expansion="auto"),
    )
    assert calls == 1
    assert queries == ["hello", "planned one", "planned two"]
    assert status == "used"


async def test_query_expansion_timeout_degrades(monkeypatch):
    class Session:
        async def load(self):
            return None

        async def get_context_for_search(self, _query):
            return {"latest_archive_overview": "prior", "current_messages": []}

    async def analyze(*_args, **_kwargs):
        await asyncio.sleep(0.05)

    monkeypatch.setattr("openviking.server.routers.search.IntentAnalyzer.analyze", analyze)
    monkeypatch.setattr(
        "openviking.server.routers.search.get_openviking_config",
        lambda: SimpleNamespace(retrieval=SimpleNamespace(recall_intent_timeout_s=0.001)),
    )
    service = SimpleNamespace(sessions=SimpleNamespace(session=lambda *_args: Session()))
    queries, status = await _recall_query_plan(
        service=service,
        ctx=_ctx(),
        request=RecallRequest(query="hello", session_id="session", query_expansion="auto"),
    )
    assert queries == ["hello"]
    assert status == "failed"
