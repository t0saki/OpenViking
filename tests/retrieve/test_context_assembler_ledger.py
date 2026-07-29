# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
from types import SimpleNamespace

from openviking.retrieve.context_assembler.ledger import LEDGER_FILENAME, RecallLedger
from openviking.retrieve.context_assembler.models import AssembledEntry
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier

SESSION_URI = "viking://user/test_user/sessions/s1"
LEDGER_URI = f"{SESSION_URI}/{LEDGER_FILENAME}"


def _ctx():
    return RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )


class _FakeVikingFS:
    def __init__(self, files=None, fail_read=False, fail_write=False):
        self.files = dict(files or {})
        self.fail_read = fail_read
        self.fail_write = fail_write
        self.writes = []

    async def read_file(self, uri, ctx=None):
        del ctx
        if self.fail_read:
            raise RuntimeError("agfs down")
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None, lock_handle=None):
        del ctx, lock_handle
        if self.fail_write:
            raise RuntimeError("read-only")
        self.files[uri] = content
        self.writes.append(uri)


def _session(turn=10):
    return SimpleNamespace(
        uri=SESSION_URI,
        meta=SimpleNamespace(total_message_count=turn),
        messages=[],
    )


def _entry(name, detail="abstract"):
    return AssembledEntry(
        uri=f"viking://user/test_user/memories/events/{name}",
        category="events",
        score=0.5,
        detail=detail,
        text="body",
    )


async def test_ledger_is_absent_when_dedup_is_off_or_session_missing():
    service = SimpleNamespace(viking_fs=_FakeVikingFS())
    assert (
        await RecallLedger.load(service=service, ctx=_ctx(), session=_session(), dedup_turns=0)
        is None
    )
    assert await RecallLedger.load(service=service, ctx=_ctx(), session=None, dedup_turns=5) is None


async def test_ledger_cools_recent_uris_only():
    payload = json.dumps(
        {
            "version": 1,
            "updated_turn": 10,
            "entries": {
                "viking://recent": {"turn": 8},
                "viking://old": {"turn": 2},
            },
        }
    )
    service = SimpleNamespace(viking_fs=_FakeVikingFS({LEDGER_URI: payload}))
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=10), dedup_turns=5
    )

    assert ledger is not None
    assert ledger.turn == 10
    assert ledger.cooled_uris() == {"viking://recent"}


async def test_future_turns_expire_after_archive_rotation():
    payload = json.dumps({"entries": {"viking://stale": {"turn": 99}}})
    service = SimpleNamespace(viking_fs=_FakeVikingFS({LEDGER_URI: payload}))
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=3), dedup_turns=5
    )

    assert ledger.cooled_uris() == set()


async def test_corrupt_ledger_starts_empty():
    service = SimpleNamespace(viking_fs=_FakeVikingFS({LEDGER_URI: "{not json"}))
    ledger = await RecallLedger.load(service=service, ctx=_ctx(), session=_session(), dedup_turns=5)

    assert ledger.status == "new"
    assert ledger.cooled_uris() == set()


async def test_malformed_records_are_skipped_and_overwritten_rather_than_raising():
    payload = json.dumps(
        {
            "entries": {
                "viking://text-turn": {"turn": "x"},
                "viking://null-turn": {"turn": None},
                "viking://not-a-record": "served",
                "viking://good": {"turn": 9},
            }
        }
    )
    fs = _FakeVikingFS({LEDGER_URI: payload})
    service = SimpleNamespace(viking_fs=fs)
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=10), dedup_turns=5
    )

    assert ledger.cooled_uris() == {"viking://good"}

    await ledger.record([_entry("a.md")])
    stored = json.loads(fs.files[LEDGER_URI])
    assert set(stored["entries"]) == {
        "viking://good",
        "viking://user/test_user/memories/events/a.md",
    }


async def test_uri_tier_entries_are_not_cooled():
    payload = json.dumps(
        {
            "entries": {
                "viking://pointer": {"turn": 9, "detail": "uri"},
                "viking://served": {"turn": 9, "detail": "abstract"},
            }
        }
    )
    service = SimpleNamespace(viking_fs=_FakeVikingFS({LEDGER_URI: payload}))
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=10), dedup_turns=5
    )

    assert ledger.cooled_uris() == {"viking://served"}


async def test_record_upserts_served_uris_at_current_turn():
    fs = _FakeVikingFS()
    service = SimpleNamespace(viking_fs=fs)
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=7), dedup_turns=3
    )
    await ledger.record([_entry("a.md", "full"), _entry("b.md")])

    stored = json.loads(fs.files[LEDGER_URI])
    assert stored["version"] == 1
    assert stored["updated_turn"] == 7
    assert stored["entries"]["viking://user/test_user/memories/events/a.md"] == {
        "turn": 7,
        "detail": "full",
    }


async def test_record_prunes_entries_beyond_the_cooldown_window():
    payload = json.dumps({"entries": {"viking://ancient": {"turn": 1}}})
    fs = _FakeVikingFS({LEDGER_URI: payload})
    service = SimpleNamespace(viking_fs=fs)
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=40), dedup_turns=5
    )
    await ledger.record([_entry("new.md")])

    stored = json.loads(fs.files[LEDGER_URI])
    assert "viking://ancient" not in stored["entries"]


async def test_record_prunes_records_left_ahead_of_the_clock():
    payload = json.dumps({"entries": {"viking://stale": {"turn": 99}}})
    fs = _FakeVikingFS({LEDGER_URI: payload})
    service = SimpleNamespace(viking_fs=fs)
    ledger = await RecallLedger.load(
        service=service, ctx=_ctx(), session=_session(turn=3), dedup_turns=5
    )
    await ledger.record([_entry("new.md")])

    stored = json.loads(fs.files[LEDGER_URI])
    assert "viking://stale" not in stored["entries"]


async def test_write_failure_is_contained():
    fs = _FakeVikingFS(fail_write=True)
    service = SimpleNamespace(viking_fs=fs)
    ledger = await RecallLedger.load(service=service, ctx=_ctx(), session=_session(), dedup_turns=5)
    await ledger.record([_entry("a.md")])

    assert ledger.status == "write_failed"


async def test_turn_falls_back_to_live_message_count():
    session = SimpleNamespace(
        uri=SESSION_URI,
        meta=SimpleNamespace(total_message_count=None),
        messages=[1, 2, 3, 4],
    )
    service = SimpleNamespace(viking_fs=_FakeVikingFS())
    ledger = await RecallLedger.load(service=service, ctx=_ctx(), session=session, dedup_turns=2)
    assert ledger.turn == 4
