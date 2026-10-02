"""Capture recovery and archive lifetime across requests, branches and restarts."""

import asyncio
import copy
import time

import pytest
from test_review_regressions import history, prepare, worker_for

from openviking_context_gateway import archives
from openviking_context_gateway.capture import reset_capture
from openviking_context_gateway.client import VikingClient, VikingError
from openviking_context_gateway.protocols import ResponseCapture


@pytest.mark.parametrize("terminal", ["completed", "failed"])
async def test_terminal_archive_without_summary_releases_commits(
    setup_kernel, credential, policy, terminal
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    viking.summary = ""
    viking.archive_status = terminal
    worker = await worker_for(store, encryption, credential, viking)
    for turns in (1, 2, 3):
        request = await prepare(
            kernel, credential, policy, [*history(turns), {"role": "user", "content": "next"}]
        )
        await worker.once()
    assert len(viking.commits) == 3
    records = await store.read(request.scope, request.session, kinds=["replacement"])
    assert not records


async def test_pending_archive_has_bounded_lifetime_and_shared_polling(
    setup_kernel, credential, monkeypatch
):
    _, store, viking, _ = setup_kernel
    value = {"archive_id": "archive_001", "created": time.time() - 1000}
    await store.put("scope", "session", "archive", "anchor", value)
    viking.summary = ""
    calls = []

    async def pending(*args):
        calls.append(1)
        await asyncio.sleep(0.01)
        return "pending"

    viking.archive_state = pending
    results = await asyncio.gather(
        *(
            archives.refresh_archive(
                store, viking, "scope", "session", "anchor", value, "synthetic", "ov"
            )
            for _ in range(12)
        )
    )
    assert len(calls) == 1
    stored = (await store.read("scope", "session"))["archive", "anchor"]
    assert stored["status"] == "abandoned"
    await archives.refresh_archive(
        store, viking, "scope", "session", "anchor", stored, "synthetic", "ov"
    )
    assert len(calls) == 1
    assert all(not r.get("summary") for r in results)


@pytest.mark.parametrize("failure", [True, False])
async def test_existing_replacement_survives_capture_pause_or_plugin(
    setup_kernel, credential, policy, failure
):
    kernel, store, _, _ = setup_kernel
    policy.update(recall=False, keep_recent_turns=1)
    messages = [*history(2), {"role": "user", "content": "next"}]
    first = await prepare(kernel, credential, policy, messages)
    await store.put(
        first.scope,
        first.session,
        "replacement",
        first.chain[1],
        {"text": "[OpenViking Session Context]\nverified summary"},
    )
    if failure:
        await store.put(
            first.scope,
            first.session,
            "capture_failed",
            "",
            {"reason": "outage", "retry_at": time.time() + 300},
        )
    else:
        await store.put(first.scope, first.session, "disabled", "", {"reason": "plugin_present"})
    again = await prepare(kernel, credential, policy, messages)
    assert "verified summary" in again.body["messages"][0]["content"]
    assert messages[0] not in again.body["messages"]


@pytest.mark.parametrize("capture", [True, False])
async def test_high_usage_without_archive_returns_immediately(
    setup_kernel, credential, policy, capture
):
    kernel, _, _, _ = setup_kernel
    policy.update(recall=False, capture=capture, context_window=2000, archive_wait_seconds=30)
    messages = [*history(1), {"role": "user", "content": "next"}]
    p = await prepare(kernel, credential, policy, messages)
    await kernel.completed(p, credential, ResponseCapture("chat", usage={"input_tokens": 1900}))
    result = await asyncio.wait_for(prepare(kernel, credential, policy, messages), 0.5)
    assert result.body["messages"] == messages
    assert result.metrics.get("degradation") != "archive_wait_timeout"


async def test_emergency_wait_only_for_known_pending_archive(
    setup_kernel, credential, policy, monkeypatch
):
    kernel, store, viking, _ = setup_kernel
    policy.update(recall=False, context_window=2000, archive_wait_seconds=0.05, keep_recent_turns=1)
    viking.summary = ""
    messages = [*history(1), {"role": "user", "content": "next"}]
    p = await prepare(kernel, credential, policy, messages)
    await kernel.completed(p, credential, ResponseCapture("chat", usage={"input_tokens": 1900}))
    await store.put(p.scope, p.session, "archive", p.chain[1], {"archive_id": "archive_001"})
    waited = await prepare(kernel, credential, policy, messages)
    assert waited.metrics["degradation"] == "archive_wait_timeout"
    value = (await store.read(p.scope, p.session))[("archive", p.chain[1])]
    await store.commit(p.scope, p.session, {("archive", p.chain[1]): {**value, "status": "failed"}})
    failed = await asyncio.wait_for(prepare(kernel, credential, policy, messages), 0.5)
    assert "degradation" not in failed.metrics


@pytest.mark.parametrize("change", ["edit", "compact"])
async def test_changed_history_resyncs_to_new_session(setup_kernel, credential, policy, change):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False)
    messages = [*history(2), {"role": "user", "content": "next"}]
    first = await prepare(kernel, credential, policy, messages)
    worker = await worker_for(store, encryption, credential, viking)
    await worker.once()
    await worker.once()
    changed = copy.deepcopy(messages)
    if change == "edit":
        changed[1]["content"] = "a corrected answer"
    else:
        changed = [
            {"role": "user", "content": "compact summary"},
            {"role": "assistant", "content": "continue"},
            {"role": "user", "content": "next"},
        ]
    revised = await prepare(kernel, credential, policy, changed)
    assert revised.capture_session != first.capture_session
    assert revised.metrics["capture_status"] == "active"
    await worker.once()
    assert viking.write_sessions[-1] != viking.write_sessions[0]
    assert changed[0]["content"] in str(viking.writes[-1])


async def test_reset_during_delivery_cannot_publish_an_old_archive(
    setup_kernel, credential, policy
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    first = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    worker = await worker_for(store, encryption, credential, viking)
    entered, released = asyncio.Event(), asyncio.Event()
    original = viking.write

    async def slow(*args):
        entered.set()
        await released.wait()
        return await original(*args)

    viking.write = slow
    task = asyncio.create_task(worker.once())
    await entered.wait()
    route = await reset_capture(store, first.scope, first.session, "manual_reset", None)
    released.set()
    await task
    assert not await store.read(first.scope, first.session, kinds=["archive"])
    revised = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    assert revised.capture_session == route["session"]
    await worker.once()
    assert viking.write_sessions[0] != viking.write_sessions[1]
    assert await store.read(first.scope, first.session, kinds=["archive"])


async def test_anonymous_continuation_reuses_observed_reply_only(setup_kernel, credential, policy):
    kernel, store, _, _ = setup_kernel
    policy.update(recall=False)

    async def anonymous(messages):
        return await kernel.prepare(
            {"messages": messages}, "chat", {}, credential, {"id": "upstream"}, policy
        )

    opening = [{"role": "user", "content": "hello"}]
    first, independent = await asyncio.gather(anonymous(opening), anonymous(opening))
    assert first.session != independent.session
    reply = {"role": "assistant", "content": "hello back", "reasoning_content": "opaque"}
    visible = {"role": "assistant", "content": "hello back"}
    await kernel.completed(first, credential, ResponseCapture("chat", reply, complete=True))
    continued = await anonymous([*opening, visible, {"role": "user", "content": "next"}])
    assert continued.session == first.session
    # A second observed identical conversation makes ownership ambiguous.
    await kernel.completed(independent, credential, ResponseCapture("chat", reply, complete=True))
    ambiguous = await anonymous([*opening, visible, {"role": "user", "content": "different"}])
    assert ambiguous.session not in {first.session, independent.session}
    assert ambiguous.capture_ov_session != first.capture_ov_session


@pytest.mark.parametrize(
    "marker, expected", [(".done", "completed"), (".failed.json", "failed"), ("", "pending")]
)
async def test_archive_state_uses_server_terminal_markers(marker, expected):
    client = VikingClient(None, "http://unused", "0.4.16")
    paths = []

    async def request(method, path, key, **kwargs):
        paths.append(path)
        if marker and path.endswith(marker):
            return "{}"
        raise VikingError("openviking_http_404", 404)

    client.request = request
    assert (
        await client.archive_state(
            "synthetic", "session", "archive_001", "viking://user/a/sessions/s/history/archive_001"
        )
        == expected
    )
    assert all("/api/v1/content/read?uri=viking%3A" in path for path in paths)


def test_missing_optional_dependencies_explain_install(monkeypatch):
    from openviking_context_gateway import cli

    monkeypatch.setattr("sys.argv", ["openviking-context-gateway"])
    monkeypatch.setattr(cli, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match=r"openviking\[context-gateway\]"):
        cli.main()


async def test_queue_upgrade_preserves_lease_and_separates_sessions(setup_kernel):
    _, store, _, _ = setup_kernel
    await store.enqueue("scope", "one", "same", {"messages": []}, 0)
    leased = await store.claim()
    with store.connect() as c:
        c.executescript("""
            ALTER TABLE queue RENAME TO current_queue;
            CREATE TABLE queue (
                id TEXT PRIMARY KEY, scope TEXT, session TEXT, anchor TEXT,
                value BLOB NOT NULL, ready REAL, lease REAL DEFAULT 0, owner TEXT,
                position INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0, UNIQUE(scope,anchor));
            INSERT INTO queue SELECT * FROM current_queue;
            DROP TABLE current_queue;
            PRAGMA user_version=3;
        """)
    await store.initialize()
    await store.enqueue("scope", "two", "same", {"messages": []}, 0)
    other = await store.claim()
    assert other["session"] == "two"
    await store.ack(leased, True)
    await store.ack(other, True)
    assert await store.claim() is None


async def test_archive_publication_retry_does_not_commit_twice(
    setup_kernel, credential, policy, monkeypatch
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    request = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    worker = await worker_for(store, encryption, credential, viking)
    commit = store.commit
    interrupted = False

    async def fail_publication(scope, session, values, **kwargs):
        nonlocal interrupted
        if any(k == "archive" for k, _ in values) and not interrupted:
            interrupted = True
            raise RuntimeError("temporary publication outage")
        return await commit(scope, session, values, **kwargs)

    monkeypatch.setattr(store, "commit", fail_publication)
    await worker.once()
    assert interrupted and len(viking.commits) == 1
    with store.connect() as c:
        c.execute("UPDATE queue SET ready=0 WHERE attempts>0")
    await worker.once()
    assert len(viking.commits) == 1
    assert await store.read(request.scope, request.session, kinds=["archive"])
