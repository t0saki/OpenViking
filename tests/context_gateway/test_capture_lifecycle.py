"""Capture recovery and archive lifetime across requests, branches and restarts."""

import asyncio
import copy
import time

import pytest
from conftest import make_due, replay_records, update_capture
from test_review_regressions import history, prepare, worker_for

from openviking_context_gateway.capture import reset_capture
from openviking_context_gateway.capture_store import Document
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
    records = await replay_records(store, request, "replacement")
    assert not records


async def test_pending_archive_has_bounded_lifetime_and_shared_polling(
    setup_kernel, credential, policy
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    viking.summary = ""
    p = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    worker = await worker_for(store, encryption, credential, viking)
    await worker.once()
    state = (await store.capture.get(p.scope, p.session)).value
    await update_capture(store, p, archive={**state["archive"], "created": time.time() - 1000})
    calls = []

    async def pending(*args):
        calls.append(1)
        await asyncio.sleep(0.01)
        return "pending"

    viking.archive_state = pending
    await asyncio.gather(*(worker.once() for _ in range(12)))
    state = (await store.capture.get(p.scope, p.session)).value
    assert state["archive"]["status"] == "abandoned"
    assert len(calls) == 1
    assert not await worker.once()


@pytest.mark.parametrize("failure", [True, False])
async def test_existing_replacement_survives_capture_pause_or_plugin(
    setup_kernel, credential, policy, failure
):
    kernel, store, _, _ = setup_kernel
    policy.update(recall=False, keep_recent_turns=1)
    messages = [*history(2), {"role": "user", "content": "next"}]
    first = await prepare(kernel, credential, policy, messages)
    await store.replay.put(
        first.scope,
        first.session,
        "replacement",
        first.chain[1],
        {"text": "[OpenViking Session Context]\nverified summary"},
    )
    if failure:
        await update_capture(
            store, first, error={"reason": "outage", "attempts": 5, "retry_at": time.time() + 300}
        )
    else:
        await store.replay.put(
            first.scope, first.session, "disabled", "", {"reason": "plugin_present"}
        )
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
    archive = {
        "status": "pending",
        "boundary": p.chain[1],
        "archive_id": "archive_001",
        "created": time.time(),
    }
    await update_capture(store, p, archive=archive, tokens=policy["takeover_tokens"])
    waited = await prepare(kernel, credential, policy, messages)
    assert waited.metrics["degradation"] == "archive_wait_timeout"
    await update_capture(store, p, archive={**archive, "status": "failed"})
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
    assert revised.capture_target != first.capture_target
    assert revised.metrics["capture_status"] == "active"
    await worker.once()
    assert viking.write_sessions[-1] != viking.write_sessions[0]
    assert changed[0]["content"] in str(
        [
            m
            for m, sid in zip(viking.writes, viking.write_sessions, strict=True)
            if sid == revised.capture_target
        ]
    )


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
    route = await reset_capture(store.capture, first.scope, first.session, "manual_reset")
    released.set()
    await task
    assert not (await store.capture.get(first.scope, first.session)).value["archive"]
    revised = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    assert revised.capture_target == route["ov_session"]
    await worker.once()
    assert viking.write_sessions[0] != viking.write_sessions[1]
    assert (await store.capture.get(first.scope, first.session)).value["archive"]


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
    assert continued.capture_target == first.capture_target
    # A second observed identical conversation makes ownership ambiguous.
    await kernel.completed(independent, credential, ResponseCapture("chat", reply, complete=True))
    ambiguous = await anonymous([*opening, visible, {"role": "user", "content": "different"}])
    assert ambiguous.session not in {first.session, independent.session}
    assert ambiguous.capture_target != first.capture_target


@pytest.mark.parametrize(
    "marker, expected", [(".done", "completed"), (".failed.json", "failed"), ("", "pending")]
)
async def test_archive_state_uses_server_terminal_markers(marker, expected):
    client = VikingClient(None, "http://unused", "0.4.16")
    paths = []

    async def request(method, path, key, **kwargs):
        paths.append(path)
        if marker and path.split("&")[0].endswith(marker):
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


async def test_capture_mailboxes_are_independent_and_lease_safe(setup_kernel):
    _, store, _, _ = setup_kernel
    await store.capture.swap("scope", "one", Document(), {"anchor": "same"}, 0)
    leased = await store.capture.claim()
    await store.capture.swap("scope", "two", Document(), {"anchor": "same"}, 0)
    other = await store.capture.claim()
    assert other["session"] == "two"
    assert not await store.capture.claim()
    await store.capture.release({**leased, "owner": "stale"})
    assert not await store.capture.claim()
    for item in (leased, other):
        await store.capture.swap("scope", item["session"], item["document"], {"done": True}, None)
        await store.capture.release(item)
    assert not await store.capture.claim()


async def test_archive_publication_retry_does_not_commit_twice(
    setup_kernel, credential, policy, monkeypatch
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    request = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    worker = await worker_for(store, encryption, credential, viking)
    put = store.replay.put
    interrupted = False

    async def fail_publication(scope, session, kind, anchor, value):
        nonlocal interrupted
        if kind == "replacement" and not interrupted:
            interrupted = True
            raise RuntimeError("temporary publication outage")
        return await put(scope, session, kind, anchor, value)

    monkeypatch.setattr(store.replay, "put", fail_publication)
    await worker.once()  # commit
    await worker.once()  # observe and try to publish
    assert interrupted and len(viking.commits) == 1
    await make_due(store)
    await worker.once()
    assert len(viking.commits) == 1
    assert await replay_records(store, request, "replacement")


@pytest.mark.parametrize("partial_batch", [False, True])
async def test_delivery_reconciles_lost_ack_and_partial_batch(
    setup_kernel, credential, policy, partial_batch
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, commit_tokens=1000000, takeover=False)
    assistants = 205 if partial_batch else 1
    messages = [
        {"role": "user", "content": "start"},
        *[{"role": "assistant", "content": f"part {i}"} for i in range(assistants)],
        {"role": "user", "content": "next"},
    ]
    p = await prepare(kernel, credential, policy, messages)
    worker = await worker_for(store, encryption, credential, viking)
    write = viking.write
    calls = 0

    async def interrupted(*args):
        nonlocal calls
        calls += 1
        if partial_batch and calls == 2:
            raise VikingError("connection_lost")
        result = await write(*args)
        if not partial_batch and calls == 1:
            raise VikingError("response_lost_after_append")
        return result

    viking.write = interrupted
    await worker.once()
    await make_due(store)
    await worker.once()
    ids = [sid for batch in viking.writes for m in batch for sid in m["source_message_ids"]]
    assert len(ids) == len(set(ids)) == assistants + 1
    state = (await store.capture.get(p.scope, p.session)).value
    assert not state["error"] and not state["pending"]
    assert state["retained"] == [{"anchor": p.chain[-2], "count": assistants + 1}]


async def test_lost_commit_response_is_resolved_before_new_messages(
    setup_kernel, credential, policy
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    p = await prepare(
        kernel, credential, policy, [*history(1), {"role": "user", "content": "next"}]
    )
    worker = await worker_for(store, encryption, credential, viking)
    commit = viking.commit

    async def interrupted(*args):
        await commit(*args)
        raise VikingError("response_lost_after_commit")

    viking.commit = interrupted
    await worker.once()
    assert len(viking.commits) == 1
    old = (await store.capture.get(p.scope, p.session)).value
    assert old["archive"]["status"] == "committing"
    await prepare(kernel, credential, policy, [*history(2), {"role": "user", "content": "next"}])
    viking.commit = commit
    await make_due(store)
    await worker.once()
    assert len(viking.commits) == 2
    assert [[m["parts"][0]["text"] for m in batch] for batch in viking.archived.values()] == [
        ["Question 0", "Answer 0"],
        ["Question 1", "Answer 1"],
    ]


@pytest.mark.parametrize("edit", [False, True])
async def test_concurrent_request_reconciles_inflight_delivery(
    setup_kernel, credential, policy, edit
):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False)
    p = await prepare(kernel, credential, policy, history(1)[:1])
    await kernel.completed(p, credential, ResponseCapture("chat", history(1)[1], complete=True))
    await make_due(store)
    worker = await worker_for(store, encryption, credential, viking)
    entered, released = asyncio.Event(), asyncio.Event()
    write = viking.write

    async def slow(*args):
        entered.set()
        await released.wait()
        return await write(*args)

    viking.write = slow
    task = asyncio.create_task(worker.once())
    await asyncio.wait_for(entered.wait(), 2)
    messages = history(2)
    if edit:
        messages[1]["content"] = "edited reply"
    try:
        revised = await prepare(
            kernel, credential, policy, [*messages, {"role": "user", "content": "next"}]
        )
    finally:
        released.set()
        await task
    await worker.once()
    batches = [
        batch
        for batch, target in zip(viking.writes, viking.write_sessions, strict=True)
        if target == revised.capture_target
    ]
    assert [m["parts"][0]["text"] for batch in batches for m in batch] == [
        m["content"] for m in messages
    ]
    assert (revised.capture_target != p.capture_target) is edit
