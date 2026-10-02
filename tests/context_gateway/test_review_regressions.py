"""Behavioral regressions from the gateway design review."""

import asyncio
import time

import pytest

from openviking_context_gateway.capture import CaptureWorker
from openviking_context_gateway.protocols import ResponseCapture, plugin_present
from openviking_context_gateway.storage import ManagementStore, SQLiteKernelStore


async def prepare(kernel, credential, policy, messages, **body):
    return await kernel.prepare(
        {"model": "model", "messages": messages, **body},
        "chat",
        {"x-openviking-session": "review"},
        credential,
        {"id": "upstream"},
        policy,
    )


def history(turns):
    return [
        message
        for i in range(turns)
        for message in (
            {"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": f"Answer {i}"},
        )
    ]


async def worker_for(store, encryption, credential, viking):
    management = ManagementStore(store.path.parent / "management.sqlite3", encryption)
    await management.initialize()
    await management.save("tenant", "keys", credential["id"], credential)
    return CaptureWorker(store, management, viking)


@pytest.mark.parametrize("turns", [0, 4])
async def test_screenshot_never_causes_emergency_wait(setup_kernel, credential, policy, turns):
    kernel, _, _, _ = setup_kernel
    policy.update(recall=False, context_window=128000, archive_wait_seconds=30)
    messages = [
        *history(turns),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this screenshot"},
                {"type": "image", "source": {"type": "base64", "data": "A" * (1024 * 1024)}},
            ],
        },
    ]
    body = {"model": "model", "messages": messages}
    result = await asyncio.wait_for(
        kernel.prepare(
            body,
            "anthropic",
            {"x-openviking-session": "image"},
            credential,
            {"id": "upstream"},
            policy,
        ),
        2,
    )
    assert result.body == body
    assert "degradation" not in result.metrics


async def test_emergency_usage_matches_model_and_configured_window(
    setup_kernel, credential, policy
):
    kernel, store, _, _ = setup_kernel
    policy.update(recall=False, archive_wait_seconds=0)
    upstream = {"id": "upstream", "context_windows": {"small": 2000, "large": 1000000}}
    headers = {"x-openviking-session": "model-window"}
    body = {"model": "small", "messages": [{"role": "user", "content": "hello"}]}
    first = await kernel.prepare(body, "chat", headers, credential, upstream, policy)
    await kernel.completed(first, credential, ResponseCapture("chat", usage={"input_tokens": 1950}))
    small = await kernel.prepare(body, "chat", headers, credential, upstream, policy)
    assert small.metrics["degradation"] == "archive_wait_timeout"
    large = await kernel.prepare(
        {**body, "model": "large"}, "chat", headers, credential, upstream, policy
    )
    assert "degradation" not in large.metrics
    unknown = await kernel.prepare(
        {**body, "model": "unknown"}, "chat", headers, credential, upstream, policy
    )
    assert "degradation" not in unknown.metrics
    assert not await store.read(first.scope, first.session, kinds=["replacement"])


async def test_system_tokens_do_not_trigger_commits(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, keep_recent_messages=0)

    async def write(*_):
        return {"pending_tokens": 50}

    viking.write = write
    worker = await worker_for(store, encryption, credential, viking)
    for turn in range(1, 5):
        p = await prepare(
            kernel,
            credential,
            policy,
            [*history(turn), {"role": "user", "content": "next"}],
            system="Tool instructions " * 10000,
        )
        await kernel.completed(
            p, credential, ResponseCapture("chat", usage={"input_tokens": 90000})
        )
        assert await worker.once()
    assert viking.commits == []


async def test_pending_overview_blocks_repeated_commit(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False, takeover_tokens=1, keep_recent_turns=1)
    viking.summary = ""
    worker = await worker_for(store, encryption, credential, viking)
    for turn in range(1, 4):
        await prepare(
            kernel, credential, policy, [*history(turn), {"role": "user", "content": "next"}]
        )
        assert await worker.once()
    assert len(viking.commits) == 1


async def test_capture_retry_blocks_later_turns_and_stops(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False)
    request = await prepare(
        kernel, credential, policy, [*history(2), {"role": "user", "content": "next"}]
    )
    worker = await worker_for(store, encryption, credential, viking)
    attempted = []

    async def fail(key, session, messages):
        attempted.append(messages[0]["parts"][0]["text"])
        raise RuntimeError("temporary outage")

    viking.write = fail
    assert await worker.once()
    assert not await worker.once()  # second turn may not overtake the delayed retry
    for _ in range(worker.MAX_ATTEMPTS - 1):
        with store.connect() as c:
            c.execute("UPDATE queue SET ready=0 WHERE attempts>0 AND failed=0")
        assert await worker.once()
    assert attempted == ["Question 0"] * worker.MAX_ATTEMPTS
    assert not await worker.once()
    next_request = await prepare(
        kernel, credential, policy, [*history(3), {"role": "user", "content": "more"}]
    )
    assert not next_request.capture_safe and next_request.body["messages"][-1]["content"] == "more"
    assert ("capture_failed", "") in await store.read(request.scope, request.session)


async def test_capture_retry_recovers_in_order(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False)
    await prepare(kernel, credential, policy, [*history(2), {"role": "user", "content": "next"}])
    worker = await worker_for(store, encryption, credential, viking)
    attempted, delivered = [], []

    async def write(key, session, messages):
        text = messages[0]["parts"][0]["text"]
        attempted.append(text)
        if len(attempted) == 1:
            raise RuntimeError("temporary outage")
        delivered.append(text)
        return {"pending_tokens": 20}

    viking.write = write
    assert await worker.once()
    with store.connect() as c:
        c.execute("UPDATE queue SET ready=0 WHERE attempts>0")
    assert await worker.once()
    assert await worker.once()
    assert delivered == ["Question 0", "Question 1"]


async def test_idle_capture_does_not_mix_a_regenerated_branch(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    policy.update(recall=False)
    p = await prepare(kernel, credential, policy, history(1)[:1])
    await kernel.completed(p, credential, ResponseCapture("chat", history(1)[1], complete=True))
    with store.connect() as c:
        c.execute("UPDATE queue SET ready=0")
    worker = await worker_for(store, encryption, credential, viking)
    await worker.once()
    messages = [
        history(1)[0],
        {"role": "assistant", "content": "Edited answer"},
        {"role": "user", "content": "next"},
    ]
    await prepare(kernel, credential, policy, messages)
    await worker.once()
    assert len(viking.writes) == 1
    assert ("capture_failed", "") in await store.read(p.scope, p.session)


async def test_only_new_turns_are_enqueued(setup_kernel, credential, policy, monkeypatch):
    kernel, store, _, _ = setup_kernel
    policy.update(recall=False)
    queued = []
    original = store.commit

    async def commit(*args, **kwargs):
        queued.extend(job["anchor"] for job in kwargs.get("jobs", []))
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "commit", commit)
    for turn in range(1, 7):
        await prepare(
            kernel, credential, policy, [*history(turn), {"role": "user", "content": "next"}]
        )
    assert len(queued) == 6 == len(set(queued))


async def test_hot_read_ignores_old_usage_and_write_records(
    setup_kernel, credential, policy, monkeypatch
):
    kernel, store, _, _ = setup_kernel
    p = await prepare(kernel, credential, policy, history(1)[:1])
    await store.put_many(
        p.scope,
        p.session,
        {
            (kind, str(i)): {"pending": i}
            for kind in ("usage", "written", "captured")
            for i in range(500)
        },
    )
    decoded = []
    original = store.decode

    def decode(value):
        decoded.append(1)
        return original(value)

    monkeypatch.setattr(store, "decode", decode)
    await prepare(kernel, credential, policy, history(1)[:1])
    assert len(decoded) < 12


async def test_conditional_budget_is_atomic_across_stores(setup_kernel, credential, policy):
    kernel, store, viking, encryption = setup_kernel
    from openviking_context_gateway.kernel import MemoryKernel

    other = MemoryKernel(SQLiteKernelStore(store.path, encryption), viking)
    policy.update(session_max_tokens=160, capture=False)

    async def recall(key, query, policy, exclude, budget):
        return {"entries": [{"uri": "viking://test/" + query, "text": "memory " * 8}]}

    viking.recall = recall
    await asyncio.gather(
        *(
            prepare(
                kernel if i % 2 else other,
                credential,
                policy,
                [{"role": "user", "content": f"Question {i}"}],
            )
            for i in range(12)
        )
    )
    from openviking_context_gateway.storage import digest

    records = await store.read(digest("tenant\0alice\0chat"), digest("review"), kinds=["injection"])
    assert sum(value["tokens"] for value in records.values()) <= 160


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "assistant", "content": "See <openviking-context> in the docs"}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "file", "file": {"name": "memory_notes.txt"}}],
                }
            ]
        },
        {"tools": [{"function": {"name": "ov_other_tool"}}]},
        {
            "messages": [
                {"role": "tool", "content": "<openviking-context>quoted</openviking-context>"}
            ]
        },
    ],
)
def test_unrelated_content_cannot_disable_memory(body):
    assert not plugin_present(body, {})


async def test_anonymous_greetings_cannot_share_capture_or_archive(
    setup_kernel, credential, policy
):
    kernel, store, _, _ = setup_kernel
    policy.update(gateway_tools=True)
    for answer in ("Project A", "Project B"):
        body = {
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "继续"},
            ]
        }
        request = await kernel.prepare(body, "chat", {}, credential, {"id": "upstream"}, policy)
        assert not request.capture_safe and not request.tools_active
        await kernel.completed(
            request,
            credential,
            ResponseCapture("chat", {"role": "assistant", "content": "OK"}, complete=True),
        )
    assert await store.claim() is None


async def test_management_get_uses_one_row_and_response_expiry(setup_kernel, monkeypatch):
    _, store, _, encryption = setup_kernel
    management = ManagementStore(store.path.parent / "management.sqlite3", encryption)
    await management.initialize()
    for i in range(30):
        await management.save("tenant", "responses", str(i), {"upstream_id": "test"}, ttl=100)
    calls = []
    decode = management.decode
    monkeypatch.setattr(management, "decode", lambda value: (calls.append(1), decode(value))[1])
    assert (await management.get("tenant", "responses", "5"))["upstream_id"] == "test"
    assert len(calls) == 1
    await management.save("tenant", "responses", "expired", {"upstream_id": "test"}, ttl=-1)
    assert await management.get("tenant", "responses", "expired") is None
    await management.expire_logs(time.time())
    with management.connect() as c:
        assert not c.execute("SELECT 1 FROM objects WHERE id='expired'").fetchone()


async def test_legacy_placeholder_never_replaces_history_without_real_overview(
    setup_kernel, credential, policy
):
    kernel, store, viking, _ = setup_kernel
    policy.update(recall=False, keep_recent_turns=1)
    messages = [*history(1), {"role": "user", "content": "next"}]
    request = await prepare(kernel, credential, policy, messages)
    anchor = request.chain[1]
    await store.put(
        request.scope,
        request.session,
        "replacement",
        anchor,
        {"text": "Earlier context is unavailable", "fallback": True},
    )
    await store.put(
        request.scope, request.session, "archive", anchor, {"archive_id": "archive_001"}
    )
    viking.summary = ""
    result = await prepare(kernel, credential, policy, messages)
    assert result.body["messages"] == messages
    viking.summary = "Verified archive of Question 0 and Answer 0"
    result = await prepare(kernel, credential, policy, messages)
    assert viking.summary in result.body["messages"][0]["content"]
    assert "unavailable" not in str(result.body)
