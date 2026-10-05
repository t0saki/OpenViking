# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One capture document per client session; the mailbox is its only authority.

Requests replace the unconfirmed tail. The sole lease holder advances delivery
and archives. A changed confirmed prefix starts a new OpenViking session. Replay
records are independent: a failed capture never invalidates an old replacement.
"""

import asyncio
import copy
import logging
import time
import uuid

import async_timeout
import orjson

from .archives import TERMINAL, observe_archive
from .capture_store import Document, LeaseLost
from .models import Policy
from .protocols import clean_text, is_user, text_content, unwrap_client
from .records import RecordKind as K
from .tool_protocols import hidden_chain

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 5
RECOVERY_SECONDS = 300


def new_capture(reason=""):
    return {
        "ov_session": "context-gateway-" + uuid.uuid4().hex,
        "delivered": "",
        "pending": [],
        "retained": [],
        "tokens": 0,
        "archive": None,
        "error": {},
        "reason": reason,
    }


def ready_at(state):
    if state.get("error"):
        return state["error"]["retry_at"]
    archive = state.get("archive")
    if archive and archive["status"] == "committing":
        return archive.get("next_check", 0)
    times = [turn["ready"] for turn in state.get("pending", [])[:1]]
    if archive and archive["status"] not in TERMINAL:
        times.append(archive.get("next_check", 0))
    return min(times) if times else None


async def reset_capture(queue, scope, session, reason):
    while True:
        old = await queue.get(scope, session)
        value = new_capture(reason)
        if await queue.swap(scope, session, old, value, None):
            return value


class CapturePipeline:
    def __init__(self, queue):
        self.queue = queue

    @staticmethod
    def turns(messages, chain, start, confirmed, idle_seconds):
        starts = [i for i, m in enumerate(messages) if i >= start and is_user(m)]
        stops = starts[1:] + ([len(messages)] if not confirmed else [])
        turns = []
        for begin, end in zip(starts, stops, strict=False):
            captured = capture_messages(messages[begin:end], chain[begin:end])
            if captured:
                turns.append(
                    {
                        "anchor": chain[end - 1],
                        "messages": captured,
                        "confirmed": confirmed,
                        "ready": 0 if confirmed else time.time() + idle_seconds,
                    }
                )
        return turns

    async def confirm(self, request, credential, policy):
        old = request.capture
        positions = {a: i for i, a in enumerate(request.capture_chain) if a}
        user_anchor = request.capture_chain[request.anchor]
        while True:
            value = old.value or new_capture()
            endpoints = [
                value["delivered"],
                *[t["anchor"] for t in value["pending"]],
            ]
            if any(a and a not in positions for a in endpoints):
                value = new_capture("history_changed")
            value = copy.deepcopy(value)
            after = positions.get(value["delivered"], -1) + 1
            pending = await asyncio.to_thread(
                self.turns, request.messages, request.capture_chain, after, True, 0
            )
            # Reusing a prepared prompt does not generate another queue write.
            value.update(
                pending=pending,
                request_anchor=user_anchor,
                credential_id=credential["id"],
                account=credential["account"],
                protocol=request.protocol,
                policy=policy.model_dump(),
            )
            if value == old.value or await self.queue.swap(
                request.scope, request.session, old, value, ready_at(value)
            ):
                request.capture = Document(value, old.version + (value != old.value))
                request.capture_target = value["ov_session"]
                self.metrics(request, policy)
                return
            old = await self.queue.get(request.scope, request.session)

    @staticmethod
    def metrics(request, policy):
        value = request.capture.value
        error = value.get("error", {})
        request.metrics.update(
            capture_status=("paused" if error["attempts"] >= MAX_ATTEMPTS else "retrying")
            if error
            else ("active" if policy.capture else "disabled"),
            capture_reason=error.get("reason", value.get("reason", "")),
            capture_retry_at=error.get("retry_at"),
        )

    async def stage(self, request, response, policy):
        messages = [*request.messages, *(response.output_items or [response.message])]
        chain = await asyncio.to_thread(hidden_chain, messages, request.protocol)
        turns = await asyncio.to_thread(
            self.turns, messages, chain, request.anchor, False, policy.idle_seconds
        )
        if not turns:
            return
        while True:
            old = await self.queue.get(request.scope, request.session)
            if (
                old.value.get("ov_session") != request.capture_target
                or old.value.get("request_anchor") != request.capture_chain[request.anchor]
            ):
                return  # A newer request or explicit reset superseded this response.
            delivered = old.value.get("delivered", "")
            # Tool continuations replace the unconfirmed tail in place. Only
            # extending a turn already delivered after idle needs a new target.
            if delivered and delivered in chain and chain.index(delivered) >= request.anchor:
                value = {
                    **old.value,
                    **new_capture("continued_after_idle"),
                    "request_anchor": request.capture_chain[request.anchor],
                }
                prior = await asyncio.to_thread(
                    self.turns, request.messages, request.capture_chain, 0, True, 0
                )
                value["pending"] = [*prior, *turns]
            else:
                value = {
                    **old.value,
                    "pending": [t for t in old.value["pending"] if t["confirmed"]] + turns,
                }
            if await self.queue.swap(request.scope, request.session, old, value, ready_at(value)):
                return


class BranchChanged(Exception):
    pass


class CaptureWorker:
    def __init__(self, store, management, viking):
        self.queue, self.replay = store.capture, store.replay
        self.management, self.viking = management, viking

    async def save(self, item, previous, value):
        """Merge worker-owned progress into a concurrently updated request tail."""
        while True:
            if await self.queue.swap(
                item["scope"],
                item["session"],
                previous,
                value,
                ready_at(value),
                owner=item["owner"],
            ):
                return Document(value, previous.version + 1)
            fresh = await self.queue.get(item["scope"], item["session"])
            if fresh.value.get("ov_session") != value["ov_session"]:
                raise BranchChanged
            # The writer may only advance through the exact queued prefix it read.
            delivered = value["delivered"]
            pending = fresh.value["pending"]
            if delivered != previous.value["delivered"]:
                index = next(
                    (i for i, turn in enumerate(pending) if turn["anchor"] == delivered), None
                )
                if index is None:
                    # An idle reply was edited while its HTTP write was in flight.
                    await reset_capture(
                        self.queue,
                        item["scope"],
                        item["session"],
                        "history_changed_during_delivery",
                    )
                    raise BranchChanged
                pending = pending[index + 1 :]
            value = {
                **fresh.value,
                **{
                    key: value[key]
                    for key in ("delivered", "retained", "tokens", "archive", "error", "idle")
                    if key in value
                },
                "pending": pending,
            }
            previous = fresh

    async def once(self):
        item = await self.queue.claim()
        if item is None:
            return False
        try:
            async with async_timeout.timeout(100):
                await self.deliver(item)
        except (BranchChanged, LeaseLost):
            pass
        except Exception as error:
            old = await self.queue.get(item["scope"], item["session"])
            if old.value.get("ov_session") == item["document"].value.get("ov_session"):
                previous = old.value.get("error", {})
                attempts = previous.get("attempts", 0) + 1
                reason = getattr(error, "reason", "delivery_failed")
                failure = {
                    "attempts": attempts,
                    "reason": reason,
                    "retry_at": time.time()
                    + (RECOVERY_SECONDS if attempts >= MAX_ATTEMPTS else 10 * 2 ** (attempts - 1)),
                }
                await self.save(item, old, {**old.value, "error": failure})
                # Record status changes only: a paused capture retries forever.
                if attempts in (1, MAX_ATTEMPTS) or reason != previous.get("reason"):
                    status = "paused" if attempts >= MAX_ATTEMPTS else "retrying"
                    await self.log(item, old.value, status, reason)
                    logger.warning("Capture %s: %s (attempt %s)", item["session"], reason, attempts)
                else:
                    logger.debug("Capture %s: %s (attempt %s)", item["session"], reason, attempts)
        finally:
            await self.queue.release(item)
        return True

    async def log(self, item, state, status, reason):
        await self.management.log(
            state["account"],
            {
                "kind": "capture",
                "session": item["session"],
                "protocol": state["protocol"],
                "credential_id": state["credential_id"],
                "capture_status": status,
                "capture_reason": reason,
            },
        )

    async def deliver(self, item):
        doc = await self.queue.get(item["scope"], item["session"])
        state = copy.deepcopy(doc.value)
        if state.get("ov_session") != item["document"].value.get("ov_session"):
            return
        disabled = await self.replay.read(item["scope"], item["session"], [""])
        key = await self.management.get(state["account"], "keys", state["credential_id"])
        if (K.DISABLED, "") in disabled or not key:
            state["pending"] = []
            state["archive"] = None
            state["error"] = {}
            await self.save(item, doc, state)
            return
        token, session = key["openviking_key"], state["ov_session"]
        policy = Policy.model_validate(state["policy"])
        if state.get("error"):
            await self.viking.health(token)
        await self.viking.create_session(token, session)
        if state.get("archive") and state["archive"]["status"] == "committing":
            doc = await self.resolve_commit(item, doc, state, token, session)
            state = copy.deepcopy(doc.value)
            if state["archive"]["status"] == "committing":
                return
        # Source IDs are provenance, not a server idempotency key. Read the live
        # tail before retrying delivery, including after a lease-holder crash.
        remote = await self.viking.capture_status(token, session)
        seen = {s for m in remote["messages"] for s in m.get("source_message_ids", [])}
        while state["pending"] and state["pending"][0]["ready"] <= time.time():
            turn = state["pending"][0]
            messages = [m for m in turn["messages"] if not set(m["source_message_ids"]) <= seen]
            for offset in range(0, len(messages), 100):
                await self.viking.write(token, session, messages[offset : offset + 100])
            state["retained"].append({"anchor": turn["anchor"], "count": len(turn["messages"])})
            state["tokens"] += sum(
                (len(orjson.dumps(m["parts"])) + 2) // 3 for m in turn["messages"]
            )
            state["delivered"] = turn["anchor"]
            state["pending"] = state["pending"][1:]
            state["idle"] = not turn["confirmed"]
            doc = await self.save(item, doc, state)
            state = copy.deepcopy(doc.value)
        archive = state.get("archive")
        if archive and archive["status"] not in TERMINAL:
            state["archive"] = await observe_archive(self.viking, archive, token, session)
            doc = await self.save(item, doc, state)
        takeover = policy.takeover and state["tokens"] >= policy.takeover_tokens
        archive = state.get("archive")
        if archive and archive["status"] == "ready" and takeover:
            await self.replay.put(
                item["scope"],
                item["session"],
                K.REPLACEMENT,
                archive["boundary"],
                {"text": "[OpenViking Session Context]\n" + archive["summary"]},
            )
        if state.get("error"):
            state["error"] = {}
            doc = await self.save(item, doc, state)
            await self.log(item, state, "active", "delivery_recovered")
        if archive and archive["status"] not in TERMINAL:
            return
        remote = await self.viking.capture_status(token, session)
        threshold = policy.takeover_tokens if takeover else policy.commit_tokens
        if remote["pending_tokens"] < threshold and not state.get("idle"):
            return
        retained, keep = 0, 0
        if not state.get("idle"):
            for turn in reversed(state["retained"]):
                if (takeover and retained >= policy.keep_recent_turns - 1) or (
                    not takeover and keep >= policy.keep_recent_messages
                ):
                    break
                retained += 1
                keep += turn["count"]
        archived = state["retained"][: len(state["retained"]) - retained]
        if not archived:
            return
        # Persist the intent before the HTTP call. A retry first checks the
        # predicted server archive, so a lost response never repeats a commit.
        state["archive"] = {
            "status": "committing",
            "boundary": archived[-1]["anchor"],
            "keep": keep,
            "archive_uri": remote["next_archive_uri"],
            "created": time.time(),
        }
        doc = await self.save(item, doc, state)
        await self.resolve_commit(item, doc, state, token, session)

    async def resolve_commit(self, item, doc, state, token, session):
        archive = await self.viking.resolve_commit(token, session, state["archive"])
        state["archive"] = archive
        state["error"] = {}
        if archive.get("committed"):
            end = next(
                i
                for i, turn in enumerate(state["retained"])
                if turn["anchor"] == archive["boundary"]
            )
            state["retained"] = state["retained"][end + 1 :]
        return await self.save(item, doc, state)


def capture_messages(messages, chain):
    """Keep complete tool pairs, using the server's ToolPart representation."""
    results = {}
    for message in messages:
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message.get("content", "")
        if message.get("type") in {"function_call_output", "custom_tool_call_output"}:
            results[message.get("call_id")] = message.get("output", "")
        for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block.get("content", "")
    output = []
    for message, anchor in zip(messages, chain, strict=True):
        if message.get("role") not in {"user", "assistant"} and message.get("type") not in {
            "function_call",
            "custom_tool_call",
        }:
            continue
        text = clean_text(unwrap_client(text_content(message)))
        parts = [{"type": "text", "text": text}] if text else []
        calls = list(message.get("tool_calls") or [])
        if message.get("type") in {"function_call", "custom_tool_call"}:
            calls.append(message)
        calls.extend(
            b
            for b in message.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ) if isinstance(message.get("content"), list) else None
        for call in calls:
            identifier = call.get("call_id", call.get("id"))
            if identifier not in results:
                continue
            function = call.get("function", call)
            arguments = function.get("arguments", call.get("input", {}))
            if isinstance(arguments, str):
                try:
                    arguments = orjson.loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            result = results[identifier]
            parts.append(
                {
                    "type": "tool",
                    "tool_id": identifier,
                    "tool_name": function.get("name", "tool"),
                    "tool_input": arguments,
                    "tool_status": "completed",
                    "tool_output": clean_text(
                        result if isinstance(result, str) else orjson.dumps(result).decode()
                    ),
                }
            )
        if parts:
            output.append(
                {
                    "role": message.get("role", "assistant"),
                    "parts": parts,
                    "source_message_ids": ["context-gateway:" + anchor],
                }
            )
    return output
