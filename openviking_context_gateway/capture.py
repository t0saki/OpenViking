# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Branch reconciliation and ordered capture, independent of HTTP and SQL."""

import asyncio
import copy
import logging
import time
import uuid

import async_timeout
import orjson

from .archives import TERMINAL, refresh_archive
from .models import Policy
from .protocols import clean_text, is_user, prefix_chain, text_content, unwrap_client
from .records import RecordKind as K

logger = logging.getLogger(__name__)


class CaptureBranchChanged(Exception):
    """A queued turn belongs to history that has already changed."""


async def reset_capture(store, scope, session, reason, expected):
    """Switch capture destinations; immutable replay stays with the client session.

    In-flight workers can finish against the old OV session. They cannot publish
    archives into the new branch, and its independent queue never waits for them.
    """
    identifier = uuid.uuid4().hex
    route = {
        "session": session + ":" + identifier,
        "ov_session": "context-gateway-" + identifier,
        "reason": reason,
        "time": time.time(),
    }
    if await store.commit(
        scope, session, {(K.CAPTURE_ROUTE, ""): route}, expected={(K.CAPTURE_ROUTE, ""): expected}
    ):
        return route
    return None


class CapturePipeline:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def job(request, credential, policy, messages, chain, start, end, confirmed):
        captured = capture_messages(messages[start:end], chain[start:end])
        if not captured:
            return None
        anchor = next((a for a in reversed(chain[start:end]) if a), "")
        first_user = next((i for i, m in enumerate(messages) if is_user(m)), start)
        return {
            "anchor": anchor,
            "position": end,
            "ready": time.time() + (0 if confirmed else policy.idle_seconds),
            "value": {
                "messages": captured,
                "credential_id": credential["id"],
                "account": credential["account"],
                "owner_session": request.session,
                "protocol": request.protocol,
                "ov_session": request.capture_ov_session,
                "policy": policy.model_dump(),
                "confirmed": confirmed,
                "previous": next((chain[i] for i in range(start - 1, -1, -1) if chain[i]), "")
                if start > first_user
                else "",
                "user_anchor": chain[start],
            },
        }

    async def confirm(self, request, credential, policy):
        key = K.CAPTURE_CURSOR, ""
        positions = {a: i for i, a in enumerate(request.chain) if a}
        starts = [i for i, m in enumerate(request.messages) if is_user(m)]
        while True:
            routes = await self.store.read(request.scope, request.session, [""], [K.CAPTURE_ROUTE])
            route = routes.get((K.CAPTURE_ROUTE, ""))
            request.capture_session = route["session"] if route else request.session
            request.capture_ov_session = (
                route["ov_session"] if route else request.root["ov_session"]
            )
            records = await self.store.read(
                request.scope,
                request.capture_session,
                [""],
                [K.CAPTURE_CURSOR, K.CAPTURE_HEAD, K.CAPTURE_FAILED],
            )
            old = records.get(key)
            cursor = dict(old or {})
            head = records.get((K.CAPTURE_HEAD, ""))
            if head is None:
                legacy = await self.store.read(
                    request.scope, request.capture_session, [""], [K.CAPTURE_STATE]
                )
                head = {"anchor": legacy.get((K.CAPTURE_STATE, ""), {}).get("last_anchor", "")}
                await self.store.put(
                    request.scope, request.capture_session, K.CAPTURE_HEAD, "", head
                )
            failure = records.get((K.CAPTURE_FAILED, ""), {})
            endpoints = [cursor.get("confirmed"), head.get("anchor")]
            changed = any(a and a not in positions for a in endpoints)
            # Old releases wrote permanent failures. Rebuild from the current
            # history once rather than inheriting an unrecoverable queue.
            restart = changed or (failure and not failure.get("retry_at"))
            if restart:
                route = await reset_capture(
                    self.store,
                    request.scope,
                    request.session,
                    "history_changed" if changed else "capture_recovered",
                    route,
                )
                if route:
                    request.metrics["capture_reason"] = route["reason"]
                continue
            request.metrics.update(
                capture_status=("paused" if failure.get("attempts", 5) >= 5 else "retrying")
                if failure
                else "active",
                capture_reason=failure.get("reason", (route or {}).get("reason", "")),
                capture_retry_at=failure.get("retry_at"),
            )
            after = positions.get(cursor.get("confirmed"), -1)

            def build_jobs(after=after):
                return [
                    job
                    for start, end in zip(starts, starts[1:], strict=False)
                    if end - 1 > after
                    if (
                        job := self.job(
                            request,
                            credential,
                            policy,
                            request.messages,
                            request.chain,
                            start,
                            end,
                            True,
                        )
                    )
                ]

            jobs = await asyncio.to_thread(build_jobs)
            staged = cursor.pop("staged", None)
            if not jobs and not staged:
                return
            if jobs:
                cursor["confirmed"] = jobs[-1]["anchor"]
            if await self.store.commit(
                request.scope,
                request.capture_session,
                {key: cursor},
                expected={key: old},
                jobs=jobs,
                cancel=[staged["anchor"]] if staged else [],
            ):
                return

    async def stage(self, request, credential, policy, response):
        routes = await self.store.read(request.scope, request.session, [""], [K.CAPTURE_ROUTE])
        route = routes.get((K.CAPTURE_ROUTE, ""))
        if (route["session"] if route else request.session) != request.capture_session:
            return  # A response from the branch before a reset finished late.
        messages = [*request.messages, *(response.output_items or [response.message])]
        chain = await asyncio.to_thread(prefix_chain, messages)
        job = await asyncio.to_thread(
            self.job,
            request,
            credential,
            policy,
            messages,
            chain,
            request.anchor,
            len(messages),
            False,
        )
        if not job:
            return
        key = K.CAPTURE_CURSOR, ""
        while True:
            records = await self.store.read(
                request.scope, request.capture_session, [""], [K.CAPTURE_CURSOR]
            )
            old = records.get(key)
            cursor = dict(old or {})
            endpoint = cursor.get("confirmed", "")
            if endpoint and (endpoint not in chain or chain.index(endpoint) >= request.anchor):
                return
            staged = cursor.get("staged")
            cursor["staged"] = {"anchor": job["anchor"], "user_anchor": chain[request.anchor]}
            if await self.store.commit(
                request.scope,
                request.capture_session,
                {key: cursor},
                expected={key: old},
                jobs=[job],
                cancel=[staged["anchor"]] if staged else [],
            ):
                return


class CaptureWorker:
    MAX_ATTEMPTS = 5
    RECOVERY_SECONDS = 300

    def __init__(self, store, management, viking):
        self.store, self.management, self.viking = store, management, viking

    async def once(self):
        item = await self.store.claim(120)
        if item is None:
            return False
        owner = item["payload"].get("owner_session", item["session"])
        try:
            async with async_timeout.timeout(100):
                await self._deliver(item)
            await self.store.commit(item["scope"], item["session"], {(K.CAPTURE_FAILED, ""): None})
            await self.store.ack(item, True)
            if item.get("attempts"):
                await self.management.log(
                    item["payload"]["account"],
                    {
                        "kind": "capture",
                        "session": owner,
                        "protocol": item["payload"].get("protocol"),
                        "credential_id": item["payload"]["credential_id"],
                        "capture_status": "active",
                        "capture_reason": "delivery_recovered",
                    },
                )
        except Exception as error:
            attempts = item.get("attempts", 0) + 1
            branch_changed = isinstance(error, CaptureBranchChanged)
            paused = attempts >= self.MAX_ATTEMPTS
            retry_at = time.time() + (self.RECOVERY_SECONDS if paused else 10 * 2 ** (attempts - 1))
            reason = (
                "history_changed" if branch_changed else getattr(error, "reason", "delivery_failed")
            )
            status = {
                "reason": reason,
                "attempts": attempts,
                "retry_at": None if branch_changed else retry_at,
            }
            await self.store.commit(
                item["scope"], item["session"], {(K.CAPTURE_FAILED, ""): status}
            )
            await self.store.ack(item, False, retry_at=retry_at, failed=branch_changed)
            await self.management.log(
                item["payload"]["account"],
                {
                    "kind": "capture",
                    "protocol": item["payload"].get("protocol"),
                    "session": owner,
                    "credential_id": item["payload"]["credential_id"],
                    "capture_status": "paused" if paused or branch_changed else "retrying",
                    **status,
                    "capture_reason": reason,
                },
            )
            logger.warning("Context Gateway capture %s: %s (attempt %s)", owner, reason, attempts)
        return True

    async def _deliver(self, item):
        scope, session, payload = item["scope"], item["session"], item["payload"]
        owner = payload.get("owner_session", session)
        routing = await self.store.read(scope, owner, [""], [K.DISABLED, K.CAPTURE_ROUTE])
        route = routing.get((K.CAPTURE_ROUTE, ""))
        if (K.DISABLED, "") in routing or (route["session"] if route else owner) != session:
            return
        key = await self.management.get(payload["account"], "keys", payload["credential_id"])
        if not key:
            return
        records = await self.store.read(
            scope, session, ["", item["id"]], [K.CREATED, K.CAPTURE_STATE, K.WRITTEN]
        )
        policy = Policy.model_validate(payload["policy"])
        token, ov_session = key["openviking_key"], payload["ov_session"]
        # After the fast retry budget, this is a periodic recovery probe. Failed
        # health checks do not retry writes; the head job continues to hold order.
        await self.viking.health(token)
        if (K.CREATED, "") not in records:
            await self.viking.create_session(token, ov_session)
            await self.store.put(scope, session, K.CREATED, "", {})
        state = copy.deepcopy(records.get((K.CAPTURE_STATE, "")) or {})
        if not state:
            legacy = await self.store.read(scope, session, kinds=[K.CAPTURED, K.ARCHIVE])
            covered = {
                a for (k, _), v in legacy.items() if k == K.ARCHIVE for a in v.get("covered", [])
            }
            state["turns"] = [
                {"anchor": a, "count": v["count"]}
                for (k, a), v in sorted(legacy.items(), key=lambda x: x[1].get("time", 0))
                if k == K.CAPTURED and a not in covered
            ]
        if state.get("last_job") != item["id"] and "previous" in payload:
            if payload["previous"] != state.get(
                "last_anchor", payload["previous"] if state.get("turns") else ""
            ):
                raise CaptureBranchChanged
        progress = records.get((K.WRITTEN, item["id"]), {})
        pending = progress.get("pending", 0)
        messages = payload["messages"]
        offset_start = (
            len(messages) if state.get("last_job") == item["id"] else progress.get("offset", 0)
        )
        for offset in range(offset_start, len(messages), 100):
            result = await self.viking.write(token, ov_session, messages[offset : offset + 100])
            pending = result.get("pending_tokens", 0)
            await self.store.commit(
                scope,
                session,
                {
                    (K.WRITTEN, item["id"]): {
                        "offset": offset + len(messages[offset : offset + 100]),
                        "pending": pending,
                    }
                },
            )
        if state.get("last_job") != item["id"]:
            state.setdefault("turns", []).append({"anchor": item["anchor"], "count": len(messages)})
            state["dialogue_tokens"] = state.get("dialogue_tokens", 0) + sum(
                (len(orjson.dumps(m["parts"])) + 2) // 3 for m in messages
            )
            state.update(last_job=item["id"], last_anchor=item["anchor"])
            await self.store.commit(
                scope,
                session,
                {(K.CAPTURE_STATE, ""): state, (K.CAPTURE_HEAD, ""): {"anchor": item["anchor"]}},
            )
        takeover = policy.takeover and state.get("dialogue_tokens", 0) >= policy.takeover_tokens
        if takeover:
            await self.store.put(scope, owner, K.TAKEOVER, "", {})
        previous = state.get("pending_archive")
        if previous:
            anchor = previous.get("boundary")
            if not anchor:
                archives = await self.store.read(scope, owner, kinds=[K.ARCHIVE])
                anchor = next(
                    (
                        a
                        for (_, a), v in archives.items()
                        if v.get("archive_id") == previous["archive_id"]
                    ),
                    None,
                )
            if anchor:
                archives = await self.store.read(scope, owner, [anchor], [K.ARCHIVE])
                archive = archives.get((K.ARCHIVE, anchor))
                if archive is None:
                    # State precedes publication. Recover a store interruption
                    # without asking OpenViking to commit the same work twice.
                    archive = {k: v for k, v in previous.items() if k != "boundary"}
                    if not await self.store.commit(
                        scope,
                        owner,
                        {(K.ARCHIVE, anchor): archive},
                        expected={(K.CAPTURE_ROUTE, ""): route},
                    ):
                        return
                if archive:
                    archive = await refresh_archive(
                        self.store, self.viking, scope, owner, anchor, archive, token, ov_session
                    )
                    if archive.get("status") not in TERMINAL:
                        return
            state.pop("pending_archive")
            await self.store.commit(scope, session, {(K.CAPTURE_STATE, ""): state})
        threshold = policy.takeover_tokens if takeover else policy.commit_tokens
        if pending < threshold and payload["confirmed"]:
            return
        turns = state.get("turns", [])
        retained, keep = 0, 0
        if payload["confirmed"]:
            for turn in reversed(turns):
                if (takeover and retained >= policy.keep_recent_turns - 1) or (
                    not takeover and keep >= policy.keep_recent_messages
                ):
                    break
                retained += 1
                keep += turn["count"]
        archived = turns[: len(turns) - retained]
        if not archived:
            return
        result = await self.viking.commit(token, ov_session, keep)
        uri = result.get("archive_uri", "")
        if not uri:
            return
        archive = {
            "archive_id": uri.rstrip("/").split("/")[-1],
            "archive_uri": uri,
            "ov_session": ov_session,
            "takeover": takeover,
            "status": "pending",
            "created": time.time(),
        }
        boundary = archived[-1]["anchor"]
        state["pending_archive"] = {**archive, "boundary": boundary}
        state["turns"] = turns[len(archived) :]
        await self.store.commit(
            scope, session, {(K.CAPTURE_STATE, ""): state, (K.WRITTEN, item["id"]): None}
        )
        await self.store.commit(
            scope, owner, {(K.ARCHIVE, boundary): archive}, expected={(K.CAPTURE_ROUTE, ""): route}
        )


def capture_messages(messages, chain):
    """Keep complete tool pairs, using the server's ToolPart representation."""
    results = {}
    for message in messages:
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message.get("content", "")
        if message.get("type") == "function_call_output":
            results[message.get("call_id")] = message.get("output", "")
        for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block.get("content", "")
    output = []
    for message, anchor in zip(messages, chain, strict=True):
        if (
            message.get("role") not in {"user", "assistant"}
            and message.get("type") != "function_call"
        ):
            continue
        text = clean_text(unwrap_client(text_content(message)))
        parts = [{"type": "text", "text": text}] if text else []
        calls = list(message.get("tool_calls") or [])
        if message.get("type") == "function_call":
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
