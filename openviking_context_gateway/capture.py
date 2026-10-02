# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Capture reconciliation and ordered delivery, independent of HTTP and SQL."""

import copy
import logging
import time

import async_timeout
import orjson

from .client import VikingError
from .models import Policy
from .protocols import clean_text, is_user, prefix_chain, text_content, unwrap_client
from .records import RecordKind as K

logger = logging.getLogger(__name__)


class CaptureBranchChanged(Exception):
    """Delivered history no longer matches the client branch."""


class CapturePipeline:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def job(request, credential, policy, messages, chain, start, end, confirmed):
        captured = capture_messages(messages[start:end], chain[start:end])
        if not captured:
            return None
        anchor = next((a for a in reversed(chain[start:end]) if a), "")
        return {
            "anchor": anchor,
            "position": end,
            "ready": time.time() + (0 if confirmed else policy.idle_seconds),
            "value": {
                "messages": captured,
                "credential_id": credential["id"],
                "account": credential["account"],
                "ov_session": request.root["ov_session"],
                "policy": policy.model_dump(),
                "confirmed": confirmed,
                "boundary": anchor,
                "previous": next((chain[i] for i in range(start - 1, -1, -1) if chain[i]), "")
                if start > next((i for i, m in enumerate(messages) if is_user(m)), start)
                else "",
                "user_anchor": chain[start],
            },
        }

    async def confirm(self, request, credential, policy):
        key = K.CAPTURE_CURSOR, ""
        while True:
            records = await self.store.read(
                request.scope, request.session, [""], [K.CAPTURE_CURSOR]
            )
            old = records.get(key)
            cursor = dict(old or {})
            endpoint = cursor.get("confirmed", "")
            positions = {a: i for i, a in enumerate(request.chain) if a}
            if endpoint and endpoint not in positions:
                # Already delivered history cannot be retracted from an OV session.
                # A divergent client branch needs a new explicit session ID.
                request.capture_safe = False
                request.metrics["capture_skip_reason"] = "history_branch_changed"
                return
            after = positions.get(endpoint, -1)
            starts = [i for i, message in enumerate(request.messages) if is_user(message)]
            jobs = [
                self.job(
                    request, credential, policy, request.messages, request.chain, start, end, True
                )
                for start, end in zip(starts, starts[1:], strict=False)
                if end - 1 > after
            ]
            jobs = [job for job in jobs if job]
            staged = cursor.get("staged")
            if not jobs and not staged:
                return
            cursor.pop("staged", None)
            if jobs:
                cursor["confirmed"] = jobs[-1]["anchor"]
            if await self.store.commit(
                request.scope,
                request.session,
                {key: cursor},
                expected={key: old},
                jobs=jobs,
                cancel=[staged["anchor"]] if staged else [],
            ):
                return

    async def stage(self, request, credential, policy, response):
        messages = [*request.messages, *(response.output_items or [response.message])]
        chain = prefix_chain(messages)
        job = self.job(
            request, credential, policy, messages, chain, request.anchor, len(messages), False
        )
        if not job:
            return
        key = K.CAPTURE_CURSOR, ""
        while True:
            records = await self.store.read(
                request.scope, request.session, [""], [K.CAPTURE_CURSOR]
            )
            old = records.get(key)
            cursor = dict(old or {})
            endpoint = cursor.get("confirmed", "")
            # Ignore an older request completing after a newer turn was confirmed.
            if endpoint and (endpoint not in chain or chain.index(endpoint) >= request.anchor):
                return
            staged = cursor.get("staged")
            cursor["staged"] = {"anchor": job["anchor"], "user_anchor": chain[request.anchor]}
            if await self.store.commit(
                request.scope,
                request.session,
                {key: cursor},
                expected={key: old},
                jobs=[job],
                cancel=[staged["anchor"]] if staged else [],
            ):
                return


class CaptureWorker:
    MAX_ATTEMPTS = 5

    def __init__(self, store, management, viking):
        self.store, self.management, self.viking = store, management, viking

    async def once(self):
        item = await self.store.claim(120)
        if item is None:
            return False
        try:
            # Finish or cancel before the lease expires. Later jobs from this
            # session remain behind this one throughout retries and failures.
            async with async_timeout.timeout(100):
                await self._deliver(item)
            await self.store.ack(item, True)
        except Exception as error:
            attempts = item.get("attempts", 0) + 1
            failed = attempts >= self.MAX_ATTEMPTS or isinstance(error, CaptureBranchChanged)
            if failed:
                await self.store.put(
                    item["scope"],
                    item["session"],
                    K.CAPTURE_FAILED,
                    "",
                    {"reason": "delivery_failed", "attempts": attempts},
                )
            await self.store.ack(
                item,
                False,
                retry_at=time.time() + min(10 * 2 ** (attempts - 1), 300),
                failed=failed,
            )
            logger.warning(
                "Context Gateway capture %s after attempt %s",
                "stopped" if failed else "retry scheduled",
                attempts,
            )
        return True

    async def _deliver(self, item):
        scope, session, payload = item["scope"], item["session"], item["payload"]
        key = await self.management.get(payload["account"], "keys", payload["credential_id"])
        records = await self.store.read(
            scope, session, ["", item["id"]], [K.DISABLED, K.CREATED, K.CAPTURE_STATE, K.WRITTEN]
        )
        if not key or (K.DISABLED, "") in records:
            return
        policy = Policy.model_validate(payload["policy"])
        token, ov_session = key["openviking_key"], payload["ov_session"]
        await self.viking.health(token)
        if (K.CREATED, "") not in records:
            await self.viking.create_session(token, ov_session)
            await self.store.put(scope, session, K.CREATED, "", {})
        state = copy.deepcopy(records.get((K.CAPTURE_STATE, "")) or {})
        if not state:
            # One-time import of the previous ledger format on upgrade.
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
            # Only sanitized messages accepted by OV count toward takeover. The
            # provider's system prompt, tool schemas and image bytes never do.
            state["dialogue_tokens"] = state.get("dialogue_tokens", 0) + sum(
                (len(orjson.dumps(m["parts"])) + 2) // 3 for m in messages
            )
            state["last_job"] = item["id"]
            state["last_anchor"] = item["anchor"]
            await self.store.commit(scope, session, {(K.CAPTURE_STATE, ""): state})
        takeover = policy.takeover and state.get("dialogue_tokens", 0) >= policy.takeover_tokens
        if takeover:
            await self.store.put(scope, session, K.TAKEOVER, "", {})
        previous = state.get("pending_archive")
        if previous:
            try:
                summary = await self.viking.overview(token, ov_session, previous["archive_id"])
            except VikingError:
                summary = ""
            if not summary:
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
        archive = {"archive_id": uri.rstrip("/").split("/")[-1], "takeover": takeover}
        state["pending_archive"] = archive
        state["turns"] = turns[len(archived) :]
        await self.store.commit(
            scope,
            session,
            {
                (K.ARCHIVE, archived[-1]["anchor"]): archive,
                (K.CAPTURE_STATE, ""): state,
                (K.WRITTEN, item["id"]): None,
            },
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
