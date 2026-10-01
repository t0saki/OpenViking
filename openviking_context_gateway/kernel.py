# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Framework-independent recall, immutable replay and capture orchestration."""

import asyncio
import copy
import html
import logging
import time
from dataclasses import dataclass, field

import orjson

from .client import VikingError
from .models import Policy
from .protocols import (
    READ_VERSIONS,
    append_context,
    classify,
    clean_text,
    is_user,
    messages_of,
    plugin_present,
    prefix_chain,
    session_id,
    strip_thinking,
    text_content,
    unwrap_client,
)
from .storage import KernelStore, digest

logger = logging.getLogger(__name__)


def token_estimate(text):
    # Conservative, deterministic bound, independent of a model tokenizer.
    return (len(text.encode("utf-8")) + 2) // 3


def render_entries(entries, budget):
    lines, uris = [], []
    lead = "<openviking-context>\nReference material retrieved from the user's OpenViking memory.\n"
    end = "\n</openviking-context>"
    for entry in entries:
        uri, content = entry.get("uri", ""), entry.get("text", "")
        if not uri or uri in uris or not content:
            continue
        line = f"\n{html.escape(uri)}\n{html.escape(content, quote=False)}"
        if token_estimate(lead + "".join(lines) + line + end) > budget:
            continue
        lines.append(line)
        uris.append(uri)
    text = lead + "".join(lines) + end if lines else ""
    return {
        "text": text,
        "uris": uris,
        "tokens": token_estimate(text),
        "reason": "recalled" if text else "empty",
    }


@dataclass
class Prepared:
    body: dict
    original: dict
    protocol: str
    scope: str
    session: str
    chain: list[str]
    messages: list[dict]
    kind: str
    anchor: int
    root: dict
    records: dict
    disabled: bool = False
    metrics: dict = field(default_factory=dict)


class MemoryKernel:
    def __init__(self, store: KernelStore, viking):
        self.store, self.viking = store, viking

    async def prepare(self, body, protocol, headers, credential, upstream, policy, counting=False):
        messages = messages_of(body, protocol)
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("invalid message list")
        chains = [
            await asyncio.to_thread(prefix_chain, messages, version) for version in READ_VERSIONS
        ]
        chain = chains[0]
        # Identity is from /health at issuance, not caller-supplied tenant headers.
        scope = digest(credential["account"] + "\0" + credential["user_id"] + "\0" + protocol)
        sid = session_id(headers, messages, chain)
        records = await self.store.read(scope, sid, [a for c in chains for a in c if a])
        root = await self.store.put(
            scope,
            sid,
            "root",
            "",
            {
                "upstream_id": upstream["id"],
                "policy": policy,
                "credential_id": credential["id"],
                "ov_session": "context-gateway-" + digest(scope + sid)[:40],
            },
        )
        policy = Policy.model_validate(
            {k: v for k, v in root["policy"].items() if k not in {"id", "revision"}}
        )
        kind, anchor = classify(body, headers, messages, counting)
        disabled = ("disabled", "") in records
        if plugin_present(body, headers):
            await self.store.put(scope, sid, "disabled", "", {"reason": "plugin_present"})
            disabled = True
        result = copy.deepcopy(body)
        result_messages = copy.deepcopy(messages)
        result["input" if protocol == "responses" else "messages"] = result_messages
        prepared = Prepared(
            result,
            body,
            protocol,
            scope,
            sid,
            chain,
            messages,
            kind,
            anchor,
            root,
            records,
            disabled,
            {"kind": kind, "replay_hits": 0, "recall_count": 0, "recall_ms": 0},
        )
        # Always replay, even after plugin detection or on auxiliary requests.
        missing = False
        for index, _message in enumerate(messages):
            decision = next(
                (
                    records.get(("injection", c[index]))
                    for c in chains
                    if ("injection", c[index]) in records
                ),
                None,
            )
            if decision:
                if decision["text"]:
                    append_context(result_messages[index], decision["text"], protocol)
                prepared.metrics["replay_hits"] += 1
            elif ("sent", chain[index]) in records:
                missing = True
        if missing and protocol == "anthropic":
            result_messages[:] = strip_thinking(result_messages)
            prepared.metrics["degradation"] = "missing_injection_record"
        if disabled:
            prepared.metrics["degradation"] = "plugin_present"
            if result_messages == messages and isinstance(body.get("input"), str):
                result["input"] = body["input"]
            return prepared
        if kind == "user" and anchor >= 0 and ("injection", chain[anchor]) not in records:
            started = time.monotonic()
            existing = [v for (k, _), v in records.items() if k == "injection"]
            budget = min(
                policy.max_tokens,
                policy.session_max_tokens - sum(x.get("tokens", 0) for x in existing),
            )
            query = clean_text(unwrap_client(text_content(messages[anchor])))[
                : policy.query_max_chars
            ]
            decision = {"text": "", "uris": [], "tokens": 0, "reason": "disabled"}
            if policy.recall and budget >= 64 and len(query) >= 3:
                try:
                    response = await self.viking.recall(
                        credential["openviking_key"],
                        query,
                        policy,
                        [uri for x in existing for uri in x.get("uris", [])],
                        budget,
                    )
                    decision = render_entries(response.get("entries", []), budget)
                except (VikingError, asyncio.TimeoutError) as error:
                    decision["reason"] = getattr(error, "reason", "recall_timeout")
            decision["_budget"] = policy.session_max_tokens
            decision = await self.store.put(scope, sid, "injection", chain[anchor], decision)
            records["injection", chain[anchor]] = decision
            if decision["text"]:
                append_context(result_messages[anchor], decision["text"], protocol)
            prepared.metrics.update(
                recall_count=len(decision["uris"]),
                recall_ms=round((time.monotonic() - started) * 1000, 2),
                recall_reason=decision["reason"],
            )
        if policy.capture and kind == "user":
            await self._confirm_history(prepared, credential, policy)
        if policy.takeover and kind in {"user", "continuation", "count"}:
            await self._takeover(prepared, credential, policy)
        if (
            result_messages == messages
            and result.get("input") == result_messages
            and isinstance(body.get("input"), str)
        ):
            result["input"] = body["input"]
        return prepared

    async def _confirm_history(self, request, credential, policy):
        # At a new user turn, the preceding turns are the branch the client kept.
        await self.store.reconcile(request.scope, request.session, request.chain[: request.anchor])
        starts = [i for i, m in enumerate(request.messages) if is_user(m)]
        for start, end in zip(starts, starts[1:], strict=False):
            await self._queue(request, credential, policy, start, end, confirmed=True)

    async def _queue(self, request, credential, policy, start, end, confirmed):
        messages = capture_messages(request.messages[start:end], request.chain[start:end])
        if not messages:
            return
        endpoint = next((a for a in reversed(request.chain[start:end]) if a), "")
        await self.store.enqueue(
            request.scope,
            request.session,
            endpoint,
            {
                "messages": messages,
                "credential_id": credential["id"],
                "account": credential["account"],
                "ov_session": request.root["ov_session"],
                "policy": policy.model_dump(),
                "confirmed": confirmed,
                "boundary": endpoint,
                "user_anchor": request.chain[start],
            },
            time.time() if confirmed else time.time() + policy.idle_seconds,
        )

    async def completed(self, request, credential, response):
        for (kind, anchor), decision in request.records.items():
            if kind == "injection" and decision.get("text") and anchor in request.chain:
                await self.store.put(request.scope, request.session, "sent", anchor, {})
        if response.usage and request.chain:
            await self.store.put(
                request.scope,
                request.session,
                "usage",
                request.chain[-1],
                {**response.usage, "time": time.time()},
            )
        if (
            request.disabled
            or request.kind not in {"user", "continuation"}
            or not response.complete
            or not response.message
        ):
            return
        policy = Policy.model_validate(
            {k: v for k, v in request.root["policy"].items() if k not in {"id", "revision"}}
        )
        if not policy.capture or request.anchor < 0:
            return
        all_messages = [*request.messages, *(response.output_items or [response.message])]
        staged = copy.copy(request)
        staged.messages = all_messages
        staged.chain = prefix_chain(all_messages)
        await self._queue(
            staged, credential, policy, request.anchor, len(all_messages), confirmed=False
        )

    async def _takeover(self, request, credential, policy):
        # Capture workers commit complete user turns and publish their exact
        # prefix boundary. Never summarize unconfirmed data or split tool pairs.
        usage = max(
            (v for (kind, _), v in request.records.items() if kind == "usage"),
            key=lambda v: v.get("time", 0),
            default={},
        ).get("input_tokens", 0)
        urgent = token_estimate(orjson.dumps(request.body).decode()) >= policy.context_window * 0.9
        if (
            usage < policy.takeover_tokens
            and not urgent
            and not any(k == "replacement" for k, _ in request.records)
        ):
            return
        deadline = time.monotonic() + (policy.archive_wait_seconds if urgent else 0)
        while True:
            if await self._replace_archive(request, credential, policy):
                return
            if not urgent or time.monotonic() >= deadline:
                break
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
            request.records = await self.store.read(request.scope, request.session, request.chain)
        if urgent:
            starts = [i for i, message in enumerate(request.messages) if is_user(message)]
            if len(starts) > policy.keep_recent_turns:
                boundary = starts[-policy.keep_recent_turns] - 1
                anchor = request.chain[boundary]
                value = await self.store.put(
                    request.scope,
                    request.session,
                    "replacement",
                    anchor,
                    {
                        "text": "[OpenViking Session Context]\nEarlier context is unavailable; the recent conversation follows.",
                        "fallback": True,
                    },
                )
                self._apply_replacement(request, boundary, value)
                request.metrics["degradation"] = "archive_wait_timeout"

    async def _replace_archive(self, request, credential, policy):
        candidates = [
            (request.chain.index(anchor), anchor, value)
            for (kind, anchor), value in request.records.items()
            if kind in {"archive", "replacement"} and anchor in request.chain
        ]
        for index, anchor, archive in sorted(candidates, key=lambda item: item[0], reverse=True):
            remaining_turns = sum(is_user(m) for m in request.messages[index + 1 :])
            if (
                remaining_turns < policy.keep_recent_turns
                and ("replacement", anchor) not in request.records
            ):
                continue
            replacement = request.records.get(("replacement", anchor))
            if replacement is None:
                try:
                    summary = await self.viking.overview(
                        credential["openviking_key"],
                        request.root["ov_session"],
                        archive["archive_id"],
                    )
                except VikingError:
                    summary = ""
                if not summary:
                    request.metrics["degradation"] = "archive_pending"
                    continue
                replacement = await self.store.put(
                    request.scope,
                    request.session,
                    "replacement",
                    anchor,
                    {"text": "[OpenViking Session Context]\n" + summary},
                )
            self._apply_replacement(request, index, replacement)
            request.metrics["archive_replayed"] = anchor
            return True
        return False

    @staticmethod
    def _apply_replacement(request, index, replacement):
        key = "input" if request.protocol == "responses" else "messages"
        prefix = [
            m for m in request.body[key][: index + 1] if m.get("role") in {"system", "developer"}
        ]
        request.body[key] = [
            *prefix,
            {"role": "user", "content": replacement["text"]},
            *strip_thinking(request.body[key][index + 1 :]),
        ]


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


class CaptureWorker:
    def __init__(self, store, management, viking):
        self.store, self.management, self.viking = store, management, viking

    async def once(self):
        item = await self.store.claim(120)
        if item is None:
            return False
        try:
            payload = item["payload"]
            key = await self.management.get(payload["account"], "keys", payload["credential_id"])
            records = await self.store.read(item["scope"], item["session"], [])
            if not key or ("disabled", "") in records:
                await self.store.ack(item, True)
                return True
            policy = Policy.model_validate(payload["policy"])
            await self.viking.health(key["openviking_key"])
            if ("created", "") not in records:
                await self.viking.create_session(key["openviking_key"], payload["ov_session"])
                await self.store.put(item["scope"], item["session"], "created", "", {})
            pending = 0
            messages = payload["messages"]
            for offset in range(0, len(messages), 100):
                # Persist progress per batch; only the write/ack crash gap can duplicate.
                batch_key = item["id"] + ":" + str(offset)
                if ("written", batch_key) not in records:
                    result = await self.viking.write(
                        key["openviking_key"],
                        payload["ov_session"],
                        messages[offset : offset + 100],
                    )
                    pending = result.get("pending_tokens", 0)
                    await self.store.put(
                        item["scope"], item["session"], "written", batch_key, {"pending": pending}
                    )
                else:
                    pending = records["written", batch_key]["pending"]
            captured = await self.store.put(
                item["scope"],
                item["session"],
                "captured",
                item["anchor"],
                {"count": len(messages), "time": time.time()},
            )
            records["captured", item["anchor"]] = captured
            usage = max(
                (v for (kind, _), v in records.items() if kind == "usage"),
                key=lambda v: v.get("time", 0),
                default={},
            ).get("input_tokens", 0)
            takeover = policy.takeover and usage >= policy.takeover_tokens
            if pending >= policy.commit_tokens or not payload["confirmed"] or takeover:
                archived = {
                    a
                    for (kind, _), v in records.items()
                    if kind == "archive"
                    for a in v.get("covered", [])
                }
                turns = sorted(
                    (
                        (anchor, value)
                        for (kind, anchor), value in records.items()
                        if kind == "captured" and anchor not in archived
                    ),
                    key=lambda x: x[1]["time"],
                )
                keep, retained = 0, []
                if payload["confirmed"]:
                    for turn in reversed(turns):
                        if (takeover and len(retained) >= policy.keep_recent_turns - 1) or (
                            not takeover and keep >= policy.keep_recent_messages
                        ):
                            break
                        retained.append(turn)
                        keep += turn[1]["count"]
                to_archive = turns[: len(turns) - len(retained)]
                if to_archive:
                    commit_key = to_archive[-1][0]
                    result = await self.viking.commit(
                        key["openviking_key"], payload["ov_session"], keep
                    )
                    uri = result.get("archive_uri", "")
                    if uri:
                        await self.store.put(
                            item["scope"],
                            item["session"],
                            "archive",
                            commit_key,
                            {
                                "archive_id": uri.rstrip("/").split("/")[-1],
                                "covered": [a for a, _ in to_archive],
                            },
                        )
            await self.store.ack(item, True)
        except Exception:
            logger.exception("Context Gateway capture will be retried")
            await self.store.ack(item, False)
        return True
