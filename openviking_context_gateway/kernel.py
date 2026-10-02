# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Framework-independent recall, immutable replay and capture orchestration."""

import asyncio
import copy
import html
import logging
import time
from dataclasses import dataclass, field

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
from .records import REPLAY_KINDS, SESSION_KINDS
from .records import RecordKind as K
from .storage import KernelStore, digest
from .tool_catalog import (
    TOOL_VERSION,
    hidden_chain,
    replay_hidden,
    select_tools,
    tool_block_reason,
)

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
    body_chain: list[str] = field(default_factory=list)
    strip_replayed_thinking: bool = False
    capture_safe: bool = True
    tools_active: bool = False
    upstream: dict = field(default_factory=dict)


class MemoryKernel:
    def __init__(self, store: KernelStore, viking):
        self.store, self.viking = store, viking

    async def _records(self, scope, sid, anchors):
        state, replay, inherited = await asyncio.gather(
            self.store.read(scope, sid, [""], SESSION_KINDS),
            self.store.read(
                scope, sid, anchors, [k for k in REPLAY_KINDS if k not in {K.INJECTION, K.HIDDEN}]
            ),
            self.store.lookup(scope, anchors, [K.INJECTION, K.HIDDEN]),
        )
        return {**state, **replay, **inherited}

    @staticmethod
    def degraded_body(body, protocol):
        if protocol == "anthropic":
            return {**body, "messages": strip_thinking(body.get("messages", []))}
        return body

    async def enhance(self, body, protocol, *args):
        try:
            prepared = await self.prepare(body, protocol, *args)
            return prepared.body, prepared, prepared.metrics
        except Exception:
            logger.exception("Context Gateway preparation failed")
            return self.degraded_body(body, protocol), None, {"degradation": "memory_store_failure"}

    async def prepare(
        self, body, protocol, headers, credential, upstream, policy, counting=False, upstreams=()
    ):
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
        capture_safe = sid is not None
        # Anonymous requests can inherit prefix records, never a shared capture
        # session. Hidden tools require a stable session/tool contract too.
        sid = sid or "anonymous-" + (next((a for a in reversed(chain) if a), digest("empty")))
        needs_hidden_chain = protocol == "chat" and any(
            m.get("role") == "assistant"
            and (
                not isinstance(m.get("content"), str) or set(m) - {"role", "content", "tool_calls"}
            )
            for m in messages
        )
        body_chain = (
            await asyncio.to_thread(hidden_chain, messages) if needs_hidden_chain else chain
        )
        records = await self._records(scope, sid, list(dict.fromkeys(["", *chain, *body_chain])))
        plugin = plugin_present(body, headers)
        root = records.get((K.ROOT, ""))
        if root is None:
            root = await self.store.put(
                scope,
                sid,
                K.ROOT,
                "",
                {
                    "upstream_id": upstream["id"],
                    "tool_version": TOOL_VERSION,
                    "tools": select_tools(body, protocol, upstream, policy)
                    if capture_safe and not plugin
                    else [],
                    "policy": policy,
                    "credential_id": credential["id"],
                    "ov_session": "context-gateway-" + digest(scope + sid)[:40],
                },
            )
        pinned = next((u for u in upstreams if u["id"] == root["upstream_id"]), None)
        if pinned:
            upstream = pinned
        policy = Policy.model_validate(
            {k: v for k, v in root["policy"].items() if k not in {"id", "revision"}}
        )
        kind, anchor = classify(body, headers, messages, counting)
        disabled = (K.DISABLED, "") in records
        if plugin:
            await self.store.put(scope, sid, K.DISABLED, "", {"reason": "plugin_present"})
            disabled = True
        result = dict(body)
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
        prepared.body_chain = body_chain
        prepared.capture_safe = capture_safe and (K.CAPTURE_FAILED, "") not in records
        prepared.upstream = upstream
        if not capture_safe:
            prepared.metrics["capture_skip_reason"] = "missing_session_id"
        elif not prepared.capture_safe:
            prepared.metrics["capture_skip_reason"] = "capture_stopped"
        if upstream["id"] != root["upstream_id"]:
            result_messages[:] = strip_thinking(result_messages)
            prepared.metrics["degradation"] = "upstream_changed"
        tools = root.get("tools", [])
        if tools:
            reason = tool_block_reason(body, protocol, upstream)
            names = {t["function"]["name"] for t in tools}
            collision = any(
                t.get("function", {}).get("name") in names for t in body.get("tools", [])
            )
            if reason or collision or not capture_safe:
                prepared.metrics["tool_skip_reason"] = reason or "tool_name_collision"
            else:
                result["tools"] = [*body.get("tools", []), *copy.deepcopy(tools)]
                prepared.tools_active = True
                if disabled or kind not in {"user", "continuation"}:
                    result["tool_choice"] = "none"
        elif policy.gateway_tools:
            prepared.metrics["tool_skip_reason"] = (
                "missing_session_id"
                if not capture_safe
                else (
                    tool_block_reason(body, protocol, upstream)
                    or "tools_not_selected_at_session_start"
                )
            )
        # Always replay, even after plugin detection or on auxiliary requests.
        missing = False
        for index, _message in enumerate(messages):
            decision = next(
                (
                    records.get((K.INJECTION, c[index]))
                    for c in chains
                    if (K.INJECTION, c[index]) in records
                ),
                None,
            )
            if decision:
                if decision["text"]:
                    append_context(result_messages[index], decision["text"], protocol)
                prepared.metrics["replay_hits"] += 1
            elif (K.SENT, chain[index]) in records:
                missing = True
        if missing and protocol == "anthropic":
            result_messages[:] = strip_thinking(result_messages)
            prepared.metrics["degradation"] = "missing_injection_record"
        if disabled:
            prepared.metrics["degradation"] = "plugin_present"
            if result_messages == messages and isinstance(body.get("input"), str):
                result["input"] = body["input"]
            self._replay_tools(prepared)
            return prepared
        if kind == "user" and anchor >= 0 and (K.INJECTION, chain[anchor]) not in records:
            decision = await self._recall(prepared, credential, policy)
            if decision["text"]:
                append_context(result_messages[anchor], decision["text"], protocol)
        if policy.capture and kind == "user" and prepared.capture_safe:
            from .capture import CapturePipeline

            await CapturePipeline(self.store).confirm(prepared, credential, policy)
        if policy.takeover and prepared.capture_safe and kind in {"user", "continuation", "count"}:
            await self._takeover(prepared, credential, policy)
        if (
            result_messages == messages
            and result.get("input") == result_messages
            and isinstance(body.get("input"), str)
        ):
            result["input"] = body["input"]
        self._replay_tools(prepared)
        return prepared

    @staticmethod
    def _replay_tools(request):
        if request.protocol == "chat":
            request.body["messages"] = replay_hidden(
                request.body["messages"], request.body_chain, request.records
            )
            if request.strip_replayed_thinking:
                request.body["messages"] = strip_thinking(request.body["messages"])

    async def _recall(self, request, credential, policy):
        started = time.monotonic()
        anchor = request.chain[request.anchor]
        state_key = K.RECALL_STATE, ""
        existing = [v for (kind, _), v in request.records.items() if kind == K.INJECTION]
        state = request.records.get(state_key)
        query = clean_text(unwrap_client(text_content(request.messages[request.anchor])))[
            : policy.query_max_chars
        ]
        while True:
            current = state or {
                "tokens": sum(v.get("tokens", 0) for v in existing),
                "uris": list(dict.fromkeys(u for v in existing for u in v.get("uris", []))),
            }
            budget = min(policy.max_tokens, policy.session_max_tokens - current["tokens"])
            inherited = await self.store.lookup(request.scope, [anchor], [K.INJECTION])
            decision = inherited.get((K.INJECTION, anchor))
            if decision is None:
                decision = {"text": "", "uris": [], "tokens": 0, "reason": "disabled"}
                if policy.recall and budget >= 64 and len(query) >= 3:
                    try:
                        response = await self.viking.recall(
                            credential["openviking_key"], query, policy, current["uris"], budget
                        )
                        decision = render_entries(response.get("entries", []), budget)
                    except (VikingError, asyncio.TimeoutError) as error:
                        decision["reason"] = getattr(error, "reason", "recall_timeout")
            values = {
                (K.INJECTION, anchor): decision,
                state_key: {
                    "tokens": current["tokens"] + decision["tokens"],
                    "uris": list(dict.fromkeys([*current["uris"], *decision["uris"]])),
                },
            }
            if await self.store.commit(
                request.scope,
                request.session,
                values,
                expected={state_key: state, (K.INJECTION, anchor): None},
                shared=[(K.INJECTION, anchor)],
            ):
                break
            fresh = await self.store.read(
                request.scope, request.session, ["", anchor], [K.RECALL_STATE, K.INJECTION]
            )
            if (K.INJECTION, anchor) in fresh:
                decision = fresh[K.INJECTION, anchor]
                break
            state = fresh.get(state_key)
        request.records[K.INJECTION, anchor] = decision
        request.metrics.update(
            recall_count=len(decision["uris"]),
            recall_ms=round((time.monotonic() - started) * 1000, 2),
            recall_reason=decision["reason"],
        )
        return decision

    async def completed(self, request, credential, response):
        chain = set(request.chain)
        await self.store.put_many(
            request.scope,
            request.session,
            {
                (K.SENT, anchor): {}
                for (kind, anchor), decision in request.records.items()
                if kind == K.INJECTION
                and decision.get("text")
                and anchor in chain
                and (K.SENT, anchor) not in request.records
            },
        )
        if response.usage:
            model = request.body.get("model", "")
            await self.store.commit(
                request.scope,
                request.session,
                {
                    (K.USAGE, ""): {
                        **(response.context_usage or response.usage),
                        "model": model,
                        "upstream_id": request.upstream.get("id"),
                        "time": time.time(),
                    }
                },
            )
        if (
            request.disabled
            or not request.capture_safe
            or request.kind not in {"user", "continuation"}
            or not response.complete
            or not response.message
            or request.anchor < 0
        ):
            return
        policy = Policy.model_validate(
            {k: v for k, v in request.root["policy"].items() if k not in {"id", "revision"}}
        )
        if policy.capture:
            from .capture import CapturePipeline

            await CapturePipeline(self.store).stage(request, credential, policy, response)

    async def _takeover(self, request, credential, policy):
        # Capture workers commit complete user turns and publish their exact
        # prefix boundary. Never summarize unconfirmed data or split tool pairs.
        usage = request.records.get((K.USAGE, ""), {})
        model = request.body.get("model", "")
        resolved = request.upstream.get("aliases", {}).get(model, model)
        window = request.upstream.get("context_windows", {}).get(resolved, policy.context_window)
        urgent = bool(
            window
            and usage.get("model") in {model, resolved}
            and usage.get("upstream_id") == request.upstream.get("id")
            and usage.get("input_tokens", 0) + usage.get("output_tokens", 0) >= window * 0.9
        )
        deadline = time.monotonic() + (policy.archive_wait_seconds if urgent else 0)
        while True:
            if await self._replace_archive(request, credential, policy):
                return
            if not urgent or time.monotonic() >= deadline:
                break
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
            request.records.update(
                await self.store.read(
                    request.scope, request.session, request.chain, [K.ARCHIVE, K.REPLACEMENT]
                )
            )
        if urgent:
            request.metrics["degradation"] = "archive_wait_timeout"
        # Never publish a placeholder as an immutable replacement. Without a
        # real overview, forward the intact history for client/native compaction.

    async def _replace_archive(self, request, credential, policy):
        positions = {anchor: i for i, anchor in enumerate(request.chain) if anchor}
        candidates = [
            (positions[anchor], anchor, value)
            for (kind, anchor), value in request.records.items()
            if kind in {K.ARCHIVE, K.REPLACEMENT}
            and anchor in positions
            and not value.get("fallback")
        ]
        for index, anchor, archive in sorted(candidates, key=lambda item: item[0], reverse=True):
            if archive.get("takeover") is False and (K.TAKEOVER, "") not in request.records:
                continue
            remaining_turns = sum(is_user(m) for m in request.messages[index + 1 :])
            if (
                remaining_turns < policy.keep_recent_turns
                and (K.REPLACEMENT, anchor) not in request.records
            ):
                continue
            replacement = request.records.get((K.REPLACEMENT, anchor))
            invalid = replacement if replacement and replacement.get("fallback") else None
            if replacement is None or invalid:
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
                value = {"text": "[OpenViking Session Context]\n" + summary}
                if invalid:
                    # Repair the previous implementation's lossy placeholder only
                    # after obtaining an overview covering that exact boundary.
                    if not await self.store.commit(
                        request.scope,
                        request.session,
                        {(K.REPLACEMENT, anchor): value},
                        expected={(K.REPLACEMENT, anchor): invalid},
                    ):
                        continue
                    replacement = value
                else:
                    replacement = await self.store.put(
                        request.scope, request.session, K.REPLACEMENT, anchor, value
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
        request.strip_replayed_thinking = True
        request.body_chain = ["" for _ in prefix] + [""] + request.body_chain[index + 1 :]
        request.body[key] = [
            *prefix,
            {"role": "user", "content": replacement["text"]},
            *strip_thinking(request.body[key][index + 1 :]),
        ]


# Compatibility imports for existing integrations.
from .capture import CaptureWorker, capture_messages  # noqa: E402,F401
