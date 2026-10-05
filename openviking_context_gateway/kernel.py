# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Framework-independent recall, immutable replay and capture orchestration."""

import asyncio
import html
import logging
import time
import uuid
from dataclasses import dataclass, field

from .archives import POLL_SECONDS
from .capture import CapturePipeline
from .capture_store import Document
from .client import VikingError
from .models import Policy
from .protocols import (
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
from .records import RecordKind as K
from .state_store import get_state
from .storage import KernelStore, digest
from .tool_catalog import (
    TOOL_VERSION,
    disable_tools,
    hidden_chain,
    replay_hidden,
    select_tools,
    tool_block_reason,
    wire_tools,
)
from .vendors import parameter_fingerprint

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
    capture_chain: list[str] = field(default_factory=list)
    strip_replayed_thinking: bool = False
    anonymous: bool = False
    capture: Document = field(default_factory=Document)
    observation: Document = field(default_factory=Document)
    capture_target: str = ""
    tools_active: bool = False
    upstream: dict = field(default_factory=dict)


class MemoryKernel:
    def __init__(self, store: KernelStore, viking):
        self.store, self.viking = store, viking

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

    async def identity(self, messages, protocol, headers, credential, tools):
        sid = session_id(headers)
        needs_hidden = protocol == "chat" and (sid is None or tools)

        def chains():
            chain = prefix_chain(messages)
            return chain, hidden_chain(messages) if needs_hidden else chain

        # Small prompts need no executor; large payloads must not block the loop.
        if len(messages) > 32 or any(len(m.get("content") or "") > 8192 for m in messages):
            chain, body_chain = await asyncio.to_thread(chains)
        else:
            chain, body_chain = chains()
        scope = digest(credential["account"] + "\0" + credential["user_id"] + "\0" + protocol)
        anonymous = sid is None
        if anonymous:
            endpoints = [
                a
                for a, m in zip(body_chain, messages, strict=True)
                if a
                and (
                    m.get("role") == "assistant"
                    or protocol == "responses"
                    and m.get("type") in {"reasoning", "function_call", "custom_tool_call"}
                )
            ]
            owners = (
                await self.store.state.read(scope, ["prefix:" + a for a in endpoints])
                if endpoints
                else {}
            )
            for anchor in reversed(endpoints):
                match = owners.get("prefix:" + anchor)
                if match:
                    sid = match.value["owners"][0] if len(match.value["owners"]) == 1 else None
                    break
            sid = sid or "anonymous-" + uuid.uuid4().hex
        return scope, sid, anonymous, chain, body_chain

    async def prepare(
        self, body, protocol, headers, credential, upstream, policy, counting=False, upstreams=()
    ):
        messages = messages_of(body, protocol)
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("invalid message list")
        scope, sid, anonymous, chain, body_chain = await self.identity(
            messages,
            protocol,
            headers,
            credential,
            (policy.get("gateway_tools", False) or policy.get("capture", True)),
        )
        records, observations, capture = await self.store.load(
            scope, sid, list(dict.fromkeys(["", *chain, *body_chain]))
        )
        plugin = plugin_present(body, headers)
        root = await self.session_root(
            records, scope, sid, body, protocol, upstream, policy, credential, plugin
        )
        if (
            protocol == "chat"
            and (root["tools"] or root["policy"]["capture"])
            and body_chain is chain
        ):
            body_chain = await asyncio.to_thread(hidden_chain, messages)
            if body_chain != chain:
                records.update(await self.store.replay.read(scope, sid, body_chain))
        upstream = next((u for u in upstreams if u["id"] == root["upstream_id"]), upstream)
        policy = Policy.model_validate(
            {k: v for k, v in root["policy"].items() if k not in {"id", "revision"}}
        )
        kind, anchor = classify(body, headers, messages, counting)
        disabled = (K.DISABLED, "") in records
        if plugin and not disabled:
            await self.store.replay.put(scope, sid, K.DISABLED, "", {"reason": "plugin_present"})
            disabled = True
        field = "input" if protocol == "responses" else "messages"
        result = {**body, field: [dict(m) for m in messages]}
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
            disabled=disabled,
            anonymous=anonymous,
            body_chain=body_chain,
            capture_chain=body_chain,
            capture=capture,
            observation=observations.get(sid, Document()),
            capture_target=capture.value.get("ov_session", ""),
            upstream=upstream,
            metrics={
                "kind": kind,
                "replay_hits": 0,
                "recall_count": 0,
                "recall_ms": 0,
                "session": sid,
                "protocol": protocol,
                "credential_id": credential["id"],
            },
        )
        CapturePipeline.metrics(prepared, policy)
        self.configure_tools(prepared, policy)
        self.replay(prepared)
        if not disabled:
            if kind == "user" and anchor >= 0 and (K.INJECTION, chain[anchor]) not in records:
                await self.recall(prepared, credential, policy)
            if policy.capture and kind == "user" and anchor >= 0:
                await CapturePipeline(self.store.capture).confirm(prepared, credential, policy)
        await self.takeover(prepared, policy)
        if prepared.tools_active:
            result[field] = replay_hidden(result[field], prepared.body_chain, records, protocol)
            if prepared.strip_replayed_thinking:
                result[field] = strip_thinking(result[field])
        if isinstance(body.get("input"), str) and result.get("input") == messages:
            result["input"] = body["input"]
        return prepared

    async def session_root(
        self, records, scope, sid, body, protocol, upstream, policy, credential, plugin
    ):
        root = records.get((K.ROOT, ""))
        if root is None:
            root = await self.store.replay.put(
                scope,
                sid,
                K.ROOT,
                "",
                {
                    "upstream_id": upstream["id"],
                    "tool_version": TOOL_VERSION,
                    "tools": select_tools(body, protocol, upstream, policy) if not plugin else [],
                    "policy": policy,
                    "credential_id": credential["id"],
                    "vendor": {
                        "prompt_cache_key": body.get("prompt_cache_key")
                        or "ovcg-" + digest(scope + sid)[:40],
                        "parameters": parameter_fingerprint(body),
                    },
                },
            )
        return root

    @staticmethod
    def configure_tools(request, policy):
        tools = request.root["tools"]
        if tools:
            reason = tool_block_reason(request.original, request.protocol, request.upstream)
            names = {t["function"]["name"] for t in tools}
            collision = any(
                t.get("function", t).get("name") in names for t in request.original.get("tools", [])
            )
            if reason or collision:
                request.metrics["tool_skip_reason"] = reason or "tool_name_collision"
                if request.protocol == "anthropic" and any(
                    (K.HIDDEN, anchor) in request.records for anchor in request.body_chain
                ):
                    request.body["messages"] = strip_thinking(request.body["messages"])
                    request.metrics["degradation"] = "hidden_tool_history_unavailable"
            else:
                request.body["tools"] = [
                    *request.original.get("tools", []),
                    *wire_tools(tools, request.protocol),
                ]
                request.tools_active = True
                if request.protocol == "responses":
                    request.body["include"] = list(
                        dict.fromkeys(
                            [*request.original.get("include", []), "reasoning.encrypted_content"]
                        )
                    )
                if request.kind != "count" and (
                    request.disabled or request.kind not in {"user", "continuation"}
                ):
                    disable_tools(request.body, request.protocol)
        elif policy.gateway_tools:
            request.metrics["tool_skip_reason"] = (
                tool_block_reason(request.original, request.protocol, request.upstream)
                or "tools_not_selected_at_session_start"
            )

    @staticmethod
    def replay(request):
        field = "input" if request.protocol == "responses" else "messages"
        messages = request.body[field]
        missing = False
        sent = set(request.observation.value.get("sent", []))
        for index, anchor in enumerate(request.chain):
            decision = request.records.get((K.INJECTION, anchor))
            if decision is not None:
                if decision["text"]:
                    append_context(messages[index], decision["text"], request.protocol)
                request.metrics["replay_hits"] += 1
            elif anchor in sent:
                missing = True
        reason = ""
        if request.upstream["id"] != request.root["upstream_id"]:
            reason = "upstream_changed"
        elif missing and request.protocol == "anthropic":
            reason = "missing_injection_record"
        if reason:
            request.body[field] = strip_thinking(messages)
            request.strip_replayed_thinking = True
            request.metrics["degradation"] = reason
        if request.disabled:
            request.metrics["degradation"] = "plugin_present"

    async def recall(self, request, credential, policy):
        started = time.monotonic()
        existing = [v for (kind, _), v in request.records.items() if kind == K.INJECTION]
        used = sum(v.get("tokens", 0) for v in existing)
        exclude = list(dict.fromkeys(u for v in existing for u in v.get("uris", [])))
        budget = min(policy.max_tokens, policy.session_max_tokens - used)
        query = clean_text(unwrap_client(text_content(request.messages[request.anchor])))[
            : policy.query_max_chars
        ]
        decision = {"text": "", "uris": [], "tokens": 0, "reason": "disabled"}
        reserved = False
        if policy.recall and budget >= 64 and len(query) >= 3:
            budget = await self.reserve_recall(request, policy, used)
            reserved = budget > 0
        if reserved:
            try:
                response = await self.viking.recall(
                    credential["openviking_key"], query, policy, exclude, budget
                )
                decision = render_entries(response.get("entries", []), budget)
            except (VikingError, asyncio.TimeoutError) as error:
                decision["reason"] = getattr(error, "reason", "recall_timeout")
        anchor = request.chain[request.anchor]
        decision = await self.store.replay.put(
            request.scope, request.session, K.INJECTION, anchor, decision
        )
        if reserved:
            await self.settle_recall(request, anchor, decision["tokens"])
        request.records[K.INJECTION, anchor] = decision
        if decision["text"]:
            field = "input" if request.protocol == "responses" else "messages"
            append_context(request.body[field][request.anchor], decision["text"], request.protocol)
        request.metrics.update(
            recall_count=len(decision["uris"]),
            recall_ms=round((time.monotonic() - started) * 1000, 2),
            recall_reason=decision["reason"],
        )

    async def reserve_recall(self, request, policy, used):
        """Reserve a bounded allowance before recall; only this document uses CAS.

        A crash can leave a conservative reservation, never overspend the cap.
        Competing requests for the same anchor share its reservation and the
        immutable decision; settling it twice cannot refund twice.
        """
        anchor, old = request.chain[request.anchor], request.observation
        while True:
            ledger = old.value.get("recall", {"spent": used, "pending": {}})
            if anchor in ledger["pending"]:
                request.observation = old
                return ledger["pending"][anchor]
            budget = min(policy.max_tokens, policy.session_max_tokens - ledger["spent"])
            if budget < 64:
                return 0
            value = {
                **old.value,
                "recall": {
                    "spent": ledger["spent"] + budget,
                    "pending": {**ledger["pending"], anchor: budget},
                },
            }
            if await self.store.state.swap(request.scope, request.session, old, value):
                request.observation = Document(value, old.version + 1)
                return budget
            old = await get_state(self.store.state, request.scope, request.session)

    async def settle_recall(self, request, anchor, tokens):
        old = request.observation
        while anchor in old.value.get("recall", {}).get("pending", {}):
            ledger = old.value["recall"]
            pending = dict(ledger["pending"])
            reserved = pending.pop(anchor)
            value = {
                **old.value,
                "recall": {"spent": ledger["spent"] - reserved + tokens, "pending": pending},
            }
            if await self.store.state.swap(request.scope, request.session, old, value):
                request.observation = Document(value, old.version + 1)
                return
            old = await get_state(self.store.state, request.scope, request.session)

    async def completed(self, request, credential, response):
        chain = set(request.chain)
        sent = [
            a
            for (k, a), v in request.records.items()
            if k == K.INJECTION and v.get("text") and a in chain
        ]
        usage = {
            **(response.context_usage or response.usage or {}),
            "model": request.body.get("model", ""),
            "upstream_id": request.upstream.get("id"),
            "time": time.time(),
        }
        old = request.observation
        while True:
            value = dict(old.value)
            if response.usage and usage["time"] >= old.value.get("usage", {}).get("time", 0):
                value["usage"] = usage
            if sent:
                value["sent"] = list(dict.fromkeys([*old.value.get("sent", []), *sent]))
            if value == old.value:
                break
            if await self.store.state.swap(request.scope, request.session, old, value):
                break
            old = await get_state(self.store.state, request.scope, request.session)
        if (
            request.anonymous
            and response.complete
            and response.message
            and request.kind in {"user", "continuation"}
        ):
            returned = [*request.messages, *(response.output_items or [response.message])]
            endpoint = (
                await asyncio.to_thread(
                    hidden_chain if request.protocol == "chat" else prefix_chain, returned
                )
            )[-1]
            if endpoint:
                key = "prefix:" + endpoint
                old = await get_state(self.store.state, request.scope, key)
                while True:
                    owners = old.value.get("owners", [])
                    if request.session in owners or len(owners) >= 2:
                        break
                    if await self.store.state.swap(
                        request.scope, key, old, {"owners": [*owners, request.session]}
                    ):
                        break
                    old = await get_state(self.store.state, request.scope, key)
        if (
            request.disabled
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
            await CapturePipeline(self.store.capture).stage(request, response, policy)

    async def takeover(self, request, policy):
        if self.replace_archive(request, policy):
            return
        usage = request.observation.value.get("usage", {})
        model = request.body.get("model", "")
        resolved = request.upstream.get("aliases", {}).get(model, model)
        window = request.upstream.get("context_windows", {}).get(resolved, policy.context_window)
        archive = request.capture.value.get("archive")
        if not (
            policy.takeover
            and policy.capture
            and not request.disabled
            and window
            and archive
            and archive["status"] == "pending"
            and not request.capture.value.get("error")
            and request.capture.value.get("tokens", 0) >= policy.takeover_tokens
            and self.eligible(request, archive["boundary"], policy)
            and usage.get("model") in {model, resolved}
            and usage.get("upstream_id") == request.upstream.get("id")
            and usage.get("input_tokens", 0) + usage.get("output_tokens", 0) >= window * 0.9
        ):
            return
        deadline = time.monotonic() + policy.archive_wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(min(POLL_SECONDS, deadline - time.monotonic()))
            request.records, _, request.capture = await self.store.load(
                request.scope, request.session, ["", *request.chain, *request.capture_chain]
            )
            if self.replace_archive(request, policy):
                return
            current = request.capture.value.get("archive")
            if not current or current["status"] != "pending" or request.capture.value.get("error"):
                return
        request.metrics["degradation"] = "archive_wait_timeout"

    @staticmethod
    def eligible(request, anchor, policy):
        if anchor not in request.capture_chain:
            return False
        index = request.capture_chain.index(anchor)
        return sum(is_user(m) for m in request.messages[index + 1 :]) >= policy.keep_recent_turns

    def replace_archive(self, request, policy):
        for index in range(len(request.capture_chain) - 1, -1, -1):
            anchor = request.capture_chain[index]
            replacement = request.records.get((K.REPLACEMENT, anchor))
            if replacement is None or not self.eligible(request, anchor, policy):
                continue
            field = "input" if request.protocol == "responses" else "messages"
            prefix = [
                m
                for m in request.body[field][: index + 1]
                if m.get("role") in {"system", "developer"}
            ]
            request.strip_replayed_thinking = True
            request.body_chain = [""] * (len(prefix) + 1) + request.body_chain[index + 1 :]
            request.body[field] = [
                *prefix,
                {"role": "user", "content": replacement["text"]},
                *strip_thinking(request.body[field][index + 1 :]),
            ]
            request.metrics["archive_replayed"] = anchor
            return True
        return False
