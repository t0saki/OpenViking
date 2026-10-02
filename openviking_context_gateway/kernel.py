# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Framework-independent recall, immutable replay and capture orchestration."""

import asyncio
import html
import logging
import time
import uuid
from dataclasses import dataclass, field

from .archives import POLL_SECONDS, refresh_archive
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
    anonymous: bool = False
    capture_session: str = ""
    capture_ov_session: str = ""
    tools_active: bool = False
    upstream: dict = field(default_factory=dict)


class MemoryKernel:
    def __init__(self, store: KernelStore, viking):
        self.store, self.viking = store, viking

    async def _records(self, scope, sid, anchors):
        local, inherited = await asyncio.gather(
            self.store.read(
                scope,
                sid,
                anchors,
                [*SESSION_KINDS, *[k for k in REPLAY_KINDS if k not in {K.INJECTION, K.HIDDEN}]],
            ),
            self.store.lookup(scope, anchors, [K.INJECTION, K.HIDDEN]),
        )
        return {**local, **inherited}

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
        anonymous = sid is None
        if anonymous:
            # Only completed assistant endpoints observed by this gateway identify
            # anonymous continuations. A common greeting alone proves nothing.
            identity_chain = (
                await asyncio.to_thread(hidden_chain, messages) if protocol == "chat" else chain
            )
            endpoints = [
                a
                for a, m in zip(identity_chain, messages, strict=True)
                if a and m.get("role") == "assistant"
            ]
            owners = (
                await self.store.owners(scope, endpoints, K.SESSION_PREFIX) if endpoints else {}
            )
            for anchor in reversed(endpoints):
                if anchor in owners:
                    sid = owners[anchor][0] if len(owners[anchor]) == 1 else None
                    break
            sid = sid or "anonymous-" + uuid.uuid4().hex
        body_chain = chain
        records = await self._records(scope, sid, list(dict.fromkeys(["", *chain])))
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
                    "tools": select_tools(body, protocol, upstream, policy) if not plugin else [],
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
        if protocol == "chat" and root.get("tools"):
            body_chain = await asyncio.to_thread(hidden_chain, messages)
            if body_chain != chain:
                records.update(await self.store.lookup(scope, body_chain, [K.HIDDEN]))
        kind, anchor = classify(body, headers, messages, counting)
        disabled = (K.DISABLED, "") in records
        if plugin:
            await self.store.put(scope, sid, K.DISABLED, "", {"reason": "plugin_present"})
            disabled = True
        result = dict(body)
        # Changed message fields are replaced, never mutated in place. Nested
        # tool/image payloads remain shared until an operation needs to change them.
        result_messages = [dict(message) for message in messages]
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
        prepared.anonymous = anonymous
        route = records.get((K.CAPTURE_ROUTE, ""))
        prepared.capture_session = route["session"] if route else sid
        prepared.capture_ov_session = route["ov_session"] if route else root["ov_session"]
        prepared.upstream = upstream
        prepared.metrics.update(session=sid, protocol=protocol, credential_id=credential["id"])
        failures = (
            records
            if prepared.capture_session == sid
            else await self.store.read(scope, prepared.capture_session, [""], [K.CAPTURE_FAILED])
        )
        failure = failures.get((K.CAPTURE_FAILED, ""), {})
        prepared.metrics.update(
            capture_status=("paused" if failure.get("attempts", 5) >= 5 else "retrying")
            if failure
            else ("active" if policy.capture else "disabled"),
            capture_reason=failure.get("reason", ""),
            capture_retry_at=failure.get("retry_at"),
        )
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
            if reason or collision:
                prepared.metrics["tool_skip_reason"] = reason or "tool_name_collision"
            else:
                result["tools"] = [*body.get("tools", []), *tools]
                prepared.tools_active = True
                if disabled or kind not in {"user", "continuation"}:
                    result["tool_choice"] = "none"
        elif policy.gateway_tools:
            prepared.metrics["tool_skip_reason"] = (
                tool_block_reason(body, protocol, upstream) or "tools_not_selected_at_session_start"
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
            await self._takeover(prepared, credential, policy, allow_new=False)
            self._replay_tools(prepared)
            return prepared
        if kind == "user" and anchor >= 0 and (K.INJECTION, chain[anchor]) not in records:
            decision = await self._recall(prepared, credential, policy)
            if decision["text"]:
                append_context(result_messages[anchor], decision["text"], protocol)
        if policy.capture and kind == "user":
            from .capture import CapturePipeline

            await CapturePipeline(self.store).confirm(prepared, credential, policy)
        await self._takeover(
            prepared,
            credential,
            policy,
            allow_new=(
                policy.takeover
                and policy.capture
                and prepared.metrics.get("capture_status") == "active"
                and kind in {"user", "continuation", "count"}
            ),
        )
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
        if request.protocol == "chat" and request.tools_active:
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
            request.anonymous
            and request.kind in {"user", "continuation"}
            and response.complete
            and response.message
        ):
            returned = [*request.messages, *(response.output_items or [response.message])]
            endpoint = (
                await asyncio.to_thread(
                    hidden_chain if request.protocol == "chat" else prefix_chain, returned
                )
            )[-1]
            if endpoint:
                await self.store.put(request.scope, request.session, K.SESSION_PREFIX, endpoint, {})
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
            from .capture import CapturePipeline

            await CapturePipeline(self.store).stage(request, credential, policy, response)

    async def _takeover(self, request, credential, policy, *, allow_new=True):
        usage = request.records.get((K.USAGE, ""), {})
        model = request.body.get("model", "")
        resolved = request.upstream.get("aliases", {}).get(model, model)
        window = request.upstream.get("context_windows", {}).get(resolved, policy.context_window)
        urgent = bool(
            allow_new
            and window
            and usage.get("model") in {model, resolved}
            and usage.get("upstream_id") == request.upstream.get("id")
            and usage.get("input_tokens", 0) + usage.get("output_tokens", 0) >= window * 0.9
        )
        deadline = time.monotonic() + (policy.archive_wait_seconds if urgent else 0)
        while True:
            replaced, pending = await self._replace_archive(request, credential, policy, allow_new)
            if replaced or not (urgent and pending):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                request.metrics["degradation"] = "archive_wait_timeout"
                return
            await asyncio.sleep(min(POLL_SECONDS, remaining))
            request.records.update(
                await self.store.read(
                    request.scope, request.session, request.chain, [K.ARCHIVE, K.REPLACEMENT]
                )
            )

    async def _replace_archive(self, request, credential, policy, allow_new=True):
        if not any(k in {K.ARCHIVE, K.REPLACEMENT} for k, _ in request.records):
            return False, False
        positions = {anchor: i for i, anchor in enumerate(request.chain) if anchor}
        candidates = sorted(
            {a for k, a in request.records if k in {K.ARCHIVE, K.REPLACEMENT} and a in positions},
            key=positions.get,
            reverse=True,
        )
        pending = False
        for anchor in candidates:
            index = positions[anchor]
            replacement = request.records.get((K.REPLACEMENT, anchor))
            invalid = replacement if replacement and replacement.get("fallback") else None
            if replacement is not None and not invalid:
                self._apply_replacement(request, index, replacement)
                request.metrics["archive_replayed"] = anchor
                return True, pending
            archive = request.records.get((K.ARCHIVE, anchor))
            if not allow_new or not archive:
                continue
            if archive.get("takeover") is False and (K.TAKEOVER, "") not in request.records:
                continue
            if sum(is_user(m) for m in request.messages[index + 1 :]) < policy.keep_recent_turns:
                continue
            archive = await refresh_archive(
                self.store,
                self.viking,
                request.scope,
                request.session,
                anchor,
                archive,
                credential["openviking_key"],
                archive.get("ov_session", request.root["ov_session"]),
            )
            request.records[K.ARCHIVE, anchor] = archive
            if archive.get("status") != "ready":
                pending |= archive.get("status") == "pending"
                request.metrics["archive_status"] = archive.get("status", "unknown")
                continue
            value = {"text": "[OpenViking Session Context]\n" + archive["summary"]}
            if invalid:
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
            return True, pending
        return False, pending

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
