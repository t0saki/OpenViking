# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Framework-independent recall, immutable replay, compaction and capture orchestration."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, replace

import orjson

from .blocks import block, gateway_note, history_hint, token_estimate
from .capture import CapturePipeline, lineage
from .capture_store import Document
from .client import VikingError
from .compaction import (
    BACKOFF_SECONDS,
    active_cut,
    apply_cut,
    cut_hint,
    cut_point,
    estimate,
    opening_block,
    replacement_text,
    summary_instruction,
    window_size,
)
from .models import Policy
from .profile import build_profile
from .protocols import (
    append_context,
    assistant_texts,
    classify,
    clean_text,
    is_user,
    messages_of,
    plugin_present,
    prefix_chain,
    session_id,
    strip_thinking,
    text_content,
    unsigned_thinking,
    unwrap_client,
)
from .records import RecordKind as K
from .state_store import get_state
from .storage import KernelStore, digest
from .tool_catalog import TOOL_VERSION, select_tools, tool_block_reason
from .tool_protocols import hidden_chain, replay_hidden, tool_protocol
from .tool_protocols.common import SummaryError
from .vendors import parameter_fingerprint
from .windows import remind, status_line

logger = logging.getLogger(__name__)
# State keys for unsigned thinking the gateway relayed, by digest of its text.
THINKING = "thinking:"


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
    # Definitions and hidden history replay, but the tool loop refuses every gateway call.
    tools_closed: bool = False
    hidden_history_unavailable: bool = False
    upstream: dict = field(default_factory=dict)
    # The estimated current context and the window it is measured against.
    context_tokens: int = 0
    context_window: int = 0


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

    async def sent_history(self, scope, messages, protocol):
        """The history as the gateway relayed it: thinking a client resent as text is dropped.

        Some providers return thinking without a signature, and clients such as pi
        resend it as a text block. Anchors drop thinking, so that text would keep
        every reply from matching its anchors in the next request.
        """
        if protocol != "anthropic":
            return messages
        texts = {id(b): THINKING + digest(b["text"]) for m in messages for b in assistant_texts(m)}
        known = await self.store.state.read(scope, list(set(texts.values()))) if texts else {}
        if not known:
            return messages
        dropped = {block for block, key in texts.items() if key in known}
        return [
            {**m, "content": [b for b in m["content"] if id(b) not in dropped]}
            if assistant_texts(m)
            else m
            for m in messages
        ]

    async def identity(self, scope, messages, protocol, headers, tools):
        sid = session_id(headers)
        needs_hidden = tool_protocol(protocol).canonicalizes_history and (sid is None or tools)

        def chains():
            chain = prefix_chain(messages)
            return chain, hidden_chain(messages, protocol) if needs_hidden else chain

        # Small prompts need no executor; large payloads must not block the loop.
        if len(messages) > 32 or any(len(m.get("content") or "") > 8192 for m in messages):
            chain, body_chain = await asyncio.to_thread(chains)
        else:
            chain, body_chain = chains()
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
        return sid, anonymous, chain, body_chain

    async def prepare(
        self,
        body,
        protocol,
        headers,
        credential,
        upstream,
        policy,
        counting=False,
        upstreams=(),
        summarize=None,
    ):
        """Build the upstream body; ``summarize(prepared, body)`` sends a summary request."""
        sent = messages_of(body, protocol)
        if not isinstance(sent, list) or not all(isinstance(m, dict) for m in sent):
            raise ValueError("invalid message list")
        scope = digest(credential["account"] + "\0" + credential["user_id"] + "\0" + protocol)
        # Anchors, capture and recall read the history as the gateway relayed it;
        # the upstream still gets exactly what the client sent.
        messages = await self.sent_history(scope, sent, protocol)
        sid, anonymous, chain, body_chain = await self.identity(
            scope,
            messages,
            protocol,
            headers,
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
            tool_protocol(protocol).canonicalizes_history
            and (root["tools"] or root["policy"]["capture"])
            and body_chain is chain
        ):
            body_chain = await asyncio.to_thread(hidden_chain, messages, protocol)
            if body_chain != chain:
                records.update(await self.store.replay.read(scope, sid, body_chain))
        upstream = next((u for u in upstreams if u["id"] == root["upstream_id"]), upstream)
        policy = Policy.model_validate(root["policy"])
        kind, anchor = classify(body, headers, messages)
        disabled = (K.DISABLED, "") in records
        if plugin and not disabled:
            await self.store.replay.put(scope, sid, K.DISABLED, "", {"reason": "plugin_present"})
            disabled = True
        # A token count gets gateway tools exactly when the request it measures would.
        owner = not disabled and kind in {"user", "continuation"}
        kind = "count" if counting else kind
        field = tool_protocol(protocol).field
        result = {**body, field: [dict(m) for m in sent]}
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
        self.configure_tools(prepared, policy, owner)
        self.replay(prepared)
        if not disabled and policy.capture and kind == "user" and anchor >= 0:
            await CapturePipeline(self.store.capture).confirm(prepared, credential, policy)
        if kind in {"user", "continuation"}:
            prepared.context_window = window_size(prepared, policy)
            prepared.context_tokens = await self.measure(prepared)
            prepared.metrics.update(
                context_tokens=prepared.context_tokens, context_window=prepared.context_window
            )
            if (
                summarize
                and policy.compaction
                and not disabled
                and prepared.context_tokens >= policy.compaction_threshold * prepared.context_window
            ):
                await self.compact(prepared, credential, policy, summarize)
        # Recall and window signals come after any new cut, so they describe the
        # context the model actually gets.
        if not disabled:
            if kind == "user" and anchor >= 0 and (K.INJECTION, chain[anchor]) not in records:
                await self.recall(prepared, credential, policy)
            await remind(self.store, prepared, policy)
        self.assemble(prepared)
        if isinstance(body.get("input"), str) and result.get("input") == sent:
            result["input"] = body["input"]
        return prepared

    async def session_root(
        self, records, scope, sid, body, protocol, upstream, policy, credential, plugin
    ):
        root = records.get((K.ROOT, ""))
        if root is None:
            # The tool list is frozen here, so a failed load leaves the session without tools.
            tools, reason = [], ""
            if (
                not plugin
                and policy.get("gateway_tools")
                and not tool_block_reason(body, protocol, upstream)
            ):
                try:
                    catalog = await self.viking.tools(credential["openviking_key"])
                    tools = select_tools(catalog, policy)
                except VikingError:
                    reason = "tools_unavailable"
            root = await self.store.replay.put(
                scope,
                sid,
                K.ROOT,
                "",
                {
                    "upstream_id": upstream["id"],
                    "tool_version": TOOL_VERSION,
                    "tools": tools,
                    "tool_skip_reason": reason,
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
    def configure_tools(request, policy, owner):
        tools = request.root["tools"]
        request.metrics["tools_tokens"] = 0
        adapter = tool_protocol(request.protocol)
        if tools:
            reason = tool_block_reason(request.original, request.protocol, request.upstream)
            names = {t["function"]["name"] for t in tools}
            collision = any(
                t.get("function", t).get("name") in names for t in request.original.get("tools", [])
            )
            hidden = any((K.HIDDEN, anchor) in request.records for anchor in request.body_chain)
            if reason or collision:
                request.metrics["tool_skip_reason"] = reason or "tool_name_collision"
                request.hidden_history_unavailable = hidden
            elif owner or hidden:
                # Only the conversation's own turns get gateway tools. Others that resend
                # history using them still replay it byte for byte, for provider caches and
                # thinking signatures, but the tool loop refuses their calls.
                adapter.add_tools(request.body, tools)
                request.tools_active, request.tools_closed = True, not owner
                request.metrics["tools_tokens"] = token_estimate(
                    orjson.dumps(adapter.wire_tools(tools)).decode()
                )
        elif policy.gateway_tools:
            request.metrics["tool_skip_reason"] = (
                tool_block_reason(request.original, request.protocol, request.upstream)
                or request.root.get("tool_skip_reason")
                or "tools_not_selected_at_session_start"
            )

    @staticmethod
    def replay(request):
        field = tool_protocol(request.protocol).field
        messages = request.body[field]
        missing = False
        sent = set(request.observation.value.get("sent", []))
        # A cut removes the messages up to it, together with any lost injection.
        cut = active_cut(request)
        for index, anchor in enumerate(request.chain):
            decision = request.records.get((K.INJECTION, anchor))
            if decision is not None:
                if decision["text"]:
                    append_context(messages[index], decision["text"], request.protocol)
                request.metrics["replay_hits"] += 1
            elif anchor in sent and index > cut:
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
        existing = [
            (index, request.records[K.INJECTION, anchor])
            for index, anchor in enumerate(request.chain)
            if (K.INJECTION, anchor) in request.records
        ]
        # The budget is per context window: a cut frees the recall it replaced.
        cut = active_cut(request)
        current = [value for index, value in existing if index > cut]
        used = sum(v.get("tokens", 0) for v in current)
        exclude = list(dict.fromkeys(u for v in current for u in v.get("uris", [])))
        budget = min(policy.max_tokens, policy.session_max_tokens - used)
        query = clean_text(unwrap_client(text_content(request.messages[request.anchor])))[
            : policy.query_max_chars
        ]
        decision = {"text": "", "uris": [], "tokens": 0, "reason": "disabled"}
        reserved = False
        if policy.recall and budget >= 64 and len(query) >= 3:
            # The window starts at the cut, or at the history's first user message.
            if cut >= 0:
                window = request.capture_chain[cut]
            else:
                window = next(
                    a
                    for m, a in zip(request.messages, request.capture_chain, strict=True)
                    if is_user(m)
                )
            budget = await self.reserve_recall(request, policy, used, window)
            reserved = budget > 0
        tools = request.root["tools"]
        lead = "Relevant memory from OpenViking."
        if any(tool["function"]["name"] == "openviking_read" for tool in tools):
            lead += " Use the openviking_read tool to expand URIs."
        overhead = token_estimate(block("gateway-recall", lead + "\n"))

        async def retrieve():
            if not reserved:
                return decision
            if budget - overhead < 64:
                return {**decision, "reason": "budget"}
            try:
                response = await self.viking.recall(
                    credential["openviking_key"], query, policy, exclude, budget - overhead
                )
                rendered = response.get("rendered") or ""
                text = lead + "\n" + rendered if rendered else ""
                return {
                    "text": text,
                    "uris": [
                        entry["uri"] for entry in response.get("entries", []) if entry.get("uri")
                    ]
                    if text
                    else [],
                    "tokens": token_estimate(block("gateway-recall", text)),
                    "reason": "recalled" if text else "empty",
                }
            except (VikingError, asyncio.TimeoutError) as error:
                return {**decision, "reason": getattr(error, "reason", "recall_timeout")}

        async def opening():
            if existing:
                return ""
            profile = await build_profile(self.viking, credential["openviking_key"], policy, tools)
            request.metrics["profile_reason"] = profile["reason"]
            # After client-side compaction, the capture document still names the earlier sessions.
            hint = history_hint(
                credential["user_id"], lineage(request.capture.value), tools, policy.capture
            )
            return block(
                "gateway-session-start",
                "\n\n".join(
                    part for part in (gateway_note(policy, tools), hint, profile["text"]) if part
                ),
            )

        decision, start = await asyncio.gather(retrieve(), opening())
        status, reminder = status_line(request, policy)
        recalled = block("gateway-recall", "\n\n".join(p for p in (decision["text"], status) if p))
        # The opening block is immutable with recall but has its own budget, and
        # the window status line costs nothing.
        decision["text"] = "\n\n".join(part for part in (start, recalled) if part)
        if reminder:
            decision["reminder"] = reminder
        anchor = request.chain[request.anchor]
        decision = await self.store.replay.put(
            request.scope, request.session, K.INJECTION, anchor, decision
        )
        if reserved:
            await self.settle_recall(request, anchor, decision["tokens"])
        request.records[K.INJECTION, anchor] = decision
        if decision["text"]:
            field = tool_protocol(request.protocol).field
            append_context(request.body[field][request.anchor], decision["text"], request.protocol)
        request.metrics.update(
            recall_count=len(decision["uris"]),
            recall_ms=round((time.monotonic() - started) * 1000, 2),
            recall_reason=decision["reason"],
        )

    async def reserve_recall(self, request, policy, used, window):
        """Reserve a bounded allowance before recall; only this document uses CAS.

        A crash can leave a conservative reservation, never overspend the cap.
        Competing requests for the same anchor share its reservation and the
        immutable decision; settling it twice cannot refund twice. A ledger for
        another context window starts over from the records in this one.
        """
        anchor, old = request.chain[request.anchor], request.observation
        while True:
            ledger = old.value.get("recall", {})
            if ledger.get("window") != window:
                ledger = {"window": window, "spent": used, "pending": {}}
            if anchor in ledger["pending"]:
                request.observation = old
                return ledger["pending"][anchor]
            budget = min(policy.max_tokens, policy.session_max_tokens - ledger["spent"])
            if budget < 64:
                return 0
            value = {
                **old.value,
                "recall": {
                    "window": window,
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
                "recall": {
                    **ledger,
                    "spent": ledger["spent"] - reserved + tokens,
                    "pending": pending,
                },
            }
            if await self.store.state.swap(request.scope, request.session, old, value):
                request.observation = Document(value, old.version + 1)
                return
            old = await get_state(self.store.state, request.scope, request.session)

    async def completed(self, request, credential, response):
        if request.protocol == "anthropic" and response.message:
            # First, so the client's next request can already be matched.
            for text in unsigned_thinking(response.message):
                await self.store.state.swap(request.scope, THINKING + digest(text), Document(), {})
        chain = set(request.chain)
        sent = [
            a
            for (k, a), v in request.records.items()
            if k == K.INJECTION and v.get("text") and a in chain
        ]
        endpoint = ""
        if response.message and request.kind in {"user", "continuation"}:
            returned = [*request.messages, *(response.output_items or [response.message])]
            endpoint = (await asyncio.to_thread(hidden_chain, returned, request.protocol))[-1]
        # The next request measures its context from this usage while the reply stays in it.
        # Without counts from the upstream, it estimates the whole request instead.
        usage = {
            **(response.context_usage or response.usage or {}),
            "anchor": endpoint,
            "time": time.time(),
        }

        def observe(value):
            if (
                usage.get("input_tokens")
                and endpoint
                and usage["time"] >= value.get("usage", {}).get("time", 0)
            ):
                value["usage"] = usage
            if sent:
                value["sent"] = list(dict.fromkeys([*value.get("sent", []), *sent]))
            if request.kind == "user" and usage["time"] >= value.get("user_at", 0):
                value["user_at"] = usage["time"]
            return value

        await self.update_observation(request, observe)
        if request.anonymous and response.complete and endpoint:
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
        policy = Policy.model_validate(request.root["policy"])
        if policy.capture:
            await CapturePipeline(self.store.capture).stage(request, response, policy)

    @staticmethod
    def assemble(request):
        """Apply the latest cut, then expand or drop hidden tool history."""
        apply_cut(request)
        adapter = tool_protocol(request.protocol)
        messages = request.body[adapter.field]
        if request.tools_active:
            messages = replay_hidden(
                messages, request.body_chain, request.records, request.protocol
            )
            if request.strip_replayed_thinking:
                messages = strip_thinking(messages)
        elif request.hidden_history_unavailable:
            cleaned = adapter.omit_hidden_history(messages)
            if cleaned != messages:
                messages = cleaned
                request.metrics.setdefault("degradation", "hidden_tool_history_unavailable")
        request.body[adapter.field] = messages

    @classmethod
    def assembled(cls, request):
        """The body the upstream would get now, without changing the request."""
        candidate = replace(request, body=dict(request.body), metrics={})
        cls.assemble(candidate)
        return candidate.body

    async def measure(self, request):
        """Estimate the context: the last reply's usage plus what the client added after it.

        Usage counts only when its reply is still in this history after the latest
        cut, so usage from subagents, other branches or before a cut is ignored.
        Otherwise the estimate covers the whole body the upstream would get.
        """
        usage = request.observation.value.get("usage", {})
        anchor, chain = usage.get("anchor"), request.capture_chain
        index = chain.index(anchor) if anchor and anchor in chain else -1
        if index > active_cut(request):
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            return tokens + await asyncio.to_thread(estimate, request.messages[index + 1 :])
        return await asyncio.to_thread(estimate, self.assembled(request))

    async def compact(self, request, credential, policy, summarize):
        """Replace the history before a new cut with a model-written summary.

        The summary covers only what the cut replaces, so every branch sharing that
        prefix can reuse the record. Failures forward the full history and back off.
        """
        failed_at = request.observation.value.get("compaction", {}).get("failed_at", 0)
        cut = cut_point(request)
        if cut <= active_cut(request) or time.time() - failed_at < BACKOFF_SECONDS:
            return
        adapter = tool_protocol(request.protocol)
        body, kept = self.assembled(request), request.body[adapter.field][cut + 1 :]
        messages = body[adapter.field]
        span = messages[: len(messages) - len(kept)]
        # A continuation is cut in the middle of the turn, before the model answered.
        in_progress = request.kind == "continuation"
        started = time.monotonic()
        try:
            # Messages after the cut are new client input that hidden history never expands.
            if messages[len(span) :] != kept:
                raise SummaryError("cut_not_found")
            instruction = summary_instruction(policy.summary_max_tokens, in_progress)
            response = await summarize(
                request,
                adapter.summary_request(body, span, instruction, policy.summary_max_tokens),
            )
            summary = adapter.summary_text(response).strip()
            if not summary:
                raise SummaryError("summary_empty")
        except SummaryError as error:
            await self.compaction_failed(request, error.reason)
            return
        except Exception:
            logger.exception("Context Gateway summary failed")
            await self.compaction_failed(request, "summary_failed")
            return
        anchor = request.capture_chain[cut]
        if policy.capture and in_progress:
            # First, so the hint names the session that receives the part being cut.
            await CapturePipeline(self.store.capture).confirm_cut(request, anchor)
        hint = cut_hint(request, credential["user_id"], policy)
        opening = opening_block(request.records, request.chain)
        text = replacement_text(summary, hint, opening, in_progress)
        # A concurrent request may have written this cut first; its text wins.
        record = await self.store.replay.put(
            request.scope,
            request.session,
            K.REPLACEMENT,
            anchor,
            {"source": "compaction", "text": text, "tokens": token_estimate(text)},
        )
        request.records[K.REPLACEMENT, anchor] = record
        request.metrics.update(
            compaction_tokens=record["tokens"],
            compaction_ms=round((time.monotonic() - started) * 1000, 2),
        )
        # Later readers see the compacted context; the metric keeps the estimate that triggered it.
        request.context_tokens = await self.measure(request)
        if request.context_tokens >= policy.compaction_threshold * request.context_window:
            # What a cut cannot remove fills the window, so summarizing again soon cannot help.
            await self.compaction_failed(request, "still_over_threshold")

    async def compaction_failed(self, request, reason):
        request.metrics["compaction_failed"] = reason
        failure = {"failed_at": time.time(), "reason": reason}
        await self.update_observation(request, lambda value: {**value, "compaction": failure})

    async def update_observation(self, request, change):
        """Apply ``change`` to the session's observation document, retrying on conflicts."""
        old = request.observation
        while True:
            value = change(dict(old.value))
            if value == old.value:
                return
            if await self.store.state.swap(request.scope, request.session, old, value):
                request.observation = Document(value, old.version + 1)
                return
            old = await get_state(self.store.state, request.scope, request.session)
