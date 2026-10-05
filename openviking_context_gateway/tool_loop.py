# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One hidden-tool lifecycle shared by native model protocol adapters."""

import asyncio
import time

import async_timeout
import orjson

from .protocols import SSEDecoder, messages_of, usage_of
from .records import RecordKind as K
from .tool_protocols import hidden_chain, tool_protocol
from .tool_protocols.common import (
    PREFIX,
    ToolLoopError,
    ToolRound,
    add_usage,
    notice_head,
    notice_tail,
)


def added_tokens(value):
    # Client history, schemas and images never consume the continuation budget.
    return (len(orjson.dumps(value)) + 2) // 3


class HiddenToolLoop:
    def __init__(self, prepared, executor, store, capture):
        self.prepared, self.executor, self.store, self.capture = prepared, executor, store, capture
        self.protocol = prepared.protocol
        self.adapter = tool_protocol(self.protocol)(prepared.body)
        self.body = {
            **prepared.body,
            self.adapter.field: list(messages_of(prepared.body, self.protocol)),
        }
        self.policy, self.allowed = prepared.root["policy"], executor.allowed
        self.deadline = time.monotonic() + self.policy.get("tool_total_seconds", 120)
        self.transcript, self.usage = [], {}
        self.final, self.hidden = None, False
        self.rounds, self.token_cost = 0, 0
        self.round = ToolRound([], [], {}, False)

    async def read(self, response):
        adapter = self.adapter
        adapter.begin()
        if response.status >= 300:
            raise ToolLoopError(
                "Model upstream rejected a tool continuation",
                response.status,
                await response.read(),
                dict(response.headers),
            )
        if response.headers.get("content-encoding"):
            raise ToolLoopError("Compressed tool responses are unsupported")
        if not self.body.get("stream"):
            adapter.load(orjson.loads(await response.read()))
        else:
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ToolLoopError("Expected an event stream")
            decoder, received = SSEDecoder(), 0
            async for chunk in response.content.iter_any():
                received += len(chunk)
                if received > 64 * 1024 * 1024:
                    raise ToolLoopError("Model stream exceeds tool response budget")
                for frame in decoder.feed(chunk):
                    value = decoder.data(frame)
                    if value is None:
                        continue
                    if "error" in value or value.get("type") == "error":
                        raise ToolLoopError("Model stream failed")
                    for event in adapter.event(value):
                        yield adapter.encode(event)
            if decoder.buffer.strip():
                raise ToolLoopError("Incomplete model event stream")
        self.round = adapter.end()
        response.close()

    def observe(self):
        self.transcript.extend(self.round.output)
        add_usage(self.usage, self.round.usage)
        normalized = usage_of({"usage": self.round.usage})
        self.capture.context_usage = normalized
        if self.rounds:
            self.token_cost += normalized["output_tokens"] or added_tokens(self.round.output)
            self.prepared.metrics["hidden_upstream_calls"] = self.rounds
        prefix = "hidden_upstream_" if self.rounds else "first_upstream_"
        for key, value in normalized.items():
            self.prepared.metrics[prefix + key] = self.prepared.metrics.get(prefix + key, 0) + value

    def stream(self, events):
        # Adapters update visible history either way; only streams send events.
        return [self.adapter.encode(e) for e in events] if self.body.get("stream") else []

    async def execute(self, calls):
        """Run gateway-owned calls, yielding the visible notice for each one."""
        if (
            self.rounds >= self.policy.get("tool_max_rounds", 5)
            or self.adapter.tool_choice(self.body) == "none"
        ):
            raise ToolLoopError("Model exceeded the hidden tool round limit")
        if not self.rounds:
            self.token_cost += added_tokens(calls)
        show = self.policy.get("show_tool_calls", True)
        events = self.adapter.open_notice() if show else []
        results = []
        for call in calls:
            skipped = self.token_cost >= self.policy.get("tool_total_tokens", 100000)
            if show:
                # The head streams before a slow call runs; its outcome follows.
                events.extend(self.adapter.notice("\n\n" + notice_head(call)))
                for event in self.stream(events):
                    yield event
            if skipped:
                result = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "Gateway tool budget reached; answer using the available results.",
                }
            else:
                result = await self.executor.execute(call)
            if show:
                events = self.adapter.notice(notice_tail(result.get("failed", False), skipped))
            results.append(result)
            self.token_cost += added_tokens(result)
        if show:
            events.extend([*self.adapter.notice("\n\n"), *self.adapter.close_notice()])
            for event in self.stream(events):
                yield event
        results = self.adapter.results(results)
        self.transcript.extend(results)
        self.body[self.adapter.field].extend([*self.round.output, *results])
        self.rounds += 1
        self.hidden = True
        self.prepared.metrics.update(hidden_rounds=self.rounds, hidden_added_tokens=self.token_cost)
        exhausted = self.token_cost >= self.policy.get("tool_total_tokens", 100000)
        if exhausted or self.rounds >= self.policy.get("tool_max_rounds", 5):
            self.adapter.disable_tools(self.body)
        if exhausted:
            self.prepared.metrics["tool_stop_reason"] = "token_budget"

    async def persist(self):
        visible = self.adapter.visible
        anchor = (
            hidden_chain([*self.prepared.messages, *visible], self.protocol)[-1] if visible else ""
        )
        if self.hidden and anchor:
            await self.store.replay.put(
                self.prepared.scope,
                self.prepared.session,
                K.HIDDEN,
                anchor,
                {
                    "messages": self.transcript,
                    "visible_count": len(visible),
                    "upstream_id": self.prepared.root["upstream_id"],
                },
            )
        elif self.hidden:
            self.prepared.metrics["degradation"] = "hidden_reply_without_anchor"
        self.final = self.adapter.final(self.usage)
        self.capture.nonstream(self.final)
        if self.round.calls and self.round.tool_handoff:
            # A successful client-tool handoff is replayable too. The next
            # continuation replaces this unconfirmed capture tail in place.
            self.capture.complete = True
        self.prepared.metrics["hidden_rounds"] = self.rounds

    def error(self, message: str) -> bytes:
        return b"".join(self.adapter.encode(event) for event in self.adapter.error(message))

    async def run(self, response, send):
        """Publish the terminal event only after the exact transcript is durable."""
        try:
            async with async_timeout.timeout_at(self.deadline):
                while True:
                    async for event in self.read(response):
                        yield event
                    self.observe()
                    owned, client = [], []
                    for call in self.round.calls:
                        name = call["function"]["name"]
                        if name.startswith(PREFIX) and name not in self.allowed:
                            raise ToolLoopError("Model called an unavailable gateway tool")
                        (owned if name in self.allowed else client).append(call)
                    if owned:
                        if not self.round.tool_handoff:
                            raise ToolLoopError("Gateway tool call has an invalid stop reason")
                        async for event in self.execute(owned):
                            yield event
                    for event in self.stream(self.adapter.publish_calls(client)):
                        yield event
                    if client or not owned:
                        await self.persist()
                        if self.body.get("stream"):
                            for event in self.adapter.terminal(self.final):
                                yield self.adapter.encode(event)
                        return
                    response = await send(self.body, max(0.01, self.deadline - time.monotonic()))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise ToolLoopError("Invalid model tool response") from error
        except asyncio.TimeoutError as error:
            raise ToolLoopError("Hidden tool request timed out", 504) from error
        finally:
            response.close()
