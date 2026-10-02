# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Chat-only hidden tool loop. Text is yielded as soon as upstream emits it."""

import asyncio
import copy
import time
import uuid

import async_timeout
import orjson

from .protocols import SSEDecoder, usage_of
from .records import RecordKind as K
from .tool_catalog import PREFIX, hidden_chain


class ToolLoopError(Exception):
    def __init__(self, reason, status=502, content=None, headers=None):
        super().__init__(reason)
        self.status, self.content, self.headers = status, content, headers or {}


def merge_delta(target, delta):
    """Keep unknown message fields as well as reasoning_content and signatures."""
    for key, value in delta.items():
        if value is None:
            target.setdefault(key, None)
        elif isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            merge_delta(target[key], value)
        elif isinstance(value, str) and key not in {"role", "type"}:
            target[key] = (target.get(key) or "") + value
        elif isinstance(value, list):
            target.setdefault(key, []).extend(copy.deepcopy(value))
        else:
            target[key] = copy.deepcopy(value)


def add_usage(total, usage):
    for key, value in usage.items():
        if isinstance(value, dict):
            add_usage(total.setdefault(key, {}), value)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


def sse(value):
    return b"data: " + orjson.dumps(value) + b"\n\n"


def limit_completion(body, remaining, input_tokens):
    """Only a hidden continuation consumes this budget; never mutate the first request."""
    if remaining <= input_tokens:
        raise ToolLoopError("Hidden tool token budget exhausted")
    name = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
    body[name] = min(body.get(name) or remaining, remaining - input_tokens)


class ChatToolLoop:
    def __init__(self, prepared, executor, store, capture):
        self.prepared, self.executor, self.store, self.capture = prepared, executor, store, capture
        self.body = copy.deepcopy(prepared.body)
        self.policy = prepared.root["policy"]
        self.allowed = executor.allowed
        self.deadline = time.monotonic() + self.policy.get("tool_total_seconds", 120)
        self.identifier = "chatcmpl-ovcg-" + uuid.uuid4().hex
        self.visible = {"role": "assistant", "content": ""}
        self.transcript, self.usage = [], {}
        self.final = None
        self.hidden = False
        self.rounds = 0
        self.token_cost = 0

    async def run(self, response, send):
        """Yield visible SSE events; nonstream callers consume and use self.final."""
        streaming = bool(self.body.get("stream"))
        try:
            async with async_timeout.timeout_at(self.deadline):
                while True:
                    if response.status >= 300:
                        raise ToolLoopError(
                            "Model upstream rejected a tool continuation",
                            response.status,
                            await response.read(),
                            dict(response.headers),
                        )
                    if response.headers.get("content-encoding"):
                        raise ToolLoopError("Compressed tool responses are unsupported")
                    message, calls, usage, finish, envelope = (
                        {"role": "assistant"},
                        {},
                        {},
                        None,
                        {},
                    )
                    if streaming:
                        if "text/event-stream" not in response.headers.get("content-type", ""):
                            raise ToolLoopError("Expected an event stream")
                        decoder = SSEDecoder()
                        received = 0
                        async for chunk in response.content.iter_any():
                            received += len(chunk)
                            if received > self.policy.get("tool_total_tokens", 100000) * 32:
                                raise ToolLoopError("Model stream exceeds tool response budget")
                            for frame in decoder.feed(chunk):
                                value = decoder.data(frame)
                                if value is None:
                                    continue
                                if "error" in value:
                                    raise ToolLoopError(
                                        "Model stream failed", content=orjson.dumps(value)
                                    )
                                envelope.update(
                                    {
                                        k: v
                                        for k, v in value.items()
                                        if k not in {"choices", "usage"}
                                    }
                                )
                                if value.get("usage"):
                                    usage.update(value["usage"])
                                choices = value.get("choices") or []
                                if not choices:
                                    # Preserve unknown top-level events, withholding usage until summed.
                                    if not value.get("usage"):
                                        yield sse({**value, "id": self.identifier})
                                    continue
                                if len(choices) != 1 or choices[0].get("index", 0) != 0:
                                    raise ToolLoopError("Tool mode requires one completion")
                                choice = choices[0]
                                delta = copy.deepcopy(choice.get("delta", {}))
                                for call in delta.pop("tool_calls", []) or []:
                                    call = dict(call)
                                    index = call.pop("index", 0)
                                    merge_delta(calls.setdefault(index, {}), call)
                                merge_delta(message, delta)
                                merge_delta(self.visible, delta)
                                finish = choice.get("finish_reason") or finish
                                if delta or any(
                                    k not in {"delta", "finish_reason", "index"} for k in choice
                                ):
                                    yield sse(
                                        {
                                            **value,
                                            "id": self.identifier,
                                            "usage": None,
                                            "choices": [
                                                {
                                                    **choice,
                                                    "index": 0,
                                                    "delta": delta,
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                        if decoder.buffer.strip():
                            raise ToolLoopError("Incomplete model event stream")
                        if calls:
                            message["tool_calls"] = [calls[i] for i in sorted(calls)]
                    else:
                        raw = await response.read()
                        try:
                            value = orjson.loads(raw)
                            choices = value["choices"]
                            if len(choices) != 1:
                                raise ValueError
                            message = copy.deepcopy(choices[0]["message"])
                            finish = choices[0]["finish_reason"]
                            usage = value.get("usage") or {}
                            envelope = value
                            merge_delta(
                                self.visible,
                                {k: v for k, v in message.items() if k != "tool_calls"},
                            )
                        except (ValueError, KeyError, TypeError) as error:
                            raise ToolLoopError("Invalid tool completion") from error
                    response.close()
                    if not finish:
                        raise ToolLoopError("Model response ended before finish_reason")
                    message.setdefault("content", None)
                    self.transcript.append(message)
                    add_usage(self.usage, usage)
                    normalized = usage_of({"usage": usage})
                    self.capture.context_usage = normalized
                    if self.rounds:
                        self.token_cost += normalized["input_tokens"] + normalized["output_tokens"]
                        self.prepared.metrics["hidden_upstream_calls"] = (
                            self.prepared.metrics.get("hidden_upstream_calls", 0) + 1
                        )
                    for key, value in normalized.items():
                        prefix = "first_upstream_" if self.rounds == 0 else "hidden_upstream_"
                        self.prepared.metrics[prefix + key] = (
                            self.prepared.metrics.get(prefix + key, 0) + value
                        )
                    all_calls = message.get("tool_calls") or []
                    if any(
                        c.get("function", {}).get("name", "").startswith(PREFIX)
                        and c["function"]["name"] not in self.allowed
                        for c in all_calls
                    ):
                        raise ToolLoopError("Model called an unavailable gateway tool")
                    owned = [
                        c for c in all_calls if c.get("function", {}).get("name") in self.allowed
                    ]
                    client = [c for c in all_calls if c not in owned]
                    if owned and finish != "tool_calls":
                        raise ToolLoopError("Gateway tool call has an invalid finish_reason")
                    if owned:
                        if (
                            self.rounds >= self.policy.get("tool_max_rounds", 5)
                            or self.body.get("tool_choice") == "none"
                        ):
                            raise ToolLoopError("Model exceeded the hidden tool round limit")
                        if self.token_cost >= self.policy.get("tool_total_tokens", 100000):
                            raise ToolLoopError("Hidden tool token budget exhausted")
                        self.hidden = True
                        results = []
                        for call in owned:
                            results.append(await self.executor.execute(call))
                        self.transcript.extend(results)
                        self.body["messages"].extend([message, *results])
                        self.rounds += 1
                        self.prepared.metrics["hidden_rounds"] = self.rounds
                    if client:
                        self.visible["tool_calls"] = client
                        if streaming:
                            yield sse(
                                {
                                    **envelope,
                                    "id": self.identifier,
                                    "usage": None,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {"index": i, **c} for i, c in enumerate(client)
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                    if client or not owned:
                        # Persist before publishing the successful terminal event.
                        if self.hidden:
                            anchor = hidden_chain([*self.prepared.messages, self.visible])[-1]
                            await self.store.put(
                                self.prepared.scope,
                                self.prepared.session,
                                K.HIDDEN,
                                anchor,
                                {
                                    "messages": self.transcript,
                                    "upstream_id": self.prepared.root["upstream_id"],
                                },
                            )
                        self.capture.message = copy.deepcopy(self.visible)
                        self.capture.usage = usage_of({"usage": self.usage})
                        self.capture.response_id = self.identifier
                        self.capture.complete = finish in {"stop", "length"}
                        self.prepared.metrics["hidden_rounds"] = self.rounds
                        self.final = {
                            **envelope,
                            "id": self.identifier,
                            "usage": self.usage,
                            "choices": [
                                {
                                    **((envelope.get("choices") or [{}])[0]),
                                    "index": 0,
                                    "message": self.visible,
                                    "finish_reason": finish,
                                }
                            ],
                        }
                        if streaming:
                            yield sse(
                                {
                                    **envelope,
                                    "id": self.identifier,
                                    "usage": None,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                                }
                            )
                            if self.body.get("stream_options", {}).get("include_usage"):
                                yield sse(
                                    {
                                        **envelope,
                                        "id": self.identifier,
                                        "choices": [],
                                        "usage": self.usage,
                                    }
                                )
                            yield b"data: [DONE]\n\n"
                        return
                    remaining = self.policy.get("tool_total_tokens", 100000) - self.token_cost
                    # Use the last provider usage plus newly added tool results.
                    # Raw request bytes (especially images) are not token counts.
                    extra = sum(len(result.get("content", "").encode()) for result in results) // 3
                    limit_completion(
                        self.body,
                        remaining,
                        normalized["input_tokens"] + normalized["output_tokens"] + extra,
                    )
                    if self.rounds >= self.policy.get("tool_max_rounds", 5):
                        self.body["tool_choice"] = "none"
                    response = await send(self.body, max(0.01, self.deadline - time.monotonic()))
        except (ValueError, TypeError, KeyError) as error:
            raise ToolLoopError("Invalid model tool response") from error
        except asyncio.TimeoutError as error:
            raise ToolLoopError("Hidden tool request timed out", 504) from error
        finally:
            response.close()
