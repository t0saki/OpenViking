"""Python side of the microbenchmarks, on the gateway source in this repository.

Usage: python pymicro.py FIXTURES_DIR [--iters N]
Prints one JSON line per measurement: {"name", "fixture", "cpu_ms", "wall_ms"} (per call,
process CPU time across all threads, median of 5 batches), plus the last prefix_chain
anchor of each body so the Rust side can check it computes the same anchors.
"""

import asyncio
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import orjson

from openviking_gateway.capture import CapturePipeline
from openviking_gateway.compaction import estimate
from openviking_gateway.kernel import MemoryKernel
from openviking_gateway.models import Policy
from openviking_gateway.protocols import (
    SSEDecoder,
    classify,
    client_openviking_tools,
    parse_body,
    plugin_present,
    prefix_chain,
)
from openviking_gateway.storage import SQLiteKernelStore
from openviking_gateway.tool_catalog import select_tools
from openviking_gateway.tool_protocols import ResponseCapture
from openviking_gateway.tool_protocols.anthropic import AnthropicProtocol

FIX = Path(sys.argv[1])
ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 20
HEADERS = {
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    "x-claude-code-session-id": "fixture-session",
}
CATALOG = json.loads((FIX / "tools.json").read_text())
POLICY = {**Policy(name="default").model_dump(), "id": "default", "revision": 1}
TOOLS = select_tools(CATALOG, POLICY)
NAMES = [t["function"]["name"].removeprefix("openviking_") for t in TOOLS]


def measure(name, fixture, fn, iters=ITERS):
    fn()  # warm
    cpu, wall = [], []
    for _ in range(5):
        c0, w0 = time.process_time(), time.perf_counter()
        for _ in range(iters):
            fn()
        cpu.append((time.process_time() - c0) * 1000 / iters)
        wall.append((time.perf_counter() - w0) * 1000 / iters)
    row = {"name": name, "fixture": fixture, "cpu_ms": round(statistics.median(cpu), 4),
           "wall_ms": round(statistics.median(wall), 4)}
    print(json.dumps(row), flush=True)
    return row


class FakeViking:
    unavailable_reason = ""

    async def tools(self, key):
        return CATALOG


def prep_benchmarks(label):
    raw = (FIX / f"body_{label}.json").read_bytes()
    body = parse_body(raw)
    messages = body["messages"]
    chain = prefix_chain(messages)
    print(json.dumps({"anchor": label, "last": chain[-1], "count": len([a for a in chain if a])}))
    measure("parse_body", label, lambda: parse_body(raw))
    measure("prefix_chain", label, lambda: prefix_chain(messages))
    measure("classify", label, lambda: classify(body, HEADERS, messages))
    measure("plugin_present", label, lambda: plugin_present(body, HEADERS))
    measure("client_openviking_tools", label, lambda: client_openviking_tools(body, NAMES))

    def thinking_digests():
        return {
            hashlib.sha256(b["text"].encode()).hexdigest()
            for m in messages
            for b in AnthropicProtocol.thinking_as_text(m)
        }

    measure("thinking_digests", label, thinking_digests)
    measure("estimate_full_body", label, lambda: estimate(body))
    measure("capture_turns_full", label,
            lambda: CapturePipeline.turns(messages, chain, 0, len(messages), True, 0))
    measure("dumps_body", label, lambda: orjson.dumps(body))

    # The whole preparation of a steady-state tool-result request on an existing session:
    # parse_body + MemoryKernel.prepare (real SQLite store) + proxy's compare and dumps.
    tmp = tempfile.mkdtemp()
    loop = asyncio.new_event_loop()
    store = SQLiteKernelStore(Path(tmp) / "kernel.sqlite3")
    loop.run_until_complete(store.initialize())
    kernel = MemoryKernel(store, FakeViking())
    credential = {"account": "bench", "user_id": "alice", "id": "cred", "openviking_key": "k"}
    upstream = {"id": "mock-anthropic", "protocol": "anthropic", "vendor": "generic",
                "base_url": "http://127.0.0.1:1", "priority": 0}

    async def full():
        parsed = parse_body(raw)
        prepared = await kernel.prepare(
            parsed, "anthropic", dict(HEADERS), credential, upstream, POLICY, False, [upstream], None
        )
        if prepared.body != parsed:
            orjson.dumps(prepared.body)
        return prepared

    # Steady state: the previous reply was relayed and its usage observed, so measure()
    # estimates only the messages after it, as on every tool-result turn under load.
    prepared = loop.run_until_complete(full())
    from openviking_gateway.capture_store import Document

    usage = {"input_tokens": 100000, "output_tokens": 500, "anchor": prepared.chain[-2],
             "time": time.time()}
    assert loop.run_until_complete(
        store.state.swap(prepared.scope, prepared.session, Document(), {"usage": usage}))
    measure("full_prepare", label, lambda: loop.run_until_complete(full()), iters=max(5, ITERS // 2))
    prepared = loop.run_until_complete(full())
    output = [{"role": "assistant", "content": [{"type": "text", "text": "x " * 3000}]}]
    measure("post_reply_chain", label, lambda: prefix_chain([*prepared.messages, *output]))
    store.close()
    loop.close()


def relay(chunks):
    adapter = AnthropicProtocol({"messages": []})
    adapter.lead_with("")
    adapter.begin()
    decoder = SSEDecoder()
    out = []
    received = 0
    for chunk in chunks:
        received += len(chunk)
        for frame in decoder.feed(chunk):
            value = decoder.data(frame)
            if value is None:
                continue
            if "error" in value or value.get("type") == "error":
                raise RuntimeError
            for event in adapter.event(value):
                out.append(adapter.encode(event))
    rnd = adapter.end()
    out.extend(adapter.encode(e) for e in adapter.publish_calls(rnd.calls))
    usage = {}
    for k, v in rnd.usage.items():
        usage[k] = v
    final = adapter.final(usage)
    ResponseCapture("anthropic").nonstream(final)
    out.extend(adapter.encode(e) for e in adapter.terminal(final))
    return out


def relay_benchmarks(n):
    chunks = [c.encode() for c in json.loads((FIX / f"sse_{n}.json").read_text())]
    out = relay(chunks)
    print(json.dumps({"relay_check": n, "events_out": len(out), "bytes_out": sum(map(len, out))}))
    measure("relay", f"sse_{n}", lambda: relay(chunks))


if __name__ == "__main__":
    for label in ("44k", "140k"):
        prep_benchmarks(label)
    for n in (200, 800, 1500):
        relay_benchmarks(n)
