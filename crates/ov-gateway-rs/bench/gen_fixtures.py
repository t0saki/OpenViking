"""Build microbenchmark fixtures shared by the Python and Rust runs.

Usage: python gen_fixtures.py BENCH_DIR OUT_DIR
  BENCH_DIR holds agent_sim.py (from the w157 harness); OUT_DIR receives
  body_44k.json / body_140k.json (Claude Code shaped continuation requests built by
  agent_sim's Session), sse_{200,800,1500}.json (lists of upstream chunks exactly as
  mock_agent_upstream.py writes them: 150 thinking tokens + N text tokens + client
  tool_use, one write per event) and tools.json (the recorded OV MCP tool catalog).
"""

import hashlib
import json
import random
import sys
import uuid

sys.path.insert(0, sys.argv[1])
import agent_sim  # noqa: E402

out = sys.argv[2]


class Args:
    e1, max_tokens, ident, plugin, key = True, 10**9, "header", False, "k"


for label, target in (("44k", 44_000), ("140k", 140_000)):
    rnd = random.Random(7)
    sess = agent_sim.Session(Args, rnd, sid="fixture-" + label)
    sess.synth_history(target)
    kind, headers, body = sess.build_request()
    raw = json.dumps(body, ensure_ascii=False).encode()
    with open(f"{out}/body_{label}.json", "wb") as f:
        f.write(raw)
    print(label, kind, len(raw), "bytes", len(raw) // 4, "tokens", len(body["messages"]), "messages")

VOCAB = ("the file function returns value config test error module import class request "
         "session gateway memory update check path handler result build deploy cache "
         "token stream parse output input index query worker latency").split()


def words(n, rnd):
    return " ".join(rnd.choice(VOCAB) for _ in range(n)) + " "


def stream(n_out, think=150, tps=60, chunk_s=0.05):
    rnd = random.Random(n_out)
    chunks = []

    def ev(name, data):
        chunks.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")

    per = max(1, round(tps * chunk_s))
    think_text = words(think, rnd)
    sig = hashlib.sha256(think_text.encode()).hexdigest() * 4
    tool = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": "Read",
            "input": {"path": "src/module_1.py", "pattern": words(4, rnd).strip()}}
    usage_in = {"input_tokens": 100000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    ev("message_start", {"type": "message_start", "message": {
        "id": "msg_x", "type": "message", "role": "assistant", "model": "mock-model", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {**usage_in, "output_tokens": 1}}})
    ev("content_block_start", {"type": "content_block_start", "index": 0,
                               "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
    for kind, index, n in (("thinking", 0, think), ("text", 1, n_out)):
        if kind == "text":
            ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            ev("content_block_start", {"type": "content_block_start", "index": 1,
                                       "content_block": {"type": "text", "text": ""}})
        sent = 0
        while sent < n:
            k = min(per, n - sent)
            delta = ({"type": "thinking_delta", "thinking": words(k, rnd)} if kind == "thinking"
                     else {"type": "text_delta", "text": words(k, rnd)})
            ev("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
            sent += k
        if kind == "thinking":
            ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "signature_delta", "signature": sig}})
    ev("content_block_stop", {"type": "content_block_stop", "index": 1})
    ev("content_block_start", {"type": "content_block_start", "index": 2, "content_block": {**tool, "input": {}}})
    ev("content_block_delta", {"type": "content_block_delta", "index": 2,
                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool["input"])}})
    ev("content_block_stop", {"type": "content_block_stop", "index": 2})
    ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                         "usage": {"output_tokens": think + n_out}})
    ev("message_stop", {"type": "message_stop"})
    return chunks


for n in (200, 800, 1500):
    chunks = stream(n)
    json.dump(chunks, open(f"{out}/sse_{n}.json", "w"))
    print("sse", n, len(chunks), "events", sum(map(len, chunks)), "bytes")

recording = sys.argv[1] + "/ov_recording.jsonl"
for line in open(recording):
    r = json.loads(line)
    if r["path"] == "/mcp" and '"tools/list"' in r["req"]:
        data = next(l[5:].strip() for l in r["resp"].splitlines() if l.startswith("data:"))
        json.dump(json.loads(data)["result"]["tools"], open(f"{out}/tools.json", "w"))
        break
