"""Inspectable provider fixture: three protocols, delayed streams and hidden tools."""

import argparse
import asyncio
import copy
import hashlib
import json
import time
import zlib

from aiohttp import web

MODELS = ["mock", "mock-silent", "mock-long", "mock-tools", "mock-silent-gzip"]
TEXT = "mock response complete"


def tool_pending(body):
    """Request a gateway search once per user turn, including replayed histories."""
    items = body.get("messages", body.get("input", []))
    if isinstance(items, str):
        items = []
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if item.get("role") == "tool" or item.get("type") == "function_call_output":
            return False
        if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
            return False
        if item.get("role") == "user":
            break
    return any(
        tool.get("function", tool).get("name") == "openviking_search"
        for tool in body.get("tools", [])
    )


def reply(protocol, body, number):
    model = body.get("model", "mock")
    use_tool = "tools" in model and tool_pending(body)
    history = body.get("messages", body.get("input", []))
    users = [item for item in history if isinstance(item, dict) and item.get("role") == "user"]
    fingerprint = json.dumps([protocol, users or history], sort_keys=True).encode()
    call_id = "call_mock_" + hashlib.sha256(fingerprint).hexdigest()[:16]
    function = {"name": "openviking_search", "arguments": '{"query":"blue"}'}
    if protocol == "chat":
        message = {"role": "assistant", "content": TEXT}
        if use_tool:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": call_id, "type": "function", "function": function}],
            }
        return {
            "id": f"chatcmpl_mock_{number}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if use_tool else "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        }
    if protocol == "anthropic":
        content = [{"type": "text", "text": TEXT}]
        if use_tool:
            content = [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": function["name"],
                    "input": {"query": "blue"},
                }
            ]
        return {
            "id": f"msg_mock_{number}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if use_tool else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 3},
        }
    output = [
        {
            "type": "message",
            "id": f"msg_mock_{number}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": TEXT, "annotations": [], "logprobs": []}],
        }
    ]
    if use_tool:
        output = [
            {
                "type": "function_call",
                "id": f"fc_mock_{number}",
                "call_id": call_id,
                **function,
                "status": "completed",
            }
        ]
    return {
        "id": f"resp_mock_{number}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 3,
            "total_tokens": 23,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def events(protocol, value):
    """Build complete provider event sequences without relying on gateway parsers."""
    if protocol == "chat":
        base = {k: value[k] for k in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"

        def chunk(delta, finish=None):
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        result = [chunk({"role": "assistant", "content": ""})]
        message = value["choices"][0]["message"]
        if message.get("tool_calls"):
            call = copy.deepcopy(message["tool_calls"][0])
            arguments = call["function"].pop("arguments")
            result.append(chunk({"tool_calls": [{"index": 0, **call}]}))
            result.append(
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]})
            )
        else:
            result.extend(chunk({"content": part}) for part in ("mock ", "response ", "complete"))
        result.append(chunk({}, value["choices"][0]["finish_reason"]))
        result.append({**base, "choices": [], "usage": value["usage"]})
        return result
    if protocol == "anthropic":
        result = [
            {
                "type": "message_start",
                "message": {
                    **value,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                },
            }
        ]
        for index, block in enumerate(value["content"]):
            initial = {**block, "text": ""} if block["type"] == "text" else {**block, "input": {}}
            result.append({"type": "content_block_start", "index": index, "content_block": initial})
            deltas = (
                [{"type": "text_delta", "text": p} for p in ("mock ", "response ", "complete")]
                if block["type"] == "text"
                else [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
            )
            result.extend(
                {"type": "content_block_delta", "index": index, "delta": d} for d in deltas
            )
            result.append({"type": "content_block_stop", "index": index})
        return [
            *result,
            {
                "type": "message_delta",
                "delta": {"stop_reason": value["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
    result = [
        {"type": name, "response": {**value, "status": "in_progress", "output": []}}
        for name in ("response.created", "response.in_progress")
    ]
    for index, item in enumerate(value["output"]):
        initial = {**item, "status": "in_progress"}
        initial["content" if item["type"] == "message" else "arguments"] = (
            [] if item["type"] == "message" else ""
        )
        result.append(
            {"type": "response.output_item.added", "output_index": index, "item": initial}
        )
        fields = {"output_index": index, "item_id": item["id"]}
        if item["type"] == "message":
            fields["content_index"] = 0
            part = item["content"][0]
            result.append(
                {"type": "response.content_part.added", **fields, "part": {**part, "text": ""}}
            )
            result.extend(
                {"type": "response.output_text.delta", **fields, "delta": p, "logprobs": []}
                for p in ("mock ", "response ", "complete")
            )
            result.extend(
                [
                    {"type": "response.output_text.done", **fields, "text": TEXT, "logprobs": []},
                    {"type": "response.content_part.done", **fields, "part": part},
                ]
            )
        else:
            result.extend(
                [
                    {
                        "type": "response.function_call_arguments.delta",
                        **fields,
                        "delta": item["arguments"],
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        **fields,
                        "arguments": item["arguments"],
                    },
                ]
            )
        result.append({"type": "response.output_item.done", "output_index": index, "item": item})
    result.append({"type": "response.completed", "response": value})
    return [{**event, "sequence_number": number} for number, event in enumerate(result)]


def create_app():
    records = []
    responses = {}
    counter = 0

    async def handle(request):
        nonlocal counter
        if request.path == "/health":
            return web.json_response({"status": "ok"})
        if request.path == "/_records":
            selected = records
            for name in ("path", "key", "source_pod"):
                if name in request.query:
                    selected = [r for r in selected if r[name] == request.query[name]]
            return web.json_response({"requests": selected})
        if request.path == "/_reset" and request.method == "POST":
            records.clear()
            responses.clear()
            return web.json_response({"status": "ok"})
        if request.path.endswith("/models"):
            return web.json_response(
                {
                    "object": "list",
                    "data": [
                        {"id": name, "object": "model", "created": 0, "owned_by": "mock"}
                        for name in MODELS
                    ],
                }
            )
        paths = {
            "/v1/chat/completions": "chat",
            "/v1/messages": "anthropic",
            "/v1/responses": "responses",
        }
        if request.path == "/v1/messages/count_tokens":
            return web.json_response({"input_tokens": 20})
        if request.path not in paths or request.method != "POST":
            raise web.HTTPNotFound()
        body = await request.json()
        counter += 1
        record = {
            "id": counter,
            "time": time.time(),
            "method": request.method,
            "path": request.path,
            "key": request.headers.get("X-API-Key", request.headers.get("Authorization", "")),
            "source_pod": request.headers.get("X-Source-Pod", request.remote),
            "body": body,
            "status": 200,
            "completed": False,
            "events": [],
        }
        records.append(record)
        protocol = paths[request.path]
        effective = body
        if protocol == "responses":
            items = body.get("input", [])
            items = [{"role": "user", "content": items}] if isinstance(items, str) else items
            if previous := body.get("previous_response_id"):
                prior = responses.get((record["key"], previous))
                if prior is None:
                    record.update(status=404, completed=True)
                    return web.json_response(
                        {"error": {"message": "Unknown previous_response_id"}}, status=404
                    )
                items = [*prior, *items]
            record["effective_input"] = items
            effective = {**body, "input": items}
        value = reply(protocol, effective, counter)
        record["response"] = value

        def remember():
            if protocol == "responses":
                responses[record["key"], value["id"]] = [
                    *record["effective_input"],
                    *value["output"],
                ]

        if not body.get("stream"):
            record["completed"] = True
            remember()
            return web.json_response(value)
        model = body.get("model", "mock")
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        compressor = None
        if "gzip" in model:
            headers["Content-Encoding"] = "gzip"
            compressor = zlib.compressobj(wbits=31)
        response = web.StreamResponse(headers=headers)
        await response.prepare(request)
        sequence = events(protocol, value)
        try:
            for index, event in enumerate(sequence):
                # Pause only after a complete SSE event has reached the client.
                if "silent" in model and index == 2:
                    record["pause_started"] = time.time()
                    await asyncio.sleep(40)
                    record["pause_ended"] = time.time()
                if "long" in model and index:
                    await asyncio.sleep(60 / (len(sequence) - 1))
                frame = "" if protocol == "chat" else f"event: {event['type']}\n"
                data = (frame + "data: " + json.dumps(event) + "\n\n").encode()
                if compressor:
                    data = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
                await response.write(data)
                record["events"].append({"time": time.time(), "type": event.get("type", "chunk")})
            if protocol == "chat":
                data = b"data: [DONE]\n\n"
                await response.write(compressor.compress(data) if compressor else data)
            if compressor:
                await response.write(compressor.flush())
            await response.write_eof()
            record["completed"] = True
            remember()
        except ConnectionResetError as error:
            record["disconnected"] = type(error).__name__
        except asyncio.CancelledError as error:
            record["disconnected"] = type(error).__name__
            raise
        finally:
            record["finished_at"] = time.time()
        return response

    app = web.Application(client_max_size=32 * 1024**2)
    app.router.add_route("*", "/{path:.*}", handle)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port, access_log=None)
