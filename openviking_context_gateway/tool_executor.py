# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bounded, user-authenticated MCP calls and signed file upload handoff."""

import asyncio
import base64
import binascii
import re
import time
import uuid
from pathlib import PurePosixPath
from urllib.parse import parse_qs, quote, urlsplit

import async_timeout
import orjson

from .capture_store import Document
from .client import VikingError
from .state_store import get_state
from .storage import digest
from .tool_catalog import CATALOG, PREFIX, attachments, has_shell

UPLOAD_URL = re.compile(
    r'https?://[^\s<>"\x27`]+/api/v1/resources/temp_upload\?token=[^\s<>"\x27`]+'
)


def upload_token(result):
    for block in result.get("content", []):
        if block.get("type") == "text":
            match = UPLOAD_URL.search(block.get("text", ""))
            if match:
                return parse_qs(urlsplit(match[0]).query).get("token", [""])[0]
    return ""


def attachment_bytes(part, limit):
    name = PurePosixPath(part.get("filename", "attachment.txt")).name
    if not name or name in {".", ".."}:
        name = "attachment.txt"
    if isinstance(part.get("text"), str):
        data = part["text"].encode()
    else:
        value = part.get("file_data", "")
        if not isinstance(value, str) or not value:
            raise ValueError(
                "Attachment bytes are unavailable; file IDs and remote URLs cannot be imported"
            )
        if value.startswith("data:"):
            header, value = value.split(",", 1)
            if not header.endswith(";base64"):
                raise ValueError("Expected a base64 attachment")
        if len(value) > (limit + 2) // 3 * 4:
            raise ValueError("Attachment exceeds upload limit")
        data = base64.b64decode(value, validate=True)
    if len(data) > limit:
        raise ValueError("Attachment exceeds upload limit")
    return name, data


class ToolExecutor:
    def __init__(self, viking, store, prepared, credential, public_url, max_upload_bytes):
        self.viking, self.store, self.request = viking, store, prepared
        self.key = credential["openviking_key"]
        self.public_url, self.max_upload_bytes = public_url.rstrip("/"), max_upload_bytes
        self.policy = prepared.root["policy"]
        self.allowed = {t["function"]["name"] for t in prepared.root.get("tools", [])}

    async def execute(self, call):
        name = call.get("function", {}).get("name", "")
        call_id = call.get("id", "")
        result = {"role": "tool", "tool_call_id": call_id}
        if name not in self.allowed or not call_id:
            return {**result, "content": '{"error":"Tool is not allowed"}'}
        # Retries sharing a history and call ID share the claim. A process crash
        # never causes automatic repetition of a potentially committed write.
        anchor = digest(
            (self.request.chain[-1] if self.request.chain else "") + orjson.dumps(call).decode()
        )
        owner = uuid.uuid4().hex
        scope, session = self.request.scope, self.request.session
        receipt_key = "tool:" + session + ":" + anchor
        empty = Document()
        claim = {"owner": owner, "time": time.time()}
        won = await self.store.state.swap(scope, receipt_key, empty, claim)
        receipt = (
            Document(claim, 1) if won else await get_state(self.store.state, scope, receipt_key)
        )
        claim = receipt.value
        timeout = self.policy.get("tool_timeout_seconds", 30)
        try:
            async with async_timeout.timeout(timeout):
                if claim["owner"] != owner:
                    while True:
                        saved = await get_state(self.store.state, scope, receipt_key)
                        if "content" in saved.value:
                            return {**result, "content": saved.value["content"]}
                        if time.time() - claim["time"] > timeout:
                            raise asyncio.TimeoutError
                        await asyncio.sleep(0.05)
                short = name.removeprefix(PREFIX)
                args = (
                    CATALOG[short][0]
                    .model_validate_json(call["function"]["arguments"])
                    .model_dump(exclude_none=True)
                )
                value = await self._call(short, args)
                encoded = orjson.dumps(value)
                limit = self.policy.get("tool_result_bytes", 65536)
                if len(encoded) > limit:
                    # Bound the serialized result too, including escaping overhead.
                    text = encoded[:limit].decode("utf-8", errors="ignore")
                    while len(orjson.dumps({"truncated": True, "text": text})) > limit:
                        text = text[: len(text) * 3 // 4]
                    value = {"truncated": True, "text": text}
                content = orjson.dumps(value).decode()
                await self.store.state.swap(
                    scope, receipt_key, receipt, {**claim, "content": content}
                )
                return {**result, "content": content}
        except asyncio.TimeoutError:
            content = '{"error":"Tool timed out or a previous attempt has an unknown outcome; inspect state before retrying writes"}'
        except (ValueError, binascii.Error, VikingError, KeyError):
            # Pydantic errors can echo complete arguments and secrets.
            content = '{"error":"Invalid tool arguments or OpenViking operation failed"}'
        if claim["owner"] == owner:
            await self.store.state.swap(scope, receipt_key, receipt, {**claim, "content": content})
        return {**result, "content": content}

    async def _call(self, name, args):
        index = args.pop("attachment_index", None)
        file = None
        if index is not None:
            parts = attachments(self.request.original)
            if index >= len(parts):
                raise ValueError("Attachment is unavailable")
            file = attachment_bytes(parts[index], self.max_upload_bytes)
            # Only a synthetic local filename is sent. The gateway never opens it.
            args["path"] = "/client-upload/" + file[0]
            if name == "add_skill":
                args.pop("data", None)
        elif name in {"add_resource", "add_skill"} and args.get("path"):
            path = args["path"]
            if not path.startswith(("https://", "http://")) and not has_shell(
                self.request.original
            ):
                raise ValueError("A local path requires a client shell or attachment")
        value = await self.viking.mcp("find" if name == "search" else name, self.key, args)
        token = upload_token(value) if name in {"add_resource", "add_skill"} else ""
        if file:
            if not token or value.get("isError"):
                raise VikingError("upload_instruction_missing")
            return await self.viking.upload(token, file[0], file[1])
        if token:
            if not self.public_url:
                raise ValueError("Set context_gateway.public_url for client uploads")
            url = self.public_url + "/context-gateway/uploads?token=" + quote(token, safe="")
            for block in value.get("content", []):
                if block.get("type") == "text":
                    block["text"] = UPLOAD_URL.sub(lambda _: url, block["text"])
        return value
