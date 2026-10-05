# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProtocolName = Literal["anthropic", "chat", "responses"]


class CaptureReset(BaseModel):
    session: str = Field(min_length=1, max_length=128)
    protocol: ProtocolName


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "Default"
    recall: bool = True
    capture: bool = True
    context_types: list[str] = Field(default_factory=lambda: ["memory", "resource", "skill"])
    quotas: dict[str, int] = Field(default_factory=dict)
    max_tokens: int = Field(default=1600, ge=64, le=32000)
    session_max_tokens: int = Field(default=6000, ge=0)
    score_threshold: float = Field(default=0.35, ge=0, le=1)
    recall_timeout: float = Field(default=2, gt=0, le=30)
    query_max_chars: int = Field(default=8000, ge=3, le=32000)
    commit_tokens: int = Field(default=20000, ge=1)
    keep_recent_messages: int = Field(default=10, ge=0, le=1000)
    idle_seconds: float = Field(default=600, ge=1)
    takeover: bool = True
    takeover_tokens: int = Field(default=30000, ge=1)
    keep_recent_turns: int = Field(default=3, ge=1, le=100)
    # Optional fallback for deployments serving one known model.
    context_window: int | None = Field(default=None, ge=1024)
    archive_wait_seconds: float = Field(default=30, ge=0, le=60)
    gateway_tools: bool = False
    allow_write_tools: bool = False
    tool_allowlist: list[
        Literal["search", "read", "list", "write", "add_resource", "add_skill"]
    ] = Field(default_factory=lambda: ["search", "read", "list"])
    tool_max_rounds: int = Field(default=5, ge=1, le=20)
    tool_timeout_seconds: float = Field(default=30, gt=0, le=120)
    tool_result_bytes: int = Field(default=65536, ge=1024, le=1048576)
    tool_total_seconds: float = Field(default=120, gt=0, le=600)
    tool_total_tokens: int = Field(default=100000, ge=1024, le=1000000)
    show_tool_calls: bool = True


class Upstream(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    protocol: ProtocolName
    base_url: str
    api_key: str = ""
    auth_mode: Literal["managed", "passthrough"] = "managed"
    headers: dict[str, str] = Field(default_factory=dict)
    models: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    priority: int = 0
    enabled: bool = True
    vendor: Literal["generic", "anthropic", "openai", "deepseek", "ark", "byteplus"] = "generic"
    allow_gateway_tools: bool = True
    coding_plan: bool = False
    allow_coding_plan: bool = False
    cache_min_tokens: int = Field(default=1024, ge=0)
    context_windows: dict[str, int] = Field(default_factory=dict)

    @field_validator("context_windows")
    @classmethod
    def validate_windows(cls, value: dict[str, int]) -> dict[str, int]:
        if any(size < 1024 for size in value.values()):
            raise ValueError("model context windows must be at least 1024 tokens")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "expected an HTTP(S) upstream URL without credentials, query or fragment"
            )
        return value.rstrip("/")

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if (
                "\r" in key + item
                or "\n" in key + item
                or key.lower() in {"host", "content-length", "transfer-encoding", "connection"}
            ):
                raise ValueError("invalid upstream header")
        return value


class KeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    openviking_key: str
    policy_id: str
    upstream_ids: list[str] = Field(min_length=1)
    models: list[str] = Field(default_factory=list)
