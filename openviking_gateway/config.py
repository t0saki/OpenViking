# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Configuration for embedded and standalone gateway hosts."""

import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PUBLIC_OPENVIKING_URLS = frozenset({"https://api.vikingdb.cn-beijing.volces.com/openviking"})


def normalize_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("expected an HTTP(S) URL without credentials, query or fragment")
    # Validate ports and canonicalize equivalent public service addresses.
    port = parsed.port
    host = parsed.hostname.lower()
    host = f"[{host}]" if ":" in host else host
    if port is not None and (parsed.scheme, port) not in {("https", 443), ("http", 80)}:
        host += f":{port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    public_url: str = ""
    storage_path: str = "~/.openviking/gateway"
    encryption_key_env: str = "OPENVIKING_GATEWAY_ENCRYPTION_KEY"
    management_key: str = Field(default="", repr=False)
    min_server_version: str = "0.4.16"
    session_ttl_days: int = Field(default=30, ge=1)
    response_ttl_seconds: int = Field(default=30 * 86400, ge=60)
    log_retention_days: int = Field(default=30, ge=1)
    max_body_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)
    upstream_timeout_seconds: float = Field(default=600, gt=0)
    health_interval_seconds: float = Field(default=60, ge=1)

    @field_validator("public_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return normalize_url(value) if value else ""

    @field_validator("management_key")
    @classmethod
    def validate_management_key(cls, value: str) -> str:
        value = os.path.expandvars(value)
        if value and (len(value) < 32 or value.startswith("$")):
            raise ValueError("management_key must resolve to at least 32 characters")
        return value

    @property
    def directory(self) -> Path:
        return Path(self.storage_path).expanduser()

    def encryption_key(self) -> str:
        """Publish one durable key before opening SQLite, even with multiple workers."""
        if key := os.environ.get(self.encryption_key_env):
            Fernet(key.encode())
            return key
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = directory / "encryption.key"
        if not target.exists():
            if (
                any(
                    (directory / name).exists() for name in ("kernel.sqlite3", "management.sqlite3")
                )
                and not target.exists()
            ):
                raise ValueError(
                    "Gateway database exists but encryption.key is missing; restore the original "
                    "key file or configure the original encryption key environment variable"
                )
            fd, temporary = tempfile.mkstemp(prefix=".encryption-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(Fernet.generate_key())
                    file.flush()
                    os.fsync(file.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
        key = target.read_text().strip()
        Fernet(key.encode())
        return key


class EmbeddedGatewayConfig(GatewayConfig):
    enabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_removed_fields(cls, value):
        removed = {"host", "port", "workers", "url", "openviking_url", "admin_token_env"}
        if isinstance(value, dict) and (found := removed.intersection(value)):
            raise ValueError(
                f"Removed embedded gateway settings: {', '.join(sorted(found))}. "
                "See docs/en/guides/22-gateway-operations.md for migration instructions"
            )
        return value


class StandaloneGatewayConfig(GatewayConfig):
    openviking_url: str = Field(min_length=1)
    host: str = "127.0.0.1"
    port: int = Field(default=1935, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=64)

    @field_validator("openviking_url")
    @classmethod
    def validate_openviking_url(cls, value: str) -> str:
        return normalize_url(value)
