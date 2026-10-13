# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Standalone gateway entry point; it never reads the server's ov.conf."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import StandaloneGatewayConfig
from .standalone import check_openviking, create_standalone_app

PREFIX = "OPENVIKING_GATEWAY_"


def load_config() -> StandaloneGatewayConfig:
    path = os.environ.get(PREFIX + "CONFIG_FILE")
    raw = json.loads(Path(path).expanduser().read_text()) if path else {}
    for field in StandaloneGatewayConfig.model_fields:
        # An empty variable, such as a compose default, never overrides the file.
        if value := os.environ.get(PREFIX + field.upper()):
            raw[field] = value
    return StandaloneGatewayConfig.model_validate(raw)


def factory():
    """uvicorn factory: every worker rebuilds the configuration from the environment."""
    return create_standalone_app(load_config())


def main():
    parser = argparse.ArgumentParser(description="OpenViking Gateway")
    parser.add_argument("--config", help="Path to a standalone gateway.conf")
    parser.add_argument("--openviking-url", help="OpenViking server URL (required)")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--storage-path")
    args = parser.parse_args()
    if args.config:
        os.environ[PREFIX + "CONFIG_FILE"] = str(Path(args.config).resolve())
    for field in ("openviking_url", "host", "port", "workers", "storage_path"):
        if (value := getattr(args, field)) is not None:
            os.environ[PREFIX + field.upper()] = str(value)
    config = load_config()
    import uvicorn

    asyncio.run(check_openviking(config))
    uvicorn.run(
        "openviking_gateway.cli:factory",
        factory=True,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level="info",
        # Signed upload tokens live in query strings; log request metadata only.
        access_log=False,
    )


if __name__ == "__main__":
    main()
