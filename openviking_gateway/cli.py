# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Standalone gateway entry point; it never reads the server's ov.conf."""

import argparse
import json
import logging
import os
from importlib.util import find_spec
from pathlib import Path

from .config import PUBLIC_OPENVIKING_URLS, StandaloneGatewayConfig

PREFIX = "OPENVIKING_GATEWAY_"


def load_config() -> StandaloneGatewayConfig:
    path = os.environ.get(PREFIX + "CONFIG_FILE")
    raw = json.loads(Path(path).expanduser().read_text()) if path else {}
    for field in StandaloneGatewayConfig.model_fields:
        if (value := os.environ.get(PREFIX + field.upper())) is not None:
            raw[field] = value
    return StandaloneGatewayConfig.model_validate(raw)


def studio_directory() -> Path | None:
    if path := os.environ.get("OPENVIKING_WEB_STUDIO_DIR"):
        return Path(path).expanduser()
    spec = find_spec("openviking")
    if spec and spec.submodule_search_locations:
        path = Path(next(iter(spec.submodule_search_locations))) / "web_studio" / "dist"
        if path.is_dir():
            return path
    return None


def create_standalone_app():
    from .app import create_app

    config = load_config()
    if not config.management_key and config.openviking_url in PUBLIC_OPENVIKING_URLS:
        logging.getLogger(__name__).warning(
            "public_openviking_without_management_key: This gateway is open to every "
            "OpenViking Cloud user. Set OPENVIKING_GATEWAY_MANAGEMENT_KEY to restrict "
            "changes to upstreams and profiles."
        )
    return create_app(config, openviking_url=config.openviking_url, studio_dir=studio_directory())


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

    uvicorn.run(
        "openviking_gateway.cli:create_standalone_app",
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
