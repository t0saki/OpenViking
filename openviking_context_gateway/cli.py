# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
import argparse
import json
import os
from pathlib import Path

from .config import ContextGatewayConfig


def load_config():
    path = Path(os.environ.get("OPENVIKING_CONFIG_FILE", "~/.openviking/ov.conf")).expanduser()
    raw = json.loads(path.read_text()) if path.exists() else {}
    config = ContextGatewayConfig.model_validate(raw.get("context_gateway", {}))
    if not config.enabled:
        raise ValueError("Set context_gateway.enabled=true in ov.conf")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="OpenViking Context Gateway (separate from VikingBot Gateway)"
    )
    parser.add_argument("--config", help="Path to ov.conf")
    args = parser.parse_args()
    if args.config:
        os.environ["OPENVIKING_CONFIG_FILE"] = str(Path(args.config).resolve())
    config = load_config()
    import uvicorn

    uvicorn.run(
        "openviking_context_gateway.app:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level="info",
        # Signed upload tokens live in query strings; use metadata-only gateway logs.
        access_log=False,
    )


if __name__ == "__main__":
    main()
