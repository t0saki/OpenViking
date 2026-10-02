# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
import json
import shlex


def connection_guides(base):
    return {
        "Claude Code": f"export ANTHROPIC_BASE_URL={shlex.quote(base)}\nexport ANTHROPIC_AUTH_TOKEN='<gateway-key>'\nexport CLAUDE_CODE_GATEWAY_HINT_HEADERS=1",
        "Codex CLI": '[model_providers.openviking]\nname = "OpenViking Context Gateway"\n'
        f'base_url = {json.dumps(base + "/v1")}\nwire_api = "responses"\n'
        'env_key = "OPENVIKING_GATEWAY_KEY"\n# Set model_provider = "openviking" at the TOML root.\n'
        "# Send full history with store=false; WebSocket requests receive HTTP 426.",
        "Chat client": f"Base URL: {base}/v1\nAPI key: <gateway-key>\nHeader for capture, takeover and hidden tools: X-OpenViking-Session: <conversation-id>",
        "Open WebUI": f"Base URL: {base}/v1\nAPI key: <gateway-key>\nHeaders:\n"
        "X-OpenViking-Session: {{CHAT_ID}}\nX-OpenViking-Task: {{TASK}}\n"
        "Enable RAG_SYSTEM_CONTEXT. One connection key maps every frontend user to one OpenViking user.",
        "OpenCode": json.dumps(
            {
                "provider": {
                    "openviking": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "OpenViking",
                        "options": {
                            "baseURL": base + "/v1",
                            "apiKey": "{env:OPENVIKING_GATEWAY_KEY}",
                        },
                        "models": {"<model>": {}},
                    }
                }
            },
            indent=2,
        ),
        "pi": json.dumps(
            {
                "providers": {
                    "openviking": {
                        "baseUrl": base + "/v1",
                        "apiKey": "OPENVIKING_GATEWAY_KEY",
                        "api": "openai-completions",
                        "models": [{"id": "<model>"}],
                    }
                }
            },
            indent=2,
        ),
    }
