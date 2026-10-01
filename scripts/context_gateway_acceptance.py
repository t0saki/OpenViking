#!/usr/bin/env python3
"""Opt-in live smoke test. Never reads or prints credentials from config files."""

import argparse
import asyncio
import os
import uuid

import aiohttp


async def run(protocol):
    base, key, model = (
        os.environ[name] for name in ("OV_CG_TEST_BASE_URL", "OV_CG_TEST_KEY", "OV_CG_TEST_MODEL")
    )
    paths = {
        "anthropic": "/v1/messages",
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
    }
    headers = {
        "Authorization": "Bearer " + key,
        "X-OpenViking-Session": "acceptance-" + uuid.uuid4().hex,
    }
    history = []
    async with aiohttp.ClientSession() as client:
        for index, prompt in enumerate(
            (
                "Remember the test project uses a blue deployment cluster. Reply briefly.",
                "Which deployment cluster does the test project use?",
                "Repeat the cluster name and nothing else.",
            ),
            1,
        ):
            history.append({"role": "user", "content": prompt})
            body = {
                "model": model,
                "stream": False,
                "input" if protocol == "responses" else "messages": history,
            }
            if protocol == "anthropic":
                headers.update(
                    {
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "thinking-binding-controls-2026-08-01",
                    }
                )
                body.update(
                    max_tokens=2048,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024,
                        "block_binding": {"prefix_mismatch_behavior": "error"},
                    },
                )
            elif protocol == "responses":
                body["store"] = False
            async with client.post(
                base.rstrip("/") + paths[protocol], json=body, headers=headers
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"turn {index}: HTTP {response.status}; inspect provider request logs"
                    )
                result = await response.json()
                if result.get("input_transformations"):
                    raise RuntimeError(f"turn {index}: provider transformed signed input")
                if protocol == "anthropic":
                    history.append({"role": "assistant", "content": result["content"]})
                elif protocol == "responses":
                    history.extend(result["output"])
                else:
                    history.append(result["choices"][0]["message"])
                print(
                    {
                        "turn": index,
                        "status": response.status,
                        "usage": result.get("usage", {}),
                        "input_transformations": result.get("input_transformations", []),
                    }
                )
    print(
        "Live smoke passed. Inspect cache coverage and run real-client/long-context acceptance separately."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["anthropic", "chat", "responses"], required=True)
    asyncio.run(run(parser.parse_args().protocol))
