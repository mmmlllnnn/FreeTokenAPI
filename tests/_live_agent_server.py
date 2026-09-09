"""Opt-in local live-test server; isolated state, private port, no saved settings."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
from pathlib import Path


def main():
    from dotenv import dotenv_values

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("deepseek", "qwen"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    values = dotenv_values(root / ".env")
    key = "DEEPSEEK_TOKENS" if args.provider == "deepseek" else "QWEN_TOKENS"
    raw = os.environ.get(key, values.get(key) or "")
    token = next((part.strip() for part in raw.split(",") if part.strip()), "")
    if not token:
        raise RuntimeError(f"No {args.provider} web credential is configured")
    os.environ.update({
        "PYTHON_DOTENV_DISABLED": "1", "DEEPSEEK_TOKENS": "", "QWEN_TOKENS": "",
        "FREETOKENAPI_LOG_FILE": "", "FREETOKENAPI_CACHE_DISABLED": "1", "FREETOKENAPI_USAGE_ENABLED": "0",
        "FREETOKENAPI_HUMAN_DELAY_MIN": "2", "FREETOKENAPI_HUMAN_DELAY_MAX": "4",
        "FREETOKENAPI_SEARCH_ENABLED": "0", "FREETOKENAPI_TIMEOUT": "90",
    })
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(root))
    import uvicorn

    from freetokenapi.accounts import AccountPool, DeepSeekAccount
    from freetokenapi.api import openai as api
    from freetokenapi.deepseek.client import DeepSeekClient
    from freetokenapi.qwen.accounts import QwenAccount
    from freetokenapi.qwen.client import QwenClient

    async def run():
        client = DeepSeekClient(token, timeout=90) if args.provider == "deepseek" else QwenClient(token, timeout=90)
        account = DeepSeekAccount(0, client) if args.provider == "deepseek" else QwenAccount(0, client)
        pool = AccountPool([account], label=args.provider)
        api.app.state.pool = pool if args.provider == "deepseek" else None
        api.app.state.qwen_pool = pool if args.provider == "qwen" else None
        api.app.state.qwen_models = []
        api.app.state.usage = None
        trace = {"upstream_requests": 0, "web_conversations": 0}
        sessions = set()
        completion = client.completion

        async def traced(*args, **kwargs):
            trace["upstream_requests"] += 1
            session = kwargs.get("chat_session_id") or kwargs.get("chat_id")
            if session:
                sessions.add(session)
                trace["web_conversations"] = len(sessions)
            return await completion(*args, **kwargs)

        client.completion = traced

        @api.app.get("/__live_test_trace")
        async def get_trace():
            return trace

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        server = uvicorn.Server(uvicorn.Config(api.app, log_level="critical", access_log=False, lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Test server did not start")
                await asyncio.sleep(0.01)
            print(json.dumps({"port": sock.getsockname()[1]}), flush=True)
            await asyncio.to_thread(sys.stdin.readline)
        finally:
            server.should_exit = True
            await task
            sock.close()
            await client.aclose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
