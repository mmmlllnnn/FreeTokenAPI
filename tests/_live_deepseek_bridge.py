"""Opt-in live smoke helper; no listener and no changes to the running service."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    from dotenv import dotenv_values

    root = Path(__file__).resolve().parents[1]
    incoming = json.load(sys.stdin)
    if incoming["body"].get("model") not in {"deepseek-web", "deepseek-web-thinking"}:
        raise ValueError("This live helper is restricted to DeepSeek Web fixture tests")
    config = dotenv_values(root / ".env")
    raw = os.environ.get("DEEPSEEK_TOKENS", config.get("DEEPSEEK_TOKENS") or "")
    token = next((value.strip() for value in raw.split(",") if value.strip()), "")
    os.environ.update({"PYTHON_DOTENV_DISABLED": "1", "DEEPSEEK_TOKENS": "", "QWEN_TOKENS": "", "FREETOKENAPI_LOG_FILE": "", "FREETOKENAPI_CACHE_DISABLED": "1", "FREETOKENAPI_USAGE_ENABLED": "0", "FREETOKENAPI_HUMAN_DELAY_MIN": "0", "FREETOKENAPI_HUMAN_DELAY_MAX": "0"})
    sys.path.insert(0, str(root))
    import httpx

    from freetokenapi.accounts import AccountPool, DeepSeekAccount
    from freetokenapi.api import openai as api
    from freetokenapi.deepseek.client import DeepSeekClient

    logging.disable(logging.CRITICAL)

    async def run():
        if not token:
            raise RuntimeError("No DeepSeek web credential is configured")
        client = DeepSeekClient(token, timeout=90)
        trace = {"proof_requests": 0, "file_statuses": []}
        original_challenge, original_wait = client.create_pow_challenge, client.wait_for_file

        async def challenge(*args, **kwargs):
            trace["proof_requests"] += 1
            return await original_challenge(*args, **kwargs)

        async def wait(info, **kwargs):
            record = {"initial": info.get("status")}
            trace["file_statuses"].append(record)
            result = await original_wait(info, **kwargs)
            record["final"] = result.get("status")
            return result

        client.create_pow_challenge, client.wait_for_file = challenge, wait
        try:
            api.app.state.pool = AccountPool([DeepSeekAccount(0, client)])
            api.app.state.qwen_pool = None
            api.app.state.qwen_models = []
            api.app.state.usage = None
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://adapter.test", timeout=180) as http:
                result = await http.post(incoming["path"], json=incoming["body"])
                print(json.dumps({"status": result.status_code, "headers": dict(result.headers), "body": result.text, "trace": trace}), flush=True)
        finally:
            await client.aclose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
