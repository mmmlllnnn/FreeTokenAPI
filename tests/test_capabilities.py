from __future__ import annotations

import json

import httpx
import pytest

from freetokenapi.capabilities import NATIVE_SEARCH_HINT, native_search_prompt
from freetokenapi.deepseek.client import DeepSeekClient
from freetokenapi.qwen.client import QwenClient


def test_native_search_has_standing_permission_beyond_current_information():
    assert "standing permission" in NATIVE_SEARCH_HINT
    assert "autonomously whenever it can help complete the task" in NATIVE_SEARCH_HINT
    assert "without waiting for an explicit search request or asking for per-search approval" in NATIVE_SEARCH_HINT
    assert "not limited to time-sensitive questions" in NATIVE_SEARCH_HINT
    assert "repeat searches as needed" in NATIVE_SEARCH_HINT
    assert "When current online information is needed" not in NATIVE_SEARCH_HINT


def test_native_search_keeps_privacy_client_permissions_and_truthfulness():
    assert "explicit no-search instructions and privacy constraints" in NATIVE_SEARCH_HINT
    assert "permissions for client-side tools remain unchanged" in NATIVE_SEARCH_HINT
    assert "Do not invent client search tool calls, search results, or sources" in NATIVE_SEARCH_HINT
    assert "only claim to have searched when the backend actually performed a search" in NATIVE_SEARCH_HINT


@pytest.mark.parametrize("prompt", ["Explain this concept", "Do not browse. Use only the supplied text.", ""])
@pytest.mark.parametrize("enabled", [False, True])
def test_search_hint_preserves_prompt_and_off_switch(prompt, enabled):
    expected = NATIVE_SEARCH_HINT + "\n\n" + prompt if enabled else prompt
    assert native_search_prompt(prompt, enabled) == expected


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_native_search_policy_reaches_both_web_backends(provider, enabled):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, content="data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    client = DeepSeekClient() if provider == "deepseek" else QwenClient()
    await client.http.aclose()
    base_url = "https://chat.deepseek.com" if provider == "deepseek" else "https://chat.qwen.ai"
    client.http = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
    prompt = "Verify the design assumptions and compare alternatives."
    try:
        if provider == "deepseek":
            response = await client.completion("test-session", prompt, None, search_enabled=enabled)
            actual_prompt = captured[-1]["prompt"]
            actual_search = captured[-1]["search_enabled"]
        else:
            response = await client.completion("test-session", prompt, None, "qwen3.8-max", search=enabled)
            message = captured[-1]["messages"][0]
            actual_prompt = message["content"]
            actual_search = message["feature_config"]["auto_search"]
        await response.aclose()
        assert actual_search is enabled
        assert actual_prompt == native_search_prompt(prompt, enabled)
    finally:
        await client.aclose()
