"""Wire shapes reproduced with the installed Pi 0.85.1 openai-completions adapter."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepseek_fixtures import capable_account_mock
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from freetokenapi import tools
from freetokenapi.api import openai as api


@pytest.fixture
def pi_backend(monkeypatch):
    calls = []

    async def backend(req):
        calls.append(req)
        response = {
            "id": "chatcmpl-pi-test", "object": "chat.completion", "created": 0, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello Pi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        if not req.stream:
            return response

        async def events():
            yield "data: " + json.dumps({**response, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello Pi"}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({**response, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    monkeypatch.setattr(api, "_chat_completions_qwen", backend)
    monkeypatch.setattr(api, "_chat_completions_deepseek", backend)
    return TestClient(api.app), calls


def pi_body(**overrides):
    # Default custom provider + reasoning:true -> developer, even for Qwen/DS.
    return {
        "model": "qwen3.8-max", "stream": True, "store": False,
        "messages": [{"role": "developer", "content": "Pi system instructions"}, {"role": "user", "content": "Hello"}],
        "stream_options": {"include_usage": True}, "max_completion_tokens": 256,
        "reasoning_effort": "medium", **overrides,
    }


@pytest.mark.parametrize("model", ["qwen3.8-max", "deepseek-web-thinking"])
@pytest.mark.parametrize("stream", [False, True])
def test_pi_default_provider_developer_message(pi_backend, model, stream):
    client, calls = pi_backend
    response = client.post("/v1/chat/completions", json=pi_body(model=model, stream=stream))
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert [m.role for m in calls[-1].messages] == ["system", "user"]
    assert calls[-1].messages[0].content == "Pi system instructions"
    assert calls[-1].thinking is True
    if stream:
        assert "text/event-stream" in response.headers["content-type"]
        assert '"finish_reason": "stop"' in response.text
        assert "data: [DONE]" in response.text
    else:
        assert response.json()["choices"][0]["message"]["content"] == "Hello Pi"


def test_pi_developer_keeps_position_and_existing_session_policy(pi_backend):
    client, calls = pi_backend
    history = [
        {"role": "developer", "content": "Prefix"},
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "developer", "content": "Later instruction"},
        {"role": "user", "content": "Follow-up"},
    ]
    response = client.post("/v1/chat/completions", json=pi_body(messages=history, session_id="old-session"))
    assert response.status_code == 200
    req = calls[-1]
    assert [m.role for m in req.messages] == ["system", "user", "assistant", "system", "user"]
    assert [m.content for m in req.messages] == [m["content"] for m in history]
    assert tools.has_interleaved_system(req.messages)
    assert req.session_id == "old-session"


@pytest.mark.parametrize("options,expected", [
    ({"thinking": {"type": "enabled"}}, True),
    ({"thinking": {"type": "disabled"}}, False),
    ({"thinking": {"type": "enabled", "clear_thinking": False}}, True),
    ({"thinking": {"type": "adaptive", "budget_tokens": 1024}}, True),
    ({"enable_thinking": True}, True),
    ({"enable_thinking": False}, False),
    ({"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}, False),
    ({"reasoning_effort": "none"}, False),
    ({"reasoning_effort": "high"}, True),
    ({"thinking": False, "enable_thinking": True, "reasoning_effort": "high"}, False),
    ({"thinking": {"type": "disabled"}, "enable_thinking": True}, False),
])
def test_pi_thinking_wire_formats(pi_backend, options, expected):
    client, calls = pi_backend
    response = client.post("/v1/chat/completions", json=pi_body(**options))
    assert response.status_code == 200
    assert calls[-1].thinking is expected


@pytest.mark.parametrize("invalid", [
    {"messages": [{"role": "PRIVATE_BAD_ROLE", "content": "PRIVATE_PROMPT"}]},
    {"thinking": {"type": "PRIVATE_BAD_TYPE", "extra": "PRIVATE_PROMPT"}},
    {"thinking": {"type": "enabled", "budget_tokens": -1}},
    {"thinking": {"type": "enabled", "clear_thinking": "PRIVATE_PROMPT"}},
    {"thinking": ["PRIVATE_PROMPT"]},
    {"thinking": {"type": []}},
    {"thinking": {"type": {}}},
    {"chat_template_kwargs": {"enable_thinking": "PRIVATE_PROMPT"}},
    {"reasoning_effort": "PRIVATE_PROMPT"},
    {"messages": "PRIVATE_PROMPT"},
])
def test_pi_validation_errors_are_openai_errors_not_empty_422(pi_backend, invalid):
    client, calls = pi_backend
    response = client.post("/v1/chat/completions", json=pi_body(**invalid))
    assert response.status_code == 400
    assert response.headers["x-request-id"]
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"] and error["param"]
    assert "PRIVATE_" not in response.text
    assert "detail" not in response.json()
    assert not calls


def test_pi_unknown_model_has_openai_error_body(monkeypatch):
    monkeypatch.setattr(api.app.state, "qwen_models", [], raising=False)
    response = TestClient(api.app).post("/v1/chat/completions", json=pi_body(model="unsupported-test-model"))
    assert response.status_code == 404
    assert "Unknown model" in response.json()["error"]["message"]


def test_pi_upstream_error_has_openai_error_body(monkeypatch):
    async def fail(req):
        raise HTTPException(429, {"error": {"message": "Please retry later"}}, headers={"Retry-After": "5"})
    monkeypatch.setattr(api, "_chat_completions_qwen", fail)
    response = TestClient(api.app).post("/v1/chat/completions", json=pi_body())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["message"] == "Please retry later"


@pytest.mark.parametrize("stream", [False, True])
def test_pi_tool_history_and_null_assistant_content(pi_backend, stream):
    client, calls = pi_backend
    history = pi_body()["messages"] + [
        {"role": "assistant", "content": None, "reasoning_content": "Replayed annotation", "tool_calls": [
            {"id": "call_pi_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"README.md"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_pi_1", "content": "Tool output"},
    ]
    response = client.post("/v1/chat/completions", json=pi_body(stream=stream, messages=history, tools=[]))
    assert response.status_code == 200
    assert calls[-1].messages[-2].content is None
    assert calls[-1].messages[-2].tool_calls[0]["function"]["name"] == "read"
    assert calls[-1].messages[-1].tool_call_id == "call_pi_1"
    assert calls[-1].messages[-1].content == "Tool output"


@pytest.mark.parametrize("protocol", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("model", ["qwen3.8-max", "deepseek-web"])
@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_pi_protocols_share_native_search_default(monkeypatch, protocol, model, default, explicit):
    monkeypatch.setattr(api.settings, "search_enabled", default)
    monkeypatch.setattr(api.app.state, "pool", MagicMock(), raising=False)
    monkeypatch.setattr(api.app.state, "qwen_pool", MagicMock(), raising=False)
    account = capable_account_mock()
    api.app.state.pool.healthy = [account]
    monkeypatch.setattr(api, "_acquire_and_build", AsyncMock(return_value=(account, None, (), "Hello", False)))
    calls = []

    async def collect(**kwargs):
        calls.append(kwargs)
        return {"id": "chatcmpl-search", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    monkeypatch.setattr(api, "_collect_non_stream", collect)
    monkeypatch.setattr(api.qwen_api, "collect_non_stream", collect)
    body = {"model": model, "stream": False}
    if protocol == "responses":
        body.update(input="Hello")
    else:
        body.update(messages=[{"role": "user", "content": "Hello"}])
    if protocol == "messages":
        body["max_tokens"] = 64
    if explicit is not None:
        body["search"] = explicit
    response = TestClient(api.app).post("/v1/" + protocol, json=body)
    assert response.status_code == 200, response.text
    assert calls[-1]["search"] is (default if explicit is None else explicit)
