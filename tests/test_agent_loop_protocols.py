"""Exercise real conversion/session/streaming code with a deterministic web backend."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from freetokenapi.accounts import AccountPool, DeepSeekAccount
from freetokenapi.api import openai as api
from freetokenapi.deepseek.client import DeepSeekSession
from freetokenapi.qwen import api as qwen_api
from freetokenapi.qwen.accounts import QwenAccount

TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {key: {"type": "string"} for key in keys},
    }}}
    for name, keys in [("Read", ["file_path"]), ("Write", ["file_path", "content"])]
]
REPLIES = [
    '{"tool_calls":[{"name":"Read","arguments":{"file_path":"source.txt"}}]}',
    '[assistant called Write({"file_path":"report.txt","content":"verified"})]',
    '[调用 Read] {"file_path":"report.txt"}',
    'Done. The written result has been verified.',
]
PATHS = {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}


def frames(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def web_response(provider, text, number, interrupted=False, partial_error=False):
    if provider == "deepseek":
        data = {"v": {"response": {"message_id": number, "status": "WIP" if interrupted else "FINISHED", "fragments": [{"type": "RESPONSE", "content": text}]}}}
        events = [data]
    else:
        events = [{"response_id": str(number), "choices": [{"delta": {"phase": "answer", "status": "typing", "content": text}}]}]
        if not interrupted:
            events.append({"choices": [{"delta": {"phase": "answer", "status": "finished", "content": ""}}]})
        if partial_error:
            events.append({"error": {"code": "Internal_Server_Error", "details": "provider failed after partial output"}})
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def setup_backend(monkeypatch, provider, replies=REPLIES, interrupted=False, partial_error=False):
    prompts = []

    async def completion(**kwargs):
        index = len(prompts)
        prompts.append(kwargs["prompt"])
        return web_response(provider, replies[index], index + 1, interrupted, partial_error)

    upstream = SimpleNamespace(
        completion=AsyncMock(side_effect=completion),
        create_session=AsyncMock(return_value=DeepSeekSession(id="web-task")),
        create_chat=AsyncMock(return_value="web-task"), create_pow_challenge=AsyncMock(return_value={}),
        stop_stream=AsyncMock(), rename_session=AsyncMock(), update_chat=AsyncMock(),
    )
    account = DeepSeekAccount(0, upstream) if provider == "deepseek" else QwenAccount(0, upstream)
    if provider == "deepseek":
        account.pow.make_header = AsyncMock(return_value={})
    pool = AccountPool([account], label=provider)
    monkeypatch.setattr(api.app.state, "pool", pool if provider == "deepseek" else None, raising=False)
    monkeypatch.setattr(api.app.state, "qwen_pool", pool if provider == "qwen" else None, raising=False)
    monkeypatch.setattr(api.app.state, "qwen_models", [], raising=False)
    monkeypatch.setattr(api.app.state, "usage", None, raising=False)
    monkeypatch.setattr(api, "_human_delay", AsyncMock())
    monkeypatch.setattr(qwen_api, "_human_delay", AsyncMock())
    return TestClient(api.app), upstream, prompts


def request_body(protocol, provider, stream):
    model = "deepseek-v4-flash" if provider == "deepseek" else "qwen3.8-max"
    body = {"model": model, "stream": stream}
    initial = [{"role": "user", "content": "Read source.txt, write report.txt, then read it to verify the saved content."}]
    if protocol == "messages":
        body.update(max_tokens=4096, messages=initial, tools=[{"name": t["function"]["name"], "input_schema": t["function"]["parameters"]} for t in TOOLS])
    elif protocol == "responses":
        body.update(input=initial, tools=[{"type": "function", **t["function"]} for t in TOOLS])
    else:
        body.update(messages=initial, tools=TOOLS)
    return body


def read_reply(protocol, stream, response):
    if not stream:
        data = response.json()
    elif protocol == "responses":
        data = next(frame["response"] for frame in frames(response.text) if frame["type"] == "response.completed")
    elif protocol == "messages":
        data = {"content": [], "stop_reason": None}
        for frame in frames(response.text):
            kind = frame["type"]
            if kind == "content_block_start":
                data["content"].append(frame["content_block"])
            elif kind == "content_block_delta":
                block = data["content"][frame["index"]]
                delta = frame["delta"]
                if delta["type"] == "input_json_delta":
                    block["input"] = json.loads(delta["partial_json"])
                elif delta["type"] == "text_delta":
                    block["text"] += delta["text"]
            elif kind == "message_delta":
                data["stop_reason"] = frame["delta"]["stop_reason"]
    else:
        data = {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}, "finish_reason": None}]}
        choice = data["choices"][0]
        by_index = {}
        for frame in frames(response.text):
            if not frame.get("choices"):
                continue
            item = frame["choices"][0]
            delta = item.get("delta", {})
            choice["message"]["content"] += delta.get("content", "")
            for call in delta.get("tool_calls", []):
                target = by_index.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                target["id"] += call.get("id", "")
                target["function"]["name"] += call.get("function", {}).get("name", "")
                target["function"]["arguments"] += call.get("function", {}).get("arguments", "")
            if item.get("finish_reason"):
                choice["finish_reason"] = item["finish_reason"]
        choice["message"]["tool_calls"] = list(by_index.values())
    if protocol == "chat":
        choice = data["choices"][0]
        return choice["message"], choice["message"].get("tool_calls", []), choice["finish_reason"]
    if protocol == "messages":
        return {"role": "assistant", "content": data["content"]}, [b for b in data["content"] if b["type"] == "tool_use"], data["stop_reason"]
    return data["output"], [item for item in data["output"] if item["type"] == "function_call"], data["status"]


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
@pytest.mark.parametrize("protocol", list(PATHS))
@pytest.mark.parametrize("stream", [False, True])
def test_agent_runs_multiple_tools_without_user_continue(monkeypatch, provider, protocol, stream):
    client, upstream, prompts = setup_backend(monkeypatch, provider)
    body = request_body(protocol, provider, stream)
    for index in range(4):
        response = client.post(PATHS[protocol], json=body)
        assert response.status_code == 200, response.text
        message, calls, finish = read_reply(protocol, stream, response)
        assert '"file_path":{"type":"string"}' in prompts[-1]
        if index:
            assert f"CURRENT_RESULT_{index}" in prompts[-1]
            assert all(f"CURRENT_RESULT_{old}" not in prompts[-1] for old in range(1, index))
        if index == 3:
            assert not calls
            assert finish == {"chat": "stop", "messages": "end_turn", "responses": "completed"}[protocol]
            break
        assert len(calls) == 1
        assert finish == {"chat": "tool_calls", "messages": "tool_use", "responses": "completed"}[protocol]
        output = f"CURRENT_RESULT_{index + 1}"
        if protocol == "chat":
            body["messages"] += [message, {"role": "tool", "tool_call_id": calls[0]["id"], "content": output}]
        elif protocol == "messages":
            body["messages"] += [message, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"], "content": output}]}]
        else:
            body["input"] += message + [{"type": "function_call_output", "call_id": calls[0]["call_id"], "output": output}]
    assert len(prompts) == 4
    (upstream.create_session if provider == "deepseek" else upstream.create_chat).assert_awaited_once()


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
@pytest.mark.parametrize("protocol", list(PATHS))
@pytest.mark.parametrize("stream", [False, True])
def test_interrupted_response_never_dispatches_buffered_tools(monkeypatch, provider, protocol, stream):
    client, _, _ = setup_backend(monkeypatch, provider, replies=REPLIES[:1], interrupted=True)
    response = client.post(PATHS[protocol], json=request_body(protocol, provider, stream))
    if stream:
        assert response.status_code == 200
        data = frames(response.text)
        assert any("error" in frame or frame.get("type") == "response.failed" for frame in data)
        assert not any(frame.get("type") in {"response.completed", "message_stop"} for frame in data)
        assert '"type": "tool_use"' not in response.text
        assert '"finish_reason": "tool_calls"' not in response.text
    else:
        assert response.status_code == 502
        assert "error" in response.json()


@pytest.mark.parametrize("stream", [False, True])
def test_qwen_error_after_content_is_not_success(monkeypatch, stream):
    client, _, _ = setup_backend(monkeypatch, "qwen", replies=REPLIES[:1], partial_error=True)
    response = client.post(PATHS["messages"], json=request_body("messages", "qwen", stream))
    if stream:
        assert any(frame.get("type") == "error" for frame in frames(response.text))
        assert not any(frame.get("type") == "message_stop" for frame in frames(response.text))
    else:
        assert response.status_code == 502
