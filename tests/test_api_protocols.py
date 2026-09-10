"""Native protocol tests; all backends are fake and no lifespan contacts accounts."""

import copy
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from freetokenapi.api.compat import (
    ChatRun,
    OutputLimit,
    ProtocolError,
    chat_chunks,
    dumps,
)
from freetokenapi.api.messages import create_messages_router
from freetokenapi.api.responses import ResponseStore, create_responses_router

MODEL = "deepseek-web"
TOOL = {"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
CLAUDE_TOOL = {"name": "weather", "input_schema": TOOL["parameters"]}
CALL = {"id": "call_weather", "type": "function", "function": {"name": "weather", "arguments": '{"city":"台北"}'}}


def parse_sse(text):
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        data = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
        if data:
            event = json.loads("\n".join(data))
            name = next(line[6:].strip() for line in frame.splitlines() if line.startswith("event:"))
            assert name == event["type"]
            events.append(event)
    return events


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.failure = None
        self.message = {"role": "assistant", "content": "你好"}
        self.finish = "stop"
        self.usage = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        self.deltas = None
        self.extra_chunks = []
        self.done = True
        self.fragment_bytes = False

    async def __call__(self, payload):
        self.calls.append(copy.deepcopy(payload))
        if self.failure:
            raise self.failure
        if not payload.get("stream"):
            return {
                "id": "chatcmpl_fake",
                "model": payload["model"],
                "choices": [{"message": copy.deepcopy(self.message), "finish_reason": self.finish}],
                "usage": self.usage,
                "session_id": "upstream_session",
            }

        async def generate():
            try:
                deltas = self.deltas if self.deltas is not None else [self.message]
                chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
                chunks += [
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": self.finish}]},
                    {"choices": [], "usage": self.usage},
                    {"choices": [], "session_id": "upstream_session"},
                ]
                chunks += self.extra_chunks
                for chunk in chunks:
                    frame = ": comment\r\nevent: ignored_chat_event\r\ndata: " + dumps(chunk) + "\r\n\r\n"
                    if self.fragment_bytes:
                        for byte in frame.encode("utf-8"):
                            yield bytes([byte])
                    else:
                        yield frame
                if self.done:
                    yield "data: [DONE]\n\n"
            finally:
                self.closed = True

        return StreamingResponse(generate(), media_type="text/event-stream")


@pytest.fixture
def native():
    backend = FakeBackend()
    store = ResponseStore()
    app = FastAPI()
    app.include_router(create_responses_router(backend, store))
    app.include_router(create_messages_router(backend))
    return TestClient(app), backend, store


def response_body(**kwargs):
    return {"model": MODEL, "input": "Hello", **kwargs}


def message_body(**kwargs):
    return {"model": MODEL, "max_tokens": 256, "messages": [{"role": "user", "content": "Hello"}], **kwargs}


@pytest.mark.parametrize("model", [MODEL, "qwen3.8-max"])
@pytest.mark.parametrize("stream", [False, True])
def test_responses_text(native, model, stream):
    client, backend, _ = native
    backend.fragment_bytes = True
    response = client.post("/v1/responses", json=response_body(model=model, stream=stream))
    assert response.status_code == 200
    assert response.headers["x-request-id"].startswith("req_")
    if stream:
        events = parse_sse(response.text)
        assert [event["type"] for event in events[:2]] == ["response.created", "response.in_progress"]
        assert events[0]["response"]["output"] == []
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        assert all(event["response_id"] == events[0]["response"]["id"] for event in events)
        assert events[-1]["type"] == "response.completed"
        assert next(event for event in events if event["type"] == "response.output_item.added")["item"]["content"] == []
        assert "[DONE]" not in response.text
        data = events[-1]["response"]
        assert backend.closed
    else:
        data = response.json()
    assert data["id"].startswith("resp_")
    assert data["object"] == "response"
    assert data["model"] == model
    assert data["status"] == "completed"
    assert data["output"][0]["content"][0]["text"] == "你好"
    assert data["usage"]["input_tokens"] == 12
    assert data["usage"]["output_tokens"] == 5
    assert data["usage"]["input_tokens_details"]["cache_write_tokens"] == 0
    assert data["session_id"] == "upstream_session"
    assert backend.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert backend.calls[0]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("stream", [False, True])
def test_messages_text(native, stream):
    client, backend, _ = native
    backend.fragment_bytes = True
    response = client.post(
        "/v1/messages?beta=true",
        headers={"anthropic-version": "2023-06-01", "x-api-key": "local"},
        json=message_body(stream=stream, system=[{"type": "text", "text": "Be concise", "cache_control": {"type": "ephemeral"}}]),
    )
    assert response.status_code == 200
    assert response.headers["request-id"].startswith("req_")
    if stream:
        events = parse_sse(response.text)
        assert [event["type"] for event in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[0]["message"]["content"] == []
        assert events[1]["content_block"] == {"type": "text", "text": ""}
        assert events[2]["delta"] == {"type": "text_delta", "text": "你好"}
        assert events[-2]["delta"]["stop_reason"] == "end_turn"
        assert events[-2]["usage"]["input_tokens"] == 12
        assert backend.closed
    else:
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"] == [{"type": "text", "text": "你好"}]
        assert data["stop_reason"] == "end_turn"
    assert backend.calls[0]["messages"][0] == {"role": "system", "content": "Be concise"}


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_tools_and_reasoning(native, protocol, stream):
    client, backend, _ = native
    backend.message = {"role": "assistant", "reasoning_content": "检查天气", "content": "稍等", "tool_calls": [CALL]}
    backend.finish = "tool_calls"
    backend.deltas = [
        {"reasoning_content": "检查"},
        {"reasoning_content": "天气"},
        {"content": "稍等"},
        {"tool_calls": [{"index": 0, "id": CALL["id"], "function": {"name": "wea", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"name": "ther", "arguments": '{"city":'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"台北"}'}}]},
    ]
    body = response_body(tools=[TOOL]) if protocol == "responses" else message_body(tools=[CLAUDE_TOOL])
    response = client.post("/v1/" + protocol, json={**body, "stream": stream})
    assert response.status_code == 200
    if stream:
        events = parse_sse(response.text)
        if protocol == "responses":
            assert any(event["type"] == "response.reasoning_summary_text.delta" for event in events)
            added = next(
                event["item"]
                for event in events
                if event["type"] == "response.output_item.added" and event["item"]["type"] == "function_call"
            )
            assert added["arguments"] == ""
            assert added["status"] == "in_progress"
            arguments = "".join(event["delta"] for event in events if event["type"] == "response.function_call_arguments.delta")
            assert json.loads(arguments) == {"city": "台北"}
            assert next(event for event in events if event["type"] == "response.function_call_arguments.done")["name"] == "weather"
            assert events[-1]["response"]["output"][-1]["call_id"] == CALL["id"]
        else:
            starts = [event["content_block"] for event in events if event["type"] == "content_block_start"]
            assert [block["type"] for block in starts] == ["thinking", "text", "tool_use"]
            assert starts[-1]["input"] == {}
            assert starts[-1]["id"] == CALL["id"]
            assert next(event for event in events if event.get("delta", {}).get("type") == "signature_delta")["delta"]["signature"] == ""
            assert events[-2]["delta"]["stop_reason"] == "tool_use"
            assert [event["index"] for event in events if event["type"] == "content_block_stop"] == [0, 1, 2]
    else:
        data = response.json()
        if protocol == "responses":
            assert [item["type"] for item in data["output"]] == ["reasoning", "message", "function_call"]
            assert data["output"][-1]["call_id"] == CALL["id"]
        else:
            assert [block["type"] for block in data["content"]] == ["thinking", "text", "tool_use"]
            assert data["content"][-1]["input"] == {"city": "台北"}
            assert data["stop_reason"] == "tool_use"


def test_responses_previous_id_tools_and_instructions(native):
    client, backend, _ = native
    backend.message = {"role": "assistant", "content": "", "tool_calls": [CALL]}
    backend.finish = "tool_calls"
    first = client.post("/v1/responses", json=response_body(instructions="OLD", tools=[TOOL])).json()
    assert client.get("/v1/responses/" + first["id"]).json() == first
    backend.message, backend.finish = {"content": "Sunny"}, "stop"
    response = client.post(
        "/v1/responses",
        json=response_body(
            previous_response_id=first["id"],
            instructions="NEW",
            tools=[TOOL],
            input=[{"type": "function_call_output", "call_id": CALL["id"], "output": "Sunny"}],
        ),
    )
    assert response.status_code == 200
    history = backend.calls[-1]["messages"]
    assert [item["role"] for item in history] == ["system", "user", "assistant", "tool"]
    assert history[0]["content"] == "NEW"
    assert "OLD" not in dumps(history)
    assert history[-2]["tool_calls"][0]["id"] == history[-1]["tool_call_id"]
    assert client.delete("/v1/responses/" + first["id"]).json()["deleted"]
    assert client.get("/v1/responses/" + first["id"]).status_code == 404
    assert client.post("/v1/responses", json=response_body(previous_response_id=first["id"])).status_code == 404


def test_store_false_and_bounded_cache(native, monkeypatch):
    client, _, cache = native
    data = client.post("/v1/responses", json=response_body(store=False)).json()
    assert not data["store"] and not cache.entries
    assert client.get("/v1/responses/" + data["id"]).status_code == 404
    cache.max_entries = 1
    first = client.post("/v1/responses", json=response_body()).json()
    second = client.post("/v1/responses", json=response_body()).json()
    assert first["id"] not in cache.entries and second["id"] in cache.entries
    cache.max_bytes = 1
    oversized = client.post("/v1/responses", json=response_body()).json()
    assert oversized["store"] is False
    monkeypatch.setattr("freetokenapi.api.responses.time.monotonic", lambda: float("inf"))
    assert client.get("/v1/responses/" + second["id"]).status_code == 404
    assert cache.size == 0


def test_responses_full_history_and_formats(native):
    client, backend, _ = native
    response = client.post(
        "/v1/responses",
        json=response_body(
            input=[
                {"role": "developer", "content": [{"type": "input_text", "text": "Policy"}]},
                {"role": "user", "content": "Weather?"},
                {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "Sunny"}]},
                {"type": "function_call_output", "call_id": "c2", "output": "Cold"},
            ],
            tools=[TOOL],
            text={"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}}},
            reasoning={"effort": "none"},
        ),
    )
    assert response.status_code == 200
    payload = backend.calls[-1]
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    assert len(payload["messages"][2]["tool_calls"]) == 2
    assert payload["thinking"] is False
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}


@pytest.mark.parametrize("stream", [False, True])
def test_custom_tools(native, stream):
    client, backend, _ = native
    backend.message = {
        "tool_calls": [
            {"id": "call_patch", "function": {"name": "apply_patch", "arguments": dumps({"input": "*** Begin Patch\n*** End Patch"})}}
        ]
    }
    backend.finish = "tool_calls"
    tool = {"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}
    result = client.post("/v1/responses", json=response_body(tools=[tool], stream=stream))
    assert result.status_code == 200
    if stream:
        events = parse_sse(result.text)
        assert any(e["type"] == "response.custom_tool_call_input.delta" for e in events)
        result = events[-1]["response"]
    else:
        result = result.json()
    item = result["output"][0]
    assert item["type"] == "custom_tool_call"
    assert item["input"] == "*** Begin Patch\n*** End Patch"
    assert "input" in backend.calls[-1]["tools"][0]["function"]["parameters"]["properties"]
    backend.message, backend.finish = {"content": "Done"}, "stop"
    again = client.post(
        "/v1/responses",
        json=response_body(
            tools=[tool],
            input=[
                {"role": "user", "content": "Change file"},
                item,
                {"type": "custom_tool_call_output", "call_id": item["call_id"], "output": "Success"},
            ],
        ),
    )
    assert again.status_code == 200
    assert backend.calls[-1]["messages"][-1] == {"role": "tool", "tool_call_id": "call_patch", "content": "Success"}


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("in_tool_result", [False, True])
def test_inline_image_content_survives_history_conversion(native, protocol, in_tool_result):
    client, backend, _ = native
    # No image is uploaded: the fake backend only records normalized content.
    data = "aGVsbG8="
    uri = "data:image/png;base64," + data
    expected = [{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": uri}}]
    if protocol == "responses":
        parts = [{"type": "input_text", "text": "Look"}, {"type": "input_image", "image_url": uri}]
        history = [{"role": "user", "content": parts}]
        if in_tool_result:
            history = [
                {"role": "user", "content": "Run"},
                {"type": "function_call", "call_id": "call_image", "name": "weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_image", "output": parts},
            ]
        body = response_body(input=history)
    else:
        parts = [{"type": "text", "text": "Look"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]
        history = [{"role": "user", "content": parts}]
        if in_tool_result:
            history = [
                {"role": "user", "content": "Run"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_image", "name": "weather", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_image", "content": parts}]},
            ]
        body = message_body(messages=history)
    response = client.post("/v1/" + protocol, json=body)
    assert response.status_code == 200, response.text
    assert backend.calls[-1]["messages"][-1]["content"] == expected
    assert backend.calls[-1]["messages"][-1]["role"] == ("tool" if in_tool_result else "user")


def test_claude_tool_results_preserve_order_and_errors(native):
    client, backend, _ = native
    response = client.post(
        "/v1/messages",
        json=message_body(
            tools=[CLAUDE_TOOL],
            messages=[
                {"role": "user", "content": "Run"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Old", "signature": ""},
                        {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "台北"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "is_error": True,
                            "content": [{"type": "text", "text": "Unavailable"}],
                        },
                        {"type": "text", "text": "Retry later"},
                    ],
                },
            ],
        ),
    )
    assert response.status_code == 200
    messages = backend.calls[-1]["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant", "tool", "user"]
    assert messages[2]["tool_call_id"] == "toolu_1"
    assert messages[2]["content"] == "[Tool execution error]\nUnavailable"
    assert messages[3]["content"] == "Retry later"
    assert "signature" not in dumps(messages)


@pytest.mark.parametrize("stream", [False, True])
def test_omitted_thinking(native, stream):
    client, backend, _ = native
    backend.message = {"reasoning_content": "private reasoning", "content": "Answer"}
    response = client.post("/v1/messages", json=message_body(stream=stream, thinking={"type": "enabled", "display": "omitted"}))
    assert response.status_code == 200
    assert "private reasoning" not in response.text
    assert backend.calls[-1]["thinking"] is True


@pytest.mark.parametrize("stream", [False, True])
def test_stop_sequence_across_chunks(native, stream):
    client, backend, _ = native
    backend.message = {"content": "Answer<END>Hidden"}
    backend.deltas = [{"content": text} for text in ["An", "swer<", "EN", "D>Hidden"]]
    response = client.post("/v1/messages", json=message_body(stream=stream, stop_sequences=["<END>"]))
    assert response.status_code == 200
    assert "Hidden" not in response.text
    if stream:
        events = parse_sse(response.text)
        assert "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta") == "Answer"
        assert events[-2]["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "<END>"}
        assert backend.closed
    else:
        assert response.json()["content"] == [{"type": "text", "text": "Answer"}]
        assert response.json()["stop_reason"] == "stop_sequence"
        assert response.json()["stop_sequence"] == "<END>"


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_output_token_limit(native, protocol, stream):
    client, backend, _ = native
    backend.message = {"content": "一二三四五"}
    body = response_body(max_output_tokens=2) if protocol == "responses" else message_body(max_tokens=2)
    response = client.post("/v1/" + protocol, json={**body, "stream": stream})
    assert response.status_code == 200
    assert "三四五" not in response.text
    if protocol == "responses":
        data = parse_sse(response.text)[-1]["response"] if stream else response.json()
        assert data["status"] == "incomplete"
        assert data["incomplete_details"] == {"reason": "max_output_tokens"}
        assert data["output"][0]["content"][0]["text"] == "一二"
    elif stream:
        assert parse_sse(response.text)[-2]["delta"]["stop_reason"] == "max_tokens"
    else:
        assert response.json()["stop_reason"] == "max_tokens"
        assert response.json()["content"][0]["text"] == "一二"


def test_count_tokens_is_local_and_includes_tools(native):
    client, backend, _ = native
    body = {"model": MODEL, "messages": [{"role": "user", "content": "你好"}]}
    first = client.post("/v1/messages/count_tokens", json=body)
    second = client.post("/v1/messages/count_tokens", json={**body, "system": "Rules", "tools": [CLAUDE_TOOL]})
    assert first.status_code == second.status_code == 200
    assert first.headers["x-freetokenapi-token-count"] == "estimate"
    assert second.json()["input_tokens"] > first.json()["input_tokens"] > 0
    assert not backend.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("background", True),
        ("conversation", "conv_remote"),
        ("tools", [{"type": "web_search"}]),
        ("input", [{"type": "input_file", "file_id": "file_x"}]),
        ("input", [{"type": "reasoning", "encrypted_content": "opaque"}]),
        ("input", [{"role": "user", "content": [{"type": "input_image", "image_url": "http://127.0.0.1/secret"}]}]),
        ("tool_choice", []),
        ("tool_choice", {"type": []}),
        ("reasoning", {"effort": []}),
        ("input", [{"role": [], "content": "bad"}]),
        ("input", [{"type": []}]),
        ("tools", [{"type": [], "name": "bad"}]),
        ("text", {"format": {"type": []}}),
        ("max_output_tokens", 0),
        ("unexpected_feature", True),
    ],
)
def test_responses_rejects_unsupported_or_malformed(native, field, value):
    client, backend, _ = native
    response = client.post("/v1/responses", json=response_body(**{field: value}))
    assert response.status_code == 400, response.text
    assert "error" in response.json() and "detail" not in response.json()
    assert not backend.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        {"messages": [{"role": "system", "content": "use top-level instead"}]},
        {"messages": [{"role": [], "content": "bad"}]},
        {"messages": [{"role": "user", "content": [{"type": []}]}]},
        {"messages": [{"role": "user", "content": [{"type": "tool_use", "id": "x", "name": "weather", "input": {}}]}]},
        {"thinking": {"type": []}},
        {"tool_choice": {"type": []}},
        {"stop_sequences": [""]},
        {"max_tokens": 0},
        {"context_management": {"edits": [{"type": "clear_tool_uses"}]}},
    ],
)
def test_messages_rejects_unsupported_or_malformed(native, changes):
    client, backend, _ = native
    response = client.post("/v1/messages", json=message_body(**changes))
    assert response.status_code == 400, response.text
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not backend.calls


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("status", [401, 404, 429, 503])
def test_upstream_http_errors_use_native_envelopes(native, protocol, status):
    client, backend, _ = native
    backend.failure = HTTPException(status, "mock upstream failure")
    body = response_body() if protocol == "responses" else message_body()
    response = client.post("/v1/" + protocol, json=body)
    assert response.status_code == status
    assert response.json()["error"]["message"] == "mock upstream failure"
    if protocol == "messages":
        assert response.json()["type"] == "error"


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("failure", ["truncated", "error"])
def test_stream_failure_is_not_reported_as_success(native, protocol, failure):
    client, backend, _ = native
    if failure == "truncated":
        backend.done = False
    else:
        backend.extra_chunks = [{"error": {"message": "provider stopped"}, "choices": []}]
    body = response_body(stream=True) if protocol == "responses" else message_body(stream=True)
    response = client.post("/v1/" + protocol, json=body)
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert any(event["type"] == "error" for event in events)
    assert not any(event["type"] in {"response.completed", "message_stop"} for event in events)
    if protocol == "responses":
        assert events[-1]["response"]["status"] == "failed"
        assert events[-1]["response"]["store"] is False
    assert backend.closed


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("valid_prefix", [False, True])
def test_invalid_tool_arguments_are_not_executed(native, protocol, valid_prefix):
    client, backend, _ = native
    backend.message = {"tool_calls": [{"id": "x", "function": {"name": "weather", "arguments": "not-json"}}]}
    if valid_prefix:
        backend.message["tool_calls"].insert(0, copy.deepcopy(CALL))
    backend.finish = "tool_calls"
    body = response_body(tools=[TOOL], stream=True) if protocol == "responses" else message_body(tools=[CLAUDE_TOOL], stream=True)
    response = client.post("/v1/" + protocol, json=body)
    events = parse_sse(response.text)
    assert any(event["type"] == "error" for event in events)
    assert not any(event.get("content_block", {}).get("type") == "tool_use" for event in events)
    assert not any(event.get("item", {}).get("type") == "function_call" for event in events)


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("violation", ["required", "parallel"])
@pytest.mark.parametrize("stream", [False, True])
def test_backend_tool_choice_violations_are_reported(native, protocol, violation, stream):
    client, backend, _ = native
    if violation == "parallel":
        backend.message = {"tool_calls": [copy.deepcopy(CALL), {**copy.deepcopy(CALL), "id": "call_second"}]}
        backend.finish = "tool_calls"
    if protocol == "responses":
        body = response_body(tools=[TOOL], stream=stream, tool_choice="required", parallel_tool_calls=False)
    else:
        body = message_body(tools=[CLAUDE_TOOL], stream=stream, tool_choice={"type": "any", "disable_parallel_tool_use": True})
    response = client.post("/v1/" + protocol, json=body)
    if not stream:
        assert response.status_code == 502, response.text
        assert "error" in response.json()
    else:
        events = parse_sse(response.text)
        assert any(event["type"] == "error" for event in events)
        assert not any(event["type"] in {"response.completed", "message_stop"} for event in events)
        assert not any(event.get("content_block", {}).get("type") == "tool_use" for event in events)
        assert not any(event.get("item", {}).get("type") == "function_call" for event in events)
        assert backend.closed


async def test_closing_chat_adapter_closes_upstream():
    backend = FakeBackend()
    payload = {"model": MODEL, "messages": [], "stream": True}
    response = await backend(payload)
    run = ChatRun(response, payload)
    iterator = run.__aiter__()
    assert await anext(iterator) == ("text", "你好")
    await iterator.aclose()
    assert backend.closed


async def test_sse_parser_handles_multiline_frames_and_bad_json():
    async def multiline():
        yield 'data: {"choices": [],\ndata: "usage": {"prompt_tokens": 2}}\n\ndata: [DONE]\n\n'

    assert [chunk async for chunk in chat_chunks(StreamingResponse(multiline()))] == [{"choices": [], "usage": {"prompt_tokens": 2}}]

    async def malformed():
        yield "data: not-json\n\n"

    with pytest.raises(ProtocolError, match="malformed"):
        _ = [chunk async for chunk in chat_chunks(StreamingResponse(malformed()))]


def test_output_limit_flushes_partial_stop_and_keeps_tool_json_atomic():
    limit = OutputLimit(128, ["STOP"])
    assert limit.text("answerST") == "answer"
    assert limit.text("", final=True) == "ST"
    limit = OutputLimit(1)
    assert limit.take('{"large":"value"}', atomic=True) == ""
    assert limit.reason == "length"


@pytest.mark.parametrize("provider,model", [("deepseek", MODEL), ("qwen", "qwen3.8-max")])
@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_real_router_reuses_existing_provider_implementation(monkeypatch, provider, model, protocol, stream):
    # Reuse existing fake web SSE fixtures, not a mock of chat_completions.
    from test_tools_api import DS_TOOL_SSE, QWEN_TOOL_SSE, FakeAccount

    import freetokenapi.api.openai as api

    account = FakeAccount([DS_TOOL_SSE if provider == "deepseek" else QWEN_TOOL_SSE])
    pool = MagicMock()
    pool.healthy = [account]
    monkeypatch.setattr(api.app.state, "pool", pool, raising=False)
    monkeypatch.setattr(api.app.state, "qwen_pool", pool, raising=False)
    monkeypatch.setattr(api, "_acquire_and_build", AsyncMock(return_value=(account, "s1", None, "request", True)))
    tool = {"type": "function", "name": "get_weather", "parameters": TOOL["parameters"]}
    body = (
        response_body(model=model, tools=[tool], stream=stream)
        if protocol == "responses"
        else message_body(model=model, tools=[{"name": "get_weather", "input_schema": TOOL["parameters"]}], stream=stream)
    )
    # No context manager: do not run real credential-checking app lifespan.
    client = TestClient(api.app)
    response = client.post("/v1/" + protocol, json=body)
    assert response.status_code == 200, response.text
    if stream:
        events = parse_sse(response.text)
        assert events[-1]["type"] == ("response.completed" if protocol == "responses" else "message_stop"), response.text
        if protocol == "responses":
            assert events[-1]["response"]["output"][-1]["name"] == "get_weather"
        else:
            assert (
                next(
                    event["content_block"]
                    for event in events
                    if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
                )["name"]
                == "get_weather"
            )
    else:
        data = response.json()
        item = data["output"][-1] if protocol == "responses" else data["content"][-1]
        assert item["name"] == "get_weather"
    assert account.client.completion.await_count == 1
    assert account.sem._value == 1


async def test_original_stream_guard_closes_nested_generator():
    import freetokenapi.api.openai as api

    closed = False

    async def source():
        nonlocal closed
        try:
            yield "first"
            yield "second"
        finally:
            closed = True

    guard = api._stream_guard(source(), MODEL)
    assert await anext(guard) == "first"
    await guard.aclose()
    assert closed


def test_real_routes_forward_qwen_images_and_reject_unknown_models(monkeypatch):
    import freetokenapi.api.openai as api

    completion = AsyncMock(return_value={"id": "chatcmpl-image", "object": "chat.completion", "created": 0,
        "model": "qwen3.8-max", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Image received"}, "finish_reason": "stop"}]})
    monkeypatch.setattr(api, "_chat_completions_qwen", completion)
    monkeypatch.setattr(api.app.state, "qwen_models", [], raising=False)
    client = TestClient(api.app)
    image = "data:image/png;base64,aGVsbG8="
    body = response_body(model="qwen3.8-max", input=[{"role": "user", "content": [{"type": "input_image", "image_url": image}]}])
    assert client.post("/v1/responses", json=body).status_code == 200
    completion.assert_awaited_once()
    assert completion.call_args.args[0].messages[0].content[0]["image_url"]["url"] == image
    completion.reset_mock()
    assert client.post("/v1/responses", json=response_body(model="unknown-model")).status_code == 404
    completion.assert_not_awaited()


def test_openapi_exposes_new_routes_without_replacing_chat():
    import freetokenapi.api.openai as api

    schema = api.app.openapi()
    assert all(path in schema["paths"] for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens"])


CLAUDE_CONTEXT_HINTS = [
    {},
    {"edits": []},
    {"edits": [{"type": "clear_thinking_20251015"}]},
    {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}}]},
    {"edits": [{"type": "clear_tool_uses_20250919"}]},
    {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 100000},
                "keep": {"type": "tool_uses", "value": 3},
                "clear_at_least": {"type": "input_tokens", "value": 2000},
                "exclude_tools": ["weather"],
                "clear_tool_inputs": True,
            }
        ]
    },
    {
        "edits": [
            {"type": "clear_thinking_20251015", "keep": "all"},
            {"type": "clear_tool_uses_20250919", "trigger": {"type": "tool_uses", "value": 5}},
        ]
    },
]


@pytest.mark.parametrize("context", CLAUDE_CONTEXT_HINTS)
@pytest.mark.parametrize("stream", [False, True])
def test_claude_context_hints_do_not_block_or_silently_clear_history(native, context, stream):
    client, backend, _ = native
    body = message_body(
        tools=[CLAUDE_TOOL],
        messages=[
            {"role": "user", "content": "Check the weather"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Old reasoning", "signature": ""},
                    {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "台北"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Keep this tool output"}]},
        ],
    )
    baseline = client.post("/v1/messages", json=body)
    assert baseline.status_code == 200
    assert "context_management" not in baseline.json()
    assert "x-freetokenapi-context-management" not in baseline.headers
    expected = backend.calls[-1]
    response = client.post(
        "/v1/messages?beta=true",
        headers={"anthropic-beta": "context-management-2025-06-27"},
        json={**body, "context_management": context, "stream": stream},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-freetokenapi-context-management"] == "not-applied"
    assert backend.calls[-1] == {**expected, "stream": stream}
    assert backend.calls[-1]["messages"][-1]["content"] == "Keep this tool output"
    assert "context_management" not in backend.calls[-1]
    if stream:
        events = parse_sse(response.text)
        assert events[0]["message"]["context_management"] == {"applied_edits": []}
        delta = next(event for event in events if event["type"] == "message_delta")
        assert delta["context_management"] == {"applied_edits": []}
        assert events[-1]["type"] == "message_stop"
        assert backend.closed
    else:
        assert response.json()["content"] == [{"type": "text", "text": "你好"}]
        assert response.json()["context_management"] == {"applied_edits": []}


@pytest.mark.parametrize("context", CLAUDE_CONTEXT_HINTS)
def test_count_tokens_accepts_context_hints_without_claiming_context_edits(native, context):
    client, backend, _ = native
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Keep the same prompt"}]}
    baseline = client.post("/v1/messages/count_tokens", json=body)
    response = client.post("/v1/messages/count_tokens", json={**body, "context_management": context})
    assert response.status_code == 200
    assert response.json() == baseline.json()
    assert response.headers["x-freetokenapi-context-management"] == "not-applied"
    assert response.headers["x-freetokenapi-token-count"] == "estimate"
    assert not backend.calls


@pytest.mark.parametrize(
    "context",
    [
        {"unknown": True},
        {"edits": "not an array"},
        {"edits": [None]},
        {"edits": [{"type": []}]},
        {"edits": [{"type": "compact_20260112"}]},
        {"edits": [{"type": "clear_thinking_20251015", "unknown": True}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": 2}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 0}}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": True}}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": [], "value": 1}}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": -1}}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": "10"}}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "input_tokens", "value": 1}}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 1, "unknown": 2}}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "exclude_tools": "weather"}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "exclude_tools": [1]}]},
        {"edits": [{"type": "clear_tool_uses_20250919", "clear_tool_inputs": "yes"}]},
    ],
)
@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
def test_invalid_or_cloud_only_context_edits_still_report_native_errors(native, context, endpoint):
    client, backend, _ = native
    body = message_body(context_management=context)
    if endpoint.endswith("count_tokens"):
        body.pop("max_tokens")
    response = client.post("/v1/" + endpoint, json=body)
    assert response.status_code == 400, response.text
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not backend.calls


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}, {"type": "enabled", "budget_tokens": 1024}, {"type": "adaptive"}])
@pytest.mark.parametrize("stream", [False, True])
def test_messages_accept_instruction_prefix_without_changing_thinking(native, role, blocked, thinking, stream):
    client, backend, _ = native
    instructions = "Keep the instruction prefix"
    content = [{"type": "text", "text": instructions, "cache_control": {"type": "ephemeral"}}] if blocked else instructions
    common = {"stream": stream, "thinking": thinking, "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}}
    canonical = message_body(system="Native instructions\n" + instructions, **common)
    baseline = client.post("/v1/messages", json=canonical)
    assert baseline.status_code == 200, baseline.text
    expected = copy.deepcopy(backend.calls[-1])
    legacy = message_body(
        system="Native instructions",
        messages=[{"role": role, "content": content}, {"role": "user", "content": "Hello"}],
        **common,
    )
    original = copy.deepcopy(legacy)
    response = client.post("/v1/messages", json=legacy)
    assert response.status_code == 200, response.text
    assert backend.calls[-1] == expected
    assert legacy == original
    assert response.headers["x-freetokenapi-context-management"] == "not-applied"
    if stream:
        events = parse_sse(response.text)
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        assert not any(event["type"] == "error" for event in events)
    else:
        assert response.json()["content"] == [{"type": "text", "text": "你好"}]


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("with_native_system", [False, True])
def test_count_tokens_normalizes_instruction_prefix_like_native_system(native, role, with_native_system):
    client, backend, _ = native
    messages = [{"role": "user", "content": "Hello"}]
    native_system = "Top-level instructions" if with_native_system else None
    combined = "Top-level instructions\nPrefix" if with_native_system else "Prefix"
    baseline = client.post("/v1/messages/count_tokens", json={"model": MODEL, "system": combined, "messages": messages})
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": MODEL, "system": native_system, "messages": [{"role": role, "content": "Prefix"}, *messages]},
    )
    assert response.status_code == 200, response.text
    assert response.json() == baseline.json()
    assert not backend.calls


def test_messages_preserve_order_of_multiple_instruction_prefixes_and_tool_history(native):
    client, backend, _ = native
    response = client.post(
        "/v1/messages",
        json=message_body(
            system="Native",
            tools=[CLAUDE_TOOL],
            messages=[
                {"role": "system", "content": "First"},
                {"role": "developer", "content": [{"type": "text", "text": "Second"}, {"type": "text", "text": "Third"}]},
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_prefix", "name": "weather", "input": {"city": "台北"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_prefix", "content": "Keep tool output"}]},
            ],
        ),
    )
    assert response.status_code == 200, response.text
    history = backend.calls[-1]["messages"]
    assert history[0] == {"role": "system", "content": "Native\nFirst\nSecond\nThird"}
    assert [message["role"] for message in history] == ["system", "user", "assistant", "tool"]
    assert history[-1] == {"role": "tool", "tool_call_id": "toolu_prefix", "content": "Keep tool output"}


@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "system", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]},
            {"role": "user", "content": "PRIVATE_PROMPT"},
        ],
        [
            {"role": "developer", "content": [{"type": "tool_use", "id": "x", "name": "weather", "input": {}}]},
            {"role": "user", "content": "PRIVATE_PROMPT"},
        ],
        [{"role": "tool", "tool_call_id": "x", "content": "PRIVATE_PROMPT"}],
        [{"role": "function", "name": "weather", "content": "PRIVATE_PROMPT"}],
        [{"role": "system", "content": {"text": "PRIVATE_PROMPT"}}, {"role": "user", "content": "Hello"}],
        [{"role": [], "content": "PRIVATE_PROMPT"}],
        [{"role": {}, "content": "PRIVATE_PROMPT"}],
    ],
)
def test_messages_instruction_compatibility_does_not_accept_other_invalid_roles_or_content(native, endpoint, messages):
    client, backend, _ = native
    body = message_body(messages=messages)
    if endpoint.endswith("count_tokens"):
        body.pop("max_tokens")
    response = client.post("/v1/" + endpoint, json=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "PRIVATE_PROMPT" not in response.text
    assert not backend.calls


def test_unknown_messages_role_error_identifies_position_without_logging_message(native):
    client, backend, _ = native
    response = client.post(
        "/v1/messages",
        json=message_body(messages=[{"role": "user", "content": "Hello"}, {"role": "tool", "content": "PRIVATE_PROMPT"}]),
    )
    assert response.status_code == 400
    error = response.json()["error"]["message"]
    assert "messages.1.role" in error and "'tool'" in error and "tool_result" in error
    assert "PRIVATE_PROMPT" not in error
    assert not backend.calls


CLIENT_METADATA_CASES = [
    None,
    {},
    {"trace_id": "PRIVATE_CLIENT_METADATA", "origin": "cc-switch"},
    {
        "turn": {"id": "PRIVATE_CLIENT_METADATA", "labels": ["local", 1, None, False]},
        # Metadata must not override real inference or transport settings.
        "model": "not-a-model",
        "thinking": False,
        "search": True,
        "base_url": "https://metadata.invalid",
    },
]


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("client_metadata", CLIENT_METADATA_CASES)
def test_client_metadata_is_ignored_without_changing_generation_or_leaking(native, protocol, stream, client_metadata):
    client, backend, store = native
    if protocol == "responses":
        body = response_body(stream=stream, metadata={"public_tag": "keep"}, reasoning={"effort": "medium"})
    else:
        body = message_body(stream=stream, metadata={"user_id": "keep"}, thinking={"type": "enabled", "budget_tokens": 1024})
    baseline = client.post("/v1/" + protocol, json=body)
    assert baseline.status_code == 200, baseline.text
    expected = copy.deepcopy(backend.calls[-1])
    request = {**body, "client_metadata": client_metadata}
    original = copy.deepcopy(request)
    response = client.post("/v1/" + protocol, json=request)
    assert response.status_code == 200, response.text
    assert backend.calls[-1] == expected
    assert request == original
    assert "client_metadata" not in dumps(backend.calls[-1])
    assert "PRIVATE_CLIENT_METADATA" not in dumps(backend.calls[-1])
    assert "client_metadata" not in response.text
    assert "PRIVATE_CLIENT_METADATA" not in response.text
    data = response.json() if not stream else None
    if stream:
        events = parse_sse(response.text)
        assert not any(event["type"] == "error" for event in events)
        assert backend.closed
        if protocol == "responses":
            assert events[-1]["type"] == "response.completed"
            data = events[-1]["response"]
        else:
            assert events[-1]["type"] == "message_stop"
    if protocol == "responses":
        assert data["metadata"] == {"public_tag": "keep"}
        assert "PRIVATE_CLIENT_METADATA" not in dumps(store.get(data["id"]))
        assert "client_metadata" not in dumps(store.get(data["id"]))


@pytest.mark.parametrize("client_metadata", CLIENT_METADATA_CASES)
def test_count_tokens_ignores_client_metadata(native, client_metadata):
    client, backend, _ = native
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]}
    baseline = client.post("/v1/messages/count_tokens", json=body)
    response = client.post("/v1/messages/count_tokens", json={**body, "client_metadata": client_metadata})
    assert response.status_code == 200, response.text
    assert response.json() == baseline.json()
    assert "PRIVATE_CLIENT_METADATA" not in response.text
    assert not backend.calls


@pytest.mark.parametrize("endpoint", ["responses", "messages", "messages/count_tokens"])
@pytest.mark.parametrize("client_metadata", [[], ["PRIVATE_CLIENT_METADATA"], "PRIVATE_CLIENT_METADATA", 7, False])
def test_client_metadata_rejects_non_objects_without_echoing_private_values(native, endpoint, client_metadata):
    client, backend, _ = native
    body = response_body() if endpoint == "responses" else message_body()
    if endpoint.endswith("count_tokens"):
        body.pop("max_tokens")
    response = client.post("/v1/" + endpoint, json={**body, "client_metadata": client_metadata})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "client_metadata" in error["message"]
    assert "PRIVATE_CLIENT_METADATA" not in response.text
    assert not backend.calls


@pytest.mark.parametrize("endpoint", ["responses", "messages", "messages/count_tokens"])
def test_client_metadata_does_not_disable_unknown_field_validation(native, endpoint):
    client, backend, _ = native
    body = response_body() if endpoint == "responses" else message_body()
    if endpoint.endswith("count_tokens"):
        body.pop("max_tokens")
    response = client.post("/v1/" + endpoint, json={**body, "client_metadata": {}, "unknown_server_feature": "PRIVATE_VALUE"})
    assert response.status_code == 400, response.text
    assert "unknown_server_feature" in response.json()["error"]["message"]
    assert "PRIVATE_VALUE" not in response.text
    assert not backend.calls


@pytest.mark.parametrize(
    "endpoint,options",
    [
        ("responses", {"background": True}),
        ("messages", {"container": "cloud-container"}),
        ("messages/count_tokens", {"mcp_servers": [{"type": "url", "url": "https://metadata.invalid"}]}),
    ],
)
def test_client_metadata_does_not_enable_unsupported_cloud_features(native, endpoint, options):
    client, backend, _ = native
    body = response_body() if endpoint == "responses" else message_body()
    if endpoint.endswith("count_tokens"):
        body.pop("max_tokens")
    response = client.post("/v1/" + endpoint, json={**body, **options, "client_metadata": {"trace_id": "PRIVATE_CLIENT_METADATA"}})
    assert response.status_code == 400, response.text
    assert "client_metadata" not in response.json()["error"]["message"]
    assert "PRIVATE_CLIENT_METADATA" not in response.text
    assert not backend.calls


def test_client_metadata_is_excluded_from_request_serialization_and_repr():
    from freetokenapi.api.compat import NativeRequest

    request = NativeRequest(client_metadata={"trace_id": "PRIVATE_CLIENT_METADATA"})
    assert "client_metadata" not in request.model_dump()
    assert "PRIVATE_CLIENT_METADATA" not in request.model_dump_json()
    assert "PRIVATE_CLIENT_METADATA" not in repr(request)
