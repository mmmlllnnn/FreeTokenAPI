"""Real client shapes; fake web responses, no accounts or local services contacted."""

import copy
import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_api_protocols import (
    CLAUDE_TOOL,
    MODEL,
    TOOL,
    FakeBackend,
    message_body,
    parse_sse,
    response_body,
)

from freetokenapi import tools as toolemu
from freetokenapi.api import openai as api
from freetokenapi.api.compat import dumps
from freetokenapi.api.messages import create_messages_router
from freetokenapi.api.responses import (
    ResponsesRequest,
    ResponseStore,
    _tool_backend_name,
    create_responses_router,
    prepare_request,
)


@pytest.fixture
def native():
    backend, store = FakeBackend(), ResponseStore()
    app = FastAPI()
    app.include_router(create_responses_router(backend, store))
    app.include_router(create_messages_router(backend))
    return TestClient(app), backend, store


def completed_response(response, stream):
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()
    events = parse_sse(response.text)
    assert events[-1]["type"] == "response.completed", response.text
    assert not any(event["type"] == "error" for event in events)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    return events[-1]["response"]


def grouped_tools(kind="function", field="tools"):
    leaf = {"type": kind, "name": "lookup", "description": "A harmless test lookup"}
    if kind == "function":
        leaf["parameters"] = {"type": "object", "properties": {"city": {"type": "string"}}}
    else:
        leaf["format"] = {"type": "text"}
    return [
        copy.deepcopy(leaf),
        {"type": "namespace", "name": "first.group", "description": "First group", field: [copy.deepcopy(leaf)]},
        {"type": "namespace", "name": "second_group", "description": "Second group", field: [copy.deepcopy(leaf)]},
    ]


def backend_call(name, arguments, call_id="call_lookup"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("as_blocks", [False, True])
@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}, {"type": "enabled", "budget_tokens": 1024}, {"type": "adaptive"}])
@pytest.mark.parametrize("stream", [False, True])
def test_messages_inline_instructions_preserve_user_and_position(native, role, as_blocks, thinking, stream):
    client, backend, _ = native
    content = [{"type": "text", "text": "Later instruction", "cache_control": {"type": "ephemeral"}}] if as_blocks else "Later instruction"
    body = message_body(
        system="Top instruction",
        messages=[{"role": "user", "content": "Actual user question"}, {"role": role, "content": content}],
        thinking=thinking,
        stream=stream,
        context_management={"edits": [{"type": "clear_thinking_20251015"}]},
    )
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    assert backend.calls[-1]["messages"] == [
        {"role": "system", "content": "Top instruction"},
        {"role": "user", "content": "Actual user question"},
        {"role": "system", "content": "Later instruction"},
    ]
    assert backend.calls[-1]["thinking"] is (None if thinking is None else thinking["type"] != "disabled")
    assert response.headers["x-freetokenapi-context-management"] == "not-applied"
    if stream:
        events = parse_sse(response.text)
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        assert not any(event["type"] == "error" for event in events)
    else:
        assert response.json()["content"] == [{"type": "text", "text": "你好"}]


@pytest.mark.parametrize("role", ["system", "developer"])
def test_messages_inline_instructions_keep_tool_result_as_untrusted_history(native, role):
    client, backend, _ = native
    response = client.post(
        "/v1/messages",
        json=message_body(
            system="Prefix",
            tools=[CLAUDE_TOOL],
            messages=[
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {}}]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Tool output, not instructions"}],
                },
                {"role": role, "content": "New instruction"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": role, "content": "Latest instruction"},
                {"role": "user", "content": "Follow-up"},
            ],
        ),
    )
    assert response.status_code == 200, response.text
    history = backend.calls[-1]["messages"]
    assert [item["role"] for item in history] == ["system", "user", "assistant", "tool", "system", "assistant", "system", "user"]
    assert history[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "Tool output, not instructions"}
    assert [history[i]["content"] for i in (0, 4, 6)] == ["Prefix", "New instruction", "Latest instruction"]


@pytest.mark.parametrize("role", ["system", "developer"])
def test_count_tokens_accepts_and_counts_inline_instructions(native, role):
    client, backend, _ = native
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Question"}]}
    baseline = client.post("/v1/messages/count_tokens", json=body)
    body["messages"].append({"role": role, "content": [{"type": "text", "text": "Additional instruction " * 20}]})
    response = client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > baseline.json()["input_tokens"]
    assert backend.calls == []


@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
        {"type": "tool_use", "name": "weather", "id": "t1", "input": {}},
    ],
)
def test_inline_system_still_rejects_nontext_blocks(native, endpoint, role, block):
    client, backend, _ = native
    body = message_body(messages=[{"role": "user", "content": "PRIVATE_QUESTION"}, {"role": role, "content": [block]}])
    if endpoint.endswith("count_tokens"):
        body.pop("max_tokens")
    response = client.post("/v1/" + endpoint, json=body)
    assert response.status_code == 400, response.text
    assert "PRIVATE_QUESTION" not in response.text
    assert not backend.calls


@pytest.mark.parametrize("has_session", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("with_prefix", [False, True])
def test_prompt_keeps_question_before_later_system_even_in_cached_session(has_session, with_tools, with_prefix):
    messages = [api.ChatMessage(role="user", content="Actual question"), api.ChatMessage(role="system", content="Later instruction")]
    if with_prefix:
        messages.insert(0, api.ChatMessage(role="system", content="Original prefix"))
    tools = [{"type": "function", "function": {"name": "weather", "parameters": TOOL["parameters"]}}] if with_tools else None
    prompt, mode = toolemu.build_prompt(messages, tools, "auto", has_session)
    ordered = "User: Actual question\nSystem: Later instruction"
    if with_prefix:
        ordered = "System: Original prefix\n" + ordered
    assert ordered in prompt
    assert prompt.count("Actual question") == prompt.count("Later instruction") == 1
    assert mode is with_tools
    assert toolemu._has_history(messages)
    assert toolemu.has_interleaved_system(messages)
    assert prompt == toolemu.build_prompt(messages, tools, "auto", False)[0]
    assert api._reduced_prompt_variants(messages, tools, "auto", None, prompt) == []


@pytest.mark.parametrize(
    "roles,expected",
    [
        ([], False),
        (["user"], False),
        (["system", "system", "user"], False),
        (["user", "assistant", "user"], False),
        (["user", "system"], True),
        (["assistant", "system"], True),
        (["user", "assistant", "tool", "system", "user"], True),
    ],
)
def test_interleaved_system_detection_does_not_disable_ordinary_sessions(roles, expected):
    assert toolemu.has_interleaved_system([api.ChatMessage(role=role, content="text") for role in roles]) is expected


@pytest.mark.parametrize("explicit_session", [None, "user_supplied_old_session"])
@pytest.mark.parametrize("model", [MODEL, "qwen3.8-max"])
async def test_inline_instructions_skip_both_explicit_and_cached_web_sessions(explicit_session, model):
    account, pool = MagicMock(), MagicMock()
    pool.acquire = AsyncMock(return_value=(account, "unexpected_reusable_session"))
    req = api.ChatCompletionRequest(
        model=model,
        session_id=explicit_session,
        messages=[api.ChatMessage(role="user", content="Original question"), api.ChatMessage(role="system", content="Inline instruction")],
    )
    result = await api._acquire_and_build(pool, req, {"model": model})
    assert result[0] is account
    assert result[1] is None
    assert result[2]
    assert result[3] == "User: Original question\nSystem: Inline instruction"
    pool.resolve_context.assert_not_called()
    pool.acquire.assert_awaited_once_with(None, api.settings.acquire_timeout)
    account.sessions.can_reuse.assert_not_called()


@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["tools", "children"])
def test_namespace_tools_restore_identity_in_json_sse_and_cache(native, kind, stream, field):
    client, backend, store = native
    alias = _tool_backend_name("lookup", "second_group")
    arguments = dumps({"input": "Exact raw input\n第二行"} if kind == "custom" else {"city": "台北"})
    backend.message = {"tool_calls": [backend_call(alias, arguments)]}
    backend.finish = "tool_calls"
    backend.fragment_bytes = True
    backend.deltas = [
        {"tool_calls": [{"index": 0, "id": "call_lookup", "function": {"name": alias[:8], "arguments": arguments[:10]}}]},
        {"tool_calls": [{"index": 0, "function": {"name": alias[8:], "arguments": arguments[10:]}}]},
    ]
    response = client.post(
        "/v1/responses",
        json=response_body(
            tools=grouped_tools(kind, field),
            stream=stream,
            tool_choice={"type": kind, "name": "lookup", "namespace": "second_group"},
            client_metadata={"trace_id": "PRIVATE_CLIENT_ID"},
        ),
    )
    data = completed_response(response, stream)
    item = data["output"][0]
    assert item["type"] == ("custom_tool_call" if kind == "custom" else "function_call")
    assert (item["name"], item["namespace"], item["call_id"]) == ("lookup", "second_group", "call_lookup")
    assert item["input" if kind == "custom" else "arguments"] == ("Exact raw input\n第二行" if kind == "custom" else arguments)
    assert alias not in response.text
    assert "PRIVATE_CLIENT_ID" not in response.text
    payload = backend.calls[-1]
    assert "client_metadata" not in dumps(payload)
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert len(names) == len(set(names)) == 3
    assert "lookup" in names and alias in names
    assert payload["tool_choice"] == {"type": "function", "function": {"name": alias}}
    cached_response, cached_history = store.get(data["id"])
    assert cached_response["output"] == data["output"]
    assert "PRIVATE_CLIENT_ID" not in dumps([cached_response, cached_history])
    assert cached_history[-1]["tool_calls"][0]["function"] == {"name": "lookup", "namespace": "second_group", "arguments": arguments}
    if stream:
        events = parse_sse(response.text)
        items = [event["item"] for event in events if event["type"] in {"response.output_item.added", "response.output_item.done"}]
        assert len(items) == 2
        assert all(item["name"] == "lookup" and item["namespace"] == "second_group" for item in items)
        assert items[0]["status"] == "in_progress" and items[1]["status"] == "completed"
        prefix = "response.custom_tool_call_input" if kind == "custom" else "response.function_call_arguments"
        assert next(event for event in events if event["type"] == prefix + ".done")["name"] == "lookup"


@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("remove_tools", [False, True])
def test_namespaced_tool_call_result_and_continuation_round_trip(native, kind, stream, cached, remove_tools):
    client, backend, store = native
    tools = grouped_tools(kind)
    alias = _tool_backend_name("lookup", "first.group")
    arguments = dumps({"input": "read-only lookup"} if kind == "custom" else {"city": "台北"})
    backend.message, backend.finish = {"tool_calls": [backend_call(alias, arguments)]}, "tool_calls"
    first = completed_response(
        client.post(
            "/v1/responses",
            json=response_body(
                input="Question",
                tools=tools,
                stream=stream,
                store=cached,
            ),
        ),
        stream,
    )
    call = first["output"][0]
    tool_result = {
        "type": "custom_tool_call_output" if kind == "custom" else "function_call_output",
        "call_id": call["call_id"],
        "output": "Sunny",
    }
    backend.message, backend.finish = {"content": "Answer based on Sunny"}, "stop"
    body = response_body(
        tools=[] if remove_tools else list(reversed(tools)),
        stream=stream,
        input=[tool_result] if cached else [{"role": "user", "content": "Question"}, call, tool_result],
        previous_response_id=first["id"] if cached else None,
    )
    second = completed_response(client.post("/v1/responses", json=body), stream)
    assert second["output"][0]["content"][0]["text"] == "Answer based on Sunny"
    history = backend.calls[-1]["messages"]
    assert [message["role"] for message in history] == ["user", "assistant", "tool"]
    assert history[1]["tool_calls"][0] == backend_call(alias, arguments)
    assert history[-1] == {"role": "tool", "tool_call_id": "call_lookup", "content": "Sunny"}
    # The store is canonical even after active tools are reordered or removed.
    _, cached_history = store.get(second["id"])
    assert cached_history[1]["tool_calls"][0]["function"] == {"name": "lookup", "namespace": "first.group", "arguments": arguments}


@pytest.mark.parametrize("stream", [False, True])
def test_same_leaf_name_in_global_and_multiple_namespaces_is_not_confused(native, stream):
    client, backend, _ = native
    identities = [None, "first.group", "second_group"]
    backend.message = {"tool_calls": [backend_call(_tool_backend_name("lookup", ns), "{}", f"call_{i}") for i, ns in enumerate(identities)]}
    backend.finish = "tool_calls"
    data = completed_response(client.post("/v1/responses", json=response_body(tools=grouped_tools(), stream=stream)), stream)
    assert [(item["name"], item.get("namespace")) for item in data["output"]] == [("lookup", ns) for ns in identities]
    assert [item["call_id"] for item in data["output"]] == ["call_0", "call_1", "call_2"]


def test_namespace_conversion_does_not_mutate_request_or_stored_history():
    tools = grouped_tools()
    req = ResponsesRequest(
        **response_body(
            tools=tools,
            input=[
                {"role": "user", "content": "Question"},
                {"type": "function_call", "name": "lookup", "namespace": "first.group", "call_id": "c1", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "Result"},
            ],
        )
    )
    original = copy.deepcopy(req.model_dump())
    payload, history, _ = prepare_request(req, ResponseStore())
    assert req.model_dump() == original
    assert tools == grouped_tools()
    assert history[1]["tool_calls"][0]["function"]["namespace"] == "first.group"
    assert "namespace" not in payload["messages"][1]["tool_calls"][0]["function"]
    payload["messages"][1]["tool_calls"][0]["function"]["name"] = "changed"
    assert history[1]["tool_calls"][0]["function"]["name"] == "lookup"


def test_namespace_aliases_are_stable_bounded_and_not_ambiguous_delimiter_splits():
    identities = [("a_b", "c"), ("a", "b_c"), ("a.b", "c"), ("a" * 120, "lookup"), ("天气", "查询"), ("x", "z" * 100)]
    aliases = [_tool_backend_name(leaf, group) for group, leaf in identities]
    assert len(set(aliases)) == len(identities)
    assert all(len(alias) <= 64 and re.fullmatch(r"[a-zA-Z0-9_-]+", alias) for alias in aliases)
    assert aliases == [_tool_backend_name(leaf, group) for group, leaf in identities]
    assert _tool_backend_name("global_name", None) == "global_name"


@pytest.mark.parametrize(
    "selector,expected_names,mode",
    [
        ({"type": "namespace", "name": "first.group"}, {"first.group"}, "required"),
        ({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "namespace", "name": "second_group"}]}, {"second_group"}, "auto"),
        (
            {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "lookup", "namespace": "first.group"}]},
            {"first.group"},
            "required",
        ),
        (
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "lookup"}, {"type": "namespace", "name": "second_group"}],
            },
            {None, "second_group"},
            "auto",
        ),
    ],
)
def test_namespace_and_allowed_tools_choices_restrict_the_actual_backend_toolset(selector, expected_names, mode):
    payload, _, _ = prepare_request(ResponsesRequest(**response_body(tools=grouped_tools(), tool_choice=selector)), ResponseStore())
    assert {tool["function"]["name"] for tool in payload["tools"]} == {_tool_backend_name("lookup", ns) for ns in expected_names}
    assert payload["tool_choice"] == mode


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "selector",
    [
        {"type": "namespace", "name": "first.group"},
        {"type": "function", "name": "lookup", "namespace": "first.group"},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "namespace", "name": "first.group"}]},
    ],
)
def test_backend_cannot_call_tools_outside_requested_namespace(native, stream, selector):
    client, backend, _ = native
    backend.message = {"tool_calls": [backend_call(_tool_backend_name("lookup", "second_group"), "{}")]}
    backend.finish = "tool_calls"
    response = client.post("/v1/responses", json=response_body(tools=grouped_tools(), tool_choice=selector, stream=stream))
    if stream:
        assert response.status_code == 200
        events = parse_sse(response.text)
        assert events[-1]["type"] == "response.failed"
        assert not any(event["type"] == "response.output_item.added" for event in events)
    else:
        assert response.status_code == 502, response.text


@pytest.mark.parametrize(
    "selector",
    [
        {"type": "namespace", "name": "missing"},
        {"type": "function", "name": "lookup", "namespace": "missing"},
        {"type": "custom", "name": "lookup", "namespace": "first.group"},
        {"type": "function", "name": "lookup", "namespace": []},
        {"type": "allowed_tools", "mode": "none", "tools": [{"type": "namespace", "name": "first.group"}]},
        {"type": "allowed_tools", "mode": "auto", "tools": []},
        {"type": "allowed_tools", "mode": "auto", "tools": [None]},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search", "name": "hosted"}]},
    ],
)
def test_invalid_namespace_tool_choices_fail_before_backend(native, selector):
    client, backend, _ = native
    response = client.post("/v1/responses", json=response_body(tools=grouped_tools(), tool_choice=selector, input="PRIVATE_QUESTION"))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"].startswith("tool_choice")
    assert "PRIVATE_QUESTION" not in response.text
    assert not backend.calls


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "namespace", "tools": []}],
        [{"type": "namespace", "name": "group"}],
        [{"type": "namespace", "name": "group", "tools": {}}],
        [{"type": "namespace", "name": "group", "tools": [None]}],
        [{"type": "namespace", "name": "group", "tools": [], "children": []}],
        [{"type": "namespace", "name": "group", "tools": [{"type": "web_search"}]}],
        [{"type": "namespace", "name": "group", "tools": [{"type": "namespace", "name": "nested", "tools": [TOOL]}]}],
        [{"type": "namespace", "name": "group", "tools": [TOOL, TOOL]}],
        [{"type": "namespace", "name": "group", "tools": [TOOL]}, {"type": "namespace", "name": "group", "tools": []}],
        [{"type": "namespace", "name": "group", "tools": [{"type": "function", "name": "x", "parameters": []}]}],
        [{"type": "namespace", "name": "group", "description": {}, "tools": [TOOL]}],
        [TOOL, TOOL],
        [{"type": "web_search"}],
    ],
)
def test_invalid_groups_do_not_silently_skip_unsupported_tools(native, tools):
    client, backend, _ = native
    response = client.post("/v1/responses", json=response_body(tools=tools, input="PRIVATE_QUESTION"))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"].startswith("tools")
    assert "PRIVATE_QUESTION" not in response.text
    assert not backend.calls


@pytest.mark.parametrize("reverse", [False, True])
def test_namespace_alias_collision_with_explicit_global_tool_is_rejected(native, reverse):
    client, backend, _ = native
    tools = [
        {"type": "function", "name": _tool_backend_name("weather", "group"), "parameters": {"type": "object"}},
        {"type": "namespace", "name": "group", "tools": [TOOL]},
    ]
    response = client.post("/v1/responses", json=response_body(tools=list(reversed(tools)) if reverse else tools))
    assert response.status_code == 400, response.text
    assert not backend.calls


def test_historical_namespace_identity_cannot_be_redirected_to_colliding_global_tool(native):
    client, backend, _ = native
    response = client.post(
        "/v1/responses",
        json=response_body(
            tools=[{"type": "function", "name": _tool_backend_name("weather", "group")}],
            input=[{"type": "function_call", "name": "weather", "namespace": "group", "call_id": "c1", "arguments": "{}"}],
        ),
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "input"
    assert not backend.calls


def test_namespace_function_nullable_parameters_and_custom_grammar_are_supported():
    req = ResponsesRequest(
        **response_body(
            tools=[
                {
                    "type": "namespace",
                    "name": "group",
                    "tools": [
                        {"type": "function", "name": "empty", "parameters": None},
                        {"type": "custom", "name": "raw", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}},
                    ],
                }
            ],
            tool_choice={"type": "allowed_tools", "mode": "required", "tools": [{"type": "custom", "name": "raw", "namespace": "group"}]},
        )
    )
    payload, _, _ = prepare_request(req, ResponseStore())
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == _tool_backend_name("raw", "group")
    assert "start: /.+/" in payload["tools"][0]["function"]["description"]
    assert payload["tool_choice"] == "required"


@pytest.mark.parametrize("provider,model", [("deepseek", MODEL), ("qwen", "qwen3.8-max")])
@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_actual_provider_pipeline_preserves_inline_instructions_and_tools(monkeypatch, provider, model, protocol, stream):
    from test_tools_api import (
        DS_PLAIN_SSE,
        DS_TOOL_SSE,
        QWEN_TOOL_SSE,
        TOOL_JSON,
        FakeAccount,
    )

    name = _tool_backend_name("get_weather", "weather_group") if protocol == "responses" else "get_weather"
    web_sse = (DS_TOOL_SSE if provider == "deepseek" else QWEN_TOOL_SSE).replace("get_weather", name)
    plain_sse = DS_PLAIN_SSE if provider == "deepseek" else QWEN_TOOL_SSE.replace(dumps(TOOL_JSON), dumps("The weather is sunny."))
    account, pool = FakeAccount([web_sse, plain_sse]), MagicMock()
    account.sessions.can_reuse.return_value = True
    pool.acquire = AsyncMock(return_value=(account, "stale_session"))
    monkeypatch.setattr(api.app.state, "pool", pool, raising=False)
    monkeypatch.setattr(api.app.state, "qwen_pool", pool, raising=False)
    # Do not mock _acquire_and_build, prompt construction, or provider dispatch.
    if protocol == "responses":
        body = response_body(
            model=model,
            stream=stream,
            session_id="explicit_old_session",
            reasoning={"effort": "none"},
            input=[{"role": "user", "content": "Original weather question"}, {"role": "developer", "content": "Later client instruction"}],
            tools=[
                {
                    "type": "namespace",
                    "name": "weather_group",
                    "tools": [{"type": "function", "name": "get_weather", "parameters": TOOL["parameters"]}],
                }
            ],
        )
    else:
        body = message_body(
            model=model,
            stream=stream,
            session_id="explicit_old_session",
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": "Original weather question"}, {"role": "system", "content": "Later client instruction"}],
            tools=[{"name": "get_weather", "input_schema": TOOL["parameters"]}],
        )
    # No lifespan: mock accounts only, never load or validate real credentials.
    response = TestClient(api.app).post("/v1/" + protocol, json=body)
    assert response.status_code == 200, response.text
    if protocol == "responses":
        item = completed_response(response, stream)["output"][-1]
        assert (item["name"], item["namespace"]) == ("get_weather", "weather_group")
    elif stream:
        events = parse_sse(response.text)
        assert events[-1]["type"] == "message_stop", response.text
        item = next(
            event["content_block"]
            for event in events
            if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
        )
        assert item["name"] == "get_weather"
    else:
        item = response.json()["content"][-1]
        assert item["name"] == "get_weather"
    account.client.completion.assert_awaited_once()
    sent = account.client.completion.await_args.kwargs
    assert "User: Original weather question\nSystem: Later client instruction" in sent["prompt"]
    assert name in sent["prompt"]
    assert sent["thinking_enabled" if provider == "deepseek" else "thinking"] is False
    pool.acquire.assert_awaited_once_with(None, api.settings.acquire_timeout)
    pool.resolve_context.assert_not_called()
    assert account.sessions.obtain.await_args.args[0] is None
    assert account.sem._value == 1

    follow_up = copy.deepcopy(body)
    if protocol == "responses":
        follow_up["input"] += [item, {"type": "function_call_output", "call_id": item["call_id"], "output": "Sunny tool result"}]
        follow_up["tool_choice"] = "none"
    else:
        follow_up["messages"] += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": item["id"], "name": "get_weather", "input": {"city": "Moscow"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": item["id"], "content": "Sunny tool result"}]},
        ]
        follow_up["tool_choice"] = {"type": "none"}
    again = TestClient(api.app).post("/v1/" + protocol, json=follow_up)
    assert again.status_code == 200, again.text
    if protocol == "responses":
        assert completed_response(again, stream)["output"][0]["type"] == "message"
    elif stream:
        assert parse_sse(again.text)[-1]["type"] == "message_stop", again.text
    else:
        assert again.json()["content"][0]["type"] == "text"
    assert account.client.completion.await_count == 2
    prompt = account.client.completion.await_args.kwargs["prompt"]
    assert (
        prompt.index("User: Original weather question")
        < prompt.index("System: Later client instruction")
        < prompt.index("Sunny tool result")
    )
    assert f'"name": "{name}"' in prompt
    assert "Client function tools are disabled" in prompt
    assert account.sessions.obtain.await_args.args[0] is None
    assert account.sem._value == 1
